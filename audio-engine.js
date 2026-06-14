// ClearEar audio engine  —  software-defined hearing aid DSP (zero-lookahead build)
// =============================================================================
// Same hearing-aid intent as before (multiband WDRC, prescriptive gain, dual
// noise reduction, speech focus, wind control, output limiting) but with EVERY
// latency-inducing node removed, plus an AGC that keeps the wearer's own voice
// from being amplified.
//
// WHY THIS VERSION IS FASTER:
//   The previous build used 8 band DynamicsCompressor nodes plus one limiter
//   DynamicsCompressor. Each compressor uses an internal lookahead buffer, and
//   Chrome silently delays the whole output to feed them. getMetrics() reported
//   ~20ms, but the hidden compressor lookahead added the rest of the delay you
//   could hear. This build has NO DynamicsCompressor nodes:
//     * Multiband WDRC is done as software gain, applied through plain GainNodes
//       that are automated from the analysis loop. GainNodes have zero lookahead.
//     * The output limiter (MPO) is a WaveShaperNode doing instantaneous
//       soft-clipping. Also zero lookahead.
//   Net effect: the only remaining latency is the unavoidable device + output
//   buffer (~20ms), which feels live.
//
// AGC (own-voice management):
//   Because the mic sits close to the wearer, their own voice is the loudest
//   thing it hears. A broadband AGC pulls overall gain down when the input is
//   loud, so the wearer's own voice is not amplified, while quieter speech from
//   someone across the room still gets full prescribed gain. It cannot tell your
//   voice apart from any other loud source, but the close-mic geometry makes it
//   effective in practice.
//
// FEEDBACK SAFETY: output is silent until a separate output device is chosen,
// and the WaveShaper limiter caps the output so it can never howl loudly. Use a
// different physical device for input vs output to avoid feedback.
//
// The public surface (window.audioEngine.*) is unchanged, so the UI keeps working.
// =============================================================================

(function () {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;

  let ctx = null;
  let stream = null;
  let src = null;
  let connected = false;
  let playbackConnected = false;
  let updateInterval = null;

  let inputDeviceId = localStorage.getItem('clearear.inputDevice') || null;
  let outputDeviceId = localStorage.getItem('clearear.outputDevice') || null;

  const BANDS = [
    { f: 60,    lo: 30,   hi: 120,   speech: false },
    { f: 170,   lo: 120,  hi: 240,   speech: false },
    { f: 310,   lo: 240,  hi: 440,   speech: true  },
    { f: 600,   lo: 440,  hi: 820,   speech: true  },
    { f: 1000,  lo: 820,  hi: 1400,  speech: true  },
    { f: 3000,  lo: 1400, hi: 4200,  speech: true  },
    { f: 6000,  lo: 4200, hi: 8400,  speech: true  },
    { f: 12000, lo: 8400, hi: 16000, speech: false }
  ];
  const NB = BANDS.length;

  // dB SPL calibration (estimate, not a calibrated meter)
  const SPL_OFFSET = 90;
  const SPL_MIN = 20, SPL_MAX = 110;

  // WDRC + NR tuning (all in 0..1 band-level terms unless noted)
  const COMP_KNEE = 0.060;        // band level where compression begins
  const COMP_MIN_FACTOR = 0.25;   // most a band can be turned down by compression
  const EXPANSION_FLOOR = 0.020;  // below this a band is hiss → squelch
  const MOD_SPEECH_DEPTH = 0.055; // modulation depth that looks speech-like
  const MAX_BAND_GAIN_DB = 18;
  const MAX_BAND_TARGET = 6.0;    // absolute linear ceiling per band (safety)

  // AGC: keep the wearer's (loud, close) own voice from being amplified.
  const AGC = { thresh: 0.12, slope: 0.85, floor: 0.35 };

  const proc = { cancel: 0.6, speech: 0.75, transparency: 0.35, name: 'Quiet space' };

  // Graph nodes
  let inputGain = null;
  let preHighpass = null;
  let inputAnalyser = null;
  let bandFilters = [];
  let bandGain = [];        // ONE gain node per band: carries WDRC * NR * makeup * AGC
  let sumBus = null;
  let masterGain = null;
  let limiterShaper = null; // WaveShaper soft-clip (MPO, zero lookahead)
  let outputGain = null;

  // Per-band analysis state
  const maxFollow = new Float32Array(NB);
  const minFollow = new Float32Array(NB);
  const bandLevel = new Float32Array(NB);
  const makeupLin = new Float32Array(NB);
  const compExp = new Float32Array(NB);
  let speechDetected = false;
  let speechHold = 0;
  let dbSmooth = 35;
  let noiseFloor = 1e-5;
  let lastSnrDb = 0;
  let lastAgc = 1;
  for (let i = 0; i < NB; i++) { maxFollow[i] = 0.02; minFollow[i] = 0.02; }

  let binCache = null;

  const prescribedDb = new Float32Array(NB);

  function dbToLin(db) { return Math.pow(10, db / 20); }
  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

  function bandRatio(i) {
    const g = Math.abs(prescribedDb[i] || 0);
    let r = 1.7 + g * 0.07;
    if (r < 1.5) r = 1.5;
    if (r > 3.5) r = 3.5;
    return r;
  }

  // Recompute prescription-derived constants (makeup gain + compression exponent).
  function refreshPrescription() {
    const eq = (window.state && Array.isArray(window.state.eq)) ? window.state.eq : null;
    for (let i = 0; i < NB; i++) {
      let g = eq ? Number(eq[i]) || 0 : 0;
      if (g > MAX_BAND_GAIN_DB) g = MAX_BAND_GAIN_DB;
      if (g < -MAX_BAND_GAIN_DB) g = -MAX_BAND_GAIN_DB;
      prescribedDb[i] = g;
      makeupLin[i] = dbToLin(g);
      compExp[i] = (1 / bandRatio(i)) - 1; // negative: gain falls as level rises
    }
  }

  // ===========================================================================
  //  REPLAY BUFFER  (AudioWorklet — no added latency)
  // ===========================================================================
  let recorderNode = null;
  let recorderModuleLoaded = false;
  let grabSeq = 0;
  const grabResolvers = {};
  let mediaRecorder = null;
  let recordedChunks = [];

  async function setupRecorderWorklet() {
    if (recorderNode || !ctx || !ctx.audioWorklet || !limiterShaper) return;
    try {
      if (!recorderModuleLoaded) {
        await ctx.audioWorklet.addModule('clearear-recorder-worklet.js');
        recorderModuleLoaded = true;
      }
      recorderNode = new AudioWorkletNode(ctx, 'clearear-recorder', {
        numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1,
        channelCountMode: 'explicit', channelInterpretation: 'speakers'
      });
      recorderNode.port.onmessage = (e) => {
        const d = e.data || {};
        const r = grabResolvers[d.id];
        if (r) { delete grabResolvers[d.id]; r({ samples: d.samples, sampleRate: d.sampleRate }); }
      };
      const sink = ctx.createGain();
      sink.gain.value = 0;
      limiterShaper.connect(recorderNode);
      recorderNode.connect(sink);
      sink.connect(ctx.destination);
    } catch (e) {
      recorderNode = null;
      console.warn('[AudioEngine] recorder worklet unavailable, using MediaRecorder:', e && e.message);
    }
  }

  function grabFromWorklet(seconds) {
    return new Promise((resolve) => {
      const id = ++grabSeq;
      grabResolvers[id] = resolve;
      try { recorderNode.port.postMessage({ cmd: 'grab', seconds: seconds, id: id }); }
      catch (e) { delete grabResolvers[id]; resolve(null); }
      setTimeout(() => { if (grabResolvers[id]) { delete grabResolvers[id]; resolve(null); } }, 600);
    });
  }

  function hasSignal(samples) {
    if (!samples || !samples.length) return false;
    const step = Math.max(1, Math.floor(samples.length / 4000));
    for (let i = 0; i < samples.length; i += step) if (Math.abs(samples[i]) > 0.0006) return true;
    return false;
  }

  function startMediaRecorder(mediaStream) {
    if (mediaRecorder) return;
    try { mediaRecorder = new MediaRecorder(mediaStream, { mimeType: 'audio/webm;codecs=opus' }); }
    catch (e) { try { mediaRecorder = new MediaRecorder(mediaStream); } catch (e2) { return; } }
    recordedChunks = [];
    mediaRecorder.ondataavailable = function (e) {
      if (e.data && e.data.size > 0) {
        recordedChunks.push({ blob: e.data, ts: Date.now() });
        const cutoff = Date.now() - 70000;
        while (recordedChunks.length && recordedChunks[0].ts < cutoff) recordedChunks.shift();
      }
    };
    try { mediaRecorder.start(1000); } catch (e) {}
  }

  function getPlaybackBuffer() { return null; }

  async function playBufferedAudio(seconds) {
    if (recorderNode) {
      const data = await grabFromWorklet(seconds);
      if (data && data.samples && hasSignal(data.samples)) {
        const playCtx = new (window.AudioContext || window.webkitAudioContext)();
        const buffer = playCtx.createBuffer(1, data.samples.length, data.sampleRate || 48000);
        buffer.getChannelData(0).set(data.samples);
        const source = playCtx.createBufferSource();
        source.buffer = buffer;
        const g = playCtx.createGain();
        g.gain.value = 1.4;
        source.connect(g);
        g.connect(playCtx.destination);
        source.start();
        source.onended = function () { try { playCtx.close(); } catch (e) {} };
        return { source: source, ctx: playCtx, durationSec: data.samples.length / (data.sampleRate || 48000) };
      }
    }
    if (!recordedChunks.length) return null;
    const cutoff = Date.now() - (seconds * 1000);
    const relevant = recordedChunks.filter(c => c.ts >= cutoff).map(c => c.blob);
    if (!relevant.length) return null;
    const blob = new Blob(relevant, { type: relevant[0].type || 'audio/webm' });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.play();
    audio.onended = function () { URL.revokeObjectURL(url); };
    return { audio: audio, durationSec: seconds };
  }

  // ===========================================================================
  //  GRAPH
  // ===========================================================================
  function ensureCtx() {
    if (!ctx) {
      const Ctor = window.AudioContext || window.webkitAudioContext;
      try { ctx = new Ctor({ latencyHint: 'interactive' }); }
      catch (e) { ctx = new Ctor(); }
    }
    return ctx;
  }

  // Soft-clip curve for the output limiter: linear-ish through normal levels,
  // saturating to a ceiling beyond. Instantaneous, so zero added latency.
  function makeSoftClipCurve(drive) {
    const n = 2048;
    const curve = new Float32Array(n);
    const norm = Math.tanh(drive);
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * 2 - 1;
      curve[i] = Math.tanh(drive * x) / norm;
    }
    return curve;
  }

  function buildGraph() {
    ensureCtx();
    if (inputGain) return;

    inputGain = ctx.createGain();
    inputGain.channelCount = 1;
    inputGain.channelCountMode = 'explicit';
    inputGain.channelInterpretation = 'speakers';
    inputGain.gain.value = 1.0;

    preHighpass = ctx.createBiquadFilter();
    preHighpass.type = 'highpass';
    preHighpass.frequency.value = 90;
    preHighpass.Q.value = 0.7;

    // Smaller FFT than before (1024 vs 2048) → ~21ms analysis window, so NR/VAD
    // and the AGC react roughly twice as fast.
    inputAnalyser = ctx.createAnalyser();
    inputAnalyser.fftSize = 1024;
    inputAnalyser.smoothingTimeConstant = 0.5;

    bandFilters = []; bandGain = [];
    sumBus = ctx.createGain();
    sumBus.gain.value = 1.0;

    refreshPrescription();

    for (let i = 0; i < NB; i++) {
      const band = BANDS[i];
      const filt = ctx.createBiquadFilter();
      if (i === 0) { filt.type = 'lowpass'; filt.frequency.value = band.hi; filt.Q.value = 0.7; }
      else if (i === NB - 1) { filt.type = 'highpass'; filt.frequency.value = band.lo; filt.Q.value = 0.7; }
      else { filt.type = 'bandpass'; filt.frequency.value = band.f; filt.Q.value = 1.4; }

      // One automated gain node per band. No compressor, no lookahead.
      const g = ctx.createGain();
      g.gain.value = makeupLin[i];

      inputAnalyser.connect(filt);
      filt.connect(g);
      g.connect(sumBus);

      bandFilters.push(filt);
      bandGain.push(g);
    }

    masterGain = ctx.createGain();
    masterGain.gain.value = 1.0;

    // Output limiter (MPO): WaveShaper soft-clip. Zero lookahead.
    limiterShaper = ctx.createWaveShaper();
    limiterShaper.curve = makeSoftClipCurve(1.4);
    limiterShaper.oversample = '2x';

    outputGain = ctx.createGain();
    outputGain.gain.value = 0.0; // silent until a real output device is chosen

    sumBus.connect(masterGain);
    masterGain.connect(limiterShaper);
    limiterShaper.connect(outputGain);
    outputGain.connect(ctx.destination);
  }

  function rebuildBinCache() {
    const binCount = inputAnalyser.frequencyBinCount;
    const nyquist = (ctx ? ctx.sampleRate : 48000) / 2;
    binCache = BANDS.map(b => ({
      lo: Math.max(0, Math.floor(b.lo / nyquist * binCount)),
      hi: Math.min(binCount - 1, Math.ceil(b.hi / nyquist * binCount))
    }));
    binCache._binCount = binCount;
  }

  // ===========================================================================
  //  ANALYSIS + CONTROL LOOP  (~60x/sec; computes all gains, applies via GainNodes)
  // ===========================================================================
  function analysisTick() {
    if (!inputAnalyser) return;
    if (!binCache || binCache._binCount !== inputAnalyser.frequencyBinCount) rebuildBinCache();
    const binCount = inputAnalyser.frequencyBinCount;
    const now = ctx.currentTime;

    // --- broadband level (time domain) → dB SPL + AGC ---
    const t = new Uint8Array(binCount);
    inputAnalyser.getByteTimeDomainData(t);
    let sq = 0;
    for (let i = 0; i < t.length; i++) { const v = (t[i] - 128) / 128; sq += v * v; }
    const rms = Math.sqrt(sq / t.length);

    // AGC: duck the loudest input (the wearer's own close-mic voice) so it is
    // not amplified, while leaving quiet/distant speech at full gain.
    let agc = 1;
    if (rms > AGC.thresh) {
      agc = Math.pow(AGC.thresh / rms, AGC.slope);
      if (agc < AGC.floor) agc = AGC.floor;
    }
    lastAgc = lastAgc * 0.6 + agc * 0.4; // smooth so it never pumps

    const dbFs = 20 * Math.log10(Math.max(rms, 1e-8));
    const dbSPL = Math.max(SPL_MIN, Math.min(SPL_MAX, dbFs + SPL_OFFSET));
    dbSmooth = dbSmooth * 0.6 + dbSPL * 0.4;

    // --- per-band spectrum analysis ---
    const freq = new Uint8Array(binCount);
    inputAnalyser.getByteFrequencyData(freq);

    let speechEnergy = 0, speechNoise = 0, lowEnergy = 0, midEnergy = 0;
    for (let i = 0; i < NB; i++) {
      const c = binCache[i];
      let sum = 0, n = 0;
      for (let b = c.lo; b <= c.hi; b++) { sum += freq[b]; n++; }
      const lvl = n > 0 ? (sum / n) / 255 : 0;
      bandLevel[i] = bandLevel[i] * 0.45 + lvl * 0.55;
      const bl = bandLevel[i];

      if (bl > maxFollow[i]) maxFollow[i] = bl;
      else maxFollow[i] = maxFollow[i] * 0.992 + bl * 0.008;
      if (bl < minFollow[i]) minFollow[i] = bl;
      else minFollow[i] = minFollow[i] * 0.9985 + bl * 0.0015;

      if (BANDS[i].speech) { speechEnergy += bl; speechNoise += minFollow[i]; }
      if (i <= 1) lowEnergy += bl;
      if (i >= 2 && i <= 5) midEnergy += bl;
    }

    const speechRatio = speechEnergy / (speechNoise + 1e-6);
    if (speechRatio > 1.6) speechHold = 14;          // ~230ms hold at 60Hz
    else if (speechHold > 0) speechHold--;
    speechDetected = speechHold > 0;

    const windy = (lowEnergy > midEnergy * 1.4) && !speechDetected;

    let gMin = 0.15 + proc.transparency * 0.55 - proc.cancel * 0.10;
    if (gMin < 0.08) gMin = 0.08;
    if (gMin > 0.92) gMin = 0.92;

    const wienerK = 0.4 + proc.cancel * 2.6;
    let snrAccum = 0, snrCount = 0;

    for (let i = 0; i < NB; i++) {
      const bl = bandLevel[i];
      const noise = minFollow[i] + 1e-6;
      const snr = bl / noise;
      snrAccum += snr; snrCount++;

      // (1) noise reduction: Wiener coefficient × modulation factor
      let nr = snr / (snr + wienerK);
      const modDepth = maxFollow[i] - minFollow[i];
      const modFactor = clamp01(modDepth / MOD_SPEECH_DEPTH);
      nr *= 0.45 + 0.55 * modFactor;

      // (2) expansion / squelch below the noise floor
      if (bl < EXPANSION_FLOOR) nr *= clamp01(bl / EXPANSION_FLOOR) * 0.6;

      // (3) speech focus
      if (BANDS[i].speech) {
        if (speechDetected) nr = Math.max(nr, 0.75 + 0.20 * proc.speech);
      } else {
        if (speechDetected) nr *= (1 - 0.45 * proc.speech);
        if (!speechDetected) nr *= (1 - 0.35 * proc.cancel);
      }
      if (windy && i <= 1) nr *= (1 - 0.6 * proc.cancel);
      if (nr < gMin) nr = gMin;
      if (nr > 1) nr = 1;

      // (4) software WDRC compression factor (zero lookahead). Gain falls as the
      //     band level rises above the kneepoint, restoring loudness balance.
      let comp = 1;
      if (bl > COMP_KNEE) {
        comp = Math.pow(bl / COMP_KNEE, compExp[i]);
        if (comp < COMP_MIN_FACTOR) comp = COMP_MIN_FACTOR;
        if (comp > 1) comp = 1;
      }

      // (5) combine: prescription × compression × NR × own-voice AGC
      let target = makeupLin[i] * comp * nr * lastAgc;
      if (target < 0) target = 0;
      if (target > MAX_BAND_TARGET) target = MAX_BAND_TARGET;

      // Fast to reduce (catch your own voice / noise), gentle to restore.
      const prev = bandGain[i].gain.value;
      const tc = target < prev ? 0.015 : 0.10;
      bandGain[i].gain.setTargetAtTime(target, now, tc);
    }

    lastSnrDb = 10 * Math.log10(Math.max(snrAccum / Math.max(1, snrCount), 1e-3));
    let nf = 0; for (let i = 0; i < NB; i++) nf += minFollow[i];
    noiseFloor = nf / NB;

    // --- UI: meter, spectrum, telemetry ---
    if (window.state) window.state.db = Math.round(dbSmooth);
    const el = document.getElementById('db-readout');
    if (el && window.state) el.textContent = Math.round(window.state.db);

    updateSpectrum(freq, binCount);

    if (typeof window.clearEarAudioTelemetry === 'function') {
      window.clearEarAudioTelemetry({
        db: window.state ? window.state.db : Math.round(dbSmooth),
        rms: rms,
        speechDetected: speechDetected,
        noiseFloor: noiseFloor,
        snrDb: lastSnrDb,
        preset: window.state && window.state.preset ? window.state.preset : 'Unknown',
        timestamp: Date.now()
      });
    }
    const meta = document.querySelector('.vis-meta');
    if (meta) meta.textContent = speechDetected ? 'Speech detected, words amplified' : 'Ambient';
  }

  function updateSpectrum(freqData, binCount) {
    if (!window.specData || !window.specData.length) return;
    const numBars = window.specData.length;
    const nyquist = (ctx ? ctx.sampleRate : 48000) / 2;
    const logMin = Math.log10(60), logMax = Math.log10(16000);
    for (let i = 0; i < numBars; i++) {
      const fLow = Math.pow(10, logMin + (i / numBars) * (logMax - logMin));
      const fHigh = Math.pow(10, logMin + ((i + 1) / numBars) * (logMax - logMin));
      const bLow = Math.max(0, Math.floor(fLow / nyquist * binCount));
      const bHigh = Math.min(binCount - 1, Math.ceil(fHigh / nyquist * binCount));
      let sum = 0, n = 0;
      for (let b = bLow; b <= bHigh; b++) { sum += freqData[b]; n++; }
      const avg = n > 0 ? sum / n / 255 : 0;
      window.specData[i] = window.specData[i] * 0.55 + avg * 0.45;
    }
  }

  // ===========================================================================
  //  PUBLIC CONTROL METHODS
  // ===========================================================================
  function setEqBands(gains) {
    if (!Array.isArray(gains)) return;
    for (let i = 0; i < NB; i++) {
      let g = Number(gains[i]) || 0;
      if (g > MAX_BAND_GAIN_DB) g = MAX_BAND_GAIN_DB;
      if (g < -MAX_BAND_GAIN_DB) g = -MAX_BAND_GAIN_DB;
      prescribedDb[i] = g;
      makeupLin[i] = dbToLin(g);
      compExp[i] = (1 / bandRatio(i)) - 1;
    }
    // The band gains pick up the new makeup on the next analysis tick.
  }

  function applyScene(scene) {
    if (!scene) return;
    proc.cancel = clamp01((scene.cancel || 0) / 100);
    proc.speech = clamp01((scene.speech || 0) / 100);
    proc.transparency = clamp01((scene.transparency || 0) / 100);
    proc.name = scene.name || proc.name;
  }

  function applyControls(controls) {
    if (!controls) return;
    proc.cancel = controls.cancel ? 0.8 : 0.25;
    proc.speech = controls.speech ? 0.8 : 0.2;
    proc.transparency = controls.transparency ? 0.92 : 0.30;
  }

  // ===========================================================================
  //  DEVICE-IN-USE HIGHLIGHT  (CSS injected here; the HTML already tags .active)
  // ===========================================================================
  function injectDeviceHighlightCSS() {
    if (document.getElementById('clearear-device-highlight')) return;
    const style = document.createElement('style');
    style.id = 'clearear-device-highlight';
    style.textContent =
      '#device-input-list .suggestion.active,#device-output-list .suggestion.active{' +
      'border-color:var(--accent)!important;background:rgba(217,249,157,0.10)!important;' +
      'color:var(--text-primary)!important;position:relative;padding-right:64px;}' +
      '#device-input-list .suggestion.active::after,#device-output-list .suggestion.active::after{' +
      "content:'in use';position:absolute;right:10px;top:50%;transform:translateY(-50%);" +
      'font-family:var(--font-mono);font-size:9px;letter-spacing:0.08em;text-transform:uppercase;' +
      'color:var(--accent);}';
    document.head.appendChild(style);
  }

  // Reflect the device actually in use so the picker highlights it even when the
  // user never explicitly chose one (i.e. the system default mic).
  function reflectActiveInputDevice() {
    if (!stream || !window.state) return;
    const track = stream.getAudioTracks ? stream.getAudioTracks()[0] : null;
    if (!track) return;
    const s = track.getSettings ? track.getSettings() : {};
    if (s.deviceId && !window.state.inputDeviceId) {
      window.state.inputDeviceId = s.deviceId;
      window.state.inputDeviceLabel = track.label || window.state.inputDeviceLabel || '';
      try { if (typeof window.updateDeviceStatus === 'function') window.updateDeviceStatus(); } catch (e) {}
    }
  }

  // ===========================================================================
  //  LIFECYCLE
  // ===========================================================================
  async function start() {
    try {
      buildGraph();
      await ctx.resume();
      if (ctx.state === 'suspended') return;

      if (!stream) {
        const audioConstraints = {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
          latency: 0
        };
        if (inputDeviceId) audioConstraints.deviceId = { exact: inputDeviceId };
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
        } catch (err) {
          inputDeviceId = null;
          localStorage.removeItem('clearear.inputDevice');
          stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, latency: 0 }
          });
        }
      }
      if (!src) src = ctx.createMediaStreamSource(stream);
      startMediaRecorder(stream);
      reflectActiveInputDevice();

      if (connected) {
        if (!updateInterval) updateInterval = setInterval(analysisTick, 1000 / 60);
        return;
      }

      src.connect(inputGain);
      inputGain.connect(preHighpass);
      preHighpass.connect(inputAnalyser);

      refreshPrescription();
      if (window.state && window.state.controls) applyControls(window.state.controls);
      if (window.clearEarSceneConfig) applyScene(window.clearEarSceneConfig);

      await setupRecorderWorklet();
      maybeEnablePlayback();

      if (!updateInterval) updateInterval = setInterval(analysisTick, 1000 / 60);

      connected = true;
      window.audioEngine.running = true;

      const lat = Math.round(((ctx.baseLatency || 0) + (ctx.outputLatency || 0)) * 1000);
      console.log('[AudioEngine] running (zero-lookahead). sampleRate', ctx.sampleRate,
        '| est. output latency ~' + lat + 'ms');
    } catch (err) {
      console.warn('[AudioEngine] start failed:', err);
    }
  }

  function stop() {
    try {
      if (updateInterval) { clearInterval(updateInterval); updateInterval = null; }
      disablePlayback();
      [src, inputGain, preHighpass].forEach(n => { if (n) { try { n.disconnect(); } catch (e) {} } });
      connected = false;
      window.audioEngine.running = false;
    } catch (err) {}
  }

  function isDefaultOutput(id) { return !id || id === 'default' || id === ''; }

  function maybeEnablePlayback() {
    if (!outputGain || !ctx) return;
    if (isDefaultOutput(outputDeviceId)) { disablePlayback(); return; }
    try {
      outputGain.gain.setTargetAtTime(1.0, ctx.currentTime, 0.05);
      playbackConnected = true;
      if (typeof ctx.setSinkId === 'function' && outputDeviceId) {
        ctx.setSinkId(outputDeviceId).catch(e => console.warn('setSinkId failed:', e));
      }
    } catch (e) { console.warn('enable playback failed:', e); }
  }

  function disablePlayback() {
    if (!outputGain || !ctx) return;
    try { outputGain.gain.setTargetAtTime(0.0, ctx.currentTime, 0.05); } catch (e) {}
    playbackConnected = false;
  }

  async function setInputDevice(deviceId) {
    inputDeviceId = deviceId || null;
    if (inputDeviceId) localStorage.setItem('clearear.inputDevice', inputDeviceId);
    else localStorage.removeItem('clearear.inputDevice');
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; src = null; }
    if (mediaRecorder) { try { mediaRecorder.stop(); } catch (e) {} mediaRecorder = null; }
    if (connected) { stop(); await start(); }
  }

  async function setOutputDevice(deviceId) {
    outputDeviceId = deviceId || null;
    if (outputDeviceId) localStorage.setItem('clearear.outputDevice', outputDeviceId);
    else localStorage.removeItem('clearear.outputDevice');
    maybeEnablePlayback();
  }

  function isPlaybackEnabled() { return playbackConnected; }
  function getInputDeviceId() { return inputDeviceId; }
  function getOutputDeviceId() { return outputDeviceId; }
  function getMetrics() {
    const baseMs = (ctx && ctx.baseLatency) ? ctx.baseLatency * 1000 : 0;
    const outMs = (ctx && ctx.outputLatency) ? ctx.outputLatency * 1000 : 0;
    return {
      db: window.state ? window.state.db : Math.round(dbSmooth),
      speechDetected: speechDetected,
      snrDb: lastSnrDb,
      noiseFloor: noiseFloor,
      agc: lastAgc,
      prescribedDb: Array.from(prescribedDb),
      strengths: { cancel: proc.cancel, speech: proc.speech, transparency: proc.transparency },
      latencyMs: Math.round(baseMs + outMs),
      baseLatencyMs: Math.round(baseMs),
      outputLatencyMs: Math.round(outMs),
      compressorsInPath: 0
    };
  }

  window.audioEngine = {
    start: start,
    stop: stop,
    running: false,
    setEqBands: setEqBands,
    applyScene: applyScene,
    applyControls: applyControls,
    setInputDevice: setInputDevice,
    setOutputDevice: setOutputDevice,
    isPlaybackEnabled: isPlaybackEnabled,
    getInputDeviceId: getInputDeviceId,
    getOutputDeviceId: getOutputDeviceId,
    getPlaybackBuffer: getPlaybackBuffer,
    playBufferedAudio: playBufferedAudio,
    getMetrics: getMetrics,
    get ctx() { return ctx; }
  };

  injectDeviceHighlightCSS();

  const unlock = function () {
    start().catch(function () {});
    document.removeEventListener('click', unlock);
    document.removeEventListener('keydown', unlock);
  };
  document.addEventListener('click', unlock);
  document.addEventListener('keydown', unlock);

  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(function () { start().catch(function () {}); }, 100);
  } else {
    document.addEventListener('DOMContentLoaded', function () {
      injectDeviceHighlightCSS();
      setTimeout(function () { start().catch(function () {}); }, 100);
    });
  }
})();