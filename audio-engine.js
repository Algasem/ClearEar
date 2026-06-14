// ClearEar audio engine  —  software-defined hearing aid DSP (low-latency build)
// =============================================================================
// Same hearing-aid processing as before (8-channel WDRC, prescriptive gain, dual
// noise reduction, speech focus, wind + transient control, MPO limiter), but
// re-plumbed for minimum latency so amplified speech feels live, like a real
// hearing aid rather than a delayed monitor.
//
// WHAT CHANGED FOR LATENCY (and why the DSP is untouched):
//   * The replay buffer no longer uses a ScriptProcessorNode. ScriptProcessor
//     runs on the main thread, and its mere presence makes Chrome buffer the
//     entire output ahead to cover main-thread jitter, which on a busy page adds
//     hundreds of ms. It is replaced by an AudioWorklet on the audio thread that
//     adds no meaningful latency.
//   * The AudioContext is created with a low-latency hint.
//   * The mic is opened raw and low-latency (no browser echo-cancel / noise-
//     suppression / AGC), so nothing buffers or re-processes ahead of our DSP.
//   * The live signal path is all native nodes (filters, compressors, gains),
//     which together add only a few ms. These do the amplification and
//     attenuation, so they are left exactly as they were.
//
// FEEDBACK SAFETY (unchanged): output is silent until a separate output device
// (earbuds) is chosen, and a brick-wall output limiter caps the level so the
// device can never howl loudly. For zero feedback, the mic and the earbuds
// should be different physical devices (e.g. built-in mic in, earbuds out).
//
// The public surface (window.audioEngine.*) and the window hooks it reads/writes
// are unchanged, so the existing UI keeps working.
// =============================================================================

(function () {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;

  // ── Context + stream ──────────────────────────────────────────────────────
  let ctx = null;
  let stream = null;
  let src = null;
  let connected = false;
  let playbackConnected = false;
  let updateInterval = null;

  let inputDeviceId = localStorage.getItem('clearear.inputDevice') || null;
  let outputDeviceId = localStorage.getItem('clearear.outputDevice') || null;

  // ── Band plan (8 channels, aligned to the UI EQ band centres) ─────────────
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

  // ── dB SPL calibration (estimate, not a calibrated meter) ────────────────
  const SPL_OFFSET = 90;
  const SPL_MIN = 20, SPL_MAX = 110;
  const EXPANSION_FLOOR = 0.020;
  const MOD_SPEECH_DEPTH = 0.055;
  const MAX_BAND_GAIN_DB = 18;

  // ── Processing strength (driven by scenes + control toggles) ──────────────
  const proc = { cancel: 0.6, speech: 0.75, transparency: 0.35, name: 'Quiet space' };

  // ── Web Audio graph nodes ────────────────────────────────────────────────
  let inputGain = null;
  let preHighpass = null;
  let inputAnalyser = null;
  let bandFilters = [];
  let bandComp = [];
  let bandNR = [];
  let bandMakeup = [];
  let sumBus = null;
  let masterGain = null;
  let limiter = null;
  let outputGain = null;

  // ── Per-band running analysis state ──────────────────────────────────────
  const maxFollow = new Float32Array(NB);
  const minFollow = new Float32Array(NB);
  const bandLevel = new Float32Array(NB);
  let speechDetected = false;
  let speechHold = 0;
  let dbSmooth = 35;
  let noiseFloor = 1e-5;
  let lastSnrDb = 0;
  for (let i = 0; i < NB; i++) { maxFollow[i] = 0.02; minFollow[i] = 0.02; }

  let binCache = null;

  const prescribedDb = new Float32Array(NB);
  function readPrescription() {
    const eq = (window.state && Array.isArray(window.state.eq)) ? window.state.eq : null;
    for (let i = 0; i < NB; i++) {
      let g = eq ? Number(eq[i]) || 0 : 0;
      if (g > MAX_BAND_GAIN_DB) g = MAX_BAND_GAIN_DB;
      if (g < -MAX_BAND_GAIN_DB) g = -MAX_BAND_GAIN_DB;
      prescribedDb[i] = g;
    }
  }
  function dbToLin(db) { return Math.pow(10, db / 20); }
  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

  // ===========================================================================
  //  REPLAY BUFFER  (AudioWorklet on the audio thread — no added latency)
  //  Falls back to MediaRecorder if the worklet can't load (older browsers).
  // ===========================================================================
  let recorderNode = null;
  let recorderModuleLoaded = false;
  let grabSeq = 0;
  const grabResolvers = {};

  let mediaRecorder = null;
  let recordedChunks = [];

  async function setupRecorderWorklet() {
    if (recorderNode || !ctx || !ctx.audioWorklet || !limiter) return;
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
      // Tap the processed output. A silent sink keeps the node scheduled without
      // adding anything audible to the live path.
      const sink = ctx.createGain();
      sink.gain.value = 0;
      limiter.connect(recorderNode);
      recorderNode.connect(sink);
      sink.connect(ctx.destination);
    } catch (e) {
      recorderNode = null; // graceful fallback to MediaRecorder
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
    for (let i = 0; i < samples.length; i += step) {
      if (Math.abs(samples[i]) > 0.0006) return true;
    }
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

  // Kept for API compatibility. The reliable path is now async (worklet), so the
  // synchronous accessor simply reports "nothing buffered".
  function getPlaybackBuffer() { return null; }

  async function playBufferedAudio(seconds) {
    // Primary: PCM from the worklet ring buffer (the processed/clarified output).
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
    // Fallback: MediaRecorder chunks (raw mic audio).
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
  //  GRAPH CONSTRUCTION
  // ===========================================================================
  function ensureCtx() {
    if (!ctx) {
      const Ctor = window.AudioContext || window.webkitAudioContext;
      // 'interactive' asks for the lowest stable output buffer the device allows.
      try { ctx = new Ctor({ latencyHint: 'interactive' }); }
      catch (e) { ctx = new Ctor(); }
    }
    return ctx;
  }

  function bandRatio(i) {
    const g = Math.abs(prescribedDb[i] || 0);
    let r = 1.7 + g * 0.07;
    if (r < 1.5) r = 1.5;
    if (r > 3.5) r = 3.5;
    return r;
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

    // AnalyserNode is a zero-latency pass-through; it only taps for measurement.
    inputAnalyser = ctx.createAnalyser();
    inputAnalyser.fftSize = 2048;
    inputAnalyser.smoothingTimeConstant = 0.6;

    bandFilters = []; bandComp = []; bandNR = []; bandMakeup = [];
    sumBus = ctx.createGain();
    sumBus.gain.value = 1.0;

    readPrescription();

    for (let i = 0; i < NB; i++) {
      const band = BANDS[i];
      const filt = ctx.createBiquadFilter();
      if (i === 0) { filt.type = 'lowpass'; filt.frequency.value = band.hi; filt.Q.value = 0.7; }
      else if (i === NB - 1) { filt.type = 'highpass'; filt.frequency.value = band.lo; filt.Q.value = 0.7; }
      else { filt.type = 'bandpass'; filt.frequency.value = band.f; filt.Q.value = 1.4; }

      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -35;
      comp.knee.value = 22;
      comp.ratio.value = bandRatio(i);
      comp.attack.value = 0.005;
      comp.release.value = 0.12;

      const nr = ctx.createGain();
      nr.gain.value = 1.0;

      const makeup = ctx.createGain();
      makeup.gain.value = dbToLin(prescribedDb[i]);

      inputAnalyser.connect(filt);
      filt.connect(comp);
      comp.connect(nr);
      nr.connect(makeup);
      makeup.connect(sumBus);

      bandFilters.push(filt);
      bandComp.push(comp);
      bandNR.push(nr);
      bandMakeup.push(makeup);
    }

    masterGain = ctx.createGain();
    masterGain.gain.value = 1.0;

    // Output limiter = Maximum Power Output + impulse/transient control.
    limiter = ctx.createDynamicsCompressor();
    limiter.threshold.value = -6;
    limiter.knee.value = 0;
    limiter.ratio.value = 20;
    limiter.attack.value = 0.002;
    limiter.release.value = 0.10;

    outputGain = ctx.createGain();
    outputGain.gain.value = 0.0; // silent until a real output device is chosen

    sumBus.connect(masterGain);
    masterGain.connect(limiter);
    limiter.connect(outputGain);
    outputGain.connect(ctx.destination);
  }

  // ===========================================================================
  //  FFT BIN MAPPING
  // ===========================================================================
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
  //  ANALYSIS + CONTROL LOOP  (~30x/sec, off the audio path; steers gains only)
  // ===========================================================================
  function analysisTick() {
    if (!inputAnalyser) return;
    if (!binCache || binCache._binCount !== inputAnalyser.frequencyBinCount) rebuildBinCache();

    const binCount = inputAnalyser.frequencyBinCount;
    const freq = new Uint8Array(binCount);
    inputAnalyser.getByteFrequencyData(freq);
    const now = ctx.currentTime;

    let speechEnergy = 0, speechNoise = 0, lowEnergy = 0, midEnergy = 0;

    for (let i = 0; i < NB; i++) {
      const c = binCache[i];
      let sum = 0, n = 0;
      for (let b = c.lo; b <= c.hi; b++) { sum += freq[b]; n++; }
      const lvl = n > 0 ? (sum / n) / 255 : 0;
      bandLevel[i] = bandLevel[i] * 0.5 + lvl * 0.5;
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
    if (speechRatio > 1.6) speechHold = 8;
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

      let g = snr / (snr + wienerK);
      const modDepth = maxFollow[i] - minFollow[i];
      const modFactor = clamp01(modDepth / MOD_SPEECH_DEPTH);
      g *= 0.45 + 0.55 * modFactor;

      if (bl < EXPANSION_FLOOR) g *= clamp01(bl / EXPANSION_FLOOR) * 0.6;

      if (BANDS[i].speech) {
        if (speechDetected) g = Math.max(g, 0.75 + 0.20 * proc.speech);
      } else {
        if (speechDetected) g *= (1 - 0.45 * proc.speech);
        if (!speechDetected) g *= (1 - 0.35 * proc.cancel);
      }

      if (windy && i <= 1) g *= (1 - 0.6 * proc.cancel);

      if (g < gMin) g = gMin;
      if (g > 1) g = 1;

      const prev = bandNR[i].gain.value;
      const tc = g < prev ? 0.05 : 0.18; // quick to attenuate, gentle to restore
      bandNR[i].gain.setTargetAtTime(g, now, tc);
    }

    lastSnrDb = 10 * Math.log10(Math.max(snrAccum / Math.max(1, snrCount), 1e-3));

    let nf = 0; for (let i = 0; i < NB; i++) nf += minFollow[i];
    noiseFloor = nf / NB;

    updateSpectrum(freq, binCount);
    updateDb();
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

  function updateDb() {
    const t = new Uint8Array(inputAnalyser.frequencyBinCount);
    inputAnalyser.getByteTimeDomainData(t);
    let sum = 0;
    for (let i = 0; i < t.length; i++) { const v = (t[i] - 128) / 128; sum += v * v; }
    const rms = Math.sqrt(sum / t.length);
    const dbFs = 20 * Math.log10(Math.max(rms, 1e-8));
    const dbSPL = Math.max(SPL_MIN, Math.min(SPL_MAX, dbFs + SPL_OFFSET));

    dbSmooth = dbSmooth * 0.6 + dbSPL * 0.4;
    if (window.state) window.state.db = Math.round(dbSmooth);
    const el = document.getElementById('db-readout');
    if (el && window.state) el.textContent = Math.round(window.state.db);

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

  // ===========================================================================
  //  PUBLIC CONTROL METHODS
  // ===========================================================================
  function setEqBands(gains) {
    if (!Array.isArray(gains)) return;
    const now = ctx ? ctx.currentTime : 0;
    for (let i = 0; i < NB; i++) {
      let g = Number(gains[i]) || 0;
      if (g > MAX_BAND_GAIN_DB) g = MAX_BAND_GAIN_DB;
      if (g < -MAX_BAND_GAIN_DB) g = -MAX_BAND_GAIN_DB;
      prescribedDb[i] = g;
      if (bandMakeup[i]) bandMakeup[i].gain.setTargetAtTime(dbToLin(g), now, 0.05);
      if (bandComp[i]) bandComp[i].ratio.setTargetAtTime(bandRatio(i), now, 0.1);
    }
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
  //  LIFECYCLE
  // ===========================================================================
  async function start() {
    try {
      buildGraph();
      await ctx.resume();
      if (ctx.state === 'suspended') return;

      if (!stream) {
        // Raw, low-latency capture so our DSP is the only thing shaping sound.
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

      if (connected) {
        if (!updateInterval) updateInterval = setInterval(analysisTick, 1000 / 30);
        return;
      }

      src.connect(inputGain);
      inputGain.connect(preHighpass);
      preHighpass.connect(inputAnalyser);

      readPrescription();
      setEqBands(Array.from(prescribedDb));
      if (window.state && window.state.controls) applyControls(window.state.controls);
      if (window.clearEarSceneConfig) applyScene(window.clearEarSceneConfig);

      await setupRecorderWorklet();
      maybeEnablePlayback();

      if (!updateInterval) updateInterval = setInterval(analysisTick, 1000 / 30);

      connected = true;
      window.audioEngine.running = true;

      const lat = Math.round(((ctx.baseLatency || 0) + (ctx.outputLatency || 0)) * 1000);
      console.log('[AudioEngine] running. sampleRate', ctx.sampleRate,
        '| est. output latency ~' + lat + 'ms (input + Bluetooth not included)');
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

  // ── Output device gating (feedback safety) ────────────────────────────────
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
      prescribedDb: Array.from(prescribedDb),
      strengths: { cancel: proc.cancel, speech: proc.speech, transparency: proc.transparency },
      latencyMs: Math.round(baseMs + outMs),
      baseLatencyMs: Math.round(baseMs),
      outputLatencyMs: Math.round(outMs)
    };
  }

  // ===========================================================================
  //  PUBLIC SURFACE  (unchanged so the existing UI keeps working)
  // ===========================================================================
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
      setTimeout(function () { start().catch(function () {}); }, 100);
    });
  }
})();