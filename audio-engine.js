// ClearEar audio engine
// Captures the selected microphone, runs the processing chain (8-band EQ,
// noise estimation, focus beam, master gain shaping), and feeds spectrum
// and dB telemetry back to the UI.
//
// PLAYBACK IS OFF BY DEFAULT. Routing the processed mic back to ctx.destination
// (system speakers) creates a feedback loop you hear as echo. Playback only
// connects when the user explicitly picks a non-default output device (earbuds,
// headphones) through the device picker. When the browser supports
// AudioContext.setSinkId, we route directly to that device.
//
// Circular audio buffer stores the last ~60 seconds for instant playback.

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

  let sceneConfig = {
    cancel: 0.18, speech: 0.35, transparency: 0.92, beam: false, name: 'Quiet space'
  };

  // Audio graph nodes, built lazily once we have a context
  let analyser = null;
  let masterGain = null;
  let splitter = null;
  let leftGain = null;
  let rightGain = null;
  let merger = null;
  let silenceSink = null;
  let proc = null;
  let eqFilters = [];

  const bandFreqs = [60, 170, 310, 600, 1000, 3000, 6000, 12000];

  let noiseFloor = 1e-5;
  let speechDetected = false;
  let dbSmooth = 35;

  // ─── Circular audio buffer (last ~65 seconds) ───
  const BUFFER_DURATION_SEC = 65;
  let ringBuffer = null;
  let ringBufferLength = 0;
  let ringWritePos = 0;
  let ringSampleRate = 48000;

  function initRingBuffer(sampleRate) {
    ringSampleRate = sampleRate;
    ringBufferLength = Math.ceil(sampleRate * BUFFER_DURATION_SEC);
    ringBuffer = new Float32Array(ringBufferLength);
    ringWritePos = 0;
  }

  function writeToRingBuffer(samples) {
    if (!ringBuffer) return;
    for (let i = 0; i < samples.length; i++) {
      ringBuffer[ringWritePos] = samples[i];
      ringWritePos = (ringWritePos + 1) % ringBufferLength;
    }
  }

  function getPlaybackBuffer(seconds) {
    if (!ringBuffer) return null;
    const requestedSamples = Math.min(Math.ceil(seconds * ringSampleRate), ringBufferLength);
    const out = new Float32Array(requestedSamples);
    let readPos = (ringWritePos - requestedSamples + ringBufferLength) % ringBufferLength;
    for (let i = 0; i < requestedSamples; i++) {
      out[i] = ringBuffer[readPos];
      readPos = (readPos + 1) % ringBufferLength;
    }
    return { samples: out, sampleRate: ringSampleRate };
  }

  function playBufferedAudio(seconds) {
    const data = getPlaybackBuffer(seconds);
    if (!data || !data.samples.length) return null;
    const playCtx = new (window.AudioContext || window.webkitAudioContext)();
    const buffer = playCtx.createBuffer(1, data.samples.length, data.sampleRate);
    buffer.getChannelData(0).set(data.samples);
    const source = playCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(playCtx.destination);
    source.start();
    source.onended = function () { playCtx.close(); };
    return { source: source, ctx: playCtx, durationSec: data.samples.length / data.sampleRate };
  }

  // ─── End ring buffer ───

  function ensureCtx() {
    if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
    return ctx;
  }

  function buildGraph() {
    ensureCtx();
    if (analyser) return;

    if (!ringBuffer) initRingBuffer(ctx.sampleRate);

    analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.75;

    silenceSink = ctx.createGain();
    silenceSink.gain.value = 0;

    masterGain = ctx.createGain();
    masterGain.gain.value = 1.0;

    splitter = ctx.createChannelSplitter(2);
    leftGain = ctx.createGain();
    rightGain = ctx.createGain();
    leftGain.gain.value = 1.0;
    rightGain.gain.value = 1.0;
    merger = ctx.createChannelMerger(2);

    eqFilters = bandFreqs.map(function (f, idx) {
      const b = ctx.createBiquadFilter();
      b.type = 'peaking';
      b.frequency.value = f;
      b.Q.value = 1.0;
      b.gain.value = (window.state && Array.isArray(window.state.eq)) ? (window.state.eq[idx] || 0) : 0;
      return b;
    });

    proc = ctx.createScriptProcessor(2048, 2, 2);
    proc.onaudioprocess = handleAudioProcess;
  }

  function rmsFromArray(buf) {
    let sum = 0;
    for (let i = 0; i < buf.length; i++) { const v = buf[i]; sum += v * v; }
    return Math.sqrt(sum / buf.length);
  }

  // Logarithmic frequency mapping for spectrum display
  function updateSpectrum() {
    if (!analyser) return;
    const binCount = analyser.frequencyBinCount;
    const data = new Uint8Array(binCount);
    analyser.getByteFrequencyData(data);

    if (!window.specData || !window.specData.length) return;
    const numBars = window.specData.length;
    const nyquist = (ctx ? ctx.sampleRate : 48000) / 2;
    const minFreq = 60;
    const maxFreq = 16000;
    const logMin = Math.log10(minFreq);
    const logMax = Math.log10(maxFreq);

    for (let i = 0; i < numBars; i++) {
      const logFreqLow = logMin + (i / numBars) * (logMax - logMin);
      const logFreqHigh = logMin + ((i + 1) / numBars) * (logMax - logMin);
      const freqLow = Math.pow(10, logFreqLow);
      const freqHigh = Math.pow(10, logFreqHigh);

      const binLow = Math.max(0, Math.floor(freqLow / nyquist * binCount));
      const binHigh = Math.min(binCount - 1, Math.ceil(freqHigh / nyquist * binCount));

      let sum = 0;
      let count = 0;
      for (let b = binLow; b <= binHigh; b++) {
        sum += data[b];
        count++;
      }
      const avg = count > 0 ? sum / count / 255 : 0;
      window.specData[i] = window.specData[i] * 0.55 + avg * 0.45;
    }
  }

  function setEqBands(gains) {
    if (!Array.isArray(gains)) return;
    gains.slice(0, eqFilters.length).forEach(function (gain, index) {
      if (eqFilters[index]) eqFilters[index].gain.value = Number(gain) || 0;
    });
  }

  function applyControls(controls) {
    if (!controls) return;
    sceneConfig = {
      cancel: controls.cancel ? 0.8 : 0.25,
      speech: controls.speech ? 0.75 : 0.2,
      transparency: controls.transparency ? 0.95 : 0.15,
      beam: !!controls.speech,
      name: sceneConfig.name
    };
  }

  function applyScene(scene) {
    if (!scene) return;
    sceneConfig = {
      cancel: Math.max(0, Math.min(1, (scene.cancel || 0) / 100)),
      speech: Math.max(0, Math.min(1, (scene.speech || 0) / 100)),
      transparency: Math.max(0, Math.min(1, (scene.transparency || 0) / 100)),
      beam: scene.beam !== false,
      name: scene.name || 'Scene'
    };
  }

  function updateDb() {
    if (!analyser) return;
    const t = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteTimeDomainData(t);
    let sum = 0;
    for (let i = 0; i < t.length; i++) { const v = (t[i] - 128) / 128; sum += v * v; }
    const rms = Math.sqrt(sum / t.length);
    const dbFs = 20 * Math.log10(Math.max(rms, 1e-8));

    // Map dBFS to approximate dB SPL
    // Typical: -60 dBFS = ~30 dB SPL (quiet room), -20 dBFS = ~70 dB SPL (conversation)
    // We use a simple linear mapping: dB SPL = dBFS + 90 (rough mic calibration)
    const dbSPL = Math.max(20, Math.min(110, dbFs + 90));
    
    // Faster smoothing so meter is responsive
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
        preset: window.state && window.state.preset ? window.state.preset : 'Unknown',
        timestamp: Date.now()
      });
    }
  }

  function handleAudioProcess(e) {
    try {
      for (let ch = 0; ch < e.outputBuffer.numberOfChannels; ch++) {
        e.outputBuffer.getChannelData(ch).fill(0);
      }

      const left = e.inputBuffer.getChannelData(0);
      const right = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : left;

      // Write to ring buffer (mono mix for playback)
      const monoMix = new Float32Array(left.length);
      for (let i = 0; i < left.length; i++) {
        monoMix[i] = (left[i] + (right !== left ? right[i] : left[i])) * 0.5;
      }
      writeToRingBuffer(monoMix);

      const lr = rmsFromArray(left);
      const rr = rmsFromArray(right);
      const curNoise = Math.max(lr, rr);

      if (!speechDetected) noiseFloor = noiseFloor * 0.995 + curNoise * 0.005;

      const freqData = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(freqData);
      const nyq = ctx.sampleRate / 2;
      const bandStart = Math.floor(300 / nyq * freqData.length);
      const bandEnd = Math.ceil(3000 / nyq * freqData.length);
      let freqSum = 0;
      for (let i = bandStart; i < bandEnd; i++) freqSum += freqData[i];
      const bandAvg = freqSum / Math.max(1, bandEnd - bandStart) / 255;

      speechDetected = bandAvg > 0.06 && (curNoise > noiseFloor * 1.3 || bandAvg > 0.08);

      if (!speechDetected) {
        const reduction = Math.max(0.35, 1 - ((curNoise - noiseFloor) * (1.25 + sceneConfig.cancel)));
        masterGain.gain.value = Math.max(reduction, sceneConfig.transparency * 0.95);
      } else {
        masterGain.gain.value = 1.0 + (sceneConfig.speech * 0.12);
      }

      const total = lr + rr + 1e-9;
      if (sceneConfig.beam) {
        const beamWidth = 0.15 + (1 - sceneConfig.cancel) * 0.2;
        leftGain.gain.value = 1 + ((lr - rr) / total) * beamWidth;
        rightGain.gain.value = 1 + ((rr - lr) / total) * beamWidth;
      } else {
        leftGain.gain.value = 1;
        rightGain.gain.value = 1;
      }

      const meta = document.querySelector('.vis-meta');
      if (meta) meta.textContent = speechDetected ? 'Speech detected, clarity boosted' : 'Ambient';
    } catch (err) {
      console.warn(err);
    }
  }

  async function start() {
    try {
      buildGraph();
      await ctx.resume();

      // If context is still suspended (autoplay policy), bail out - will retry on user gesture
      if (ctx.state === 'suspended') {
        console.log('[AudioEngine] Context suspended, waiting for user gesture');
        return;
      }

      if (!ringBuffer) initRingBuffer(ctx.sampleRate);

      if (!stream) {
        const audioConstraints = {
          channelCount: 2,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: false
        };
        if (inputDeviceId) audioConstraints.deviceId = { exact: inputDeviceId };
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
        } catch (err) {
          console.warn('Falling back to default microphone:', err);
          inputDeviceId = null;
          localStorage.removeItem('clearear.inputDevice');
          stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              channelCount: 2,
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: false
            }
          });
        }
      }
      if (!src) src = ctx.createMediaStreamSource(stream);
      if (connected) {
        // Already connected but maybe context was just resumed - restart the update interval
        if (!updateInterval) {
          updateInterval = setInterval(function () { updateSpectrum(); updateDb(); }, 1000 / 30);
        }
        return;
      }

      if (window.state && Array.isArray(window.state.eq)) setEqBands(window.state.eq);
      if (window.state && window.state.controls) applyControls(window.state.controls);
      if (window.clearEarSceneConfig) applyScene(window.clearEarSceneConfig);

      let node = src;
      eqFilters.forEach(function (f) { node.connect(f); node = f; });
      node.connect(analyser);

      // Connect analyser to splitter for stereo processing
      analyser.connect(splitter);
      splitter.connect(leftGain, 0);
      splitter.connect(rightGain, 1);
      leftGain.connect(merger, 0, 0);
      rightGain.connect(merger, 0, 1);
      merger.connect(masterGain);

      // CRITICAL: Connect masterGain to silenceSink so audio actually flows through
      // the analyser node even when playback is disabled. Without this, the analyser
      // sits in a dead branch and never gets data.
      masterGain.connect(silenceSink);
      silenceSink.connect(ctx.destination);

      // ScriptProcessor for ring buffer recording and speech detection
      if (!proc._connected) {
        node.connect(proc);
        proc.connect(ctx.createGain()); // dummy destination to keep proc alive
        proc._connected = true;
      }

      maybeEnablePlayback();

      if (!updateInterval) {
        updateInterval = setInterval(function () {
          updateSpectrum();
          updateDb();
        }, 1000 / 30);
      }

      connected = true;
      window.audioEngine.running = true;
      console.log('[AudioEngine] Connected and running. Sample rate:', ctx.sampleRate);
    } catch (err) {
      console.warn('Audio start failed:', err);
    }
  }

  function stop() {
    try {
      if (proc && proc._connected) { proc.disconnect(); proc._connected = false; }
      disablePlayback();
      [masterGain, merger, leftGain, rightGain, splitter, analyser]
        .concat(eqFilters)
        .forEach(function (n) { if (n) { try { n.disconnect(); } catch (e) {} } });
      if (updateInterval) { clearInterval(updateInterval); updateInterval = null; }
      connected = false;
      window.audioEngine.running = false;
    } catch (err) {}
  }

  function isDefaultOutput(id) {
    return !id || id === 'default' || id === '';
  }

  function maybeEnablePlayback() {
    if (!masterGain || !ctx) return;
    if (isDefaultOutput(outputDeviceId)) {
      disablePlayback();
      return;
    }
    if (playbackConnected) return;
    try {
      masterGain.connect(ctx.destination);
      playbackConnected = true;
      if (typeof ctx.setSinkId === 'function' && outputDeviceId) {
        ctx.setSinkId(outputDeviceId).catch(function (e) {
          console.warn('setSinkId not supported or failed:', e);
        });
      }
    } catch (e) {
      console.warn('Could not enable playback:', e);
    }
  }

  function disablePlayback() {
    if (!masterGain || !ctx) return;
    if (!playbackConnected) return;
    try { masterGain.disconnect(ctx.destination); } catch (e) {}
    playbackConnected = false;
  }

  async function setInputDevice(deviceId) {
    inputDeviceId = deviceId || null;
    if (inputDeviceId) localStorage.setItem('clearear.inputDevice', inputDeviceId);
    else localStorage.removeItem('clearear.inputDevice');
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
      src = null;
    }
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

  document.addEventListener('input', function (e) {
    const t = e.target;
    if (!t.matches || !t.matches('.eq-slider')) return;
    const idx = Number(t.dataset.band);
    const val = Number(t.value);
    if (eqFilters[idx]) eqFilters[idx].gain.value = val;
    if (window.state && Array.isArray(window.state.eq)) window.state.eq[idx] = val;
  });

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
    get ctx() { return ctx; }
  };

  const unlock = function () {
    if (document.getElementById('power-toggle') && document.getElementById('power-toggle').classList.contains('on')) {
      start();
    }
    document.removeEventListener('click', unlock);
    document.removeEventListener('keydown', unlock);
  };
  document.addEventListener('click', unlock);
  document.addEventListener('keydown', unlock);

  const powerEl = document.getElementById('power-toggle');
  if (powerEl && powerEl.classList.contains('on')) start();
})();