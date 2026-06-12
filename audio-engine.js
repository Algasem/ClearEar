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
  const freqData = new Uint8Array(1024);

  let noiseFloor = 1e-5;
  let speechDetected = false;
  let dbOffset = 85;
  let dbSmooth = 42;
  let dbCalFrames = 0;
  let dbCalAccum = 0;

  function ensureCtx() {
    if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
    return ctx;
  }

  function buildGraph() {
    ensureCtx();
    if (analyser) return;

    analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;

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

  function updateSpectrum() {
    if (!analyser) return;
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    if (window.specData && window.specData.length) {
      const n = Math.min(window.specData.length, data.length);
      for (let i = 0; i < n; i++) window.specData[i] = data[i] / 255;
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

    if (dbCalFrames < 90) {
      dbCalAccum += dbFs;
      dbCalFrames += 1;
      if (dbCalFrames === 90) {
        const avgDbFs = dbCalAccum / dbCalFrames;
        dbOffset = 35 - avgDbFs;
      }
    }

    const estimate = Math.max(20, Math.min(110, dbFs + dbOffset));
    dbSmooth = dbSmooth * 0.85 + estimate * 0.15;
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
      // Zero the output buffer so nothing reaches destination through this path
      for (let ch = 0; ch < e.outputBuffer.numberOfChannels; ch++) {
        e.outputBuffer.getChannelData(ch).fill(0);
      }

      const left = e.inputBuffer.getChannelData(0);
      const right = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : left;
      const lr = rmsFromArray(left);
      const rr = rmsFromArray(right);
      const curNoise = Math.max(lr, rr);

      if (!speechDetected) noiseFloor = noiseFloor * 0.995 + curNoise * 0.005;

      analyser.getByteFrequencyData(freqData);
      const nyq = ctx.sampleRate / 2;
      const bandStart = Math.floor(300 / nyq * freqData.length);
      const bandEnd = Math.ceil(3000 / nyq * freqData.length);
      let sum = 0;
      for (let i = bandStart; i < bandEnd; i++) sum += freqData[i];
      const bandAvg = sum / Math.max(1, bandEnd - bandStart) / 255;

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
          // If the saved device is no longer available, fall back to default
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
      if (connected) return;

      dbCalFrames = 0;
      dbCalAccum = 0;

      if (window.state && Array.isArray(window.state.eq)) setEqBands(window.state.eq);
      if (window.state && window.state.controls) applyControls(window.state.controls);
      if (window.clearEarSceneConfig) applyScene(window.clearEarSceneConfig);

      // Chain: src -> EQ filters in series -> analyser -> splitter -> L/R gains -> merger -> masterGain
      // masterGain is intentionally NOT connected to ctx.destination by default.
      let node = src;
      eqFilters.forEach(function (f) { node.connect(f); node = f; });
      node.connect(analyser);
      analyser.connect(splitter);
      splitter.connect(leftGain, 0);
      splitter.connect(rightGain, 1);
      leftGain.connect(merger, 0, 0);
      rightGain.connect(merger, 0, 1);
      merger.connect(masterGain);

      // Sidechain for processing telemetry (no audio reaches destination through this path)
      if (!proc._connected) {
        node.connect(proc);
        proc.connect(silenceSink);
        silenceSink.connect(ctx.destination);
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
    } catch (err) {
      // ignore
    }
  }

  function isDefaultOutput(id) {
    return !id || id === 'default' || id === '';
  }

  function maybeEnablePlayback() {
    if (!masterGain || !ctx) return;
    if (isDefaultOutput(outputDeviceId)) {
      // Default = system speakers, would cause feedback. Stay silent.
      disablePlayback();
      return;
    }
    if (playbackConnected) return;
    try {
      masterGain.connect(ctx.destination);
      playbackConnected = true;
      // Route to the specific output if supported
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

    if (connected) {
      stop();
      await start();
    }
  }

  async function setOutputDevice(deviceId) {
    outputDeviceId = deviceId || null;
    if (outputDeviceId) localStorage.setItem('clearear.outputDevice', outputDeviceId);
    else localStorage.removeItem('clearear.outputDevice');
    maybeEnablePlayback();
  }

  function isPlaybackEnabled() {
    return playbackConnected;
  }

  function getInputDeviceId() { return inputDeviceId; }
  function getOutputDeviceId() { return outputDeviceId; }

  // Live EQ slider hookup
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
    get ctx() { return ctx; }
  };

  // Try to resume on the first user gesture so autoplay policies do not block us
  const unlock = function () {
    if (document.getElementById('power-toggle') && document.getElementById('power-toggle').classList.contains('on')) {
      start();
    }
    document.removeEventListener('click', unlock);
    document.removeEventListener('keydown', unlock);
  };
  document.addEventListener('click', unlock);
  document.addEventListener('keydown', unlock);

  // Auto-start if power is already on at load
  const powerEl = document.getElementById('power-toggle');
  if (powerEl && powerEl.classList.contains('on')) start();
})();