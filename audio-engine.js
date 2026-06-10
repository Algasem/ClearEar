// Audio engine for ClearEar
// Captures microphone, applies 8-band EQ, simple noise gating/VAD, and provides analyser data

(async function(){
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;

  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  let stream = null;
  let src = null;
  let eqOutput = null;
  let connected = false;
  let sceneConfig = {
    cancel: 0.18,
    speech: 0.35,
    transparency: 0.92,
    beam: false,
    name: 'Quiet space'
  };
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  const freqData = new Uint8Array(analyser.frequencyBinCount);
  const silenceSink = ctx.createGain();
  silenceSink.gain.value = 0;

  const masterGain = ctx.createGain();
  masterGain.gain.value = 1.0;

  // EQ band center freqs (matches UI labels)
  const bandFreqs = [60,170,310,600,1000,3000,6000,12000];
  const eqFilters = bandFreqs.map(f => {
    const b = ctx.createBiquadFilter();
    b.type = 'peaking';
    b.frequency.value = f;
    b.Q.value = 1.0;
    b.gain.value = (window.state && window.state.eq) ? window.state.eq[bandFreqs.indexOf(f)] || 0 : 0;
    return b;
  });

  // Per-channel gains for focus beam
  const splitter = ctx.createChannelSplitter(2);
  const leftGain = ctx.createGain();
  const rightGain = ctx.createGain();
  leftGain.gain.value = 1.0; rightGain.gain.value = 1.0;
  const merger = ctx.createChannelMerger(2);

  // ScriptProcessor for simple VAD/noise estimation
  const bufSize = 2048;
  const proc = ctx.createScriptProcessor(bufSize, 2, 2);

  let noiseFloor = 1e-5;
  let speechDetected = false;
  let dbOffset = 85;
  let dbSmooth = 42;
  let dbCalFrames = 0;
  let dbCalAccum = 0;

  // Simple RMS helper
  function rmsFromArray(buf) {
    let sum = 0;
    for (let i=0;i<buf.length;i++){ const v = buf[i]; sum += v*v; }
    return Math.sqrt(sum / buf.length);
  }

  // Update UI-visible spectrum (mutates global specData array used by drawSpectrum)
  function updateSpectrum() {
    analyser.getByteFrequencyData(freqData);
    if (window.specData && window.specData.length) {
      const n = Math.min(window.specData.length, freqData.length);
      for (let i=0;i<n;i++) window.specData[i] = freqData[i] / 255;
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

  // Update dB readout using analyser time domain
  function updateDb() {
    const t = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteTimeDomainData(t);
    // Compute dBFS from time domain and map to a rough dB SPL estimate via startup calibration.
    let sum=0; for (let i=0;i<t.length;i++){ const v=(t[i]-128)/128; sum+=v*v; }
    const rms = Math.sqrt(sum / t.length);
    const dbFs = 20 * Math.log10(Math.max(rms, 1e-8));

    if (dbCalFrames < 90) {
      dbCalAccum += dbFs;
      dbCalFrames += 1;
      if (dbCalFrames === 90) {
        const avgDbFs = dbCalAccum / dbCalFrames;
        // Anchor startup ambient to ~35 dB so quiet rooms don't read as 90+.
        dbOffset = 35 - avgDbFs;
      }
    }

    const estimate = Math.max(20, Math.min(110, dbFs + dbOffset));
    dbSmooth = dbSmooth * 0.85 + estimate * 0.15;
    if (window.state) window.state.db = Math.round(dbSmooth);
    const el = document.getElementById('db-readout'); if (el) el.textContent = Math.round(window.state.db);
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

  // Hook EQ slider changes (delegated)
  document.addEventListener('input', e => {
    const t = e.target; if (!t.matches || !t.matches('.eq-slider')) return;
    const idx = Number(t.dataset.band);
    const val = Number(t.value);
    if (eqFilters[idx]) eqFilters[idx].gain.value = val;
    if (window.state && Array.isArray(window.state.eq)) window.state.eq[idx] = val;
  });

  // Power toggle listener to start/stop audio
  const powerEl = document.getElementById('power-toggle');
  if (powerEl) {
    powerEl.addEventListener('click', () => {
      const running = powerEl.classList.contains('on');
      if (running) start(); else stop();
    });
  }

  async function start() {
    try {
      await ctx.resume();
      if (!stream) {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 2,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: false
          }
        });
      }
      if (!src) src = ctx.createMediaStreamSource(stream);
      if (connected) return;

      dbCalFrames = 0;
      dbCalAccum = 0;

      if (window.state && Array.isArray(window.state.eq)) {
        setEqBands(window.state.eq);
      }
      if (window.state && window.state.controls) {
        applyControls(window.state.controls);
      }
      if (window.clearEarSceneConfig) {
        applyScene(window.clearEarSceneConfig);
      }

      // Connect: src -> EQ chain -> analyser -> splitter -> L/R gains -> merger -> master -> destination
      let node = src;
      // connect through EQ filters in series
      eqFilters.forEach(f => { node.connect(f); node = f; });
      eqOutput = node;
      node.connect(analyser);
      analyser.connect(splitter);

      splitter.connect(leftGain, 0); splitter.connect(rightGain, 1);
      leftGain.connect(merger, 0, 0);
      rightGain.connect(merger, 0, 1);

      // after merging, feed to masterGain then destination
      merger.connect(masterGain);
      masterGain.connect(ctx.destination);

      // connect proc for VAD (non-blocking)
      if (!proc._connected) {
        node.connect(proc);
        proc.connect(silenceSink);
        silenceSink.connect(ctx.destination);
        proc._connected = true;
      }

      // Initialize filter gains from state
      if (window.state && Array.isArray(window.state.eq)) {
        window.state.eq.forEach((v,i)=>{ if (eqFilters[i]) eqFilters[i].gain.value = v; });
      }

      // periodic updates
      if (!start._interval) start._interval = setInterval(()=>{ updateSpectrum(); updateDb(); }, 1000/30);
      connected = true;
      window.audioEngine.running = true;
      const status = document.getElementById('captions-status-text');
      if (status && status.textContent === 'Speech recognition unavailable') {
        status.textContent = 'Transcribing live';
      }
    } catch (err) {
      console.warn('Audio start failed', err);
    }
  }

  function stop() {
    try {
      // disconnect
      if (proc && proc._connected) { proc.disconnect(); proc._connected = false; }
      if (masterGain) masterGain.disconnect();
      if (merger) merger.disconnect();
      if (leftGain) leftGain.disconnect();
      if (rightGain) rightGain.disconnect();
      if (splitter) splitter.disconnect();
      if (analyser) analyser.disconnect();
      if (eqOutput) eqOutput.disconnect();
      if (start._interval) { clearInterval(start._interval); start._interval = null; }
      connected = false;
      window.audioEngine.running = false;
    } catch(e){/* ignore */}
  }

  // ScriptProcessor processing for VAD/noise gate
  proc.onaudioprocess = function(e){
    try {
      for (let ch = 0; ch < e.outputBuffer.numberOfChannels; ch++) {
        e.outputBuffer.getChannelData(ch).fill(0);
      }

      const left = e.inputBuffer.getChannelData(0);
      const right = e.inputBuffer.numberOfChannels>1 ? e.inputBuffer.getChannelData(1) : left;
      const lr = rmsFromArray(left); const rr = rmsFromArray(right);
      const curNoise = Math.max(lr, rr);

      // slowly adapt noise floor when no speech
      if (!speechDetected) noiseFloor = noiseFloor * 0.995 + curNoise * 0.005;

      // speech band energy check using analyser (approx)
      analyser.getByteFrequencyData(freqData);
      const nyq = ctx.sampleRate/2;
      const bandStart = Math.floor(300 / nyq * freqData.length);
      const bandEnd = Math.ceil(3000 / nyq * freqData.length);
      let sum = 0; for (let i=bandStart;i<bandEnd;i++) sum += freqData[i];
      const bandAvg = sum / Math.max(1, bandEnd-bandStart) / 255;

      speechDetected = bandAvg > 0.06 && (curNoise > noiseFloor*1.3 || bandAvg > 0.08);

      // adjust master gain slightly when heavy noise and no speech (simple NR)
      if (!speechDetected) {
        const reduction = Math.max(0.35, 1 - ((curNoise - noiseFloor) * (1.25 + sceneConfig.cancel)));
        masterGain.gain.value = Math.max(reduction, sceneConfig.transparency * 0.95);
      } else {
        masterGain.gain.value = 1.0 + (sceneConfig.speech * 0.12);
      }

      // focus beam: compare left/right energy and boost stronger side slightly
      const l = lr; const r = rr;
      const total = l + r + 1e-9;
      if (sceneConfig.beam) {
        const beamWidth = 0.15 + (1 - sceneConfig.cancel) * 0.2;
        const lGain = 1 + ((l - r) / total) * beamWidth;
        const rGain = 1 + ((r - l) / total) * beamWidth;
        leftGain.gain.value = lGain;
        rightGain.gain.value = rGain;
      } else {
        leftGain.gain.value = 1;
        rightGain.gain.value = 1;
      }

      // update UI speech indicator
      const meta = document.querySelector('.vis-meta');
      if (meta) meta.textContent = speechDetected ? 'Speech detected · clarity boosted' : 'Ambient';

    } catch (err) { console.warn(err); }
  };

  // expose start/stop for console
  window.audioEngine = { start, stop, ctx, running: false, setEqBands, applyScene, applyControls };

  // auto-start if power toggle was on
  if (document.getElementById('power-toggle')?.classList.contains('on')) start();

  // Ensure first user gesture can resume audio on stricter autoplay policies.
  const unlock = () => {
    if (document.getElementById('power-toggle')?.classList.contains('on')) start();
    document.removeEventListener('click', unlock);
    document.removeEventListener('keydown', unlock);
  };
  document.addEventListener('click', unlock);
  document.addEventListener('keydown', unlock);

})();
