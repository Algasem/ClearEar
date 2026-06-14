// ClearEar replay recorder  —  AudioWorklet
// =============================================================================
// Maintains a ~65 second circular buffer of the (already processed) hearing-aid
// output, entirely on the audio render thread. This replaces the old
// ScriptProcessorNode, which ran on the main thread and forced the whole audio
// context to buffer ahead, adding hundreds of milliseconds of latency.
//
// A pass-through worklet processes audio in 128-sample render quanta, so it adds
// no meaningful latency to the live signal. The main thread only talks to it
// when the user actually asks for an instant replay (a "grab" message).
// =============================================================================

class ClearEarRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufLen = Math.max(1, Math.ceil(sampleRate * 65)); // sampleRate is a worklet global
    this.buf = new Float32Array(this.bufLen);
    this.writePos = 0;

    this.port.onmessage = (e) => {
      const msg = e.data || {};
      if (msg.cmd === 'grab') {
        const want = Math.min(Math.ceil((msg.seconds || 10) * sampleRate), this.bufLen);
        const out = new Float32Array(want);
        let read = (this.writePos - want + this.bufLen) % this.bufLen;
        for (let i = 0; i < want; i++) {
          out[i] = this.buf[read];
          read = (read + 1) % this.bufLen;
        }
        this.port.postMessage({ id: msg.id, samples: out, sampleRate: sampleRate }, [out.buffer]);
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const ch = input[0];
      const buf = this.buf;
      const len = this.bufLen;
      let w = this.writePos;
      for (let i = 0; i < ch.length; i++) {
        buf[w] = ch[i];
        w = (w + 1) % len;
      }
      this.writePos = w;
    }
    return true;
  }
}

registerProcessor('clearear-recorder', ClearEarRecorder);