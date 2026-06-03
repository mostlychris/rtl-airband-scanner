# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
python3 app.py                          # default port 8080
python3 app.py --listen-port 9000
python3 app.py --config /path/to/cfg.json
python3 app.py --debug                  # log every chunk: freq, dB, squelch state
```

## Dependencies

```bash
sudo apt install rtl-sdr python3-numpy ffmpeg
pip install fastapi "uvicorn[standard]" pyrtlsdr scipy
```

## Configuration

`scanner_config.json` (copied from `scanner_config.example.json`) drives `app.py`. Channel values support two forms:

```json
"channels": {
  "446.000": "Label string",
  "443.700": { "label": "Repeater", "squelch_rms": 0.056, "gain": "25.4", "pl": 100.0 }
}
```

Top-level config keys accepted by `app.py`: `name`, `host`, `port`, `channels`, `skipped`, `squelch_rms`, `squelch_hold`, `ppm`, `modulation`, `device`, `gain`, `samp_rate`, `scan_dwell`, `fir_taps`.

Note: the old `squelch` integer key is ignored — use `squelch_rms` (float 0.0–1.0, default 0.05).

## Architecture

### `app.py`

Single-file FastAPI server (~2100 lines). The entire HTML/CSS/JS frontend is embedded in the `PAGE` string constant near the top.

**Backend layers (top to bottom in the file):**

1. **CTCSS detection** (`_ctcss_analyze`) — FFT-based PL tone detection on demodulated audio; gates squelch when a `pl` tone is configured per channel.

2. **`_RtlSdr` (ctypes wrapper, ~line 1070)** — thin wrapper around `librtlsdr.so` via ctypes. Handles `open`, `close`, `set_center_freq`, `set_sample_rate`, `set_gain`, `set_bandwidth`, `start_async`, `stop_async`. Also provides `_usb_reset` for USB power-cycling stale devices.

3. **`RTLFMScanner` (~line 1252)** — the core scan loop. Runs in a background thread. Opens the RTL-SDR once, then retuning between frequencies with `set_center_freq` (no USB open/close per hop). For each frequency dwell:
   - Reads IQ chunks from an `asyncio.Queue` fed by the `_RtlSdr` async callback
   - FIR anti-aliasing + stride decimation (scipy `lfilter`)
   - FM discriminator (angle of conjugate product)
   - 75 µs de-emphasis IIR
   - Phase-variance squelch with hysteresis
   - CTCSS gating (accumulates 4096-sample windows)
   - Emits `freq_change`/`freq_clear`/`signal` events to the FastAPI layer via `_emit()`
   - Emits raw 16-bit PCM via `_audio_cb()` when squelch is open

4. **`WsManager` + FastAPI endpoints (~line 1734)** — broadcasts scanner events to all WebSocket clients. REST API:
   - `PUT /api/channel` — add/edit a channel
   - `DELETE /api/channel/{freq}` — remove a channel
   - `POST /api/skip` — toggle a frequency in/out of the scan rotation
   - `POST /api/hold` — lock/unlock the scanner to a specific frequency
   - `POST /api/resume` — force an immediate scan advance (NEXT button)
   - `GET /debug` — live scanner state
   - `WS /ws` — control channel (sends full state on connect, then incremental events)
   - `WS /ws/audio` — raw PCM stream (16-bit signed, mono, 24 kHz)

**Frontend (embedded in `PAGE`):**

- Pure vanilla JS + Web Audio API. No build step.
- `AudioWorklet` (`PCMRingProcessor`) runs in the browser's audio render thread with a ring buffer for jitter-free playback. Falls back silently on plain HTTP (AudioWorklet requires HTTPS or localhost).
- Audio graph: `PCMRingProcessor` → HP BiquadFilter → LP BiquadFilter → Volume GainNode → Gate GainNode → destination. The gate is used for squelch tail suppression.
- Control WebSocket (`/ws`) drives all UI updates. Scanner state is normalized into `S.streams[mount]`.
- Audio settings (`vol`, `hp`, `lp`, `sqtail`) are persisted in `localStorage`.

## Signal processing details

- Hardware sample rate: 240 kHz default (must be an RTL-SDR supported rate)
- Decimation factor: `hw_rate / AUDIO_RATE` (e.g. 240000 / 24000 = 10×)
- FIR coefficients: 127-tap lowpass at `1.0 / decimate` (cutoff = audio Nyquist)
- FM scale: `AUDIO_RATE / (2π × 75000)` — normalises deviation for 75 kHz max deviation
- Squelch: phase-variance method; `var(Δφ) ≈ π²/3` for noise, near 0 for a captured carrier
- CTCSS window: 4096 samples (~170 ms at 24 kHz); FFT bin resolution ~5.9 Hz
