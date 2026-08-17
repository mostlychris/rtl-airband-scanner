# RTL Scanner

A web-based RTL-SDR frequency scanner with a live browser UI, real-time audio streaming, and a glass-themed interface. Runs as a self-contained Python server — no external SDR software required.

<img width="379" height="222" alt="scannerscreen" src="https://github.com/user-attachments/assets/dad4d81b-d73c-4ba8-a655-9bdeebec69aa" />

## Requirements

- RTL-SDR dongle (RTL2832U-based)
- Linux host (Raspberry Pi or any x86 Linux)
- Python 3.9+

```bash
sudo apt install rtl-sdr python3-numpy ffmpeg
pip install fastapi "uvicorn[standard]" pyrtlsdr scipy
```

## Installation

```bash
git clone https://github.com/mostlychris/rtl-scanner.git
cd rtl-scanner
cp scanner_config.example.json scanner_config.json
nano scanner_config.json      # add your frequencies and labels
python3 app.py
```

Open **http://\<host-ip\>:8080** in any browser on your network.

## Running

```bash
python3 app.py                          # default port 8080
python3 app.py --listen-port 9000
python3 app.py --config /path/to/cfg.json
python3 app.py --debug                  # log squelch/CTCSS decisions per chunk
```

### Running as a systemd service

```ini
[Unit]
Description=RTL Airband Scanner
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/chris/rtl-airband-scanner/app.py
WorkingDirectory=/home/chris/rtl-airband-scanner
Restart=on-failure
User=chris

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable rtl-airband-scanner
sudo systemctl start rtl-airband-scanner
```

## Configuration

Copy `scanner_config.example.json` to `scanner_config.json`. All fields are optional except `channels`.

```json
{
  "name": "My Scanner",
  "host": "0.0.0.0",
  "port": 8080,
  "device": "0",
  "gain": "48",
  "squelch_rms": 0.05,
  "squelch_hold": 2.0,
  "ppm": 0,
  "modulation": "fm",
  "samp_rate": 240000,
  "scan_dwell": 0.25,
  "fir_taps": 127,
  "channels": {
    "154.250": "Fire Dispatch",
    "155.400": { "label": "EMS Channel", "squelch_rms": 0.08, "gain": "32" },
    "162.400": { "label": "NOAA Weather", "pl": 100.0 },
    "155.160": { "label": "Police", "modulation": "nfm", "bank": "Law" }
  }
}
```

### Top-level config keys

| Key | Description | Default |
|---|---|---|
| `name` | Scanner name shown in the browser UI | `"SDR Scanner"` |
| `host` | Listen address | `"0.0.0.0"` |
| `port` | HTTP port | `8080` |
| `device` | RTL-SDR device index or serial string | `"0"` |
| `gain` | RF gain in dB, or `"auto"` | `"48"` |
| `squelch_rms` | Phase-variance squelch threshold 0.0–1.0 | `0.05` |
| `squelch_hold` | Seconds to stay on a frequency after signal drops | `2.0` |
| `ppm` | PPM frequency correction for your dongle | `0` |
| `modulation` | Default demodulation mode: `"fm"`, `"nfm"`, or `"am"` | `"fm"` |
| `samp_rate` | Hardware sample rate in Hz (RTL-SDR valid: 225001–300000 or 900001–3200000) | `240000` |
| `scan_dwell` | Seconds to listen on each inactive frequency before moving on | `0.25` |
| `fir_taps` | Anti-alias FIR filter length (higher = sharper; more CPU) | `127` |

### Per-channel config fields

Channels can be a plain string label or an object with any of these fields:

| Field | Description |
|---|---|
| `label` | Channel name displayed in the UI |
| `bank` | Bank group name (e.g. `"Police"`, `"Fire"`) for bulk enable/disable |
| `modulation` | Per-channel mode: `"fm"`, `"nfm"`, or `"am"` |
| `squelch_rms` | Squelch threshold override for this channel |
| `gain` | RF gain override for this channel |
| `pl` | CTCSS/PL tone in Hz — squelch only opens when this tone is detected |
| `hp_filter` | Sub-audio notch filter in Hz — removes a specific tone from audio output without affecting CTCSS detection |
| `bandwidth` | Channel bandwidth in kHz (e.g. `12.5`, `25`) — auto-selects FM deviation |

## Browser UI

### Scanner card

Each scanner shows the current frequency or channel label, a segmented signal meter with live dB readout, and action buttons:

- **SKIP** — exclude the active frequency from the scan rotation (re-appears in the channel list as dimmed; click SCAN to restore)
- **HOLD** — lock the scanner to the current frequency indefinitely
- **NEXT** — advance immediately without waiting for the dwell timer
- **EDIT** — open the channel edit modal for the active frequency

### ☰ Channels modal

Click the **☰ Channels (N)** button in the scanner card header to open the full-width channel bank modal. From here you can:

- **Add** a new frequency (＋ Add Frequency at the bottom)
- **Edit** any channel — opens the channel edit modal with all per-channel fields
- **Skip / Scan** — toggle a channel in/out of the scan rotation
- **Delete** a channel
- **Hold** — click any channel row to tune directly to that frequency
- **Scan Banks** — if channels are assigned to named banks, toggle entire banks on/off at the top of the modal

All changes take effect immediately and persist to `scanner_config.json`.

### Channel edit modal

Accessible from the EDIT button on the active channel or from the Channels modal. Fields:

- Frequency (MHz), Label, Bank
- Mode (FM / NFM / AM), Channel Width (kHz)
- CTCSS / PL Tone (Hz)
- Sub-audio tone filter (Hz) — notch filter to remove a CTCSS tone from speaker output
- Squelch RMS, RF Gain

### Audio controls

Click **Audio** to start streaming. The controls bar appears beneath the scanner card:

- **Volume** knob
- **LP Cut** knob — low-pass filter cutoff (default 1.5 kHz; range 1–8 kHz)
- **SQ Tail** — toggle browser-side gate that mutes the noise burst at squelch close

Audio settings are persisted in `localStorage`.

### Activity log

Collapsible log at the bottom of the page showing the last 30 transmissions with timestamp, scanner name, frequency, and channel label.

## How it works

`app.py` is a single-file FastAPI server (~3800 lines) with the entire HTML/CSS/JS frontend embedded as a string constant. No build step, no frontend toolchain.

**Signal path (antenna → speaker):**

1. RTL-SDR dongle samples at 240 kHz IQ via `librtlsdr` (ctypes wrapper — no pyrtlsdr dependency at runtime)
2. 127-tap Blackman-Harris FIR lowpass + 10× decimation → 24 kHz audio rate
3. Demodulation: FM discriminator (conjugate product angle) or AM envelope detection
4. 75 µs de-emphasis IIR lowpass (FM/NFM only)
5. Optional per-channel sub-audio notch filter (`iirnotch`, Q=35) — removes CTCSS tone from speaker output while leaving the unfiltered signal available for CTCSS detection
6. Phase-variance squelch: `var(Δφ) ≈ π²/3` for noise; near 0 for a captured carrier
7. Optional CTCSS/PL gating — exact per-tone DFT on 4096-sample windows (~170 ms), ~5.9 Hz bin resolution
8. Raw 16-bit PCM → `/ws/audio` WebSocket and `/stream` WAV endpoint
9. Browser `AudioWorklet` ring buffer → LP `BiquadFilter` → volume `GainNode` → gate `GainNode` → speaker

**Scanner loop:**

The scanner opens the RTL-SDR once and retunes between frequencies with `set_center_freq` — no USB reconnect per hop. Skipped channels and disabled-bank channels are filtered out of the active list each iteration; they consume zero dwell time. When squelch opens the scanner locks to that frequency and streams audio until the signal drops, then holds for `squelch_hold` seconds. SKIP, HOLD, and bank toggles take effect at the start of the next iteration.

## REST API

| Method | Endpoint | Body / Params | Description |
|---|---|---|---|
| `PUT` | `/api/channel` | JSON: `freq`, `label`, `squelch_rms`, `gain`, `pl`, `bank`, `modulation`, `bandwidth`, `hp_filter` | Add or update a channel |
| `DELETE` | `/api/channel/{freq}` | — | Remove a channel |
| `POST` | `/api/skip` | JSON: `{ "freq": "154.250" }` | Toggle a frequency in/out of scan rotation |
| `POST` | `/api/hold` | JSON: `{ "freq": "154.250" }` or `{}` to release | Lock/unlock scanner to a frequency |
| `POST` | `/api/resume` | — | Force immediate scan advance (NEXT) |
| `POST` | `/api/bank` | JSON: `{ "bank": "Fire", "enabled": false }` | Enable or disable a scan bank |
| `GET` | `/debug` | — | Live scanner state as JSON |
| `WS` | `/ws` | — | Control WebSocket — full state on connect, then incremental events |
| `WS` | `/ws/audio` | Query: `?lp=1500` | Raw PCM stream (16-bit signed, mono, 24 kHz) |
| `GET` | `/stream` | — | WAV audio stream for `<audio>` element |
