# RTL Scanner

A web-based RTL-SDR frequency scanner with a live browser UI and real-time audio streaming. Runs as a self-contained Python server — no external SDR software required.

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

## Configuration

Copy `scanner_config.example.json` to `scanner_config.json`. All fields are optional except `channels`.

```json
{
  "name": "My Scanner",
    "host": "0.0.0.0",
      "port": 8080,
        "gain": "48",
          "squelch_rms": 0.05,
            "squelch_hold": 2.0,
              "ppm": 0,
                "modulation": "fm",
                  "samp_rate": 240000,
                    "scan_dwell": 0.5,
                      "channels": {
                          "154.250": "Fire Dispatch",
                              "155.400": { "label": "EMS Channel", "squelch_rms": 0.08, "gain": "32" },
                                  "162.400": { "label": "NOAA Weather", "pl": 100.0 }
                                    }
                                    }
                                    ```
                                    
                                    ### Channel config fields
                                    
                                    | Field | Description |
                                    |---|---|
                                    | `channels` | Map of `"frequency_mhz": "label"` or `"frequency_mhz": { ... }` |
                                    | `squelch_rms` | Phase-variance squelch threshold 0.0–1.0 (default `0.05`) |
                                    | `squelch_hold` | Seconds to hold on an active frequency after signal drops (default `2.0`) |
                                    | `gain` | SDR gain in dB, or `"auto"` (default `"48"`) |
                                    | `ppm` | PPM frequency correction for your dongle (default `0`) |
                                    | `modulation` | `"fm"` only (default `"fm"`) |
                                    | `samp_rate` | Hardware sample rate in Hz; must be RTL-SDR supported (default `240000`) |
                                    | `scan_dwell` | Seconds to listen on each inactive frequency before moving on (default `0.5`) |
                                    | `pl` | CTCSS/PL tone in Hz — squelch only opens when this tone is detected |
                                    
                                    Per-channel overrides (`squelch_rms`, `gain`, `pl`) take precedence over top-level defaults.
                                    
                                    ## Running
                                    
                                    ```bash
                                    python3 app.py                          # default port 8080
                                    python3 app.py --listen-port 9000
                                    python3 app.py --config /path/to/cfg.json
                                    python3 app.py --debug                  # log squelch/CTCSS decisions per chunk
                                    ```
                                    
                                    ## Browser UI features
                                    
                                    - **Live signal meter** — dB level updated in real time
                                    - **Audio streaming** — browser connects to `/stream` for WAV audio; starts automatically on page load
                                    - **HP / LP filters** — adjustable high-pass and low-pass in the audio chain
                                    - **Volume control** — per-session, persisted in localStorage
                                    - **SQ TAIL CUT** — mutes the brief noise burst at squelch close (browser-side gate)
                                    - **SKIP** — remove a frequency from the scan rotation without deleting it
                                    - **HOLD** — lock the scanner to one frequency indefinitely
                                    - **NEXT** — advance immediately without waiting for the dwell timer
                                    - **Channel bank** — add, edit, or remove channels live; changes persist in `scanner_config.json`
                                    - **Activity log** — scrollable history of recent transmissions with timestamps
                                    
                                    ## How it works
                                    
                                    `app.py` is a single-file FastAPI server with the entire HTML/CSS/JS UI embedded. No build step, no external dependencies beyond the pip packages above.
                                    
                                    **Signal path (antenna → speaker):**
                                    
                                    1. RTL-SDR dongle samples at 240 kHz IQ
                                    2. 127-tap FIR lowpass anti-alias filter + 10× decimation → 24 kHz
                                    3. FM discriminator (angle of conjugate product)
                                    4. 75 µs de-emphasis IIR
                                    5. Phase-variance squelch — noise gives `var(Δφ) ≈ π²/3`; a captured carrier drives it near zero
                                    6. Optional CTCSS/PL gating (FFT on 4096-sample windows, ~5.9 Hz bin resolution)
                                    7. Raw 16-bit PCM streamed to `/stream` (WAV) and `/ws/audio` (WebSocket)
                                    8. Browser Web Audio graph: HP filter → LP filter → volume → destination
                                    
                                    **Scanner loop:**
                                    
                                    The scanner tunes the SDR to each frequency in turn. When squelch opens it stays on that frequency and streams audio until the signal drops, then holds for `squelch_hold` seconds before scanning on. SKIP and HOLD are applied per loop iteration without reconnecting the USB device.
                                    
                                    ## REST API
                                    
                                    | Method | Endpoint | Description |
                                    |---|---|---|
                                    | `PUT` | `/api/channel` | Add or update a channel |
                                    | `DELETE` | `/api/channel/{freq}` | Remove a channel |
                                    | `POST` | `/api/skip` | Toggle a frequency in/out of scan rotation |
                                    | `POST` | `/api/hold` | Lock/unlock scanner to a frequency |
                                    | `POST` | `/api/resume` | Force immediate scan advance (NEXT) |
                                    | `GET` | `/debug` | Live scanner state (JSON) |
                                    | `WS` | `/ws` | Control channel — full state on connect, then events |
                                    | `WS` | `/ws/audio` | Raw PCM stream (16-bit signed, mono, 24 kHz) |
                                    | `GET` | `/stream` | WAV audio stream for `<audio>` element |
                                    