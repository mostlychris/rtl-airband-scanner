# RTL-Airband Scanner

Terminal UI for [RTLSDR-Airband](https://github.com/rtl-airband/RTLSDR-Airband).
Connects to one or more local Icecast streams, reads ICY metadata to detect the
active frequency in real time, and shows a live display in your SSH terminal.

```
┌─ RTL-Airband Scanner ─────────────────────────────────────────────────────────┐
│ ┌─ Ham / GMRS  /ham.mp3  ● connected ──┐  ┌─ Air Traffic  /air.mp3  ● ──┐    │
│ │  Frequency     Label        Since    │  │  Frequency   Label   Since  │    │
│ │  443.700 MHz   443.700   ○           │  │  ...                        │    │
│ │  446.000 MHz   446.000   ◉ ACTIVE    │  │                             │    │
│ │                09:14:32              │  │                             │    │
│ └──────────────────────────────────────┘  └─────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Requirements

- Raspberry Pi running RTLSDR-Airband
- An Icecast server receiving the RTLSDR-Airband output
- `send_scan_freq_tags = true` in each Icecast output block in `rtl_airband.conf`
- Python 3.9+

## Installation (Pi)

```bash
pip3 install fastapi "uvicorn[standard]"
git clone https://github.com/mostlychris/rtl-airband-scanner.git
cd rtl-airband-scanner
cp scanner_config.example.json scanner_config.json
nano scanner_config.json      # set host IP and add channel labels
python3 app.py
```

Then open **http://\<pi-ip\>:8080** in any browser on your network.

## Configuration

Copy `scanner_config.example.json` to `scanner_config.json` and fill in:

| Field | Description |
|---|---|
| `host` | IP or hostname of your Icecast server |
| `port` | Icecast port (default 8000) |
| `streams[].mount` | Mount point, e.g. `/ham.mp3` |
| `streams[].name` | Display name shown in the UI |
| `streams[].channels` | Map of `"frequency": "label"` pairs |

Channel labels are optional — if omitted the active frequency is auto-detected
from the ICY stream title.

### RTLSDR-Airband config snippet

Each Icecast output block needs `send_scan_freq_tags = true`:

```
outputs:(
  {
    type = "icecast";
    server = "YOUR_ICECAST_IP";
    port = 8000;
    mountpoint = "ham.mp3";
    send_scan_freq_tags = true;
  }
);
```

## Usage

```bash
python3 app.py                          # default port 8080
python3 app.py --listen-port 9000       # custom port
python3 app.py --config /path/to/cfg.json
```

## How it works

- **Frequency detection** — the Pi connects to each Icecast stream and reads
  the ICY `StreamTitle` metadata (updated by RTLSDR-Airband when squelch opens).
- **Real-time UI** — a WebSocket pushes every channel change to the browser instantly.
- **Audio** — the browser connects to `/audio/<mount>` on the Pi; Python proxies
  the Icecast stream directly with no re-encoding. Click the audio button in the
  browser to start; the player auto-switches to whichever stream just became active.
  Click a stream card to lock audio to that stream.
