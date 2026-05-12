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

- Raspberry Pi (or any Linux box) running RTLSDR-Airband
- An Icecast server receiving the RTLSDR-Airband output
- `send_scan_freq_tags = true` in each Icecast output block in `rtl_airband.conf`

## Installation

```bash
sudo apt install ffmpeg python3-pip
pip3 install rich
git clone https://github.com/mostlychris/rtl-airband-scanner.git
cd rtl-airband-scanner
cp scanner_config.example.json scanner_config.json
# Edit scanner_config.json with your Icecast server IP and channel labels
```

## Configuration

Copy `scanner_config.example.json` to `scanner_config.json` and fill in:

| Field | Description |
|---|---|
| `host` | IP or hostname of your Icecast server |
| `port` | Icecast port (default 8000) |
| `streams[].mount` | Mount point, e.g. `/ham.mp3` |
| `streams[].name` | Display name shown in the UI |
| `streams[].channels` | Map of `"frequency": "label"` pairs |

Channel labels are optional — if omitted the app auto-detects the active frequency
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
# Use scanner_config.json (recommended)
python3 scanner.py

# Override host/mounts on the command line
python3 scanner.py --host 192.168.1.100 --mount /ham.mp3 --mount /air.mp3

# Display only — audio handled by RTLSDR-Airband PulseAudio (default)
python3 scanner.py

# Let Python handle audio via PulseAudio (disable rtl_airband pulse output first)
python3 scanner.py --audio
```

## Audio

By default the scanner is **display only** — audio continues through the
existing RTLSDR-Airband → PulseAudio path.

Pass `--audio` to have Python route the Icecast audio through `ffmpeg` →
PulseAudio instead. This gives tighter control over buffering but you must
disable the corresponding `pulse` output in `rtl_airband.conf` first to avoid
hearing every transmission twice.
