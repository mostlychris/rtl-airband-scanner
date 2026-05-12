#!/usr/bin/env python3
"""
RTL-Airband Scanner — Raspberry Pi edition
Monitors one or more RTLSDR-Airband Icecast streams, shows the active
frequency per stream in a live terminal UI.

Default mode: display only.
  Audio continues via the existing RTLSDR-Airband → PulseAudio → PC path.

Optional --audio mode:
  Routes Icecast audio through ffmpeg → PulseAudio so the Python app owns
  the buffer.  Disable the rtl_airband pulse output for that stream first
  to avoid hearing the same transmission twice.

Pi setup:
    sudo apt install ffmpeg python3-pip
    pip3 install rich

RTLSDR-Airband requires:  send_scan_freq_tags = true  in each icecast output.
This makes the scanner post the active frequency as the ICY StreamTitle when
the squelch opens.

Quickstart for your setup:
    python3 scanner.py                               # uses scanner_config.json
    python3 scanner.py --host 172.31.10.192 \\
        --mount /ham.mp3 --mount /air.mp3

Config file (scanner_config.json — place next to scanner.py):
{
  "host": "172.31.10.192",
  "port": 8000,
  "streams": [
    { "mount": "/ham.mp3", "name": "Ham Radio" },
    { "mount": "/air.mp3", "name": "Air Traffic" }
  ],
  "channels": {
    "146.940": "Rptr 146.940",
    "118.100": "ATIS",
    "119.000": "Ground"
  }
}
"""

from __future__ import annotations

import re
import sys
import json
import time
import shutil
import struct
import socket
import threading
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import deque

# ── Third-party: only rich is required ─────────────────────────────────────────
try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel
    from rich.columns import Columns
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None
    print("Install rich for a better display:  pip3 install rich")

# ── Audio ───────────────────────────────────────────────────────────────────────
FFMPEG      = shutil.which("ffmpeg")
SAMPLE_RATE = 44100
CHANNELS    = 1


def _detect_audio_backend() -> str | None:
    if shutil.which("pactl"):
        try:
            r = subprocess.run(["pactl", "info"], capture_output=True, timeout=3)
            if r.returncode == 0:
                return "pulse"
        except Exception:
            pass
    if shutil.which("aplay"):
        return "alsa"
    return None


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    if console:
        console.print(f"[dim]{ts}[/] {msg}")
    else:
        print(f"{ts}  {msg}")


# ── ICY stream ──────────────────────────────────────────────────────────────────

class IcyStream:
    """
    Raw socket connection to an Icecast endpoint.
    Strips the interleaved ICY metadata blocks from the audio bytes so
    both can be consumed independently.
    """
    RECV_SIZE = 2048

    def __init__(self, host: str, port: int, mount: str,
                 user: str = "", password: str = ""):
        self.host          = host
        self.port          = port
        self.mount         = mount
        self.user          = user
        self.password      = password
        self.metaint       = 0
        self.current_title = ""
        self.content_type  = ""
        self._sock         = None
        self._closed       = False

    def connect(self) -> dict:
        self._sock = socket.create_connection((self.host, self.port), timeout=10)
        self._sock.settimeout(30)

        lines = [
            f"GET {self.mount} HTTP/1.0",
            f"Host: {self.host}:{self.port}",
            "User-Agent: RTLAirbandScanner/1.0",
            "Icy-MetaData: 1",
        ]
        if self.user:
            import base64
            creds = base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
            lines.append(f"Authorization: Basic {creds}")
        lines += ["Connection: close", "", ""]
        self._sock.sendall("\r\n".join(lines).encode())

        raw = b""
        while b"\r\n\r\n" not in raw:
            b = self._sock.recv(1)
            if not b:
                raise ConnectionError("Server closed during handshake")
            raw += b

        headers = {}
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                headers[k.strip().lower()] = v.strip()

        self.metaint      = int(headers.get("icy-metaint", 0))
        self.content_type = headers.get("content-type", "unknown")
        return headers

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise EOFError("Stream ended")
            buf.extend(chunk)
        return bytes(buf)

    def _read_meta_block(self):
        length = struct.unpack("B", self._recv_exact(1))[0] * 16
        if length:
            raw = self._recv_exact(length).decode("utf-8", errors="replace").rstrip("\x00")
            m = re.search(r"StreamTitle='([^']*)'", raw)
            if m:
                self.current_title = m.group(1)

    def iter_audio(self):
        """
        Yield clean audio bytes (metadata stripped).
        Must be continuously consumed — the socket buffer stalls otherwise.
        """
        if not self.metaint:
            while not self._closed:
                try:
                    data = self._sock.recv(self.RECV_SIZE)
                except Exception:
                    break
                if not data:
                    break
                yield data
            return

        remaining = self.metaint
        while not self._closed:
            to_read = min(self.RECV_SIZE, remaining)
            try:
                data = self._sock.recv(to_read)
            except Exception:
                break
            if not data:
                break
            yield data
            remaining -= len(data)
            if remaining <= 0:
                try:
                    self._read_meta_block()
                except Exception:
                    break
                remaining = self.metaint

    def close(self):
        self._closed = True
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass


# ── Per-stream monitor ──────────────────────────────────────────────────────────

class StreamMonitor:
    """
    Manages one Icecast connection in a background thread.
    Tracks active frequency from ICY StreamTitle.
    Thread-safe for reading active_freq / history from the UI thread.
    """
    RECONNECT_DELAY = 5

    def __init__(self, name: str, host: str, port: int, mount: str,
                 channels: dict, user: str = "", password: str = ""):
        self.name     = name
        self.host     = host
        self.port     = port
        self.mount    = mount
        self.channels = channels   # {freq_str: label}
        self.user     = user
        self.password = password

        self._stream        = None
        self._active_freq   = None
        self._active_since: datetime | None = None
        self._history: deque[tuple[datetime, str, str]] = deque(maxlen=20)
        self._lock          = threading.Lock()
        self._audio_queue: deque[bytes] = deque(maxlen=8)
        self._running       = False
        self._connected     = False

    # ── public (thread-safe) ────────────────────────────────────────────────────

    @property
    def active_freq(self) -> str | None:
        with self._lock:
            return self._active_freq

    @property
    def active_since(self) -> datetime | None:
        with self._lock:
            return self._active_since

    @property
    def history(self) -> list:
        with self._lock:
            return list(self._history)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def current_title(self) -> str:
        return self._stream.current_title if self._stream else ""

    def drain_audio(self) -> bytes | None:
        """Return next audio chunk, or None if queue is empty."""
        try:
            return self._audio_queue.popleft()
        except IndexError:
            return None

    # ── lifecycle ───────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True,
                             name=f"stream-{self.mount}")
        t.start()

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.close()

    # ── internal ────────────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            self._connected = False
            self._stream    = IcyStream(
                self.host, self.port, self.mount, self.user, self.password
            )
            try:
                self._stream.connect()
                self._connected = True
            except Exception as exc:
                _log(f"[red]{self.name}: connection failed — {exc}[/]")
                time.sleep(self.RECONNECT_DELAY)
                continue

            for chunk in self._stream.iter_audio():
                if not self._running:
                    break
                self._audio_queue.append(chunk)

                title = self._stream.current_title
                freq  = _match_title(title, self.channels)
                with self._lock:
                    if freq and freq != self._active_freq:
                        self._active_freq  = freq
                        self._active_since = datetime.now()
                        label = self.channels.get(freq, "")
                        self._history.appendleft((datetime.now(), freq, label))

            if self._running:
                self._connected = False
                _log(f"[yellow]{self.name}: stream dropped — reconnecting in "
                     f"{self.RECONNECT_DELAY}s[/]")
                time.sleep(self.RECONNECT_DELAY)


# ── Channel matching ────────────────────────────────────────────────────────────

def _match_title(title: str, channels: dict) -> str | None:
    """
    Map an ICY StreamTitle to a known channel frequency key.
    RTLSDR-Airband (send_scan_freq_tags = true) sets StreamTitle to the
    channel label, typically the frequency string e.g. "118.100".
    """
    if not title:
        return None
    for freq, label in channels.items():
        if freq in title or (label and label in title):
            return freq
    m = re.search(r"(\d{2,3}\.\d{1,4})", title)
    return m.group(1) if m else None


# ── Audio sink (optional) ───────────────────────────────────────────────────────

class AudioSink:
    """
    Routes compressed audio → ffmpeg → PulseAudio or ALSA.
    ffmpeg's stdin is the raw Icecast audio bytes (MP3 or whatever format).
    PulseAudio on the Pi forwards automatically to your PC.
    """

    def __init__(self, backend: str, stream_name: str = "RTL-Airband Scanner"):
        self._backend     = backend
        self._stream_name = stream_name
        self._proc        = None

    def start(self):
        if not FFMPEG:
            _log("[yellow]ffmpeg not found — audio disabled (sudo apt install ffmpeg)[/]")
            return

        sink = (
            ["-f", "pulse", self._stream_name]
            if self._backend == "pulse"
            else ["-f", "alsa", "default"]
        )
        self._proc = subprocess.Popen(
            [
                FFMPEG, "-loglevel", "quiet",
                "-probesize",       "32",    # minimise startup buffering
                "-analyzeduration", "0",
                "-i", "pipe:0",
                "-ar", str(SAMPLE_RATE),
                "-ac", str(CHANNELS),
                "-af", "aresample=async=1",  # absorb clock drift silently
            ] + sink,
            stdin=subprocess.PIPE,
            bufsize=0,
        )

    def feed(self, data: bytes):
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(data)
            except (BrokenPipeError, OSError):
                pass

    def stop(self):
        if self._proc:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


# ── Main scanner ────────────────────────────────────────────────────────────────

class Scanner:
    def __init__(self, monitors: list[StreamMonitor],
                 sink: AudioSink | None = None,
                 audio_mount: str = ""):
        self._monitors    = monitors
        self._sink        = sink
        self._audio_mount = audio_mount  # which mount to route audio from
        self._running     = False

    def run(self):
        self._running = True

        for m in self._monitors:
            m.start()

        if self._sink:
            self._sink.start()
            # Feed audio from the chosen stream to the sink in a background thread
            threading.Thread(
                target=self._audio_loop, daemon=True, name="audio-feed"
            ).start()

        try:
            self._ui_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            for m in self._monitors:
                m.stop()
            if self._sink:
                self._sink.stop()

    def _audio_loop(self):
        target_mount = self._audio_mount or (
            self._monitors[0].mount if self._monitors else ""
        )
        while self._running:
            fed = False
            for m in self._monitors:
                if m.mount == target_mount:
                    chunk = m.drain_audio()
                    if chunk and self._sink and self._sink.alive:
                        self._sink.feed(chunk)
                        fed = True
            if not fed:
                time.sleep(0.005)

    # ── UI ──────────────────────────────────────────────────────────────────────

    def _ui_loop(self):
        if HAS_RICH:
            with Live(refresh_per_second=8, screen=False) as live:
                while self._running:
                    live.update(self._render())
                    time.sleep(0.12)
        else:
            seen = {}
            while self._running:
                for m in self._monitors:
                    freq = m.active_freq
                    if freq and seen.get(m.mount) != freq:
                        seen[m.mount] = freq
                        label = m.channels.get(freq, "")
                        ts    = datetime.now().strftime("%H:%M:%S")
                        print(f"{ts}  [{m.name}]  ACTIVE: {freq} MHz  {label}")
                time.sleep(0.2)

    def _render(self):
        panels = []
        for m in self._monitors:
            panels.append(self._render_stream(m))

        title_parts = ["[bold blue]RTL-Airband Scanner[/]"]
        if self._sink and self._sink.alive:
            title_parts.append(
                f"[dim]audio → {_detect_audio_backend() or '?'}[/]"
            )
        else:
            title_parts.append("[dim]audio: RTLSDR-Airband PulseAudio[/]")

        if len(panels) == 1:
            return Panel(
                panels[0],
                title="  ".join(title_parts),
                border_style="blue",
                padding=(1, 2),
            )

        return Panel(
            Columns(panels, equal=True, expand=True),
            title="  ".join(title_parts),
            border_style="blue",
            padding=(1, 2),
        )

    def _render_stream(self, m: StreamMonitor):
        status_line = (
            "[green]● connected[/]" if m.connected
            else "[red]○ connecting…[/]"
        )
        raw_title = m.current_title or "—"

        tbl = Table(box=None, padding=(0, 1), show_header=True)
        tbl.add_column("Frequency", style="cyan",  width=13)
        tbl.add_column("Label",                    width=20)
        tbl.add_column("",                         width=9)   # status dot
        tbl.add_column("Since",                    width=9)

        for freq in sorted(m.channels):
            label = m.channels[freq]
            if freq == m.active_freq:
                since = m.active_since.strftime("%H:%M:%S") if m.active_since else ""
                tbl.add_row(
                    Text(f"{freq} MHz", style="bold green"),
                    Text(label,         style="bold green"),
                    Text("◉ ACTIVE",    style="bold green"),
                    Text(since,         style="green"),
                )
            else:
                tbl.add_row(f"{freq} MHz", label, "○", "")

        if not m.channels:
            if m.active_freq:
                tbl.add_row(
                    Text(f"{m.active_freq} MHz", style="bold green"),
                    Text("(auto-detected)", style="dim green"),
                    Text("◉ ACTIVE", style="bold green"),
                    Text(
                        m.active_since.strftime("%H:%M:%S") if m.active_since else "",
                        style="green",
                    ),
                )
            else:
                tbl.add_row("[dim]—[/dim]", "[dim]no channels configured[/dim]", "", "")

        hist = "\n".join(
            f"[dim]{ts.strftime('%H:%M:%S')}[/]  {f} MHz  {lb}"
            for ts, f, lb in m.history[:5]
        ) or "[dim]no activity yet[/dim]"

        return Panel(
            Group(tbl, Text(""), Text(hist, no_wrap=False)),
            title=f"[bold]{m.name}[/]  [dim]{m.mount}[/]  {status_line}",
            subtitle=f"[dim]stream title: {raw_title}[/]",
            border_style="cyan",
            padding=(0, 1),
        )


# ── Config file ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = Path(__file__).parent / "scanner_config.json"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── CLI ──────────────────────────────────────────────────────────────────────────

def parse_channels(spec: str) -> dict:
    """'118.1:ATIS, 119.0:Ground' → {'118.1': 'ATIS', '119.0': 'Ground'}"""
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        freq, _, label = part.partition(":")
        out[freq.strip()] = label.strip() if label else freq.strip()
    return out


def main():
    p = argparse.ArgumentParser(
        description="RTL-Airband Scanner (Pi) — monitor active frequency via Icecast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scanner.py                          # uses scanner_config.json\n"
            "  python3 scanner.py --host 172.31.10.192 --mount /ham.mp3 --mount /air.mp3\n"
            "  python3 scanner.py --audio                  # Python handles audio\n"
        ),
    )
    p.add_argument("--config",   default="", metavar="FILE",
                   help=f"JSON config file (default: {DEFAULT_CONFIG.name} if present)")
    p.add_argument("--host",     default="",
                   help="Icecast host (default: 172.31.10.192)")
    p.add_argument("--port",     type=int, default=0,
                   help="Icecast port (default: 8000)")
    p.add_argument("--mount",    action="append", default=[], metavar="PATH",
                   help="Mount point, e.g. /ham.mp3  (repeat for multiple streams)")
    p.add_argument("--name",     action="append", default=[], metavar="LABEL",
                   help="Display name for each --mount (same order)")
    p.add_argument("--channels", default="",
                   help='Comma-separated freq:label pairs, e.g. "146.940:Rptr,118.1:ATIS"')
    p.add_argument("--user",     default="", help="Icecast listener username (if required)")
    p.add_argument("--password", default="", help="Icecast listener password (if required)")

    audio_grp = p.add_mutually_exclusive_group()
    audio_grp.add_argument(
        "--audio", dest="audio", action="store_true",
        help="Enable audio: ffmpeg → PulseAudio/ALSA "
             "(disable rtl_airband pulse output first to avoid echo)")
    audio_grp.add_argument(
        "--no-audio", dest="audio", action="store_false",
        help="Display only — audio handled by RTLSDR-Airband (default)")
    p.set_defaults(audio=False)

    p.add_argument("--audio-mount", default="",
                   help="Which mount to play audio from when --audio is set "
                        "(default: first --mount)")
    args = p.parse_args()

    # ── Load config file ────────────────────────────────────────────────────────
    cfg: dict = {}
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    if config_path.exists():
        try:
            cfg = load_config(config_path)
            _log(f"Loaded config: {config_path}")
        except Exception as exc:
            _log(f"[yellow]Config file error: {exc} — using defaults[/]")

    # CLI overrides config
    host     = args.host     or cfg.get("host",     "172.31.10.192")
    port     = args.port     or cfg.get("port",     8000)
    user     = args.user     or cfg.get("user",     "")
    password = args.password or cfg.get("password", "")
    channels = (parse_channels(args.channels)
                if args.channels
                else cfg.get("channels", {}))

    # Build stream list: CLI --mount args take precedence over config
    if args.mount:
        names = args.name or []
        stream_list = [
            {"mount": m, "name": names[i] if i < len(names) else Path(m).stem}
            for i, m in enumerate(args.mount)
        ]
    elif "streams" in cfg:
        stream_list = cfg["streams"]
    else:
        stream_list = [{"mount": "/ham.mp3", "name": "Ham Radio"},
                       {"mount": "/air.mp3", "name": "Air Traffic"}]

    monitors = [
        StreamMonitor(
            name=s.get("name", Path(s["mount"]).stem),
            host=host,
            port=port,
            mount=s["mount"],
            channels=s.get("channels", channels),
            user=user,
            password=password,
        )
        for s in stream_list
    ]

    # ── Audio ───────────────────────────────────────────────────────────────────
    sink = None
    if args.audio:
        backend = _detect_audio_backend()
        if backend and FFMPEG:
            _log(f"Audio: ffmpeg → {backend}  "
                 "[dim](disable rtl_airband pulse output to avoid echo)[/]")
            sink = AudioSink(backend)
        else:
            _log("[yellow]Audio requested but no backend found — display only[/]")

    urls = "  ".join(f"http://{host}:{port}{s['mount']}" for s in stream_list)
    _log(f"Streams: {urls}")

    Scanner(monitors, sink=sink, audio_mount=args.audio_mount).run()


if __name__ == "__main__":
    main()
