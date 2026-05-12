#!/usr/bin/env python3
"""
RTL-Airband Scanner — Web App
Run on the Raspberry Pi; open http://<pi-ip>:8080 in any browser.

Setup:
    pip install fastapi "uvicorn[standard]"
    sudo apt install ffmpeg
    python3 app.py
"""
from __future__ import annotations

import re, json, struct, socket, asyncio, threading, argparse, shutil, uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse

FFMPEG = shutil.which("ffmpeg")

# ── Embedded page ──────────────────────────────────────────────────────────────
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RTL-Airband Scanner</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--card:#161b22;--card2:#21262d;--border:#30363d;
  --text:#e6edf3;--muted:#7d8590;--green:#3fb950;--gdim:rgba(63,185,80,.12);
  --gborder:rgba(63,185,80,.35);--blue:#58a6ff;--red:#f85149;--yellow:#d29922;
  --mono:'SF Mono','Fira Code','Consolas',monospace;
}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px;min-height:100vh}
header{background:var(--card);border-bottom:1px solid var(--border);padding:12px 20px;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:100}
h1{font-size:15px;font-weight:600;letter-spacing:.02em;display:flex;align-items:center;gap:8px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted);flex-shrink:0;transition:background .3s}
.dot.ok{background:var(--green)}.dot.err{background:var(--red)}
.st{font-size:12px;color:var(--muted)}
.spacer{flex:1}
.abtn{background:var(--card2);border:1px solid var(--border);border-radius:6px;color:var(--text);cursor:pointer;font-size:12px;padding:5px 12px;display:flex;align-items:center;gap:6px;transition:all .15s}
.abtn:hover{border-color:var(--blue);color:var(--blue)}
.abtn.on{border-color:var(--green);color:var(--green);background:var(--gdim)}
.asrc{font-size:11px;color:var(--muted)}
main{max-width:1100px;margin:0 auto;padding:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-bottom:20px}
.scard{background:var(--card);border:1px solid var(--border);border-radius:8px;overflow:hidden;cursor:pointer;transition:border-color .2s}
.scard:hover{border-color:#484f58}
.scard.playing{border-color:var(--green)}
.scard.locked{border-color:var(--blue)}
.shdr{padding:10px 14px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.sname{font-weight:600;font-size:13px}
.smount{font-size:11px;color:var(--muted);font-family:var(--mono);margin-left:2px}
.sconn{margin-left:auto;font-size:11px}
.sconn.ok{color:var(--green)}.sconn.err{color:var(--red)}.sconn.warn{color:var(--yellow)}
.serr{font-size:11px;color:var(--red);padding:6px 14px;background:rgba(248,81,73,.08);border-bottom:1px solid rgba(248,81,73,.2)}
.chlist{padding:4px 0}
.ch{display:flex;align-items:center;gap:10px;padding:6px 14px;transition:background .15s}
.ch.active{background:var(--gdim);border-left:3px solid var(--green);padding-left:11px}
.ch-dot{font-size:10px;color:var(--muted);width:12px;flex-shrink:0}
.ch.active .ch-dot{color:var(--green)}
.ch-f{font-family:var(--mono);font-size:13px;font-weight:500;width:80px;flex-shrink:0}
.ch.active .ch-f{color:var(--green)}
.ch-l{color:var(--muted);font-size:12px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ch.active .ch-l{color:var(--text)}
.ch-t{font-size:11px;color:var(--muted);font-family:var(--mono);flex-shrink:0}
.noch{padding:14px;color:var(--muted);font-size:12px;text-align:center}
.hint{padding:8px 14px;font-size:11px;color:var(--muted);border-top:1px solid var(--border)}
.acard{background:var(--card);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.ahdr{padding:10px 14px;border-bottom:1px solid var(--border);font-size:11px;font-weight:600;color:var(--muted);letter-spacing:.06em;text-transform:uppercase}
.arow{display:flex;align-items:center;gap:14px;padding:7px 14px;border-bottom:1px solid var(--border);font-size:12px}
.arow:last-child{border-bottom:none}
.at{font-family:var(--mono);color:var(--muted);width:58px;flex-shrink:0}
.as{color:var(--muted);width:96px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.af{font-family:var(--mono);font-weight:600;width:82px;flex-shrink:0}
.al{color:var(--muted);flex:1}
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.75);display:flex;align-items:center;justify-content:center;z-index:200}
.overlay.hidden{display:none}
.obox{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:28px 36px;text-align:center;max-width:400px}
.obox h2{font-size:17px;margin-bottom:8px}
.obox p{color:var(--muted);font-size:13px;margin-bottom:22px;line-height:1.6}
.obtn{background:var(--green);border:none;border-radius:6px;color:#000;cursor:pointer;font-size:13px;font-weight:600;padding:9px 24px;margin:4px}
.obtn.skip{background:var(--card2);border:1px solid var(--border);color:var(--text);font-weight:400}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.blink{animation:pulse 1.4s ease-in-out infinite}
</style>
</head>
<body>
<header>
  <h1>
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2">
      <circle cx="12" cy="12" r="2"/>
      <path d="M16.24 7.76a6 6 0 010 8.49M7.76 16.24a6 6 0 010-8.49M20.49 3.51a12 12 0 010 16.97M3.51 20.49a12 12 0 010-16.97"/>
    </svg>
    RTL-Airband Scanner
  </h1>
  <div class="dot" id="wdot"></div>
  <span class="st" id="wst">Connecting…</span>
  <div class="spacer"></div>
  <span class="asrc" id="asrc"></span>
  <button class="abtn" id="abtn" onclick="toggleAudio()">
    <span id="aico">🔇</span><span id="albl">Enable Audio</span>
  </button>
</header>
<main>
  <div class="grid" id="grid"></div>
  <div class="acard">
    <div class="ahdr">Recent Activity</div>
    <div id="actlist">
      <div class="arow"><span class="at" style="color:#484f58">—</span><span style="color:#484f58;font-size:12px">No activity yet</span></div>
    </div>
  </div>
</main>
<div class="overlay hidden" id="overlay">
  <div class="obox">
    <h2>🔊 Enable Audio?</h2>
    <p>Audio streams at ~150 ms latency using PCM over WebSocket.<br>
       The player auto-follows whichever stream becomes active.<br>
       Click a stream card to lock it to one source.</p>
    <button class="obtn" onclick="enableAudio()">Enable Audio</button>
    <button class="obtn skip" onclick="closeOverlay()">Display Only</button>
  </div>
</div>
<script>
// ── State ──────────────────────────────────────────────────────────────────────
const S = { streams:{}, playing:null, audioOn:false, locked:null };
let ws, wsRetry=0;
let actItems = [];

// ── Audio (Web Audio API + PCM over WebSocket) ─────────────────────────────────
// Each stream gets its own WebSocket delivering raw s16le PCM at 44100 Hz mono.
// We schedule chunks 100 ms ahead — total latency ≈ 100 ms + one-chunk ≈ 120 ms.
const RATE = 44100;
let actx   = null;   // AudioContext
let audWs  = null;   // current audio WebSocket
let audMount = null; // which mount is feeding actx
let nextAt = 0;      // scheduled up-to time (AudioContext clock)

function initAudioCtx() {
  if (actx) return;
  actx = new AudioContext({ sampleRate: RATE, latencyHint: 'interactive' });
}

function openAudioStream(mount) {
  if (audMount === mount && audWs && audWs.readyState === WebSocket.OPEN) return;
  if (audWs) { audWs.close(); audWs = null; }
  audMount = mount;
  nextAt   = 0;
  initAudioCtx();
  if (actx.state === 'suspended') actx.resume();

  const url = 'ws://' + location.host + '/ws/audio' + mount;
  audWs = new WebSocket(url);
  audWs.binaryType = 'arraybuffer';

  audWs.onopen = () => updateAudioUI();

  audWs.onmessage = ({ data }) => {
    if (!actx) return;
    // data is raw s16le PCM
    const s16 = new Int16Array(data);
    const buf = actx.createBuffer(1, s16.length, RATE);
    const ch  = buf.getChannelData(0);
    for (let i = 0; i < s16.length; i++) ch[i] = s16[i] / 32768;
    const src = actx.createBufferSource();
    src.buffer = buf;
    src.connect(actx.destination);
    const now = actx.currentTime;
    if (nextAt < now + 0.1) nextAt = now + 0.1;  // stay 100 ms ahead
    src.start(nextAt);
    nextAt += buf.duration;
  };

  audWs.onclose = () => { if (audMount === mount) updateAudioUI(); };
  audWs.onerror = () => { if (audMount === mount) updateAudioUI(); };
}

function closeAudio() {
  if (audWs) { audWs.close(); audWs = null; }
  audMount = null;
  nextAt   = 0;
  if (actx) { actx.close(); actx = null; }
}

// ── WebSocket (control) ────────────────────────────────────────────────────────
function connect() {
  ws = new WebSocket('ws://' + location.host + '/ws');
  ws.onopen  = () => { wsRetry=0; setWsSt(true); };
  ws.onclose = () => { setWsSt(false); setTimeout(connect, Math.min(2000*(++wsRetry),15000)); };
  ws.onmessage = e => onMsg(JSON.parse(e.data));
}
function setWsSt(ok) {
  document.getElementById('wdot').className = 'dot ' + (ok ? 'ok' : 'err');
  document.getElementById('wst').textContent = ok ? 'Connected' : 'Reconnecting…';
}

// ── Message handler ────────────────────────────────────────────────────────────
function onMsg(m) {
  if (m.type === 'state') {
    m.streams.forEach(s => {
      S.streams[s.mount] = s;
      (s.history || []).forEach(h => pushActivity(s.name, h.freq, h.label, h.time));
    });
    renderAll();
    autoSelect();
  } else if (m.type === 'freq_change') {
    const s = S.streams[m.mount]; if (!s) return;
    s.activeFreq  = m.freq;
    s.activeSince = m.time;
    s.lastError   = null;
    updateCard(m.mount);
    pushActivity(m.name, m.freq, m.label, m.time);
    if (S.audioOn && (!S.locked || S.locked === m.mount)) switchAudio(m.mount);
  } else if (m.type === 'conn') {
    const s = S.streams[m.mount]; if (!s) return;
    s.connected = m.connected;
    s.lastError  = m.error || null;
    updateCard(m.mount);
  }
}

// ── Render ─────────────────────────────────────────────────────────────────────
function renderAll() {
  const g = document.getElementById('grid'); g.innerHTML = '';
  Object.values(S.streams).forEach(s => {
    const d = document.createElement('div');
    d.className = cardClass(s);
    d.id = 'sc' + eid(s.mount);
    d.onclick = () => lockTo(s.mount);
    d.innerHTML = cardHtml(s);
    g.appendChild(d);
  });
}
function updateCard(mount) {
  const d = document.getElementById('sc' + eid(mount)); if (!d) return;
  const s = S.streams[mount];
  d.className = cardClass(s);
  d.innerHTML = cardHtml(s);
}
function cardClass(s) {
  let c = 'scard';
  if (S.locked === s.mount) c += ' locked';
  else if (audMount === s.mount && S.audioOn) c += ' playing';
  return c;
}
function cardHtml(s) {
  const connHtml = s.connected
    ? '<span class="sconn ok">● live</span>'
    : '<span class="sconn err">○ connecting…</span>';
  const spk = (audMount===s.mount && S.audioOn)
    ? ' <span class="blink" style="color:var(--green);font-size:10px">🔊</span>' : '';
  const lockBadge = S.locked===s.mount
    ? ' <span style="font-size:10px;color:var(--blue)">🔒 locked</span>' : '';

  const errHtml = s.lastError
    ? `<div class="serr">⚠ ${s.lastError}</div>` : '';

  const chs   = s.channels || {};
  const freqs = Object.keys(chs).sort();
  let rows = '';
  if (freqs.length) {
    freqs.forEach(f => {
      const lbl = chs[f] || ''; const act = f === s.activeFreq;
      const since = act && s.activeSince ? new Date(s.activeSince).toLocaleTimeString() : '';
      rows += `<div class="ch${act?' active':''}">
        <span class="ch-dot">${act?'◉':'○'}</span>
        <span class="ch-f">${f}</span>
        <span class="ch-l">${lbl!==f?lbl:''}</span>
        <span class="ch-t">${since}</span>
      </div>`;
    });
  } else if (s.activeFreq) {
    rows = `<div class="ch active">
      <span class="ch-dot">◉</span>
      <span class="ch-f">${s.activeFreq}</span>
      <span class="ch-l" style="font-style:italic;color:var(--muted)">auto-detected</span>
      <span class="ch-t">${s.activeSince?new Date(s.activeSince).toLocaleTimeString():''}</span>
    </div>`;
  } else {
    rows = '<div class="noch">No activity yet</div>';
  }
  const hintTxt = S.locked===s.mount ? 'Click to unlock' : 'Click to lock audio here';
  return `<div class="shdr">
    <span class="sname">${s.name}${spk}${lockBadge}</span>
    <span class="smount">${s.mount}</span>
    ${connHtml}
  </div>${errHtml}<div class="chlist">${rows}</div>
  <div class="hint">${hintTxt}</div>`;
}
function eid(m) { return m.replace(/[^a-zA-Z0-9]/g,'_'); }

// ── Activity ───────────────────────────────────────────────────────────────────
function pushActivity(name, freq, label, iso) {
  actItems.unshift({ t: new Date(iso).toLocaleTimeString(), name, freq, label: label||'' });
  actItems = actItems.slice(0, 30);
  document.getElementById('actlist').innerHTML = actItems.map(a =>
    `<div class="arow">
      <span class="at">${a.t}</span><span class="as">${a.name}</span>
      <span class="af">${a.freq} MHz</span>
      <span class="al">${a.label!==a.freq?a.label:''}</span>
    </div>`).join('');
}

// ── Audio controls ─────────────────────────────────────────────────────────────
function autoSelect() {
  let best=null, bestT=0;
  Object.values(S.streams).forEach(s => {
    if (s.activeSince) { const t=new Date(s.activeSince).getTime(); if(t>bestT){bestT=t;best=s.mount;} }
  });
  if (best) S.playing = best;
}
function switchAudio(mount) {
  S.playing = mount;
  if (S.audioOn) openAudioStream(mount);
  updateAudioUI();
  Object.keys(S.streams).forEach(m => updateCard(m));
}
function lockTo(mount) {
  if (S.locked === mount) { S.locked = null; }
  else { S.locked = mount; if (S.audioOn) switchAudio(mount); }
  updateAudioUI();
  Object.keys(S.streams).forEach(m => updateCard(m));
}
function toggleAudio() {
  if (!S.audioOn) { document.getElementById('overlay').classList.remove('hidden'); }
  else { S.audioOn=false; closeAudio(); updateAudioUI(); Object.keys(S.streams).forEach(m=>updateCard(m)); }
}
function enableAudio() {
  S.audioOn = true; closeOverlay();
  autoSelect();
  if (S.locked) switchAudio(S.locked);
  else if (S.playing) switchAudio(S.playing);
  updateAudioUI();
}
function closeOverlay() { document.getElementById('overlay').classList.add('hidden'); }
function updateAudioUI() {
  const btn = document.getElementById('abtn');
  const src = document.getElementById('asrc');
  const connected = audWs && audWs.readyState === WebSocket.OPEN;
  if (S.audioOn && audMount) {
    const s = S.streams[audMount];
    btn.className = 'abtn on';
    document.getElementById('aico').textContent = connected ? '🔊' : '⏳';
    document.getElementById('albl').textContent  = S.locked ? 'Locked' : 'Auto';
    src.textContent = s ? s.name : '';
  } else {
    btn.className = 'abtn';
    document.getElementById('aico').textContent = '🔇';
    document.getElementById('albl').textContent  = 'Enable Audio';
    src.textContent = '';
  }
}

// ── Init ───────────────────────────────────────────────────────────────────────
connect();
setTimeout(() => document.getElementById('overlay').classList.remove('hidden'), 900);
</script>
</body>
</html>
"""

# ── IcyStream ──────────────────────────────────────────────────────────────────
class IcyStream:
    RECV_SIZE = 2048

    def __init__(self, host: str, port: int, mount: str):
        self.host          = host
        self.port          = port
        self.mount         = mount
        self.metaint       = 0
        self.current_title = ""
        self._sock         = None
        self._closed       = False

    def connect(self) -> dict:
        self._sock = socket.create_connection((self.host, self.port), timeout=10)
        self._sock.settimeout(30)
        req = "\r\n".join([
            f"GET {self.mount} HTTP/1.0",
            f"Host: {self.host}:{self.port}",
            "User-Agent: RTLAirbandScanner/1.0",
            "Icy-MetaData: 1",
            "Connection: close", "", ""
        ])
        self._sock.sendall(req.encode())
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
        self.metaint = int(headers.get("icy-metaint", 0))
        return headers

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise EOFError("Stream ended")
            buf.extend(chunk)
        return bytes(buf)

    def _read_meta(self):
        length = struct.unpack("B", self._recv_exact(1))[0] * 16
        if length:
            raw = self._recv_exact(length).decode("utf-8", errors="replace").rstrip("\x00")
            m = re.search(r"StreamTitle='([^']*)'", raw)
            if m:
                self.current_title = m.group(1)

    def iter_audio(self):
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
                    self._read_meta()
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


# ── StreamMonitor ──────────────────────────────────────────────────────────────
def _match_title(title: str, channels: dict) -> str | None:
    if not title:
        return None
    for freq, label in channels.items():
        if freq in title or (label and label in title):
            return freq
    m = re.search(r"(\d{2,3}\.\d{1,4})", title)
    return m.group(1) if m else None


class StreamMonitor:
    RECONNECT = 5

    def __init__(self, name: str, host: str, port: int, mount: str,
                 channels: dict, on_event=None):
        self.name      = name
        self.host      = host
        self.port      = port
        self.mount     = mount
        self.channels  = channels
        self._on_event = on_event

        self._stream       = None
        self._active_freq  = None
        self._active_since = None
        self._history: deque = deque(maxlen=20)
        self._lock         = threading.Lock()
        self._running      = False
        self.connected     = False

    @property
    def active_freq(self):
        with self._lock: return self._active_freq

    @property
    def active_since(self):
        with self._lock: return self._active_since

    @property
    def history(self):
        with self._lock: return list(self._history)

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True,
                         name=f"mon-{self.mount}").start()

    def stop(self):
        self._running = False
        if self._stream: self._stream.close()

    def _emit(self, event: dict):
        if self._on_event: self._on_event(event)

    def _loop(self):
        import time
        last_error: str | None = None
        while self._running:
            self.connected = False
            self._emit({"type": "conn", "mount": self.mount,
                        "connected": False, "error": last_error})

            self._stream = IcyStream(self.host, self.port, self.mount)
            try:
                self._stream.connect()
            except Exception as exc:
                last_error = str(exc)
                print(f"[{self.name}] connect failed: {last_error}")
                self._emit({
                    "type": "conn", "mount": self.mount,
                    "connected": False, "error": last_error,
                })
                time.sleep(self.RECONNECT)
                continue

            last_error = None
            self.connected = True
            self._emit({"type": "conn", "mount": self.mount,
                        "connected": True, "error": None})

            for chunk in self._stream.iter_audio():
                if not self._running: break
                freq = _match_title(self._stream.current_title, self.channels)
                with self._lock:
                    if freq and freq != self._active_freq:
                        self._active_freq  = freq
                        self._active_since = datetime.now()
                        label = self.channels.get(freq, "")
                        self._history.appendleft((datetime.now(), freq, label))
                        self._emit({
                            "type": "freq_change", "mount": self.mount,
                            "name": self.name, "freq": freq, "label": label,
                            "time": datetime.now().isoformat(),
                        })

            if self._running:
                self.connected = False
                self._emit({"type": "conn", "mount": self.mount,
                            "connected": False, "error": "Stream ended"})
                print(f"[{self.name}] stream ended, reconnecting in {self.RECONNECT}s")
                time.sleep(self.RECONNECT)


# ── WebSocket manager ──────────────────────────────────────────────────────────
class WsManager:
    def __init__(self):
        self._clients: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock: self._clients.append(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            try: self._clients.remove(ws)
            except ValueError: pass

    async def broadcast(self, data: dict):
        msg = json.dumps(data, default=str)
        async with self._lock:
            dead = []
            for ws in self._clients:
                try: await ws.send_text(msg)
                except Exception: dead.append(ws)
            for ws in dead:
                try: self._clients.remove(ws)
                except ValueError: pass


# ── App state ──────────────────────────────────────────────────────────────────
monitors: list[StreamMonitor]           = []
cfg:      dict                          = {}
wsman:    WsManager                     = WsManager()
_evq:     asyncio.Queue | None          = None
_evloop:  asyncio.AbstractEventLoop | None = None


def _emit(event: dict):
    if _evloop and _evq:
        asyncio.run_coroutine_threadsafe(_evq.put(event), _evloop)


async def _bcast_loop():
    while True:
        event = await _evq.get()
        await wsman.broadcast(event)


def _state() -> dict:
    return {
        "type": "state",
        "streams": [
            {
                "mount":      m.mount,
                "name":       m.name,
                "connected":  m.connected,
                "activeFreq": m.active_freq,
                "activeSince": m.active_since.isoformat() if m.active_since else None,
                "history": [
                    {"time": ts.isoformat(), "freq": f, "label": lb}
                    for ts, f, lb in m.history[:10]
                ],
                "channels": m.channels,
                "lastError": None,
            }
            for m in monitors
        ],
    }


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _evq, _evloop
    _evloop = asyncio.get_running_loop()
    _evq    = asyncio.Queue()
    asyncio.create_task(_bcast_loop())
    for m in monitors: m.start()
    yield
    for m in monitors: m.stop()


app = FastAPI(lifespan=lifespan)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE


@app.websocket("/ws/audio/{mount_path:path}")
async def audio_ws(ws: WebSocket, mount_path: str):
    """
    Stream raw PCM (s16le, 44100 Hz, mono) via WebSocket.
    ffmpeg connects to Icecast, decodes whatever format, outputs PCM.
    The browser's Web Audio API plays it with ~100 ms lookahead.
    """
    if not FFMPEG:
        await ws.close(code=1011, reason="ffmpeg not found on server")
        return

    await ws.accept()
    host  = cfg.get("host", "localhost")
    port  = cfg.get("port", 8000)
    mount = "/" + mount_path.lstrip("/")
    url   = f"http://{host}:{port}{mount}"

    proc = await asyncio.create_subprocess_exec(
        FFMPEG,
        "-loglevel",  "quiet",
        "-reconnect", "1", "-reconnect_streamed", "1",
        "-i",         url,
        "-f",         "s16le",   # raw PCM, signed 16-bit little-endian
        "-ar",        "44100",   # sample rate the browser AudioContext expects
        "-ac",        "1",       # mono
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    # 4096 bytes = 1024 samples = ~23 ms per chunk at 44100 Hz
    CHUNK = 4096
    try:
        while True:
            data = await asyncio.wait_for(proc.stdout.read(CHUNK), timeout=10.0)
            if not data:
                break
            await ws.send_bytes(data)
    except (WebSocketDisconnect, asyncio.TimeoutError, Exception):
        pass
    finally:
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await wsman.connect(ws)
    try:
        await ws.send_text(json.dumps(_state(), default=str))
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "ping"}))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        await wsman.disconnect(ws)


# ── Entry point ────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = Path(__file__).parent / "scanner_config.json"


def main():
    global cfg, monitors

    p = argparse.ArgumentParser(description="RTL-Airband Scanner Web App")
    p.add_argument("--config",       default=str(DEFAULT_CONFIG))
    p.add_argument("--listen-port",  type=int, default=8080)
    args = p.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"Config loaded: {config_path}")
    else:
        print(f"Warning: {config_path} not found — using defaults")

    host     = cfg.get("host",    "172.31.10.192")
    port     = cfg.get("port",    8000)
    channels = cfg.get("channels", {})
    streams  = cfg.get("streams", [
        {"mount": "/ham.mp3", "name": "Ham / GMRS"},
        {"mount": "/air.mp3", "name": "Air Traffic"},
    ])

    monitors = [
        StreamMonitor(
            name=s.get("name", Path(s["mount"]).stem),
            host=host, port=port, mount=s["mount"],
            channels=s.get("channels", channels),
            on_event=_emit,
        )
        for s in streams
    ]

    if not FFMPEG:
        print("WARNING: ffmpeg not found — audio will not work (sudo apt install ffmpeg)")

    print(f"Open http://<pi-ip>:{args.listen_port} in your browser")
    uvicorn.run(app, host="0.0.0.0", port=args.listen_port, log_level="warning")


if __name__ == "__main__":
    main()
