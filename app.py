#!/usr/bin/env python3
"""
RTL-Airband Scanner — Web App
Run on the Raspberry Pi; open http://<pi-ip>:8080 in any browser on your network.

Setup:
    pip install fastapi "uvicorn[standard]"
    python3 app.py
"""
from __future__ import annotations

import re, json, struct, socket, asyncio, threading, argparse, uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse

# ── Embedded web page ──────────────────────────────────────────────────────────
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
  --gborder:rgba(63,185,80,.35);--blue:#58a6ff;--red:#f85149;
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
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;overflow:hidden;cursor:pointer;transition:border-color .2s}
.card:hover{border-color:#484f58}
.card.playing{border-color:var(--green)}
.card-hdr{padding:10px 14px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px}
.card-name{font-weight:600;font-size:13px}
.card-mount{font-size:11px;color:var(--muted);font-family:var(--mono);margin-left:4px}
.card-conn{margin-left:auto;font-size:11px}
.card-conn.ok{color:var(--green)}.card-conn.err{color:var(--red)}
.ch-list{padding:4px 0}
.ch{display:flex;align-items:center;gap:10px;padding:6px 14px;transition:background .15s}
.ch.active{background:var(--gdim);border-left:3px solid var(--green);padding-left:11px}
.ch-dot{font-size:10px;color:var(--muted);width:12px;flex-shrink:0}
.ch.active .ch-dot{color:var(--green)}
.ch-freq{font-family:var(--mono);font-size:13px;font-weight:500;width:78px;flex-shrink:0}
.ch.active .ch-freq{color:var(--green)}
.ch-lbl{color:var(--muted);font-size:12px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ch.active .ch-lbl{color:var(--text)}
.ch-since{font-size:11px;color:var(--muted);font-family:var(--mono);flex-shrink:0}
.no-ch{padding:14px;color:var(--muted);font-size:12px;text-align:center}
.act-card{background:var(--card);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.act-hdr{padding:10px 14px;border-bottom:1px solid var(--border);font-size:11px;font-weight:600;color:var(--muted);letter-spacing:.06em;text-transform:uppercase}
.act-row{display:flex;align-items:center;gap:14px;padding:7px 14px;border-bottom:1px solid var(--border);font-size:12px}
.act-row:last-child{border-bottom:none}
.at{font-family:var(--mono);color:var(--muted);width:58px;flex-shrink:0}
.as{color:var(--muted);width:96px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.af{font-family:var(--mono);font-weight:600;width:82px;flex-shrink:0}
.al{color:var(--muted);flex:1}
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.75);display:flex;align-items:center;justify-content:center;z-index:200}
.overlay.hidden{display:none}
.obox{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:28px 36px;text-align:center;max-width:380px}
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
  <div class="act-card">
    <div class="act-hdr">Recent Activity</div>
    <div id="actlist"><div class="act-row"><span class="at" style="color:#484f58">—</span><span style="color:#484f58;font-size:12px">No activity yet</span></div></div>
  </div>
</main>
<div class="overlay hidden" id="overlay">
  <div class="obox">
    <h2>🔊 Enable Audio?</h2>
    <p>Your browser requires a click before audio can play.<br>The scanner will automatically follow the active frequency.</p>
    <button class="obtn" onclick="enableAudio()">Enable Audio</button>
    <button class="obtn skip" onclick="closeOverlay()">Display Only</button>
  </div>
</div>
<audio id="aud" preload="none"></audio>
<script>
const S={streams:{},playing:null,audioOn:false,locked:null};
const aud=document.getElementById('aud');
let ws,retry=0,activity=[];

function connect(){
  ws=new WebSocket('ws://'+location.host+'/ws');
  ws.onopen=()=>{retry=0;wsSt(true)};
  ws.onclose=()=>{wsSt(false);setTimeout(connect,Math.min(2000*(++retry),15000))};
  ws.onmessage=e=>onMsg(JSON.parse(e.data));
}
function wsSt(ok){
  document.getElementById('wdot').className='dot '+(ok?'ok':'err');
  document.getElementById('wst').textContent=ok?'Connected':'Reconnecting…';
}
function onMsg(m){
  if(m.type==='state'){
    m.streams.forEach(s=>{
      S.streams[s.mount]=s;
      (s.history||[]).forEach(h=>pushActivity(s.name,h.freq,h.label,h.time));
    });
    renderAll();autoAudio();
  } else if(m.type==='freq_change'){
    const s=S.streams[m.mount];if(!s)return;
    s.activeFreq=m.freq;s.activeSince=m.time;
    updateCard(m.mount);
    pushActivity(m.name,m.freq,m.label,m.time);
    if(S.audioOn&&(!S.locked||S.locked===m.mount))playMount(m.mount);
  } else if(m.type==='conn'){
    const s=S.streams[m.mount];if(s){s.connected=m.connected;updateCard(m.mount);}
  }
}

// ── Render ──────────────────────────────────────────────────────────────────
function renderAll(){
  const g=document.getElementById('grid');g.innerHTML='';
  Object.values(S.streams).forEach(s=>{
    const d=document.createElement('div');
    d.className='card'+(S.playing===s.mount?' playing':'');
    d.id='card'+eid(s.mount);d.onclick=()=>lockAudio(s.mount);
    d.innerHTML=cardHtml(s);g.appendChild(d);
  });
}
function updateCard(mount){
  const d=document.getElementById('card'+eid(mount));if(!d)return;
  const s=S.streams[mount];
  d.className='card'+(S.playing===mount?' playing':'');
  d.innerHTML=cardHtml(s);
}
function cardHtml(s){
  const conn=s.connected
    ?'<span class="card-conn ok">● live</span>'
    :'<span class="card-conn err">○ connecting…</span>';
  const chs=s.channels||{};
  const freqs=Object.keys(chs).sort();
  const spk=(S.playing===s.mount&&S.audioOn)?' <span class="blink" style="color:var(--green);font-size:10px">🔊</span>':'';
  let rows='';
  if(freqs.length){
    freqs.forEach(f=>{
      const lbl=chs[f]||'';const act=f===s.activeFreq;
      const since=act&&s.activeSince?new Date(s.activeSince).toLocaleTimeString():'';
      rows+=`<div class="ch${act?' active':''}">
        <span class="ch-dot">${act?'◉':'○'}</span>
        <span class="ch-freq">${f}</span>
        <span class="ch-lbl">${lbl!==f?lbl:''}</span>
        <span class="ch-since">${since}</span>
      </div>`;
    });
  } else if(s.activeFreq){
    rows=`<div class="ch active">
      <span class="ch-dot">◉</span>
      <span class="ch-freq">${s.activeFreq}</span>
      <span class="ch-lbl" style="font-style:italic;color:var(--muted)">auto-detected</span>
      <span class="ch-since">${s.activeSince?new Date(s.activeSince).toLocaleTimeString():''}</span>
    </div>`;
  } else {
    rows='<div class="no-ch">No activity yet</div>';
  }
  return `<div class="card-hdr">
    <span class="card-name">${s.name}${spk}</span>
    <span class="card-mount">${s.mount}</span>
    ${conn}
  </div><div class="ch-list">${rows}</div>`;
}
function eid(m){return m.replace(/[^a-zA-Z0-9]/g,'_')}

// ── Activity feed ────────────────────────────────────────────────────────────
function pushActivity(name,freq,label,iso){
  activity.unshift({t:new Date(iso).toLocaleTimeString(),name,freq,label:label||''});
  activity=activity.slice(0,30);
  const list=document.getElementById('actlist');
  list.innerHTML=activity.map(a=>`<div class="act-row">
    <span class="at">${a.t}</span>
    <span class="as">${a.name}</span>
    <span class="af">${a.freq} MHz</span>
    <span class="al">${a.label!==a.freq?a.label:''}</span>
  </div>`).join('');
}

// ── Audio ────────────────────────────────────────────────────────────────────
function autoAudio(){
  let best=null,bestT=0;
  Object.values(S.streams).forEach(s=>{
    if(s.activeSince){const t=new Date(s.activeSince).getTime();if(t>bestT){bestT=t;best=s.mount;}}
  });
  if(best)S.playing=best;
}
function playMount(mount){
  if(S.playing===mount&&!aud.paused&&!aud.ended)return;
  S.playing=mount;
  aud.src='/audio'+mount;
  aud.play().catch(()=>{});
  updateAudioUI();
  Object.keys(S.streams).forEach(m=>updateCard(m));
}
// Nudge playback toward live edge to combat buffer build-up
setInterval(()=>{
  if(!aud.src||aud.paused||!aud.buffered.length)return;
  const edge=aud.buffered.end(aud.buffered.length-1);
  if(edge-aud.currentTime>4)aud.currentTime=edge-0.5;
},3000);

function toggleAudio(){
  if(!S.audioOn){document.getElementById('overlay').classList.remove('hidden');}
  else{S.audioOn=false;aud.pause();aud.src='';updateAudioUI();}
}
function enableAudio(){
  S.audioOn=true;closeOverlay();autoAudio();
  if(S.playing)playMount(S.playing);
  updateAudioUI();
}
function closeOverlay(){document.getElementById('overlay').classList.add('hidden');}
function lockAudio(mount){
  S.locked=(S.locked===mount)?null:mount;
  if(S.audioOn&&S.locked)playMount(mount);
  updateAudioUI();
}
function updateAudioUI(){
  const btn=document.getElementById('abtn');
  const src=document.getElementById('asrc');
  if(S.audioOn&&S.playing){
    const s=S.streams[S.playing];
    btn.className='abtn on';
    document.getElementById('aico').textContent='🔊';
    document.getElementById('albl').textContent=S.locked?'Locked':'Auto';
    src.textContent=s?s.name:'';
  } else {
    btn.className='abtn';
    document.getElementById('aico').textContent='🔇';
    document.getElementById('albl').textContent='Enable Audio';
    src.textContent='';
  }
}

connect();
setTimeout(()=>document.getElementById('overlay').classList.remove('hidden'),900);
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
        with self._lock:
            return self._active_freq

    @property
    def active_since(self):
        with self._lock:
            return self._active_since

    @property
    def history(self):
        with self._lock:
            return list(self._history)

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True,
                         name=f"mon-{self.mount}").start()

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.close()

    def _emit(self, event: dict):
        if self._on_event:
            self._on_event(event)

    def _loop(self):
        while self._running:
            self.connected = False
            self._emit({"type": "conn", "mount": self.mount, "connected": False})
            self._stream = IcyStream(self.host, self.port, self.mount)
            try:
                self._stream.connect()
            except Exception as exc:
                print(f"[{self.name}] connection failed: {exc}")
                import time; time.sleep(self.RECONNECT)
                continue
            self.connected = True
            self._emit({"type": "conn", "mount": self.mount, "connected": True})

            for chunk in self._stream.iter_audio():
                if not self._running:
                    break
                title = self._stream.current_title
                freq  = _match_title(title, self.channels)
                with self._lock:
                    if freq and freq != self._active_freq:
                        self._active_freq  = freq
                        self._active_since = datetime.now()
                        label = self.channels.get(freq, "")
                        self._history.appendleft(
                            (datetime.now(), freq, label)
                        )
                        self._emit({
                            "type":  "freq_change",
                            "mount": self.mount,
                            "name":  self.name,
                            "freq":  freq,
                            "label": label,
                            "time":  datetime.now().isoformat(),
                        })

            if self._running:
                self.connected = False
                self._emit({"type": "conn", "mount": self.mount, "connected": False})
                print(f"[{self.name}] stream dropped, reconnecting in {self.RECONNECT}s")
                import time; time.sleep(self.RECONNECT)


# ── WebSocket manager ──────────────────────────────────────────────────────────
class WsManager:
    def __init__(self):
        self._clients: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.append(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            try:
                self._clients.remove(ws)
            except ValueError:
                pass

    async def broadcast(self, data: dict):
        msg = json.dumps(data, default=str)
        async with self._lock:
            dead = []
            for ws in self._clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                try:
                    self._clients.remove(ws)
                except ValueError:
                    pass


# ── App state ──────────────────────────────────────────────────────────────────
monitors:  list[StreamMonitor]          = []
cfg:       dict                         = {}
wsman:     WsManager                    = WsManager()
_evqueue:  asyncio.Queue | None         = None
_evloop:   asyncio.AbstractEventLoop | None = None


def _emit(event: dict):
    """Bridge: called from sync monitor threads, queues into async event loop."""
    if _evloop and _evqueue:
        asyncio.run_coroutine_threadsafe(_evqueue.put(event), _evloop)


async def _broadcast_loop():
    while True:
        event = await _evqueue.get()
        await wsman.broadcast(event)


def _get_state() -> dict:
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
            }
            for m in monitors
        ],
    }


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _evqueue, _evloop
    _evloop  = asyncio.get_running_loop()
    _evqueue = asyncio.Queue()
    asyncio.create_task(_broadcast_loop())
    for m in monitors:
        m.start()
    yield
    for m in monitors:
        m.stop()


app = FastAPI(lifespan=lifespan)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE


@app.get("/audio/{mount_path:path}")
async def audio_proxy(mount_path: str, request: Request):
    """Proxy the Icecast stream to the browser as a clean audio stream."""
    host  = cfg.get("host", "localhost")
    port  = cfg.get("port", 8000)
    mount = "/" + mount_path.lstrip("/")

    async def generate():
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=10
            )
        except Exception as exc:
            print(f"Audio proxy connect failed: {exc}")
            return
        try:
            req = "\r\n".join([
                f"GET {mount} HTTP/1.0",
                f"Host: {host}:{port}",
                "User-Agent: RTLAirbandScanner/1.0",
                "Connection: close", "", ""
            ])
            writer.write(req.encode())
            await writer.drain()
            # Skip HTTP response headers
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = await asyncio.wait_for(reader.read(1), timeout=10)
                if not chunk:
                    return
                buf += chunk
            # Stream audio to browser
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(reader.read(8192), timeout=2.0)
                    if not data:
                        break
                    yield data
                except asyncio.TimeoutError:
                    continue
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    return StreamingResponse(
        generate(),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await wsman.connect(ws)
    try:
        await ws.send_text(json.dumps(_get_state(), default=str))
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
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--listen-port", type=int, default=8080,
                   help="Web server port (default: 8080)")
    args = p.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"Config: {config_path}")
    else:
        print(f"Warning: {config_path} not found, using defaults")

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
            host=host,
            port=port,
            mount=s["mount"],
            channels=s.get("channels", channels),
            on_event=_emit,
        )
        for s in streams
    ]

    print(f"Open http://<pi-ip>:{args.listen_port} in your browser")
    uvicorn.run(app, host="0.0.0.0", port=args.listen_port, log_level="warning")


if __name__ == "__main__":
    main()
