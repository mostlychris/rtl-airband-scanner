#!/usr/bin/env python3
"""
RTL-Airband Scanner — rtl_fm subprocess mode.
No RTLSDR-Airband, Icecast, or pyrtlsdr required.

Install:
    sudo apt install rtl-sdr python3-numpy
    pip install fastapi "uvicorn[standard]"
    cp scanner_config.example.json scanner_config.json
    nano scanner_config.json
    python3 app.py
"""
from __future__ import annotations

import re, json, asyncio, threading, argparse, subprocess, shutil, uvicorn
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

AUDIO_RATE = 25000   # PCM output rate; must match -r flag passed to rtl_fm


# Silence chunk sent to audio WebSocket clients every 5 s when no signal is
# active — keeps the connection alive so the browser never needs to re-enable.
_AUDIO_KEEPALIVE = bytes(int(AUDIO_RATE * 2 * 0.1))         # 100 ms of zeros

# rtl_fm prints these on every invocation — suppress them so only real errors show
_RTLFM_NOISE = re.compile(
    r"Found \d+ device|Using device \d+|Found .+ tuner|"
    r"Tuner gain set|Tuner error set|Tuned to \d+|Oversampling|"
    r"Buffer size|sample rate is|Allocating \d+|Sampling at|Output at|"
    r"Signal caught|User cancel",
    re.IGNORECASE,
)

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
.sqbar{height:4px;background:var(--card2);border-radius:2px;margin:0 14px 10px;overflow:hidden}
.sqfill{height:100%;background:var(--muted);border-radius:2px;transition:width .15s,background .15s;width:0%}
.sqfill.active{background:var(--green)}
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
    <p>Audio is demodulated on the Pi and streamed as PCM over WebSocket.<br>
       Latency is typically under 1 second.<br>
       Click a stream card to lock audio to that stream.</p>
    <button class="obtn" onclick="enableAudio()">Enable Audio</button>
    <button class="obtn skip" onclick="closeOverlay()">Display Only</button>
  </div>
</div>
<script>
// ── State ──────────────────────────────────────────────────────────────────────
const S = { streams:{}, playing:null, audioOn:false, locked:null };
let ws, wsRetry=0;
let actItems = [];

// ── Audio (Web Audio API, PCM via WebSocket) ───────────────────────────────────
let PCM_RATE  = 25000;
let actx      = null;
let audFilt   = null;   // persistent lowpass filter node
let audWs     = null;
let audMount  = null;
let nextAt    = 0;

function initAudioCtx() {
  if (actx) return;
  actx = new AudioContext({ latencyHint: 'interactive' });
  audFilt = actx.createBiquadFilter();
  audFilt.type = 'lowpass';
  audFilt.frequency.value = 3000;  // 3 kHz: pass voice, block FM noise above Nyquist
  audFilt.Q.value = 0.707;         // Butterworth — no resonance peak
  audFilt.connect(actx.destination);
}

function openAudioStream(mount) {
  if (audWs && audWs.readyState === WebSocket.OPEN) return;
  if (audWs) { audWs.close(); audWs = null; }
  audMount = mount;
  nextAt   = 0;
  initAudioCtx();
  if (actx && actx.state === 'suspended') actx.resume();

  audWs = new WebSocket('ws://' + location.host + '/ws/audio');
  audWs.binaryType = 'arraybuffer';
  audWs.onopen  = () => updateAudioUI();
  audWs.onerror = () => {};  // onclose always follows
  audWs.onclose = () => {
    const m = audMount;
    audMount = null; nextAt = 0; updateAudioUI();
    // Auto-reconnect when audio is still enabled (i.e. closed unexpectedly)
    if (S.audioOn && m) setTimeout(() => { if (S.audioOn) openAudioStream(m); }, 2000);
  };

  const MAX_QUEUE = 1.5;
  audWs.onmessage = ({ data }) => {
    if (!actx) return;
    if (actx.state === 'suspended') { actx.resume(); return; }
    if (actx.state !== 'running') return;
    const now = actx.currentTime;
    if (nextAt > now + MAX_QUEUE) return;
    const s16 = new Int16Array(data);
    const buf = actx.createBuffer(1, s16.length, PCM_RATE);
    const ch  = buf.getChannelData(0);
    for (let i = 0; i < s16.length; i++) ch[i] = s16[i] / 32768;
    const src = actx.createBufferSource();
    src.buffer = buf; src.connect(audFilt || actx.destination);
    if (nextAt < now + 0.05) nextAt = now + 0.05;
    src.start(nextAt);
    nextAt += buf.duration;
  };
}

function closeAudio() {
  if (audWs) { audWs.close(); audWs = null; }
  audMount = null; nextAt = 0;
  if (actx) actx.suspend();  // suspend rather than close — avoids gesture requirement on re-enable
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
    if (m.audio_rate) PCM_RATE = m.audio_rate;
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
  } else if (m.type === 'freq_clear') {
    const s = S.streams[m.mount]; if (!s) return;
    s.activeFreq  = null;
    s.activeSince = null;
    updateCard(m.mount);
  } else if (m.type === 'signal') {
    const d = document.getElementById('sqfill_' + eid(m.mount));
    if (!d) return;
    const pct = Math.min(100, Math.max(0, (m.db + 60) * 100 / 40));
    d.style.width = pct + '%';
    d.className = 'sqfill' + (m.active ? ' active' : '');
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
    ? '<span class="sconn ok">● scanning</span>'
    : '<span class="sconn err">○ ' + (s.lastError ? 'error' : 'opening…') + '</span>';
  const spk = (audMount===s.mount && S.audioOn)
    ? ' <span class="blink" style="color:var(--green);font-size:10px">🔊</span>' : '';
  const lockBadge = S.locked===s.mount
    ? ' <span style="font-size:10px;color:var(--blue)">🔒 locked</span>' : '';
  const errHtml = s.lastError
    ? '<div class="serr">⚠ ' + s.lastError + '</div>' : '';

  const chs   = s.channels || {};
  const freqs = Object.keys(chs).sort((a,b) => parseFloat(a)-parseFloat(b));
  let rows = '';
  if (freqs.length) {
    freqs.forEach(f => {
      const lbl = chs[f] || ''; const act = f === s.activeFreq;
      const since = act && s.activeSince ? new Date(s.activeSince).toLocaleTimeString() : '';
      rows += '<div class="ch' + (act?' active':'') + '">'
        + '<span class="ch-dot">' + (act?'◉':'○') + '</span>'
        + '<span class="ch-f">' + f + '</span>'
        + '<span class="ch-l">' + (lbl!==f?lbl:'') + '</span>'
        + '<span class="ch-t">' + since + '</span>'
        + '</div>';
    });
  } else if (s.activeFreq) {
    rows = '<div class="ch active">'
      + '<span class="ch-dot">◉</span>'
      + '<span class="ch-f">' + s.activeFreq + '</span>'
      + '<span class="ch-l" style="font-style:italic;color:var(--muted)">detected</span>'
      + '<span class="ch-t">' + (s.activeSince?new Date(s.activeSince).toLocaleTimeString():'') + '</span>'
      + '</div>';
  } else {
    rows = '<div class="noch">Scanning…</div>';
  }
  const hintTxt = S.locked===s.mount ? 'Click to unlock' : 'Click to lock audio here';
  return '<div class="shdr"><span class="sname">' + s.name + spk + lockBadge + '</span>'
    + connHtml + '</div>' + errHtml
    + '<div class="chlist">' + rows + '</div>'
    + '<div class="sqbar"><div class="sqfill" id="sqfill_' + eid(s.mount) + '"></div></div>'
    + '<div class="hint">' + hintTxt + '</div>';
}
function eid(m) { return m.replace(/[^a-zA-Z0-9]/g,'_'); }

// ── Activity ───────────────────────────────────────────────────────────────────
function pushActivity(name, freq, label, iso) {
  actItems.unshift({ t: new Date(iso).toLocaleTimeString(), name, freq, label: label||'' });
  actItems = actItems.slice(0, 30);
  document.getElementById('actlist').innerHTML = actItems.map(a =>
    '<div class="arow"><span class="at">' + a.t + '</span><span class="as">' + a.name + '</span>'
    + '<span class="af">' + a.freq + ' MHz</span>'
    + '<span class="al">' + (a.label!==a.freq?a.label:'') + '</span></div>').join('');
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
  const target = S.locked || S.playing
    || (Object.values(S.streams).find(s => s.connected) || {}).mount;
  if (target) switchAudio(target);
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

# ── Scanner (rtl_fm subprocess) ────────────────────────────────────────────────
class RTLFMScanner:
    """
    Runs rtl_fm in scan mode, reads raw PCM from stdout, parses stderr for
    frequency info, and detects squelch via RMS level.
    """
    CHUNK_SECS = 0.1   # seconds of audio per processing chunk

    def __init__(self, name: str, channels: dict[str, str],
                 squelch: int = 70, squelch_rms: float = 0.003,
                 squelch_hold: float = 2.0,
                 channel_squelch: dict[str, float] | None = None,
                 ppm: int = 0, modulation: str = "fm",
                 device: str = "0", gain: str = "auto",
                 samp_rate: int = 250000, scan_dwell: float = 0.5,
                 debug: bool = False,
                 on_event=None, on_audio=None):
        self.name            = name
        self.channels        = channels
        self.frequencies     = sorted(float(f) for f in channels)
        self.squelch         = squelch
        self.squelch_rms     = squelch_rms    # default Python-side threshold (0.0–1.0)
        self.squelch_hold    = squelch_hold   # seconds before clearing inactive freq
        self.channel_squelch = channel_squelch or {}  # per-freq overrides
        self.debug           = debug
        self.ppm          = ppm
        self.modulation   = modulation
        self.device       = str(device)
        self.gain         = str(gain)
        self.samp_rate    = samp_rate
        self.scan_dwell   = scan_dwell
        # rtl_fm can only downsample, not upsample. Output rate must be ≤ samp_rate.
        self.audio_rate   = min(samp_rate, AUDIO_RATE)
        self._on_event  = on_event
        self._on_audio  = on_audio

        self._running      = False
        self._lock         = threading.Lock()
        self._active_freq  = None
        self._active_since = None
        self._history      = deque(maxlen=20)
        self.connected     = False
        self.last_error: str | None = None

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
        threading.Thread(target=self._loop, daemon=True, name="scanner").start()

    def stop(self):
        self._running = False

    def _emit(self, evt: dict):
        if self._on_event: self._on_event(evt)

    def _emit_audio(self, pcm: bytes):
        if self._on_audio: self._on_audio(pcm)

    @staticmethod
    def _rms(data: bytes) -> float:
        s = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(s ** 2))) if len(s) else 0.0

    def _loop(self):
        import time
        while self._running:
            self.connected = False
            self.last_error = None
            self._emit({"type": "conn", "mount": "sdr", "connected": False, "error": None})
            try:
                self._run()
            except Exception as exc:
                self.last_error = str(exc)
                print(f"[Scanner] error: {exc}")
                self._emit({"type": "conn", "mount": "sdr",
                            "connected": False, "error": str(exc)})
            time.sleep(5)

    def _run(self):
        import time, sys

        if not shutil.which("rtl_fm"):
            raise RuntimeError("rtl_fm not found — sudo apt install rtl-sdr")

        freq_keys = list(self.channels.keys())   # preserve config order
        n_freqs   = len(freq_keys)
        CHUNK     = int(self.audio_rate * 2 * self.CHUNK_SECS)

        overrides = ", ".join(f"{f}={v}" for f, v in sorted(self.channel_squelch.items()))
        print(f"[Scanner] started — {n_freqs} freq(s), scan_dwell={self.scan_dwell}s, "
              f"squelch_rms={self.squelch_rms}, squelch_hold={self.squelch_hold}s, "
              f"mod={self.modulation}, samp_rate={self.samp_rate}, audio_rate={self.audio_rate}"
              + (f", per-channel: {overrides}" if overrides else ""))

        self.connected = True
        self._emit({"type": "conn", "mount": "sdr", "connected": True, "error": None})

        scan_idx = 0

        while self._running:
            freq_str  = freq_keys[scan_idx]
            freq_mhz  = float(freq_str)
            label     = self.channels[freq_str]
            threshold = self.channel_squelch.get(freq_str, self.squelch_rms)

            cmd = ["rtl_fm",
                   "-f", f"{freq_mhz:.3f}M",
                   "-M", self.modulation,
                   "-s", str(self.samp_rate),
                   "-r", str(self.audio_rate),
                   "-p", str(self.ppm),
                   "-d", self.device]
            if self.squelch > 0:
                cmd += ["-l", str(self.squelch)]
            if self.gain.lower() != "auto":
                cmd += ["-g", self.gain]
            cmd += ["-"]

            if self.debug:
                print(f"[scan] → {freq_str} MHz  ({label})")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            def _drain_stderr(p=proc):
                for line in p.stderr:
                    text = line.decode("utf-8", errors="replace").strip()
                    if text and not _RTLFM_NOISE.search(text):
                        print(f"[rtl_fm] {text}")
            threading.Thread(target=_drain_stderr, daemon=True).start()

            proc_start   = time.time()
            dwell_start  = None    # set after first PCM chunk arrives
            squelch_open = False
            last_sig_t   = 0.0

            try:
                while self._running:
                    data = proc.stdout.read(CHUNK)
                    if not data:
                        break

                    if dwell_start is None:
                        dwell_start = time.time()

                    rms    = self._rms(data)
                    db     = 20.0 * np.log10(max(rms, 1e-9))
                    active = rms > threshold

                    self._emit({"type": "signal", "mount": "sdr",
                                "db": round(db, 1), "active": active})

                    if active:
                        last_sig_t  = time.time()
                        dwell_start = time.time()   # reset dwell while signal present

                        if not squelch_open or self._active_freq != freq_str:
                            squelch_open = True
                            with self._lock:
                                now   = datetime.now()
                                self._active_freq  = freq_str
                                self._active_since = now
                                self._history.appendleft((now, freq_str, label))
                            if self.debug:
                                print(f"[Scanner] active: {freq_str} MHz  ({db:.1f} dB)")
                            self._emit({
                                "type":  "freq_change", "mount": "sdr",
                                "name":  self.name, "freq": freq_str, "label": label,
                                "time":  now.isoformat(),
                            })

                        self._emit_audio(data)

                    else:
                        if squelch_open and time.time() - last_sig_t > self.squelch_hold:
                            squelch_open = False
                            with self._lock:
                                self._active_freq  = None
                                self._active_since = None
                            if self.debug:
                                print("[Scanner] squelch closed")
                            self._emit({"type": "freq_clear", "mount": "sdr"})
                            if n_freqs > 1:
                                break   # advance to next frequency

                        elif not squelch_open and n_freqs > 1:
                            now_t = time.time()
                            if dwell_start is not None and now_t - dwell_start > self.scan_dwell:
                                break
                            if dwell_start is None and now_t - proc_start > self.scan_dwell + 5.0:
                                if self.debug:
                                    print(f"[scan] {freq_str}: rtl_fm startup timeout")
                                break

            finally:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()

            scan_idx = (scan_idx + 1) % n_freqs
            if n_freqs > 1 and self._running:
                time.sleep(0.05)   # brief pause between hops for USB device to settle


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
scanner: RTLFMScanner | None            = None
wsman:   WsManager                      = WsManager()
_evq:    asyncio.Queue | None           = None
_evloop: asyncio.AbstractEventLoop | None = None
_audio_clients: list[asyncio.Queue]    = []


def _emit(event: dict):
    if _evloop and _evq:
        asyncio.run_coroutine_threadsafe(_evq.put(event), _evloop)


def _audio_cb(pcm: bytes):
    if _evloop and _audio_clients:
        asyncio.run_coroutine_threadsafe(_dispatch_audio(pcm), _evloop)


async def _dispatch_audio(pcm: bytes):
    for q in list(_audio_clients):
        try: q.put_nowait(pcm)
        except asyncio.QueueFull: pass


async def _bcast_loop():
    while True:
        event = await _evq.get()
        await wsman.broadcast(event)


def _state() -> dict:
    s = scanner
    return {
        "type":       "state",
        "audio_rate": s.audio_rate,
        "streams": [{
            "mount":      "sdr",
            "name":       s.name,
            "connected":  s.connected,
            "activeFreq": s.active_freq,
            "activeSince": s.active_since.isoformat() if s.active_since else None,
            "history": [{"time": t.isoformat(), "freq": f, "label": lb}
                        for t, f, lb in s.history[:10]],
            "channels":  s.channels,
            "lastError": s.last_error,
        }],
    }


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI()


@app.on_event("startup")
async def _startup():
    global _evq, _evloop
    _evloop = asyncio.get_running_loop()
    _evq    = asyncio.Queue()
    asyncio.create_task(_bcast_loop())
    scanner.start()
    print("Scanner started")


@app.on_event("shutdown")
async def _shutdown():
    if scanner: scanner.stop()


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE


@app.get("/debug")
async def debug():
    return {
        "connected":     scanner.connected if scanner else None,
        "active_freq":   scanner.active_freq if scanner else None,
        "audio_clients": len(_audio_clients),
        "queue_size":    _evq.qsize() if _evq else -1,
    }


@app.websocket("/ws/audio")
async def audio_ws(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=30)
    _audio_clients.append(q)
    try:
        while True:
            try:
                data = await asyncio.wait_for(q.get(), timeout=5.0)
            except asyncio.TimeoutError:
                rate = scanner.audio_rate if scanner else AUDIO_RATE
                data = bytes(int(rate * 2 * 0.1))   # 100 ms silence keepalive
            await ws.send_bytes(data)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        try: _audio_clients.remove(q)
        except ValueError: pass


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await wsman.connect(ws)

    async def _keepalive():
        while True:
            await asyncio.sleep(25)
            try: await ws.send_text(json.dumps({"type": "ping"}))
            except Exception: return

    ka = asyncio.create_task(_keepalive())
    try:
        await ws.send_text(json.dumps(_state(), default=str))
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        ka.cancel()
        await wsman.disconnect(ws)


# ── Entry point ────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = Path(__file__).parent / "scanner_config.json"


def main():
    global scanner

    p = argparse.ArgumentParser(description="RTL-Airband Scanner")
    p.add_argument("--config",      default=str(DEFAULT_CONFIG))
    p.add_argument("--listen-port", type=int, default=8080)
    p.add_argument("--debug",       action="store_true",
                   help="Log every chunk above threshold: freq, dB, threshold source, squelch state")
    args = p.parse_args()

    cfg: dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"Config: {config_path}")
    else:
        print(f"Warning: {config_path} not found — using defaults")

    # Parse channels: supports "freq": "label" or "freq": {"label": "...", "squelch_rms": 0.056}
    raw_channels    = cfg.get("channels", {"446.000": "446.000"})
    channels        : dict[str, str]   = {}
    channel_squelch : dict[str, float] = {}
    for freq, val in raw_channels.items():
        if isinstance(val, dict):
            channels[freq] = val.get("label", freq)
            if "squelch_rms" in val:
                channel_squelch[freq] = float(val["squelch_rms"])
        else:
            channels[freq] = str(val)

    scanner = RTLFMScanner(
        name            = cfg.get("name", "Scanner"),
        channels        = channels,
        squelch         = cfg.get("squelch", 70),
        squelch_rms     = cfg.get("squelch_rms", 0.003),
        squelch_hold    = cfg.get("squelch_hold", 2.0),
        channel_squelch = channel_squelch,
        ppm             = cfg.get("ppm", 0),
        modulation      = cfg.get("modulation", "fm"),
        device          = cfg.get("device", "0"),
        gain            = cfg.get("gain", "auto"),
        samp_rate       = cfg.get("samp_rate", 250000),
        scan_dwell      = cfg.get("scan_dwell", 0.5),
        debug           = args.debug,
        on_event        = _emit,
        on_audio        = _audio_cb,
    )

    print(f"Open http://<pi-ip>:{args.listen_port} in your browser")
    uvicorn.run(app, host="0.0.0.0", port=args.listen_port, log_level="warning")


if __name__ == "__main__":
    main()
