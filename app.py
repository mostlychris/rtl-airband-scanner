#!/usr/bin/env python3
"""
SDR Scanner — pyrtlsdr direct mode.

Install:
    sudo apt install rtl-sdr python3-numpy
    pip install fastapi "uvicorn[standard]" pyrtlsdr
    cp scanner_config.example.json scanner_config.json
    nano scanner_config.json
    python3 app.py
"""
from __future__ import annotations

import json, asyncio, threading, argparse, uvicorn, struct, signal
import numpy as np
from scipy.signal import lfilter, firwin, butter, lfilter_zi, iirnotch
from pathlib import Path
from datetime import datetime
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response

VERSION    = "3.1.0"
AUDIO_RATE = 24000   # PCM output rate; default hw_rate = 240,000 Hz (10× oversample)


def _wav_header(sample_rate: int = AUDIO_RATE) -> bytes:
    """WAV header for a streaming (infinite-length) 16-bit mono PCM stream."""
    byte_rate = sample_rate * 2
    return (
        b'RIFF' + struct.pack('<I', 0xFFFFFFFF) +
        b'WAVE' +
        b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, byte_rate, 2, 16) +
        b'data' + struct.pack('<I', 0xFFFFFFFF)
    )


# ── CTCSS (PL tone) detection ──────────────────────────────────────────────────
_CTCSS_TONES = [
    67.0, 69.3, 71.9, 74.4, 77.0, 79.7, 82.5, 85.4, 88.5, 91.5, 94.8, 97.4,
    100.0, 103.5, 107.2, 110.9, 114.8, 118.8, 123.0, 127.3, 131.8, 136.5,
    141.3, 146.2, 150.0, 151.4, 156.7, 159.8, 162.2, 165.5, 167.9, 171.3,
    173.8, 177.3, 179.9, 183.5, 186.2, 189.9, 192.8, 196.6, 199.5, 203.5,
    206.5, 210.7, 218.1, 225.7, 229.1, 233.6, 241.8, 250.3, 254.1,
]
_CTCSS_WINDOW = 4096   # samples to accumulate before evaluating (~170 ms at 24 kHz)


def _ctcss_analyze(buf: np.ndarray, sample_rate: float,
                   target_hz: float = 0.0) -> tuple[bool, float | None]:
    """Analyze buf for CTCSS tone presence.

    Returns (gated_open, detected_tone):
      - detected_tone: the CTCSS tone with clearly dominant power, or None.
      - gated_open: True if no target is configured (target_hz == 0) OR
        the target tone is dominant over all tones more than 5 Hz away.

    Uses exact per-tone DFT (not nearest FFT bin) so that closely-spaced
    tones such as 97.4 Hz and 100.0 Hz are always measured independently.
    Both the detection display and gating use the same dominance criterion,
    so a tone shown in the UI will reliably gate the squelch.
    """
    n = len(buf)
    win     = np.hanning(n)
    win_buf = buf.astype(np.float64) * win
    norm    = float(np.dot(win, win))   # sum(win²) — normalises to amplitude²

    # Vectorised exact DFT at every CTCSS frequency in one matrix multiply.
    # phases shape: (num_tones, n)
    tones_arr = np.array(_CTCSS_TONES, dtype=np.float64)
    t         = np.arange(n, dtype=np.float64) / sample_rate
    phases    = 2.0 * np.pi * np.outer(tones_arr, t)
    re        = np.cos(phases) @ win_buf   # (num_tones,)
    im        = np.sin(phases) @ win_buf
    powers    = (re * re + im * im) / norm

    tone_powers = list(zip(_CTCSS_TONES, powers.tolist()))

    def _is_dominant(target: float, tgt_pwr: float) -> bool:
        """True when tgt_pwr is >= every excluding-tone power AND above their mean."""
        exc = [p for hz, p in tone_powers if abs(hz - target) >= 5.0]
        exc.append(tgt_pwr)
        return tgt_pwr >= max(exc) and tgt_pwr > sum(exc) / len(exc)

    # Detected tone for display — only shown when it also passes the dominance
    # test, so the UI never reports a tone that the gate would reject.
    best_idx  = int(np.argmax(powers))
    best_tone = _CTCSS_TONES[best_idx]
    detected  = best_tone if _is_dominant(best_tone, float(powers[best_idx])) else None

    if target_hz > 0.0:
        # Map the configured Hz to the nearest entry in the CTCSS table.
        tgt_idx  = min(range(len(_CTCSS_TONES)),
                       key=lambda i: abs(_CTCSS_TONES[i] - target_hz))
        tgt_pwr  = float(powers[tgt_idx])
        gated    = _is_dominant(_CTCSS_TONES[tgt_idx], tgt_pwr)
    else:
        gated = True   # no filter configured — always pass

    return gated, detected


# ── Broadcastify / Icecast feeder ─────────────────────────────────────────────
import subprocess as _subprocess

class BroadcastifyFeeder:
    """Continuous MP3 stream to a Broadcastify (Icecast) server.

    Reads raw 16-bit mono PCM from an internal queue and pipes it to ffmpeg,
    which encodes to MP3 and pushes via the Icecast protocol.  Self-generates
    silence between scanner transmissions so the stream stays alive.
    Auto-reconnects on disconnect with a 5-second delay.
    """
    _CHUNK_SAMPLES = 1200          # 50 ms of silence at 24 kHz
    _RECONNECT_DELAY = 5.0

    def __init__(self, server: str, port: int, mountpoint: str, password: str,
                 bitrate: int = 32, sample_rate: int = AUDIO_RATE,
                 on_status=None):
        self.url         = f"icecast://source:{password}@{server}:{port}{mountpoint}"
        self.bitrate     = bitrate
        self.sample_rate = sample_rate
        self._on_status  = on_status   # callback(connected: bool, error: str|None)
        self._q: _q_mod.Queue = _q_mod.Queue(maxsize=200)
        self._running    = False
        self._thread     = None
        self.connected   = False
        self.last_error: str | None = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True,
                                          name="broadcastify-feeder")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8.0)

    def send(self, pcm: bytes):
        """Feed a chunk of raw 16-bit mono PCM. Drop oldest if queue full."""
        try: self._q.put_nowait(pcm)
        except _q_mod.Full:
            try: self._q.get_nowait()
            except _q_mod.Empty: pass
            try: self._q.put_nowait(pcm)
            except _q_mod.Full: pass

    def _notify(self, connected: bool, error: str | None = None):
        self.connected  = connected
        self.last_error = error
        if self._on_status:
            try: self._on_status(connected, error)
            except Exception: pass

    def _run(self):
        import time as _t
        silence = bytes(self._CHUNK_SAMPLES * 2)
        chunk_secs = self._CHUNK_SAMPLES / self.sample_rate

        while self._running:
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-f', 's16le', '-ar', str(self.sample_rate), '-ac', '1',
                '-i', 'pipe:0',
                '-c:a', 'libmp3lame', '-b:a', f'{self.bitrate}k',
                '-f', 'mp3', self.url,
            ]
            proc = None
            try:
                proc = _subprocess.Popen(cmd, stdin=_subprocess.PIPE,
                                          stderr=_subprocess.PIPE)
                self._notify(True)
                print(f"[Broadcastify] Connected → {self.url.split('@')[-1]}")

                while self._running:
                    if proc.poll() is not None:
                        stderr = proc.stderr.read().decode(errors='replace').strip()
                        raise RuntimeError(f"ffmpeg exited: {stderr or 'no output'}")
                    try:
                        chunk = self._q.get(timeout=chunk_secs)
                    except _q_mod.Empty:
                        chunk = silence
                    proc.stdin.write(chunk)
                    proc.stdin.flush()

            except Exception as exc:
                err = str(exc)
                print(f"[Broadcastify] Error: {err}")
                self._notify(False, err)
            else:
                self._notify(False)

            if proc is not None:
                try: proc.stdin.close()
                except Exception: pass
                try: proc.wait(timeout=3)
                except Exception:
                    try: proc.kill()
                    except Exception: pass

            if self._running:
                _t.sleep(self._RECONNECT_DELAY)

        self._notify(False)
        print("[Broadcastify] Feeder stopped")

import queue as _q_mod   # alias so it doesn't shadow inner-function imports


# ── Embedded page ──────────────────────────────────────────────────────────────
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#0a0d0f">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<link rel="manifest" href="/manifest.json">
<link rel="apple-touch-icon" href="/icon.svg">
<title>SDR Scanner</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#04080f;--card:rgba(15,25,55,0.55);--card2:rgba(20,35,70,0.5);--border:rgba(80,130,220,0.2);
  --panel:rgba(5,10,28,0.75);--panel-b:rgba(10,18,50,0.6);
  --text:#ccd8f0;--muted:#7fa8d8;--dim:#1a2a4a;
  --green:#2dff6e;--gdim:rgba(45,255,110,.07);--gborder:rgba(45,255,110,.28);
  --amber:#ffaa00;--cyan:#00d4ff;--blue:#4d8aff;--red:#ff4455;--purple:#9966dd;--yellow:#ffcc00;
  --mono:'SF Mono','Fira Code','Consolas',monospace;
  --glow:0 0 6px var(--green),0 0 14px rgba(45,255,110,.35);
  --glow-sm:0 0 4px rgba(45,255,110,.5);
  --amber-glow:0 0 5px rgba(255,170,0,.45);
  --cyan-glow:0 0 5px rgba(0,212,255,.4);
  --glass-border:rgba(100,160,255,0.22);
  --glass-hi:rgba(140,190,255,0.10);
}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px;min-height:100vh;
  background-image:
    radial-gradient(ellipse 90% 70% at 15% 5%, rgba(20,60,160,0.7) 0%,transparent 55%),
    radial-gradient(ellipse 70% 55% at 85% 85%, rgba(10,40,120,0.6) 0%,transparent 50%),
    radial-gradient(ellipse 60% 50% at 70% 20%, rgba(30,80,200,0.35) 0%,transparent 45%),
    radial-gradient(ellipse 80% 60% at 30% 80%, rgba(8,30,90,0.5) 0%,transparent 50%),
    linear-gradient(160deg,#04091a 0%,#020610 50%,#03080f 100%);
}

/* ── Header ───────────────────────────────────────────────── */
header{
  background:rgba(8,18,50,0.75);
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border-bottom:1px solid var(--glass-border);
  box-shadow:0 4px 24px rgba(0,0,10,.8),inset 0 1px 0 var(--glass-hi);
  padding:10px 20px;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:100
}
h1{font-size:12px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:#6a8aaa;display:flex;align-items:center;gap:8px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted);flex-shrink:0;transition:background .3s,box-shadow .3s}
.dot.ok{background:var(--green);box-shadow:var(--glow)}
.dot.err{background:var(--red);box-shadow:0 0 6px var(--red)}
.st{font-size:10px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase}
.spacer{flex:1}
.abtn{
  background:rgba(12,24,65,0.65);border:1px solid var(--glass-border);border-radius:4px;
  color:#5a7a9a;cursor:pointer;font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  padding:5px 14px;display:flex;align-items:center;gap:6px;transition:all .15s;
  box-shadow:0 2px 0 rgba(4,6,9,.5),inset 0 1px 0 var(--glass-hi)
}
.abtn:hover{border-color:var(--green);color:var(--green);box-shadow:0 0 10px rgba(45,255,110,.18),0 2px 0 rgba(4,6,9,.5)}
.abtn:active{transform:translateY(1px);box-shadow:0 1px 0 rgba(4,6,9,.5)}
.abtn.on{border-color:rgba(45,255,110,.5);color:var(--green);background:rgba(45,255,110,.06)}
.asrc{font-size:10px;color:var(--cyan);letter-spacing:.08em;text-transform:uppercase;opacity:.8}
.bcast-wrap{display:flex;align-items:center;gap:5px;flex-shrink:0}
.bcast-dot{width:6px;height:6px;border-radius:50%;background:var(--dim);flex-shrink:0;transition:background .3s,box-shadow .3s}
.bcast-dot.live{background:#ff4020;box-shadow:0 0 6px rgba(255,64,32,.7),0 0 14px rgba(255,64,32,.35)}
.bcast-dot.err{background:var(--red);box-shadow:0 0 5px var(--red)}
.bcast-lbl{font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);transition:color .3s}
.bcast-lbl.live{color:#ff6040}
.bcast-lbl.err{color:var(--red)}

/* ── Layout ───────────────────────────────────────────────── */
.app-layout{max-width:1260px;margin:0 auto}
main{padding:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-bottom:20px}

/* ── Audio controls inline bar ────────────────────────────── */
.acontrols{
  display:flex;flex-direction:row;align-items:center;
  background:rgba(8,18,52,0.78);border-top:1px solid var(--glass-border);
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
}
.acontrols.hidden{display:none}
#ac-body{display:flex;flex-direction:row;align-items:center;flex-wrap:wrap;gap:4px 18px;padding:4px 14px;flex:1}
#ac-acts-wrap{flex:1;display:flex;justify-content:center;align-items:center}
#ac-acts .sc-btn{padding:4px 7px;font-size:7px;letter-spacing:.08em;white-space:nowrap}
#ac-acts .sc-acts{display:flex;flex-direction:column;gap:2px;padding:0;background:none;border:none}
.ac-knob-wrap{display:flex;flex-direction:column;align-items:center;gap:2px;padding:4px 0 2px}
.ac-knob-lbl{font-size:7px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--cyan);opacity:.65}
.ac-knob{cursor:ns-resize;display:block;touch-action:none}
.ac-knob-val{font-family:var(--mono);font-size:10px;color:var(--amber);letter-spacing:.05em}
.ac-seg-group{padding:6px 0 2px}
.ac-seg-lbl{font-size:7px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--cyan);opacity:.65;margin-bottom:5px}
.ac-seg{display:flex;flex-direction:column;gap:2px}
.ac-seg-btn{flex:1;background:rgba(6,10,22,0.7);border:1px solid rgba(26,32,53,0.8);color:var(--muted);border-radius:2px;padding:5px 6px;font-size:9px;font-weight:700;letter-spacing:.06em;cursor:pointer;font-family:var(--mono);transition:all .15s;line-height:1}
.ac-seg-btn:hover{border-color:rgba(42,58,74,.7);color:var(--text)}
.ac-seg-btn.active{background:rgba(45,255,110,.07);border-color:rgba(45,255,110,.4);color:var(--green)}
.ac-tog-row{display:flex;align-items:center;gap:7px;padding:6px 0;cursor:pointer;user-select:none}
.ac-sw{width:26px;height:14px;background:#111827;border:1px solid #1a2035;border-radius:7px;flex-shrink:0;position:relative;transition:background .2s,border-color .2s}
.ac-sw-t{position:absolute;top:2px;left:2px;width:8px;height:8px;border-radius:50%;background:#2a3a4a;transition:left .2s,background .2s}
input:checked~.ac-sw{background:rgba(45,255,110,.15);border-color:rgba(45,255,110,.35)}
input:checked~.ac-sw .ac-sw-t{left:14px;background:var(--green)}
.ac-tog-lbl{font-size:8px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}

/* ── Scanner card ─────────────────────────────────────────── */
.scard{
  background:rgba(12,22,60,0.52);border:1px solid var(--glass-border);border-radius:8px;
  overflow:clip;transition:border-color .2s,box-shadow .2s;
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  box-shadow:0 6px 32px rgba(0,5,30,.7),inset 0 1px 0 var(--glass-hi)
}
.sc-sticky{position:sticky;top:43px;z-index:10;background:rgba(10,20,55,0.88);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px)}

.scard.playing{border-color:rgba(45,255,110,.4);box-shadow:0 0 24px rgba(45,255,110,.12),0 6px 32px rgba(0,5,30,.7),inset 0 1px 0 rgba(45,255,110,.06)}

/* ── Panel header (stream name + status) ─────────────────── */
.sc-panel-hdr{
  display:flex;align-items:center;gap:8px;padding:7px 12px 6px;
  background:linear-gradient(180deg,rgba(20,38,90,0.75) 0%,rgba(10,20,55,0.7) 100%);
  border-bottom:1px solid var(--glass-border);
}
.sc-name{font-size:10px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:#b0c4d8;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sc-chl-btn{background:rgba(12,24,65,.6);border:1px solid var(--glass-border);border-radius:3px;color:var(--muted);cursor:pointer;font-size:8px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;padding:3px 10px;white-space:nowrap;transition:all .15s;flex-shrink:0}
.sc-chl-btn:hover{border-color:rgba(100,160,255,.4);color:var(--cyan);box-shadow:var(--cyan-glow)}
.sc-status{display:flex;align-items:center;gap:5px;font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;flex-shrink:0}
.sc-led{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.sc-status.ok{color:var(--green)}.sc-status.ok .sc-led{background:var(--green);box-shadow:var(--glow-sm)}
.sc-status.scanning .sc-led,.sc-status.scanning{animation:pulse 1.4s ease-in-out infinite}
.sc-status.err{color:var(--red)}.sc-status.err .sc-led{background:var(--red);box-shadow:0 0 5px var(--red)}
.sc-status.warn{color:var(--yellow)}.sc-status.warn .sc-led{background:var(--yellow)}
.serr{font-size:10px;color:var(--red);padding:5px 12px;background:rgba(255,68,85,.07);border-bottom:1px solid rgba(255,68,85,.15);letter-spacing:.06em}

/* ── Frequency display (LCD panel) ───────────────────────── */
.sc-display{
  background:rgba(4,8,28,0.72);
  padding:10px 14px 8px;
  border-bottom:1px solid rgba(40,70,160,0.25);
  box-sizing:border-box;overflow:hidden;
}
.sc-lbl-row{display:flex;align-items:center;gap:10px;min-width:0}
.sc-lbl{
  font-family:var(--mono);font-size:26px;font-weight:600;
  letter-spacing:.04em;line-height:1;
  color:var(--muted);text-transform:uppercase;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  flex:1;min-width:0;
  transition:color .4s,text-shadow .4s;
}
.sc-display.active .sc-lbl{color:var(--green);text-shadow:var(--glow)}
.sc-meta{display:flex;align-items:center;gap:10px;margin-top:6px;height:16px;overflow:hidden}
.sc-freq{
  font-family:var(--mono);font-size:13px;font-weight:500;
  letter-spacing:.06em;color:var(--muted);
  transition:color .3s,text-shadow .3s;flex:1;
}
.sc-display.active .sc-freq{color:var(--amber);text-shadow:var(--amber-glow)}
.sc-unit{font-size:11px;color:var(--muted);margin-left:3px}
.sc-timer{font-family:var(--mono);font-size:11px;color:var(--muted);transition:color .3s;flex-shrink:0}
.sc-display.active .sc-timer{color:var(--cyan);opacity:.8}
.sc-pl{font-family:var(--mono);font-size:10px;color:var(--purple);letter-spacing:.08em;flex-shrink:0;opacity:.85}
.sc-ctcss{font-family:var(--mono);font-size:10px;letter-spacing:.08em;flex-shrink:0;transition:color .3s}
.sc-ctcss.info{color:var(--cyan);opacity:.85}
.sc-ctcss.match{color:var(--green);text-shadow:var(--glow-sm)}

/* ── Signal meter (inline segmented blocks) ────────────────── */
.sc-meter{display:flex;align-items:center;gap:5px;flex-shrink:0;margin-left:auto}
.sc-segs{display:flex;gap:2px;align-items:flex-end}
.sc-seg{
  width:5px;border-radius:1px;
  background:var(--dim);transition:background .1s,box-shadow .1s;
}
/* varying heights for a classic bar-graph look */
.sc-seg:nth-child(-n+4){height:8px}
.sc-seg:nth-child(n+5):nth-child(-n+8){height:11px}
.sc-seg:nth-child(n+9):nth-child(-n+11){height:14px}
.sc-seg:nth-child(n+12){height:17px}
/* lit colours per zone */
.sc-seg.lit-g{background:#2dff6e;box-shadow:0 0 4px rgba(45,255,110,.6)}
.sc-seg.lit-y{background:#ffb800;box-shadow:0 0 4px rgba(255,184,0,.5)}
.sc-seg.lit-r{background:#ff3344;box-shadow:0 0 4px rgba(255,51,68,.5)}
.sc-db{font-family:var(--mono);font-size:10px;color:var(--muted);width:9ch;text-align:right;white-space:nowrap;transition:color .3s;flex-shrink:0}
.sc-display.active .sc-db{color:var(--green)}

/* ── Action buttons (SKIP / EDIT / DEL) ─────────────────── */
.sc-acts{
  display:flex;gap:5px;padding:8px 10px;
  background:rgba(8,16,46,0.65);border-bottom:1px solid var(--glass-border);
}
.sc-btn{
  flex:1;background:rgba(10,20,55,0.55);border:1px solid rgba(60,100,200,0.2);border-radius:3px;
  color:#4a5a6a;cursor:pointer;font-size:9px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  padding:5px 2px;text-align:center;
  transition:all .1s;
  box-shadow:0 2px 0 rgba(4,6,9,.6),inset 0 1px 0 var(--glass-hi);
}
.sc-btn:active{transform:translateY(1px);box-shadow:0 1px 0 #040609}
.sc-btn.skip:hover{border-color:rgba(255,170,0,.45);color:var(--amber);box-shadow:0 0 6px rgba(255,170,0,.15),0 2px 0 #040609}
.sc-btn.skip.active{border-color:rgba(255,170,0,.4);color:var(--amber);background:rgba(255,170,0,.06)}
.sc-btn.hold:hover{border-color:rgba(255,170,0,.45);color:var(--amber);box-shadow:0 0 6px rgba(255,170,0,.15),0 2px 0 #040609}
.sc-btn.hold.active{border-color:rgba(255,170,0,.55);color:var(--amber);background:rgba(255,170,0,.08);text-shadow:0 0 6px rgba(255,170,0,.6)}
.sc-btn.resume:hover{border-color:rgba(45,255,110,.45);color:var(--green);box-shadow:0 0 6px rgba(45,255,110,.15),0 2px 0 #040609}
.sc-btn.edit:hover{border-color:rgba(0,212,255,.4);color:var(--cyan);box-shadow:var(--cyan-glow),0 2px 0 #040609}
.sc-btn.del:hover{border-color:rgba(255,68,85,.4);color:var(--red);box-shadow:0 0 6px rgba(255,68,85,.15),0 2px 0 #040609}
.sc-btn:disabled,.sc-acts.idle .sc-btn{opacity:.2;pointer-events:none}
.sc-acts.idle .sc-btn.hold.active{opacity:1;pointer-events:auto}

/* ── Channel bank list ─────────────────────────────────────── */
.chlist{padding:3px 0}
.sc-chl-hdr{
  padding:5px 12px;font-size:8px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;
  color:#5a8acc;border-bottom:1px solid var(--glass-border);background:rgba(10,20,58,0.65);
  cursor:pointer;user-select:none;display:flex;align-items:center;justify-content:space-between;
}
.sc-chl-hdr:hover{color:var(--cyan)}
.coll-arrow{font-size:8px;opacity:.5}
.ch{display:flex;align-items:center;gap:8px;padding:4px 12px;transition:background .15s;cursor:pointer}
.ch:hover{background:var(--card2)}
.ch.active{background:var(--gdim);border-left:2px solid var(--green);padding-left:10px}
.ch.active:hover{background:rgba(45,255,110,.1)}
.ch.held{background:rgba(255,176,0,.06);border-left:2px solid var(--amber);padding-left:10px}
.ch.held:hover{background:rgba(255,176,0,.12)}
.ch-dot{font-size:9px;color:var(--muted);width:10px;flex-shrink:0;transition:color .2s}
.ch.active .ch-dot{color:var(--green);text-shadow:var(--glow-sm)}
.ch-f{font-family:var(--mono);font-size:12px;font-weight:600;width:74px;flex-shrink:0;color:#3a5a6a;transition:color .2s}
.ch.active .ch-f{color:var(--amber);text-shadow:var(--amber-glow)}
.ch-l{color:var(--muted);font-size:11px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;letter-spacing:.02em}
.ch.active .ch-l{color:var(--text)}
.ch-t{font-size:10px;color:var(--muted);font-family:var(--mono);flex-shrink:0;transition:color .2s}
.ch.active .ch-t{color:var(--cyan);opacity:.7}
.noch{padding:12px;color:var(--muted);font-size:10px;text-align:center;letter-spacing:.15em;text-transform:uppercase}
.ch-acts{display:flex;gap:3px;margin-left:auto;flex-shrink:0;padding-left:6px}
.ch-btn{
  background:rgba(8,18,32,0.7);border:1px solid rgba(26,42,62,0.7);cursor:pointer;
  padding:2px 8px;font-size:8px;font-weight:700;color:var(--muted);border-radius:2px;
  letter-spacing:.1em;text-transform:uppercase;white-space:nowrap;line-height:1.6;
  transition:background .12s,border-color .12s,color .12s;
}
.ch-btn:hover{background:rgba(18,32,64,.7);border-color:rgba(42,74,112,.6);color:var(--text)}
.ch-btn.del:hover{background:#200a0a;border-color:#502020;color:var(--red)}
.ch-btn.skip:hover{background:#1a1200;border-color:#504020;color:var(--amber)}
.ch.skipped .ch-f,.ch.skipped .ch-l,.ch.skipped .ch-t,.ch.skipped .ch-dot{opacity:.45}
.ch.skipped .ch-btn.skip{color:var(--amber);border-color:#503010;background:#140e00}
.ch-edit-row{display:flex;align-items:center;gap:6px;padding:5px 12px;flex-wrap:wrap}
.ch-edit-in{background:rgba(4,8,18,0.7);border:1px solid rgba(30,42,62,0.8);color:var(--text);border-radius:3px;padding:2px 6px;font-size:11px;font-family:var(--mono);min-width:0}
.ch-edit-lbl{flex:1}.ch-edit-sq{width:64px}.ch-edit-gn{width:72px}.ch-edit-pl{width:64px}
.ch-pl{font-size:9px;color:var(--purple);letter-spacing:.05em;flex-shrink:0;opacity:.8}
.ch-sq{font-size:9px;color:var(--cyan);letter-spacing:.05em;flex-shrink:0;opacity:.65}
.ch-gn{font-size:9px;color:var(--amber);letter-spacing:.05em;flex-shrink:0;opacity:.65}
.ch-save{background:rgba(45,255,110,.08);border:1px solid rgba(45,255,110,.3);border-radius:3px;color:var(--green);cursor:pointer;font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:2px 10px;white-space:nowrap}
.ch-cancel{background:none;border:1px solid #1e2a3e;border-radius:3px;color:var(--muted);cursor:pointer;font-size:9px;letter-spacing:.06em;padding:2px 8px}
.ch-add-btn{display:flex;align-items:center;gap:5px;padding:5px 12px;font-size:10px;color:var(--muted);cursor:pointer;border-top:1px solid var(--glass-border);transition:color .15s;letter-spacing:.1em;text-transform:uppercase}
.ch-add-btn:hover{color:var(--cyan)}
.ch-bank-badge{font-size:8px;color:#5a8aaa;background:#0a1824;border:1px solid #1a3040;border-radius:2px;padding:1px 5px;flex-shrink:0;letter-spacing:.05em}
.ch-mode-badge{font-size:8px;color:#aa7a5a;background:#181008;border:1px solid #302010;border-radius:2px;padding:1px 5px;flex-shrink:0;letter-spacing:.05em;text-transform:uppercase}

/* ── Channel edit modal ─────────────────────────────────────── */
.modal-backdrop{
  position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:200;
  display:flex;align-items:center;justify-content:center;
  opacity:0;pointer-events:none;transition:opacity .15s;
  backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);
}
.modal-backdrop.open{opacity:1;pointer-events:auto}
.modal{
  background:rgba(10,22,65,0.82);border:1px solid var(--glass-border);border-radius:8px;
  width:min(480px,96vw);max-height:90vh;overflow-y:auto;
  backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
  box-shadow:0 8px 48px rgba(0,5,30,.9),inset 0 1px 0 var(--glass-hi);
  transform:translateY(8px);transition:transform .15s;
}
.modal-backdrop.open .modal{transform:translateY(0)}
.modal-hdr{
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 18px;border-bottom:1px solid var(--glass-border);
  font-size:12px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--cyan);
}
.modal-close{background:rgba(255,255,255,.04);border:1px solid var(--glass-border);border-radius:3px;color:var(--muted);
  cursor:pointer;font-size:13px;padding:1px 7px;line-height:1.4}
.modal-close:hover{color:var(--text);border-color:rgba(100,160,220,.3)}
.modal-body{padding:16px 18px;display:grid;grid-template-columns:1fr 1fr;gap:10px 14px}
.modal-field{display:flex;flex-direction:column;gap:4px}
.modal-field.full{grid-column:1/-1}.modal-lbl{font-size:9px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#5a8aaa}
.modal-in{
  background:rgba(4,8,18,0.7);border:1px solid rgba(30,42,62,0.9);color:var(--text);border-radius:3px;
  padding:6px 9px;font-size:11px;font-family:var(--mono);width:100%;box-sizing:border-box;
}
.modal-in:focus{outline:none;border-color:rgba(42,96,144,.8);box-shadow:0 0 0 2px rgba(42,96,144,.15)}
select.modal-in{cursor:pointer}
.modal-foot{
  display:flex;gap:8px;justify-content:flex-end;
  padding:12px 18px;border-top:1px solid var(--glass-border);
}
.modal-save{
  background:rgba(45,255,110,.1);border:1px solid rgba(45,255,110,.35);
  border-radius:3px;color:var(--green);cursor:pointer;
  font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;padding:6px 18px;
}
.modal-save:hover{background:rgba(45,255,110,.18);box-shadow:0 0 12px rgba(45,255,110,.15)}
.modal-cancel{
  background:rgba(255,255,255,.03);border:1px solid rgba(30,42,62,0.8);border-radius:3px;color:var(--muted);
  cursor:pointer;font-size:10px;letter-spacing:.08em;padding:6px 12px;
}
.modal-cancel:hover{border-color:rgba(42,74,110,.7);color:var(--text)}

/* ── Channel bank modal ────────────────────────────────────── */
.chbank-modal{
  background:rgba(10,22,65,0.88);border:1px solid var(--glass-border);border-radius:8px;
  width:min(860px,98vw);max-height:88vh;display:flex;flex-direction:column;
  backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
  box-shadow:0 8px 48px rgba(0,5,30,.9),inset 0 1px 0 var(--glass-hi);
  transform:translateY(8px);transition:transform .15s;
}
.modal-backdrop.open .chbank-modal{transform:translateY(0)}
.chbank-modal .modal-hdr{flex-shrink:0}
.chbank-modal-body{flex:1;overflow-y:auto;min-height:0}
.chbank-footer{
  flex-shrink:0;padding:8px 12px;border-top:1px solid var(--glass-border);
  display:flex;align-items:center;justify-content:space-between;gap:8px;
}
.chbank-add{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--muted);cursor:pointer;transition:color .15s;letter-spacing:.1em;text-transform:uppercase;padding:3px 0}
.chbank-add:hover{color:var(--cyan)}
.chbank-close{background:rgba(255,255,255,.03);border:1px solid rgba(30,42,62,.8);border-radius:3px;color:var(--muted);cursor:pointer;font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:5px 16px}
.chbank-close:hover{border-color:rgba(42,74,110,.7);color:var(--text)}

/* ── Banks panel ───────────────────────────────────────────── */
.banks-panel{background:rgba(8,18,52,0.6);border-bottom:1px solid var(--glass-border)}
.banks-hdr{
  padding:5px 12px;font-size:8px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;
  color:#5a7a9a;display:flex;align-items:center;gap:8px;
}
.banks-list{display:flex;flex-wrap:wrap;gap:4px;padding:4px 12px 8px}
.bank-btn{
  background:rgba(8,18,32,0.7);border:1px solid rgba(26,42,64,0.8);border-radius:3px;
  color:var(--muted);font-size:9px;letter-spacing:.08em;text-transform:uppercase;
  cursor:pointer;padding:3px 10px;transition:background .15s,border-color .15s,color .15s;
  white-space:nowrap;
}
.bank-btn.enabled{background:rgba(45,255,110,.07);border-color:rgba(45,255,110,.3);color:var(--green)}
.bank-btn:hover{border-color:rgba(42,74,112,.7);color:var(--text)}
.bank-btn.enabled:hover{background:rgba(45,255,110,.12)}

/* ── Activity log ─────────────────────────────────────────── */
.acard{background:rgba(12,22,60,0.52);border:1px solid var(--glass-border);border-radius:8px;overflow:hidden;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);box-shadow:0 6px 32px rgba(0,5,30,.7),inset 0 1px 0 var(--glass-hi)}
.ahdr{
  padding:7px 12px;border-bottom:1px solid rgba(60,40,100,0.4);font-size:8px;font-weight:700;
  color:#8a6acc;letter-spacing:.2em;text-transform:uppercase;background:rgba(16,12,42,0.7);
  cursor:pointer;user-select:none;display:flex;align-items:center;justify-content:space-between;
}
.ahdr:hover{color:var(--purple)}
.arow{display:flex;align-items:center;gap:12px;padding:5px 12px;border-bottom:1px solid var(--panel-b);font-size:11px}
.arow:last-child{border-bottom:none}
.at{font-family:var(--mono);color:var(--cyan);width:58px;flex-shrink:0;opacity:.7}
.as{color:var(--muted);width:90px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:10px;letter-spacing:.04em;text-transform:uppercase}
.af{font-family:var(--mono);font-weight:600;width:82px;flex-shrink:0;color:var(--amber)}
.al{color:var(--text);flex:1;font-size:11px;opacity:.8}

.obtn:hover{background:rgba(45,255,110,.16);box-shadow:0 0 12px rgba(45,255,110,.2)}
.obtn.skip{background:#0a0e18;border:1px solid #1e2a3e;color:var(--muted);font-weight:400;letter-spacing:.06em}

/* ── Animations ───────────────────────────────────────────── */
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.blink{animation:pulse 1.2s ease-in-out infinite}

/* ── Responsive ────────────────────────────────────────────── */

@media (max-width:700px){
  main{padding:10px}
  .grid{grid-template-columns:1fr;gap:10px;margin-bottom:10px}
  .sc-lbl{font-size:20px}
  .sc-btn{padding:8px 4px;font-size:8px}
  .ch{padding:8px 12px}
  header{padding:8px 14px;gap:8px}
  .st{display:none}
  h1{font-size:11px;letter-spacing:.14em}
}

@media (max-width:400px){
  .sc-lbl{font-size:16px}
  .sc-freq{font-size:12px}
  .sc-acts{gap:3px;padding:6px 8px}
  .abtn{padding:5px 10px}
  main{padding:8px}
  .grid{gap:8px}
}
</style>
</head>
<body>
<header>
  <h1>
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4a6a8a" stroke-width="2.5">
      <circle cx="12" cy="12" r="2"/>
      <path d="M16.24 7.76a6 6 0 010 8.49M7.76 16.24a6 6 0 010-8.49M20.49 3.51a12 12 0 010 16.97M3.51 20.49a12 12 0 010-16.97"/>
    </svg>
    SDR Scanner
  </h1>
  <div class="dot" id="wdot"></div>
  <span class="st" id="wst">Connecting…</span>
  <span class="st" style="opacity:.35;font-size:.7em;margin-left:6px">v__VERSION__</span>
  <span id="wscount" style="font-size:.65em;color:var(--muted);margin-left:6px" title="Active connections"></span>
  <span id="wslog" style="font-size:.65em;color:#f84;margin-left:6px" title="Last disconnect — persists until next disconnect"></span>
  <div class="spacer"></div>
  <div class="bcast-wrap" id="bcastWrap" style="display:none" title="Broadcastify feed status">
    <div class="bcast-dot" id="bcastDot"></div>
    <span class="bcast-lbl" id="bcastLbl">BCF</span>
  </div>
  <span class="asrc" id="asrc"></span>
  <span id="audio-lag-display" style="font-size:.65em;color:var(--muted);margin-right:6px;display:none" title="Measured audio buffer depth"></span>
  <button class="abtn" id="abtn" onclick="toggleAudio()">
    <span id="aico">◼</span><span id="albl">Audio Off</span>
  </button>
</header>
<div class="app-layout">
<main>
  <div class="grid" id="grid"></div>
  <div class="acard">
    <div class="ahdr" onclick="toggleActLog()">Activity Log <span id="actArrow">▶</span></div>
    <div id="actlist" style="display:none">
      <div class="arow"><span class="at" style="color:#1d2e1a">—</span><span style="color:#1d2e1a;font-size:11px;letter-spacing:.1em;text-transform:uppercase">No activity recorded</span></div>
    </div>
  </div>
</main>
<div class="acontrols hidden" id="acontrols">
  <div id="ac-body">
  <div class="ac-knob-wrap">
    <div class="ac-knob-lbl">Volume</div>
    <canvas id="aVolKnob" class="ac-knob" width="52" height="52"></canvas>
    <div class="ac-knob-val" id="aVolLbl">100%</div>
  </div>
  <div class="ac-knob-wrap">
    <div class="ac-knob-lbl">LP Cut</div>
    <canvas id="aLPKnob" class="ac-knob" width="52" height="52"></canvas>
    <div class="ac-knob-val" id="aLPLbl">1.5k</div>
  </div>
  <div class="ac-seg-group">
    <div class="ac-seg-lbl">SQ Tail</div>
    <div class="ac-seg">
      <button class="ac-seg-btn" id="aSqTailBtn" onclick="toggleSqTail()">ON</button>
    </div>
  </div>
  <div id="aWakeLockRow" style="display:none">
    <div class="ac-seg-group">
      <div class="ac-seg-lbl">Screen</div>
      <div class="ac-seg">
        <button class="ac-seg-btn" id="aWakeLockBtn" onclick="toggleWakeLock()">ON</button>
      </div>
    </div>
  </div>
  <div id="ac-acts-wrap"><div id="ac-acts"></div></div>
  </div>
</div>
</div>
<!-- Channel bank modal -->
<div class="modal-backdrop" id="chBankModal" onclick="if(event.target===this)closeChBankModal()">
  <div class="chbank-modal">
    <div class="modal-hdr">
      <span id="chBankModalTitle">Channel Bank</span>
      <button class="modal-close" onclick="closeChBankModal()">✕</button>
    </div>
    <div class="chbank-modal-body">
      <div id="chBankModalBanks"></div>
      <div class="chlist" id="chBankModalList"></div>
    </div>
    <div class="chbank-footer">
      <div class="chbank-add" onclick="closeChBankModal();showAddChannel()">＋ Add Frequency</div>
      <button class="chbank-close" onclick="closeChBankModal()">Close</button>
    </div>
  </div>
</div>
<!-- Channel edit/add modal -->
<div class="modal-backdrop" id="chModal" onclick="if(event.target===this)closeChModal()">
  <div class="modal">
    <div class="modal-hdr"><span id="chModalTitle">Edit Channel</span><button class="modal-close" onclick="closeChModal()">✕</button></div>
    <div class="modal-body">
      <div class="modal-field" id="chModalFreqField">
        <label class="modal-lbl">Frequency (MHz)</label>
        <input class="modal-in" id="chModalFreq" type="number" step="0.0001" placeholder="e.g. 446.000">
      </div>
      <div class="modal-field">
        <label class="modal-lbl">Label</label>
        <input class="modal-in" id="chModalLabel" placeholder="Channel name">
      </div>
      <div class="modal-field">
        <label class="modal-lbl">Bank</label>
        <input class="modal-in" id="chModalBank" placeholder="e.g. Police, Fire (optional)">
      </div>
      <div class="modal-field">
        <label class="modal-lbl">Mode</label>
        <select class="modal-in" id="chModalMode">
          <option value="">FM (default)</option>
          <option value="fm">FM — Land Mobile (±5 kHz)</option>
          <option value="nfm">NFM — Narrow FM (±2.5 kHz)</option>
          <option value="am">AM — Amplitude Modulation</option>
        </select>
      </div>
      <div class="modal-field">
        <label class="modal-lbl">Channel Width (kHz)</label>
        <select class="modal-in" id="chModalBW">
          <option value="">Auto</option>
          <option value="6.25">6.25 kHz (Narrowband)</option>
          <option value="8.33">8.33 kHz (Aviation)</option>
          <option value="12.5">12.5 kHz (NFM)</option>
          <option value="25">25 kHz (FM Land Mobile)</option>
          <option value="30">30 kHz (FM)</option>
        </select>
      </div>
      <div class="modal-field">
        <label class="modal-lbl">CTCSS / PL Tone (Hz)</label>
        <input class="modal-in" id="chModalPL" type="number" step="0.1" min="0" placeholder="e.g. 100.0 (blank = off)">
      </div>
      <div class="modal-field">
        <label class="modal-lbl">Sub-audio tone filter (Hz)</label>
        <input class="modal-in" id="chModalHPF" type="number" step="0.1" min="0" max="1000" placeholder="e.g. 179.9 (blank = off)">
      </div>
      <div class="modal-field">
        <label class="modal-lbl">Squelch RMS</label>
        <input class="modal-in" id="chModalSQ" type="number" step="0.001" min="0.001" max="0.5" placeholder="e.g. 0.050">
      </div>
      <div class="modal-field">
        <label class="modal-lbl">RF Gain</label>
        <input class="modal-in" id="chModalGain" placeholder="auto or dB (e.g. 25.4)">
      </div>
    </div>
    <div class="modal-foot">
      <button class="modal-cancel" onclick="closeChModal()">Cancel</button>
      <button class="modal-save" onclick="saveChModal()">Save Channel</button>
    </div>
  </div>
</div>
<script>
// ── State ──────────────────────────────────────────────────────────────────────
const S = { streams:{}, playing:null, audioOn:(localStorage.getItem('a_on')==='true') };
let ws, wsRetry=0;
let actItems = [];
let _editFreq    = null;   // freq string currently open in edit mode, or null
let _addingCh    = false;  // whether the add-channel form is shown
const _chCollapsed = {};   // { [mount]: bool } — per-stream channel bank collapse state
let _actCollapsed  = true;
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Audio ─────────────────────────────────────────────────────────────────────
//
// Three paths depending on the runtime environment:
//
//  1. Native Android app  (window.AndroidNative defined)
//     ScannerService MediaPlayer — audio runs OUTSIDE the WebView entirely.
//     JS calls window.AndroidNative.startAudio(url) / stopAudio() / setVolume(v).
//     The service has its own WiFi lock + foreground service; it survives
//     background, screen-off, Doze, and long silences without any WebView audio.
//
//  2. Desktop browser
//     <audio src=/stream> + Web Audio graph (HP/LP BiquadFilters, volume, gate).
//
//  3. Android browser / TWA  (isAndroid && not native app)
//     MSE + WebSocket + ffmpeg.  Retained as fallback for browser users where
//     Android's background restrictions still apply.

// ── Audio settings (persisted in localStorage) ────────────────────────────────
const A = {
  vol:    Math.min(1, Math.max(0, parseFloat(localStorage.getItem('a_vol') ?? '1') || 1)),
  lp:     parseInt(  localStorage.getItem('a_lp')     ?? '1500', 10),
  sqtail: (localStorage.getItem('a_sqtail') ?? 'false') === 'true',
};

// Detect native Android app via JavascriptInterface injected by MainActivity
const _isNativeApp = typeof window.AndroidNative !== 'undefined';
const _isAndroidBrowser = /Android/i.test(navigator.userAgent) && !_isNativeApp;

let _sqActive        = true;
let _gateRaf         = null;
let _gateGain        = 1.0;
let _gateCloseTimer  = null;   // handle for pending delayed _setGate(false)
let audMount         = null;

// Audio lag tracking via per-connection output sample counter.
//
// The server tracks q.out_samples for each /stream client: every PCM chunk
// sent (real audio AND injected silence) increments it.  audio_stats broadcasts
// this once per second.  The browser's _audEl.currentTime × AUDIO_RATE equals
// samples played since the stream connected.  Both counters reset to 0 at
// connection time, so lag = (out_samples - currentTime × AUDIO_RATE) / AUDIO_RATE.
//
// This is correct even during long idle periods where only silence is injected
// (scanner.audio_seq does NOT advance during silence, so using it collapses lag
// to 0 after idle time and the UI fires 4+ seconds before audio is heard).
const _AUDIO_RATE    = 24000;    // must match server AUDIO_RATE
let _serverAudioSeq  = 0;        // latest audio_seq from audio_stats (display only)
let _lastOutSamples  = 0;        // latest out_samples from audio_stats
let _lagEstMs        = 0;        // current lag estimate in ms

function _tickLag() {
  // Interpolate between audio_stats broadcasts using the last known out_samples.
  // As the browser plays more audio (currentTime increases), lag decreases
  // correctly without waiting for the next audio_stats update.
  if (_mseActive || _isNativeApp || !_audEl || _audEl.paused || !_lastOutSamples) return;
  if (_audEl.currentTime < 0.3) return;
  const raw = Math.max(0, (_lastOutSamples - _audEl.currentTime * _AUDIO_RATE) / _AUDIO_RATE * 1000);
  if (raw < 30000) _lagEstMs = raw;
}
setInterval(_tickLag, 200);

// Returns how many milliseconds the audio lags behind the WebSocket events.
// Desktop WAV: derived from monotonic sequence numbers (accurate).
// MSE/Android: SourceBuffer buffered-range depth (good enough).
// Native app: 0 (no pipeline buffering).
function _audioLagMs() {
  if (_isNativeApp) return 0;
  if (_mseActive && _mseSb && _mseSb.buffered.length && _audEl) {
    return Math.max(0,
      (_mseSb.buffered.end(_mseSb.buffered.length - 1) - _audEl.currentTime) * 1000);
  }
  if (!_mseActive) return _lagEstMs;
  return 0;
}
let _audEl     = null;

// Web Audio API nodes (used by native app + desktop paths)
let _audioCtx  = null;
let _hpNode    = null;   // BiquadFilter highpass
let _lpNode    = null;   // BiquadFilter lowpass
let _volNode   = null;   // GainNode — user volume
let _gateNode  = null;   // GainNode — squelch gate

// MSE / WebSocket path variables (Android browser only)
let _mseAbort  = null;
let _mseActive = false;
let _mseSb     = null;
let _retries   = 0;
let _stallTimer = null;

function _initWebAudioGraph() {
  if (_audioCtx) return;
  try {
    _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = _audioCtx.createMediaElementSource(_audEl);
    _hpNode  = _audioCtx.createBiquadFilter();
    _hpNode.type = 'highpass';
    _hpNode.frequency.value = 10;  // transparent — HPF now done per-channel in scan loop
    _lpNode  = _audioCtx.createBiquadFilter();
    _lpNode.type = 'lowpass';
    _lpNode.frequency.value = A.lp;
    _volNode  = _audioCtx.createGain();
    _volNode.gain.value  = A.vol;
    _gateNode = _audioCtx.createGain();
    _gateNode.gain.value = _gateGain;
    src.connect(_hpNode);
    _hpNode.connect(_lpNode);
    _lpNode.connect(_volNode);
    _volNode.connect(_gateNode);
    _gateNode.connect(_audioCtx.destination);
  } catch (e) {
    // Web Audio not available — fall back to _audEl.volume control
    _audioCtx = null;
  }
}

function _applyVolume() {
  if (_isNativeApp) {
    window.AndroidNative.setVolume(A.vol * _gateGain);
  } else if (_volNode && _gateNode) {
    _volNode.gain.value  = A.vol;
    // Cancel any in-progress automation before setting value directly,
    // otherwise a running setTargetAtTime(0) ramp overrides the assignment.
    _gateNode.gain.cancelScheduledValues(0);
    _gateNode.gain.value = _gateGain;
  } else if (_audEl) {
    _audEl.volume = Math.min(1, Math.max(0, A.vol * _gateGain));
  }
}

function _setGate(open) {
  if (_gateRaf) { cancelAnimationFrame(_gateRaf); _gateRaf = null; }
  // Opening the gate always cancels any pending delayed close from a prior
  // transmission.  Without this, a 5+ second delayed close fires during the
  // next transmission and silences it.
  if (open && _gateCloseTimer) {
    clearTimeout(_gateCloseTimer);
    _gateCloseTimer = null;
    console.log('[audio] cancelled stale gate-close timer');
  }
  const prev = _gateGain;
  _gateGain = open ? 1.0 : 0.0;
  console.log(`[audio] _setGate(${open}) prev=${prev.toFixed(2)} sqActive=${_sqActive} sqtail=${A.sqtail} ctx=${_audioCtx ? _audioCtx.state : 'none'} nodeGain=${_gateNode ? _gateNode.gain.value.toFixed(3) : 'n/a'}`);
  if (_isNativeApp) {
    const from = prev;
    const to   = _gateGain;
    const STEPS = 10, STEP_MS = 4;  // 40 ms total
    for (let i = 1; i <= STEPS; i++) {
      (function(step) {
        setTimeout(function() {
          if (typeof window.AndroidNative !== 'undefined') {
            const g = from + (to - from) * step / STEPS;
            window.AndroidNative.setVolume(A.vol * Math.max(0, Math.min(1, g)));
          }
        }, step * STEP_MS);
      })(i);
    }
    return;
  }
  if (_gateNode && _audioCtx) {
    // cancelScheduledValues(0) cancels ALL events including any in-progress
    // setTargetAtTime ramp that started in the past — cancelScheduledValues(currentTime)
    // only cancels future-scheduled events and leaves past-started ramps running.
    _gateNode.gain.cancelScheduledValues(0);
    _gateNode.gain.value = _gateGain;
  } else {
    _applyVolume();
  }
  _dbgAudioStatus();
}

function _dbgAudioStatus() {
  const ctxState = _audioCtx ? _audioCtx.state : 'none';
  const gateVal  = _gateNode ? _gateNode.gain.value.toFixed(3) : String(_gateGain);
  const elState  = _audEl ? (_audEl.paused ? 'paused' : 'playing') + '/rs' + _audEl.readyState : 'none';
  const lag      = _lagEstMs.toFixed(0);
  // Console only — UI is updated by audio_stats WS events (has server seq+queue)
  console.log(`[audio-dbg] ctx=${ctxState} gate=${gateVal} _gateGain=${_gateGain} _sqActive=${_sqActive} sqtail=${A.sqtail} el=${elState} lagMs=${lag} mse=${_mseActive}`);
}
setInterval(_dbgAudioStatus, 2000);

function _initAudEl() {
  // Native app: audio lives entirely in ScannerService (MediaPlayer).
  // No HTMLAudioElement or Web Audio graph is needed — skip.
  if (_isNativeApp) return;

  if (_audEl) return;
  _audEl = new Audio();
  _audEl.volume = Math.min(1, Math.max(0, A.vol));
  _audEl.addEventListener('playing', () => {
    _retries = 0; clearTimeout(_stallTimer);
    console.log('[audio] playing readyState=' + _audEl.readyState + ' currentTime=' + _audEl.currentTime.toFixed(2) + ' gain=' + (_gateNode ? _gateNode.gain.value.toFixed(3) : _gateGain));
    _dbgAudioStatus();
  });
  _audEl.addEventListener('pause', () => {
    console.log('[audio] paused audioOn=' + S.audioOn);
    if (!S.audioOn) return;
    _audEl.play().catch((e) => console.warn('[audio] re-play failed:', e));
  });
  _audEl.addEventListener('waiting', () => console.log('[audio] waiting (buffering)'));
  _audEl.addEventListener('canplay', () => console.log('[audio] canplay readyState=' + _audEl.readyState));
  _audEl.addEventListener('stalled', () => {
    console.warn('[audio] stalled mseActive=' + _mseActive);
    if (!S.audioOn) return;
    if (_mseActive) return;   // MSE watchdog handles recovery
    clearTimeout(_stallTimer);
    _stallTimer = setTimeout(_reloadStream, 4000);
  });
  _audEl.onerror = () => {
    console.error('[audio] error code=' + (_audEl.error ? _audEl.error.code : '?') + ' msg=' + (_audEl.error ? _audEl.error.message : '?'));
    if (!S.audioOn) return;
    _retries++;
    setTimeout(_reloadStream, Math.min(1000 * _retries, 30000));
  };

  if (_isAndroidBrowser && window.MediaSource && MediaSource.isTypeSupported('audio/mpeg')) {
    // Android browser / TWA: use MSE + WebSocket + ffmpeg to work around
    // browser background restrictions.
    _mseActive = true;
    _mseConnect();
  } else {
    // Desktop: simple WAV stream with Web Audio graph for HP/LP filtering,
    // volume, and squelch gate.
    // Both out_samples (server) and currentTime (browser) reset to 0 at connect,
    // so no offset is needed — reset our tracker so lag starts fresh.
    _lastOutSamples = 0;
    _lagEstMs = 0;
    _audEl.src = '/stream';
    _initWebAudioGraph();
  }
}

async function _mseConnect() {
  // Abort any previous connection (filter change, watchdog reconnect, etc.)
  if (_mseAbort) { _mseAbort.abort(); }
  _mseAbort = new AbortController();
  // Capture the controller for *this* invocation.  If _mseConnect is called
  // again before this one exits (filter change, watchdog), the global
  // _mseAbort is replaced but our local reference stays valid, so the
  // reconnect-or-not check at the end tests the right signal.
  const abort = _mseAbort;
  const ms     = new MediaSource();
  const objUrl = URL.createObjectURL(ms);
  _audEl.src   = objUrl;

  await new Promise(res => ms.addEventListener('sourceopen', res, { once: true }));
  URL.revokeObjectURL(objUrl);

  let sb;
  try {
    sb = ms.addSourceBuffer('audio/mpeg');
    _mseSb = sb;   // expose for buffer-depth reads (e.g. squelch gate delay)
  } catch (e) {
    // MSE not supported for this type — fall back to direct src
    _mseActive  = false;
    _audEl.src = '/stream.mp3';
    return;
  }

  const waitSb = () => sb.updating
    ? new Promise(r => sb.addEventListener('updateend', r, { once: true }))
    : Promise.resolve();

  const TARGET_LAG = 0.9;   // seconds behind live edge to target after a seek
  const MAX_LAG    = 2.5;   // seconds; seek only if genuinely far behind

  // canplay handler: resumes playback and (once) seeks to the live edge.
  //
  // IMPORTANT — not registered with { once: true }.  If canplay fires while
  // the buffer is too shallow to seek, we still call play() but skip the seek.
  // With { once: true } that early fire consumes the handler and play() is
  // never called, leaving the element paused after a reconnect.  Without it,
  // every subsequent canplay (e.g. after a seek-induced waiting→canplay cycle)
  // will also attempt play() — harmless when already playing.
  let _jumped = false;
  const _jumpToLive = () => {
    if (!sb || !sb.buffered.length) return;
    // Always resume if paused — covers reconnect paths where openAudioStream
    // / play() is not called again by the caller.
    if (_audEl.paused) _audEl.play().catch(() => {});
    // Seek to live edge once, but only when the buffer is deep enough to
    // land TARGET_LAG behind live with data still ahead to play.
    if (!_jumped) {
      const liveEdge = sb.buffered.end(sb.buffered.length - 1);
      if (liveEdge >= TARGET_LAG + 0.5) {
        _jumped = true;
        try { _audEl.currentTime = Math.max(0, liveEdge - TARGET_LAG); } catch (_) {}
      }
    }
  };
  _audEl.addEventListener('canplay', _jumpToLive);

  // Ongoing watchdog: seek forward only when genuinely far behind AND Chrome
  // is actively playing (readyState 4 = HAVE_ENOUGH_DATA).  Skipping the seek
  // when readyState < 4 avoids a seek→waiting→lag-grows→seek loop where the
  // watchdog fires repeatedly while Chrome is already buffering at the live edge.
  const _watchdog = setInterval(() => {
    if (!sb || !_audEl || _audEl.paused) return;
    if (_audEl.readyState < 4) return;   // already buffering; don't pile on seeks
    if (!sb.buffered.length) return;
    const liveEdge = sb.buffered.end(sb.buffered.length - 1);
    if (liveEdge - _audEl.currentTime > MAX_LAG) {
      try { _audEl.currentTime = liveEdge - TARGET_LAG; } catch (_) {}
    }
  }, 500);

  // Use WebSocket instead of HTTP fetch for audio delivery.  Android kills
  // long-running HTTP streaming responses after a few minutes of background
  // silence; WebSocket connections survive because Chrome manages them as
  // persistent connections (same as rdio-scanner's approach).
  const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const audioWs = new WebSocket(`${wsProto}//${location.host}/ws/audio?lp=${A.lp}`);
  audioWs.binaryType = 'arraybuffer';

  // Wire abort signal → WebSocket close
  const _onAbort = () => audioWs.close(1000, 'reconnect');
  abort.signal.addEventListener('abort', _onAbort, { once: true });

  await new Promise((resolve, reject) => {
    audioWs.onopen    = resolve;
    audioWs.onerror   = reject;
    audioWs.onclose   = reject;
  }).catch(() => {});   // errors handled below via onclose

  // SourceBuffer operations must be serialised — appendBuffer and remove
  // both throw InvalidStateError if called while an update is already in
  // progress.  WebSocket onmessage fires for every incoming frame, so
  // concurrent async handlers easily race each other.  We queue raw frames
  // and drain one at a time through a single async loop.
  const _sbQueue = [];
  let   _sbDraining = false;

  const _drainSb = async () => {
    if (_sbDraining) return;
    _sbDraining = true;
    while (_sbQueue.length) {
      const data = _sbQueue.shift();
      if (!data || !data.byteLength || ms.readyState !== 'open') continue;
      await waitSb();
      try { sb.appendBuffer(new Uint8Array(data)); } catch (_) { continue; }
      await waitSb();
      // Trim old data behind the playhead to bound memory use
      if (sb.buffered.length && _audEl.currentTime > 0.5) {
        const trimTo = _audEl.currentTime - 0.2;
        if (trimTo > sb.buffered.start(0)) {
          try { sb.remove(sb.buffered.start(0), trimTo); } catch (_) {}
          await waitSb();
        }
      }
    }
    _sbDraining = false;
  };

  audioWs.onmessage = (e) => {
    _sbQueue.push(e.data);
    _drainSb();
  };

  await new Promise(resolve => { audioWs.onclose = resolve; });

  abort.signal.removeEventListener('abort', _onAbort);
  clearInterval(_watchdog);
  _audEl.removeEventListener('canplay', _jumpToLive);
  if (_mseSb === sb) _mseSb = null;

  // Reconnect unless this close was triggered by *our* abort (filter change
  // or explicit close).  Use the locally captured controller so a newer
  // _mseConnect invocation's abort doesn't suppress our reconnect.
  if (!abort.signal.aborted && S.audioOn)
    setTimeout(_mseConnect, Math.min(1000 * (++_retries), 30000));
}

function _reloadStream() {
  if (!_audEl || !S.audioOn) return;
  if (_mseActive) {
    _mseConnect();
  } else {
    _lastOutSamples = 0;
    _lagEstMs = 0;
    _audEl.load();
    _audEl.play().catch(() => updateAudioUI());
  }
}

function openAudioStream(mount) {
  audMount  = mount;
  _sqActive = true;
  _gateGain = 1.0;
  console.log(`[audio] openAudioStream(${mount}) sqtail=${A.sqtail} mseActive=${_mseActive} ctxState=${_audioCtx ? _audioCtx.state : 'none'} paused=${_audEl ? _audEl.paused : 'n/a'}`);
  if (_isNativeApp) {
    // Hand off to ScannerService — AudioTrack handles PCM directly.
    window.AndroidNative.startAudio(location.origin + '/stream?lp=' + A.lp);
    _applyVolume();
    _updateMediaSession(true);
    updateAudioUI();
    return;
  }
  _initAudEl();
  // Always open the gate explicitly here so we don't rely on gate state
  // left over from a previous transmission or a prior _setGate(false) call.
  _setGate(true);
  // Resume AudioContext if suspended (browsers require user gesture before
  // AudioContext can run; this call is inside a user-gesture handler).
  if (_audioCtx && _audioCtx.state === 'suspended') {
    console.log('[audio] resuming suspended AudioContext');
    _audioCtx.resume().then(() => console.log('[audio] AudioContext resumed:', _audioCtx.state));
  }
  if (_audEl.paused) {
    console.log('[audio] _audEl paused — calling play()');
    _audEl.play().catch((e) => {
      console.warn('[audio] play() failed:', e);
    });
  }
  _updateMediaSession(true);
  if (_isAndroidBrowser) _acquireWakeLock();
  updateAudioUI();
  _dbgAudioStatus();
}

function closeAudio() {
  if (_isNativeApp) {
    window.AndroidNative.stopAudio();
  } else if (_audEl) {
    _audEl.pause();
  }
  audMount = null;
  _releaseWakeLock();
  _updateMediaSession(false);
}

function _updateMediaSession(playing) {
  if (!('mediaSession' in navigator)) return;
  navigator.mediaSession.metadata = new MediaMetadata({ title: 'RTL Scanner' });
  navigator.mediaSession.playbackState = playing ? 'playing' : 'paused';
  // Action handlers are required for Android to fully integrate the tab into
  // the media system (lock-screen controls, background process priority).
  navigator.mediaSession.setActionHandler('play',  () => { if (_audEl) _audEl.play(); _updateMediaSession(true); });
  navigator.mediaSession.setActionHandler('pause', () => { if (_audEl) _audEl.pause(); _updateMediaSession(false); });
  navigator.mediaSession.setActionHandler('stop',  () => closeAudio());
}


// ── WebSocket (control) ────────────────────────────────────────────────────────
let _wsPingTimer = null;

function connect() {
  const _wsp = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(_wsp + '//' + location.host + '/ws');
  let _wsConnectedAt = null;

  ws.onopen = () => {
    wsRetry = 0;
    _wsConnectedAt = new Date();
    setWsSt(true);
    // Log reconnects so the user can see the gap duration in the activity log
    const log = document.getElementById('wslog');
    if (log && log.textContent)
      pushActivity('Network', '—', 'Reconnected', _wsConnectedAt.toISOString());
    clearInterval(_wsPingTimer);
    _wsPingTimer = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN)
        ws.send(JSON.stringify({type: 'ping'}));
    }, 15000);
  };
  ws.onclose = (e) => {
    clearInterval(_wsPingTimer);
    _wsPingTimer = null;
    setWsSt(false);
    const now = new Date();
    const ts  = now.toLocaleTimeString();
    const why = e.reason || '';
    // Update the persistent disconnect badge in the header
    const log = document.getElementById('wslog');
    if (log) log.textContent = `✕${e.code}${why ? ' '+why : ''} @${ts}`;
    // Add an entry to the activity log so the history survives reconnect
    pushActivity('Network', `✕${e.code}`, 'Disconnected', now.toISOString());
    console.warn('[ws] closed code:', e.code, 'clean:', e.wasClean, 'reason:', why);
    wsRetry++;
    const delay = (e.code === 1006 && wsRetry <= 3) ? 0
                : Math.min(1000 * wsRetry, 15000);
    setTimeout(connect, delay);
  };
  ws.onmessage = e => onMsg(JSON.parse(e.data));
}
function setWsSt(ok) {
  document.getElementById('wdot').className = 'dot ' + (ok ? 'ok' : 'err');
  document.getElementById('wst').textContent = ok ? 'Connected' : 'Reconnecting…';
  // wslog intentionally NOT cleared on reconnect — leave last disconnect code
  // visible so the user can read the code after the connection recovers.
}

// ── Message handler ────────────────────────────────────────────────────────────
function onMsg(m) {
  if (m.type === 'state') {

    m.streams.forEach(s => {
      s.channelSquelch = s.channelSquelch || {};
      s.channelGain    = s.channelGain    || {};
      s.channelPL      = s.channelPL      || {};
      s.channelBank       = s.channelBank       || {};
      s.channelModulation = s.channelModulation || {};
      s.channelBandwidth  = s.channelBandwidth  || {};
      s.channelHpFilter   = s.channelHpFilter   || {};
      s.banks             = s.banks             || {};
      s.skipped        = s.skipped        || [];
      s.holdFreq       = s.holdFreq       || null;
      s.squelchHoldMs  = (s.squelchHold || 2.0) * 1000;
      S.streams[s.mount] = s;
      (s.history || []).forEach(h => pushActivity(s.name, h.freq, h.label, h.time));
    });
    renderAll();
    autoSelect();
    if (m.broadcastify) {
      _updateBcastUI(m.broadcastify.connected, m.broadcastify.error, m.broadcastify.enabled);
    }
    // Resume audio on reconnect / page load.
    // For the native app S.audioOn is always true, so this also handles the
    // initial auto-start without any user interaction required.
    if (S.audioOn && !audMount) {
      const target = S.playing || (Object.values(S.streams).find(s=>s.connected)||{}).mount;
      if (target) {
        // Resume AudioContext before play() — required when AudioContext was
        // created before a user gesture (native app autoplay path).
        if (_audioCtx && _audioCtx.state === 'suspended') _audioCtx.resume().catch(()=>{});
        switchAudio(target);
      }
      updateAudioUI();
    }
  } else if (m.type === 'audio_stats') {
    if (m.audio_seq !== undefined) _serverAudioSeq = m.audio_seq;
    if (m.out_samples !== undefined) _lastOutSamples = m.out_samples;
    // Desktop lag: out_samples and currentTime both reset to 0 at stream connect,
    // so no offset arithmetic needed.  out_samples includes injected silence, so
    // lag stays accurate during idle periods (unlike audio_seq which only counts
    // real signal audio and collapses lag to 0 after quiet periods).
    if (!_mseActive && !_isNativeApp && _lastOutSamples && _audEl && _audEl.currentTime > 0.3) {
      const raw = Math.max(0, (_lastOutSamples - _audEl.currentTime * _AUDIO_RATE) / _AUDIO_RATE * 1000);
      if (raw < 30000) _lagEstMs = raw;
    }
    // Header lag badge
    const lagDisplay = document.getElementById('audio-lag-display');
    if (lagDisplay && S.audioOn && !_isNativeApp) {
      const ms = Math.round(_lagEstMs);
      lagDisplay.style.display = ms > 50 ? '' : 'none';
      lagDisplay.textContent = ms + ' ms delay';
    }
    const dbgEl = document.getElementById('audio-dbg-line');
    if (dbgEl && dbgEl.style.display !== 'none') {
      const lagMs   = _lagEstMs.toFixed(0);
      const gateVal = _gateNode ? _gateNode.gain.value.toFixed(2) : String(_gateGain);
      const elState = _audEl ? (_audEl.paused ? 'paused' : 'playing') + '/rs' + _audEl.readyState : 'none';
      const ctx     = _audioCtx ? _audioCtx.state : 'none';
      dbgEl.textContent = `seq:${m.audio_seq} out:${m.out_samples} q:${m.queue_depth} lag:${lagMs}ms ctx:${ctx} gate:${gateVal} sq:${_sqActive?1:0} ${elState}`;
    }
  } else if (m.type === 'freq_change') {
    const s = S.streams[m.mount]; if (!s) return;
    // pendingFreq is set immediately — used for stale-signal-handler detection
    // and per-channel squelch threshold lookups.  activeFreq / activeSince are
    // set inside _doFreqChange (delayed by lag) so the card only shows the
    // channel name/frequency when the audio actually arrives at the speaker.
    s.pendingFreq   = m.freq;
    s.lastError     = null;
    s.detectedCTCSS = null;
    // Snapshot lag now so all UI events in this transmission use the same delay.
    const _fcLag = _audioLagMs();
    s.txLagMs = _fcLag;
    console.log(`[ws] freq_change ${m.freq} mount=${m.mount} audMount=${audMount} lag=${_fcLag}ms sqtail=${A.sqtail}`);
    const _doFreqChange = () => {
      s.activeFreq  = m.freq;
      s.activeSince = m.time;
      updateCard(m.mount, true); pushActivity(m.name, m.freq, m.label, m.time);
      if (!_isNativeApp && A.sqtail && m.mount === (audMount || 'sdr')) {
        _sqActive = true;
        console.log('[ws] freq_change gate re-armed via _doFreqChange');
      }
    };
    if (_fcLag > 100) setTimeout(_doFreqChange, _fcLag); else _doFreqChange();
    if (S.audioOn) switchAudio(m.mount);
  } else if (m.type === 'freq_clear') {
    const s = S.streams[m.mount]; if (!s) return;
    s.pendingFreq   = null;
    s.activeFreq    = null;
    s.activeSince   = null;
    s.detectedCTCSS = null;
    s.txLagMs       = undefined;
    // freq_clear arrives squelch_hold seconds after signal dropped.  Audio has
    // already been silent for squelch_hold ms (server held it open that long).
    // The correct display delay is max(0, lag - squelch_hold) so the card clears
    // when the silence actually reaches the speaker, not squelch_hold seconds later.
    const _fclLag = Math.max(0, _audioLagMs() - (s.squelchHoldMs || 2000));
    if (_fclLag > 100) setTimeout(() => updateCard(m.mount, true), _fclLag);
    else updateCard(m.mount, true);
  } else if (m.type === 'signal') {
    const s2 = S.streams[m.mount];
    const _sigLag = (s2 && s2.txLagMs != null) ? s2.txLagMs : _audioLagMs();
    // Use pendingFreq (set immediately on freq_change) for stale detection —
    // activeFreq is delayed and may still be null or the previous frequency.
    const _sigFreq = s2 ? s2.pendingFreq : null;
    const _doSignalUI = () => {
      const st = S.streams[m.mount];
      // Drop update if a newer freq_change has superseded this transmission.
      if (st && st.pendingFreq !== _sigFreq) return;
      const thr = st ? (st.pendingFreq && st.channelSquelch && st.channelSquelch[st.pendingFreq]
        ? st.channelSquelch[st.pendingFreq] : (st.defaultSquelch || 0.05)) : 0.05;
      const thrDb = 20 * Math.log10(Math.max(thr, 1e-9));
      _updateMeter(m.mount, m.db, thrDb, m.active);
      // Update detected CTCSS tone display in-place (avoids full card re-render)
      const s = S.streams[m.mount];
      if (s && m.ctcss !== undefined) {
        const prev = s.detectedCTCSS;
        s.detectedCTCSS = m.ctcss || null;
        if (s.detectedCTCSS !== prev) {
          const el = document.getElementById('sc_ctcss_' + eid(m.mount));
          if (el) {
            const cpl = s.channelPL || {};
            const configured = s.activeFreq && cpl[s.activeFreq];
            el.textContent  = s.detectedCTCSS ? ('◈ ' + s.detectedCTCSS + ' Hz') : '';
            el.className    = 'sc-ctcss' + (s.detectedCTCSS
              ? (configured ? ' match' : ' info') : '');
          }
        }
      }
    };
    if (_sigLag > 100) setTimeout(_doSignalUI, _sigLag); else _doSignalUI();
    // Squelch tail suppression gate close — all paths.
    // Fire when signal goes inactive, delayed by the audio buffer depth so the
    // gate closes exactly when the listener hears the carrier drop.
    // On desktop lagMs ≈ 0 so the gate fires immediately at signal-drop time
    // (no tail).  On MSE/Android lagMs ≈ 2 s so it fires 2 s later (synced).
    // Native app also fires immediately (no buffer lag).
    // Gate open is handled by the freq_change callback for browser paths and
    // by the active→inactive→active transition below for the native app.
    if (A.sqtail && m.mount === (audMount || 'sdr') && !m.active && _sqActive) {
      _sqActive = false;
      const lagMs = _audioLagMs();
      console.log(`[ws] signal inactive → closing gate in ${lagMs}ms`);
      if (_gateCloseTimer) { clearTimeout(_gateCloseTimer); _gateCloseTimer = null; }
      if (lagMs > 0) {
        _gateCloseTimer = setTimeout(() => { _gateCloseTimer = null; _setGate(false); }, lagMs);
      } else {
        _setGate(false);
      }
    } else if (A.sqtail && m.mount === (audMount || 'sdr') && !m.active && !_sqActive) {
      console.log('[ws] signal inactive but _sqActive=false — gate already closed, skipping');
    }
    // Gate open on signal{active:true} — for ALL paths, not just native.
    // freq_change opens the gate when the scanner moves to a new frequency.
    // But if signal returns on the SAME frequency (hold mode, or fast retune),
    // no freq_change fires and the gate stays closed indefinitely.
    // Delay by lag so the gate opens when the audio actually reaches the speaker.
    if (A.sqtail && m.mount === (audMount || 'sdr') && m.active && !_sqActive) {
      _sqActive = true;
      const openLag = _isNativeApp ? 0 : _audioLagMs();
      console.log(`[ws] signal active → opening gate in ${openLag}ms (same-freq resume or native)`);
      if (openLag > 100) setTimeout(() => _setGate(true), openLag);
      else _setGate(true);
    }
  } else if (m.type === 'channels_update') {
    const s = S.streams[m.mount]; if (!s) return;
    s.channels       = m.channels;
    s.channelSquelch = m.channelSquelch;
    s.channelGain    = m.channelGain    || {};
    s.channelPL      = m.channelPL      || {};
    s.channelBank       = m.channelBank       || {};
    s.channelModulation = m.channelModulation || {};
    s.channelBandwidth  = m.channelBandwidth  || {};
    s.channelHpFilter   = m.channelHpFilter   || {};
    s.banks             = m.banks             || {};
    s.defaultSquelch = m.defaultSquelch;
    s.defaultGain    = m.defaultGain;
    s.skipped        = m.skipped || [];
    updateCard(m.mount);
  } else if (m.type === 'hold_update') {
    const s = S.streams[m.mount]; if (!s) return;
    s.holdFreq = m.holdFreq;
    updateCard(m.mount);
  } else if (m.type === 'conn') {
    const s = S.streams[m.mount]; if (!s) return;
    s.connected = m.connected;
    s.lastError  = m.error || null;
    updateCard(m.mount);
  } else if (m.type === 'ws_clients') {
    const el = document.getElementById('wscount');
    if (el) el.textContent = m.count > 0 ? `·${m.count}` : '';
  } else if (m.type === 'bcast_status') {
    _updateBcastUI(m.connected, m.error, true);
  }
}

function _updateBcastUI(connected, error, enabled) {
  const wrap = document.getElementById('bcastWrap');
  const dot  = document.getElementById('bcastDot');
  const lbl  = document.getElementById('bcastLbl');
  if (!wrap) return;
  if (!enabled) { wrap.style.display = 'none'; return; }
  wrap.style.display = 'flex';
  if (error) {
    dot.className = 'bcast-dot err';
    lbl.className = 'bcast-lbl err';
    lbl.title     = 'BCF ERROR: ' + error;
  } else if (connected) {
    dot.className = 'bcast-dot live';
    lbl.className = 'bcast-lbl live';
    lbl.title     = 'Broadcastify LIVE';
  } else {
    dot.className = 'bcast-dot';
    lbl.className = 'bcast-lbl';
    lbl.title     = 'Broadcastify connecting…';
  }
}

// ── Signal meter helpers ───────────────────────────────────────────────────────
const _SEG_TOTAL = 12;
// Segment zones: 0-3 green, 4-7 green, 8-10 yellow, 11 red
function _segClass(i) {
  if (i >= 11) return 'lit-r';
  if (i >= 8)  return 'lit-y';
  return 'lit-g';
}
function _segHtml(litCount) {
  let h = '';
  for (let i = 0; i < _SEG_TOTAL; i++) {
    h += '<div class="sc-seg' + (i < litCount ? ' ' + _segClass(i) : '') + '"></div>';
  }
  return h;
}
function _updateMeter(mount, db, thrDb, active) {
  const meter = document.getElementById('sqsegs_' + eid(mount));
  const lv    = document.getElementById('sqlevel_' + eid(mount));
  const wrap  = meter && meter.closest('.sc-meter');
  if (wrap) wrap.style.visibility = active ? '' : 'hidden';
  if (meter && active) {
    const pct      = Math.min(1, Math.max(0, (db - thrDb) / (-thrDb)));
    const litCount = Math.round(pct * _SEG_TOTAL);
    meter.innerHTML = _segHtml(litCount);
  }
  if (lv) lv.textContent = active ? db.toFixed(1) + ' dB' : '';
}

// ── Render ─────────────────────────────────────────────────────────────────────
function _actsHtml(s) {
  const af = s.activeFreq;
  const heldF = s.holdFreq || null;
  const isHeld = !!heldF;
  const skpSet = new Set(s.skipped || []);
  const afSkp = af && skpSet.has(af);
  const noAf = !af;
  const holdTarget = af || heldF;
  return '<div class="sc-acts' + (noAf && !isHeld ? ' idle' : '') + '">'
    + '<button class="sc-btn skip' + (afSkp?' active':'') + '" onclick="event.stopPropagation();' + (af?'skipChannel(\''+af+'\')':'') + '" title="' + (afSkp?'Resume scan':'Skip channel') + '">'
    + (afSkp ? '▶ SCAN' : '⊘ SKIP') + '</button>'
    + '<button class="sc-btn hold' + (isHeld?' active':'') + '" onclick="event.stopPropagation();' + (holdTarget?'holdChannel(\''+holdTarget+'\')':'') + '" title="' + (isHeld?'Release hold — resume scanning':'Hold on current frequency') + '">'
    + (isHeld ? '⏹ HELD' : '⏸ HOLD') + '</button>'
    + '<button class="sc-btn resume" onclick="event.stopPropagation();resumeScan()" title="Skip to next frequency now"' + (af && !isHeld ? '' : ' style="visibility:hidden"') + '>▶▶ NEXT</button>'
    + '<button class="sc-btn edit" onclick="event.stopPropagation();' + (af?'editChannel(\''+af+'\')':'') + '" title="Edit label/squelch">✏ EDIT</button>'
    + '<button class="sc-btn del" onclick="event.stopPropagation();' + (af?'deleteChannel(\''+af+'\')':'') + '" title="Remove channel">✕ DEL</button>'
    + '</div>';
}
function _updateAcActs(mount) {
  const el = document.getElementById('ac-acts');
  if (!el) return;
  const s = S.streams[mount];
  if (s) el.innerHTML = _actsHtml(s);
}
function _placeAudioControls() {
  const slot = document.querySelector('.ac-inline-slot');
  const ac = document.getElementById('acontrols');
  if (slot && ac) slot.appendChild(ac);
}
function _rescueAudioControls() {
  const ac = document.getElementById('acontrols');
  if (ac && ac.parentElement && ac.parentElement !== document.body) document.body.appendChild(ac);
}
function renderAll() {
  _rescueAudioControls();
  const g = document.getElementById('grid'); g.innerHTML = '';
  Object.values(S.streams).forEach(s => {
    const d = document.createElement('div');
    d.className = cardClass(s);
    d.id = 'sc' + eid(s.mount);
    d.innerHTML = cardHtml(s);
    g.appendChild(d);
    _placeAudioControls();
    _updateAcActs(s.mount);
  });
}
function updateCard(mount, skipIfEditing) {
  const d = document.getElementById('sc' + eid(mount)); if (!d) return;
  // Don't blow away an active edit row on routine scan events (freq changes, etc.)
  if (skipIfEditing && (_editFreq !== null || _addingCh)) return;
  const s = S.streams[mount];
  d.className = cardClass(s);
  _rescueAudioControls();
  d.innerHTML = cardHtml(s);
  _placeAudioControls();
  _updateAcActs(mount);
  refreshChBankModal();
}
function cardClass(s) {
  let c = 'scard';
  if (audMount === s.mount && S.audioOn) c += ' playing';
  return c;
}
function cardHtml(s) {
  const chs    = s.channels || {};
  const csq    = s.channelSquelch || {};
  const cgain  = s.channelGain        || {};
  const cpl    = s.channelPL          || {};
  const cbank  = s.channelBank        || {};
  const cmod   = s.channelModulation  || {};
  const cbw    = s.channelBandwidth   || {};
  const banks  = s.banks              || {};
  const skpSet = new Set(s.skipped || []);
  const defSq  = (s.defaultSquelch || 0.05).toFixed(3);
  const defGn  = s.defaultGain || 'auto';
  const freqs  = Object.keys(chs).sort((a,b) => parseFloat(a)-parseFloat(b));
  const af      = s.activeFreq;
  const heldF   = s.holdFreq || null;
  const isHeld  = !!heldF;
  const rawLbl  = af ? (chs[af] && chs[af] !== af ? chs[af] : null) : null;
  const holdLbl = heldF ? (chs[heldF] && chs[heldF] !== heldF ? chs[heldF] : null) : null;
  const primary = rawLbl || af || holdLbl || heldF || 'SCANNING';
  const since   = af && s.activeSince ? new Date(s.activeSince).toLocaleTimeString() : '';
  const isRx    = !!af;

  // Panel header: stream name + status LED
  const holdBadge = isHeld
    ? ' <span style="font-size:9px;color:var(--amber);letter-spacing:.1em">⏸ HOLD</span>' : '';
  const chCount = freqs.length;
  const connStatus = s.connected
    ? '<span class="sc-status ok' + (!isRx && !isHeld ? ' scanning' : '') + '"><span class="sc-led"></span>' + (isHeld ? 'HOLD' : 'SCANNING') + '</span>'
    : '<span class="sc-status ' + (s.lastError ? 'err' : 'warn') + '"><span class="sc-led"></span>' + (s.lastError ? 'ERROR' : 'OPENING') + '</span>';
  const errHtml = s.lastError
    ? '<div class="serr">⚠ ' + escHtml(s.lastError) + '</div>' : '';

  // Channel bank rows
  let rows = '';
  if (freqs.length) {
    freqs.forEach(f => {
      const lbl  = chs[f] || '';
      const act  = f === af;
      const skp  = skpSet.has(f);
      const sq   = f in csq ? csq[f].toFixed(3) : defSq;
      const gn   = f in cgain ? cgain[f] : defGn;
      const pl   = f in cpl  ? cpl[f]   : 0;
      const bk   = cbank[f] || '';
      const md   = cmod[f]  || '';
      const t    = act && s.activeSince ? new Date(s.activeSince).toLocaleTimeString() : '';
      const isHeld = f === heldF;
      rows += '<div class="ch' + (act?' active':'') + (isHeld&&!act?' held':'') + (skp?' skipped':'') + '" onclick="event.stopPropagation();holdChannel(\'' + f + '\')" title="' + (isHeld?'Release hold':'Tune to ' + f + ' MHz') + '">'
        + '<span class="ch-dot">' + (skp ? '─' : act ? '◉' : '○') + '</span>'
        + '<span class="ch-f">' + f + '</span>'
        + '<span class="ch-l">' + escHtml(lbl!==f?lbl:'') + '</span>'
        + (bk ? '<span class="ch-bank-badge">' + escHtml(bk) + '</span>' : '')
        + (md ? '<span class="ch-mode-badge">' + escHtml(md.toUpperCase()) + '</span>' : '')
        + (pl ? '<span class="ch-pl">PL ' + pl + '</span>' : '')
        + '<span class="ch-sq">SQ ' + sq + '</span>'
        + '<span class="ch-gn">G ' + (gn || 'auto') + '</span>'
        + '<span class="ch-t">' + t + '</span>'
        + '<div class="ch-acts">'
        + '<button class="ch-btn skip" onclick="event.stopPropagation();skipChannel(\'' + f + '\')">' + (skp?'SCAN':'SKIP') + '</button>'
        + '<button class="ch-btn" onclick="event.stopPropagation();editChannel(\'' + f + '\')">EDIT</button>'
        + '<button class="ch-btn del" onclick="event.stopPropagation();deleteChannel(\'' + f + '\')">DEL</button>'
        + '</div></div>';
    });
  } else if (af) {
    rows = '<div class="ch active">'
      + '<span class="ch-dot">◉</span>'
      + '<span class="ch-f">' + af + '</span>'
      + '<span class="ch-l" style="letter-spacing:.08em;text-transform:uppercase">Detected</span>'
      + '<span class="ch-t">' + since + '</span>'
      + '</div>';
  } else {
    rows = '<div class="noch">Scanning…</div>';
  }

  const addArea = '<div class="ch-add-btn" onclick="event.stopPropagation();showAddChannel()">＋ Add Frequency</div>';

  // Show freq below label when active (rawLbl case), or when holding silently (holdLbl case)
  const metaFreq    = (af && rawLbl) ? af : (!af && heldF && holdLbl) ? heldF : null;
  const activePL    = af && cpl[af] ? cpl[af] : (heldF && cpl[heldF] ? cpl[heldF] : null);
  const initCTCSS   = s.detectedCTCSS || null;
  const ctcssMatch  = initCTCSS && activePL;
  // sc-freq is always rendered (hidden when inactive) so the meta row height never changes
  const metaHtml = '<div class="sc-meta">'
      + '<span class="sc-freq"' + (metaFreq ? '' : ' style="visibility:hidden"') + '>'
      + (metaFreq || '–') + '<span class="sc-unit">MHz</span></span>'
      + (activePL ? '<span class="sc-pl">PL ' + activePL + ' Hz</span>' : '')
      + '<span class="sc-ctcss' + (initCTCSS ? (ctcssMatch ? ' match' : ' info') : '') + '" id="sc_ctcss_' + eid(s.mount) + '">'
      + (initCTCSS ? '◈ ' + initCTCSS + ' Hz' : '') + '</span>'
      + (since ? '<span class="sc-timer">' + since + '</span>' : '')
      + '</div>';

  // Default to open when the channel list is short enough to fit without scrolling.
  // User can toggle; once toggled the explicit state is remembered in _chCollapsed.
  const _defaultCollapsed = Object.keys(s.channels || {}).length > 8;
  const collapsed = _chCollapsed[s.mount] !== undefined ? _chCollapsed[s.mount] : _defaultCollapsed;
  const chBankHtml = '<div class="sc-chl-hdr" onclick="event.stopPropagation();toggleChBank(' + escHtml(JSON.stringify(s.mount)) + ')">'
    + 'Channel Bank<span class="coll-arrow">' + (collapsed ? '▶' : '▼') + '</span></div>'
    + (collapsed ? '' : rows + addArea);

  // Banks panel (only render when >1 bank or when the single bank is not Default)
  const bankNames = Object.keys(banks);
  const showBanks = bankNames.length > 1 || (bankNames.length === 1 && bankNames[0] !== 'Default');
  const bankBtns = bankNames.sort().map(b => {
    const on = banks[b] !== false;
    return '<button class="bank-btn' + (on ? ' enabled' : '') + '" onclick="event.stopPropagation();toggleBank(' + escHtml(JSON.stringify(s.mount)) + ',' + escHtml(JSON.stringify(b)) + ',' + (on?'false':'true') + ')">' + escHtml(b) + '</button>';
  }).join('');
  const banksPanelHtml = showBanks
    ? '<div class="banks-panel"><div class="banks-hdr">SCAN BANKS</div><div class="banks-list">' + bankBtns + '</div></div>'
    : '';

  return '<div class="sc-sticky">'
    + '<div class="sc-panel-hdr"><span class="sc-name">' + escHtml(s.name) + holdBadge + '</span>'
    + '<button class="sc-chl-btn" onclick="event.stopPropagation();openChBankModal(' + escHtml(JSON.stringify(s.mount)) + ')">☰ Channels' + (chCount ? ' (' + chCount + ')' : '') + '</button>'
    + connStatus + '</div>'
    + errHtml
    + '<div class="sc-display' + (isRx ? ' active' : '') + '">'
    + '<div class="sc-lbl-row">'
    + '<div class="sc-lbl">' + escHtml(primary) + '</div>'
    + '<div class="sc-meter" style="' + (isRx ? '' : 'visibility:hidden') + '"><div class="sc-segs" id="sqsegs_' + eid(s.mount) + '">' + _segHtml(0) + '</div><span class="sc-db" id="sqlevel_' + eid(s.mount) + '"></span></div>'
    + '</div>'
    + metaHtml
    + '</div>'
    + '<div class="ac-inline-slot"></div>';
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

function toggleAudio() {
  if (S.audioOn) {
    S.audioOn = false; localStorage.setItem('a_on','false'); closeAudio(); updateAudioUI(); Object.keys(S.streams).forEach(m=>updateCard(m));
  } else {
    _startAudio();
  }
}
function _startAudio() {
  S.audioOn = true; localStorage.setItem('a_on','true');
  _gateGain = 1.0;
  if (_isNativeApp) {
    audMount = audMount
      || S.playing
      || (Object.values(S.streams).find(s => s.connected) || {}).mount;
    if (audMount) openAudioStream(audMount);
    _applyVolume();
  } else {
    _initAudEl();
    _applyVolume();
    if (_audEl.error) _audEl.load();
    _audEl.play().catch(() => {});
  }
  if (!_isNativeApp) {
    audMount = audMount
      || S.playing
      || (Object.values(S.streams).find(s => s.connected) || {}).mount
      || 'sdr';
  }
  _updateMediaSession(true);
  updateAudioUI();
  Object.keys(S.streams).forEach(m => updateCard(m));
}
function updateAudioUI() {
  const ac = document.getElementById('acontrols');
  ac.classList.toggle('hidden', !S.audioOn);
  _placeAudioControls();
  const btn = document.getElementById('abtn');
  const src = document.getElementById('asrc');
  // Native: MediaPlayer is always "connected" once audMount is set
  const connected = _isNativeApp ? !!audMount : (_audEl && !_audEl.paused);
  if (S.audioOn) {
    const s = audMount ? S.streams[audMount] : null;
    btn.className = 'abtn on';
    document.getElementById('aico').textContent = (audMount && connected) ? '▶' : '…';
    document.getElementById('albl').textContent  = 'Audio Follow';
    src.textContent = s ? s.name.toUpperCase() : '';
  } else {
    btn.className = 'abtn';
    document.getElementById('aico').textContent = '◼';
    document.getElementById('albl').textContent  = 'Audio Off';
    src.textContent = '';
  }
}

// ── Collapse toggles ───────────────────────────────────────────────────────────
function toggleChBank(mount) {
  const defaultCollapsed = Object.keys(((S.streams[mount] || {}).channels) || {}).length > 8;
  const current = _chCollapsed[mount] !== undefined ? _chCollapsed[mount] : defaultCollapsed;
  _chCollapsed[mount] = !current;
  updateCard(mount);
}
function toggleActLog() {
  _actCollapsed = !_actCollapsed;
  document.getElementById('actlist').style.display = _actCollapsed ? 'none' : '';
  const arrow = document.getElementById('actArrow');
  if (arrow) arrow.textContent = _actCollapsed ? '▶' : '▼';
}

// ── Channel bank modal ─────────────────────────────────────────────────────────
let _chBankMount = null;

function openChBankModal(mount) {
  _chBankMount = mount;
  const s = S.streams[mount] || {};
  document.getElementById('chBankModalTitle').textContent = (s.name || mount) + ' — Channel Bank';
  _renderChBankModal();
  document.getElementById('chBankModal').classList.add('open');
}

function closeChBankModal() {
  _chBankMount = null;
  document.getElementById('chBankModal').classList.remove('open');
}

function refreshChBankModal() {
  if (_chBankMount) _renderChBankModal();
}

function _renderChBankModal() {
  const mount = _chBankMount;
  const s = S.streams[mount];
  if (!s) return;
  const chs   = s.channels || {};
  const csq   = s.channelSquelch || {};
  const cgain = s.channelGain || {};
  const cpl   = s.channelPL || {};
  const cbank = s.channelBank || {};
  const cmod  = s.channelModulation || {};
  const skpSet = new Set(s.skipped || []);
  const banks  = s.banks || {};
  const af     = s.activeFreq;
  const heldF  = s.holdFreq || null;
  const defSq  = (s.defaultSquelch || 0.05).toFixed(3);
  const defGn  = s.defaultGain || 'auto';
  const freqs  = Object.keys(chs).sort((a,b) => parseFloat(a)-parseFloat(b));

  // Banks panel
  const bankNames = Object.keys(banks);
  const showBanks = bankNames.length > 1 || (bankNames.length === 1 && bankNames[0] !== 'Default');
  let banksHtml = '';
  if (showBanks) {
    const btns = bankNames.sort().map(b => {
      const on = banks[b] !== false;
      return '<button class="bank-btn' + (on ? ' enabled' : '') + '" onclick="toggleBank(' + escHtml(JSON.stringify(mount)) + ',' + escHtml(JSON.stringify(b)) + ',' + (on?'false':'true') + ')">' + escHtml(b) + '</button>';
    }).join('');
    banksHtml = '<div class="banks-panel"><div class="banks-hdr">SCAN BANKS</div><div class="banks-list">' + btns + '</div></div>';
  }
  document.getElementById('chBankModalBanks').innerHTML = banksHtml;

  // Channel rows
  let rows = '';
  if (freqs.length) {
    freqs.forEach(f => {
      const lbl    = chs[f] || '';
      const act    = f === af;
      const held   = f === heldF;
      const skp    = skpSet.has(f);
      const sq     = f in csq ? csq[f].toFixed(3) : defSq;
      const gn     = f in cgain ? cgain[f] : defGn;
      const pl     = f in cpl ? cpl[f] : 0;
      const bk     = cbank[f] || '';
      const md     = cmod[f] || '';
      const t      = act && s.activeSince ? new Date(s.activeSince).toLocaleTimeString() : '';
      rows += '<div class="ch' + (act?' active':'') + (held&&!act?' held':'') + (skp?' skipped':'') + '" onclick="holdChannel(\'' + f + '\')" title="' + (held?'Release hold':'Tune to ' + f + ' MHz') + '">'
        + '<span class="ch-dot">' + (skp ? '─' : act ? '◉' : '○') + '</span>'
        + '<span class="ch-f">' + f + '</span>'
        + '<span class="ch-l">' + escHtml(lbl!==f?lbl:'') + '</span>'
        + (bk ? '<span class="ch-bank-badge">' + escHtml(bk) + '</span>' : '')
        + (md ? '<span class="ch-mode-badge">' + escHtml(md.toUpperCase()) + '</span>' : '')
        + (pl ? '<span class="ch-pl">PL ' + pl + '</span>' : '')
        + '<span class="ch-sq">SQ ' + sq + '</span>'
        + '<span class="ch-gn">G ' + (gn || 'auto') + '</span>'
        + '<span class="ch-t">' + t + '</span>'
        + '<div class="ch-acts">'
        + '<button class="ch-btn skip" onclick="event.stopPropagation();skipChannel(\'' + f + '\')">' + (skp?'SCAN':'SKIP') + '</button>'
        + '<button class="ch-btn" onclick="event.stopPropagation();editChannel(\'' + f + '\')">EDIT</button>'
        + '<button class="ch-btn del" onclick="event.stopPropagation();deleteChannel(\'' + f + '\')">DEL</button>'
        + '</div></div>';
    });
  } else {
    rows = '<div class="noch">No channels configured</div>';
  }
  document.getElementById('chBankModalList').innerHTML = rows;
}

// ── Channel management ─────────────────────────────────────────────────────────
let _chModalFreq = null;  // null = add mode, string = edit mode

function openChModal(freq) {
  // edit mode
  const mount = Object.keys(S.streams)[0];
  const s = mount ? S.streams[mount] : null;
  const chs   = s ? (s.channels || {}) : {};
  const csq   = s ? (s.channelSquelch || {}) : {};
  const cgain = s ? (s.channelGain || {}) : {};
  const cpl   = s ? (s.channelPL || {}) : {};
  const cbank = s ? (s.channelBank || {}) : {};
  const cmod  = s ? (s.channelModulation || {}) : {};
  const cbw   = s ? (s.channelBandwidth || {}) : {};
  const chpf  = s ? (s.channelHpFilter  || {}) : {};
  const defSq = s ? (s.defaultSquelch || 0.05) : 0.05;
  const defGn = s ? (s.defaultGain || 'auto') : 'auto';

  _chModalFreq = freq;
  document.getElementById('chModalTitle').textContent = 'Edit Channel';
  document.getElementById('chModalFreqField').style.display = 'none';
  document.getElementById('chModalFreq').value = freq;
  document.getElementById('chModalLabel').value = (chs[freq] && chs[freq] !== freq ? chs[freq] : '');
  document.getElementById('chModalBank').value  = cbank[freq] || '';
  document.getElementById('chModalMode').value  = cmod[freq]  || '';
  document.getElementById('chModalBW').value    = cbw[freq]   || '';
  document.getElementById('chModalPL').value    = cpl[freq]   || '';
  document.getElementById('chModalHPF').value    = chpf[freq] || '';
  document.getElementById('chModalSQ').value    = freq in csq ? csq[freq].toFixed(3) : defSq.toFixed(3);
  document.getElementById('chModalGain').value  = cgain[freq] || defGn;
  document.getElementById('chModal').classList.add('open');
  setTimeout(() => document.getElementById('chModalLabel').focus(), 50);
}

function showAddChannel() {
  _chModalFreq = null;
  document.getElementById('chModalTitle').textContent = 'Add Channel';
  document.getElementById('chModalFreqField').style.display = '';
  document.getElementById('chModalFreq').value  = '';
  document.getElementById('chModalLabel').value = '';
  document.getElementById('chModalBank').value  = '';
  document.getElementById('chModalMode').value  = '';
  document.getElementById('chModalBW').value    = '';
  document.getElementById('chModalPL').value    = '';
  document.getElementById('chModalHPF').value    = '';
  const mount = Object.keys(S.streams)[0];
  const s = mount ? S.streams[mount] : null;
  document.getElementById('chModalSQ').value  = s ? (s.defaultSquelch || 0.05).toFixed(3) : '0.050';
  document.getElementById('chModalGain').value = s ? (s.defaultGain || 'auto') : 'auto';
  document.getElementById('chModal').classList.add('open');
  setTimeout(() => document.getElementById('chModalFreq').focus(), 50);
}

function closeChModal() {
  document.getElementById('chModal').classList.remove('open');
  _chModalFreq = null;
  // Re-open bank modal if it was open when edit was triggered
  if (_chBankMount) document.getElementById('chBankModal').classList.add('open');
}

function saveChModal() {
  const freq  = _chModalFreq || (document.getElementById('chModalFreq').value || '').trim();
  const label = (document.getElementById('chModalLabel').value || '').trim();
  const sq    = parseFloat(document.getElementById('chModalSQ').value);
  const pl    = parseFloat(document.getElementById('chModalPL').value);
  const gn    = (document.getElementById('chModalGain').value || '').trim();
  const bk    = (document.getElementById('chModalBank').value || '').trim();
  const md    = (document.getElementById('chModalMode').value || '').trim();
  const bw    = parseFloat(document.getElementById('chModalBW').value) || 0;
  const hpf   = parseFloat(document.getElementById('chModalHPF').value) || 0;
  if (!freq) return;
  fetch('/api/channel', {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      freq, label: label || freq,
      squelch_rms: isNaN(sq) ? null : sq,
      gain: gn, pl: isNaN(pl) ? 0 : pl,
      bank: bk, modulation: md,
      bandwidth: bw || null,
      hp_filter: hpf,
    }),
  }).then(() => closeChModal()).catch(e => console.error('[api]', e));
}

function editChannel(freq) {
  // Hide bank modal while edit modal is open; closeChModal() restores it.
  document.getElementById('chBankModal').classList.remove('open');
  openChModal(freq);
}
function cancelEdit() { closeChModal(); }

function deleteChannel(freq) {
  if (!confirm('Remove ' + freq + ' MHz from scanner?')) return;
  fetch('/api/channel/' + encodeURIComponent(freq), { method: 'DELETE' })
    .catch(e => console.error('[api]', e));
}
function skipChannel(freq) {
  fetch('/api/skip', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ freq }),
  }).catch(e => console.error('[api]', e));
}
function resumeScan() {
  fetch('/api/resume', { method: 'POST' }).catch(e => console.error('[api]', e));
}
function holdChannel(freq) {
  fetch('/api/hold', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ freq }),
  }).catch(e => console.error('[api]', e));
}
function toggleBank(mount, bank, enabled) {
  fetch('/api/bank', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ bank, enabled }),
  }).catch(e => console.error('[api]', e));
}

// ── Audio filter / volume controls ────────────────────────────────────────────
// ── Knob controls ──────────────────────────────────────────────────────────────
const _knobs = {};
function _knobSetup(id, cfg) {
  const c = document.getElementById(id);
  if (!c) return;
  _knobs[id] = {...cfg, c};
  _drawKnob(id);
  let sy, sv;
  const move = y => {
    const range = cfg.max - cfg.min;
    let v = sv + ((sy - y) / 120) * range;
    v = Math.max(cfg.min, Math.min(cfg.max, v));
    if (cfg.step) v = Math.round(v / cfg.step) * cfg.step;
    _knobs[id].value = v;
    _drawKnob(id);
    cfg.onChange(v);
  };
  c.addEventListener('mousedown', e => {
    sy = e.clientY; sv = _knobs[id].value;
    const mm = e2 => move(e2.clientY);
    const mu = () => { removeEventListener('mousemove', mm); removeEventListener('mouseup', mu); };
    addEventListener('mousemove', mm); addEventListener('mouseup', mu);
    e.preventDefault();
  });
  c.addEventListener('touchstart', e => { sy = e.touches[0].clientY; sv = _knobs[id].value; e.preventDefault(); }, {passive:false});
  c.addEventListener('touchmove',  e => { move(e.touches[0].clientY); e.preventDefault(); }, {passive:false});
}
function _drawKnob(id) {
  const k = _knobs[id]; if (!k) return;
  const c = k.c, ctx = c.getContext('2d'), w = c.width, cx = w/2, cy = w/2, r = cx - 5;
  ctx.clearRect(0, 0, w, w);
  const sa = Math.PI * 0.75, ea = Math.PI * 2.25;
  const pct = (k.value - k.min) / (k.max - k.min);
  const va = sa + pct * (ea - sa);
  // Track
  ctx.beginPath(); ctx.arc(cx, cy, r, sa, ea);
  ctx.strokeStyle = '#111827'; ctx.lineWidth = 4; ctx.lineCap = 'round'; ctx.stroke();
  // Value arc
  if (pct > 0) {
    ctx.beginPath(); ctx.arc(cx, cy, r, sa, va);
    ctx.strokeStyle = k.color; ctx.lineWidth = 4; ctx.lineCap = 'round'; ctx.stroke();
  }
  // Pointer
  ctx.beginPath();
  ctx.arc(cx + Math.cos(va) * (r - 1), cy + Math.sin(va) * (r - 1), 3, 0, Math.PI * 2);
  ctx.fillStyle = k.color; ctx.fill();
  // Center
  ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx.fillStyle = '#1a2a3a'; ctx.fill();
}
function _knobSet(id, value) {
  if (_knobs[id]) { _knobs[id].value = value; _drawKnob(id); }
}

function setVol(v) {
  v = Math.round(v);
  A.vol = v / 100;
  localStorage.setItem('a_vol', A.vol);
  document.getElementById('aVolLbl').textContent = v + '%';
  _knobSet('aVolKnob', v);
  _applyVolume();
}
function setHP(v) {
  // kept as no-op for backwards compatibility with any bookmarked calls
  if (_isNativeApp && audMount) {
    openAudioStream(audMount);
  }
}
function setLP(v) {
  v = Math.round(v / 500) * 500;
  A.lp = v;
  localStorage.setItem('a_lp', A.lp);
  document.getElementById('aLPLbl').textContent = (A.lp / 1000).toFixed(1) + 'k';
  _knobSet('aLPKnob', v);
  if (_lpNode) {
    _lpNode.frequency.value = A.lp;
  } else if (_mseActive && S.audioOn) {
    _mseConnect();
  } else if (_isNativeApp && audMount) {
    openAudioStream(audMount);
  }
}
function toggleSqTail() { setSqTail(!A.sqtail); }
function setSqTail(v) {
  A.sqtail = v;
  localStorage.setItem('a_sqtail', v);
  const btn = document.getElementById('aSqTailBtn');
  if (btn) btn.classList.toggle('active', !!v);
  if (!v) { _gateGain = 1.0; _applyVolume(); }
}

function initControls() {
  const vol = Math.round(A.vol * 100);
  _knobSetup('aVolKnob', { min:0, max:150, value:vol, step:1, color:'#2dff6e',
    onChange: v => { document.getElementById('aVolLbl').textContent = Math.round(v) + '%'; A.vol = v/100; localStorage.setItem('a_vol', A.vol); _applyVolume(); }
  });
  document.getElementById('aVolLbl').textContent = vol + '%';
  _knobSetup('aLPKnob', { min:1000, max:8000, value:A.lp, step:500, color:'#ffb000',
    onChange: v => { v = Math.round(v/500)*500; document.getElementById('aLPLbl').textContent = (v/1000).toFixed(1)+'k'; A.lp=v; localStorage.setItem('a_lp',v); if(_lpNode)_lpNode.frequency.value=v; else if(_mseActive&&S.audioOn)_mseConnect(); else if(_isNativeApp&&audMount)openAudioStream(audMount); }
  });
  document.getElementById('aLPLbl').textContent = (A.lp / 1000).toFixed(1) + 'k';
  const sqBtn = document.getElementById('aSqTailBtn');
  if (sqBtn) sqBtn.classList.toggle('active', !!A.sqtail);
  if (_isAndroidBrowser && 'wakeLock' in navigator)
    document.getElementById('aWakeLockRow').style.display = '';
}

// ── Screen Wake Lock (Android browser keep-alive only) ────────────────────────
// Only used on the Android browser path.  The native app has a WiFi lock +
// foreground service — those are more effective and don't drain the battery
// by keeping the screen on.
let _wakeLock = null;
let _wakeLockWanted = false;  // tracks user intent; prevents re-acquire after manual release

async function _acquireWakeLock() {
  if (!('wakeLock' in navigator) || _wakeLock) return;
  _wakeLockWanted = true;
  try {
    _wakeLock = await navigator.wakeLock.request('screen');
    _wakeLock.addEventListener('release', () => {
      _wakeLock = null;
      // Only re-acquire if the USER still wants it (not a manual release)
      // AND the OS released it involuntarily (screen-on restoration on resume).
      if (_wakeLockWanted && S.audioOn && document.visibilityState === 'visible')
        setTimeout(_acquireWakeLock, 1000);
    });
    const btn = document.getElementById('aWakeLockBtn');
    if (btn) btn.classList.add('active');
  } catch (_) {}
}
function _releaseWakeLock() {
  _wakeLockWanted = false;   // user explicitly turned it off — don't re-acquire
  if (_wakeLock) { _wakeLock.release(); _wakeLock = null; }
  const btn = document.getElementById('aWakeLockBtn');
  if (btn) btn.classList.remove('active');
}
function toggleWakeLock() { setWakeLock(!_wakeLockWanted); }
function setWakeLock(on) {
  if (on) _acquireWakeLock(); else _releaseWakeLock();
}

// ── Init ───────────────────────────────────────────────────────────────────────
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible') return;
  // Force-reconnect the control WS immediately (bypasses exponential backoff)
  if (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
    wsRetry = 0;
    connect();
  }
  // Resume the audio element if it was paused while locked.
  // visibilitychange counts as user activation in Chrome, so play() works here.
  if (S.audioOn && _audEl && _audEl.paused) {
    _audEl.play().catch(() => {});
  }
});
if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(() => {});

// Audio is always on — start immediately without requiring a click.
S.audioOn = true;
localStorage.setItem('a_on', 'true');

_initAudEl();
connect();
initControls();
// Resume AudioContext on first pointer interaction (browser autoplay policy).
document.addEventListener('pointerdown', function _resumeCtx() {
  if (_audioCtx && _audioCtx.state === 'suspended') _audioCtx.resume().catch(() => {});
}, { once: true, passive: true });
updateAudioUI();
</script>
</body>
</html>
"""

# ── librtlsdr ctypes wrapper ───────────────────────────────────────────────────
class _RtlSdr:
    """
    Thin ctypes wrapper around librtlsdr.so — no pyrtlsdr package required.
    Uses only functions present in all librtlsdr versions (avoids
    rtlsdr_set_dithering and other symbols added in newer forks).
    """

    def __init__(self, device_index: int = 0):
        import ctypes, ctypes.util
        _RtlSdr._usb_reset(device_index)
        lib = None
        for name in ("librtlsdr.so.0", "librtlsdr.so"):
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                pass
        if lib is None:
            path = ctypes.util.find_library("rtlsdr")
            if path:
                lib = ctypes.CDLL(path)
        if lib is None:
            raise RuntimeError("librtlsdr not found — sudo apt install rtl-sdr")
        self._lib = lib
        self._dev = ctypes.c_void_p()
        r = lib.rtlsdr_open(ctypes.byref(self._dev), device_index)
        if r != 0:
            raise RuntimeError(f"rtlsdr_open(index={device_index}) failed: {r}")

    @staticmethod
    def _usb_reset(device_index: int) -> None:
        """Power-cycle the USB device to clear stale RTL2832U I2C/pipe state."""
        try:
            import glob, os, time
        except ImportError:
            return
        devs = []
        for vpath in sorted(glob.glob("/sys/bus/usb/devices/*/idVendor")):
            try:
                if open(vpath).read().strip() != "0bda":
                    continue
                pid = open(vpath.replace("idVendor", "idProduct")).read().strip()
                if pid not in ("2832", "2838", "2820", "0832"):
                    continue
                base    = os.path.dirname(vpath)
                bus     = int(open(os.path.join(base, "busnum")).read())
                devnum  = int(open(os.path.join(base, "devnum")).read())
                sysname = os.path.basename(base)
                devs.append((bus, devnum, sysname, base))
            except Exception:
                pass
        devs.sort()
        if device_index >= len(devs):
            return
        bus, devnum, sysname, base = devs[device_index]

        # Primary: deauthorize then reauthorize — software equivalent of unplug/replug.
        # This fully power-cycles the RTL2832U and clears stuck I2C state that
        # survives a plain USB reset.
        auth = os.path.join(base, "authorized")
        try:
            with open(auth, "w") as f: f.write("0")
            time.sleep(0.5)
            with open(auth, "w") as f: f.write("1")
            time.sleep(2.0)
            print(f"[Scanner] USB power-cycle: {sysname}")
            return
        except Exception:
            pass

        # Fallback: USBDEVFS_RESET (resets USB protocol layer only)
        try:
            import fcntl
            node = f"/dev/bus/usb/{bus:03d}/{devnum:03d}"
            with open(node, "wb") as fh:
                fcntl.ioctl(fh, USBDEVFS_RESET, 0)
            time.sleep(0.5)
            print(f"[Scanner] USB reset: {node}")
        except Exception as e:
            print(f"[Scanner] USB reset failed ({sysname}): {e}")

    @staticmethod
    def index_for_serial(lib_path_hint: str, serial: str) -> int:
        import ctypes
        lib = ctypes.CDLL(lib_path_hint)
        n = lib.rtlsdr_get_device_count()
        for i in range(n):
            buf = ctypes.create_string_buffer(256)
            lib.rtlsdr_get_device_usb_strings(i, None, None, buf)
            if buf.value.decode(errors="replace") == serial:
                return i
        raise RuntimeError(f"RTL-SDR with serial '{serial}' not found")

    def set_sample_rate(self, rate: int):
        r = self._lib.rtlsdr_set_sample_rate(self._dev, int(rate))
        if r != 0:
            raise RuntimeError(f"rtlsdr_set_sample_rate({rate}) failed: {r}")

    def set_center_freq(self, freq: int):
        r = self._lib.rtlsdr_set_center_freq(self._dev, int(freq))
        if r != 0:
            raise RuntimeError(f"rtlsdr_set_center_freq({freq/1e6:.3f} MHz) failed: {r}")

    def set_freq_correction(self, ppm: int):
        r = self._lib.rtlsdr_set_freq_correction(self._dev, int(ppm))
        if r not in (0, -2):  # -2 = already at this value, not an error
            raise RuntimeError(f"rtlsdr_set_freq_correction({ppm}) failed: {r}")

    def set_gain(self, gain):
        if str(gain).lower() == "auto":
            r = self._lib.rtlsdr_set_tuner_gain_mode(self._dev, 0)
        else:
            r = self._lib.rtlsdr_set_tuner_gain_mode(self._dev, 1)
            if r == 0:
                r = self._lib.rtlsdr_set_tuner_gain(self._dev, int(float(gain) * 10))
        if r != 0:
            raise RuntimeError(f"set_gain({gain}) failed: {r}")

    def reset_buffer(self):
        self._lib.rtlsdr_reset_buffer(self._dev)

    def start_async(self, callback, buf_len: int = 0):
        """
        Block until cancel_async() is called, invoking callback(samples) for
        each USB packet.  Runs rtlsdr_read_async in the calling thread — run
        this in a daemon thread.  buf_len must be a multiple of 512 bytes.
        """
        import ctypes
        _CB  = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint8),
                                 ctypes.c_uint32, ctypes.c_void_p)
        _AT  = ctypes.c_ubyte * buf_len   # pre-built array type — avoids per-call alloc

        def _c_cb(buf, length, _ctx):
            n   = int(length)
            # Zero-copy view of librtlsdr's buffer, then .copy() before it's reused
            raw = np.frombuffer(
                _AT.from_address(ctypes.cast(buf, ctypes.c_void_p).value),
                dtype=np.uint8, count=n).copy()
            f = (raw.astype(np.float32) - 127.5) / 127.5
            callback((f[::2] + 1j * f[1::2]).astype(np.complex64))

        self._async_cb = _CB(_c_cb)   # keep ref — GC of this crashes the process
        self._lib.rtlsdr_read_async(self._dev, self._async_cb, None,
                                     ctypes.c_uint32(0), ctypes.c_uint32(buf_len))

    def cancel_async(self):
        if self._dev:
            self._lib.rtlsdr_cancel_async(self._dev)

    def read_samples(self, num_samples: int) -> "np.ndarray":
        import ctypes
        n_bytes = num_samples * 2          # interleaved I + Q, 1 byte each
        buf     = (ctypes.c_uint8 * n_bytes)()
        n_read  = ctypes.c_int()
        r = self._lib.rtlsdr_read_sync(
            self._dev, buf, ctypes.c_int(n_bytes), ctypes.byref(n_read))
        if r != 0:
            raise RuntimeError(f"rtlsdr_read_sync failed: {r}")
        raw = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        iq  = (raw - 127.5) / 127.5
        return (iq[::2] + 1j * iq[1::2]).astype(np.complex64)

    def close(self):
        if self._dev:
            self._lib.rtlsdr_close(self._dev)
            self._dev = None


# ── Scanner ────────────────────────────────────────────────────────────────────
class RTLFMScanner:
    """
    Opens the RTL-SDR device once via librtlsdr (ctypes), retuning between
    frequencies with set_center_freq(). FM demodulation and decimation are
    done in Python/numpy — no rtl_fm subprocess, no USB open/close per hop.
    """
    CHUNK_SECS = 0.05  # seconds of audio per processing chunk (smaller = faster scan response)

    def __init__(self, name: str, channels: dict[str, str],
                 squelch_rms: float = 0.05,
                 squelch_hold: float = 2.0,
                 channel_squelch: dict[str, float] | None = None,
                 channel_gain: dict[str, str] | None = None,
                 channel_pl: dict[str, float] | None = None,
                 channel_bank: dict[str, str] | None = None,
                 channel_modulation: dict[str, str] | None = None,
                 channel_bandwidth: dict[str, float] | None = None,
                 channel_hp_filter: dict[str, float] | None = None,
                 banks_enabled: dict[str, bool] | None = None,
                 skipped: set[str] | None = None,
                 ppm: int = 0, modulation: str = "fm",
                 device: str = "0", gain: str = "auto",
                 samp_rate: int = 240000, scan_dwell: float = 0.25,
                 fir_taps: int = 127,
                 debug: bool = False,
                 on_event=None, on_audio=None):
        self.name            = name
        self.channels        = channels
        self.frequencies     = sorted(float(f) for f in channels)
        self.squelch_rms     = squelch_rms    # default phase-variance threshold (0.0–1.0)
        self.squelch_hold    = squelch_hold   # seconds before clearing inactive freq
        self.channel_squelch = channel_squelch or {}  # per-freq squelch overrides
        self.channel_gain    = channel_gain    or {}  # per-freq gain overrides (e.g. "25.4" or "auto")
        self.channel_pl         = channel_pl         or {}  # per-freq CTCSS tone (Hz); 0.0 or absent = disabled
        self.channel_bank       = channel_bank       or {}  # per-freq bank name ('' = Default)
        self.channel_modulation = channel_modulation or {}  # per-freq mode: 'fm','nfm','am' (absent = global modulation)
        self.channel_bandwidth  = channel_bandwidth  or {}  # per-freq channel bandwidth in kHz (absent = auto)
        self.channel_hp_filter  = channel_hp_filter  or {}  # per-freq HPF cutoff Hz (0/absent = off)
        self.banks_enabled      = banks_enabled      or {}  # bank name → enabled (absent = True)
        self.skipped         = skipped or set()       # freqs excluded from scan rotation
        self.debug           = debug
        self.ppm          = ppm
        self.modulation   = modulation
        self.device       = str(device)
        self.gain         = str(gain)
        self.samp_rate    = samp_rate
        self.fir_taps     = fir_taps
        self.scan_dwell   = scan_dwell
        self.audio_rate   = AUDIO_RATE
        self._on_event      = on_event
        self._on_audio      = on_audio
        self._broadcastify: "BroadcastifyFeeder | None" = None

        self._running      = False
        self._lock         = threading.Lock()
        self._active_freq  = None
        self._active_since = None
        self._history      = deque(maxlen=20)
        self.hold_freq: str | None = None   # non-None = scanner locked to this frequency
        self.connected     = False
        self.last_error: str | None = None
        # Monotonic sample counter — incremented by len(audio_samples) each time
        # audio is emitted.  Embedded in freq_change so the client knows exactly
        # which sample position the transmission started at; also in audio_stats
        # telemetry so the client can compute true lag without wall-clock estimates.
        self.audio_seq: int = 0
        self._resume_event = threading.Event()  # set to force immediate scan advance
        self._sdr_ref: "_RtlSdr | None" = None  # set while device is open; used for clean shutdown

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
        self._thread = threading.Thread(target=self._loop, daemon=True, name="scanner")
        self._thread.start()

    def stop(self):
        self._running = False

    def stop_and_join(self, timeout: float = 6.0) -> None:
        """Signal the scan loop to exit and wait for the device to close cleanly."""
        self._running = False
        self._resume_event.set()   # unblock any dwell sleep
        # Close the device directly — rtlsdr_close() calls cancel_async internally
        # and releases the USB device, which reliably unblocks rtlsdr_read_async.
        # cancel_async() alone is sometimes insufficient.  The scan loop's finally
        # block will call close() again; _RtlSdr.close() is idempotent (_dev guard).
        sdr = self._sdr_ref
        if sdr is not None:
            try: sdr.close()
            except Exception: pass
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def set_channel(self, freq: str, label: str,
                    squelch_rms: float | None = None,
                    gain: str | None = None,
                    pl: float | None = None,
                    bank: str | None = None,
                    modulation: str | None = None,
                    bandwidth: float | None = None,
                    hp_filter: float | None = None) -> None:
        with self._lock:
            self.channels[freq] = label
            if squelch_rms is not None:
                self.channel_squelch[freq] = squelch_rms
            elif freq in self.channel_squelch:
                del self.channel_squelch[freq]
            if gain is not None:
                self.channel_gain[freq] = gain
            elif freq in self.channel_gain:
                del self.channel_gain[freq]
            if pl is not None:
                if pl > 0.0:
                    self.channel_pl[freq] = pl
                else:
                    self.channel_pl.pop(freq, None)
            if bank is not None:
                self.channel_bank[freq] = bank.strip()
            if modulation is not None:
                m = modulation.strip().lower()
                if m:
                    self.channel_modulation[freq] = m
                else:
                    self.channel_modulation.pop(freq, None)
            if bandwidth is not None:
                if bandwidth > 0:
                    self.channel_bandwidth[freq] = bandwidth
                else:
                    self.channel_bandwidth.pop(freq, None)
            if hp_filter is not None:
                if hp_filter > 0:
                    self.channel_hp_filter[freq] = hp_filter
                else:
                    self.channel_hp_filter.pop(freq, None)

    def remove_channel(self, freq: str) -> None:
        with self._lock:
            self.channels.pop(freq, None)
            self.channel_squelch.pop(freq, None)
            self.channel_gain.pop(freq, None)
            self.channel_pl.pop(freq, None)
            self.channel_bank.pop(freq, None)
            self.channel_modulation.pop(freq, None)
            self.channel_bandwidth.pop(freq, None)
            self.channel_hp_filter.pop(freq, None)
            self.skipped.discard(freq)
            if self._active_freq == freq:
                self._active_freq  = None
                self._active_since = None

    def toggle_skip(self, freq: str) -> bool:
        """Toggle freq in/out of the scan rotation. Returns True if now skipped."""
        with self._lock:
            if freq in self.skipped:
                self.skipped.discard(freq)
                return False
            else:
                self.skipped.add(freq)
                if self._active_freq == freq:
                    self._active_freq  = None
                    self._active_since = None
                return True

    def toggle_hold(self, freq: str) -> bool:
        """Lock scanner to freq (returns True) or release hold (returns False)."""
        with self._lock:
            if self.hold_freq == freq:
                self.hold_freq = None
                return False
            else:
                self.hold_freq = freq
                self._resume_event.set()  # break out of current dwell immediately
                return True

    def set_bank_enabled(self, bank: str, enabled: bool) -> None:
        with self._lock:
            self.banks_enabled[bank] = enabled
        self._resume_event.set()  # skip any held freq that may now be excluded

    def _bank_list_locked(self) -> dict[str, bool]:
        """Return {bank_name: enabled}. Caller must hold self._lock."""
        banks: dict[str, bool] = {}
        for freq in self.channels:
            b = self.channel_bank.get(freq, '') or 'Default'
            if b not in banks:
                banks[b] = self.banks_enabled.get(b, True)
        if not banks:
            banks['Default'] = self.banks_enabled.get('Default', True)
        return banks

    def bank_list(self) -> dict[str, bool]:
        """Return {bank_name: enabled} for all known banks including Default."""
        with self._lock:
            return self._bank_list_locked()

    def resume_scan(self) -> None:
        """Force the scan loop to advance to the next frequency immediately."""
        self._resume_event.set()

    def _emit(self, evt: dict):
        if self._on_event: self._on_event(evt)

    def set_broadcastify(self, feeder: "BroadcastifyFeeder | None"):
        self._broadcastify = feeder

    def _emit_audio(self, pcm: bytes):
        # Advance the sequence counter before dispatching so that any freq_change
        # emitted in the same chunk sees the sample position AFTER this audio block.
        self.audio_seq += len(pcm) // 2   # bytes → 16-bit samples
        if self._on_audio: self._on_audio(pcm)
        if self._broadcastify: self._broadcastify.send(pcm)

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
        import time
        import queue as _q
        import threading

        freq_keys = list(self.channels.keys())
        n_freqs   = len(freq_keys)

        # Derive decimation from samp_rate, rounded to nearest integer multiple of
        # audio_rate so the division is exact.  Lower samp_rate = narrower capture
        # bandwidth = better adjacent-channel rejection.
        # RTL-SDR valid sample rates: 225,001–300,000 Hz OR 900,001–3,200,000 Hz.
        # The range 300,001–900,000 Hz is not supported by the RTL2832U hardware.
        # Snap any out-of-range value to 240,000 (best selectivity, low range) or
        # 960,000 (safe high range) depending on which side of the gap we're on.
        _sr = self.samp_rate
        if _sr < 225_001 or (300_001 <= _sr <= 900_000):
            _snapped = 240_000 if _sr <= 600_000 else 960_000
            print(f"[Scanner] samp_rate {_sr} Hz is outside RTL-SDR valid ranges "
                  f"(225001–300000 or 900001–3200000) — using {_snapped} Hz")
            _sr = _snapped
        decimate = max(1, round(_sr / self.audio_rate))
        hw_rate  = self.audio_rate * decimate

        # FIR anti-aliasing filter for decimation.
        # Cutoff is set exactly at the output Nyquist (1.0/decimate normalized) so that
        # decimated noise samples remain uncorrelated — preserving the π²/3 phase-variance
        # baseline the squelch relies on.  More taps sharpens the transition band so
        # adjacent channels 25 kHz away fall in the stopband.
        # Default 127-tap Blackman-Harris at 240 kHz: stopband starts ~19.5 kHz,
        # giving ~92 dB rejection of signals 25 kHz away (GMRS/MURS channel spacing).
        fir_coeffs = firwin(self.fir_taps, 1.0 / decimate, window='blackmanharris')
        # Align to USB max-packet-size boundary: RTL-SDR never sends short packets,
        # so a non-multiple-of-512 transfer causes LIBUSB_ERROR_OVERFLOW (-8).
        # chunk_n × 2 bytes must be a multiple of 512 → chunk_n multiple of 256.
        chunk_n = ((int(hw_rate * self.CHUNK_SECS) + 255) // 256) * 256
        buf_len = chunk_n * 2   # bytes per rtlsdr_read_async callback

        fm_scale      = self.audio_rate / (2.0 * np.pi * 5000.0)
        _NOISE_VAR    = np.pi ** 2 / 3.0
        _SQUELCH_FADE = 240   # 10 ms ramp at 24 kHz, applied at squelch open/close
        _OPEN_DEBOUNCE = 3    # consecutive active chunks required to open squelch (~150 ms)
        # De-emphasis: 1-pole IIR lowpass at 2122 Hz (τ = 75 μs North-American standard).
        _deemph_alpha = float(np.exp(-1.0 / (self.audio_rate * 75e-6)))
        _deemph_beta  = 1.0 - _deemph_alpha

        overrides = ", ".join(f"{f}={v}" for f, v in sorted(self.channel_squelch.items()))
        # Transition-band analysis: stopband edge ≈ cutoff + (8/taps)*Nyquist
        _nyq = hw_rate / 2
        _cutoff_hz  = round(1.0 / decimate * _nyq)
        _stopband_hz = round(_cutoff_hz + (8 / self.fir_taps) * _nyq)
        print(f"[Scanner] librtlsdr/async — {n_freqs} freq(s), hw_rate={hw_rate}, "
              f"decimate×{decimate}→{self.audio_rate} Hz, "
              f"FIR {self.fir_taps}-tap BH: passband 0–{_cutoff_hz} Hz, "
              f"stopband from ~{_stopband_hz} Hz, "
              f"scan_dwell={self.scan_dwell}s, squelch_rms={self.squelch_rms}"
              + (f", per-channel overrides: {overrides}" if overrides else ""))
        for fk in freq_keys:
            thr = self.channel_squelch.get(fk, self.squelch_rms)
            print(f"[Scanner]   {fk} MHz  thr={thr}  ({20*np.log10(max(thr,1e-9)):.1f} dB)")

        dev_index = int(self.device) if self.device.isdigit() else \
                    _RtlSdr.index_for_serial("librtlsdr.so.0", self.device)
        sdr = _RtlSdr(device_index=dev_index)
        self._sdr_ref = sdr

        iq_q           = _q.Queue(maxsize=32)
        _reader_thread = [None]

        def _cb(samples):
            # Drop oldest chunk if the processing loop is falling behind so the
            # queue never fills with stale audio.
            if iq_q.full():
                try: iq_q.get_nowait()
                except _q.Empty: pass
            try: iq_q.put_nowait(samples)
            except _q.Full: pass

        def _stop_reader():
            try: sdr.cancel_async()
            except Exception: pass
            t = _reader_thread[0]
            if t is not None:
                t.join(timeout=3.0)
                _reader_thread[0] = None
            while True:
                try: iq_q.get_nowait()
                except _q.Empty: break

        def _start_reader(freq_hz):
            # set_center_freq and reset_buffer must be called while async is stopped.
            sdr.set_center_freq(freq_hz)
            sdr.reset_buffer()
            # librtlsdr prints "Allocating 15 zero-copy buffers" directly to fd 2
            # on every rtlsdr_read_async call.  Redirect stderr to /dev/null for
            # the duration of async init; restore it once the first callback fires
            # (which only happens after initialization is complete).
            _saved_stderr = None
            if not self.debug:
                import os as _os
                _devnull = _os.open(_os.devnull, _os.O_WRONLY)
                _saved_stderr = _os.dup(2)
                _os.dup2(_devnull, 2)
                _os.close(_devnull)
            t = threading.Thread(
                target=sdr.start_async,
                args=(_cb,),
                kwargs={"buf_len": buf_len},
                daemon=True,
            )
            t.start()
            _reader_thread[0] = t
            # Discard first callback — contains samples buffered before the retune.
            # Restore stderr here: the first callback fires only after librtlsdr
            # finishes its initialization (and has already printed the message).
            try:
                iq_q.get(timeout=5.0)
            except _q.Empty:
                raise RuntimeError("rtlsdr async reader did not deliver samples — device stalled?")
            finally:
                if _saved_stderr is not None:
                    import os as _os
                    _os.dup2(_saved_stderr, 2)
                    _os.close(_saved_stderr)

        try:
            sdr.set_sample_rate(hw_rate)
            # Tune to the first scan frequency BEFORE applying ppm correction.
            # rtlsdr_set_freq_correction internally calls set_center_freq(dev->freq)
            # to re-apply the correction; if called before any tune, dev->freq is
            # 100 MHz (set by rtlsdr_open init), and the 100 MHz + 1008000 Hz combo
            # fails to lock the R820T PLL on this librtlsdr version.
            sdr.set_center_freq(int(float(freq_keys[0]) * 1_000_000))
            sdr.set_freq_correction(self.ppm)
            sdr.set_gain(self.gain)

            self.connected = True
            self._emit({"type": "conn", "mount": "sdr", "connected": True, "error": None})

            scan_idx = 0

            while self._running:
                # Rebuild the channel list each iteration so additions/removals/skips
                # made through the UI take effect without restarting the scanner.
                with self._lock:
                    freq_keys = [f for f in self.channels
                                 if f not in self.skipped
                                 and self.banks_enabled.get(
                                     self.channel_bank.get(f, '') or 'Default', True)]
                n_freqs = len(freq_keys)
                if n_freqs == 0:
                    time.sleep(0.1)
                    continue
                scan_idx = scan_idx % n_freqs

                freq_str  = freq_keys[scan_idx]
                with self._lock:
                    # Hold overrides the normal scan rotation.
                    hf = self.hold_freq
                    if hf and hf in freq_keys:
                        freq_str = hf
                        scan_idx = freq_keys.index(hf)
                    label       = self.channels.get(freq_str, freq_str)
                    threshold   = self.channel_squelch.get(freq_str, self.squelch_rms)
                    eff_gain    = self.channel_gain.get(freq_str, self.gain)
                    pl_tone     = self.channel_pl.get(freq_str, 0.0)
                    ch_mode     = self.channel_modulation.get(freq_str, self.modulation).lower()
                    ch_bw_khz   = self.channel_bandwidth.get(freq_str, 0.0)  # 0 = auto
                    ch_hp_filter = self.channel_hp_filter.get(freq_str, 0.0)  # Hz; 0 = off
                freq_hz   = int(float(freq_str) * 1_000_000)

                if self.debug:
                    print(f"[scan] → {freq_str} MHz  ({label})  gain={eff_gain}")

                _stop_reader()
                sdr.set_gain(eff_gain)
                _start_reader(freq_hz)

                dwell_start     = time.time()
                squelch_open    = False
                last_sig_t      = 0.0
                last_iq         = None   # per-frequency; valid across chunks with async continuity
                deemph_z        = 0.0
                hp_pl_b = hp_pl_a = hp_pl_zi = None  # computed once per hop if ch_hp_filter > 0
                if ch_hp_filter > 0:
                    # Notch at the exact tone frequency; Q=35 → ~±3 Hz bandwidth.
                    hp_pl_b, hp_pl_a = iirnotch(ch_hp_filter, Q=35.0, fs=self.audio_rate)
                    hp_pl_zi = lfilter_zi(hp_pl_b, hp_pl_a) * 0.0
                _last_dbg_state = None
                ctcss_buf: list      = []    # accumulation buffer for CTCSS detection
                ctcss_detected: bool = (pl_tone == 0.0)  # pessimistic for PL channels; True when no PL configured
                detected_ctcss: float | None = None   # last detected tone for display
                # For a known PL tone accumulate ~6 complete cycles before analysing;
                # much shorter than the 4096-sample default, allowing detection to
                # complete during the carrier debounce window (parallel, not sequential).
                ctcss_window: int = (max(512, int(self.audio_rate / pl_tone * 6))
                                     if pl_tone > 0.0 else _CTCSS_WINDOW)
                self._resume_event.clear()  # reset any pending resume from previous dwell
                fir_zi_i = np.zeros(len(fir_coeffs) - 1)  # FIR state reset on each freq hop
                fir_zi_q = np.zeros(len(fir_coeffs) - 1)
                sq_just_opened = False
                sq_just_closed = False
                open_debounce  = 0

                need_freq_clear = False
                while self._running:
                    # Exit inner loop immediately if this freq was skipped mid-dwell,
                    # or if hold_freq was changed to a different frequency (click-to-tune
                    # while active). Checking hold_freq here avoids a race where
                    # _resume_event is set by toggle_hold but then cleared by the outer
                    # loop's resume_event.clear() before the inner loop gets to check it.
                    with self._lock:
                        if freq_str in self.skipped:
                            if squelch_open:
                                self._active_freq  = None
                                self._active_since = None
                                need_freq_clear    = True
                            break
                        _hf = self.hold_freq
                        if _hf is not None and _hf != freq_str:
                            if squelch_open:
                                self._active_freq  = None
                                self._active_since = None
                                need_freq_clear    = True
                            break
                        threshold = self.channel_squelch.get(freq_str, self.squelch_rms)
                        new_gain  = self.channel_gain.get(freq_str, self.gain)
                        new_pl    = self.channel_pl.get(freq_str, 0.0)
                    if new_gain != eff_gain:
                        eff_gain = new_gain
                        sdr.set_gain(eff_gain)
                    if new_pl != pl_tone:
                        pl_tone        = new_pl
                        ctcss_buf      = []
                        ctcss_detected = (pl_tone == 0.0)
                        detected_ctcss = None
                        ctcss_window   = (max(512, int(self.audio_rate / pl_tone * 6))
                                          if pl_tone > 0.0 else _CTCSS_WINDOW)
                    try:
                        raw = iq_q.get(timeout=0.5)
                    except _q.Empty:
                        if not self._running:
                            break   # clean shutdown — exit inner chunk loop
                        raise RuntimeError("rtlsdr async read timed out — device stalled?")

                    # Stage 1 — FIR anti-aliasing + stride decimation.
                    # Filter all samples to maintain continuous zi state, then stride-decimate.
                    filt_i, fir_zi_i = lfilter(fir_coeffs, [1.0], raw.real, zi=fir_zi_i)
                    filt_q, fir_zi_q = lfilter(fir_coeffs, [1.0], raw.imag, zi=fir_zi_q)
                    n_out = len(raw) // decimate
                    iq_if = (filt_i[:n_out * decimate:decimate]
                             + 1j * filt_q[:n_out * decimate:decimate]).astype(np.complex64)
                    iq_if -= iq_if.mean()   # remove RTL-SDR LO leakage (DC offset at 0 Hz)

                    # Stage 2 — demodulation at audio_rate.
                    if ch_mode == 'am':
                        # AM envelope detection: magnitude minus smoothed DC.
                        env = np.abs(iq_if).astype(np.float32)
                        env -= env.mean()
                        # Normalise to ±1 (typical AM envelope is 0–1 normalised IQ).
                        peak = float(np.max(np.abs(env))) or 1.0
                        audio = np.clip(env / peak, -1.0, 1.0)
                        audio_for_ctcss = audio  # notch filter not applied to AM
                        # Squelch via envelope RMS (no phase-variance for AM).
                        rms_am = float(np.sqrt(np.mean(env ** 2)))
                        noise_ratio  = max(0.0, 1.0 - rms_am * 20.0)
                        signal_level = 1.0 - noise_ratio
                        rms          = signal_level
                        demod        = env  # for CTCSS buffer (won't be used but keeps code path same)
                        # No de-emphasis for AM.
                        deemph_z = 0.0
                    else:
                        # FM / NFM — FM discriminator (phase-difference method).
                        # Deviation scale: if channel bandwidth is set, derive from it
                        # (land mobile standard: max deviation ≈ channel_bw / 5).
                        # Otherwise fall back to mode default.
                        if ch_bw_khz > 0:
                            deviation_hz = ch_bw_khz * 1000.0 / 5.0
                        elif ch_mode == 'nfm':
                            deviation_hz = 2500.0   # ±2.5 kHz (12.5 kHz channel standard)
                        else:
                            deviation_hz = 5000.0   # ±5.0 kHz (25 kHz channel standard)
                        ch_fm_scale = self.audio_rate / (2.0 * np.pi * deviation_hz)

                        # Prepend last IQ sample to maintain phase continuity across chunks.
                        if last_iq is not None:
                            iq_ext = np.concatenate(([last_iq], iq_if))
                        else:
                            iq_ext = iq_if
                        demod = np.angle(iq_ext[1:] * np.conj(iq_ext[:-1]))

                        audio = (demod * ch_fm_scale).astype(np.float32)
                        np.clip(audio, -1.0, 1.0, out=audio)
                        # De-emphasis: 1-pole IIR (y[n] = β·x[n] + α·y[n-1]).
                        audio_f64, zf = lfilter(
                            [_deemph_beta], [1.0, -_deemph_alpha],
                            audio.astype(np.float64), zi=np.array([deemph_z])
                        )
                        deemph_z = float(zf[0])
                        audio = np.clip(audio_f64, -1.0, 1.0).astype(np.float32)
                        # Preserve pre-notch audio for CTCSS analysis — the notch
                        # filter removes the sub-audio tone from the speaker output
                        # but CTCSS detection must see the unfiltered signal.
                        audio_for_ctcss = audio
                        if hp_pl_b is not None:
                            audio_hp, hp_pl_zi = lfilter(hp_pl_b, hp_pl_a, audio, zi=hp_pl_zi)
                            audio = np.clip(audio_hp, -1.0, 1.0).astype(np.float32)

                        # Phase-variance squelch: noise gives var(Δφ) ≈ π²/3;
                        # a captured FM carrier drives it near zero.
                        noise_ratio  = min(float(np.var(demod)) / _NOISE_VAR, 1.0)
                        signal_level = 1.0 - noise_ratio
                        rms          = signal_level

                    last_iq = iq_if[-1]
                    db     = 20.0 * np.log10(max(rms, 1e-9))
                    # Hysteresis: squelch opens above threshold, stays open above
                    # 75 % of threshold.  50 % was too loose — a weak carrier sitting
                    # just below threshold kept refreshing the hold timer indefinitely.
                    close_thr = threshold * 0.75

                    # CTCSS detection — accumulate whenever a carrier is present (above
                    # squelch threshold) so that on PL-gated channels the tone is
                    # confirmed *before* squelch opens, eliminating the false-open burst.
                    # On non-PL channels ctcss_detected is always True (no gating needed).
                    signal_present = rms > (close_thr if squelch_open else threshold)
                    if signal_present or squelch_open:
                        ctcss_buf.extend(audio_for_ctcss.tolist())
                        if len(ctcss_buf) >= ctcss_window:
                            ctcss_detected, detected_ctcss = _ctcss_analyze(
                                np.array(ctcss_buf[:ctcss_window], dtype=np.float32),
                                self.audio_rate, pl_tone,
                            )
                            if self.debug:
                                print(f"[ctcss] {freq_str}: detected={detected_ctcss}  "
                                      f"pl={pl_tone}  gated={'open' if ctcss_detected else 'closed'}")
                            ctcss_buf = ctcss_buf[ctcss_window:]
                    elif pl_tone > 0.0:
                        # Carrier gone — reset so the next transmission is evaluated fresh.
                        ctcss_buf.clear()
                        ctcss_detected = False

                    active = signal_present and ctcss_detected
                    if self.debug:
                        dbg_state = (freq_str, squelch_open, active)
                        if squelch_open or dbg_state != _last_dbg_state:
                            print(f"[squelch] {freq_str}: sig={rms:.3f} "
                                  f"({db:.1f} dB)  thr={threshold}"
                                  f"  {'OPEN' if squelch_open else 'closed'}"
                                  f"  {'ACTIVE' if active else 'inactive'}")
                        _last_dbg_state = dbg_state

                    self._emit({"type": "signal", "mount": "sdr",
                                "db": round(db, 1), "active": active,
                                "ctcss": detected_ctcss})

                    # On PL channels, count the open debounce from carrier presence
                    # alone so that CTCSS detection and RF debounce run in parallel.
                    # Once both are satisfied the squelch opens immediately.
                    # On non-PL channels ctcss_detected is always True, so
                    # carrier_ok == active and behaviour is unchanged.
                    carrier_ok = signal_present if pl_tone > 0.0 else active

                    if carrier_ok:
                        if squelch_open:
                            # Hold-timer refresh rules:
                            # • PL channels: carrier alone (signal_present) refreshes the
                            #   timer so CTCSS detection gaps don't drain it prematurely.
                            # • Non-PL channels: require rms > full threshold (not just the
                            #   hysteresis close_thr) so that noise bursts that briefly
                            #   exceed close_thr can't hold the squelch open indefinitely
                            #   after a weak spike originally opened it.
                            if pl_tone > 0.0 or rms > threshold:
                                last_sig_t = time.time()
                            if active:
                                # Both carrier and PL confirmed: also refresh dwell timer.
                                dwell_start = time.time()
                            open_debounce = 0
                        else:
                            # Debounce: require consecutive carrier chunks before opening.
                            # Prevents brief noise spikes from triggering hold.
                            open_debounce += 1
                            if open_debounce >= _OPEN_DEBOUNCE and ctcss_detected:
                                last_sig_t  = time.time()
                                dwell_start = time.time()
                                squelch_open    = True
                                sq_just_opened  = True
                                sq_just_closed  = False
                                with self._lock:
                                    now   = datetime.now()
                                    self._active_freq  = freq_str
                                    self._active_since = now
                                    self._history.appendleft((now, freq_str, label))
                                if self.debug:
                                    print(f"[Scanner] active: {freq_str} MHz  ({db:.1f} dB)")
                                self._emit({
                                    "type":      "freq_change", "mount": "sdr",
                                    "name":      self.name, "freq": freq_str, "label": label,
                                    "time":      now.isoformat(),
                                    "audio_seq": self.audio_seq,
                                })
                    else:
                        open_debounce = 0

                    # Pre-compute close condition (scan advance, not audio gate).
                    should_close_sq = False
                    holding = False
                    if not active:
                        with self._lock:
                            holding = self.hold_freq == freq_str
                        if squelch_open and time.time() - last_sig_t > self.squelch_hold:
                            should_close_sq = True

                    # Audio gate: emit only while signal is active.
                    # On the first inactive chunk apply a short fade-out to avoid
                    # a click, then go silent for the rest of the hold period.
                    # sq_just_closed is True for exactly one chunk (the transition).
                    if squelch_open and active:
                        out = audio
                        if sq_just_opened:
                            out = audio.copy()
                            n = min(_SQUELCH_FADE, len(out))
                            out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
                            sq_just_opened = False
                        sq_just_closed = True
                        pcm = (out * 32767).astype(np.int16).tobytes()
                        self._emit_audio(pcm)
                    elif squelch_open and sq_just_closed:
                        # One fade-out chunk at the active→inactive transition.
                        out = audio.copy()
                        n = min(_SQUELCH_FADE, len(out))
                        out[:n] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
                        out[n:] = 0.0
                        sq_just_closed = False
                        pcm = (out * 32767).astype(np.int16).tobytes()
                        self._emit_audio(pcm)
                    # else: squelch_open but inactive — emit nothing (silence)

                    # Resume button: advance immediately, closing squelch cleanly first.
                    if self._resume_event.is_set() and n_freqs > 1:
                        self._resume_event.clear()
                        if squelch_open:
                            squelch_open = False
                            with self._lock:
                                self._active_freq  = None
                                self._active_since = None
                            self._emit({"type": "freq_clear", "mount": "sdr"})
                        break

                    if should_close_sq:
                        squelch_open = False
                        with self._lock:
                            self._active_freq  = None
                            self._active_since = None
                        if self.debug:
                            print("[Scanner] squelch closed")
                        self._emit({"type": "freq_clear", "mount": "sdr"})
                        if n_freqs > 1 and not holding:
                            break
                    elif not active and not squelch_open and n_freqs > 1 and not holding:
                        if time.time() - dwell_start > self.scan_dwell:
                            break

                if need_freq_clear:
                    self._emit({"type": "freq_clear", "mount": "sdr"})

                scan_idx = (scan_idx + 1) % n_freqs

        finally:
            _stop_reader()
            self._sdr_ref = None
            try:
                sdr.close()
            except Exception:
                pass


# ── WebSocket manager ──────────────────────────────────────────────────────────
class WsManager:
    def __init__(self):
        self._clients: list[WebSocket] = []
        self._lock = asyncio.Lock()
        # Per-client send locks so broadcast and the per-client keepalive
        # task never write to the same socket concurrently.
        self._send_locks: dict[int, asyncio.Lock] = {}

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.append(ws)
            self._send_locks[id(ws)] = asyncio.Lock()

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            try: self._clients.remove(ws)
            except ValueError: pass
            self._send_locks.pop(id(ws), None)

    async def send_one(self, ws: WebSocket, msg: str) -> bool:
        """Send msg to a single client under its per-socket lock.
        Returns False if the send failed (client dead)."""
        lock = self._send_locks.get(id(ws))
        if lock is None:
            return False
        async with lock:
            try:
                await ws.send_text(msg)
                return True
            except Exception:
                return False

    async def broadcast(self, data: dict):
        msg = json.dumps(data, default=str)
        async with self._lock:
            clients = list(self._clients)
        dead = []
        for ws in clients:
            if not await self.send_one(ws, msg):
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    try: self._clients.remove(ws)
                    except ValueError: pass
                    self._send_locks.pop(id(ws), None)


# ── App state ──────────────────────────────────────────────────────────────────
scanner:      RTLFMScanner | None           = None
wsman:        WsManager                     = WsManager()
_evq:         asyncio.Queue | None          = None
_evloop:      asyncio.AbstractEventLoop | None = None
_audio_clients: list[asyncio.Queue]         = []
_config_path: Path | None                  = None


def _save_config() -> None:
    """Persist current scanner channel list back to scanner_config.json."""
    if not _config_path:
        return
    try:
        if _config_path.exists():
            with open(_config_path) as f:
                cfg = json.load(f)
        else:
            cfg = {}
        with scanner._lock:
            new_channels = {}
            for freq, lbl in scanner.channels.items():
                entry: dict = {"label": lbl}
                if freq in scanner.channel_squelch:
                    entry["squelch_rms"] = scanner.channel_squelch[freq]
                if freq in scanner.channel_gain:
                    entry["gain"] = scanner.channel_gain[freq]
                if freq in scanner.channel_pl:
                    entry["pl"] = scanner.channel_pl[freq]
                bk = scanner.channel_bank.get(freq, '')
                if bk:
                    entry["bank"] = bk
                md = scanner.channel_modulation.get(freq, '')
                if md:
                    entry["modulation"] = md
                bw = scanner.channel_bandwidth.get(freq, 0.0)
                if bw:
                    entry["bandwidth"] = bw
                hpf_hz = scanner.channel_hp_filter.get(freq, 0.0)
                if hpf_hz > 0:
                    entry["hp_filter"] = hpf_hz
                new_channels[freq] = entry if len(entry) > 1 else lbl
        cfg["channels"] = new_channels
        with scanner._lock:
            cfg["skipped"] = sorted(scanner.skipped)
            cfg["banks_enabled"] = {k: v for k, v in scanner.banks_enabled.items() if not v}
        with open(_config_path, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[config] save failed: {e}")


def _channels_event() -> dict:
    with scanner._lock:
        return {
            "type":           "channels_update",
            "mount":          "sdr",
            "channels":       dict(scanner.channels),
            "channelSquelch": dict(scanner.channel_squelch),
            "channelGain":    dict(scanner.channel_gain),
            "channelPL":      dict(scanner.channel_pl),
            "channelBank":        dict(scanner.channel_bank),
            "channelModulation":  dict(scanner.channel_modulation),
            "channelBandwidth":   dict(scanner.channel_bandwidth),
            "channelHpFilter":    dict(scanner.channel_hp_filter),
            "banks":              scanner._bank_list_locked(),
            "defaultSquelch":     scanner.squelch_rms,
            "defaultGain":    scanner.gain,
            "skipped":        sorted(scanner.skipped),
        }


def _emit(event: dict):
    if _evloop and _evq:
        asyncio.run_coroutine_threadsafe(_evq.put(event), _evloop)


_audio_stats_seq: int = 0          # audio_seq value at last stats broadcast
_audio_stats_max_depth: int = 0    # rolling max queue depth since last broadcast

def _audio_cb(pcm: bytes):
    if _evloop and _audio_clients:
        asyncio.run_coroutine_threadsafe(_dispatch_audio(pcm), _evloop)


async def _dispatch_audio(pcm: bytes):
    global _audio_stats_max_depth
    max_depth = 0
    for q in list(_audio_clients):
        # Drop-oldest policy: if the queue is full, evict the head chunk
        # before pushing the new one.  This bounds latency — the queue
        # represents unplayed audio; if a client can't keep up, older
        # audio is discarded rather than making lag grow without bound.
        if q.full():
            try: q.get_nowait()
            except Exception: pass
        try:
            q.put_nowait(pcm)
            depth = q.qsize()
            if depth > max_depth:
                max_depth = depth
        except asyncio.QueueFull:
            pass
    if max_depth > _audio_stats_max_depth:
        _audio_stats_max_depth = max_depth


async def _bcast_loop():
    while True:
        event = await _evq.get()
        await wsman.broadcast(event)


async def _audio_stats_loop():
    """Broadcast audio_stats once per second so clients can compute true lag
    from the monotonic audio_seq counter instead of wall-clock estimates."""
    global _audio_stats_max_depth
    while True:
        await asyncio.sleep(1.0)
        if not scanner or not wsman._clients:
            _audio_stats_max_depth = 0
            continue
        seq   = scanner.audio_seq
        depth = _audio_stats_max_depth
        _audio_stats_max_depth = 0
        # Sum per-stream output counters (includes injected silence chunks so
        # the browser can compute true lag without discounting idle periods).
        out_sum = sum(getattr(q, 'out_samples', 0) for q in _audio_clients)
        await wsman.broadcast({
            "type":        "audio_stats",
            "audio_seq":   seq,
            "out_samples": out_sum,         # total samples delivered (incl. silence)
            "queue_depth": depth,
            "clients":     len(_audio_clients),
        })


_bcast_feeder: "BroadcastifyFeeder | None" = None


def _bcast_status_event(connected: bool, error: str | None):
    """Called by BroadcastifyFeeder on connect/disconnect; relays to WS clients."""
    _emit({"type": "bcast_status", "connected": connected,
           "error": error, "url": _bcast_feeder.url.split('@')[-1] if _bcast_feeder else ""})


def _state() -> dict:
    s = scanner
    with s._lock:
        channels        = dict(s.channels)
        channel_squelch = dict(s.channel_squelch)
        channel_gain    = dict(s.channel_gain)
        channel_pl      = dict(s.channel_pl)
        channel_bank       = dict(s.channel_bank)
        channel_modulation = dict(s.channel_modulation)
        channel_bandwidth  = dict(s.channel_bandwidth)
        channel_hp_filter  = dict(s.channel_hp_filter)
        banks              = s._bank_list_locked()
        skipped         = sorted(s.skipped)
        hold_freq       = s.hold_freq
    bcast = _bcast_feeder
    return {
        "type":       "state",
        "audio_rate": s.audio_rate,
        "broadcastify": {
            "enabled":   bcast is not None,
            "connected": bcast.connected if bcast else False,
            "error":     bcast.last_error if bcast else None,
            "url":       bcast.url.split('@')[-1] if bcast else "",
        },
        "streams": [{
            "mount":          "sdr",
            "name":           s.name,
            "connected":      s.connected,
            "activeFreq":     s.active_freq,
            "activeSince":    s.active_since.isoformat() if s.active_since else None,
            "history": [{"time": t.isoformat(), "freq": f, "label": lb}
                        for t, f, lb in s.history[:10]],
            "channels":       channels,
            "channelSquelch": channel_squelch,
            "channelGain":    channel_gain,
            "channelPL":          channel_pl,
            "channelBank":        channel_bank,
            "channelModulation":  channel_modulation,
            "channelBandwidth":   channel_bandwidth,
            "channelHpFilter":    channel_hp_filter,
            "banks":              banks,
            "defaultSquelch": s.squelch_rms,
            "defaultGain":    s.gain,
            "squelchHold":    s.squelch_hold,
            "holdFreq":       hold_freq,
            "skipped":        skipped,
            "lastError":      s.last_error,
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
    asyncio.create_task(_audio_stats_loop())
    if _bcast_feeder:
        scanner.set_broadcastify(_bcast_feeder)
        _bcast_feeder.start()
    scanner.start()
    print("Scanner started")


@app.on_event("shutdown")
async def _shutdown():
    if _bcast_feeder:
        await asyncio.get_event_loop().run_in_executor(None, _bcast_feeder.stop)
    if scanner:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: scanner.stop_and_join(timeout=15.0))


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=PAGE.replace('__VERSION__', VERSION, 1),
                        headers={"Cache-Control": "no-store"})


@app.get("/manifest.json")
async def pwa_manifest():
    name = (scanner.name if scanner else None) or "RTL Scanner"
    return JSONResponse(
        {
            "name": name,
            "short_name": name[:15],
            "display": "standalone",
            "display_override": ["standalone", "minimal-ui"],
            "start_url": "/",
            "scope": "/",
            "background_color": "#0a0d0f",
            "theme_color": "#0a0d0f",
            "categories": ["music", "utilities"],
            "orientation": "portrait",
            "icons": [
                # Relative paths — resolved against the manifest URL by the browser
                # and by PWABuilder, so they always point to the correct public origin
                # regardless of reverse-proxy configuration.
                {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
                {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
                {"src": "/icon.svg",     "sizes": "any",     "type": "image/svg+xml", "purpose": "any maskable"},
            ],
        },
        headers={"Content-Type": "application/manifest+json"},
    )


_ICON_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <rect width="512" height="512" fill="#0a0d0f"/>
  <circle cx="256" cy="296" r="18" fill="#2dff6e"/>
  <path d="M196 256 a80 80 0 0 1 120 0" fill="none" stroke="#2dff6e" stroke-width="18" stroke-linecap="round"/>
  <path d="M158 218 a130 130 0 0 1 196 0" fill="none" stroke="#2dff6e" stroke-width="14" stroke-linecap="round" opacity=".65"/>
  <path d="M118 180 a182 182 0 0 1 276 0" fill="none" stroke="#2dff6e" stroke-width="10" stroke-linecap="round" opacity=".35"/>
</svg>"""

@app.get("/icon.svg")
async def pwa_icon():
    return Response(content=_ICON_SVG, media_type="image/svg+xml")


def _make_icon_png(size: int) -> bytes:
    """Render the scanner icon as a PNG using only numpy + zlib (no cairosvg).

    Reproduces the SVG: dark background, green centre dot, three concentric
    arc segments opening upward (radio-wave motif).
    """
    import zlib, struct

    img = np.full((size, size, 3), [10, 13, 15], dtype=np.uint8)   # #0a0d0f bg

    # Coordinate grids
    Y, X = np.mgrid[0:size, 0:size]
    # Dot is at (cx, cy) — slightly below centre, matching the SVG viewBox
    cx  = size / 2
    cy  = size * 296 / 512          # SVG: circle cy="296" in 512-high canvas
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    green = np.array([45, 255, 110], dtype=np.float32)   # #2dff6e
    bg    = np.array([10, 13,  15],  dtype=np.float32)   # #0a0d0f

    # Filled centre dot
    r_dot = size * 18 / 512
    img[dist <= r_dot] = green.astype(np.uint8)

    # Three arcs (upper half only: Y <= cy)
    arc_params = [
        (size * 80  / 512, size * 9  / 512, 1.00),   # inner  sw=18 → half-width=9
        (size * 130 / 512, size * 7  / 512, 0.65),   # middle sw=14 → half-width=7
        (size * 182 / 512, size * 5  / 512, 0.35),   # outer  sw=10 → half-width=5
    ]
    above = Y <= cy
    for radius, half_w, alpha in arc_params:
        ring = (dist >= radius - half_w) & (dist <= radius + half_w) & above
        blended = (green * alpha + bg * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        img[ring] = blended

    # --- Encode as PNG (RGB, 8-bit, no alpha) ---
    def _chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return (struct.pack('>I', len(data)) + payload +
                struct.pack('>I', zlib.crc32(payload) & 0xFFFFFFFF))

    ihdr = struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)   # RGB colour type=2
    # Filter byte 0 (None) prepended to every row
    raw_rows = b''.join(b'\x00' + img[row].tobytes() for row in range(size))
    idat = zlib.compress(raw_rows, 6)

    return (b'\x89PNG\r\n\x1a\n'
            + _chunk(b'IHDR', ihdr)
            + _chunk(b'IDAT', idat)
            + _chunk(b'IEND', b''))


_PNG_CACHE: dict[int, bytes] = {}

@app.get("/icon-512.png")
async def pwa_icon_512():
    if 512 not in _PNG_CACHE:
        _PNG_CACHE[512] = _make_icon_png(512)
    return Response(content=_PNG_CACHE[512],
                    media_type="image/png",
                    headers={"Content-Type": "image/png",
                             "Cache-Control": "public, max-age=86400"})


@app.get("/icon-192.png")
async def pwa_icon_192():
    if 192 not in _PNG_CACHE:
        _PNG_CACHE[192] = _make_icon_png(192)
    return Response(content=_PNG_CACHE[192],
                    media_type="image/png",
                    headers={"Content-Type": "image/png",
                             "Cache-Control": "public, max-age=86400"})


@app.get("/sw.js")
async def service_worker():
    js = f"""/* v{VERSION} */
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', () => self.clients.claim());
self.addEventListener('fetch', e => {{
  // Let the browser handle the audio stream and WebSocket natively
  if (e.request.url.includes('/stream') || e.request.url.includes('/ws')) return;
  e.respondWith(fetch(e.request).catch(() => new Response('', {{status: 503}})));
}});
"""
    return Response(content=js, media_type="application/javascript",
                    headers={"Service-Worker-Allowed": "/"})


@app.put("/api/channel")
async def api_put_channel(request: Request):
    body = await request.json()
    freq_raw = str(body.get("freq", "")).strip()
    if not freq_raw:
        return {"ok": False, "error": "freq required"}
    try:
        freq = f"{float(freq_raw):.3f}"
    except ValueError:
        return {"ok": False, "error": "invalid frequency"}
    label = str(body.get("label", freq)).strip() or freq
    sq_raw = body.get("squelch_rms")
    squelch_rms = float(sq_raw) if sq_raw is not None else None
    # gain: only modify if the key was explicitly sent; empty string means "clear override"
    if "gain" in body:
        g = str(body["gain"]).strip()
        gain_arg: str | None = g if g else None
    else:
        gain_arg = ...  # sentinel: don't touch existing gain
    # pl: 0 or empty string means "disable CTCSS"; absent means don't change
    if "pl" in body:
        try:
            pl_arg: float | None = float(body["pl"])
        except (TypeError, ValueError):
            pl_arg = 0.0
    else:
        pl_arg = ...  # sentinel: don't touch existing pl
    bank_arg: str | None = body.get("bank", None)
    if bank_arg is not None:
        bank_arg = str(bank_arg).strip()
    mod_arg: str | None = body.get("modulation", None)
    if mod_arg is not None:
        mod_arg = str(mod_arg).strip().lower()
    bw_raw = body.get("bandwidth")
    bw_arg: float | None = float(bw_raw) if bw_raw not in (None, '', 0) else None
    hp_filter_arg: float | None = (float(body["hp_filter"]) if body["hp_filter"] else 0.0) if "hp_filter" in body else None
    if gain_arg is ... and pl_arg is ...:
        scanner.set_channel(freq, label, squelch_rms, bank=bank_arg, modulation=mod_arg, bandwidth=bw_arg, hp_filter=hp_filter_arg)
    elif gain_arg is ...:
        scanner.set_channel(freq, label, squelch_rms, pl=pl_arg, bank=bank_arg, modulation=mod_arg, bandwidth=bw_arg, hp_filter=hp_filter_arg)
    elif pl_arg is ...:
        scanner.set_channel(freq, label, squelch_rms, gain_arg, bank=bank_arg, modulation=mod_arg, bandwidth=bw_arg, hp_filter=hp_filter_arg)
    else:
        scanner.set_channel(freq, label, squelch_rms, gain_arg, pl_arg, bank=bank_arg, modulation=mod_arg, bandwidth=bw_arg, hp_filter=hp_filter_arg)
    _save_config()
    _emit(_channels_event())
    return {"ok": True, "freq": freq}


@app.delete("/api/channel/{freq:path}")
async def api_delete_channel(freq: str):
    scanner.remove_channel(freq)
    _save_config()
    _emit(_channels_event())
    return {"ok": True}


@app.post("/api/skip")
async def api_toggle_skip(request: Request):
    body = await request.json()
    freq = str(body.get("freq", "")).strip()
    if not freq:
        return {"ok": False, "error": "freq required"}
    now_skipped = scanner.toggle_skip(freq)
    _save_config()
    _emit(_channels_event())
    return {"ok": True, "skipped": now_skipped}


@app.post("/api/bank")
async def api_set_bank(request: Request):
    body = await request.json()
    bank = str(body.get("bank", "")).strip()
    if not bank:
        return {"ok": False, "error": "bank required"}
    enabled = bool(body.get("enabled", True))
    scanner.set_bank_enabled(bank, enabled)
    _save_config()
    _emit(_channels_event())
    return {"ok": True, "bank": bank, "enabled": enabled}


@app.post("/api/hold")
async def api_toggle_hold(request: Request):
    body = await request.json()
    freq = str(body.get("freq", "")).strip()
    if not freq:
        return {"ok": False, "error": "freq required"}
    now_held = scanner.toggle_hold(freq)
    _emit({"type": "hold_update", "mount": "sdr", "holdFreq": freq if now_held else None})
    return {"ok": True, "held": now_held}


@app.post("/api/resume")
async def api_resume():
    scanner.resume_scan()
    return {"ok": True}


@app.get("/debug")
async def debug():
    import shutil
    return {
        "connected":     scanner.connected if scanner else None,
        "active_freq":   scanner.active_freq if scanner else None,
        "audio_clients": len(_audio_clients),
        "queue_size":    _evq.qsize() if _evq else -1,
        "ffmpeg":        shutil.which("ffmpeg") or "NOT FOUND — sudo apt install ffmpeg",
    }



@app.get("/stream/tone")
async def audio_tone(request: Request):
    """Diagnostic: streams a pure 1 kHz sine wave instead of scanner audio.
    Open https://<host>/stream/tone in the Android app (paste in browser, or
    temporarily point AndroidNative.startAudio at this URL) to test whether
    AudioTrack plays a clean tone.  Clean tone = noise is in the scanner PCM.
    Noisy tone = AudioTrack or HAL is adding noise on this device."""
    rate = AUDIO_RATE
    t = 0.0
    freq = 1000.0
    chunk_samples = rate // 10   # 100 ms chunks
    silence = bytes(chunk_samples * 2)

    async def generate():
        nonlocal t
        yield _wav_header(rate)
        import time as _time
        next_send = _time.monotonic()
        try:
            while True:
                if await request.is_disconnected():
                    break
                phase = np.linspace(t, t + chunk_samples, chunk_samples,
                                    endpoint=False, dtype=np.float64)
                t += chunk_samples
                sine = (np.sin(2 * np.pi * freq / rate * phase) * 16000).astype(np.int16)
                yield sine.tobytes()
                # Pace to real-time accounting for elapsed work time so drift
                # doesn't accumulate into long gaps that cause underruns.
                next_send += chunk_samples / rate
                delay = next_send - _time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
        finally:
            pass

    return StreamingResponse(generate(), media_type="audio/wav",
                             headers={"Cache-Control": "no-cache"})


@app.get("/stream/tone48")
async def audio_tone48(request: Request):
    """Diagnostic: same pure 1 kHz sine wave as /stream/tone but at 48 kHz.
    Use with a 48 kHz AudioTrack to test whether 24→48 kHz HAL resampling is
    the noise source.  If /stream/tone (24 kHz) sounds noisy but this is clean,
    the device HAL dislikes 24 kHz input."""
    rate = 48000
    t = 0.0
    freq = 1000.0
    chunk_samples = rate // 10   # 100 ms chunks at 48 kHz

    async def generate():
        nonlocal t
        yield _wav_header(rate)
        import time as _time
        next_send = _time.monotonic()
        try:
            while True:
                if await request.is_disconnected():
                    break
                phase = np.linspace(t, t + chunk_samples, chunk_samples,
                                    endpoint=False, dtype=np.float64)
                t += chunk_samples
                sine = (np.sin(2 * np.pi * freq / rate * phase) * 16000).astype(np.int16)
                yield sine.tobytes()
                next_send += chunk_samples / rate
                delay = next_send - _time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
        finally:
            pass

    return StreamingResponse(generate(), media_type="audio/wav",
                             headers={"Cache-Control": "no-cache"})


@app.get("/stream")
async def audio_stream(request: Request, hp: int = 0, lp: int = 0):
    """HTTP audio stream — serves a never-ending WAV.

    Optional query params mirror the browser's Web Audio BiquadFilters so
    AudioTrack clients (native Android app) can request the same filtering
    the browser applies in-client.  Pass the user's current settings:
      hp  — highpass cutoff Hz (0 = off)
      lp  — lowpass  cutoff Hz (0 = off; browser default is 3000 Hz)

    The browser does its own filtering with Web Audio API, so it should
    request lp=0&hp=0 (the default) and filter client-side as before.
    The native Android app has no Web Audio graph, so it should request
    lp=3000 (or whatever A.lp is) to get the same tonal character.
    """
    rate = scanner.audio_rate if scanner else AUDIO_RATE
    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=30)
    _audio_clients.append(q)
    # Silence timeout must be LONGER than the scanner's audio chunk rate
    # (~50 ms) so silence is never injected mid-transmission.  The asyncio
    # event loop can be delayed 100-400 ms by scanner callbacks; a 500 ms
    # timeout absorbs that jitter.  Silence chunk size matches the timeout so
    # the stream delivers the correct byte rate (rate*2 bytes/s) during
    # squelch-closed periods without starving AudioTrack.
    silence = bytes(int(rate * 2 * 0.5))   # 500 ms of zero PCM

    # Build IIR filter chain when the caller requests HP/LP.
    # We use 4th-order Butterworth — same order as the browser's BiquadFilter
    # cascade.  Filter state (zi) is kept across chunks so there are no
    # discontinuities at chunk boundaries.
    nyq = rate / 2.0
    _b_hp, _a_hp, _zi_hp = None, None, None
    _b_lp, _a_lp, _zi_lp = None, None, None
    if hp > 0 and hp < nyq:
        _b_hp, _a_hp = butter(2, hp / nyq, btype='high')
        _zi_hp = lfilter_zi(_b_hp, _a_hp) * 0.0   # start at zero
    if lp > 0 and lp < nyq:
        _b_lp, _a_lp = butter(2, lp / nyq, btype='low')
        _zi_lp = lfilter_zi(_b_lp, _a_lp) * 0.0

    def _apply_filters(raw: bytes) -> bytes:
        """Apply HP/LP IIR filters in-place and return filtered PCM bytes."""
        nonlocal _zi_hp, _zi_lp
        if _b_hp is None and _b_lp is None:
            return raw
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if _b_hp is not None:
            pcm, _zi_hp = lfilter(_b_hp, _a_hp, pcm, zi=_zi_hp)
        if _b_lp is not None:
            pcm, _zi_lp = lfilter(_b_lp, _a_lp, pcm, zi=_zi_lp)
        return np.clip(pcm, -32768, 32767).astype(np.int16).tobytes()

    # Per-connection output sample counter.  Incremented for every PCM chunk
    # sent — real audio AND injected silence.  Exposed as q.out_samples so
    # the audio_stats loop can include it.  This lets the browser compute lag
    # as (server_audio_seq - q.out_samples) / rate * 1000, which stays correct
    # even while silence is being injected between transmissions.
    q.out_samples = 0   # type: ignore[attr-defined]

    async def generate():
        yield _wav_header(rate)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    data = silence
                filtered = _apply_filters(data)
                q.out_samples += len(filtered) // 2
                yield filtered
        finally:
            try: _audio_clients.remove(q)
            except ValueError: pass

    return StreamingResponse(
        generate(),
        media_type="audio/wav",
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/stream.mp3")
async def audio_stream_mp3(request: Request,
                           hp: int = 0, lp: int = 3000):
    """MP3 stream for Android — ffmpeg encodes PCM to MP3 so Android's media
    stack treats the tab as a radio app, surviving Doze mode indefinitely.
    Query params:
      hp  — highpass cutoff Hz (0 = off, e.g. 100 or 300)
      lp  — lowpass  cutoff Hz (default 3000; 0 or >=8000 = off)
    Requires: sudo apt install ffmpeg"""
    rate = scanner.audio_rate if scanner else AUDIO_RATE
    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=30)
    _audio_clients.append(q)
    # 100 ms chunks — short enough to keep the encoder pipeline warm between
    # real audio bursts without injecting audible gaps mid-transmission.
    silence = bytes(int(rate * 2 * 0.1))

    # Build audio filter chain from HP/LP params.
    # HP: clamp to sane range; 0 means disabled.
    # LP: default 3000 Hz; 0 or >=8000 treated as disabled (no filter needed
    #     at or above the audio Nyquist at 24 kHz).
    af_parts: list[str] = []
    hp_hz = max(0, min(hp, 1000))
    lp_hz = max(0, lp)
    if hp_hz > 0:
        af_parts.append(f'highpass=f={hp_hz}')
    if 0 < lp_hz < 8000:
        af_parts.append(f'lowpass=f={lp_hz}')
    af_args = ['-af', ','.join(af_parts)] if af_parts else []

    try:
        proc = await asyncio.create_subprocess_exec(
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-fflags', '+flush_packets',          # flush output after every frame
            '-f', 's16le', '-ar', str(rate), '-ac', '1', '-i', 'pipe:0',
            *af_args,
            '-c:a', 'libmp3lame', '-b:a', '32k',
            '-reservoir', '0',      # disable bit reservoir — each frame flushes immediately
            '-f', 'mp3', 'pipe:1',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        try: _audio_clients.remove(q)
        except ValueError: pass
        return Response('ffmpeg not found — run: sudo apt install ffmpeg',
                        status_code=503, media_type='text/plain')

    # Shared stop signal — whichever side exits first cancels the other,
    # preventing _feed from writing to a dead pipe (BrokenPipeError).
    stop = asyncio.Event()

    async def _feed():
        # Prime ffmpeg with one silence chunk immediately so it starts
        # producing MP3 frames before the browser calls play().
        try:
            proc.stdin.write(silence)
            await proc.stdin.drain()
        except (BrokenPipeError, OSError):
            stop.set()
            return
        try:
            while not stop.is_set():
                try:
                    data = await asyncio.wait_for(q.get(), timeout=0.1)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    data = silence
                if stop.is_set():
                    break
                try:
                    proc.stdin.write(data)
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
        except Exception:
            pass
        finally:
            stop.set()
            try: proc.stdin.close()
            except Exception: pass
            try: _audio_clients.remove(q)
            except ValueError: pass

    feed_task = asyncio.create_task(_feed())

    async def generate():
        try:
            while not stop.is_set():
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                yield chunk
        finally:
            stop.set()
            feed_task.cancel()
            try: proc.kill()
            except Exception: pass
            await proc.wait()

    return StreamingResponse(
        generate(),
        media_type='audio/mpeg',
        headers={'Cache-Control': 'no-cache'},
    )


@app.websocket("/ws/audio")
async def ws_audio_endpoint(ws: WebSocket,
                            hp: int = 0, lp: int = 3000):
    """WebSocket MP3 audio stream for Android.  Sends binary MP3 frames so
    the browser can feed them directly into a MediaSource SourceBuffer.
    Using WebSocket instead of HTTP streaming avoids Android's background
    network restrictions that kill long-running HTTP responses after a few
    minutes of inactivity.

    Query params: hp (highpass Hz, 0=off), lp (lowpass Hz, default 3000)."""
    await ws.accept()

    rate    = scanner.audio_rate if scanner else AUDIO_RATE
    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=30)
    _audio_clients.append(q)
    silence = bytes(int(rate * 0.1) * 2)   # 100 ms zero PCM

    # Build ffmpeg audio filter chain
    af_parts: list[str] = []
    if 0 < max(0, min(hp, 1000)):
        af_parts.append(f'highpass=f={max(0, min(hp, 1000))}')
    if 0 < lp < 8000:
        af_parts.append(f'lowpass=f={lp}')
    af_args = ['-af', ','.join(af_parts)] if af_parts else []

    try:
        proc = await asyncio.create_subprocess_exec(
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-fflags', '+flush_packets',
            '-f', 's16le', '-ar', str(rate), '-ac', '1', '-i', 'pipe:0',
            *af_args,
            '-c:a', 'libmp3lame', '-b:a', '32k',
            '-reservoir', '0',
            '-f', 'mp3', 'pipe:1',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        try: _audio_clients.remove(q)
        except ValueError: pass
        await ws.close(1011, 'ffmpeg not found')
        return

    stop = asyncio.Event()

    async def _feed():
        try:
            proc.stdin.write(silence)
            await proc.stdin.drain()
        except (BrokenPipeError, OSError):
            stop.set(); return
        try:
            while not stop.is_set():
                try:
                    data = await asyncio.wait_for(q.get(), timeout=0.1)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    data = silence
                if stop.is_set(): break
                try:
                    proc.stdin.write(data)
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
        except Exception:
            pass
        finally:
            stop.set()
            try: proc.stdin.close()
            except Exception: pass
            try: _audio_clients.remove(q)
            except ValueError: pass

    feed_task = asyncio.create_task(_feed())

    last_chunk_t = [asyncio.get_event_loop().time()]   # mutable cell for _watchdog

    async def _stream():
        try:
            while not stop.is_set():
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                last_chunk_t[0] = asyncio.get_event_loop().time()
                try:
                    await ws.send_bytes(chunk)
                except Exception:
                    break
        finally:
            stop.set()

    stream_task = asyncio.create_task(_stream())

    async def _watchdog():
        """Stop if ffmpeg exits OR if its stdout stalls for > 5 s.
        Checking last_chunk_t catches the hung-pipe case where ffmpeg is still
        running but producing no output (e.g. encoder deadlock)."""
        STALL_SECS = 5.0
        while not stop.is_set():
            await asyncio.sleep(STALL_SECS)
            if stop.is_set():
                break
            if proc.returncode is not None:
                stop.set()
                break
            if asyncio.get_event_loop().time() - last_chunk_t[0] > STALL_SECS:
                stop.set()
                break

    watchdog_task = asyncio.create_task(_watchdog())

    try:
        while not stop.is_set():
            try:
                await asyncio.wait_for(ws.receive_bytes(), timeout=60)
            except asyncio.TimeoutError:
                pass
            except Exception:
                break
    finally:
        stop.set()
        feed_task.cancel()
        stream_task.cancel()
        watchdog_task.cancel()
        try: proc.kill()
        except Exception: pass
        await proc.wait()
        try: await ws.close()
        except Exception: pass


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await wsman.connect(ws)
    _emit({"type": "ws_clients", "count": len(wsman._clients)})

    async def _keepalive():
        # 10 s interval — tight enough that Android's WiFi power-save has less
        # opportunity to declare the connection idle between heartbeats.
        # Uses wsman.send_one() so the ping is serialised with broadcast sends
        # via the per-socket lock — prevents concurrent writes on the same WS.
        ping = json.dumps({"type": "ping"})
        while True:
            await asyncio.sleep(10)
            if not await wsman.send_one(ws, ping):
                return

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
        _emit({"type": "ws_clients", "count": len(wsman._clients)})


# ── Entry point ────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = Path(__file__).parent / "scanner_config.json"


def main():
    global scanner, _config_path

    p = argparse.ArgumentParser(description="SDR Scanner")
    p.add_argument("--config",      default=str(DEFAULT_CONFIG))
    p.add_argument("--listen-port", type=int, default=8080)
    p.add_argument("--debug",       action="store_true",
                   help="Log every chunk above threshold: freq, dB, threshold source, squelch state")
    args = p.parse_args()

    cfg: dict = {}
    config_path  = Path(args.config)
    _config_path = config_path
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"Config: {config_path}")
    else:
        print(f"Warning: {config_path} not found — using defaults")

    # Parse channels: supports "freq": "label" or "freq": {"label": "...", "squelch_rms": 0.056, "gain": "25.4", "pl": 100.0}
    raw_channels    = cfg.get("channels", {"446.000": "446.000"})
    channels        : dict[str, str]   = {}
    channel_squelch : dict[str, float] = {}
    channel_gain    : dict[str, str]   = {}
    channel_pl      : dict[str, float] = {}
    channel_bank        : dict[str, str]   = {}
    channel_modulation  : dict[str, str]   = {}
    channel_bandwidth   : dict[str, float] = {}
    channel_hp_filter   : dict[str, bool]  = {}
    for freq, val in raw_channels.items():
        if isinstance(val, dict):
            channels[freq] = val.get("label", freq)
            if "squelch_rms" in val:
                channel_squelch[freq] = float(val["squelch_rms"])
            if "gain" in val:
                channel_gain[freq] = str(val["gain"])
            if "pl" in val and float(val["pl"]) > 0.0:
                channel_pl[freq] = float(val["pl"])
            if "bank" in val and str(val["bank"]).strip():
                channel_bank[freq] = str(val["bank"]).strip()
            if "modulation" in val and str(val["modulation"]).strip():
                channel_modulation[freq] = str(val["modulation"]).strip().lower()
            if "bandwidth" in val:
                try:
                    bw = float(val["bandwidth"])
                    if bw > 0:
                        channel_bandwidth[freq] = bw
                except (TypeError, ValueError):
                    pass
            try:
                hpf_hz = float(val.get("hp_filter") or 0)
                if hpf_hz > 0:
                    channel_hp_filter[freq] = hpf_hz
            except (TypeError, ValueError):
                pass
        else:
            channels[freq] = str(val)

    # banks_enabled only stores disabled banks (absent = enabled); invert on load
    raw_banks_enabled: dict[str, bool] = cfg.get("banks_enabled", {})

    # ── Broadcastify feeder ───────────────────────────────────────────────────
    global _bcast_feeder
    bcast_cfg = cfg.get("broadcastify", {})
    if bcast_cfg.get("enabled") and bcast_cfg.get("server") and bcast_cfg.get("mountpoint"):
        _bcast_feeder = BroadcastifyFeeder(
            server      = bcast_cfg["server"],
            port        = int(bcast_cfg.get("port", 80)),
            mountpoint  = bcast_cfg["mountpoint"],
            password    = bcast_cfg.get("password", ""),
            bitrate     = int(bcast_cfg.get("bitrate", 32)),
            sample_rate = AUDIO_RATE,
            on_status   = _bcast_status_event,
        )
        print(f"[Broadcastify] Feed configured → {_bcast_feeder.url.split('@')[-1]}")
    else:
        _bcast_feeder = None

    scanner = RTLFMScanner(
        name            = cfg.get("name", "Scanner"),
        channels        = channels,
        squelch_rms     = cfg.get("squelch_rms", 0.05),
        squelch_hold    = cfg.get("squelch_hold", 2.0),
        channel_squelch = channel_squelch,
        channel_gain    = channel_gain,
        channel_pl      = channel_pl,
        channel_bank       = channel_bank,
        channel_modulation = channel_modulation,
        channel_bandwidth  = channel_bandwidth,
        channel_hp_filter  = channel_hp_filter,
        banks_enabled      = raw_banks_enabled,
        skipped         = set(cfg.get("skipped", [])),
        ppm             = cfg.get("ppm", 0),
        modulation      = cfg.get("modulation", "fm"),
        device          = cfg.get("device", "0"),
        gain            = cfg.get("gain", "auto"),
        samp_rate       = cfg.get("samp_rate", 240000),
        scan_dwell      = cfg.get("scan_dwell", 0.25),
        fir_taps        = cfg.get("fir_taps", 127),
        debug           = args.debug,
        on_event        = _emit,
        on_audio        = _audio_cb,
    )

    # SIGTERM is handled by uvicorn (which fires @app.on_event("shutdown")).
    # SIGINT from a terminal would normally raise KeyboardInterrupt through uvicorn,
    # but install a handler anyway so interactive Ctrl-C also closes the device cleanly.
    def _sigint_handler(sig, frame):
        print(f"\n[Scanner] SIGINT — closing RTL-SDR device…")
        scanner.stop_and_join(timeout=8.0)
        raise SystemExit(0)
    signal.signal(signal.SIGINT, _sigint_handler)

    print(f"Open http://<pi-ip>:{args.listen_port} in your browser")
    uvicorn.run(
        app,
        host           = "0.0.0.0",
        port           = args.listen_port,
        log_level      = "warning",
        # Disable uvicorn's protocol-level WebSocket pings.  Android WebView
        # (and some other clients) do not reliably respond to server-initiated
        # protocol pings; uvicorn then closes with 1011 after ws_ping_timeout.
        # Application-level keepalive is handled by the JS client sending a
        # JSON {type:"ping"} text frame every 15 s, which is sufficient to
        # keep the TCP connection alive through Android's WiFi power-save.
        ws_ping_interval = None,
    )


if __name__ == "__main__":
    main()
