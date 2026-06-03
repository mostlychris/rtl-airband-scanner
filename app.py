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

import json, asyncio, threading, argparse, uvicorn, struct
import numpy as np
from scipy.signal import lfilter, firwin
from pathlib import Path
from datetime import datetime
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response

VERSION    = "2.6.5"
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
      - detected_tone: the standard tone with the highest power if it is
        dominant over the average (i.e. a tone is clearly present), else None.
      - gated_open: True if no target is configured (target_hz == 0) OR
        the detected tone matches the configured target.

    Same dominant-tone logic as RTLSDR-Airband's Goertzel implementation.
    """
    n = len(buf)
    spectrum = np.abs(np.fft.rfft(buf * np.hanning(n))) ** 2
    bin_hz = sample_rate / n

    def _bin_power(freq: float) -> float:
        b = min(round(freq / bin_hz), len(spectrum) - 1)
        return float(spectrum[b])

    tone_powers = [(t, _bin_power(t)) for t in _CTCSS_TONES]
    avg_power   = sum(p for _, p in tone_powers) / len(tone_powers)
    best_tone, best_power = max(tone_powers, key=lambda x: x[1])

    detected = best_tone if best_power > avg_power else None

    if target_hz > 0.0:
        # Gating: target must be the dominant tone
        target_power = _bin_power(target_hz)
        all_excluding = [p for t, p in tone_powers if abs(t - target_hz) >= 5.0]
        all_excluding.append(target_power)
        avg_ex  = sum(all_excluding) / len(all_excluding)
        max_ex  = max(all_excluding)
        gated   = target_power >= max_ex and target_power > avg_ex
    else:
        gated = True   # no filter configured — always pass

    return gated, detected


# Silence chunk sent to audio WebSocket clients every 5 s when no signal is
# active — keeps the connection alive so the browser never needs to re-enable.
_AUDIO_KEEPALIVE = bytes(int(AUDIO_RATE * 2 * 0.1))         # 100 ms of zeros

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
  --bg:#090b0e;--card:#0d1117;--card2:#141b22;--border:#21262d;
  --panel:#05080d;--panel-b:#0d1520;
  --text:#c9d1d9;--muted:#8aa0b0;--dim:#1e2d3a;
  --green:#2dff6e;--gdim:rgba(45,255,110,.07);--gborder:rgba(45,255,110,.28);
  --amber:#ffaa00;--cyan:#00d4ff;--blue:#4d8aff;--red:#ff4455;--purple:#9966dd;--yellow:#ffcc00;
  --mono:'SF Mono','Fira Code','Consolas',monospace;
  --glow:0 0 6px var(--green),0 0 14px rgba(45,255,110,.35);
  --glow-sm:0 0 4px rgba(45,255,110,.5);
  --amber-glow:0 0 5px rgba(255,170,0,.45);
  --cyan-glow:0 0 5px rgba(0,212,255,.4);
}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px;min-height:100vh;
  background-image:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.07) 2px,rgba(0,0,0,.07) 4px)}

/* ── Header ───────────────────────────────────────────────── */
header{
  background:linear-gradient(180deg,#0e1420 0%,#090d14 100%);
  border-bottom:2px solid #141e2e;
  box-shadow:0 3px 10px rgba(0,0,0,.8),inset 0 1px 0 rgba(77,138,255,.06);
  padding:10px 20px;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:100
}
h1{font-size:12px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:#6a8aaa;display:flex;align-items:center;gap:8px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted);flex-shrink:0;transition:background .3s,box-shadow .3s}
.dot.ok{background:var(--green);box-shadow:var(--glow)}
.dot.err{background:var(--red);box-shadow:0 0 6px var(--red)}
.st{font-size:10px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase}
.spacer{flex:1}
.abtn{
  background:#0d1220;border:1px solid #1e2a3e;border-radius:3px;
  color:#5a7a9a;cursor:pointer;font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  padding:5px 14px;display:flex;align-items:center;gap:6px;transition:all .15s;
  box-shadow:0 2px 0 #040609,inset 0 1px 0 rgba(255,255,255,.03)
}
.abtn:hover{border-color:var(--green);color:var(--green);box-shadow:0 0 8px rgba(45,255,110,.18),0 2px 0 #040609}
.abtn:active{transform:translateY(1px);box-shadow:0 1px 0 #040609}
.abtn.on{border-color:rgba(45,255,110,.5);color:var(--green);background:var(--gdim)}
.asrc{font-size:10px;color:var(--cyan);letter-spacing:.08em;text-transform:uppercase;opacity:.8}

/* ── Layout ───────────────────────────────────────────────── */
.app-layout{display:flex;align-items:flex-start;max-width:1260px;margin:0 auto}
main{flex:1;min-width:0;padding:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-bottom:20px}

/* ── Audio controls sidebar ───────────────────────────────── */
.acontrols{
  width:152px;flex-shrink:0;
  display:flex;flex-direction:column;gap:16px;
  padding:16px 12px;
  background:#0b0f18;border-left:1px solid #1a2035;
  position:sticky;top:43px;height:calc(100vh - 43px);overflow-y:auto;
}
.acontrols.hidden{display:none}
.ac-title{
  font-size:8px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;
  color:var(--cyan);opacity:.6;padding-bottom:10px;border-bottom:1px solid var(--panel-b);
}
.actl{display:flex;flex-direction:column;gap:5px}
.actl label{color:var(--muted);user-select:none;letter-spacing:.1em;text-transform:uppercase;font-size:9px}
input[type=range].aslider{width:100%;accent-color:var(--amber);cursor:pointer}
select.asel{width:100%;background:#0a0e18;border:1px solid #1a2035;color:var(--text);border-radius:3px;padding:3px 6px;font-size:11px;cursor:pointer}
.atog{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--muted);cursor:pointer;user-select:none;letter-spacing:.08em;text-transform:uppercase}
.atog input[type=checkbox]{accent-color:var(--green);cursor:pointer;flex-shrink:0}
.avlbl{font-family:var(--mono);font-size:11px;color:var(--amber)}

/* ── Scanner card ─────────────────────────────────────────── */
.scard{
  background:var(--card);border:1px solid var(--border);border-radius:4px;
  overflow:hidden;cursor:pointer;transition:border-color .2s,box-shadow .2s;
  box-shadow:0 4px 16px rgba(0,0,0,.6),inset 0 1px 0 rgba(255,255,255,.02)
}
.scard:hover{border-color:#2a3448}
.scard.playing{border-color:rgba(45,255,110,.45);box-shadow:0 0 18px rgba(45,255,110,.1),0 4px 16px rgba(0,0,0,.6)}
.scard.locked{border-color:rgba(77,138,255,.45);box-shadow:0 0 18px rgba(77,138,255,.12),0 4px 16px rgba(0,0,0,.6)}

/* ── Panel header (stream name + status) ─────────────────── */
.sc-panel-hdr{
  display:flex;align-items:center;gap:8px;padding:7px 12px 6px;
  background:linear-gradient(180deg,#111826 0%,#0a1018 100%);
  border-bottom:1px solid #141e30;
}
.sc-name{font-size:10px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:#b0c4d8;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sc-status{display:flex;align-items:center;gap:5px;font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;flex-shrink:0}
.sc-led{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.sc-status.ok{color:var(--green)}.sc-status.ok .sc-led{background:var(--green);box-shadow:var(--glow-sm)}
.sc-status.err{color:var(--red)}.sc-status.err .sc-led{background:var(--red);box-shadow:0 0 5px var(--red)}
.sc-status.warn{color:var(--yellow)}.sc-status.warn .sc-led{background:var(--yellow)}
.serr{font-size:10px;color:var(--red);padding:5px 12px;background:rgba(255,68,85,.07);border-bottom:1px solid rgba(255,68,85,.15);letter-spacing:.06em}

/* ── Frequency display (LCD panel) ───────────────────────── */
.sc-display{
  background:var(--panel);
  padding:12px 14px 10px;
  border-bottom:1px solid var(--panel-b);
  min-height:70px;
}
.sc-lbl{
  font-family:var(--mono);font-size:26px;font-weight:600;
  letter-spacing:.04em;line-height:1;
  color:var(--dim);text-transform:uppercase;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  transition:color .4s,text-shadow .4s;
}
.sc-display.active .sc-lbl{color:var(--green);text-shadow:var(--glow)}
.sc-meta{display:flex;align-items:baseline;gap:10px;margin-top:6px}
.sc-freq{
  font-family:var(--mono);font-size:13px;font-weight:500;
  letter-spacing:.06em;color:var(--dim);
  transition:color .3s,text-shadow .3s;flex:1;
}
.sc-display.active .sc-freq{color:var(--amber);text-shadow:var(--amber-glow)}
.sc-unit{font-size:11px;color:var(--muted);margin-left:3px}
.sc-timer{font-family:var(--mono);font-size:11px;color:var(--dim);transition:color .3s;flex-shrink:0}
.sc-display.active .sc-timer{color:var(--cyan);opacity:.8}
.sc-pl{font-family:var(--mono);font-size:10px;color:var(--purple);letter-spacing:.08em;flex-shrink:0;opacity:.85}
.sc-ctcss{font-family:var(--mono);font-size:10px;letter-spacing:.08em;flex-shrink:0;transition:color .3s}
.sc-ctcss.info{color:var(--cyan);opacity:.85}
.sc-ctcss.match{color:var(--green);text-shadow:var(--glow-sm)}

/* ── Signal meter (segmented LED bar) ─────────────────────── */
.sqbar{
  height:8px;background:var(--panel);padding:0 14px;margin-bottom:0;
  display:flex;align-items:center;border-bottom:1px solid var(--panel-b);
}
.sqfill-wrap{flex:1;height:4px;background:var(--dim);border-radius:2px;overflow:hidden;position:relative}
.sqfill{
  height:100%;width:0%;
  background:linear-gradient(90deg,#14a80a 0%,#39ff14 55%,#ccff00 75%,#ffcc00 88%,#ff5000 100%);
  transition:width .12s linear;
  position:relative;
}
.sqfill::after{
  content:'';position:absolute;inset:0;
  background-image:repeating-linear-gradient(90deg,transparent 0,transparent 5px,var(--panel) 5px,var(--panel) 7px);
}
.sqfill.active{box-shadow:0 0 4px rgba(57,255,20,.6)}
.sq-label{font-size:8px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-left:8px;flex-shrink:0}

/* ── Action buttons (SKIP / EDIT / DEL) ─────────────────── */
.sc-acts{
  display:flex;gap:5px;padding:8px 10px;
  background:#0c1018;border-bottom:1px solid #1a2035;
}
.sc-btn{
  flex:1;background:#0a0e18;border:1px solid #1e2a3e;border-radius:3px;
  color:#4a5a6a;cursor:pointer;font-size:9px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  padding:5px 2px;text-align:center;
  transition:all .1s;
  box-shadow:0 2px 0 #040609,inset 0 1px 0 rgba(255,255,255,.02);
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
  color:#5a7a9a;border-bottom:1px solid #141e30;background:#0a1220;
  cursor:pointer;user-select:none;display:flex;align-items:center;justify-content:space-between;
}
.sc-chl-hdr:hover{color:var(--cyan)}
.coll-arrow{font-size:8px;opacity:.5}
.ch{display:flex;align-items:center;gap:8px;padding:4px 12px;transition:background .15s;cursor:default}
.ch:hover{background:var(--card2)}
.ch.active{background:var(--gdim);border-left:2px solid var(--green);padding-left:10px}
.ch.active:hover{background:rgba(45,255,110,.1)}
.ch-dot{font-size:9px;color:var(--dim);width:10px;flex-shrink:0;transition:color .2s}
.ch.active .ch-dot{color:var(--green);text-shadow:var(--glow-sm)}
.ch-f{font-family:var(--mono);font-size:12px;font-weight:600;width:74px;flex-shrink:0;color:#3a5a6a;transition:color .2s}
.ch.active .ch-f{color:var(--amber);text-shadow:var(--amber-glow)}
.ch-l{color:var(--muted);font-size:11px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;letter-spacing:.02em}
.ch.active .ch-l{color:var(--text)}
.ch-t{font-size:10px;color:var(--dim);font-family:var(--mono);flex-shrink:0;transition:color .2s}
.ch.active .ch-t{color:var(--cyan);opacity:.7}
.noch{padding:12px;color:var(--dim);font-size:10px;text-align:center;letter-spacing:.15em;text-transform:uppercase}
.ch-acts{display:flex;gap:2px;margin-left:auto;flex-shrink:0;visibility:hidden}
.ch:hover .ch-acts,.ch.skipped .ch-acts{visibility:visible}
.ch-icon{background:none;border:none;cursor:pointer;padding:1px 5px;font-size:10px;color:var(--muted);border-radius:2px;line-height:1}
.ch-icon:hover{background:var(--card2);color:var(--text)}
.ch-icon.del:hover{color:var(--red)}
.ch-icon.skip:hover{color:var(--amber)}
.ch.skipped .ch-f,.ch.skipped .ch-l,.ch.skipped .ch-t,.ch.skipped .ch-dot{opacity:.2}
.ch.skipped .ch-icon.skip{color:var(--amber)}
.ch-edit-row{display:flex;align-items:center;gap:6px;padding:5px 12px;flex-wrap:wrap}
.ch-edit-in{background:#080c16;border:1px solid #1e2a3e;color:var(--text);border-radius:3px;padding:2px 6px;font-size:11px;font-family:var(--mono);min-width:0}
.ch-edit-lbl{flex:1}.ch-edit-sq{width:64px}.ch-edit-gn{width:72px}.ch-edit-pl{width:64px}
.ch-pl{font-size:9px;color:var(--purple);letter-spacing:.05em;flex-shrink:0;opacity:.8}
.ch-save{background:rgba(45,255,110,.08);border:1px solid rgba(45,255,110,.3);border-radius:3px;color:var(--green);cursor:pointer;font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:2px 10px;white-space:nowrap}
.ch-cancel{background:none;border:1px solid #1e2a3e;border-radius:3px;color:var(--muted);cursor:pointer;font-size:9px;letter-spacing:.06em;padding:2px 8px}
.ch-add-btn{display:flex;align-items:center;gap:5px;padding:5px 12px;font-size:10px;color:var(--muted);cursor:pointer;border-top:1px solid #141e30;transition:color .15s;letter-spacing:.1em;text-transform:uppercase}
.ch-add-btn:hover{color:var(--cyan)}

/* ── Activity log ─────────────────────────────────────────── */
.acard{background:var(--card);border:1px solid var(--border);border-radius:4px;overflow:hidden}
.ahdr{
  padding:7px 12px;border-bottom:1px solid #1a1030;font-size:8px;font-weight:700;
  color:#7a5a9a;letter-spacing:.2em;text-transform:uppercase;background:#100c1e;
  cursor:pointer;user-select:none;display:flex;align-items:center;justify-content:space-between;
}
.ahdr:hover{color:var(--purple)}
.arow{display:flex;align-items:center;gap:12px;padding:5px 12px;border-bottom:1px solid var(--panel-b);font-size:11px}
.arow:last-child{border-bottom:none}
.at{font-family:var(--mono);color:var(--cyan);width:58px;flex-shrink:0;opacity:.7}
.as{color:var(--muted);width:90px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:10px;letter-spacing:.04em;text-transform:uppercase}
.af{font-family:var(--mono);font-weight:600;width:82px;flex-shrink:0;color:var(--amber)}
.al{color:var(--text);flex:1;font-size:11px;opacity:.8}

/* ── Overlay ──────────────────────────────────────────────── */
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.88);display:flex;align-items:center;justify-content:center;z-index:200}
.overlay.hidden{display:none}
.obox{background:var(--card);border:1px solid #1e2a3e;border-radius:6px;padding:28px 36px;text-align:center;max-width:400px;box-shadow:0 0 40px rgba(45,255,110,.06)}
.obox h2{font-size:14px;margin-bottom:8px;color:var(--green);letter-spacing:.15em;text-transform:uppercase;text-shadow:var(--glow-sm)}
.obox p{color:var(--muted);font-size:12px;margin-bottom:22px;line-height:1.7}
.obtn{background:rgba(45,255,110,.08);border:1px solid rgba(45,255,110,.3);border-radius:3px;color:var(--green);cursor:pointer;font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;padding:9px 24px;margin:4px;transition:all .15s}
.obtn:hover{background:rgba(45,255,110,.16);box-shadow:0 0 12px rgba(45,255,110,.2)}
.obtn.skip{background:#0a0e18;border:1px solid #1e2a3e;color:var(--muted);font-weight:400;letter-spacing:.06em}

/* ── Animations ───────────────────────────────────────────── */
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.blink{animation:pulse 1.2s ease-in-out infinite}

/* ── Responsive ────────────────────────────────────────────── */

/* Narrow desktop / large tablet: tighten sidebar */
@media (max-width:900px){
  .acontrols{width:128px;padding:12px 10px;gap:12px}
}

/* Tablet / mobile: sidebar flips to horizontal bar above content */
@media (max-width:700px){
  .app-layout{flex-direction:column;align-items:stretch}
  main{padding:10px}
  .grid{grid-template-columns:1fr;gap:10px;margin-bottom:10px}

  /* Sidebar becomes a compact controls bar above the grid */
  .acontrols{
    width:100%;order:-1;
    flex-direction:row;flex-wrap:wrap;align-items:center;gap:8px 16px;
    position:static;height:auto;overflow:visible;
    border-left:none;border-bottom:1px solid #1a2035;
    padding:8px 14px
  }
  .ac-title{display:none}
  .actl{flex-direction:row;align-items:center;gap:6px}
  .actl label{font-size:8px;white-space:nowrap}
  input[type=range].aslider{width:72px}
  select.asel{width:auto;min-width:80px}
  .atog{font-size:9px;white-space:nowrap}
  .avlbl{min-width:26px}

  /* Scanner display: scale down primary text */
  .sc-lbl{font-size:20px}

  /* Larger touch targets for buttons */
  .sc-btn{padding:8px 4px;font-size:8px}
  .ch{padding:8px 12px}

  /* Always show per-row actions (no hover on touch screens) */
  .ch-acts{visibility:visible}

  /* Header: drop status text, keep dot indicator */
  header{padding:8px 14px;gap:8px}
  .st{display:none}
  h1{font-size:11px;letter-spacing:.14em}
}

/* Small phones */
@media (max-width:420px){
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
  <div class="spacer"></div>
  <span class="asrc" id="asrc"></span>
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
  <div class="ac-title">Audio Controls</div>
  <div class="actl">
    <label for="aVol">Volume</label>
    <input type="range" class="aslider" id="aVol" min="0" max="150" value="100" oninput="setVol(this.value)">
    <span class="avlbl" id="aVolLbl">100%</span>
  </div>
  <div class="actl">
    <label for="aHP">HP Filter</label>
    <select class="asel" id="aHP" onchange="setHP(this.value)">
      <option value="0">Off</option>
      <option value="100">100 Hz</option>
      <option value="300">300 Hz (PL)</option>
    </select>
  </div>
  <div class="actl">
    <label for="aLP">LP Filter</label>
    <input type="range" class="aslider" id="aLP" min="2000" max="8000" step="500" value="3000" oninput="setLP(this.value)">
    <span class="avlbl" id="aLPLbl">3.0 kHz</span>
  </div>
  <div class="actl">
    <label class="atog">
      <input type="checkbox" id="aSqTail" onchange="setSqTail(this.checked)">
      SQ Tail Cut
    </label>
  </div>
</div>
</div>
<div class="overlay hidden" id="overlay">
  <div class="obox">
    <h2>◼ Enable Audio</h2>
    <p>Audio is demodulated on the device and streamed as PCM over WebSocket.<br>
       Latency is typically under 1 second.<br>
       Click a stream panel to lock audio to that stream.</p>
    <button class="obtn" onclick="enableAudio()">▶ Enable Audio</button>
    <button class="obtn skip" onclick="closeOverlay()">Display Only</button>
  </div>
</div>
<script>
// ── State ──────────────────────────────────────────────────────────────────────
const S = { streams:{}, playing:null, audioOn:(localStorage.getItem('a_on')==='true'), locked:null };
let ws, wsRetry=0;
let actItems = [];
let _editFreq    = null;   // freq string currently open in edit mode, or null
let _addingCh    = false;  // whether the add-channel form is shown
const _chCollapsed = {};   // { [mount]: bool } — per-stream channel bank collapse state
let _actCollapsed  = true;
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Audio (HTTP stream → HTMLAudioElement → Web Audio filter chain) ────────────
//
// The browser fetches /stream (a never-ending WAV) via a plain <audio> element.
// Android's media system treats HTMLAudioElement playback as native media, which
// keeps the renderer alive and WebSockets open through screen lock — the same
// mechanism used by apps like Rdio Scanner.
//
// createMediaElementSource() routes the element's output through the Web Audio
// graph so the HP/LP filters, volume, and squelch-tail gate all still apply.
// Graph: <audio src=/stream> → MediaElementSource → HP → LP → Vol → Gate → out

// ── Audio settings (persisted in localStorage) ────────────────────────────────
const A = {
  vol:    parseFloat(localStorage.getItem('a_vol')    ?? '1'),
  hp:     parseInt(  localStorage.getItem('a_hp')     ?? '0',   10),
  lp:     parseInt(  localStorage.getItem('a_lp')     ?? '3000', 10),
  sqtail: (localStorage.getItem('a_sqtail') ?? 'false') === 'true',
};
let _sqActive = true;   // tracks last squelch state to drive gate transitions
let _gateRaf  = null;   // requestAnimationFrame handle for squelch-tail fade
let _gateGain = 1.0;    // 0–1 multiplier applied on top of A.vol
let audMount  = null;
let _audEl    = null;   // HTMLAudioElement — plays /stream natively (no AudioContext
                        // interception) so Android's OS media system sees real audio

function _applyVolume() {
  if (_audEl) _audEl.volume = Math.min(1, Math.max(0, A.vol * _gateGain));
}

function _setGate(open) {
  if (_gateRaf) { cancelAnimationFrame(_gateRaf); _gateRaf = null; }
  if (open) {
    _gateGain = 1.0;
    _applyVolume();
  } else {
    // Ramp volume to 0 over ~50 ms to mute squelch-tail noise
    const start   = performance.now();
    const startGain = _gateGain;
    const RAMP_MS = 50;
    const step = ts => {
      _gateGain = Math.max(0, startGain * (1 - (ts - start) / RAMP_MS));
      _applyVolume();
      if (_gateGain > 0) _gateRaf = requestAnimationFrame(step);
      else _gateRaf = null;
    };
    _gateRaf = requestAnimationFrame(step);
  }
}

function _initAudEl() {
  if (_audEl) return;
  _audEl = new Audio();
  // Android's Doze mode blocks WAV streams after ~5 min of silence.
  // MP3 via /stream.mp3 is treated as native radio media, surviving Doze
  // with a proper foreground service.  PC browsers keep the WAV stream.
  _audEl.src = /Android/i.test(navigator.userAgent) ? '/stream.mp3' : '/stream';
  _audEl.volume = A.vol;

  let _retries   = 0;
  let _stallTimer = null;

  _audEl.addEventListener('playing', () => {
    _retries = 0;
    clearTimeout(_stallTimer);
  });

  // Exponential backoff on stream errors so a persistently broken stream
  // (e.g. ffmpeg not installed) doesn't spam retries or the overlay.
  _audEl.onerror = () => {
    if (!S.audioOn) return;
    _retries++;
    const delay = Math.min(1000 * _retries, 30000);
    console.warn(`[audio] stream error — retry ${_retries} in ${delay} ms`);
    setTimeout(_reloadStream, delay);
  };

  // Stalled: give the browser 4 s to recover before forcing a reload.
  _audEl.addEventListener('stalled', () => {
    if (!S.audioOn) return;
    clearTimeout(_stallTimer);
    _stallTimer = setTimeout(_reloadStream, 4000);
  });
}

function _reloadStream() {
  if (!_audEl || !S.audioOn) return;
  _audEl.load();
  // Automatic retry — do NOT show the overlay.  If the stream is broken
  // (e.g. ffmpeg missing) the user will see the spinner; they can tap the
  // audio button once to retry manually.  The overlay only appears when the
  // user explicitly tries to enable audio and the browser blocks play().
  _audEl.play().catch(() => updateAudioUI());
}

function openAudioStream(mount) {
  audMount  = mount;
  _sqActive = true;
  _gateGain = 1.0;
  _initAudEl();
  _applyVolume();
  if (_audEl.paused) {
    _audEl.play().catch(() => {
      document.getElementById('overlay').classList.remove('hidden');
    });
  }
  _updateMediaSession(true);
  updateAudioUI();
}

function closeAudio() {
  if (_audEl) _audEl.pause();
  audMount = null;
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
function connect() {
  const _wsp = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(_wsp + '//' + location.host + '/ws');
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
      s.channelSquelch = s.channelSquelch || {};
      s.channelGain    = s.channelGain    || {};
      s.channelPL      = s.channelPL      || {};
      s.skipped        = s.skipped        || [];
      s.holdFreq       = s.holdFreq       || null;
      S.streams[s.mount] = s;
      (s.history || []).forEach(h => pushActivity(s.name, h.freq, h.label, h.time));
    });
    renderAll();
    autoSelect();
    // Resume audio without overlay if the user had it enabled before the refresh
    if (S.audioOn && !audMount) {
      const target = S.locked || S.playing || (Object.values(S.streams).find(s=>s.connected)||{}).mount;
      if (target) switchAudio(target);
      updateAudioUI();
    }
  } else if (m.type === 'freq_change') {
    const s = S.streams[m.mount]; if (!s) return;
    s.activeFreq    = m.freq;
    s.activeSince   = m.time;
    s.lastError     = null;
    s.detectedCTCSS = null;   // clear on each new frequency
    updateCard(m.mount, true);
    pushActivity(m.name, m.freq, m.label, m.time);
    if (S.audioOn && (!S.locked || S.locked === m.mount)) switchAudio(m.mount);
  } else if (m.type === 'freq_clear') {
    const s = S.streams[m.mount]; if (!s) return;
    s.activeFreq    = null;
    s.activeSince   = null;
    s.detectedCTCSS = null;
    updateCard(m.mount, true);
  } else if (m.type === 'signal') {
    const d = document.getElementById('sqfill_' + eid(m.mount));
    if (d) {
      const pct = m.active ? Math.min(100, Math.max(0, (m.db + 60) * 100 / 40)) : 0;
      d.style.width = pct + '%';
      d.className = 'sqfill' + (m.active ? ' active' : '');
    }
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
    // Squelch tail suppression: gate audio closed on signal drop, open on signal open.
    // Only applied to the currently playing mount so background activity is ignored.
    if (A.sqtail && m.mount === (audMount || 'sdr') && m.active !== _sqActive) {
      _setGate(m.active);
      _sqActive = m.active;
    }
  } else if (m.type === 'channels_update') {
    const s = S.streams[m.mount]; if (!s) return;
    s.channels       = m.channels;
    s.channelSquelch = m.channelSquelch;
    s.channelGain    = m.channelGain    || {};
    s.channelPL      = m.channelPL      || {};
    s.defaultSquelch = m.defaultSquelch;
    s.defaultGain    = m.defaultGain;
    s.skipped        = m.skipped || [];
    _editFreq = null; _addingCh = false;
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
  }
}

// ── Render ─────────────────────────────────────────────────────────────────────
function renderAll() {
  const g = document.getElementById('grid'); g.innerHTML = '';
  Object.values(S.streams).forEach(s => {
    const d = document.createElement('div');
    d.className = cardClass(s);
    d.id = 'sc' + eid(s.mount);
    d.onclick = e => { if (!e.target.closest('.chlist')) lockTo(s.mount); };
    d.innerHTML = cardHtml(s);
    g.appendChild(d);
  });
}
function updateCard(mount, skipIfEditing) {
  const d = document.getElementById('sc' + eid(mount)); if (!d) return;
  // Don't blow away an active edit row on routine scan events (freq changes, etc.)
  if (skipIfEditing && (_editFreq !== null || _addingCh)) return;
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
  const chs    = s.channels || {};
  const csq    = s.channelSquelch || {};
  const cgain  = s.channelGain    || {};
  const cpl    = s.channelPL      || {};
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
  const rxBadge = (audMount===s.mount && S.audioOn)
    ? ' <span class="blink" style="color:var(--green);font-size:9px;letter-spacing:.1em">▶ RX</span>' : '';
  const lockBadge = S.locked===s.mount
    ? ' <span style="font-size:9px;color:var(--blue);letter-spacing:.1em">⬡ LOCK</span>' : '';
  const holdBadge = isHeld
    ? ' <span style="font-size:9px;color:var(--amber);letter-spacing:.1em">⏸ HOLD</span>' : '';
  const connStatus = s.connected
    ? '<span class="sc-status ok"><span class="sc-led"></span>' + (isHeld ? 'HOLD' : 'SCANNING') + '</span>'
    : '<span class="sc-status ' + (s.lastError ? 'err' : 'warn') + '"><span class="sc-led"></span>' + (s.lastError ? 'ERROR' : 'OPENING') + '</span>';
  const errHtml = s.lastError
    ? '<div class="serr">⚠ ' + escHtml(s.lastError) + '</div>' : '';

  // Action buttons (always rendered; disabled when no active freq)
  const noAf   = !af;
  const afSkp  = af && skpSet.has(af);
  // HOLD target: active freq if available, otherwise the currently held freq (to release)
  const holdTarget = af || heldF;
  const actsHtml = '<div class="sc-acts' + (noAf && !isHeld ? ' idle' : '') + '">'
    + '<button class="sc-btn skip' + (afSkp?' active':'') + '" onclick="event.stopPropagation();' + (af?'skipChannel(\''+af+'\')':'') + '" title="' + (afSkp?'Resume scan':'Skip channel') + '">'
    + (afSkp ? '▶ SCAN' : '⊘ SKIP') + '</button>'
    + '<button class="sc-btn hold' + (isHeld?' active':'') + '" onclick="event.stopPropagation();' + (holdTarget?'holdChannel(\''+holdTarget+'\')':'') + '" title="' + (isHeld?'Release hold — resume scanning':'Hold on current frequency') + '">'
    + (isHeld ? '⏹ HELD' : '⏸ HOLD') + '</button>'
    + (af && !isHeld ? '<button class="sc-btn resume" onclick="event.stopPropagation();resumeScan()" title="Skip to next frequency now">▶▶ NEXT</button>' : '')
    + '<button class="sc-btn edit" onclick="event.stopPropagation();' + (af?'editChannel(\''+af+'\')':'') + '" title="Edit label/squelch">✏ EDIT</button>'
    + '<button class="sc-btn del" onclick="event.stopPropagation();' + (af?'deleteChannel(\''+af+'\')':'') + '" title="Remove channel">✕ DEL</button>'
    + '</div>';

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
      const t    = act && s.activeSince ? new Date(s.activeSince).toLocaleTimeString() : '';
      if (f === _editFreq) {
        rows += '<div class="ch-edit-row" onclick="event.stopPropagation()">'
          + '<span class="ch-f" style="font-size:10px;color:var(--muted);flex-shrink:0">' + f + '</span>'
          + '<input class="ch-edit-in ch-edit-lbl" id="ch-edit-label" value="' + escHtml(lbl !== f ? lbl : '') + '" placeholder="Label">'
          + '<input class="ch-edit-in ch-edit-sq" id="ch-edit-sq" type="number" value="' + sq + '" step="0.001" min="0.001" max="0.5" title="Squelch RMS">'
          + '<input class="ch-edit-in ch-edit-gn" id="ch-edit-gain" value="' + escHtml(gn) + '" placeholder="Gain (auto or dB)" title="RF gain: auto, or tenths-of-dB (e.g. 25.4)">'
          + '<input class="ch-edit-in ch-edit-pl" id="ch-edit-pl" type="number" value="' + (pl || '') + '" step="0.1" min="0" placeholder="PL Hz (0=off)" title="CTCSS tone Hz, 0 or blank to disable">'
          + '<button class="ch-save" onclick="saveChannel(\'' + f + '\')">SAVE</button>'
          + '<button class="ch-cancel" onclick="cancelEdit()">✕</button>'
          + '</div>';
      } else {
        rows += '<div class="ch' + (act?' active':'') + (skp?' skipped':'') + '">'
          + '<span class="ch-dot">' + (skp ? '─' : act ? '◉' : '○') + '</span>'
          + '<span class="ch-f">' + f + '</span>'
          + '<span class="ch-l">' + escHtml(lbl!==f?lbl:'') + '</span>'
          + (pl ? '<span class="ch-pl">PL ' + pl + '</span>' : '')
          + '<span class="ch-t">' + t + '</span>'
          + '<div class="ch-acts">'
          + '<button class="ch-icon skip' + (skp?' skipped':'') + '" onclick="event.stopPropagation();skipChannel(\'' + f + '\')" title="' + (skp?'Include in scan':'Skip frequency') + '">' + (skp?'▶':'⊘') + '</button>'
          + '<button class="ch-icon" onclick="event.stopPropagation();editChannel(\'' + f + '\')" title="Edit">✏</button>'
          + '<button class="ch-icon del" onclick="event.stopPropagation();deleteChannel(\'' + f + '\')" title="Remove">✕</button>'
          + '</div></div>';
      }
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

  let addArea = '';
  if (_addingCh) {
    addArea = '<div class="ch-edit-row" onclick="event.stopPropagation()">'
      + '<input class="ch-edit-in" id="ch-add-freq" placeholder="MHz" style="width:60px">'
      + '<input class="ch-edit-in ch-edit-lbl" id="ch-add-label" placeholder="Label">'
      + '<input class="ch-edit-in ch-edit-sq" id="ch-add-sq" type="number" value="' + defSq + '" step="0.001" min="0.001" max="0.5" title="Squelch RMS">'
      + '<input class="ch-edit-in ch-edit-gn" id="ch-add-gain" value="' + escHtml(defGn) + '" placeholder="Gain" title="RF gain: auto or dB">'
      + '<input class="ch-edit-in ch-edit-pl" id="ch-add-pl" type="number" step="0.1" min="0" placeholder="PL Hz (0=off)" title="CTCSS tone Hz, 0 or blank to disable">'
      + '<button class="ch-save" onclick="addChannel()">ADD</button>'
      + '<button class="ch-cancel" onclick="cancelEdit()">✕</button>'
      + '</div>';
  } else {
    addArea = '<div class="ch-add-btn" onclick="event.stopPropagation();showAddChannel()">＋ Add Frequency</div>';
  }

  // Show freq below label when active (rawLbl case), or when holding silently (holdLbl case)
  const metaFreq    = (af && rawLbl) ? af : (!af && heldF && holdLbl) ? heldF : null;
  const activePL    = af && cpl[af] ? cpl[af] : (heldF && cpl[heldF] ? cpl[heldF] : null);
  const initCTCSS   = s.detectedCTCSS || null;
  const ctcssMatch  = initCTCSS && activePL;
  const metaHtml = (metaFreq || since || activePL || true)   // always render for in-place updates
    ? '<div class="sc-meta">'
      + (metaFreq ? '<span class="sc-freq">' + metaFreq + '<span class="sc-unit">MHz</span></span>' : '')
      + (activePL ? '<span class="sc-pl">PL ' + activePL + ' Hz</span>' : '')
      + '<span class="sc-ctcss' + (initCTCSS ? (ctcssMatch ? ' match' : ' info') : '') + '" id="sc_ctcss_' + eid(s.mount) + '">'
      + (initCTCSS ? '◈ ' + initCTCSS + ' Hz' : '') + '</span>'
      + (since ? '<span class="sc-timer">' + since + '</span>' : '')
      + '</div>'
    : '';

  const collapsed = _chCollapsed[s.mount] !== false;
  const chBankHtml = '<div class="sc-chl-hdr" onclick="event.stopPropagation();toggleChBank(' + escHtml(JSON.stringify(s.mount)) + ')">'
    + 'Channel Bank<span class="coll-arrow">' + (collapsed ? '▶' : '▼') + '</span></div>'
    + (collapsed ? '' : rows + addArea);

  return '<div class="sc-panel-hdr"><span class="sc-name">' + escHtml(s.name) + rxBadge + lockBadge + holdBadge + '</span>' + connStatus + '</div>'
    + errHtml
    + '<div class="sc-display' + (isRx ? ' active' : '') + '">'
    + '<div class="sc-lbl">' + escHtml(primary) + '</div>'
    + metaHtml
    + '</div>'
    + '<div class="sqbar"><div class="sqfill-wrap"><div class="sqfill' + (isRx?' active':'') + '" id="sqfill_' + eid(s.mount) + '"></div></div><span class="sq-label">SIG</span></div>'
    + actsHtml
    + '<div class="chlist">' + chBankHtml + '</div>';
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
  else { S.audioOn=false; localStorage.setItem('a_on','false'); closeAudio(); updateAudioUI(); Object.keys(S.streams).forEach(m=>updateCard(m)); }
}
function enableAudio() {
  S.audioOn = true; localStorage.setItem('a_on','true'); closeOverlay();
  // Always start the stream element right here — this is a gesture context,
  // so play() is guaranteed to succeed regardless of whether S.streams is
  // populated yet.  openAudioStream (called when state arrives) will no-op
  // since the element is already playing.
  _initAudEl();
  _gateGain = 1.0; _applyVolume();
  _audEl.load();   // clear any error state before playing
  _audEl.play().catch(() => {});
  audMount = audMount
    || S.locked || S.playing
    || (Object.values(S.streams).find(s => s.connected) || {}).mount
    || 'sdr';
  _updateMediaSession(true);
  updateAudioUI();
  Object.keys(S.streams).forEach(m => updateCard(m));
}
function closeOverlay() { document.getElementById('overlay').classList.add('hidden'); }
function updateAudioUI() {
  document.getElementById('acontrols').classList.toggle('hidden', !S.audioOn);
  const btn = document.getElementById('abtn');
  const src = document.getElementById('asrc');
  const connected = _audEl && !_audEl.paused;
  if (S.audioOn) {
    const s = audMount ? S.streams[audMount] : null;
    btn.className = 'abtn on';
    document.getElementById('aico').textContent = (audMount && connected) ? '▶' : '…';
    document.getElementById('albl').textContent  = S.locked ? 'Audio Lock' : 'Audio Auto';
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
  _chCollapsed[mount] = !_chCollapsed[mount];
  updateCard(mount);
}
function toggleActLog() {
  _actCollapsed = !_actCollapsed;
  document.getElementById('actlist').style.display = _actCollapsed ? 'none' : '';
  const arrow = document.getElementById('actArrow');
  if (arrow) arrow.textContent = _actCollapsed ? '▶' : '▼';
}

// ── Channel management ─────────────────────────────────────────────────────────
function editChannel(freq) {
  _editFreq = freq; _addingCh = false;
  Object.keys(S.streams).forEach(m => updateCard(m));
}
function cancelEdit() {
  _editFreq = null; _addingCh = false;
  Object.keys(S.streams).forEach(m => updateCard(m));
}
function saveChannel(freq) {
  const label = (document.getElementById('ch-edit-label').value || '').trim();
  const sq    = parseFloat(document.getElementById('ch-edit-sq').value);
  const plEl  = document.getElementById('ch-edit-pl');
  const pl    = plEl ? parseFloat(plEl.value) : NaN;
  const gn    = (document.getElementById('ch-edit-gain').value || '').trim();
  fetch('/api/channel', {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ freq, label: label || freq, squelch_rms: isNaN(sq) ? null : sq, gain: gn, pl: isNaN(pl) ? 0 : pl }),
  }).catch(e => console.error('[api]', e));
}
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
function showAddChannel() {
  _editFreq = null; _addingCh = true;
  Object.keys(S.streams).forEach(m => updateCard(m));
  setTimeout(() => { const el = document.getElementById('ch-add-freq'); if (el) el.focus(); }, 50);
}
function addChannel() {
  const freq  = (document.getElementById('ch-add-freq').value  || '').trim();
  const label = (document.getElementById('ch-add-label').value || '').trim();
  const sq    = parseFloat(document.getElementById('ch-add-sq').value);
  const gn    = (document.getElementById('ch-add-gain').value  || '').trim();
  const pl    = parseFloat(document.getElementById('ch-add-pl').value);
  if (!freq) return;
  fetch('/api/channel', {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ freq, label: label || freq, squelch_rms: isNaN(sq) ? null : sq, gain: gn, pl: isNaN(pl) ? 0 : pl }),
  }).catch(e => console.error('[api]', e));
}

// ── Audio filter / volume controls ────────────────────────────────────────────
function setVol(v) {
  A.vol = v / 100;
  localStorage.setItem('a_vol', A.vol);
  document.getElementById('aVolLbl').textContent = v + '%';
  _applyVolume();
}
function setHP(v) {
  A.hp = parseInt(v, 10);
  localStorage.setItem('a_hp', A.hp);
  // HP filter unavailable without AudioContext interception; setting persisted only.
}
function setLP(v) {
  A.lp = parseInt(v, 10);
  localStorage.setItem('a_lp', A.lp);
  document.getElementById('aLPLbl').textContent = (A.lp / 1000).toFixed(1) + ' kHz';
  // LP filter unavailable without AudioContext interception; setting persisted only.
}
function setSqTail(v) {
  A.sqtail = v;
  localStorage.setItem('a_sqtail', v);
  if (!v) { _gateGain = 1.0; _applyVolume(); }
}
function initControls() {
  const vol = Math.round(A.vol * 100);
  document.getElementById('aVol').value = vol;
  document.getElementById('aVolLbl').textContent = vol + '%';
  document.getElementById('aLP').value = A.lp;
  document.getElementById('aLPLbl').textContent = (A.lp / 1000).toFixed(1) + ' kHz';
  document.getElementById('aHP').value = String(A.hp);
  document.getElementById('aSqTail').checked = A.sqtail;
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
    _audEl.play().catch(() => {
      document.getElementById('overlay').classList.remove('hidden');
    });
  }
});
if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(() => {});
_initAudEl();   // pre-create the element so the /stream connection opens early
connect();
initControls();
if (!S.audioOn) setTimeout(() => document.getElementById('overlay').classList.remove('hidden'), 900);
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
                 skipped: set[str] | None = None,
                 ppm: int = 0, modulation: str = "fm",
                 device: str = "0", gain: str = "auto",
                 samp_rate: int = 240000, scan_dwell: float = 0.5,
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
        self.channel_pl      = channel_pl      or {}  # per-freq CTCSS tone (Hz); 0.0 or absent = disabled
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
        self._on_event  = on_event
        self._on_audio  = on_audio

        self._running      = False
        self._lock         = threading.Lock()
        self._active_freq  = None
        self._active_since = None
        self._history      = deque(maxlen=20)
        self.hold_freq: str | None = None   # non-None = scanner locked to this frequency
        self.connected     = False
        self.last_error: str | None = None
        self._resume_event = threading.Event()  # set to force immediate scan advance

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

    def set_channel(self, freq: str, label: str,
                    squelch_rms: float | None = None,
                    gain: str | None = None,
                    pl: float | None = None) -> None:
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

    def remove_channel(self, freq: str) -> None:
        with self._lock:
            self.channels.pop(freq, None)
            self.channel_squelch.pop(freq, None)
            self.channel_gain.pop(freq, None)
            self.channel_pl.pop(freq, None)
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
                return True

    def resume_scan(self) -> None:
        """Force the scan loop to advance to the next frequency immediately."""
        self._resume_event.set()

    def _emit(self, evt: dict):
        if self._on_event: self._on_event(evt)

    def _emit_audio(self, pcm: bytes):
        if self._on_audio: self._on_audio(pcm)

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
                    freq_keys = [f for f in self.channels if f not in self.skipped]
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
                    label     = self.channels.get(freq_str, freq_str)
                    threshold = self.channel_squelch.get(freq_str, self.squelch_rms)
                    eff_gain  = self.channel_gain.get(freq_str, self.gain)
                    pl_tone   = self.channel_pl.get(freq_str, 0.0)
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
                _last_dbg_state = None
                ctcss_buf: list      = []    # accumulation buffer for CTCSS detection
                ctcss_detected: bool = (pl_tone == 0.0)  # pessimistic for PL channels; True when no PL configured
                detected_ctcss: float | None = None   # last detected tone for display
                self._resume_event.clear()  # reset any pending resume from previous dwell
                fir_zi_i = np.zeros(len(fir_coeffs) - 1)  # FIR state reset on each freq hop
                fir_zi_q = np.zeros(len(fir_coeffs) - 1)
                sq_just_opened = False
                open_debounce  = 0

                while self._running:
                    # Exit inner loop immediately if this freq was skipped mid-dwell.
                    with self._lock:
                        if freq_str in self.skipped:
                            break
                    try:
                        raw = iq_q.get(timeout=5.0)
                    except _q.Empty:
                        raise RuntimeError("rtlsdr async read timed out — device stalled?")

                    # Stage 1 — FIR anti-aliasing + stride decimation.
                    # Filter all samples to maintain continuous zi state, then stride-decimate.
                    filt_i, fir_zi_i = lfilter(fir_coeffs, [1.0], raw.real, zi=fir_zi_i)
                    filt_q, fir_zi_q = lfilter(fir_coeffs, [1.0], raw.imag, zi=fir_zi_q)
                    n_out = len(raw) // decimate
                    iq_if = (filt_i[:n_out * decimate:decimate]
                             + 1j * filt_q[:n_out * decimate:decimate]).astype(np.complex64)
                    iq_if -= iq_if.mean()   # remove RTL-SDR LO leakage (DC offset at 0 Hz)

                    # Stage 2 — FM discriminator at audio_rate.
                    # With rtlsdr_read_async the hardware streams continuously with zero
                    # inter-callback gaps, so last_iq (the final decimated sample of the
                    # previous chunk) connects directly to iq_if[0] in time.  Prepending
                    # it restores cross-chunk phase continuity without aliasing.
                    if last_iq is not None:
                        iq_ext = np.concatenate(([last_iq], iq_if))
                    else:
                        iq_ext = iq_if
                    last_iq = iq_if[-1]
                    demod = np.angle(iq_ext[1:] * np.conj(iq_ext[:-1]))

                    audio = (demod * fm_scale).astype(np.float32)
                    np.clip(audio, -1.0, 1.0, out=audio)
                    # De-emphasis: 1-pole IIR (y[n] = β·x[n] + α·y[n-1]).
                    audio_f64, zf = lfilter(
                        [_deemph_beta], [1.0, -_deemph_alpha],
                        audio.astype(np.float64), zi=np.array([deemph_z])
                    )
                    deemph_z = float(zf[0])
                    audio = np.clip(audio_f64, -1.0, 1.0).astype(np.float32)

                    # Phase-variance squelch: noise gives var(Δφ) ≈ π²/3;
                    # a captured FM carrier drives it near zero.
                    noise_ratio  = min(float(np.var(demod)) / _NOISE_VAR, 1.0)
                    signal_level = 1.0 - noise_ratio
                    rms    = signal_level
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
                        ctcss_buf.extend(audio.tolist())
                        if len(ctcss_buf) >= _CTCSS_WINDOW:
                            ctcss_detected, detected_ctcss = _ctcss_analyze(
                                np.array(ctcss_buf[:_CTCSS_WINDOW], dtype=np.float32),
                                self.audio_rate, pl_tone,
                            )
                            if self.debug:
                                print(f"[ctcss] {freq_str}: detected={detected_ctcss}  "
                                      f"pl={pl_tone}  gated={'open' if ctcss_detected else 'closed'}")
                            ctcss_buf = ctcss_buf[_CTCSS_WINDOW:]
                    elif pl_tone > 0.0:
                        # Carrier gone — reset so the next transmission is evaluated fresh.
                        ctcss_buf.clear()
                        ctcss_detected = False

                    active = signal_present and ctcss_detected
                    if self.debug:
                        dbg_state = (freq_str, squelch_open, active)
                        if dbg_state != _last_dbg_state:
                            print(f"[squelch] {freq_str}: sig={signal_level:.3f} "
                                  f"({db:.1f} dB)  thr={threshold}"
                                  f"  {'OPEN' if squelch_open else 'closed'}"
                                  f"  {'ACTIVE' if active else 'inactive'}")
                            _last_dbg_state = dbg_state

                    self._emit({"type": "signal", "mount": "sdr",
                                "db": round(db, 1), "active": active,
                                "ctcss": detected_ctcss})

                    if active:
                        if squelch_open:
                            # Already open: extend hold/dwell timers normally.
                            last_sig_t  = time.time()
                            dwell_start = time.time()
                            open_debounce = 0
                        else:
                            # Debounce: require consecutive active chunks before opening.
                            # Prevents brief noise spikes from triggering hold.
                            open_debounce += 1
                            if open_debounce >= _OPEN_DEBOUNCE:
                                last_sig_t  = time.time()
                                dwell_start = time.time()
                                squelch_open    = True
                                sq_just_opened  = True
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
                    else:
                        open_debounce = 0

                    # Pre-compute close condition before emitting audio so the
                    # final chunk can be faded out rather than cut off abruptly.
                    should_close_sq = False
                    holding = False
                    if not active:
                        with self._lock:
                            holding = self.hold_freq == freq_str
                        if squelch_open and time.time() - last_sig_t > self.squelch_hold:
                            should_close_sq = True

                    if squelch_open:
                        out = audio
                        if sq_just_opened:
                            out = audio.copy()
                            n = min(_SQUELCH_FADE, len(out))
                            out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
                            sq_just_opened = False
                        if should_close_sq:
                            if out is audio:
                                out = audio.copy()
                            n = min(_SQUELCH_FADE, len(out))
                            out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
                        pcm = (out * 32767).astype(np.int16).tobytes()
                        self._emit_audio(pcm)

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

                scan_idx = (scan_idx + 1) % n_freqs

        finally:
            _stop_reader()
            try:
                sdr.close()
            except Exception:
                pass


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
                new_channels[freq] = entry if len(entry) > 1 else lbl
        cfg["channels"] = new_channels
        with scanner._lock:
            cfg["skipped"] = sorted(scanner.skipped)
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
            "defaultSquelch": scanner.squelch_rms,
            "defaultGain":    scanner.gain,
            "skipped":        sorted(scanner.skipped),
        }


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
    with s._lock:
        channels        = dict(s.channels)
        channel_squelch = dict(s.channel_squelch)
        channel_gain    = dict(s.channel_gain)
        channel_pl      = dict(s.channel_pl)
        skipped         = sorted(s.skipped)
        hold_freq       = s.hold_freq
    return {
        "type":       "state",
        "audio_rate": s.audio_rate,
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
            "channelPL":      channel_pl,
            "defaultSquelch": s.squelch_rms,
            "defaultGain":    s.gain,
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
    scanner.start()
    print("Scanner started")


@app.on_event("shutdown")
async def _shutdown():
    if scanner: scanner.stop()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=PAGE.replace('__VERSION__', VERSION, 1),
                        headers={"Cache-Control": "no-store"})


@app.get("/manifest.json")
async def pwa_manifest():
    name = (scanner.name if scanner else None) or "RTL Scanner"
    return JSONResponse({
        "name": name,
        "short_name": name[:15],
        "display": "standalone",
        "start_url": "/",
        "background_color": "#0a0d0f",
        "theme_color": "#0a0d0f",
        "icons": [
            {"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any maskable"},
        ],
    })


@app.get("/icon.svg")
async def pwa_icon():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <rect width="512" height="512" fill="#0a0d0f"/>
  <circle cx="256" cy="296" r="18" fill="#2dff6e"/>
  <path d="M196 256 a80 80 0 0 1 120 0" fill="none" stroke="#2dff6e" stroke-width="18" stroke-linecap="round"/>
  <path d="M158 218 a130 130 0 0 1 196 0" fill="none" stroke="#2dff6e" stroke-width="14" stroke-linecap="round" opacity=".65"/>
  <path d="M118 180 a182 182 0 0 1 276 0" fill="none" stroke="#2dff6e" stroke-width="10" stroke-linecap="round" opacity=".35"/>
</svg>"""
    return Response(content=svg, media_type="image/svg+xml")


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
    if gain_arg is ... and pl_arg is ...:
        scanner.set_channel(freq, label, squelch_rms)
    elif gain_arg is ...:
        scanner.set_channel(freq, label, squelch_rms, pl=pl_arg)
    elif pl_arg is ...:
        scanner.set_channel(freq, label, squelch_rms, gain_arg)
    else:
        scanner.set_channel(freq, label, squelch_rms, gain_arg, pl_arg)
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



@app.get("/stream")
async def audio_stream(request: Request):
    """HTTP audio stream — serves a never-ending WAV so browsers treat the
    tab as native media (required for background playback on Android)."""
    rate = scanner.audio_rate if scanner else AUDIO_RATE
    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=30)
    _audio_clients.append(q)
    # Send up to 1 s of silence when the queue is empty (squelch closed / no
    # signal).  1 s is long enough that the <audio> element never stalls, and
    # short enough that real audio chunks — which arrive every ~250 ms when the
    # scanner is transmitting — always beat the timeout and no silence is
    # injected mid-transmission.
    silence = bytes(int(rate * 2 * 0.5))   # 500 ms

    async def generate():
        yield _wav_header(rate)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    data = silence
                yield data
        finally:
            try: _audio_clients.remove(q)
            except ValueError: pass

    return StreamingResponse(
        generate(),
        media_type="audio/wav",
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/stream.mp3")
async def audio_stream_mp3(request: Request):
    """MP3 stream for Android — ffmpeg encodes PCM to MP3 so Android's media
    stack treats the tab as a radio app, surviving Doze mode indefinitely.
    Requires: sudo apt install ffmpeg"""
    rate = scanner.audio_rate if scanner else AUDIO_RATE
    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=30)
    _audio_clients.append(q)
    silence = bytes(int(rate * 2 * 0.5))

    try:
        proc = await asyncio.create_subprocess_exec(
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-f', 's16le', '-ar', str(rate), '-ac', '1', '-i', 'pipe:0',
            '-c:a', 'libmp3lame', '-b:a', '32k', '-f', 'mp3', 'pipe:1',
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
        try:
            while not stop.is_set():
                try:
                    data = await asyncio.wait_for(q.get(), timeout=1.0)
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
    for freq, val in raw_channels.items():
        if isinstance(val, dict):
            channels[freq] = val.get("label", freq)
            if "squelch_rms" in val:
                channel_squelch[freq] = float(val["squelch_rms"])
            if "gain" in val:
                channel_gain[freq] = str(val["gain"])
            if "pl" in val and float(val["pl"]) > 0.0:
                channel_pl[freq] = float(val["pl"])
        else:
            channels[freq] = str(val)

    scanner = RTLFMScanner(
        name            = cfg.get("name", "Scanner"),
        channels        = channels,
        squelch_rms     = cfg.get("squelch_rms", 0.05),
        squelch_hold    = cfg.get("squelch_hold", 2.0),
        channel_squelch = channel_squelch,
        channel_gain    = channel_gain,
        channel_pl      = channel_pl,
        skipped         = set(cfg.get("skipped", [])),
        ppm             = cfg.get("ppm", 0),
        modulation      = cfg.get("modulation", "fm"),
        device          = cfg.get("device", "0"),
        gain            = cfg.get("gain", "auto"),
        samp_rate       = cfg.get("samp_rate", 240000),
        scan_dwell      = cfg.get("scan_dwell", 0.5),
        fir_taps        = cfg.get("fir_taps", 127),
        debug           = args.debug,
        on_event        = _emit,
        on_audio        = _audio_cb,
    )

    print(f"Open http://<pi-ip>:{args.listen_port} in your browser")
    uvicorn.run(app, host="0.0.0.0", port=args.listen_port, log_level="warning")


if __name__ == "__main__":
    main()
