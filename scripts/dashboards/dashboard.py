"""Hogan Trading Bot — Live Dashboard

Launch:
    streamlit run dashboard.py

Auto-refreshes every 30 seconds. All data comes from the SQLite DB —
safe to run while the bot is running.
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so ``hogan_bot`` is importable
# regardless of the working directory Streamlit was launched from.
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh  # type: ignore[import]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hogan Bot",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Auto-refresh (every 30 seconds)
# ---------------------------------------------------------------------------
try:
    st_autorefresh(interval=30_000, key="autorefresh")
except Exception:
    pass  # streamlit-autorefresh may not be installed

# ---------------------------------------------------------------------------
# Sidebar — config paths
# ---------------------------------------------------------------------------
st.sidebar.header("Config")
DB_PATH = st.sidebar.text_input("SQLite DB", value="data/hogan.db")
MODEL_PATH = st.sidebar.text_input("Model", value="models/hogan_logreg.pkl")
REGISTRY_PATH = st.sidebar.text_input("Registry", value="models/registry.jsonl")
AUTO_REFRESH = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
if st.sidebar.button("Refresh now"):
    st.rerun()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CACHE_TTL = 25  # seconds


@st.cache_data(ttl=_CACHE_TTL)
def _load_equity(db: str, limit: int = 5000) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db)
        df = pd.read_sql_query(
            f"SELECT ts_ms, equity_usd, drawdown FROM equity_snapshots ORDER BY ts_ms DESC LIMIT {limit}",
            conn,
        )
        conn.close()
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        return df.sort_values("ts")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=_CACHE_TTL)
def _load_paper_trades(db: str) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db)
        df = pd.read_sql_query(
            """SELECT trade_id, symbol, side, entry_price, exit_price, qty,
                      realized_pnl, pnl_pct, entry_fee, exit_fee,
                      open_ts_ms, close_ts_ms, close_reason,
                      ml_up_prob, strategy_conf, vol_ratio
               FROM paper_trades ORDER BY open_ts_ms DESC""",
            conn,
        )
        # Join trade explanations (Mistral/Ollama) — fill_id = symbol_open_ts_ms
        try:
            expl = pd.read_sql_query(
                "SELECT fill_id, explanation, model_used FROM trade_explanations",
                conn,
            )
            if not expl.empty:
                df["fill_id"] = df["symbol"] + "_" + df["open_ts_ms"].astype(str)
                df = df.merge(expl[["fill_id", "explanation", "model_used"]], on="fill_id", how="left")
                df = df.drop(columns=["fill_id"], errors="ignore")
        except Exception:
            pass
        conn.close()
        df["opened"] = pd.to_datetime(df["open_ts_ms"], unit="ms", utc=True)
        df["closed"] = pd.to_datetime(df["close_ts_ms"], unit="ms", utc=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=_CACHE_TTL)
def _load_candles(db: str, symbol: str, limit: int = 200, timeframe: str = "1h") -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db)
        df = pd.read_sql_query(
            "SELECT ts_ms, open, high, low, close, volume FROM candles "
            "WHERE symbol=? AND timeframe=? ORDER BY ts_ms DESC LIMIT ?",
            conn, params=(symbol, timeframe, limit),
        )
        conn.close()
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        return df.sort_values("ts")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=_CACHE_TTL)
def _load_onchain_latest(db: str) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db)
        df = pd.read_sql_query(
            """SELECT metric, value, date FROM onchain_metrics
               WHERE (metric, date) IN (
                   SELECT metric, MAX(date) FROM onchain_metrics GROUP BY metric
               ) ORDER BY metric""",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _load_model_info(model_path: str) -> dict:
    try:
        with open(model_path, "rb") as f:
            art = pickle.load(f)
        info = {"model_type": type(art.model).__name__, "features": len(art.feature_columns)}
        if hasattr(art.model, "feature_importances_"):
            import numpy as np
            info["importances"] = dict(zip(art.feature_columns, art.model.feature_importances_))
        elif hasattr(art.model, "coef_"):
            import numpy as np
            info["importances"] = dict(zip(art.feature_columns, np.abs(art.model.coef_[0])))
        return info
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300)
def _load_registry(registry_path: str) -> list[dict]:
    try:
        rows = []
        with open(registry_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    except Exception:
        return []


def _pnl_color(val):
    if pd.isna(val):
        return ""
    return "color: #2ecc71" if val >= 0 else "color: #e74c3c"


# ---------------------------------------------------------------------------
# Swarm helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=_CACHE_TTL)
def _load_agent_votes(db: str, limit: int = 400) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db)
        df = pd.read_sql_query(
            f"""SELECT sd.ts_ms, sav.agent_id, sav.action, sav.confidence,
                       sav.veto, sav.size_scale
                FROM swarm_agent_votes sav
                JOIN swarm_decisions sd ON sav.decision_id = sd.id
                ORDER BY sd.ts_ms DESC LIMIT {int(limit)}""",
            conn,
        )
        conn.close()
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=_CACHE_TTL)
def _load_veto_ledger(db: str, limit: int = 200) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db)
        df = pd.read_sql_query(
            f"""SELECT sav.ts_ms, sav.agent_id, sav.block_reasons_json,
                       sd.final_action AS swarm_action, sd.agreement
                FROM swarm_agent_votes sav
                JOIN swarm_decisions sd ON sav.decision_id = sd.id
                WHERE sav.veto = 1
                ORDER BY sav.ts_ms DESC LIMIT {int(limit)}""",
            conn,
        )
        conn.close()
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
            df["reasons"] = df["block_reasons_json"].apply(
                lambda x: ", ".join(json.loads(x)) if x else ""
            )
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=_CACHE_TTL)
def _load_decision_detail(db: str, decision_id: int) -> dict:
    """Load full detail for a single swarm decision + its votes + baseline."""
    try:
        conn = sqlite3.connect(db)
        dec = pd.read_sql_query(
            "SELECT * FROM swarm_decisions WHERE id = ?", conn,
            params=(decision_id,),
        )
        votes = pd.read_sql_query(
            "SELECT agent_id, action, confidence, size_scale, veto, "
            "       block_reasons_json, expected_edge_bps "
            "FROM swarm_agent_votes WHERE decision_id = ?", conn,
            params=(decision_id,),
        )
        baseline = pd.DataFrame()
        if not dec.empty:
            ts_ms = int(dec.iloc[0]["ts_ms"])
            symbol = dec.iloc[0]["symbol"]
            baseline = pd.read_sql_query(
                "SELECT final_action, final_confidence, position_size, "
                "       regime, ml_up_prob, conf_scale, block_reasons_json "
                "FROM decision_log WHERE ts_ms = ? AND symbol = ? LIMIT 1",
                conn, params=(ts_ms, symbol),
            )
        conn.close()
        return {"decision": dec, "votes": votes, "baseline": baseline}
    except Exception:
        return {"decision": pd.DataFrame(), "votes": pd.DataFrame(), "baseline": pd.DataFrame()}


@st.cache_data(ttl=_CACHE_TTL)
def _load_promotion_stats(db: str) -> dict:
    """Collect counts needed for the Promotion Readiness Card."""
    stats = {
        "shadow_decisions": 0, "would_trade": 0,
        "veto_events": 0, "distinct_regimes": 0,
        "mean_agreement": 0.0,
    }
    try:
        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT COUNT(*) FROM swarm_decisions WHERE mode = 'shadow'"
        ).fetchone()
        stats["shadow_decisions"] = row[0] if row else 0

        row = conn.execute(
            "SELECT COUNT(*) FROM swarm_decisions "
            "WHERE mode = 'shadow' AND final_action IN ('buy', 'sell')"
        ).fetchone()
        stats["would_trade"] = row[0] if row else 0

        row = conn.execute(
            "SELECT COUNT(*) FROM swarm_decisions WHERE mode = 'shadow' AND vetoed = 1"
        ).fetchone()
        stats["veto_events"] = row[0] if row else 0

        row = conn.execute(
            "SELECT COUNT(DISTINCT json_extract(decision_json, '$.regime')) "
            "FROM swarm_decisions WHERE mode = 'shadow'"
        ).fetchone()
        stats["distinct_regimes"] = row[0] if row else 0

        row = conn.execute(
            "SELECT AVG(agreement) FROM swarm_decisions WHERE mode = 'shadow'"
        ).fetchone()
        stats["mean_agreement"] = round(row[0] or 0.0, 3)

        conn.close()
    except Exception:
        pass
    return stats


def _bruno_html(mood: str = "neutral", regime: str = "unknown", quip: str = "") -> str:
    """Return self-contained HTML/JS for the animated Bruno the Bull mascot."""
    return f"""
<canvas id="brunoCanvas" width="200" height="260"
        style="cursor:pointer;display:block;margin:0 auto"></canvas>
<div id="moodBadge" style="text-align:center;margin-top:4px;font-family:sans-serif;
     font-size:11px;font-weight:700;letter-spacing:2px;padding:3px 10px;
     border-radius:8px;display:inline-block;width:100%;box-sizing:border-box"></div>
<div id="quipText" style="text-align:center;margin-top:4px;font-family:sans-serif;
     font-size:10px;color:#8898aa;min-height:16px"></div>
<script>
const cv=document.getElementById('brunoCanvas'),c=cv.getContext('2d');
const mood='{mood}', regime='{regime}', quip=`{quip}`;
const B={{breath:0,bob:0,blinkT:0,blink:false,eyeTX:0,eyeTY:0,eyeX:0,eyeY:0,
  mouth:mood==='bullish'||mood==='excited'?1:mood==='bearish'||mood==='worried'?-1:0,
  tailA:0,sparks:[],particles:[],petT:0,snorePhase:0}};
const sleeping=mood==='sleeping';
const moodLabels={{bullish:'BULLISH',bearish:'BEARISH',neutral:'WATCHING',
  excited:'ON FIRE',worried:'CONCERNED',sleeping:'SLEEPING'}};
const moodColors={{bullish:'#10b981',bearish:'#ef4444',neutral:'#f59e0b',
  excited:'#fbbf24',worried:'#f87171',sleeping:'#475569'}};
const badgeBg={{bullish:'rgba(16,185,129,.12)',bearish:'rgba(239,68,68,.12)',
  neutral:'rgba(245,158,11,.1)',excited:'rgba(251,191,36,.15)',
  worried:'rgba(239,68,68,.08)',sleeping:'rgba(100,116,139,.06)'}};
const mb=document.getElementById('moodBadge');
mb.textContent=moodLabels[mood]||'WATCHING';
mb.style.color=moodColors[mood]||'#f59e0b';
mb.style.background=badgeBg[mood]||'rgba(245,158,11,.1)';
document.getElementById('quipText').textContent=quip;
const accessory=regime==='trending'?'sunglasses':regime==='volatile'?'hardhat':
  regime==='ranging'?'sleepingcap':regime==='risk_off'?'scarf':'none';

function draw(t){{
  const W=200,HH=260;c.clearRect(0,0,W,HH);
  B.breath+=.025;B.bob+=.018;
  const by=Math.sin(B.breath)*2.5,bb=Math.sin(B.bob)*1.5;
  const cx=100,cy=148+by+bb;
  const bodyC=mood==='bullish'||mood==='excited'?'#fbbf24':mood==='bearish'||mood==='worried'?'#8faab8':sleeping?'#a0937c':'#d4a574';
  const dkC=mood==='bullish'||mood==='excited'?'#b47e0a':mood==='bearish'||mood==='worried'?'#607888':sleeping?'#7a7060':'#9c7040';

  B.particles=B.particles.filter(p=>{{
    p.life-=.015;p.x+=p.vx;p.y+=p.vy;p.vy+=.05;
    if(p.life<=0)return false;
    c.save();c.globalAlpha=p.life;c.fillStyle=p.color;
    c.beginPath();c.arc(p.x,p.y,p.sz,0,Math.PI*2);c.fill();c.restore();
    return true;
  }});
  if((mood==='bullish'||mood==='excited')&&Math.random()<.12)
    B.sparks.push({{x:20+Math.random()*160,y:10+Math.random()*90,l:1,s:Math.random()*3+1}});
  B.sparks=B.sparks.filter(s=>{{
    s.l-=.012;s.y-=.4;if(s.l<=0)return false;
    c.save();c.globalAlpha=s.l;c.fillStyle='#fde68a';
    c.translate(s.x,s.y);c.rotate(t/300+s.x);
    for(let i=0;i<4;i++){{c.fillRect(-.5,-s.s,1,s.s*2);c.rotate(Math.PI/4);}}
    c.restore();return true;
  }});
  if(mood==='bearish'||mood==='worried'){{
    c.save();c.globalAlpha=.2;c.strokeStyle='#60a5fa';c.lineWidth=1;
    for(let i=0;i<5;i++){{const rx=15+i*38,ry=((t/50+i*35)%130)+10;
    c.beginPath();c.moveTo(rx,ry);c.lineTo(rx-1,ry+7);c.stroke();}}c.restore();
  }}
  c.save();c.globalAlpha=.1;c.fillStyle='#000';
  c.beginPath();c.ellipse(cx,238,38,6,0,0,Math.PI*2);c.fill();c.restore();
  B.tailA+=(mood==='bullish'||mood==='excited'?.07:sleeping?.008:.025);
  const ts=Math.sin(B.tailA)*(mood==='excited'?18:12);
  c.strokeStyle=bodyC;c.lineWidth=4.5;c.lineCap='round';
  c.beginPath();c.moveTo(cx+40,cy-8);c.quadraticCurveTo(cx+55+ts,cy-28,cx+48+ts*1.3,cy-44);c.stroke();
  c.fillStyle=dkC;c.beginPath();c.arc(cx+48+ts*1.3,cy-46,4.5,0,Math.PI*2);c.fill();
  for(let lx of [-20,-7,7,20]){{
    const lb=Math.sin(B.bob+lx*.12)*2;
    const stomp=mood==='excited'?Math.abs(Math.sin(t/100+lx))*3:0;
    c.fillStyle=bodyC;c.beginPath();c.roundRect(cx+lx-4.5,cy+38+lb-stomp,9,20,4);c.fill();
    c.fillStyle=dkC;c.beginPath();c.roundRect(cx+lx-5.5,cy+54+lb-stomp,11,4.5,3);c.fill();
  }}
  c.fillStyle=bodyC;c.beginPath();c.ellipse(cx,cy,44,50,0,0,Math.PI*2);c.fill();
  c.save();c.globalAlpha=.18;c.fillStyle='#fff';
  c.beginPath();c.ellipse(cx,cy+9,24,28,0,0,Math.PI*2);c.fill();c.restore();
  const hy=cy-56+by*.4+(sleeping?4:0);
  const headTilt=sleeping?.08:0;
  c.save();c.translate(cx,hy);c.rotate(headTilt);
  c.fillStyle=bodyC;c.beginPath();c.ellipse(0,0,29,25,0,0,Math.PI*2);c.fill();
  c.save();c.globalAlpha=.12;c.fillStyle='#fff';c.beginPath();c.ellipse(0,5,18,13,0,0,Math.PI*2);c.fill();c.restore();
  c.strokeStyle='#c9a96e';c.lineWidth=3;c.lineCap='round';
  c.beginPath();c.moveTo(-18,-18);c.quadraticCurveTo(-27,-36,-14,-40);c.stroke();
  c.beginPath();c.moveTo(18,-18);c.quadraticCurveTo(27,-36,14,-40);c.stroke();
  c.strokeStyle='#fef3c7';c.lineWidth=1.8;
  c.beginPath();c.moveTo(-16,-37);c.lineTo(-14,-40);c.stroke();
  c.beginPath();c.moveTo(16,-37);c.lineTo(14,-40);c.stroke();
  c.fillStyle=bodyC;
  c.beginPath();c.ellipse(-26,-6,8,5.5,-.4,0,Math.PI*2);c.fill();
  c.beginPath();c.ellipse(26,-6,8,5.5,.4,0,Math.PI*2);c.fill();
  c.fillStyle='#e8a0a0';
  c.beginPath();c.ellipse(-26,-6,4,2.8,-.4,0,Math.PI*2);c.fill();
  c.beginPath();c.ellipse(26,-6,4,2.8,.4,0,Math.PI*2);c.fill();
  B.blinkT++;
  if(B.blinkT>150+Math.random()*100){{B.blink=true;B.blinkT=0;}}
  if(B.blink){{B.blinkT++;if(B.blinkT>7)B.blink=false;}}
  B.eyeX+=(B.eyeTX-B.eyeX)*.08;B.eyeY+=(B.eyeTY-B.eyeY)*.08;
  if(sleeping){{
    c.strokeStyle=dkC;c.lineWidth=2;
    c.beginPath();c.moveTo(-14,-2);c.lineTo(-7,-2);c.stroke();
    c.beginPath();c.moveTo(7,-2);c.lineTo(14,-2);c.stroke();
  }}else{{
    const eH=B.blink?1:6.5;
    for(let s of[-1,1]){{
      const ex=s*12,ey=-2;
      c.fillStyle='#fff';c.beginPath();c.ellipse(ex,ey,7.5,eH,0,0,Math.PI*2);c.fill();
      if(!B.blink){{
        const px=mood==='worried'?s*1:0;
        c.fillStyle='#1e293b';c.beginPath();c.ellipse(ex+B.eyeX*2+px,ey+B.eyeY*1.5,3.8,4.2,0,0,Math.PI*2);c.fill();
        c.fillStyle='#fff';c.beginPath();c.arc(ex+B.eyeX*2+1.5+px,ey+B.eyeY*1.5-1.5,1.2,0,Math.PI*2);c.fill();
      }}
    }}
  }}
  c.strokeStyle=dkC;c.lineWidth=2;c.lineCap='round';
  const bOff=mood==='bearish'||mood==='worried'?-3:mood==='bullish'||mood==='excited'?3:0;
  c.beginPath();c.moveTo(-19,-12-bOff);c.lineTo(-5,-12+bOff*.5);c.stroke();
  c.beginPath();c.moveTo(19,-12-bOff);c.lineTo(5,-12+bOff*.5);c.stroke();
  c.fillStyle=dkC;
  c.beginPath();c.ellipse(-4,6,2,1.6,0,0,Math.PI*2);c.fill();
  c.beginPath();c.ellipse(4,6,2,1.6,0,0,Math.PI*2);c.fill();
  c.strokeStyle='#fbbf24';c.lineWidth=1.6;
  c.beginPath();c.arc(0,9,3.5,.3,Math.PI-.3);c.stroke();
  c.strokeStyle=dkC;c.lineWidth=1.6;
  c.beginPath();c.moveTo(-8,14);c.quadraticCurveTo(0,14+B.mouth*7,8,14);c.stroke();
  if(B.petT>0){{
    B.petT--;c.save();c.globalAlpha=B.petT/40*.4;
    c.fillStyle='#f87171';
    c.beginPath();c.ellipse(-16,5,5,3.5,0,0,Math.PI*2);c.fill();
    c.beginPath();c.ellipse(16,5,5,3.5,0,0,Math.PI*2);c.fill();
    c.restore();
  }}
  c.restore();
  if(accessory==='sunglasses'&&!sleeping){{
    c.save();c.translate(cx,hy);c.rotate(headTilt);
    c.fillStyle='rgba(30,41,59,.85)';c.strokeStyle='#475569';c.lineWidth=1.3;
    c.beginPath();c.roundRect(-20,-7,15,10,3);c.fill();c.stroke();
    c.beginPath();c.roundRect(5,-7,15,10,3);c.fill();c.stroke();
    c.strokeStyle='#475569';c.lineWidth=1.8;c.beginPath();c.moveTo(-5,-2);c.lineTo(5,-2);c.stroke();
    c.restore();
  }}
  if(accessory==='hardhat'){{
    c.save();c.translate(cx,hy);
    const hhy=-22+Math.sin(B.bob*1.3)*1;
    c.fillStyle='#eab308';
    c.beginPath();c.ellipse(0,hhy+5,30,3.5,0,0,Math.PI*2);c.fill();
    c.beginPath();c.moveTo(-20,hhy+2);c.quadraticCurveTo(0,hhy-16,20,hhy+2);c.lineTo(20,hhy+5);c.lineTo(-20,hhy+5);c.fill();
    c.restore();
  }}
  if(accessory==='scarf'){{
    c.save();c.translate(cx,hy);
    c.fillStyle='#dc2626';c.globalAlpha=.8;
    c.beginPath();c.ellipse(0,22,25,5,0,0,Math.PI*2);c.fill();
    c.fillRect(7,22,5,16);c.fillRect(12,22,4,12);
    c.restore();
  }}
  if((mood==='bullish'||mood==='excited')&&accessory!=='hardhat'){{
    c.save();c.translate(cx,hy);
    const hty=-24+Math.sin(B.bob*1.3)*1.5;
    c.fillStyle='#92400e';
    c.beginPath();c.ellipse(0,hty+6,34,5,0,0,Math.PI*2);c.fill();
    c.beginPath();c.moveTo(-16,hty+3);c.quadraticCurveTo(0,hty-13,16,hty+3);c.lineTo(16,hty+6);c.lineTo(-16,hty+6);c.fill();
    c.fillStyle='#fbbf24';c.beginPath();c.arc(0,hty+1,2.2,0,Math.PI*2);c.fill();
    c.restore();
  }}
  if(mood==='bearish'||mood==='worried'){{
    c.save();c.globalAlpha=.3+Math.sin(t/400)*.1;c.translate(cx,hy);
    c.fillStyle='#475569';
    c.beginPath();c.arc(-7,-40,10,0,Math.PI*2);c.fill();
    c.arc(7,-42,8,0,Math.PI*2);c.fill();
    c.arc(0,-34,9,0,Math.PI*2);c.fill();
    c.restore();
  }}
  if(sleeping){{
    B.snorePhase+=.02;
    c.save();c.globalAlpha=.5+Math.sin(B.snorePhase)*.2;
    c.fillStyle='#94a3b8';c.font='bold 13px sans-serif';
    const zy=hy-32+Math.sin(B.snorePhase*2)*5;
    c.fillText('z',cx+22,zy);c.font='bold 9px sans-serif';c.fillText('z',cx+32,zy-11);
    c.font='bold 7px sans-serif';c.fillText('z',cx+38,zy-20);
    c.restore();
  }}
  if(mood==='neutral'){{
    c.save();c.globalAlpha=.35+Math.sin(t/300)*.2;
    for(let i=0;i<3;i++){{
      const dx=cx+28+i*7,dy=hy-20-i*5+Math.sin(t/300+i)*3;
      c.fillStyle='#f59e0b';c.beginPath();c.arc(dx,dy,2.2-i*.3,0,Math.PI*2);c.fill();
    }}c.restore();
  }}
  requestAnimationFrame(draw);
}}
cv.addEventListener('mousemove',e=>{{const r=cv.getBoundingClientRect();B.eyeTX=((e.clientX-r.left)/200-.5)*2;B.eyeTY=((e.clientY-r.top)/260-.5)*2;}});
cv.addEventListener('mouseleave',()=>{{B.eyeTX=0;B.eyeTY=0;}});
cv.addEventListener('click',()=>{{
  B.petT=40;
  for(let i=0;i<4;i++) B.particles.push({{x:100+Math.random()*30-15,y:80,vx:(Math.random()-.5)*2,vy:-Math.random()*3-1,life:1,color:'#f87171',sz:2.5}});
}});
requestAnimationFrame(draw);
</script>"""


def _determine_mood(
    win_rate: float, drawdown: float, total_pnl: float, num_open: int
) -> tuple[str, str]:
    """Pick Bruno's mood and a quip based on current trading state."""
    import random
    if drawdown > 8:
        quips = ["Stay cautious...", "Bears prowling", "Rough waters", "Patience..."]
        return "worried", random.choice(quips)
    if win_rate >= 65 and total_pnl > 0:
        quips = ["BOOM! Nice trades!", "That's what I'm talking about!", "Bulls in control!"]
        return "excited", random.choice(quips)
    if total_pnl > 0 and win_rate >= 50:
        quips = ["Let's ride!", "Looking strong", "Send it!", "Yeehaw!"]
        return "bullish", random.choice(quips)
    if total_pnl < 0:
        quips = ["Not feeling it", "Getting choppy", "Need to be careful"]
        return "bearish", random.choice(quips)
    if num_open == 0:
        quips = ["Watching...", "Hmm, ranging", "Waiting for setup", "Flat market"]
        return "neutral", random.choice(quips)
    quips = ["Watching...", "Nothing yet", "Waiting for setup"]
    return "neutral", random.choice(quips)


# ---------------------------------------------------------------------------
# Main tab layout
# ---------------------------------------------------------------------------
tab_live, tab_os, tab_signals, tab_model, tab_data, tab_backtest, tab_swarm, tab_replay = st.tabs([
    "Live Trading", "Market OS", "Signals & Candles", "ML Models", "Data Coverage", "Backtest", "Swarm", "Swarm Replay"
])

# ===========================================================================
# TAB 1 — LIVE TRADING
# ===========================================================================
with tab_live:
    st.header("Live Paper Trading")

    trades = _load_paper_trades(DB_PATH)
    equity = _load_equity(DB_PATH)

    # ── KPI row ──────────────────────────────────────────────────────────────
    closed = trades[trades["exit_price"].notna()] if not trades.empty else pd.DataFrame()
    open_trades = trades[trades["exit_price"].isna()] if not trades.empty else pd.DataFrame()

    cur_equity = equity["equity_usd"].iloc[-1] if not equity.empty else 10000.0
    start_equity = equity["equity_usd"].iloc[0] if not equity.empty else 10000.0
    total_pnl = cur_equity - start_equity
    total_pnl_pct = (total_pnl / start_equity * 100) if start_equity else 0.0

    win_rate = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    if not closed.empty and "realized_pnl" in closed.columns:
        wins = closed[closed["realized_pnl"] > 0]
        losses = closed[closed["realized_pnl"] <= 0]
        win_rate = len(wins) / len(closed) * 100
        avg_win = wins["realized_pnl"].mean() if not wins.empty else 0.0
        avg_loss = losses["realized_pnl"].mean() if not losses.empty else 0.0

    max_dd = equity["drawdown"].max() * 100 if not equity.empty and "drawdown" in equity.columns else 0.0

    # ── Bruno the Bull mascot + KPIs side-by-side ────────────────────────────
    bruno_col, kpi_col = st.columns([1, 4])
    with bruno_col:
        _mood, _quip = _determine_mood(win_rate, max_dd, total_pnl, len(open_trades))
        components.html(_bruno_html(mood=_mood, regime="unknown", quip=_quip), height=330)

    with kpi_col:
        k1, k2, k3 = st.columns(3)
        k1.metric("Equity", f"${cur_equity:,.2f}", delta=f"${total_pnl:+,.2f}")
        k2.metric("Total P&L", f"${total_pnl:+,.2f}", delta=f"{total_pnl_pct:+.2f}%")
        k3.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{len(closed)} closed trades")

        k4, k5, k6 = st.columns(3)
        k4.metric("Avg Win / Loss", f"${avg_win:.2f} / ${avg_loss:.2f}")
        k5.metric("Max Drawdown", f"{max_dd:.2f}%")
        k6.metric("Open Positions", len(open_trades))

    st.divider()

    # ── Equity curve ─────────────────────────────────────────────────────────
    col_eq, col_dd = st.columns([3, 1])
    with col_eq:
        if not equity.empty:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=equity["ts"], y=equity["equity_usd"],
                mode="lines", name="Equity",
                line=dict(color="#3498db", width=2),
                fill="tozeroy", fillcolor="rgba(52,152,219,0.1)",
            ))
            fig_eq.update_layout(
                title="Equity Curve (USD)", height=280,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title=None, yaxis_title="USD",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_eq, width='stretch')
        else:
            st.info("No equity snapshots yet — start the bot.")

    with col_dd:
        if not equity.empty and "drawdown" in equity.columns:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=equity["ts"], y=equity["drawdown"] * 100,
                mode="lines", name="Drawdown",
                line=dict(color="#e74c3c", width=1.5),
                fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
            ))
            fig_dd.update_layout(
                title="Drawdown %", height=280,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title=None, yaxis_title="%",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dd, width='stretch')

    st.divider()

    # ── Open positions ────────────────────────────────────────────────────────
    st.subheader(f"Open Positions ({len(open_trades)})")
    if not open_trades.empty:
        disp = open_trades[[
            "symbol", "side", "entry_price", "qty", "ml_up_prob", "strategy_conf", "opened"
        ]].copy()
        disp["notional_usd"] = (disp["qty"] * disp["entry_price"]).round(2)
        disp["opened"] = disp["opened"].dt.strftime("%m-%d %H:%M")
        st.dataframe(disp, width='stretch', hide_index=True)
    else:
        st.info("No open positions right now.")

    # ── Closed trades ─────────────────────────────────────────────────────────
    st.subheader(f"Trade Journal — {len(closed)} Closed Trades")
    if not closed.empty:
        # P&L over time area chart
        pnl_cum = closed.sort_values("close_ts_ms").copy()
        pnl_cum["cumulative_pnl"] = pnl_cum["realized_pnl"].cumsum()
        pnl_cum["closed_dt"] = pd.to_datetime(pnl_cum["close_ts_ms"], unit="ms", utc=True)

        if len(pnl_cum) >= 2:
            fig_pnl = go.Figure()
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in pnl_cum["cumulative_pnl"]]
            fig_pnl.add_trace(go.Scatter(
                x=pnl_cum["closed_dt"], y=pnl_cum["cumulative_pnl"],
                mode="lines+markers", name="Cumulative P&L",
                line=dict(color="#2ecc71", width=2),
                fill="tozeroy",
                fillcolor="rgba(46,204,113,0.12)",
            ))
            fig_pnl.update_layout(
                title="Cumulative Realized P&L", height=220,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pnl, width='stretch')

        # Per-symbol P&L bar
        sym_pnl = closed.groupby("symbol")["realized_pnl"].sum().reset_index()
        if len(sym_pnl) > 0:
            fig_sym = px.bar(
                sym_pnl, x="symbol", y="realized_pnl",
                color="realized_pnl", color_continuous_scale=["#e74c3c", "#2ecc71"],
                title="P&L by Symbol", height=180,
            )
            fig_sym.update_layout(
                margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_sym, width='stretch')

        # Close reason breakdown
        reason_counts = closed.groupby("close_reason").size().reset_index(name="count")
        fig_reason = px.pie(
            reason_counts, names="close_reason", values="count",
            title="Close Reason Breakdown", height=200,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_reason.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_reason, width='stretch')

        # Trade table (include explanation if available)
        base_cols = [
            "symbol", "side", "entry_price", "exit_price", "qty",
            "realized_pnl", "pnl_pct", "close_reason",
            "ml_up_prob", "strategy_conf", "opened", "closed"
        ]
        if "explanation" in closed.columns:
            base_cols.insert(base_cols.index("close_reason") + 1, "explanation")
        disp2 = closed[[c for c in base_cols if c in closed.columns]].copy()
        disp2["opened"] = disp2["opened"].dt.strftime("%m-%d %H:%M")
        disp2["closed"] = disp2["closed"].dt.strftime("%m-%d %H:%M")
        disp2["pnl_pct"] = (disp2["pnl_pct"] * 100).round(3)
        disp2["realized_pnl"] = disp2["realized_pnl"].round(4)
        disp2 = disp2.sort_values("closed", ascending=False).head(100)
        st.dataframe(
            disp2.style.map(_pnl_color, subset=["realized_pnl"]),
            width='stretch', hide_index=True,
        )
    else:
        st.info("No closed trades yet. Positions will appear here once they exit.")

    # ── Win/Loss histogram ────────────────────────────────────────────────────
    if not closed.empty and len(closed) >= 5:
        st.subheader("P&L Distribution")
        fig_hist = px.histogram(
            closed, x="realized_pnl", nbins=30,
            color_discrete_sequence=["#3498db"],
            title="Trade P&L Distribution",
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig_hist.update_layout(
            height=220, margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hist, width='stretch')


# ===========================================================================
# TAB 2 — MARKET OS (Agent Pipeline + Forecast + Risk)
# ===========================================================================
with tab_os:
    st.header("Market Operating System")

    os_col1, os_col2 = st.columns(2)
    os_sym = os_col1.selectbox("Symbol", ["BTC/USD", "ETH/USD"], key="os_sym")
    os_tf = os_col2.selectbox("Timeframe", ["1h", "4h", "1d"], key="os_tf")
    os_candles = _load_candles(DB_PATH, os_sym, limit=200, timeframe=os_tf)

    if os_candles.empty:
        st.info("No candle data available.")
    else:
        try:
            from hogan_bot.forecast import compute_forecast
            from hogan_bot.risk_head import compute_risk
            from hogan_bot.regime import detect_regime
            from hogan_bot.agent_pipeline import SentimentAgent, MacroAgent

            fc = compute_forecast(os_candles)
            rk = compute_risk(os_candles, stop_pct=0.0184, tp_pct=0.0572)

            st.subheader("Forecast Head")
            fc_cols = st.columns(4)
            for i, h in enumerate(["4h", "12h", "24h"]):
                p = fc.direction_prob.get(h, 0.5)
                er = fc.expected_return.get(h, 0.0)
                fc_cols[i].metric(f"Up Prob ({h})", f"{p:.1%}", delta=f"{er:+.2f}% E[R]")
            fc_cols[3].metric("Trend Persist", f"{fc.trend_persistence:.0%}",
                              delta=f"conf={fc.confidence:.0%}")

            st.subheader("Risk Head")
            rk_cols = st.columns(5)
            rk_cols[0].metric("Ann. Vol", f"{rk.expected_vol_pct:.1f}%")
            rk_cols[1].metric("Max Adv. Exc.", f"{rk.max_adverse_pct:.2f}%")
            rk_cols[2].metric("Stop-Hit Prob", f"{rk.stop_hit_prob:.0%}")
            rk_cols[3].metric("E[Hold]", f"{rk.expected_hold_bars:.0f} bars")
            rk_cols[4].metric("Risk Regime", rk.regime_risk,
                              delta=f"scale={rk.position_scale:.0%}")

            st.subheader("Market Regime")
            try:
                regime = detect_regime(os_candles)
                reg_cols = st.columns(4)
                reg_cols[0].metric("Regime", regime.regime)
                reg_cols[1].metric("ADX", f"{regime.adx:.1f}")
                reg_cols[2].metric("ATR Rank", f"{regime.atr_pct_rank:.0%}")
                reg_cols[3].metric("Confidence", f"{regime.confidence:.0%}")
            except Exception:
                st.info("Regime detection requires more data.")

            st.subheader("Agent Votes (Current Bar)")
            try:
                conn_os = sqlite3.connect(DB_PATH)
                sent = SentimentAgent(conn=conn_os, symbol=os_sym).analyze()
                macro = MacroAgent(conn=conn_os, symbol=os_sym).analyze()
                conn_os.close()

                ag_cols = st.columns(3)
                ag_cols[0].metric("Sentiment", f"{sent.bias} ({sent.strength:.2f})")
                ag_cols[1].metric("Macro", f"{macro.regime}")
                ag_cols[2].metric("Risk On", str(macro.risk_on))

                if sent.details:
                    with st.expander("Sentiment Details"):
                        st.json(sent.details)
                if macro.details:
                    with st.expander("Macro Details"):
                        st.json({k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in macro.details.items()})
            except Exception as e:
                st.warning(f"Agent data unavailable: {e}")

            st.subheader("Feature Freshness")
            try:
                conn_fresh = sqlite3.connect(DB_PATH)
                freshness = pd.read_sql_query("""
                    SELECT metric,
                           MAX(date) as latest_date,
                           COUNT(*) as total_rows
                    FROM onchain_metrics
                    WHERE symbol = ?
                    GROUP BY metric
                    ORDER BY latest_date DESC
                """, conn_fresh, params=(os_sym,))
                deriv_fresh = pd.read_sql_query("""
                    SELECT metric,
                           datetime(MAX(ts_ms)/1000, 'unixepoch') as latest_ts,
                           COUNT(*) as total_rows
                    FROM derivatives_metrics
                    WHERE symbol = ?
                    GROUP BY metric
                    ORDER BY latest_ts DESC
                """, conn_fresh, params=(os_sym,))
                conn_fresh.close()

                if not freshness.empty:
                    st.dataframe(freshness, width='stretch', hide_index=True)
                if not deriv_fresh.empty:
                    st.dataframe(deriv_fresh, width='stretch', hide_index=True)
            except Exception:
                st.info("No feature freshness data available.")

        except ImportError as e:
            st.error(f"Missing module: {e}")
        except Exception as e:
            st.error(f"Market OS error: {e}")


# ===========================================================================
# TAB 3 — SIGNALS & CANDLES
# ===========================================================================
with tab_signals:
    st.header("Signals & Candles")

    sig_col1, sig_col2, sig_col3 = st.columns(3)
    sig_sym = sig_col1.selectbox("Symbol", ["BTC/USD", "ETH/USD"], key="sig_sym")
    sig_tf = sig_col2.selectbox("Timeframe", ["1h", "15m", "5m", "4h", "1d"], key="sig_tf")
    sig_limit = sig_col3.number_input("Bars", value=200, min_value=50, max_value=2000, step=50, key="sig_lim")
    candles = _load_candles(DB_PATH, sig_sym, limit=int(sig_limit), timeframe=sig_tf)

    if candles.empty:
        st.warning(f"No candles in DB for {sig_sym}")
    else:
        # Candlestick chart
        fig_candle = go.Figure(data=[go.Candlestick(
            x=candles["ts"],
            open=candles["open"], high=candles["high"],
            low=candles["low"], close=candles["close"],
            increasing_line_color="#2ecc71",
            decreasing_line_color="#e74c3c",
            name=sig_sym,
        )])

        # Overlay EMA cloud (fast 8/9, slow 34/50)
        import numpy as np
        for period, color, name in [(8, "#f39c12", "EMA8"), (9, "#e67e22", "EMA9"),
                                     (34, "#9b59b6", "EMA34"), (50, "#8e44ad", "EMA50")]:
            ema = candles["close"].ewm(span=period, adjust=False).mean()
            fig_candle.add_trace(go.Scatter(
                x=candles["ts"], y=ema, mode="lines",
                line=dict(width=1, color=color), name=name, opacity=0.6,
            ))

        fig_candle.update_layout(
            title=f"{sig_sym} — {sig_tf} candles (last {sig_limit} bars)",
            xaxis_rangeslider_visible=False,
            height=480,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_candle, width='stretch')

        # Volume bar chart
        fig_vol = go.Figure()
        vol_colors = ["#2ecc71" if c >= o else "#e74c3c"
                      for c, o in zip(candles["close"], candles["open"])]
        fig_vol.add_trace(go.Bar(x=candles["ts"], y=candles["volume"],
                                  marker_color=vol_colors, name="Volume"))
        avg_vol = candles["volume"].rolling(10).mean()
        fig_vol.add_trace(go.Scatter(x=candles["ts"], y=avg_vol,
                                      mode="lines", line=dict(color="#3498db", width=1.5),
                                      name="Avg(10)"))
        fig_vol.update_layout(
            title="Volume", height=180,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_vol, width='stretch')

        # Latest candle stats
        last = candles.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Close", f"${last['close']:,.2f}")
        prev = candles.iloc[-2]["close"] if len(candles) > 1 else last["close"]
        c2.metric("Change", f"${last['close'] - prev:+.2f}", delta=f"{(last['close']-prev)/prev*100:+.3f}%")
        c3.metric("High / Low", f"${last['high']:,.2f} / ${last['low']:,.2f}")
        avg_v = candles["volume"].rolling(10).mean().iloc[-1]
        c4.metric("Vol Ratio", f"{last['volume'] / avg_v:.2f}x")


# ===========================================================================
# TAB 3 — ML MODELS
# ===========================================================================
with tab_model:
    st.header("ML Model & Registry")

    # Registry table
    st.subheader("Model Registry")
    reg_rows = _load_registry(REGISTRY_PATH)
    if reg_rows:
        reg_df = pd.json_normalize(reg_rows)
        for col in ("new_score", "current_score"):
            if col in reg_df.columns:
                reg_df[col] = reg_df[col].map(lambda v: f"{v:.4f}" if v is not None else "—")
        st.dataframe(reg_df, width='stretch', hide_index=True)
    else:
        st.info("No registry entries found.")

    # Feature importance
    st.subheader("Feature Importances")
    model_info = _load_model_info(MODEL_PATH)
    if "error" in model_info:
        st.error(model_info["error"])
    else:
        m1, m2 = st.columns(2)
        m1.metric("Model type", model_info.get("model_type", "unknown"))
        m2.metric("Feature count", model_info.get("features", "?"))

        if "importances" in model_info:
            imp_df = (
                pd.DataFrame({
                    "feature": list(model_info["importances"].keys()),
                    "importance": list(model_info["importances"].values()),
                })
                .sort_values("importance", ascending=True)
                .tail(43)
            )
            fig_imp = px.bar(
                imp_df, x="importance", y="feature", orientation="h",
                title="Feature Importance", height=700,
                color="importance", color_continuous_scale=["#3498db", "#e74c3c"],
            )
            fig_imp.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_imp, width='stretch')

    # Training commands
    st.subheader("Retrain commands")
    st.code(
        "# Standard retrain (last 25k bars, XGBoost)\n"
        "python -m hogan_bot.retrain --from-db --window-bars 25000 --model-type xgboost --horizon-bars 12\n\n"
        "# With paper trade feedback labels (after 50+ closed trades)\n"
        "python -m hogan_bot.retrain --from-db --window-bars 25000 --model-type xgboost --use-paper-labels\n\n"
        "# Force promote (override score check)\n"
        "python -m hogan_bot.retrain --from-db --window-bars 25000 --model-type xgboost --force-promote",
        language="bash",
    )


# ===========================================================================
# TAB 4 — DATA COVERAGE
# ===========================================================================
with tab_data:
    st.header("Data Coverage")

    onchain = _load_onchain_latest(DB_PATH)

    if onchain.empty:
        st.warning("No onchain_metrics data found.")
    else:
        # Group by metric category
        st.subheader(f"Latest External Metrics ({len(onchain)} metrics)")
        st.dataframe(onchain, width='stretch', hide_index=True)

    # Candle coverage
    st.subheader("OHLCV Candle Coverage")
    try:
        conn = sqlite3.connect(DB_PATH)
        candle_cov = pd.read_sql_query(
            """SELECT symbol, timeframe, COUNT(*) as candles,
                      date(MIN(ts_ms)/1000, 'unixepoch') as earliest,
                      date(MAX(ts_ms)/1000, 'unixepoch') as latest
               FROM candles GROUP BY symbol, timeframe ORDER BY symbol, timeframe""",
            conn,
        )
        conn.close()
        st.dataframe(candle_cov, width='stretch', hide_index=True)
    except Exception as e:
        st.error(f"Could not load candle coverage: {e}")

    # Refresh commands
    st.subheader("Refresh commands")
    st.code(
        "# Run daily data refresh\npython refresh_daily.py\n\n"
        "# Backfill 1yr of 5m candles via Alpaca\n"
        "python -m hogan_bot.fetch_alpaca --backfill-bars 100000 --symbol BTC/USD\n"
        "python -m hogan_bot.fetch_alpaca --backfill-bars 100000 --symbol ETH/USD",
        language="bash",
    )


# ===========================================================================
# TAB 5 — BACKTEST
# ===========================================================================
with tab_backtest:
    st.header("Quick Backtest")

    col1, col2, col3 = st.columns(3)
    bt_symbol = col1.selectbox("Symbol", ["BTC/USD", "ETH/USD"], key="bt_sym")
    bt_tf = col2.selectbox("Timeframe", ["1h", "4h", "1d"], key="bt_tf")
    bt_limit = col3.number_input("Bars", value=2000, min_value=200, step=200, key="bt_lim")

    opt1, opt2, opt3 = st.columns(3)
    bt_use_ml = opt1.checkbox("Use ML filter", value=True, key="bt_ml")
    bt_shorts = opt2.checkbox("Enable shorts", value=True, key="bt_shorts")
    bt_ml_sizer = opt3.checkbox("ML as sizer", value=False, key="bt_sizer")

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                from hogan_bot.backtest import run_backtest_on_candles
                from hogan_bot.config import load_config
                from hogan_bot.storage import get_connection, load_candles

                cfg = load_config()
                conn_bt = get_connection(DB_PATH)
                bt_candles = load_candles(conn_bt, bt_symbol, bt_tf, limit=int(bt_limit))
                conn_bt.close()

                if bt_candles.empty:
                    st.warning(f"No candles found for {bt_symbol}/{bt_tf}.")
                else:
                    ml_model = None
                    if bt_use_ml and Path(MODEL_PATH).exists():
                        import pickle
                        with open(MODEL_PATH, "rb") as f:
                            ml_model = pickle.load(f)

                    result = run_backtest_on_candles(
                        candles=bt_candles,
                        symbol=bt_symbol,
                        timeframe=bt_tf,
                        starting_balance_usd=10_000.0,
                        aggressive_allocation=cfg.aggressive_allocation,
                        max_risk_per_trade=cfg.max_risk_per_trade,
                        max_drawdown=cfg.max_drawdown,
                        short_ma_window=cfg.short_ma_window,
                        long_ma_window=cfg.long_ma_window,
                        volume_window=cfg.volume_window,
                        volume_threshold=cfg.volume_threshold,
                        fee_rate=cfg.fee_rate,
                        ml_model=ml_model,
                        ml_buy_threshold=cfg.ml_buy_threshold,
                        ml_sell_threshold=cfg.ml_sell_threshold,
                        ml_confidence_sizing=cfg.ml_confidence_sizing,
                        use_ema_clouds=cfg.use_ema_clouds,
                        use_ict=cfg.use_ict,
                        signal_mode=cfg.signal_mode,
                        min_vote_margin=cfg.signal_min_vote_margin,
                        trailing_stop_pct=cfg.trailing_stop_pct,
                        take_profit_pct=cfg.take_profit_pct,
                        trail_activation_pct=cfg.trail_activation_pct,
                        enable_shorts=bt_shorts,
                        use_ml_as_sizer=bt_ml_sizer,
                        max_hold_hours=cfg.max_hold_hours,
                        short_max_hold_hours=cfg.short_max_hold_hours,
                        min_edge_multiple=cfg.min_edge_multiple,
                        min_final_confidence=cfg.min_final_confidence,
                        min_tech_confidence=cfg.min_tech_confidence,
                        min_regime_confidence=cfg.min_regime_confidence,
                        max_whipsaws=cfg.max_whipsaws,
                        reversal_confidence_mult=cfg.reversal_confidence_multiplier,
                        db_path=DB_PATH,
                    )

                    summary = result.summary_dict()
                    m1, m2, m3, m4, m5 = st.columns(5)
                    ret = summary.get("total_return_pct", 0)
                    m1.metric("Return", f"{ret:.2f}%", delta="profit" if ret > 0 else "loss")
                    m2.metric("Sharpe", f"{summary.get('sharpe_ratio', 0):.3f}")
                    m3.metric("Max DD", f"{summary.get('max_drawdown_pct', 0):.2f}%")
                    m4.metric("Trades", summary.get("total_trades", 0))
                    m5.metric("Win Rate", f"{summary.get('win_rate_pct', 0):.1f}%")

                    with st.expander("Full metrics"):
                        st.json(summary)

                    if result.equity_curve:
                        eq_df = pd.DataFrame({
                            "bar": range(len(result.equity_curve)),
                            "equity_usd": result.equity_curve,
                        })
                        fig_bt = px.line(eq_df, x="bar", y="equity_usd", title="Backtest Equity Curve")
                        fig_bt.update_layout(
                            height=300, margin=dict(l=0, r=0, t=30, b=0),
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_bt, width='stretch')

            except Exception as exc:
                st.error(f"Backtest failed: {exc}")

# ===========================================================================
# TAB 7 — SWARM DECISION LAYER
# ===========================================================================
with tab_swarm:
    st.header("Swarm Decision Layer")

    try:
        _sw_conn = sqlite3.connect(DB_PATH)

        _sw_has_tables = True
        try:
            _sw_conn.execute("SELECT 1 FROM swarm_decisions LIMIT 1")
        except Exception:
            _sw_has_tables = False

        if not _sw_has_tables:
            st.info("Swarm tables not found in this database. Enable swarm mode to start logging decisions.")
        else:
            _sw_count = _sw_conn.execute("SELECT COUNT(*) FROM swarm_decisions").fetchone()[0]

            if _sw_count == 0:
                st.metric("Total Swarm Decisions", 0)
                st.info("No swarm decisions logged yet. Run the bot with HOGAN_SWARM_ENABLED=true to start.")
            else:
                # ── Section 1: Live Swarm Snapshot ────────────────────
                st.subheader("Live Swarm Snapshot")
                try:
                    from hogan_bot.swarm_observability import load_latest_swarm_decision
                    _snap = load_latest_swarm_decision(_sw_conn)
                    if not _snap.empty:
                        _s = _snap.iloc[0]
                        snap_cols = st.columns(6)
                        snap_cols[0].metric("Decisions", _sw_count)
                        snap_cols[1].metric("Mode", _s.get("mode", "—"))
                        snap_cols[2].metric("Action", _s.get("final_action", "—"))
                        snap_cols[3].metric("Confidence", f"{_s.get('final_conf', 0):.0%}")
                        snap_cols[4].metric("Agreement", f"{_s.get('agreement', 0):.0%}")
                        _vetoed = "Yes" if _s.get("vetoed") else "No"
                        snap_cols[5].metric("Vetoed", _vetoed)

                        _snap_ts = pd.to_datetime(_s.get("ts_ms", 0), unit="ms", utc=True)
                        _age = pd.Timestamp.now("UTC") - _snap_ts
                        _age_str = f"{_age.total_seconds() / 60:.0f}m ago" if _age.total_seconds() < 3600 else f"{_age.total_seconds() / 3600:.1f}h ago"
                        st.caption(f"Last update: {_snap_ts.strftime('%Y-%m-%d %H:%M UTC')} ({_age_str}) | Symbol: {_s.get('symbol', '?')} | TF: {_s.get('timeframe', '?')}")
                    else:
                        st.metric("Total Swarm Decisions", _sw_count)
                except Exception:
                    st.metric("Total Swarm Decisions", _sw_count)

                st.divider()

                col_a, col_b = st.columns(2)

                with col_a:
                    st.subheader("Consensus Over Time")
                    df_agree = pd.read_sql_query(
                        "SELECT ts_ms, agreement, entropy, final_action "
                        "FROM swarm_decisions ORDER BY ts_ms DESC LIMIT 200",
                        _sw_conn,
                    )
                    if not df_agree.empty:
                        df_agree["ts"] = pd.to_datetime(df_agree["ts_ms"], unit="ms")
                        import plotly.graph_objects as go
                        fig_sw = go.Figure()
                        fig_sw.add_trace(go.Scatter(
                            x=df_agree["ts"], y=df_agree["agreement"],
                            name="Agreement", line=dict(color="#00cc96"),
                        ))
                        fig_sw.add_trace(go.Scatter(
                            x=df_agree["ts"], y=df_agree["entropy"],
                            name="Entropy", line=dict(color="#ef553b"),
                        ))
                        fig_sw.update_layout(
                            height=300, margin=dict(l=20, r=20, t=30, b=20),
                            yaxis_title="Score",
                        )
                        st.plotly_chart(fig_sw, width='stretch')

                with col_b:
                    st.subheader("Weight History")
                    df_wt = pd.read_sql_query(
                        "SELECT ts_ms, weights_json, source "
                        "FROM swarm_weight_snapshots ORDER BY ts_ms DESC LIMIT 100",
                        _sw_conn,
                    )
                    if not df_wt.empty:
                        import json as _json
                        df_wt["ts"] = pd.to_datetime(df_wt["ts_ms"], unit="ms")
                        _weight_records = []
                        for _, row in df_wt.iterrows():
                            w = _json.loads(row["weights_json"])
                            for agent, weight in w.items():
                                _weight_records.append({
                                    "ts": row["ts"], "agent": agent, "weight": weight,
                                })
                        if _weight_records:
                            df_wr = pd.DataFrame(_weight_records)
                            import plotly.express as px
                            fig_wt = px.line(df_wr, x="ts", y="weight", color="agent")
                            fig_wt.update_layout(
                                height=300, margin=dict(l=20, r=20, t=30, b=20),
                            )
                            st.plotly_chart(fig_wt, width='stretch')
                    else:
                        st.info("No weight snapshots yet.")

                st.subheader("Top Veto Reasons")
                df_vetoes = pd.read_sql_query(
                    "SELECT agent_id, block_reasons_json "
                    "FROM swarm_agent_votes WHERE veto = 1 "
                    "ORDER BY ts_ms DESC LIMIT 500",
                    _sw_conn,
                )
                if not df_vetoes.empty:
                    import json as _json
                    _reason_counts: dict[str, int] = {}
                    for _, row in df_vetoes.iterrows():
                        reasons = _json.loads(row["block_reasons_json"])
                        for r in reasons:
                            key = f"{row['agent_id']}: {r}"
                            _reason_counts[key] = _reason_counts.get(key, 0) + 1
                    if _reason_counts:
                        df_rc = pd.DataFrame(
                            sorted(_reason_counts.items(), key=lambda x: -x[1])[:15],
                            columns=["Reason", "Count"],
                        )
                        st.bar_chart(df_rc.set_index("Reason"))
                else:
                    st.info("No veto events logged yet.")

                st.subheader("Shadow vs Baseline Divergence")
                df_div = pd.read_sql_query(
                    "SELECT ts_ms, final_action, mode, agreement "
                    "FROM swarm_decisions WHERE mode = 'shadow' "
                    "ORDER BY ts_ms DESC LIMIT 100",
                    _sw_conn,
                )
                if not df_div.empty:
                    df_div["ts"] = pd.to_datetime(df_div["ts_ms"], unit="ms")
                    st.dataframe(
                        df_div[["ts", "final_action", "agreement"]].head(20),
                        width='stretch',
                    )
                else:
                    st.info("No shadow decisions logged yet.")

                # ── Panel A: Agent Voting Board ──────────────────────────
                st.subheader("Agent Voting Board")
                _vb_bars = st.slider("Recent bars", 10, 100, 25, key="vb_bars")
                df_vb = _load_agent_votes(DB_PATH, limit=_vb_bars * 10)
                if not df_vb.empty:
                    pivot = df_vb.pivot_table(
                        index="ts", columns="agent_id",
                        values="action", aggfunc="first",
                    ).tail(_vb_bars)
                    action_map = {"buy": 1.0, "sell": -1.0, "hold": 0.0}
                    pivot_num = pivot.map(lambda x: action_map.get(str(x), 0.0))
                    fig_vb = px.imshow(
                        pivot_num.T,
                        x=[t.strftime("%m-%d %H:%M") for t in pivot_num.index],
                        y=list(pivot_num.columns),
                        color_continuous_scale=[[0, "#e74c3c"], [0.5, "#95a5a6"], [1, "#2ecc71"]],
                        zmin=-1, zmax=1,
                        labels={"color": "Action (buy=1, sell=-1)"},
                        aspect="auto",
                    )
                    fig_vb.update_layout(
                        height=250, margin=dict(l=0, r=0, t=10, b=0),
                    )
                    st.plotly_chart(fig_vb, width='stretch')
                else:
                    st.info("No agent votes with decision_id found.")

                # ── Panel B: Veto Ledger ─────────────────────────────────
                st.subheader("Veto Ledger")
                df_vl = _load_veto_ledger(DB_PATH)
                if not df_vl.empty:
                    vl_m1, vl_m2 = st.columns(2)
                    vl_m1.metric("Total Vetoes", len(df_vl))
                    all_reasons: dict[str, int] = {}
                    for _, r in df_vl.iterrows():
                        for reason in json.loads(r["block_reasons_json"]):
                            all_reasons[reason] = all_reasons.get(reason, 0) + 1
                    if all_reasons:
                        top3 = sorted(all_reasons.items(), key=lambda x: -x[1])[:3]
                        vl_m2.metric("Top Reason", top3[0][0] if top3 else "—")
                        fig_vr = px.bar(
                            x=[r for r, _ in top3], y=[c for _, c in top3],
                            labels={"x": "Reason", "y": "Count"},
                            title="Top 3 Veto Reasons", height=200,
                        )
                        fig_vr.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig_vr, width='stretch')
                    st.dataframe(
                        df_vl[["ts", "agent_id", "reasons", "swarm_action", "agreement"]].head(50),
                        width='stretch', hide_index=True,
                    )
                else:
                    st.info("No vetoes logged yet.")

                # ── Panel C: Replay by Decision ──────────────────────────
                st.subheader("Replay by Decision")
                df_recent = pd.read_sql_query(
                    "SELECT id, ts_ms, final_action, symbol, agreement, vetoed "
                    "FROM swarm_decisions ORDER BY ts_ms DESC LIMIT 50",
                    _sw_conn,
                )
                if not df_recent.empty:
                    df_recent["label"] = (
                        pd.to_datetime(df_recent["ts_ms"], unit="ms").dt.strftime("%m-%d %H:%M")
                        + " | " + df_recent["final_action"]
                    )
                    chosen = st.selectbox(
                        "Select decision", df_recent["label"].tolist(), key="replay_sel",
                    )
                    chosen_idx = df_recent[df_recent["label"] == chosen].index[0]
                    chosen_id = int(df_recent.loc[chosen_idx, "id"])
                    detail = _load_decision_detail(DB_PATH, chosen_id)

                    dec_df = detail["decision"]
                    votes_df = detail["votes"]
                    base_df = detail["baseline"]

                    if not dec_df.empty:
                        d = dec_df.iloc[0]
                        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
                        rc1.metric("Action", d.get("final_action", "—"))
                        rc2.metric("Confidence", f"{d.get('confidence', 0):.2f}")
                        rc3.metric("Agreement", f"{d.get('agreement', 0):.2f}")
                        rc4.metric("Entropy", f"{d.get('entropy', 0):.3f}")
                        rc5.metric("Vetoed", "Yes" if d.get("vetoed") else "No")

                    if not votes_df.empty:
                        st.markdown("**Per-Agent Votes**")
                        votes_df["reasons"] = votes_df["block_reasons_json"].apply(
                            lambda x: ", ".join(json.loads(x)) if x else ""
                        )
                        st.dataframe(
                            votes_df[["agent_id", "action", "confidence", "size_scale", "veto", "reasons"]],
                            width='stretch', hide_index=True,
                        )

                    if not base_df.empty:
                        st.markdown("**Baseline Comparison**")
                        b = base_df.iloc[0]
                        bc1, bc2, bc3 = st.columns(3)
                        bc1.metric("Baseline Action", b.get("final_action", "—"))
                        bc2.metric("Baseline Confidence", f"{b.get('final_confidence', 0):.2f}")
                        bc3.metric("Regime", b.get("regime", "—"))
                    else:
                        st.info("No matching baseline decision found for this timestamp.")

                    # Decision Story (plain-English)
                    try:
                        from hogan_bot.swarm_replay import render_decision_story
                        if not dec_df.empty:
                            _bl_row = base_df.iloc[0] if not base_df.empty else None
                            story = render_decision_story(
                                dec_df.iloc[0], votes=votes_df, baseline=_bl_row,
                            )
                            with st.expander("Decision Story (plain English)", expanded=False):
                                st.markdown(story)
                    except Exception:
                        pass

                    # Raw JSON
                    if not dec_df.empty:
                        with st.expander("Raw Decision JSON", expanded=False):
                            _dj = dec_df.iloc[0].get("decision_json", "{}")
                            try:
                                st.json(json.loads(_dj) if isinstance(_dj, str) else _dj)
                            except Exception:
                                st.code(_dj)
                else:
                    st.info("No swarm decisions to replay.")

                # ── Panel D: Promotion Readiness Card ────────────────────
                st.subheader("Promotion Readiness")
                promo = _load_promotion_stats(DB_PATH)
                pr1, pr2, pr3, pr4 = st.columns(4)
                _targets = {
                    "shadow_decisions": 300, "would_trade": 100,
                    "veto_events": 50, "distinct_regimes": 3,
                }
                pr1.metric("Shadow Decisions", f"{promo['shadow_decisions']} / {_targets['shadow_decisions']}")
                pr2.metric("Would-Trade", f"{promo['would_trade']} / {_targets['would_trade']}")
                pr3.metric("Veto Events", f"{promo['veto_events']} / {_targets['veto_events']}")
                pr4.metric("Regime Coverage", f"{promo['distinct_regimes']} / {_targets['distinct_regimes']}")

                all_met = all(
                    promo[k] >= v for k, v in _targets.items()
                )
                for key, target in _targets.items():
                    pct = min(promo[key] / target, 1.0) if target else 0.0
                    st.progress(pct, text=f"{key}: {promo[key]}/{target}")

                if all_met:
                    st.success("READY — All shadow sample targets met. Consider running promotion_check.py.")
                else:
                    st.warning("COLLECTING — Shadow sample targets not yet met.")
                st.metric("Mean Agreement", f"{promo['mean_agreement']:.3f}")

                # ── Section 5: Learning & Drift ──────────────────────
                st.subheader("Learning & Drift")
                try:
                    from hogan_bot.swarm_observability import (
                        load_swarm_decisions, load_swarm_score_calibration,
                    )
                    from hogan_bot.swarm_metrics import (
                        compute_disagreement_stats, compute_trade_density,
                        compute_agent_leaderboard,
                    )

                    _drift_decisions = load_swarm_decisions(_sw_conn, limit=500)

                    if not _drift_decisions.empty:
                        # Disagreement stats
                        dis_stats = compute_disagreement_stats(_drift_decisions)
                        ds1, ds2, ds3 = st.columns(3)
                        ds1.metric("Mean Agreement", f"{dis_stats['mean_agreement']:.0%}")
                        ds2.metric("Mean Entropy", f"{dis_stats['mean_entropy']:.3f}")
                        ds3.metric("High Disagreement %", f"{dis_stats['high_disagreement_pct']:.1%}")

                        # Trade density chart
                        density = compute_trade_density(_drift_decisions, bucket_hours=24)
                        if not density.empty:
                            fig_density = go.Figure()
                            fig_density.add_trace(go.Bar(
                                x=density["bucket_start"], y=density["trades"],
                                name="Trades", marker_color="#2ecc71",
                            ))
                            fig_density.add_trace(go.Bar(
                                x=density["bucket_start"], y=density["holds"],
                                name="Holds", marker_color="#95a5a6",
                            ))
                            fig_density.update_layout(
                                title="Trade Density (24h buckets)",
                                barmode="stack", height=250,
                                margin=dict(l=0, r=0, t=30, b=0),
                            )
                            st.plotly_chart(fig_density, width='stretch')

                        # Agent leaderboard
                        _lb_votes = _load_agent_votes(DB_PATH, limit=2000)
                        if not _lb_votes.empty:
                            leaderboard = compute_agent_leaderboard(_lb_votes)
                            if not leaderboard.empty:
                                st.markdown("**Agent Leaderboard**")
                                lb_display = leaderboard[[
                                    c for c in ["agent_id", "vote_count", "veto_count",
                                                "veto_rate", "mean_confidence",
                                                "buys", "sells", "holds"]
                                    if c in leaderboard.columns
                                ]]
                                st.dataframe(lb_display, width='stretch', hide_index=True)

                        # Score calibration (if outcomes exist)
                        try:
                            cal_df = load_swarm_score_calibration(_sw_conn)
                            if not cal_df.empty and "forward_60m_bps" in cal_df.columns:
                                _has_outcomes = cal_df["forward_60m_bps"].notna().any()
                                if _has_outcomes:
                                    from hogan_bot.swarm_metrics import compute_opportunity_monotonicity
                                    mono = compute_opportunity_monotonicity(cal_df)
                                    st.markdown("**Score Calibration**")
                                    mc1, mc2 = st.columns(2)
                                    mc1.metric("Score-Return Correlation", f"{mono['correlation']:.3f}")
                                    mc2.metric("Monotonic", "Yes" if mono['monotonic'] else "No")
                                    if mono["bins"]:
                                        bins_df = pd.DataFrame(mono["bins"])
                                        fig_cal = px.bar(
                                            bins_df, x="bin", y="mean_return",
                                            title="Mean Forward Return by Confidence Bin",
                                            height=200,
                                        )
                                        fig_cal.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                                        st.plotly_chart(fig_cal, width='stretch')
                        except Exception:
                            pass
                    else:
                        st.info("Not enough decisions for drift analysis.")
                except Exception as _drift_err:
                    st.info(f"Learning & Drift data unavailable: {_drift_err}")

                # ── Persisted Promotion Reports ──────────────────────
                try:
                    from hogan_bot.swarm_observability import load_swarm_promotion_status
                    _promo_report = load_swarm_promotion_status(_sw_conn)
                    if not _promo_report.empty:
                        st.subheader("Latest Promotion Report")
                        _pr = _promo_report.iloc[0]
                        pr1, pr2, pr3 = st.columns(3)
                        pr1.metric("Phase", _pr.get("phase", "—"))
                        pr2.metric("Recommendation", _pr.get("recommendation", "—"))
                        _bl_count = len(json.loads(_pr.get("blockers_json", "[]")))
                        pr3.metric("Blockers", _bl_count)
                        with st.expander("Full Report"):
                            st.text(_pr.get("summary", "No summary available."))
                            try:
                                st.json(json.loads(_pr.get("gates_json", "[]")))
                            except Exception:
                                pass
                except Exception:
                    pass

            # ── Section: Daily Digest ────────────────────────
            try:
                st.subheader("Daily Digest")
                from hogan_bot.swarm_daily_digest import build_digest
                _digest = build_digest(_sw_conn)
                _sev_colors = {"healthy": "green", "watch": "blue", "warning": "orange", "critical": "red"}
                _sev_color = _sev_colors.get(_digest.severity, "gray")
                st.markdown(f"**Severity:** :{_sev_color}[{_digest.severity.upper()}]")
                st.markdown(f"> {_digest.headline}")

                dig_cols = st.columns(4)
                dig_cols[0].metric("Decisions", _digest.metrics.get("decision_count", 0))
                dig_cols[1].metric("Would-Trades", _digest.metrics.get("would_trade_count", 0))
                _vr_pct = f"{_digest.metrics.get('veto_ratio', 0):.0%}"
                dig_cols[2].metric("Veto Ratio", _vr_pct)
                dig_cols[3].metric("Regimes", _digest.metrics.get("distinct_regimes", 0))

                if _digest.flags:
                    with st.expander(f"Flags ({len(_digest.flags)})", expanded=True):
                        for _f in _digest.flags:
                            _icon = {"critical": "🔴", "warning": "🟡", "watch": "🔵"}.get(_f.level, "⚪")
                            st.markdown(f"{_icon} **[{_f.level.upper()}]** {_f.message}")

                if _digest.operator_actions:
                    with st.expander("Operator Actions Today", expanded=True):
                        for _i, _a in enumerate(_digest.operator_actions, 1):
                            st.markdown(f"{_i}. {_a}")

                if _digest.replay_candidates:
                    with st.expander(f"Replay Shortlist ({len(_digest.replay_candidates)})"):
                        for _rc in _digest.replay_candidates[:8]:
                            st.markdown(f"- **#{_rc.decision_id}** {_rc.symbol} {_rc.ts_iso} — {_rc.reason}")

                with st.expander("Full Markdown Digest"):
                    st.markdown(_digest.summary_md)
            except Exception as _dig_err:
                st.info(f"Daily Digest unavailable: {_dig_err}")

            # ── Section: Weekly Review ────────────────────────
            try:
                st.subheader("Weekly Review")
                from hogan_bot.swarm_weekly_review import build_weekly_review
                _wreview = build_weekly_review(_sw_conn)
                _wr_colors = {"healthy": "green", "watch": "blue", "warning": "orange", "critical": "red"}
                _wr_color = _wr_colors.get(_wreview.severity, "gray")
                st.markdown(f"**Severity:** :{_wr_color}[{_wreview.severity.upper()}]  **Recommendation:** {_wreview.recommendation}")
                st.markdown(f"> {_wreview.headline}")

                wr_cols = st.columns(5)
                wr_cols[0].metric("Decisions", _wreview.metrics.get("decision_count", 0))
                wr_cols[1].metric("Would-Trades", _wreview.metrics.get("would_trade_count", 0))
                wr_cols[2].metric("Vetoes", _wreview.metrics.get("veto_count", 0))
                _wvr = f"{_wreview.metrics.get('veto_ratio', 0):.0%}"
                wr_cols[3].metric("Veto Ratio", _wvr)
                wr_cols[4].metric("Regimes", _wreview.metrics.get("distinct_regimes", 0))

                _dom = _wreview.metrics.get("dominant_veto_agent")
                if _dom:
                    _dom_share = _wreview.metrics.get("dominant_veto_agent_share", 0)
                    st.warning(f"Dominant veto agent: **{_dom}** ({_dom_share:.0%} of vetoes)")

                if _wreview.flags:
                    with st.expander(f"Flags ({len(_wreview.flags)})", expanded=True):
                        for _wf in _wreview.flags:
                            _wicon = {"critical": "🔴", "warning": "🟡", "watch": "🔵"}.get(_wf.level, "⚪")
                            st.markdown(f"{_wicon} **[{_wf.level.upper()}]** {_wf.message}")

                if _wreview.agent_scores:
                    with st.expander("Agent Leaderboard"):
                        import pandas as _wr_pd
                        _al_data = [{
                            "Agent": a.agent_id, "Decisions": a.decisions, "Vetoes": a.vetoes,
                            "Hold Rate": f"{a.hold_rate:.0%}",
                            "Confidence": f"{a.mean_confidence:.2f}" if a.mean_confidence else "—",
                        } for a in _wreview.agent_scores]
                        st.dataframe(_wr_pd.DataFrame(_al_data), width='stretch')

                _wow_avail = _wreview.metrics.get("prior_week_available", False)
                if _wow_avail:
                    with st.expander("Week-over-Week Deltas"):
                        wow_cols = st.columns(3)
                        _d_delta = _wreview.metrics.get("decision_count_wow_delta", 0)
                        _wt_delta = _wreview.metrics.get("would_trade_wow_delta", 0)
                        _vr_delta = _wreview.metrics.get("veto_ratio_wow_delta", 0)
                        wow_cols[0].metric("Decisions", _wreview.metrics.get("decision_count", 0), delta=_d_delta)
                        wow_cols[1].metric("Would-Trades", _wreview.metrics.get("would_trade_count", 0), delta=_wt_delta)
                        wow_cols[2].metric("Veto Ratio", f"{_wreview.metrics.get('veto_ratio', 0):.1%}", delta=f"{_vr_delta:+.1%}", delta_color="inverse")

                if _wreview.operator_actions:
                    with st.expander("Operator Actions This Week", expanded=True):
                        for _wi, _wa in enumerate(_wreview.operator_actions, 1):
                            st.markdown(f"{_wi}. {_wa}")

                if _wreview.cursor_actions:
                    with st.expander("Cursor Actions This Week"):
                        for _wi, _wa in enumerate(_wreview.cursor_actions, 1):
                            st.markdown(f"{_wi}. {_wa}")

                if _wreview.replay_candidates:
                    with st.expander(f"Weekly Replay Shortlist ({len(_wreview.replay_candidates)})"):
                        for _wrc in _wreview.replay_candidates[:10]:
                            st.markdown(f"- **#{_wrc.decision_id}** [{_wrc.category}] {_wrc.symbol} {_wrc.ts_iso} — {_wrc.reason}")

                with st.expander("Full Markdown Review"):
                    st.markdown(_wreview.summary_md)
            except Exception as _wr_err:
                st.info(f"Weekly Review unavailable: {_wr_err}")

            # ── Section: Thresholds & Quarantine ─────────────────
            try:
                st.subheader("Thresholds & Quarantine")
                from hogan_bot.storage import _create_schema as _tq_schema
                _tq_schema(_sw_conn)

                # Panel 1 — Stall Status
                from hogan_bot.stall_detection import get_latest_stall_alerts, compute_stall_summary
                _stall_status = compute_stall_summary(_sw_conn)
                _stall_colors = {"healthy": "green", "info": "blue", "warning": "orange", "critical": "red"}
                st.markdown(f"**Stall Status:** :{_stall_colors.get(_stall_status, 'gray')}[{_stall_status.upper()}]")

                _stall_alerts = get_latest_stall_alerts(_sw_conn, limit=5)
                if _stall_alerts:
                    for _sa in _stall_alerts:
                        _sa_icon = {"critical": "🔴", "warn": "🟡", "info": "🔵"}.get(_sa["severity"], "⚪")
                        st.markdown(f"{_sa_icon} **{_sa['code']}** — {_sa['notes']}")

                # Panel 2 — Agent Mode Control
                from hogan_bot.agent_quarantine import get_all_agent_modes
                _all_modes = get_all_agent_modes(_sw_conn)
                if _all_modes:
                    with st.expander("Agent Modes", expanded=True):
                        import pandas as _tq_pd
                        _mode_data = [
                            {"Agent": s.agent_id, "Mode": s.mode, "Reason": s.reason,
                             "Operator": s.operator, "Changed": s.changed_at[:19] if s.changed_at else "—"}
                            for s in _all_modes.values()
                        ]
                        st.dataframe(_tq_pd.DataFrame(_mode_data), width='stretch')
                else:
                    st.info("No agent mode overrides set. All agents running in `active` mode.")

                # Panel 3 — Pre-veto vs Post-veto
                try:
                    _pv_row = _sw_conn.execute(
                        """SELECT
                            SUM(CASE WHEN pre_veto_action IN ('buy','sell') THEN 1 ELSE 0 END),
                            SUM(CASE WHEN final_action IN ('buy','sell') AND vetoed=0 THEN 1 ELSE 0 END),
                            AVG(pre_veto_agreement), AVG(agreement),
                            AVG(pre_veto_confidence), AVG(final_conf),
                            COUNT(*)
                           FROM swarm_decisions WHERE pre_veto_action IS NOT NULL""",
                    ).fetchone()
                    if _pv_row and _pv_row[6] and _pv_row[6] > 0:
                        with st.expander("Pre-Veto vs Post-Veto"):
                            pv_cols = st.columns(3)
                            pv_cols[0].metric("Pre-Veto Trades", _pv_row[0] or 0)
                            pv_cols[0].metric("Post-Veto Trades", _pv_row[1] or 0)
                            pv_cols[1].metric("Pre-Veto Agreement", f"{_pv_row[2]:.3f}" if _pv_row[2] else "—")
                            pv_cols[1].metric("Post-Veto Agreement", f"{_pv_row[3]:.3f}" if _pv_row[3] else "—")
                            pv_cols[2].metric("Pre-Veto Confidence", f"{_pv_row[4]:.3f}" if _pv_row[4] else "—")
                            pv_cols[2].metric("Post-Veto Confidence", f"{_pv_row[5]:.3f}" if _pv_row[5] else "—")
                except Exception:
                    pass

                # Panel 4 — Dominant Veto Agents
                try:
                    _dom_rows = _sw_conn.execute(
                        """SELECT sav.agent_id, COUNT(*) as cnt
                           FROM swarm_agent_votes sav WHERE sav.veto = 1
                           GROUP BY sav.agent_id ORDER BY cnt DESC LIMIT 5""",
                    ).fetchall()
                    if _dom_rows:
                        _total_vetoes = sum(r[1] for r in _dom_rows)
                        with st.expander("Dominant Veto Agents"):
                            import pandas as _tq_pd2
                            _dom_data = [
                                {"Agent": r[0], "Vetoes": r[1],
                                 "Share": f"{r[1]/_total_vetoes:.0%}" if _total_vetoes else "—"}
                                for r in _dom_rows
                            ]
                            st.dataframe(_tq_pd2.DataFrame(_dom_data), width='stretch')
                except Exception:
                    pass

                # Panel 5 — Threshold Bundle History
                try:
                    from hogan_bot.threshold_registry import list_bundles, get_change_history
                    _tb_agents = _sw_conn.execute(
                        "SELECT DISTINCT agent_id FROM swarm_threshold_bundles ORDER BY agent_id",
                    ).fetchall()
                    if _tb_agents:
                        with st.expander("Threshold Bundles"):
                            for (_tb_aid,) in _tb_agents:
                                st.markdown(f"**{_tb_aid}**")
                                _bundles = list_bundles(_tb_aid, _sw_conn)
                                for _b in _bundles[:5]:
                                    _status = "ACTIVE" if _b.active else "inactive"
                                    st.markdown(f"- `{_b.bundle_id}` v{_b.version} ({_status}) — {_b.notes or 'no notes'}")
                except Exception:
                    pass

                # Panel 6 — Suggested Actions
                _suggest: list[str] = []
                if _stall_status == "critical":
                    _suggest.append("Review and relax dominant veto agent thresholds in shadow mode.")
                if _all_modes:
                    for _am_s in _all_modes.values():
                        if _am_s.mode == "active":
                            pass
                if any(a.get("code") == "DOMINANT_VETO_AGENT" for a in _stall_alerts):
                    _da = next((a for a in _stall_alerts if a["code"] == "DOMINANT_VETO_AGENT"), None)
                    if _da:
                        _suggest.append(f"Consider setting dominant agent to `no_veto` mode: {_da['notes']}")
                if any(a.get("code") == "REGIME_BLINDNESS" for a in _stall_alerts):
                    _suggest.append("Fix regime logging before trusting promotion readiness.")
                if _suggest:
                    with st.expander("Suggested Actions"):
                        for _si, _sa_txt in enumerate(_suggest, 1):
                            st.markdown(f"{_si}. {_sa_txt}")

            except Exception as _tq_err:
                st.info(f"Thresholds & Quarantine unavailable: {_tq_err}")

        _sw_conn.close()
    except Exception as exc:
        st.error(f"Error loading swarm data: {exc}")

# ===========================================================================
# TAB 8 — SWARM REPLAY
# ===========================================================================
with tab_replay:
    st.header("Swarm Replay")

    try:
        _rp_conn = sqlite3.connect(DB_PATH)
        _rp_has_tables = True
        try:
            _rp_conn.execute("SELECT 1 FROM swarm_decisions LIMIT 1")
        except Exception:
            _rp_has_tables = False

        if not _rp_has_tables:
            st.info("Swarm tables not found. Enable swarm mode to start logging decisions for replay.")
        else:
            # Ensure newer tables (swarm_attribution, swarm_outcomes, etc.) exist
            try:
                from hogan_bot.storage import _create_schema
                _create_schema(_rp_conn)
            except Exception:
                pass

            from hogan_bot.swarm_replay_queries import ReplayFilter, list_replay_decisions, get_replay_decision
            from hogan_bot.swarm_attribution import classify_outcome, compute_full_attribution, build_learning_note
            from hogan_bot.swarm_replay import render_decision_story

            # ── Zone A: Replay Selector ───────────────────────────
            st.subheader("Replay Selector")
            za1, za2, za3, za4 = st.columns(4)
            _rp_symbol = za1.text_input("Symbol", value="BTC/USD", key="rp_sym")
            _rp_source = za2.selectbox(
                "Source filter",
                ["all", "traded", "vetoed", "skipped", "swarm"],
                key="rp_source",
            )
            _rp_sort = za3.selectbox(
                "Sort by",
                ["latest", "highest_opportunity", "biggest_winner", "biggest_loser",
                 "highest_disagreement", "veto_events"],
                key="rp_sort",
            )
            _rp_limit = za4.number_input("Max results", min_value=10, max_value=500, value=100, key="rp_limit")

            flt = ReplayFilter(
                symbol=_rp_symbol if _rp_symbol else None,
                source=_rp_source if _rp_source != "all" else None,
                sort_by=_rp_sort,
                limit=int(_rp_limit),
            )
            decisions = list_replay_decisions(_rp_conn, flt)

            if not decisions:
                st.info("No swarm decisions found matching your filters.")
            else:
                # Build selector labels
                _labels = []
                for d in decisions:
                    ts = pd.to_datetime(d["ts_ms"], unit="ms").strftime("%m-%d %H:%M")
                    action = d.get("swarm_action", "?")
                    label_tag = d.get("attr_label") or d.get("outcome_label") or ""
                    _labels.append(f"#{d['id']} | {ts} | {action} | {label_tag}")

                chosen_label = st.selectbox("Select decision to replay", _labels, key="rp_dec_sel")
                chosen_idx = _labels.index(chosen_label)
                chosen_dec_id = decisions[chosen_idx]["id"]

                replay = get_replay_decision(_rp_conn, chosen_dec_id)

                if replay and replay["decision"]:
                    dec = replay["decision"]
                    votes_list = replay["votes"]
                    outcome = replay["outcome"]
                    attribution = replay["attribution"]
                    baseline = replay["baseline_compare"]
                    candles_df = replay["candles"]
                    similar = replay["similar_events"]

                    # Compute attribution on the fly if not persisted
                    if not attribution and outcome:
                        attr = compute_full_attribution(dec, outcome, baseline)
                        note = build_learning_note(
                            dec, votes_list, outcome, attr,
                        )
                        attr["learning_note"] = note
                        attribution = attr

                    # ── Zone B: Decision Summary Strip ────────────
                    st.subheader("Decision Summary")
                    _ts_str = pd.to_datetime(dec.get("ts_ms", 0), unit="ms").strftime("%Y-%m-%d %H:%M UTC")
                    zb_cols = st.columns(6)
                    zb_cols[0].metric("Timestamp", _ts_str)
                    zb_cols[1].metric("Symbol", dec.get("symbol", "—"))
                    zb_cols[2].metric("Swarm Action", dec.get("final_action", "—"))
                    zb_cols[3].metric("Confidence", f"{dec.get('final_conf', 0):.0%}")
                    zb_cols[4].metric("Agreement", f"{dec.get('agreement', 0):.0%}")
                    _out_label = (attribution or {}).get("outcome_label", "Pending")
                    zb_cols[5].metric("Outcome", _out_label)

                    zb2_cols = st.columns(5)
                    zb2_cols[0].metric("Mode", dec.get("mode", "—"))
                    _baseline_action = baseline.get("final_action", "—") if baseline else "—"
                    zb2_cols[1].metric("Baseline Action", _baseline_action)
                    zb2_cols[2].metric("Vetoed", "Yes" if dec.get("vetoed") else "No")
                    zb2_cols[3].metric("Entropy", f"{dec.get('entropy', 0):.3f}")
                    _fwd = outcome.get("forward_60m_bps") if outcome else None
                    zb2_cols[4].metric("60m Return", f"{_fwd:+.1f} bps" if _fwd is not None else "—")

                    st.divider()

                    # ── Zone C: Market State & Chart ──────────────
                    st.subheader("Market Context")
                    if not candles_df.empty and "close" in candles_df.columns:
                        import plotly.graph_objects as go
                        fig_rp = go.Figure()
                        candles_df["ts"] = pd.to_datetime(candles_df["ts_ms"], unit="ms")
                        fig_rp.add_trace(go.Candlestick(
                            x=candles_df["ts"],
                            open=candles_df["open"], high=candles_df["high"],
                            low=candles_df["low"], close=candles_df["close"],
                            name="Price",
                        ))
                        # Decision marker
                        dec_ts = pd.to_datetime(dec.get("ts_ms", 0), unit="ms")
                        dec_price = candles_df.loc[
                            candles_df["ts_ms"] == dec.get("ts_ms"), "close"
                        ]
                        if not dec_price.empty:
                            _marker_color = {"buy": "#2ecc71", "sell": "#e74c3c"}.get(
                                dec.get("final_action", "hold"), "#f39c12"
                            )
                            fig_rp.add_trace(go.Scatter(
                                x=[dec_ts], y=[float(dec_price.iloc[0])],
                                mode="markers", marker=dict(size=14, color=_marker_color, symbol="diamond"),
                                name=f"Decision: {dec.get('final_action', 'hold')}",
                            ))
                        fig_rp.update_layout(
                            height=350, margin=dict(l=0, r=0, t=10, b=0),
                            xaxis_rangeslider_visible=False,
                        )
                        st.plotly_chart(fig_rp, width='stretch')
                    else:
                        st.info("No candle data available around this decision.")

                    # Market state table
                    try:
                        _dj = json.loads(dec.get("decision_json", "{}") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        _dj = {}
                    _block_reasons = []
                    try:
                        _block_reasons = json.loads(dec.get("block_reasons_json", "[]") or "[]")
                    except (json.JSONDecodeError, TypeError):
                        pass
                    ms_cols = st.columns(4)
                    ms_cols[0].metric("Regime", _dj.get("regime", "—"))
                    ms_cols[1].metric("ATR %", f"{_dj.get('atr_pct', '—')}")
                    ms_cols[2].metric("Mode", dec.get("mode", "—"))
                    if _block_reasons:
                        ms_cols[3].metric("Blockers", len(_block_reasons))
                        with st.expander("Block reasons"):
                            for br in _block_reasons:
                                st.text(f"• {br}")
                    else:
                        ms_cols[3].metric("Blockers", 0)

                    st.divider()

                    # ── Zone D: Agent Board + Controller Story ────
                    zd_left, zd_right = st.columns(2)

                    with zd_left:
                        st.subheader("Agent Voting Board")
                        if votes_list:
                            for v in votes_list:
                                _veto_badge = " [VETO]" if v.get("veto") else ""
                                _conf = v.get("confidence", 0)
                                st.markdown(
                                    f"**{v.get('agent_id', '?')}**{_veto_badge} — "
                                    f"`{v.get('action', '?')}` ({_conf:.0%})"
                                )
                                v_cols = st.columns(3)
                                v_cols[0].caption(f"Edge: {v.get('expected_edge_bps', '—')} bps")
                                v_cols[1].caption(f"Size: {v.get('size_scale', 1.0):.2f}")
                                try:
                                    _vr = json.loads(v.get("block_reasons_json", "[]") or "[]")
                                except (json.JSONDecodeError, TypeError):
                                    _vr = []
                                if _vr:
                                    v_cols[2].caption(f"Reasons: {', '.join(_vr[:2])}")
                        else:
                            st.info("No agent votes recorded for this decision.")

                    with zd_right:
                        st.subheader("Decision Story")
                        _story = render_decision_story(
                            dec,
                            votes=pd.DataFrame(votes_list) if votes_list else None,
                            baseline=baseline,
                        )
                        st.markdown(_story)

                    st.divider()

                    # ── Zone E: Outcome / Attribution / Similar / Learning ──
                    st.subheader("Analysis")
                    e_tab1, e_tab2, e_tab3, e_tab4, e_tab5 = st.tabs([
                        "Outcome", "Attribution", "Similar Events", "Learning Note", "Diagnostics"
                    ])

                    with e_tab1:
                        if outcome:
                            oc1, oc2, oc3, oc4 = st.columns(4)
                            oc1.metric("60m Return", f"{outcome.get('forward_60m_bps', 0):+.1f} bps")
                            oc2.metric("MAE", f"{outcome.get('mae_bps', 0):.1f} bps")
                            oc3.metric("MFE", f"{outcome.get('mfe_bps', 0):.1f} bps")
                            oc4.metric("Label", outcome.get("outcome_label", "—"))

                            oc5, oc6, oc7 = st.columns(3)
                            oc5.metric("Trade Taken", "Yes" if outcome.get("was_trade_taken") else "No")
                            oc6.metric("Veto Correct", {1: "Yes", 0: "No"}.get(
                                outcome.get("was_veto_correct"), "N/A"))
                            oc7.metric("Skip Correct", {1: "Yes", 0: "No"}.get(
                                outcome.get("was_skip_correct"), "N/A"))

                            _5m = outcome.get("forward_5m_bps")
                            _15m = outcome.get("forward_15m_bps")
                            _30m = outcome.get("forward_30m_bps")
                            _60m = outcome.get("forward_60m_bps")
                            if any(x is not None for x in [_5m, _15m, _30m, _60m]):
                                markout_df = pd.DataFrame({
                                    "window": ["5m", "15m", "30m", "60m"],
                                    "bps": [_5m or 0, _15m or 0, _30m or 0, _60m or 0],
                                })
                                st.bar_chart(markout_df.set_index("window"))
                        else:
                            st.info("Outcome not yet available — forward window may not have matured.")

                    with e_tab2:
                        if attribution:
                            attr_names = ["direction", "veto", "posture", "entry", "cost", "disagreement"]
                            attr_vals = [attribution.get(f"{n}_attr", 0) for n in attr_names]
                            attr_df = pd.DataFrame({"component": attr_names, "score": attr_vals})

                            st.bar_chart(attr_df.set_index("component"))
                            st.caption("Score range: -1 (detracted) to +1 (contributed)")

                            with st.expander("Raw attribution"):
                                st.json({k: v for k, v in attribution.items() if k != "learning_note"})
                        else:
                            st.info("Attribution not yet computed — requires outcome data.")

                    with e_tab3:
                        if similar:
                            st.markdown(f"**Top {len(similar)} similar historical decisions:**")
                            for i, s in enumerate(similar):
                                s_ts = pd.to_datetime(s.get("ts_ms", 0), unit="ms").strftime("%m-%d %H:%M")
                                s_action = s.get("final_action", "?")
                                s_fwd = s.get("forward_60m_bps")
                                s_label = s.get("attr_label") or s.get("outcome_label") or ""
                                fwd_str = f"{s_fwd:+.1f}bps" if s_fwd is not None else "—"
                                st.text(f"{i+1}. #{s.get('id','?')} | {s_ts} | {s_action} | {fwd_str} | {s_label}")
                        else:
                            st.info("No similar events found.")

                    with e_tab4:
                        if attribution and attribution.get("learning_note"):
                            st.markdown(attribution["learning_note"])
                        elif attribution:
                            note = build_learning_note(dec, votes_list, outcome or {}, attribution)
                            st.markdown(note)
                        else:
                            st.info("Learning note requires outcome and attribution data.")

                    with e_tab5:
                        st.markdown("**Diagnostics**")
                        _diag_cols = st.columns(3)
                        _diag_cols[0].metric("Votes recorded", len(votes_list))
                        _diag_cols[1].metric("Outcome available", "Yes" if outcome else "No")
                        _diag_cols[2].metric("Attribution available", "Yes" if attribution else "No")

                        _expected_agents = 4
                        if len(votes_list) < _expected_agents:
                            st.warning(f"Expected {_expected_agents} agent votes, found {len(votes_list)}.")

                        with st.expander("Raw decision JSON"):
                            st.json(_dj)

                else:
                    st.warning("Could not load replay data for the selected decision.")

        _rp_conn.close()
    except Exception as exc:
        st.error(f"Error loading replay data: {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"Hogan Trading Bot · Paper Mode · Last refresh: {pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M:%S UTC')} "
    f"· DB: {DB_PATH}"
)
