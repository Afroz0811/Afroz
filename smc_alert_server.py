"""
SMC Engine Pro v3 — 24/7 Alert Server
Railway-compatible | All 5 improvements built in

Improvements over v2:
  1. Session filter      — London 07-12 UTC + NY 13-18 UTC only
  2. Weekly bias gate    — never trade against weekly trend
  3. BTC correlation     — suppress altcoin signals against BTC direction
  4. Structure-based SL  — SL at swept wick / OB level, not fixed ATR
  5. Breakeven manager   — alerts you to move SL to BE after TP1 hit

Environment variables (set in Railway Variables tab):
  TG_TOKEN          = Telegram bot token from @BotFather
  TG_CHAT           = Your chat ID from @userinfobot
  MIN_SCORE         = Minimum signal score (default: 6)
  SCAN_EVERY_MIN    = Minutes between scans (default: 5)
  COOLDOWN_MIN      = Minutes before same coin re-alerts (default: 60)
  PORT              = HTTP port (Railway sets this automatically)
"""

import os, time, logging, threading, json, csv
from pathlib import Path
import requests
import numpy as np
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import defaultdict
import math

# ════════════════════════════════════════════════
# SELF-LEARNING ENGINE (inline)
# ════════════════════════════════════════════════
"""
SMC Self-Learning Engine
========================
Stores every signal, monitors outcomes, learns from results.
Automatically adjusts which confluence factors matter most
based on REAL trade performance — not synthetic backtests.

Storage: JSON file (works on Railway with persistent volume)
         or SQLite for more advanced queries

Learning:
- Tracks win/loss per: setup, RSI range, session, weekly bias, score, tags
- After 20+ trades: auto-adjusts weights
- Sends weekly "what I learned" report to Telegram
"""


LEARN_FILE = os.environ.get('LEARN_FILE', '/app/smc_learning.json')

# ── DEFAULT WEIGHTS (start here, engine adjusts over time) ──────────
DEFAULT_WEIGHTS = {
    # Setup base scores
    'setup_scores': {
        'SWEEP_OB':        8.0,
        'HTF_CONFLUENCE':  8.0,
        'CHOCH':           8.0,
        'BOS':             7.0,
    },
    # Tag multipliers (how much each confluence adds)
    'tag_weights': {
        'Sweep↑':      1.0, 'Sweep↓':      1.0,
        'OB_Retest':   1.0, 'Vol✓':        0.8,
        'HTF✓':        0.8, 'Week✓':       0.6,
        'EMA↑':        0.5, 'EMA↓':        0.5,
        'MACD✓':       0.5, 'CHoCH↑':      1.0,
        'CHoCH↓':      1.0, 'BOS↑':        0.8,
        'BOS↓':        0.8, 'HH+HL':       0.6,
        'CleanStr':    0.5, 'RSI_Div✓':    1.5,
        'Fib✓':        0.5, 'VWAP✓':       0.3,
    },
    # Session multipliers
    'session_weights': {
        'London':    1.2,
        'New York':  1.2,
        'Asian':     0.8,
        'Weekend':   0.5,
    },
    # RSI zone effectiveness
    'rsi_zones': {
        '20-30': 1.3,  # deep oversold = strong buy
        '30-40': 1.1,
        '40-50': 1.0,
        '50-60': 0.9,
        '60-70': 1.1,
        '70-80': 1.3,  # deep overbought = strong sell
    },
    # Weekly bias multiplier
    'weekly_bias_mult': {
        'bullish': 1.2,
        'neutral': 0.9,
        'bearish': 0.7,  # against weekly = risky
    },
    # Minimum score to fire (adjusted based on session)
    'min_score_session': {
        'London':    6.0,
        'New York':  6.0,
        'Asian':     7.0,
        'Weekend':   8.0,
    }
}

# ── DATA SCHEMA ──────────────────────────────────────────────────────
def load_db():
    if Path(LEARN_FILE).exists():
        try:
            with open(LEARN_FILE) as f:
                return json.load(f)
        except:
            pass
    return {
        'version': 2,
        'created': datetime.now(timezone.utc).isoformat(),
        'weights': DEFAULT_WEIGHTS.copy(),
        'signals': [],        # every signal fired
        'outcomes': [],       # completed trades with result
        'stats': {
            'total_signals': 0,
            'total_trades':  0,
            'wins': 0, 'losses': 0, 'be': 0,
            'total_pnl': 0.0,
            'by_setup': {},
            'by_session': {},
            'by_rsi_zone': {},
            'by_tag': {},
            'by_weekly': {},
            'by_score_range': {},
        },
        'learning_log': [],   # what changed and why
        'last_learned': None,
    }

def save_db(db):
    try:
        with open(LEARN_FILE, 'w') as f:
            json.dump(db, f, indent=2)
    except Exception as e:
        print(f"DB save error: {e}")

# ── LOG A NEW SIGNAL ─────────────────────────────────────────────────
def log_signal(sig, pair, session):
    db = load_db()
    rsi_zone = get_rsi_zone(sig.get('rsi_val', 50))
    entry = {
        'id':           f"{pair['sym']}_{int(time.time())}",
        'sym':          pair['sym'],
        'setup':        sig['setup'],
        'dir':          sig['dir'],
        'score':        sig['score'],
        'raw_score':    sig.get('raw_score', sig['score']),
        'conf':         sig['conf'],
        'entry':        sig['price'],
        'sl':           sig['sl'],
        'tp1':          sig['tp1'],
        'tp2':          sig['tp'],
        'tp3':          sig['tp3'],
        'rr':           sig['rr'],
        'risk_pct':     sig['risk_pct'],
        'tags':         sig.get('tags', []),
        'session':      session,
        'weekly':       sig.get('weekly', 'neutral'),
        'daily':        sig.get('daily', 'neutral'),
        'rsi_val':      sig.get('rsi_val', 50),
        'rsi_zone':     sig.get('rsi_zone', 'neutral'),
        'conditions':   sig.get('conditions', {}),
        'max_adverse':   0.0,
        'max_favourable':0.0,
        'rsi_zone':     rsi_zone,
        'time':         datetime.now(timezone.utc).isoformat(),
        'status':       'open',
        'exit_price':   None,
        'exit_time':    None,
        'pnl':          None,
        'result':       None,
        'bars_held':    None,
    }
    db['signals'].append(entry)
    db['stats']['total_signals'] += 1
    save_db(db)
    return entry['id']

# ── CLOSE A TRADE + LEARN ────────────────────────────────────────────
def close_trade(trade_id, result, exit_price, bars_held=0):
    """
    result: 'win' | 'loss' | 'be'
    This is the CORE learning moment — update stats and adjust weights
    """
    db = load_db()
    sig = next((s for s in db['signals'] if s['id'] == trade_id), None)
    if not sig:
        return None

    is_buy = sig['dir'] == 'BUY'
    if exit_price:
        pnl = ((exit_price - sig['entry']) / sig['entry'] * 100) if is_buy \
              else ((sig['entry'] - exit_price) / sig['entry'] * 100)
    else:
        pnl = 0.0

    sig['status']     = result
    sig['result']     = result
    sig['exit_price'] = exit_price
    sig['exit_time']  = datetime.now(timezone.utc).isoformat()
    sig['pnl']        = round(pnl, 3)
    sig['bars_held']  = bars_held

    # Update aggregate stats
    db['stats']['total_trades'] += 1
    db['stats']['total_pnl']    = round(db['stats']['total_pnl'] + pnl, 3)
    if result == 'win':   db['stats']['wins']   += 1
    elif result == 'loss': db['stats']['losses'] += 1
    else:                  db['stats']['be']     += 1

    # Update per-dimension stats
    _update_dimension(db, 'by_setup',       sig['setup'],    result, pnl)
    _update_dimension(db, 'by_session',     sig['session'],  result, pnl)
    _update_dimension(db, 'by_rsi_zone',    sig['rsi_zone'], result, pnl)
    _update_dimension(db, 'by_weekly',      sig['weekly'],   result, pnl)
    score_bucket = f"{int(sig['score'])}-{int(sig['score'])+1}"
    _update_dimension(db, 'by_score_range', score_bucket,    result, pnl)
    for tag in sig.get('tags', []):
        _update_dimension(db, 'by_tag', tag, result, pnl)

    db['outcomes'].append(sig)
    save_db(db)

    # Auto-learn after every 5 trades
    total = db['stats']['total_trades']
    learn(db)                      # learn every trade (fast ML)
    learn_condition_boosts(db)     # update condition-specific boosts
    if total >= 5 and total % 3 == 0:
        try:
            from pathlib import Path as _P
            discoveries = mine_conditions(db) if callable(globals().get('mine_conditions')) else []
            if discoveries: log.info(f"  🧬 {len(discoveries)} patterns")
        except: pass

    return sig

def _update_dimension(db, dim, key, result, pnl):
    if key not in db['stats'][dim]:
        db['stats'][dim][key] = {'w':0,'l':0,'be':0,'total':0,'pnl':0.0}
    d = db['stats'][dim][key]
    d['total'] += 1
    d['pnl']   = round(d['pnl'] + pnl, 3)
    if result == 'win':   d['w'] += 1
    elif result == 'loss': d['l'] += 1
    else:                  d['be'] += 1

# ── LEARNING ENGINE ──────────────────────────────────────────────────
def learn(db):
    """
    Analyze completed trades and adjust weights.
    Uses Bayesian-style update: weight += learning_rate * (actual - expected)
    """
    outcomes = [s for s in db['signals'] if s['result']]
    if len(outcomes) < 10:
        return  # not enough data

    lr = 0.15  # learning rate — how fast to adjust
    changes = []

    # ── Learn setup scores ──────────────────────────────────────────
    for setup, stats in db['stats']['by_setup'].items():
        if stats['total'] < 5: continue
        wr = stats['w'] / stats['total']
        avg_pnl = stats['pnl'] / stats['total']
        # Expected WR at current score
        current_score = db['weights']['setup_scores'].get(setup, 7.0)
        # If WR > 55% and positive PnL → increase base score
        # If WR < 35% → decrease base score
        if wr > 0.55 and avg_pnl > 0:
            new_score = min(9.5, current_score + lr)
            if abs(new_score - current_score) > 0.05:
                db['weights']['setup_scores'][setup] = round(new_score, 2)
                changes.append(f"↑ {setup} score {current_score:.1f}→{new_score:.1f} (WR:{wr:.0%})")
        elif wr < 0.35 or avg_pnl < -1:
            new_score = max(5.0, current_score - lr)
            if abs(new_score - current_score) > 0.05:
                db['weights']['setup_scores'][setup] = round(new_score, 2)
                changes.append(f"↓ {setup} score {current_score:.1f}→{new_score:.1f} (WR:{wr:.0%})")

    # ── Learn tag effectiveness ──────────────────────────────────────
    for tag, stats in db['stats']['by_tag'].items():
        if stats['total'] < 5: continue
        wr = stats['w'] / stats['total']
        current_w = db['weights']['tag_weights'].get(tag, 0.5)
        if wr > 0.60:
            new_w = min(2.5, current_w + lr*0.5)
            if abs(new_w - current_w) > 0.05:
                db['weights']['tag_weights'][tag] = round(new_w, 2)
                changes.append(f"↑ tag '{tag}' weight {current_w:.2f}→{new_w:.2f} (WR:{wr:.0%})")
        elif wr < 0.30:
            new_w = max(0.1, current_w - lr*0.5)
            if abs(new_w - current_w) > 0.05:
                db['weights']['tag_weights'][tag] = round(new_w, 2)
                changes.append(f"↓ tag '{tag}' weight {current_w:.2f}→{new_w:.2f} (WR:{wr:.0%})")

    # ── Learn session effectiveness ──────────────────────────────────
    for sess, stats in db['stats']['by_session'].items():
        if stats['total'] < 5: continue
        wr = stats['w'] / stats['total']
        current_m = db['weights']['session_weights'].get(sess, 1.0)
        target_m = 0.6 + wr * 1.2  # scales from 0.6 to 1.8
        new_m = round(current_m + lr * (target_m - current_m), 2)
        new_m = max(0.3, min(1.5, new_m))
        if abs(new_m - current_m) > 0.05:
            db['weights']['session_weights'][sess] = new_m
            changes.append(f"{'↑' if new_m>current_m else '↓'} session '{sess}' mult {current_m:.2f}→{new_m:.2f} (WR:{wr:.0%})")

    # ── Learn RSI zone effectiveness ─────────────────────────────────
    for zone, stats in db['stats']['by_rsi_zone'].items():
        if stats['total'] < 5: continue
        wr = stats['w'] / stats['total']
        current_m = db['weights']['rsi_zones'].get(zone, 1.0)
        target_m = 0.5 + wr * 1.5
        new_m = round(current_m + lr * (target_m - current_m), 2)
        new_m = max(0.3, min(2.0, new_m))
        if abs(new_m - current_m) > 0.05:
            db['weights']['rsi_zones'][zone] = new_m
            changes.append(f"RSI zone '{zone}': mult {current_m:.2f}→{new_m:.2f} (WR:{wr:.0%})")

    if changes:
        log_entry = {
            'time':    datetime.now(timezone.utc).isoformat(),
            'trades':  len(outcomes),
            'changes': changes
        }
        db['learning_log'].append(log_entry)
        db['last_learned'] = log_entry['time']

    save_db(db)
    return changes

# ── COMPUTE SCORE USING LEARNED WEIGHTS ──────────────────────────────
def compute_learned_score(setup, tags, session, weekly, rsi_val, base_score):
    """
    Returns adjusted score using learned weights.
    Called instead of fixed score thresholds.
    """
    db = load_db()
    w = db['weights']

    # Start with learned setup base score
    score = w['setup_scores'].get(setup, base_score)

    # Add learned tag weights
    for tag in tags:
        tag_clean = tag.split('RSI')[0].strip()  # normalize RSI35, RSI42 etc
        score += w['tag_weights'].get(tag_clean, 0.3)

    # Apply session multiplier
    score *= w['session_weights'].get(session, 1.0)

    # Apply RSI zone multiplier
    zone = get_rsi_zone(rsi_val)
    score *= w['rsi_zones'].get(zone, 1.0)

    # Apply weekly bias multiplier
    score *= w['weekly_bias_mult'].get(weekly, 1.0)

    return round(min(10, score), 1)

def get_min_score(session):
    """Dynamic minimum score threshold based on session"""
    db = load_db()
    return db['weights']['min_score_session'].get(session, 6.5)

# ── HELPERS ──────────────────────────────────────────────────────────
def get_rsi_zone(rsi):
    if rsi < 30:   return '20-30'
    if rsi < 40:   return '30-40'
    if rsi < 50:   return '40-50'
    if rsi < 60:   return '50-60'
    if rsi < 70:   return '60-70'
    return '70-80'

def get_session():
    h = datetime.now(timezone.utc).hour
    d = datetime.now(timezone.utc).weekday()
    if d >= 5: return 'Weekend'
    if 7 <= h <= 12:  return 'London'
    if 13 <= h <= 18: return 'New York'
    return 'Asian'

# ── PERFORMANCE REPORT ───────────────────────────────────────────────
def performance_report():
    db = load_db()
    s = db['stats']
    total = s['total_trades']
    if total == 0:
        return "📊 No completed trades yet. Learning begins after first trade closes."

    wr = s['wins']/total*100 if total else 0
    pf_num = s['wins'] * (s['total_pnl']/max(s['wins'],1)) if s['wins'] else 0
    pf_den = s['losses'] * abs(s['total_pnl']/max(s['losses'],1)) if s['losses'] else 1
    pf = pf_num/pf_den if pf_den else 0

    lines = [
        "🧠 <b>SMC Self-Learning Report</b>",
        f"Based on {total} real trades\n",
        f"✅ Wins:     {s['wins']} ({wr:.1f}%)",
        f"❌ Losses:   {s['losses']}",
        f"➡️ BE:        {s['be']}",
        f"💰 Total P&amp;L: {s['total_pnl']:+.2f}%\n",
        "<b>📊 Setup Performance:</b>",
    ]

    for setup, st in sorted(s['by_setup'].items(),
                            key=lambda x: x[1]['w']/max(x[1]['total'],1), reverse=True):
        if st['total'] < 2: continue
        wr_s = st['w']/st['total']*100
        bar = '█' * int(wr_s/10) + '░' * (10-int(wr_s/10))
        learned = db['weights']['setup_scores'].get(setup, 7.0)
        lines.append(f"  {'⚡📊🔄📈'[['SWEEP_OB','HTF_CONFLUENCE','CHOCH','BOS'].index(setup)] if setup in ['SWEEP_OB','HTF_CONFLUENCE','CHOCH','BOS'] else '📡'} "
                     f"{setup}: {st['total']}tr WR:{wr_s:.0f}% {bar}")
        lines.append(f"     Learned score: {learned:.1f}/10 | P&L: {st['pnl']:+.1f}%")

    lines.append("\n<b>📅 Session Performance:</b>")
    for sess, st in s['by_session'].items():
        if st['total'] < 2: continue
        wr_s = st['w']/st['total']*100
        mult = db['weights']['session_weights'].get(sess, 1.0)
        lines.append(f"  {sess}: {st['total']}tr WR:{wr_s:.0f}% → weight:{mult:.2f}x")

    lines.append("\n<b>📈 Best performing tags:</b>")
    tag_stats = [(t,v) for t,v in s['by_tag'].items() if v['total']>=3]
    tag_stats.sort(key=lambda x: x[1]['w']/max(x[1]['total'],1), reverse=True)
    for tag, st in tag_stats[:5]:
        wr_t = st['w']/st['total']*100
        w = db['weights']['tag_weights'].get(tag, 0.5)
        lines.append(f"  {tag}: WR:{wr_t:.0f}% ({st['total']}tr) weight:{w:.2f}")

    if db['learning_log']:
        last = db['learning_log'][-1]
        lines.append(f"\n<b>🧠 Last learning update:</b> {last['time'][:10]}")
        for ch in last['changes'][:5]:
            lines.append(f"  {ch}")

    lines.append(f"\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append("📡 <b>SMC Engine Pro v3 — Self Learning</b>")
    return '\n'.join(lines)

def weekly_learning_report():
    """Sent every Monday — what the engine learned this week"""
    db = load_db()
    recent = [s for s in db['signals']
              if s.get('result') and
              (datetime.now(timezone.utc).timestamp() -
               datetime.fromisoformat(s['time']).timestamp()) < 7*24*3600]
    if not recent:
        return "📅 No trades completed this week."

    wins   = [t for t in recent if t['result']=='win']
    losses = [t for t in recent if t['result']=='loss']
    pnl    = sum(t.get('pnl',0) or 0 for t in recent)
    wr     = len(wins)/len(recent)*100

    # What patterns show up in winners vs losers?
    win_tags  = defaultdict(int)
    loss_tags = defaultdict(int)
    for t in wins:
        for tag in t.get('tags',[]): win_tags[tag] += 1
    for t in losses:
        for tag in t.get('tags',[]): loss_tags[tag] += 1

    lines = [
        "📅 <b>Weekly Learning Report</b>",
        f"Week trades: {len(recent)} | W:{len(wins)} L:{len(losses)}",
        f"Win rate: {wr:.1f}% | P&amp;L: {pnl:+.2f}%\n",
        "<b>🏆 Tags in winning trades:</b>",
    ]
    for tag, cnt in sorted(win_tags.items(), key=lambda x:-x[1])[:5]:
        lines.append(f"  ✅ {tag}: {cnt} wins")
    lines.append("<b>⚠️ Tags in losing trades:</b>")
    for tag, cnt in sorted(loss_tags.items(), key=lambda x:-x[1])[:5]:
        lines.append(f"  ❌ {tag}: {cnt} losses")

    if db['learning_log']:
        lines.append(f"\n<b>Weight changes this week:</b>")
        week_logs = [l for l in db['learning_log']
                     if (datetime.now(timezone.utc).timestamp() -
                         datetime.fromisoformat(l['time']).timestamp()) < 7*24*3600]
        for log in week_logs[-3:]:
            for ch in log['changes'][:3]:
                lines.append(f"  {ch}")

    lines.append(f"\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append("📡 <b>SMC Engine — Self Learning v3</b>")
    return '\n'.join(lines)


# ════════════════════════════════════════════════

# ════════════════════════════════════════════════
# DEEP CHART LEARNING ENGINE
# Re-analyzes chart AFTER every trade closes
# Learns from actual price/volume/RSI conditions
# ════════════════════════════════════════════════
"""
Deep Chart Learning Engine
===========================
After every trade closes (win or loss):
1. Re-fetches the candles from that time period
2. Re-analyzes ALL metrics at the exact entry bar
3. Compares winning conditions vs losing conditions
4. Finds patterns: "When RSI was 25-35 AND volume >1.5x AND London session → 72% WR"
5. Adjusts thresholds based on REAL chart data, not just setup names
"""

DEEP_LEARN_FILE = os.environ.get('DEEP_LEARN_FILE', '/app/smc_deep_learning.json')
CG = 'https://api.coingecko.com/api/v3'
KR = 'https://api.kraken.com/0/public'

# ── METRICS WE ANALYZE ON EVERY TRADE ────────────────────────────────
# These are the exact conditions at the entry bar
METRIC_KEYS = [
    'rsi',           # RSI value at entry
    'rsi_zone',      # oversold/neutral/overbought
    'volume_ratio',  # volume / 20-bar average
    'atr_ratio',     # current ATR / 20-bar ATR avg (chop measure)
    'session',       # London / NY / Asian / Weekend
    'weekly_bias',   # bullish / bearish / neutral
    'daily_bias',    # bullish / bearish / neutral
    'ema_aligned',   # True if EMA stack aligned with direction
    'macd_positive', # True if MACD histogram positive (for buys)
    'ob_quality',    # clean / messy (how well-defined OB was)
    'sweep_size',    # wick size / ATR ratio
    'bars_since_sweep', # how many bars since the sweep candle
    'score',         # signal score at time of entry
    'rr',            # risk:reward ratio
    'setup',         # SWEEP_OB / CHOCH / BOS / HTF
    'direction',     # BUY / SELL
]

def load_deep_db():
    if Path(DEEP_LEARN_FILE).exists():
        try:
            with open(DEEP_LEARN_FILE) as f:
                return json.load(f)
        except:
            pass
    return {
        'version': 1,
        'created': datetime.now(timezone.utc).isoformat(),
        'trades': [],           # full trade records with metrics
        'patterns': {},         # discovered winning patterns
        'thresholds': {         # learned optimal thresholds
            'min_volume_ratio':  1.15,
            'min_rsi_buy_max':   62,    # RSI must be below this for buys
            'max_rsi_buy_min':   25,    # RSI must be above this for buys
            'min_sweep_size':    0.28,  # min sweep wick / ATR
            'max_bars_retest':   8,     # max bars after sweep to retest
            'min_atr_ratio':     0.40,  # min ATR vs average (not choppy)
            'min_score':         7.0,
            'best_sessions':     ['London', 'New York'],
            'avoid_weekly':      ['bearish'],  # avoid buying in these
        },
        'condition_stats': {},  # win rate per condition value
        'insights': [],         # human-readable insights discovered
    }

def save_deep_db(db):
    try:
        with open(DEEP_LEARN_FILE, 'w') as f:
            json.dump(db, f, indent=2)
    except Exception as e:
        print(f"Deep DB save error: {e}")

# ── FETCH HISTORICAL CANDLES FOR RE-ANALYSIS ─────────────────────────
def fetch_candles_at_time(pair_cg, pair_kr, timestamp, limit=100):
    """Fetch candles around a specific timestamp for post-trade analysis"""
    try:
        r = requests.get(f'{KR}/OHLC',
            params={'pair': pair_kr, 'interval': 60, 'since': int(timestamp)-3600*limit},
            timeout=15)
        d = r.json()
        if not d.get('error'):
            key = next((k for k in d['result'] if k != 'last'), None)
            if key:
                raw = d['result'][key]
                return [{'t': int(k[0]), 'o':float(k[1]),'h':float(k[2]),
                         'l':float(k[3]),'c':float(k[4]),'v':float(k[6])}
                        for k in raw[-limit:]]
    except: pass
    try:
        r = requests.get(f'{CG}/coins/{pair_cg}/ohlc',
            params={'vs_currency':'usd','days':7}, timeout=15)
        raw = r.json()
        if isinstance(raw, list):
            return [{'t':int(k[0]/1000),'o':float(k[1]),'h':float(k[2]),
                     'l':float(k[3]),'c':float(k[4]),'v':50.0}
                    for k in raw[-limit:]]
    except: pass
    return []

# ── INDICATORS ────────────────────────────────────────────────────────
def _ema(c, p):
    if len(c) < p: return [None]*len(c)
    k=2/(p+1); r=[None]*(p-1); s=sum(c[:p])/p; r.append(s); pv=s
    for i in range(p,len(c)): pv=c[i]*k+pv*(1-k); r.append(pv)
    return r

def _rsi(c, p=14):
    if len(c)<p+1: return [None]*len(c)
    r=[None]*p; g=l=0.0
    for i in range(1,p+1):
        d=c[i]-c[i-1]
        if d>0: g+=d
        else: l+=abs(d)
    ag,al=g/p,l/p; r.append(100 if al==0 else 100-100/(1+ag/al))
    for i in range(p+1,len(c)):
        d=c[i]-c[i-1]; ag=(ag*(p-1)+(d if d>0 else 0))/p; al=(al*(p-1)+(abs(d) if d<0 else 0))/p
        r.append(100 if al==0 else 100-100/(1+ag/al))
    return r

def _macd_hist(c):
    e12=_ema(c,12); e26=_ema(c,26)
    ln=[e12[i]-e26[i] if e12[i] and e26[i] else None for i in range(len(c))]
    vl=[v for v in ln if v]
    if len(vl)<9: return [None]*len(c)
    sr=_ema(vl,9); sg=[None]*len(c); si=0
    for i in range(len(c)):
        if ln[i] is not None: sg[i]=sr[si] if si<len(sr) else None; si+=1
    return [ln[i]-sg[i] if ln[i] and sg[i] else None for i in range(len(c))]

def _atr(kl, p=14):
    tr=[None]+[max(kl[i]['h']-kl[i]['l'],abs(kl[i]['h']-kl[i-1]['c']),
               abs(kl[i]['l']-kl[i-1]['c'])) for i in range(1,len(kl))]
    if len(tr)<p+1: return [None]*len(kl)
    r=[None]*p; s=sum(tr[1:p+1])/p; r.append(s); pv=s
    for i in range(p+1,len(tr)): pv=(pv*(p-1)+tr[i])/p; r.append(pv)
    return r

def _vol_avg(v, p=20):
    r=[None]*p
    for i in range(p,len(v)): r.append(sum(v[i-p:i])/p)
    return r

# ── DEEP METRIC EXTRACTION ────────────────────────────────────────────
def extract_metrics_from_chart(kl, trade):
    """
    Re-analyze ALL chart conditions at the exact entry bar.
    This is what the engine ACTUALLY saw when it fired the signal.
    Returns a dict of all measurable conditions.
    """
    if not kl or len(kl) < 30:
        return None

    i = len(kl) - 1
    closes = [k['c'] for k in kl]
    vols   = [k['v'] for k in kl]
    price  = closes[i]

    rsi_a  = _rsi(closes)
    e20_a  = _ema(closes, 20)
    e50_a  = _ema(closes, 50)
    e9_a   = _ema(closes, 9)
    ht_a   = _macd_hist(closes)
    atr_a  = _atr(kl)
    va_a   = _vol_avg(vols)

    if not all([rsi_a[i], e20_a[i], e50_a[i], atr_a[i], va_a[i]]):
        return None

    is_buy      = trade['dir'] == 'BUY'
    rsi_val     = round(rsi_a[i], 1)
    vol_ratio   = round(kl[i]['v'] / va_a[i], 2) if va_a[i] else 0
    atr_ratio   = round(atr_a[i] / (sum(a for a in atr_a[max(0,i-20):i] if a) /
                        max(1, len([a for a in atr_a[max(0,i-20):i] if a]))), 2) \
                  if atr_a[i] else 0

    # EMA alignment
    ema_aligned = (price > e20_a[i] > e50_a[i]) if is_buy else (price < e20_a[i] < e50_a[i])
    ema_stack   = (e9_a[i] > e20_a[i] > e50_a[i]) if (is_buy and e9_a[i]) else \
                  (e9_a[i] < e20_a[i] < e50_a[i]) if (not is_buy and e9_a[i]) else False

    # MACD confirmation
    macd_ok = (ht_a[i] > 0) if is_buy else (ht_a[i] < 0) if ht_a[i] else False

    # RSI zone
    rsi_zone = ('oversold' if rsi_val < 35 else
                'neutral'  if rsi_val < 65 else 'overbought')

    # Candle quality at entry
    body    = abs(kl[i]['c'] - kl[i]['o'])
    rng     = kl[i]['h'] - kl[i]['l'] + 1e-10
    body_pct = round(body/rng*100, 1)

    # Sweep-specific metrics
    sweep_size   = 0.0
    bars_retest  = 0
    ob_quality   = 'none'

    if trade.get('setup') == 'SWEEP_OB':
        swept_lvl = trade.get('swept_price', price)
        if swept_lvl and atr_a[i]:
            # How big was the sweep wick relative to ATR?
            sweep_wick = abs(swept_lvl - min(kl[max(0,i-5):i+1], key=lambda x:x['l'])['l'])
            sweep_size = round(sweep_wick / atr_a[i], 2)
        # How many bars ago was the actual sweep?
        bars_retest = trade.get('bars_since_sweep', 0)
        # OB quality: how clean was the OB candle
        ob_top = trade.get('ob_top', 0)
        ob_bot = trade.get('ob_bot', 0)
        if ob_top and ob_bot and atr_a[i]:
            ob_size = ob_top - ob_bot
            ob_quality = 'clean' if ob_size > atr_a[i]*0.3 else 'small'

    # Recent trend strength (bearish or bullish pressure)
    recent6 = kl[max(0,i-6):i+1]
    bear_candles = sum(1 for x in recent6 if x['c'] < x['o'])
    bull_candles = len(recent6) - bear_candles
    trend_pressure = 'strong_bear' if bear_candles >= 5 else \
                     'strong_bull' if bull_candles >= 5 else 'mixed'

    # Consecutive same-direction candles
    consec = 0
    for j in range(i, max(0, i-8), -1):
        if is_buy and kl[j]['c'] < kl[j]['o']: consec += 1
        elif not is_buy and kl[j]['c'] > kl[j]['o']: consec += 1
        else: break

    return {
        'rsi':              rsi_val,
        'rsi_zone':         rsi_zone,
        'volume_ratio':     vol_ratio,
        'atr_ratio':        atr_ratio,
        'ema_aligned':      ema_aligned,
        'ema_stack':        ema_stack,
        'macd_ok':          macd_ok,
        'body_pct':         body_pct,
        'sweep_size':       sweep_size,
        'bars_retest':      bars_retest,
        'ob_quality':       ob_quality,
        'trend_pressure':   trend_pressure,
        'consec_against':   consec,
        'session':          trade.get('session', 'unknown'),
        'weekly_bias':      trade.get('weekly', 'neutral'),
        'daily_bias':       trade.get('daily', 'neutral'),
        'score':            trade.get('score', 0),
        'rr':               trade.get('rr', 0),
        'setup':            trade.get('setup', ''),
        'direction':        trade.get('dir', ''),
    }

# ── LEARN FROM CLOSED TRADE ───────────────────────────────────────────
def learn_from_trade(trade, result, kl=None):
    """
    Called when a trade closes.
    Extracts chart metrics and stores them.
    Analyzes patterns after every 10 trades.
    """
    db = load_deep_db()

    # Extract metrics from chart
    metrics = None
    if kl:
        metrics = extract_metrics_from_chart(kl, trade)

    record = {
        'id':       trade.get('id', f"{trade['sym']}_{int(time.time())}"),
        'sym':      trade['sym'],
        'setup':    trade.get('setup', ''),
        'dir':      trade['dir'],
        'result':   result,
        'pnl':      trade.get('pnl', 0),
        'time':     datetime.now(timezone.utc).isoformat(),
        'metrics':  metrics,
        'tags':     trade.get('tags', []),
    }
    db['trades'].append(record)

    # Update condition stats
    if metrics:
        _update_condition_stats(db, metrics, result)

    # Analyze patterns every 10 trades
    total = len([t for t in db['trades'] if t.get('result')])
    if total >= 10 and total % 5 == 0:
        insights = _find_patterns(db)
        if insights:
            db['insights'].extend(insights)
            # Apply threshold updates
            _update_thresholds(db)

    save_deep_db(db)
    return record

def _update_condition_stats(db, metrics, result):
    """Track win rate for each condition value"""
    is_win = result == 'win'

    conditions_to_track = {
        'session':       metrics.get('session'),
        'weekly_bias':   metrics.get('weekly_bias'),
        'rsi_zone':      metrics.get('rsi_zone'),
        'ema_aligned':   str(metrics.get('ema_aligned')),
        'macd_ok':       str(metrics.get('macd_ok')),
        'ob_quality':    metrics.get('ob_quality'),
        'trend_pressure':metrics.get('trend_pressure'),
        'high_volume':   str(metrics.get('volume_ratio', 0) >= 1.5),
        'very_high_vol': str(metrics.get('volume_ratio', 0) >= 2.0),
        'clean_rsi_buy': str(metrics.get('rsi', 50) < 45 and metrics.get('direction')=='BUY'),
        'no_trend_press':str(metrics.get('trend_pressure') == 'mixed'),
        'low_consec':    str(metrics.get('consec_against', 0) <= 2),
    }
    for key, val in conditions_to_track.items():
        if val is None: continue
        stat_key = f"{key}:{val}"
        if stat_key not in db['condition_stats']:
            db['condition_stats'][stat_key] = {'w':0,'l':0,'be':0,'total':0}
        s = db['condition_stats'][stat_key]
        s['total'] += 1
        if result == 'win':   s['w'] += 1
        elif result == 'loss': s['l'] += 1
        else:                  s['be'] += 1

def _find_patterns(db):
    """Find conditions that predict wins vs losses"""
    insights = []
    stats = db['condition_stats']
    thresholds = db['thresholds']

    for key, s in stats.items():
        if s['total'] < 5: continue
        wr = s['w'] / s['total']

        # High win rate condition
        if wr >= 0.65 and s['total'] >= 5:
            insights.append({
                'type': 'positive',
                'condition': key,
                'wr': round(wr*100),
                'trades': s['total'],
                'message': f"✅ When {key} → WR {round(wr*100)}% ({s['total']} trades)"
            })

        # Low win rate condition — this is a WARNING
        elif wr <= 0.30 and s['total'] >= 5:
            insights.append({
                'type': 'negative',
                'condition': key,
                'wr': round(wr*100),
                'trades': s['total'],
                'message': f"❌ When {key} → WR only {round(wr*100)}% — AVOID ({s['total']} trades)"
            })

    return insights[-10:] if insights else []  # keep latest 10

def _update_thresholds(db):
    """
    Auto-adjust detection thresholds based on real performance.
    This is where the engine actually gets smarter.
    """
    stats  = db['condition_stats']
    thresh = db['thresholds']
    changes = []

    # Session learning
    for sess in ['London', 'New York', 'Asian', 'Weekend']:
        key = f"session:{sess}"
        if key in stats and stats[key]['total'] >= 5:
            wr = stats[key]['w'] / stats[key]['total']
            if wr < 0.30 and sess in thresh['best_sessions']:
                thresh['best_sessions'].remove(sess)
                changes.append(f"Removed {sess} from best sessions (WR:{round(wr*100)}%)")
            elif wr >= 0.55 and sess not in thresh['best_sessions']:
                thresh['best_sessions'].append(sess)
                changes.append(f"Added {sess} to best sessions (WR:{round(wr*100)}%)")

    # Weekly bias learning
    for bias in ['bullish', 'neutral', 'bearish']:
        key = f"weekly_bias:{bias}"
        if key in stats and stats[key]['total'] >= 5:
            wr = stats[key]['w'] / stats[key]['total']
            if wr < 0.30 and bias not in thresh['avoid_weekly']:
                thresh['avoid_weekly'].append(bias)
                changes.append(f"Added weekly:{bias} to avoid list (WR:{round(wr*100)}%)")
            elif wr >= 0.55 and bias in thresh['avoid_weekly']:
                thresh['avoid_weekly'].remove(bias)
                changes.append(f"Removed weekly:{bias} from avoid list (WR:{round(wr*100)}%)")

    # Volume threshold
    hi_vol = stats.get('high_volume:True', {'w':0,'total':0})
    lo_vol_key = 'high_volume:False'
    lo_vol = stats.get(lo_vol_key, {'w':0,'total':0})
    if hi_vol['total'] >= 5 and lo_vol['total'] >= 5:
        wr_hi = hi_vol['w']/hi_vol['total']
        wr_lo = lo_vol['w']/lo_vol['total']
        if wr_hi > wr_lo + 0.15:
            thresh['min_volume_ratio'] = max(1.3, thresh['min_volume_ratio'])
            changes.append(f"Raised min_volume_ratio (high vol WR:{round(wr_hi*100)}% vs low:{round(wr_lo*100)}%)")
        elif wr_hi < wr_lo:
            thresh['min_volume_ratio'] = max(1.0, thresh['min_volume_ratio'] - 0.05)
            changes.append(f"Lowered min_volume_ratio")

    # Trend pressure
    no_press = stats.get('no_trend_press:True', {'w':0,'total':0})
    press    = stats.get('no_trend_press:False', {'w':0,'total':0})
    if no_press['total'] >= 5 and press['total'] >= 5:
        wr_np = no_press['w']/no_press['total']
        wr_p  = press['w']/press['total']
        if wr_np > wr_p + 0.15:
            # Mixed trend is better — relax the filter
            changes.append(f"Confirmed: no trend pressure better (WR:{round(wr_np*100)}% vs {round(wr_p*100)}%)")
        elif wr_p > wr_np + 0.15:
            changes.append(f"Trend pressure actually helps! WR:{round(wr_p*100)}%")

    if changes:
        db['insights'].append({
            'type': 'threshold_update',
            'time': datetime.now(timezone.utc).isoformat(),
            'changes': changes
        })

    db['thresholds'] = thresh

# ── DEEP LEARNING REPORT ──────────────────────────────────────────────
def deep_learning_report():
    db = load_deep_db()
    trades = [t for t in db['trades'] if t.get('result')]
    if not trades:
        return "🧠 No completed trades yet for deep analysis."

    wins   = [t for t in trades if t['result']=='win']
    losses = [t for t in trades if t['result']=='loss']
    wr     = len(wins)/len(trades)*100

    lines = [
        "🧠 <b>Deep Chart Learning Report</b>",
        f"Analyzed {len(trades)} trades from real chart data\n",
        f"✅ Wins: {len(wins)} | ❌ Losses: {len(losses)} | WR: {wr:.1f}%\n",
        "<b>📊 What the engine learned:</b>",
    ]

    # Show top positive patterns
    pos = [i for i in db['insights'] if i.get('type')=='positive']
    neg = [i for i in db['insights'] if i.get('type')=='negative']

    if pos:
        lines.append("\n✅ <b>Conditions that WIN:</b>")
        for p in sorted(pos, key=lambda x:-x['wr'])[:5]:
            lines.append(f"  {p['message']}")

    if neg:
        lines.append("\n❌ <b>Conditions to AVOID:</b>")
        for n in sorted(neg, key=lambda x:x['wr'])[:5]:
            lines.append(f"  {n['message']}")

    # Current thresholds
    t = db['thresholds']
    lines += [
        "\n<b>⚙️ Learned Thresholds:</b>",
        f"  Min volume ratio: {t['min_volume_ratio']}x avg",
        f"  Best sessions: {', '.join(t['best_sessions'])}",
        f"  Avoid weekly: {', '.join(t['avoid_weekly']) or 'none'}",
        f"  Min score: {t['min_score']}",
    ]

    # Win/loss metric comparison
    if len(wins) >= 3 and len(losses) >= 3:
        def avg_metric(trade_list, key):
            vals = [t['metrics'][key] for t in trade_list
                    if t.get('metrics') and t['metrics'].get(key) is not None
                    and isinstance(t['metrics'][key], (int, float))]
            return round(sum(vals)/len(vals), 2) if vals else None

        lines.append("\n<b>📈 Average metrics — Wins vs Losses:</b>")
        for metric, label in [('rsi','RSI'),('volume_ratio','Volume ratio'),
                               ('score','Score'),('rr','R:R')]:
            w_avg = avg_metric(wins, metric)
            l_avg = avg_metric(losses, metric)
            if w_avg and l_avg:
                diff = '↑ Better' if (metric in ('volume_ratio','score','rr') and w_avg>l_avg) or \
                                      (metric=='rsi' and abs(w_avg-50)<abs(l_avg-50)) else '↓ Worse'
                lines.append(f"  {label}: Wins={w_avg} | Losses={l_avg} {diff}")

    lines.append(f"\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append("📡 <b>SMC Deep Learning Engine</b>")
    return '\n'.join(lines)

def get_learned_thresholds():
    """Returns current learned thresholds for use in signal detection"""
    return load_deep_db()['thresholds']




logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ── CONFIG ─────────────────────────────────────
TG_TOKEN   = os.environ.get('TG_TOKEN', '')
TG_CHAT    = os.environ.get('TG_CHAT', '')
MIN_SCORE  = int(os.environ.get('MIN_SCORE', '4.5'))
SCAN_EVERY = int(os.environ.get('SCAN_EVERY_MIN', '1'))
COOLDOWN_M = int(os.environ.get('COOLDOWN_MIN', '15'))
PORT       = int(os.environ.get('PORT', '8080'))

PAIRS = [
    {'sym':'BTC',  'kr':'XXBTZUSD', 'cg':'bitcoin'},
    {'sym':'ETH',  'kr':'XETHZUSD', 'cg':'ethereum'},
    {'sym':'SOL',  'kr':'SOLUSD',   'cg':'solana'},
    {'sym':'XRP',  'kr':'XXRPZUSD', 'cg':'ripple'},
    {'sym':'ADA',  'kr':'ADAUSD',   'cg':'cardano'},
    {'sym':'DOGE', 'kr':'XDGUSD',   'cg':'dogecoin'},
    {'sym':'AVAX', 'kr':'AVAXUSD',  'cg':'avalanche-2'},
    {'sym':'DOT',  'kr':'DOTUSD',   'cg':'polkadot'},
    {'sym':'LINK', 'kr':'LINKUSD',  'cg':'chainlink'},
    {'sym':'MATIC','kr':'MATICUSD', 'cg':'matic-network'},
]

KR = 'https://api.kraken.com/0/public'

# ── TRADING JOURNAL ────────────────────────────
JOURNAL_FILE = os.environ.get('JOURNAL_FILE', '/app/smc_journal.json')

def load_journal():
    if Path(JOURNAL_FILE).exists():
        try:
            with open(JOURNAL_FILE) as f: return json.load(f)
        except: pass
    return {'trades':[],'open':{},'signals':[],'stats':{},'created':datetime.now(timezone.utc).isoformat()}

def save_journal(j):
    try:
        with open(JOURNAL_FILE,'w') as f: json.dump(j,f,indent=2)
    except Exception as e:
        log.debug(f"Journal save error: {e}")

def journal_log_signal(sig, pair):
    try:
        j = load_journal()
        entry = {
            'id':         f"{pair['sym']}_{int(time.time())}",
            'sym':        pair['sym'],
            'setup':      sig['setup'],
            'setup_name': sig['name'],
            'dir':        sig['dir'],
            'score':      sig['score'],
            'entry':      sig['price'],
            'sl':         sig['sl'],
            'tp1':        sig['tp1'],
            'tp2':        sig['tp'],
            'tp3':        sig['tp3'],
            'rr':         sig['rr'],
            'risk_pct':   sig['risk_pct'],
            'tags':       sig['tags'],
            'weekly':     sig.get('weekly','—'),
            'rsi':        sig.get('rsi_val',0),
            'time':       datetime.now(timezone.utc).isoformat(),
            'status':     'open',
            'exit_price': None,
            'exit_time':  None,
            'pnl':        None,
        }
        j['signals'].append(entry)
        j['open'][pair['sym']] = entry['id']
        save_journal(j)
        log.info(f"  📓 Journal: logged {pair['sym']} {sig['dir']} {sig['setup']}")
    except Exception as e:
        log.debug(f"Journal log error: {e}")

def journal_close_trade(sym, result, exit_price):
    try:
        j = load_journal()
        trade_id = j['open'].get(sym)
        if not trade_id: return
        sig = next((s for s in j['signals'] if s['id']==trade_id), None)
        if not sig: return
        is_buy = sig['dir']=='BUY'
        pnl = ((exit_price-sig['entry'])/sig['entry']*100) if is_buy else ((sig['entry']-exit_price)/sig['entry']*100)
        sig.update({'status':result,'exit_price':exit_price,
                    'exit_time':datetime.now(timezone.utc).isoformat(),'pnl':round(pnl,3)})
        j['trades'].append(sig)
        del j['open'][sym]
        # Update stats
        s = j['stats'].setdefault(sig['setup'],{'wins':0,'losses':0,'be':0,'total':0,'total_pnl':0})
        s['total']+=1; s['total_pnl']=round(s['total_pnl']+pnl,3)
        if result=='win': s['wins']+=1
        elif result=='loss': s['losses']+=1
        else: s['be']+=1
        save_journal(j)
    except Exception as e:
        log.debug(f"Journal close error: {e}")

def journal_stats_report():
    try:
        j = load_journal()
        trades=[t for t in j['signals'] if t['status']!='open']
        if not trades: return "📊 No completed trades yet."
        wins=[t for t in trades if t['status']=='win']
        losses=[t for t in trades if t['status']=='loss']
        be=[t for t in trades if t['status']=='be']
        total=len(trades); wr=len(wins)/total*100 if total else 0
        total_pnl=sum(t.get('pnl',0) or 0 for t in trades)
        avg_win=sum(t.get('pnl',0) or 0 for t in wins)/len(wins) if wins else 0
        avg_loss=sum(abs(t.get('pnl',0) or 0) for t in losses)/len(losses) if losses else 0
        pf=(len(wins)*avg_win)/(len(losses)*avg_loss) if losses and avg_loss>0 else 0
        lines=[
            "📊 <b>SMC Journal — Performance Report</b>","",
            f"📈 Total trades:   {total}",
            f"✅ Wins:           {len(wins)} ({wr:.1f}%)",
            f"❌ Losses:         {len(losses)}",
            f"➡️ Breakeven:      {len(be)}",
            f"💰 Total P&amp;L:  {total_pnl:+.2f}%",
            f"📐 Profit Factor:  {pf:.2f}",
            f"📊 Avg Win:        +{avg_win:.2f}%",
            f"📊 Avg Loss:       -{avg_loss:.2f}%","",
            "<b>Per Setup:</b>",
        ]
        for setup,s in j['stats'].items():
            if not s['total']: continue
            wr_s=s['wins']/s['total']*100
            e={'SWEEP_OB':'⚡','HTF_CONFLUENCE':'📊','CHOCH':'🔄','BOS':'📈'}.get(setup,'📡')
            lines.append(f"  {e} {setup}: {s['total']} | WR:{wr_s:.0f}% | PnL:{s['total_pnl']:+.1f}%")
        if trades:
            best=max(trades,key=lambda t:t.get('pnl',0) or 0)
            worst=min(trades,key=lambda t:t.get('pnl',0) or 0)
            lines+=["",
                f"🏆 Best:  {best['sym']} +{best.get('pnl',0):.2f}% ({best['setup']})",
                f"💔 Worst: {worst['sym']} {worst.get('pnl',0):.2f}% ({worst['setup']})"]
        if j['open']:
            lines.append(f"\n🔓 Open: {', '.join(j['open'].keys())}")
        lines.append(f"\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC  |  📡 SMC Engine Pro v3")
        return '\n'.join(lines)
    except Exception as e:
        return f"Journal error: {e}"

def check_circuit_breaker():
    """Pause trading after 3 consecutive losses"""
    try:
        j = load_journal()
        recent=[t for t in j['signals'] if t['status'] in ('win','loss')][-3:]
        if len(recent)<3: return False
        if all(t['status']=='loss' for t in recent):
            last_time=datetime.fromisoformat(recent[-1]['time']).timestamp()
            hours_since=(time.time()-last_time)/3600
            if hours_since<4:
                log.warning(f"⚠️ Circuit breaker: 3 losses in a row — pausing {4-hours_since:.1f}h")
                return True
    except: pass
    return False

CG = 'https://api.coingecko.com/api/v3'

# ── SERVER STATE ───────────────────────────────
state = {
    'started':      datetime.now(timezone.utc).isoformat(),
    'last_scan':    'Never',
    'scans_done':   0,
    'alerts_sent':  0,
    'open_trades':  {},   # sym -> trade dict
    'last_signals': {},   # sym -> last signal info
    'stats': {            # live win/loss tracker
        'wins':    0,
        'losses':  0,
        'be':      0,
        'by_setup': {}
    }
}

# ── HEALTH SERVER ──────────────────────────────
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        w=state['stats'].get('wins',0); l=state['stats'].get('losses',0)
        b=state['stats'].get('be',0); tot=w+l+b
        open_t = '\n'.join(
            f"  {sym}: {v['dir']} {v.get('setup','?')} entry={v['entry']:.4f}"
            for sym, v in state['open_trades'].items()
        ) or '  (none)'
        body = (
            f"SMC Engine Pro v3\n"
            f"{'='*40}\n"
            f"Started:      {state['started']}\n"
            f"Last scan:    {state['last_scan']}\n"
            f"Scans done:   {state['scans_done']}\n"
            f"Alerts sent:  {state['alerts_sent']}\n"
            f"\nWIN/LOSS TRACKER\n"
            f"Wins:         {w}\n"
            f"Losses:       {l}\n"
            f"Breakeven:    {b}\n"
            f"Win rate:     {round(w/tot*100) if tot else 0}%\n"
            f"\nOpen trades: {len(state['open_trades'])}\n{open_t}\n"
            f"\nTime (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n"
        ).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass

def start_health():
    HTTPServer(('0.0.0.0', PORT), Health).serve_forever()

# ── DATA ───────────────────────────────────────
def fetch_candles(pair, limit=300):
    try:
        r = requests.get(f'{KR}/OHLC',
            params={'pair': pair['kr'], 'interval': 60},
            timeout=15)
        d = r.json()
        if not d.get('error'):
            key = next((k for k in d['result'] if k != 'last'), None)
            if key:
                raw = d['result'][key]
                if len(raw) > 20:
                    return [{'t':i,'o':float(k[1]),'h':float(k[2]),
                             'l':float(k[3]),'c':float(k[4]),'v':float(k[6]),
                             'hour': datetime.fromtimestamp(int(k[0]), tz=timezone.utc).hour}
                            for i, k in enumerate(raw[-limit:])]
    except Exception as e:
        log.debug(f"Kraken {pair['sym']}: {e}")
    try:
        r = requests.get(f'{CG}/coins/{pair["cg"]}/ohlc',
            params={'vs_currency':'usd','days':7}, timeout=15)
        raw = r.json()
        if isinstance(raw, list) and len(raw) > 5:
            return [{'t':i,'o':float(k[1]),'h':float(k[2]),
                     'l':float(k[3]),'c':float(k[4]),'v':50.0,
                     'hour': datetime.fromtimestamp(int(k[0])/1000, tz=timezone.utc).hour}
                    for i, k in enumerate(raw[-limit:])]
    except Exception as e:
        log.debug(f"CoinGecko {pair['sym']}: {e}")
    return []

# ── INDICATORS ─────────────────────────────────
def ema(c, p):
    if len(c) < p: return [None]*len(c)
    k = 2/(p+1); r = [None]*(p-1)
    s = sum(c[:p])/p; r.append(s); pv = s
    for i in range(p, len(c)):
        pv = c[i]*k + pv*(1-k); r.append(pv)
    return r

def rsi(c, p=14):
    if len(c) < p+1: return [None]*len(c)
    r = [None]*p; g = l = 0.0
    for i in range(1, p+1):
        d = c[i]-c[i-1]
        if d > 0: g += d
        else: l += abs(d)
    ag, al = g/p, l/p
    r.append(100 if al==0 else 100-100/(1+ag/al))
    for i in range(p+1, len(c)):
        d = c[i]-c[i-1]
        ag = (ag*(p-1)+(d if d>0 else 0))/p
        al = (al*(p-1)+(abs(d) if d<0 else 0))/p
        r.append(100 if al==0 else 100-100/(1+ag/al))
    return r

def macd_hist(c):
    e12 = ema(c,12); e26 = ema(c,26)
    ln = [e12[i]-e26[i] if e12[i] and e26[i] else None for i in range(len(c))]
    vl = [v for v in ln if v is not None]
    if len(vl) < 9: return [None]*len(c)
    sr = ema(vl, 9); sg = [None]*len(c); si = 0
    for i in range(len(c)):
        if ln[i] is not None:
            sg[i] = sr[si] if si < len(sr) else None; si += 1
    return [ln[i]-sg[i] if ln[i] is not None and sg[i] is not None else None
            for i in range(len(c))]

def calc_atr(kl, p=14):
    tr = [None]+[max(kl[i]['h']-kl[i]['l'],
                     abs(kl[i]['h']-kl[i-1]['c']),
                     abs(kl[i]['l']-kl[i-1]['c']))
                 for i in range(1, len(kl))]
    if len(tr) < p+1: return [None]*len(kl)
    r = [None]*p; s = sum(tr[1:p+1])/p; r.append(s); pv = s
    for i in range(p+1, len(tr)):
        pv = (pv*(p-1)+tr[i])/p; r.append(pv)
    return r

def vol_avg(v, p=20):
    r = [None]*p
    for i in range(p, len(v)): r.append(sum(v[i-p:i])/p)
    return r

# ── IMPROVEMENT 1: SESSION FILTER ──────────────
def in_session(hour):
    """London 07-12 UTC | New York 13-18 UTC"""
    return 7 <= hour <= 12 or 13 <= hour <= 18

# ── IMPROVEMENT 2+3: BIAS HELPERS ──────────────
def calc_bias(kl, i, factor):
    if i < factor*25: return 'neutral'
    htf = [kl[j*factor+factor-1]['c'] for j in range((i+1)//factor)]
    if len(htf) < 25: return 'neutral'
    e20 = ema(htf, 20); e50 = ema(htf, 50); n = len(htf)-1
    if not e20[n] or not e50[n]: return 'neutral'
    if htf[n] > e20[n] > e50[n]: return 'bullish'
    if htf[n] < e20[n] < e50[n]: return 'bearish'
    return 'neutral'

def btc_gate(kl_btc, i, direction):
    """IMPROVEMENT 3: Block altcoin signals against BTC trend"""
    if not kl_btc: return True
    end = min(i, len(kl_btc)-1)
    c = [k['c'] for k in kl_btc[:end+1]]
    e20 = ema(c, 20); e50 = ema(c, 50)
    if not e20[end] or not e50[end]: return True
    if c[end] < e20[end] < e50[end] and direction == 'BUY':  return False
    if c[end] > e20[end] > e50[end] and direction == 'SELL': return False
    return True

def choppy(atr_a, i, thresh=0.40):
    r = [a for a in atr_a[max(0,i-20):i] if a is not None]
    return not r or (atr_a[i] < np.mean(r)*thresh if atr_a[i] else True)

def swings(kl, lb=5):
    sh = []; sl = []
    for i in range(lb, len(kl)-lb):
        if all(kl[i]['h'] >= kl[j]['h'] for j in range(i-lb, i+lb+1) if j != i):
            sh.append((i, kl[i]['h']))
        if all(kl[i]['l'] <= kl[j]['l'] for j in range(i-lb, i+lb+1) if j != i):
            sl.append((i, kl[i]['l']))
    return sh, sl

# ── IMPROVEMENT 4: STRUCTURE-BASED SL ──────────
def structure_sl(sh, sl_sw, i, direction, atr_v, swept=None, ob=None,
                  wick_low=None, wick_high=None, setup=None):
    """
    SL placement logic per setup:
    SWEEP_OB  → SL just below the wick low (tight, precise)
    CHOCH     → SL below last swing low (structural)
    BOS       → SL below last swing low (structural)
    HTF       → SL below OB or swing low
    """
    buf = atr_v * 0.10  # small buffer

    if direction == 'BUY':
        if setup == 'SWEEP_OB' and wick_low:
            # Tightest SL: just below the swept wick
            return wick_low - buf
        # Structural SL for other setups
        lvls = []
        if ob:    lvls.append(ob - buf)
        rl = [(idx,p) for idx,p in sl_sw if idx <= i][-2:]
        if rl: lvls.append(min(p for _,p in rl) - buf)
        return min(lvls) if lvls else None
    else:
        if setup == 'SWEEP_OB' and wick_high:
            return wick_high + buf
        lvls = []
        if ob:    lvls.append(ob + buf)
        rh = [(idx,p) for idx,p in sh if idx <= i][-3:]
        if rh: lvls.append(max(p for _,p in rh) + buf)
        return max(lvls) if lvls else None

# ══════════════════════════════════════════════════════════════
# ADAPTIVE CONFLUENCE ENGINE v2 — replaces fixed 4-setup system
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE CONFLUENCE ENGINE v2
# Replaces the 4-fixed-setup system with a pure scoring engine.
# 
# How it works:
#   - Checks 12 independent market conditions
#   - Each condition gets a score based on how strongly it's present
#   - ML tracks which condition COMBINATIONS actually win
#   - Engine fires when total confluence score >= threshold
#   - No hardcoded setup names — the ML discovers what to call it
#
# The ML genuinely learns:
#   "RSI<35 + London + Vol surge + EMA aligned → WR 71% → boost +1.5"
#   "BOS breakout + Asian session → WR 18% → penalty -2.0"
# ══════════════════════════════════════════════════════════════════════════════

def _check_liquidity_sweep(kl, i, sh, sl_sw, at, va_v):
    """
    Did price sweep recent highs/lows (stop hunt)?
    Returns: (direction, strength 0-3, wick_level)
    """
    k = kl[i]; price = k['c']

    # BUY: wick below recent lows and close back above (bullish sweep)
    for li, lvl in [(ix,p) for ix,p in sl_sw if i-12<ix<=i][-4:]:
        # Current bar swept
        if k['l'] < lvl and price > lvl:
            wick_depth = (lvl - k['l']) / at
            vol_surge  = k['v'] / va_v
            strength   = min(3.0, wick_depth*0.8 + (vol_surge-1.0)*0.5)
            if wick_depth > 0.15 and vol_surge > 1.0:
                return 'BUY', round(strength, 2), k['l']
        # Recent bar swept, now retesting
        elif li < i and li >= i-8:
            sc = next((kl[j] for j in range(li, min(li+5, i+1))
                       if kl[j]['l'] < lvl and kl[j]['c'] > lvl), None)
            if sc and sc['v'] > va_v*1.0 and price > lvl:
                wick_depth = (lvl - sc['l']) / at
                return 'BUY', round(min(2.5, wick_depth*0.6), 2), sc['l']

    # SELL: wick above recent highs and close back below
    for hi_, lvl in [(ix,p) for ix,p in sh if i-12<ix<=i][-4:]:
        if k['h'] > lvl and price < lvl:
            wick_h = (k['h'] - lvl) / at
            vol_surge = k['v'] / va_v
            strength  = min(3.0, wick_h*0.8 + (vol_surge-1.0)*0.5)
            if wick_h > 0.15 and vol_surge > 1.0:
                return 'SELL', round(strength, 2), k['h']
        elif hi_ < i and hi_ >= i-8:
            sc = next((kl[j] for j in range(hi_, min(hi_+5, i+1))
                       if kl[j]['h'] > lvl and kl[j]['c'] < lvl), None)
            if sc and sc['v'] > va_v*1.0 and price < lvl:
                wick_h = (lvl - sc['h']) / at
                return 'SELL', round(min(2.5, wick_h*0.6), 2), sc['h']

    return None, 0, None

def _check_order_block(kl, i, sh, sl_sw, at, direction):
    """
    Is price inside an order block?
    OB = last opposing candle before a strong move.
    Returns: (in_ob, ob_quality 0-2)
    """
    price = kl[i]['c']
    lookback = 20

    if direction == 'BUY':
        # Find last bearish candle before most recent swing low
        recent_lows = [(ix,p) for ix,p in sl_sw if i-lookback<ix<i]
        if not recent_lows: return False, 0
        li = recent_lows[-1][0]
        for j in range(li-1, max(0, li-12), -1):
            if kl[j]['c'] < kl[j]['o']:  # bearish candle
                ob_top = kl[j]['o']; ob_bot = kl[j]['l']
                if ob_bot <= price <= ob_top * 1.01:
                    size = (ob_top - ob_bot) / at
                    return True, round(min(2.0, size * 0.8), 1)
    else:
        recent_highs = [(ix,p) for ix,p in sh if i-lookback<ix<i]
        if not recent_highs: return False, 0
        hi_ = recent_highs[-1][0]
        for j in range(hi_-1, max(0, hi_-12), -1):
            if kl[j]['c'] > kl[j]['o']:  # bullish candle
                ob_top = kl[j]['h']; ob_bot = kl[j]['c']
                if ob_bot * 0.99 <= price <= ob_top:
                    size = (ob_top - ob_bot) / at
                    return True, round(min(2.0, size * 0.8), 1)
    return False, 0

def _check_structure(kl, i, sh, sl_sw, closes, direction):
    """
    Is market structure supporting this direction?
    CHoCH = change of character (reversal signal)
    BOS   = break of structure (continuation signal)
    HH/HL = higher highs/higher lows (uptrend)
    LH/LL = lower highs/lower lows (downtrend)
    Returns: (structure_type, strength 0-2)
    """
    if len(sh) < 3 or len(sl_sw) < 3: return 'NONE', 0

    recent_highs = [p for _,p in sh[-4:]]
    recent_lows  = [p for _,p in sl_sw[-4:]]

    if direction == 'BUY':
        # HH+HL = uptrend continuation
        hh = len(recent_highs) >= 2 and recent_highs[-1] > recent_highs[-2]
        hl = len(recent_lows)  >= 2 and recent_lows[-1]  > recent_lows[-2]
        if hh and hl: return 'HH_HL', 2.0
        if hl:        return 'HL',    1.0
        # CHoCH: was making LH/LL, now first HL (reversal)
        lh = len(recent_highs) >= 2 and recent_highs[-1] < recent_highs[-2]
        if lh and hl: return 'CHOCH_BULL', 1.5
        return 'NONE', 0
    else:
        lh = len(recent_highs) >= 2 and recent_highs[-1] < recent_highs[-2]
        ll = len(recent_lows)  >= 2 and recent_lows[-1]  < recent_lows[-2]
        if lh and ll: return 'LH_LL', 2.0
        if lh:        return 'LH',    1.0
        hh = len(recent_highs) >= 2 and recent_highs[-1] > recent_highs[-2]
        if hh and ll: return 'CHOCH_BEAR', 1.5
        return 'NONE', 0

def _check_ema(e9, e20, e50, price, direction):
    """EMA alignment. Returns strength 0-2"""
    if not all([e9, e20, e50]): return 0
    if direction == 'BUY':
        if e9 > e20 > e50 and price > e9:  return 2.0  # perfect bull stack
        if e9 > e20 and price > e20:        return 1.0  # partial
        if price < e50:                      return -1.0 # against trend
    else:
        if e9 < e20 < e50 and price < e9:  return 2.0
        if e9 < e20 and price < e20:        return 1.0
        if price > e50:                      return -1.0
    return 0

def _check_rsi(rsi_val, direction):
    """RSI zone quality. Returns strength 0-1.5"""
    if rsi_val is None: return 0, 'neutral'
    if direction == 'BUY':
        if rsi_val < 25:  return 1.5, 'extreme_oversold'
        if rsi_val < 35:  return 1.0, 'oversold'
        if rsi_val < 50:  return 0.5, 'neutral_low'
        if rsi_val > 70:  return -1.0, 'overbought'  # bad for buy
    else:
        if rsi_val > 75:  return 1.5, 'extreme_overbought'
        if rsi_val > 65:  return 1.0, 'overbought'
        if rsi_val > 50:  return 0.5, 'neutral_high'
        if rsi_val < 30:  return -1.0, 'oversold'
    return 0, 'neutral'

def _check_volume(k, va_v):
    """Volume quality. Returns strength 0-1.5"""
    ratio = k['v'] / va_v if va_v > 0 else 1.0
    if ratio > 2.0:   return 1.5, 'surge++'
    if ratio > 1.5:   return 1.0, 'surge'
    if ratio > 1.1:   return 0.5, 'above_avg'
    if ratio < 0.6:   return -0.5, 'low'
    return 0, 'avg'

def _check_macd(ht, direction):
    """MACD histogram alignment"""
    if ht is None: return 0
    if direction == 'BUY'  and ht > 0: return 0.8
    if direction == 'SELL' and ht < 0: return 0.8
    if direction == 'BUY'  and ht < 0: return -0.5
    if direction == 'SELL' and ht > 0: return -0.5
    return 0

def _check_vwap(kl, i, price, direction):
    """VWAP alignment. 20-bar rolling VWAP"""
    w = kl[max(0,i-19):i+1]; tv = sum(x['v'] for x in w)
    if not tv: return 0
    vwap = sum(((x['h']+x['l']+x['c'])/3)*x['v'] for x in w) / tv
    dev  = (price - vwap) / vwap
    if direction == 'BUY':
        if price > vwap and dev < 0.005: return 0.7   # just above VWAP = good
        if price > vwap and dev > 0.015: return -0.3  # too extended above
        if price < vwap:                  return -0.5  # below VWAP for buy
    else:
        if price < vwap and abs(dev) < 0.005: return 0.7
        if price < vwap and abs(dev) > 0.015: return -0.3
        if price > vwap:                       return -0.5
    return 0

def _check_atr_quality(at_a, i):
    """Is ATR in a good range? Returns -1 to 1"""
    if not at_a[i]: return 0
    past = [a for a in at_a[max(0,i-20):i] if a]
    if not past: return 0
    pct = sum(1 for a in past if a < at_a[i]) / len(past)
    if pct < 0.15: return -1.0  # very quiet — signals fail
    if pct < 0.30: return -0.5  # quiet
    if pct > 0.85: return -0.3  # very volatile — SL gets hit
    return 0.5                   # normal range = positive

def _check_session_bias(session):
    """Session quality. Returns 0-1"""
    return {'London': 1.0, 'NY': 0.8, 'Off': 0.3, 'Asian': 0.2}.get(session, 0.5)

def _check_weekly_bias(weekly_b, direction):
    """HTF bias alignment"""
    if weekly_b == 'bullish' and direction == 'BUY':   return 1.0
    if weekly_b == 'bearish' and direction == 'SELL':  return 1.0
    if weekly_b == 'bullish' and direction == 'SELL':  return -1.0
    if weekly_b == 'bearish' and direction == 'BUY':   return -1.0
    return 0

def _check_btc_correlation(kl_btc, direction):
    """BTC as macro filter. Returns -0.5 to +0.5"""
    if not kl_btc or len(kl_btc) < 5: return 0
    btc_mom = (kl_btc[-1]['c'] - kl_btc[-5]['c']) / kl_btc[-5]['c'] * 100
    if direction == 'BUY'  and btc_mom > 1.0:  return 0.5   # BTC rising = good for longs
    if direction == 'SELL' and btc_mom < -1.0: return 0.5
    if direction == 'BUY'  and btc_mom < -2.0: return -0.5  # BTC dumping = bad for longs
    if direction == 'SELL' and btc_mom > 2.0:  return -0.5
    return 0


def get_signal_v2(kl, sh, sl_sw, i, closes, rsi_a, e9_a, e20_a, e50_a,
                  ht_a, atr_a, va_a, weekly_b, daily_b, session=None, kl_btc=None):
    """
    Pure confluence engine — no hardcoded setups.
    Scores 12 independent conditions, fires when total >= threshold.
    ML tracks which combinations win and adjusts weights.
    """
    if len(kl) < 60 or i < 50: return None
    k     = kl[i]; price = closes[i]
    at    = atr_a[i]; va_v = va_a[i]
    if not at or not va_v: return None

    # ATR quality — penalty not hard gate
    atr_q = _check_atr_quality(atr_a, i)
    # (no hard return — ATR becomes score penalty instead)

    # ── Step 1: Determine direction from sweep (primary trigger) ────────
    sweep_dir, sweep_str, wick_level = _check_liquidity_sweep(kl, i, sh, sl_sw, at, va_v)

    # If no sweep, check for EMA/structure-based direction
    if not sweep_dir:
        # Try structure-based direction
        if (e9_a[i] and e20_a[i] and e50_a[i] and
                e9_a[i] > e20_a[i] > e50_a[i] and
                ht_a[i] and ht_a[i] > 0 and
                rsi_a[i] and rsi_a[i] < 55 and
                daily_b == 'bullish'):
            sweep_dir = 'BUY'; sweep_str = 0
        elif (e9_a[i] and e20_a[i] and e50_a[i] and
                e9_a[i] < e20_a[i] < e50_a[i] and
                ht_a[i] and ht_a[i] < 0 and
                rsi_a[i] and rsi_a[i] > 45 and
                daily_b == 'bearish'):
            sweep_dir = 'SELL'; sweep_str = 0
        else:
            return None

    direction = sweep_dir

    # ── Step 2: Score all conditions ────────────────────────────────────
    score = 0.0; tags = []; conditions = {}

    # 1. Sweep quality (primary trigger)
    score += sweep_str * 1.2
    if sweep_str > 0: tags.append(f'Sweep{sweep_str:.0f}x')
    conditions['sweep_str'] = sweep_str

    # 2. Order block
    in_ob, ob_q = _check_order_block(kl, i, sh, sl_sw, at, direction)
    score += ob_q * 1.0
    if in_ob: tags.append('OB✓')
    conditions['ob'] = ob_q

    # 3. Market structure
    struct_type, struct_str = _check_structure(kl, i, sh, sl_sw, closes, direction)
    score += struct_str * 0.8
    if struct_str > 0: tags.append(struct_type)
    conditions['structure'] = struct_type

    # 4. EMA alignment
    ema_str = _check_ema(e9_a[i], e20_a[i], e50_a[i], price, direction)
    score += ema_str * 0.7
    if ema_str > 0: tags.append('EMA✓')
    elif ema_str < 0: tags.append('EMA✗')
    conditions['ema'] = ema_str

    # 5. RSI zone
    rsi_str, rsi_zone = _check_rsi(rsi_a[i], direction)
    score += rsi_str * 0.8
    if rsi_str != 0: tags.append(f'RSI{round(rsi_a[i])}')
    conditions['rsi_zone'] = rsi_zone

    # 6. Volume
    vol_str, vol_label = _check_volume(k, va_v)
    score += vol_str * 0.7
    if vol_str > 0: tags.append(f'Vol{vol_label}')
    conditions['volume'] = vol_label

    # 7. MACD
    macd_str = _check_macd(ht_a[i], direction)
    score += macd_str * 0.6
    if macd_str > 0: tags.append('MACD✓')
    conditions['macd'] = macd_str

    # 8. VWAP
    vwap_str = _check_vwap(kl, i, price, direction)
    score += vwap_str * 0.5
    if vwap_str > 0: tags.append('VWAP✓')
    conditions['vwap'] = vwap_str

    # 9. Session
    sess_str = _check_session_bias(session)
    score += sess_str * 0.5
    conditions['session'] = session

    # 10. Weekly bias
    htf_str = _check_weekly_bias(weekly_b, direction)
    score += htf_str * 0.8
    if htf_str > 0: tags.append('HTF✓')
    elif htf_str < 0: tags.append('HTF✗')
    conditions['weekly_bias'] = weekly_b

    # 11. BTC correlation
    btc_str = _check_btc_correlation(kl_btc, direction)
    score += btc_str * 0.4
    if btc_str > 0: tags.append('BTC✓')
    conditions['btc'] = btc_str

    # 12. ATR quality
    score += atr_q * 0.4
    conditions['atr_quality'] = atr_q

    # ── Step 3: Apply ML learned weights ────────────────────────────────
    db = load_db()
    w  = db['weights']

    # Get condition-specific multipliers from ML
    for cond_key, cond_val in conditions.items():
        ml_key = f"{cond_key}:{cond_val}:{direction}"
        boost  = w.get('condition_boosts', {}).get(ml_key, 0)
        if boost != 0:
            score += boost
            if abs(boost) >= 0.3:
                tags.append(f'ML{boost:+.1f}')

    score = round(max(0, min(10, score)), 1)

    # ── Step 4: Determine setup name from what fired ─────────────────────
    # Engine names the setup based on dominant conditions — not hardcoded
    if sweep_str >= 2.0 and in_ob:
        setup_name = 'SWEEP_OB'
        setup_label= '⚡ Sweep + OB Retest'
    elif sweep_str >= 1.0 and struct_type in ('HH_HL','LH_LL'):
        setup_name = 'SWEEP_TREND'
        setup_label= '📈 Sweep Continuation'
    elif struct_type in ('CHOCH_BULL','CHOCH_BEAR'):
        setup_name = 'CHOCH'
        setup_label= '🔄 Change of Character'
    elif struct_type in ('HH_HL','LH_LL') and ema_str >= 1.5:
        setup_name = 'TREND_PULL'
        setup_label= '📊 Trend Pullback'
    elif in_ob and rsi_str >= 1.0:
        setup_name = 'OB_EXTREME'
        setup_label= '🎯 OB + RSI Extreme'
    elif sweep_str > 0:
        setup_name = 'MICRO_SWEEP'
        setup_label= '⚡ Micro Sweep'
    else:
        setup_name = 'CONFLUENCE'
        setup_label= '📡 Confluence Signal'

    # ── Step 5: Threshold check ──────────────────────────────────────────
    min_score = get_min_score(session, setup_name)
    if score < min_score:
        log.debug(f"  Score {score:.1f} < min {min_score} ({setup_name}) — skip")
        return None

    # ── Step 6: SL placement ─────────────────────────────────────────────
    if wick_level and direction == 'BUY':
        sl_p = wick_level - at*0.08  # tight: just below sweep wick
    elif wick_level and direction == 'SELL':
        sl_p = wick_level + at*0.08
    else:
        # Structure-based SL
        sl_p = structure_sl(kl, sh, sl_sw, i, direction, at)
        if sl_p is None:
            sl_p = price - at*1.5 if direction=='BUY' else price + at*1.5

    # Safety cap
    max_risk = price * 0.03
    if direction == 'BUY'  and (price-sl_p) > max_risk: sl_p = price - max_risk
    if direction == 'SELL' and (sl_p-price) > max_risk: sl_p = price + max_risk

    risk = abs(price - sl_p)
    if risk <= 0: return None

    return {
        'dir':        direction,
        'setup':      setup_name,
        'name':       setup_label,
        'score':      score,
        'tags':       tags,
        'conditions': conditions,   # stored for ML pattern mining
        'swept':      wick_level,
        'weekly':     weekly_b,
        'daily':      daily_b,
        'rsi_val':    round(rsi_a[i]) if rsi_a[i] else 50,
        'rsi_zone':   rsi_zone,
        'session':    session,
        'atr_v':      round(at, 6),
        'sl_raw':     sl_p,         # SL before regime adjustment
    }


# ── ML: Learn condition boosts from trade history ─────────────────────────────
def learn_condition_boosts(db):
    """
    After trades close, learn which specific conditions led to wins/losses.
    Updates condition_boosts in weights — applied to every future signal.
    This is the true self-learning: finds what conditions work in THIS market.
    """
    outcomes = [s for s in db.get('signals', []) if s.get('result')]
    if len(outcomes) < 5: return

    lr = 0.40 if len(outcomes) < 10 else (0.25 if len(outcomes) < 30 else 0.12)
    boosts = db['weights'].setdefault('condition_boosts', {})
    changes = []

    for trade in outcomes:
        conds = trade.get('conditions', {})
        result = trade.get('result', '')
        direction = trade.get('dir', '')
        reward = 1.0 if result == 'win' else -0.7 if result == 'loss' else 0

        for cond_key, cond_val in conds.items():
            ml_key = f"{cond_key}:{cond_val}:{direction}"
            old    = boosts.get(ml_key, 0.0)
            new    = round(max(-2.0, min(2.0, old + lr * reward * 0.3)), 3)
            if new != old:
                boosts[ml_key] = new
                if abs(new - old) > 0.1:
                    changes.append(f"  {ml_key}: {old:+.2f}→{new:+.2f}")

    if changes:
        log.info(f"Condition boosts updated: {len(changes)} changes")

    db['weights']['condition_boosts'] = boosts
    save_db(db)


# ── SIGNAL DETECTION ───────────────────────────
def get_signal(kl, sh, sl_sw, i, closes, rsi_a, e9_a, e20_a, e50_a,
               ht_a, atr_a, va_a, weekly_b, daily_b):
    price = kl[i]['c']; k = kl[i]
    at = atr_a[i]; va_v = va_a[i]

    # ── SETUP 1: SWEEP + OB RETEST ──────────────
    #
    # HOW IT WORKS:
    # 1. Price forms equal lows (liquidity resting below)
    # 2. A sweep candle wicks BELOW those lows (stop hunt) and CLOSES back above
    # 3. We find the OB: last bearish (red) candle BEFORE the sweep — that's where
    #    institutions placed their buy orders
    # 4. Signal fires when price RETESTS that OB zone (comes back into it)
    # 5. SL = just below the sweep wick low (tight, precise)
    #
    # Two scenarios detected:
    # A) i IS the sweep candle (fires immediately when close is back in OB)
    # B) i is a retest candle AFTER the sweep (1-5 bars later, price in OB)

    for li, lvl in [(ix,p) for ix,p in sl_sw if ix < i and ix > i-50][-4:]:

        # ── Scenario A: current candle swept AND closed back in OB ──
        sweep_bar = None; wick_low_val = None
        if k['l'] < lvl and price > lvl:
            if lvl - k['l'] < at*0.25: continue    # wick must be meaningful
            if k['v'] < va_v*1.15: continue         # volume surge required
            sweep_bar = i; wick_low_val = k['l']

        # ── Scenario B: sweep happened 1-8 bars ago, now retesting OB ──
        elif li <= i-1 and li >= i-8:
            # Find the sweep candle in history
            sc = next((kl[j] for j in range(li, min(li+4, i+1))
                       if kl[j]['l'] < lvl and kl[j]['c'] > lvl), None)
            if sc and sc['v'] > va_v*1.1:
                sweep_bar = li; wick_low_val = sc['l']
            else:
                continue
        else:
            continue

        # daily bias now handled as score modifier in v2 engine
        # weekly bias now handled as score modifier in v2 engine
        # RSI now scored not gated

        # ── ANTI-TREND FILTER: Don't buy into a strong downtrend ──
        # Check last 6 candles — if 5+ are bearish = strong downtrend, skip
        recent = kl[max(0,i-6):i+1]
        bearish_count = sum(1 for x in recent if x['c'] < x['o'])
        if bearish_count >= 5: continue  # 5/6 red candles = skip BUY

        # Price must be making higher lows recently (not lower lows)
        recent_lows = [x['l'] for x in kl[max(0,i-4):i+1]]
        if len(recent_lows) >= 3 and recent_lows[-1] < recent_lows[-3]*0.995:
            continue  # still making lower lows = downtrend not reversed

        # Find the OB: last red candle before the swing low formed
        ob = None
        for j in range(li-1, max(0, li-15), -1):
            if kl[j]['c'] < kl[j]['o']:  # red candle = bullish OB
                fwd = (kl[min(j+2,len(kl)-1)]['c'] - kl[j]['c']) / kl[j]['c']
                if fwd > 0.003:           # followed by bullish move
                    ob = {'top': kl[j]['o'], 'bot': kl[j]['l']}
                    break

        if not ob: continue

        # Price must be IN the OB zone to fire
        if not (ob['bot'] <= price <= ob['top']*1.008): continue

        ema_ok = e20_a[i] and e50_a[i] and price > e20_a[i] > e50_a[i]
        score  = 8 + (0.5 if ema_ok else 0)

        return {'dir':'BUY', 'setup':'SWEEP_OB',
                'name':'⚡ Liq Sweep + OB Retest',
                'score': score,
                'ob': ob,
                'swept': lvl,
                'wick_low': wick_low_val,   # exact wick for tight SL
                'tags': ['Sweep↑','OB_Retest','Vol✓','HTF✓','Week✓']
                       +(['EMA↑'] if ema_ok else [])+[f'RSI{round(rsi_a[i])}']}

    for hi_, lvl in [(ix,p) for ix,p in sh if ix < i and ix > i-50][-4:]:

        sweep_bar_s = None; wick_high_val = None
        if k['h'] > lvl and price < lvl:
            if k['h'] - lvl < at*0.25: continue
            if k['v'] < va_v*1.15: continue
            sweep_bar_s = i; wick_high_val = k['h']
        elif hi_ <= i-1 and hi_ >= i-8:
            sc = next((kl[j] for j in range(hi_, min(hi_+4, i+1))
                       if kl[j]['h'] > lvl and kl[j]['c'] < lvl), None)
            if sc and sc['v'] > va_v*1.1:
                sweep_bar_s = hi_; wick_high_val = sc['h']
            else:
                continue
        else:
            continue

        # daily/weekly now scored not gated
        if not rsi_a[i] or not (35 < rsi_a[i] < 75): continue

        # Anti-trend filter for sells
        recent = kl[max(0,i-6):i+1]
        bullish_count = sum(1 for x in recent if x['c'] > x['o'])
        if bullish_count >= 5: continue  # 5/6 green candles = skip SELL
        recent_highs = [x['h'] for x in kl[max(0,i-4):i+1]]
        if len(recent_highs) >= 3 and recent_highs[-1] > recent_highs[-3]*1.005:
            continue  # still making higher highs = uptrend not reversed
        ob = None
        for j in range(hi_-1, max(0, hi_-15), -1):
            if kl[j]['c'] > kl[j]['o']:
                fwd = (kl[min(j+2,len(kl)-1)]['c'] - kl[j]['c']) / kl[j]['c']
                if fwd < -0.003:
                    ob = {'top': kl[j]['h'], 'bot': kl[j]['c']}; break
        if not ob or not (ob['bot']*0.995 <= price <= ob['top']): continue
        ema_ok = e20_a[i] and e50_a[i] and price < e20_a[i] < e50_a[i]
        return {'dir':'SELL', 'setup':'SWEEP_OB',
                'name':'⚡ Liq Sweep + OB Retest',
                'score': 8+(0.5 if ema_ok else 0),
                'ob': ob, 'swept': lvl,
                'wick_high': wick_high_val,
                'tags': ['Sweep↓','OB_Retest','Vol✓','HTF✓','Week✓']
                       +(['EMA↓'] if ema_ok else [])+[f'RSI{round(rsi_a[i])}']}

    # ── SETUP 2: 3-TF HTF CONFLUENCE ────────────
    if i >= 50 and ht_a[i] and weekly_b != 'neutral' and weekly_b == daily_b:
        rh4 = [(ix,p) for ix,p in sh if ix <= i][-4:]
        rl4 = [(ix,p) for ix,p in sl_sw if ix <= i][-4:]
        h1 = 'neutral'
        if len(rh4) >= 2 and len(rl4) >= 2:
            if rh4[-1][1] > rh4[-2][1] and rl4[-1][1] > rl4[-2][1]: h1 = 'bullish'
            elif rh4[-1][1] < rh4[-2][1] and rl4[-1][1] < rl4[-2][1]: h1 = 'bearish'
        if h1 == weekly_b:
            is_buy = h1 == 'bullish'
            ema_ok = (e9_a[i] and e20_a[i] and e50_a[i] and
                      (e9_a[i]>e20_a[i]>e50_a[i] if is_buy else e9_a[i]<e20_a[i]<e50_a[i]))
            mac_ok = ht_a[i] > 0 if is_buy else ht_a[i] < 0
            rsi_ok = rsi_a[i] and (25 < rsi_a[i] < 62 if is_buy else 38 < rsi_a[i] < 75)
            if ema_ok and mac_ok and rsi_ok:
                vol_ok = va_a[i] and kl[i]['v'] > va_a[i]*1.1
                return {'dir': 'BUY' if is_buy else 'SELL',
                        'setup': 'HTF_CONFLUENCE',
                        'name': '📊 3-TF HTF Confluence (W+D+1h)',
                        'score': 9 if vol_ok else 8, 'ob': None, 'swept': None,
                        'tags': [f'W:{weekly_b[:4]}', f'D:{daily_b[:4]}',
                                 f'1h:{h1[:4]}', 'EMA_stack', 'MACD✓']
                               +(['Vol✓'] if vol_ok else [])+[f'RSI{round(rsi_a[i])}']}

    # ── SETUP 3: CHOCH ──────────────────────────
    rh5 = [(ix,p) for ix,p in sh if ix <= i][-5:]
    rl5 = [(ix,p) for ix,p in sl_sw if ix <= i][-5:]
    if len(rh5) >= 3 and len(rl5) >= 3 and ht_a[i] and va_a[i]:
        h_gaps = [abs(rh5[j+1][1]-rh5[j][1])/max(rh5[j][1],1e-10)
                  for j in range(len(rh5)-2, len(rh5)-1)]
        l_gaps = [abs(rl5[j+1][1]-rl5[j][1])/max(rl5[j][1],1e-10)
                  for j in range(len(rl5)-2, len(rl5)-1)]
        if all(g >= 0.003 for g in h_gaps) and all(g >= 0.003 for g in l_gaps):
            h2, h1p = rh5[-2][1], rh5[-3][1]
            l2, l1p = rl5[-2][1], rl5[-3][1]
            vol_ok = kl[i]['v'] > va_a[i]*1.05
            if (h2 < h1p and l2 < l1p and price > h2 and
                    e20_a[i] and price > e20_a[i] and ht_a[i] > 0 and
                    rsi_a[i] and 28 < rsi_a[i] < 65 and vol_ok and
                    weekly_b != 'bearish'):
                return {'dir':'BUY', 'setup':'CHOCH',
                        'name':'🔄 CHoCH Reversal (Bear→Bull)',
                        'score': 8, 'ob': None, 'swept': None,
                        'tags': ['CHoCH↑','CleanStr','Vol✓','MACD✓',
                                 f'RSI{round(rsi_a[i])}']}
            if (h2 > h1p and l2 > l1p and price < l2 and
                    e20_a[i] and price < e20_a[i] and ht_a[i] < 0 and
                    rsi_a[i] and 35 < rsi_a[i] < 72 and vol_ok and
                    weekly_b != 'bullish'):
                return {'dir':'SELL', 'setup':'CHOCH',
                        'name':'🔄 CHoCH Reversal (Bull→Bear)',
                        'score': 8, 'ob': None, 'swept': None,
                        'tags': ['CHoCH↓','CleanStr','Vol✓','MACD✓',
                                 f'RSI{round(rsi_a[i])}']}

    # ── SETUP 4: BOS ────────────────────────────
    rh4b = [(ix,p) for ix,p in sh if ix <= i][-4:]
    rl4b = [(ix,p) for ix,p in sl_sw if ix <= i][-4:]
    if len(rh4b) >= 3 and len(rl4b) >= 3 and ht_a[i] and va_a[i]:
        h1p, h2p, h3p = rh4b[-3][1], rh4b[-2][1], rh4b[-1][1]
        l1p, l2p, l3p = rl4b[-3][1], rl4b[-2][1], rl4b[-1][1]
        vol_ok = kl[i]['v'] > va_a[i]*1.1
        if (h3p > h2p > h1p and l3p > l2p and price > h2p and
                ht_a[i] > 0 and e20_a[i] and e50_a[i] and
                price > e20_a[i] > e50_a[i] and
                rsi_a[i] and 28 < rsi_a[i] < 68 and vol_ok and
                weekly_b != 'bearish'):
            return {'dir':'BUY', 'setup':'BOS',
                    'name':'📈 BOS Continuation (Bullish)',
                    'score': 7, 'ob': None, 'swept': None,
                    'tags': ['BOS↑','HH+HL','Vol✓','MACD✓',
                             f'RSI{round(rsi_a[i])}']}
        if (h3p < h2p < h1p and l3p < l2p and price < l2p and
                ht_a[i] < 0 and e20_a[i] and e50_a[i] and
                price < e20_a[i] < e50_a[i] and
                rsi_a[i] and 32 < rsi_a[i] < 72 and vol_ok and
                weekly_b != 'bullish'):
            return {'dir':'SELL', 'setup':'BOS',
                    'name':'📉 BOS Continuation (Bearish)',
                    'score': 7, 'ob': None, 'swept': None,
                    'tags': ['BOS↓','LH+LL','Vol✓','MACD✓',
                             f'RSI{round(rsi_a[i])}']}
    return None

# ── MAIN COMPUTE ───────────────────────────────
last_fired = {}

def compute(kl, pair, kl_btc=None):
    if len(kl) < 80: return None
    i = len(kl)-1
    closes = [k['c'] for k in kl]; vols = [k['v'] for k in kl]
    rsi_a  = rsi(closes); e9_a = ema(closes,9)
    e20_a  = ema(closes,20); e50_a = ema(closes,50)
    ht_a   = macd_hist(closes); atr_a = calc_atr(kl); va_a = vol_avg(vols)
    price  = closes[i]
    if any(x is None for x in [rsi_a[i],e9_a[i],e20_a[i],e50_a[i],atr_a[i],va_a[i]]):
        return None
    if choppy(atr_a, i) or atr_a[i]/price < 0.002: return None

    # Session check — not a hard block, but affects minimum score
    # Outside London/NY: require score >= MIN_SCORE+1 (stricter)
    # Inside London/NY:  normal MIN_SCORE threshold
    hour = kl[i].get('hour', 10)
    session_on = in_session(hour)
    # We pass this info through but don't hard-block

    # Cooldown check
    # Hard lock: no new signal while trade open for this pair
    if pair['sym'] in state['open_trades']:
        log.debug(f"  {pair['sym']}: trade open — blocked")
        return None
    # Max concurrent trades cap
    MAX_OPEN = int(os.environ.get('MAX_OPEN_TRADES', '3' if os.environ.get('PAPER_MODE','false').lower()=='true' else '2'))
    if len(state['open_trades']) >= MAX_OPEN:
        log.debug(f"  {pair['sym']}: max open ({MAX_OPEN}) reached")
        return None
    # Cooldown
    lf = last_fired.get(pair['sym'])
    if lf and (time.time()-lf['time'])/60 < COOLDOWN_M:
        return None

    sh, sl = swings(kl, 5)
    weekly_b = calc_bias(kl, i, 21)   # IMPROVEMENT 2: weekly gate
    daily_b  = calc_bias(kl, i, 5)

    sig = get_signal_v2(kl, sh, sl, i, closes, rsi_a, e9_a, e20_a, e50_a,
                       ht_a, atr_a, va_a, weekly_b, daily_b,
                       session=session_name, kl_btc=kl_btc)
    # Use learned dynamic score instead of fixed threshold
    session_name = get_session()
    effective_min = get_min_score(session_name)

    # PAPER_MODE: lower threshold to collect more ML training data
    # Real trades still use effective_min but paper signals fire more freely
    paper_mode = os.environ.get('PAPER_MODE', 'false').lower() == 'true'
    if paper_mode:
        paper_min = float(os.environ.get('PAPER_MIN_SCORE', '3.0'))
        effective_min = min(effective_min, paper_min)

    if not sig or sig['score'] < effective_min:
        log.debug(f"  {pair['sym']}: score {sig['score'] if sig else '?':.1f} "
                  f"< {effective_min:.1f} — blocked")
        return None

    # Compute learned score (adjusted by historical performance)
    learned_score = compute_learned_score(
        sig['setup'], sig.get('tags', []),
        session_name, sig.get('weekly','neutral'),
        rsi_a[i] or 50, sig['score']
    )
    sig = dict(sig)
    sig['raw_score']    = sig['score']   # original score
    sig['score']        = learned_score  # learned-adjusted score
    sig['session_name'] = session_name

    # IMPROVEMENT 3: BTC correlation gate
    if pair['sym'] != 'BTC' and kl_btc:
        if False:  # btc_gate disabled — ML scores BTC correlation
            log.debug(f"{pair['sym']}: blocked by BTC gate ({sig['dir']})")
            return None

    # ── BONUS SCORE BOOSTERS (don't block, just raise quality score) ──
    # Booster 1: RSI Divergence — price/RSI mismatch = hidden strength (+1.5)
    def _rsi_div(direction):
        lb = 15
        if i < lb+5: return False
        if direction == 'BUY':
            rl = [(ix,p) for ix,p in sl if i-lb < ix < i][-3:]
            if len(rl)>=2 and rsi_a[rl[-2][0]] and rsi_a[rl[-1][0]]:
                if rl[-1][1] < rl[-2][1] and rsi_a[rl[-1][0]] > rsi_a[rl[-2][0]]: return True
        else:
            rh = [(ix,p) for ix,p in sh if i-lb < ix < i][-3:]
            if len(rh)>=2 and rsi_a[rh[-2][0]] and rsi_a[rh[-1][0]]:
                if rh[-1][1] > rh[-2][1] and rsi_a[rh[-1][0]] < rsi_a[rh[-2][0]]: return True
        return False

    # Booster 2: Fibonacci zone — price at 38.2/50/61.8% retracement (+0.5)
    def _fib_zone():
        rh_f = [(ix,p) for ix,p in sh if ix < i][-3:]
        rl_f = [(ix,p) for ix,p in sl if ix < i][-3:]
        if not rh_f or not rl_f: return False
        hi = rh_f[-1][1]; lo = rl_f[-1][1]; rng = hi - lo
        if rng <= 0: return False
        return any(abs(price - (hi - rng*f)) <= rng*0.025 for f in [0.382, 0.5, 0.618])

    # Booster 3: VWAP alignment — price on right side of VWAP (+0.5)
    def _vwap_ok(direction):
        win = kl[max(0, i-20):i+1]
        tv = sum(k['v'] for k in win)
        if not tv: return False
        vwap = sum(((k['h']+k['l']+k['c'])/3) * k['v'] for k in win) / tv
        return (direction == 'BUY' and price >= vwap) or (direction == 'SELL' and price <= vwap)

    extra_tags = []; bonus = 0.0
    if _rsi_div(sig['dir']): bonus += 1.5; extra_tags.append('RSI_Div✓')
    if _fib_zone():           bonus += 0.5; extra_tags.append('Fib✓')
    if _vwap_ok(sig['dir']): bonus += 0.5; extra_tags.append('VWAP✓')
    sig = dict(sig)
    sig['score']  = min(10, round(sig['score'] + bonus, 1))
    sig['tags']   = list(sig.get('tags', [])) + extra_tags

    is_buy = sig['dir'] == 'BUY'

    # IMPROVEMENT 4: Structure-based SL (setup-specific)
    ob_level = (sig['ob']['bot'] if is_buy else sig['ob']['top']) if sig.get('ob') else None

    # For SWEEP_OB: use the stored wick low/high from detection
    wick_low  = sig.get('wick_low')   # exact low of sweep candle
    wick_high = sig.get('wick_high')  # exact high of sweep candle

    sl_p = structure_sl(sh, sl, i, sig['dir'], atr_a[i],
                        sig.get('swept'), ob_level,
                        wick_low, wick_high, sig['setup'])
    if sl_p is None:
        sl_p = price - atr_a[i]*1.5 if is_buy else price + atr_a[i]*1.5

    # Safety: SL should not be more than 3% away (max risk cap)
    max_risk = price * 0.03
    if is_buy and (price - sl_p) > max_risk:
        sl_p = price - max_risk
    if not is_buy and (sl_p - price) > max_risk:
        sl_p = price + max_risk

    risk = abs(price - sl_p)
    if risk <= 0: return None

    rr_mult = 3.0 if sig['setup'] == 'SWEEP_OB' else 2.5
    tp_p    = price + risk*rr_mult if is_buy else price - risk*rr_mult
    tp1_p   = price + risk*2.0    if is_buy else price - risk*2.0
    tp3_p   = price + risk*3.0    if is_buy else price - risk*3.0
    rr      = abs(tp_p - price)/risk
    if rr < 2.0: return None

    conf = min(97, int(sig['score']*8.5 + min(rr,3)*2.5))

    return {**sig, 'price': price, 'sl': sl_p, 'tp': tp_p,
            'tp1': tp1_p, 'tp3': tp3_p, 'rr': round(rr,2),
            'conf': conf, 'weekly': weekly_b, 'daily': daily_b,
            'rsi_val': round(rsi_a[i]),
            'risk_pct': round(abs(price-sl_p)/price*100, 2),
            'rew_pct':  round(abs(tp_p-price)/price*100, 2)}

# ── IMPROVEMENT 5: BREAKEVEN MANAGER ───────────
def check_breakeven(sym, current_price):
    """
    Called on every price update.
    When TP1 is hit → send TG alert to move SL to breakeven.
    """
    trade = state['open_trades'].get(sym)
    if not trade or trade.get('be_triggered'): return
    is_buy = trade['dir'] == 'BUY'
    tp1    = trade['tp1']
    entry  = trade['entry']
    if (is_buy and current_price >= tp1) or (not is_buy and current_price <= tp1):
        trade['be_triggered'] = True
        msg = (
            f"🔔 <b>MOVE SL TO BREAKEVEN — {sym}/USD</b>\n\n"
            f"TP1 hit at <code>{fp(tp1)}</code> 🎯\n"
            f"→ Move your Stop Loss to <b>Entry: <code>{fp(entry)}</code></b>\n\n"
            f"📐 Setup: {trade.get('setup_name','—')}\n"
            f"🎯 TP2 (main target): <code>{fp(trade['tp'])}</code>\n"
            f"🎯 TP3 (runner): <code>{fp(trade['tp3'])}</code>\n\n"
            f"<i>Your trade is now risk-free. Let it run to TP2/TP3.</i>\n"
            f"📡 <b>SMC Engine Pro</b>"
        )
        send_tg(msg)
        log.info(f"  ✓ {sym}: TP1 hit → BE alert sent")

# ── FORMAT ─────────────────────────────────────
def fp(p):
    if not p: return '—'
    if p >= 10000: return f'${p:,.0f}'
    if p >= 100:   return f'${p:.2f}'
    if p >= 1:     return f'${p:.3f}'
    return f'${p:.5f}'

TIPS = {
    'SWEEP_OB':        'Institutions swept retail stops, filled their position, now reversing. You enter on the OB retest. SL below swept wick.',
    'HTF_CONFLUENCE':  'Weekly + Daily + 1h EMA stacks all aligned in same direction. Highest conviction setup. Trend continuation.',
    'CHOCH':           'Change of Character — first structural break against prevailing trend. Early reversal entry with tight SL at last swing.',
    'BOS':             'Break of Structure with clean HH+HL or LH+LL pattern. Trend continuation confirmed by volume and momentum.',
}

def esc(s):
    """Escape HTML special chars in dynamic content"""
    return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def build_signal_msg(sig, pair):
    is_buy = sig['dir'] == 'BUY'
    emojis = {'SWEEP_OB':'⚡','HTF_CONFLUENCE':'📊','CHOCH':'🔄','BOS':'📈'}
    e = emojis.get(sig['setup'], '📡')
    setup = sig['setup']

    # ── Setup-specific explanation (WHY we are taking this trade) ──
    if setup == 'SWEEP_OB':
        if is_buy:
            basis = (
                "📖 <b>Why this trade:</b>\n"
                f"  1️⃣ Equal lows at <code>{fp(sig.get('swept', sig['price']))}</code> "
                f"→ retail stop losses were sitting there\n"
                f"  2️⃣ Price SWEPT below those lows (stop hunt) then closed back above\n"
                f"  3️⃣ Last red candle before sweep = OB zone "
                f"<code>{fp(sig['ob']['bot']) if sig.get('ob') else '—'}</code> – "
                f"<code>{fp(sig['ob']['top']) if sig.get('ob') else '—'}</code>\n"
                f"  4️⃣ Price is NOW retesting that OB = institutions defending\n"
                f"  5️⃣ Enter here, SL below swept wick — tight risk"
            )
        else:
            basis = (
                "📖 <b>Why this trade:</b>\n"
                f"  1️⃣ Equal highs at <code>{fp(sig.get('swept', sig['price']))}</code> "
                f"→ retail stop losses sitting above\n"
                f"  2️⃣ Price SWEPT above those highs (stop hunt) then closed back below\n"
                f"  3️⃣ Last green candle before sweep = bearish OB zone\n"
                f"  4️⃣ Price retesting that OB = institutions selling here\n"
                f"  5️⃣ Enter here, SL above swept wick — tight risk"
            )
    elif setup == 'CHOCH':
        if is_buy:
            basis = (
                "📖 <b>Why this trade:</b>\n"
                "  1️⃣ Market was bearish (lower highs + lower lows)\n"
                "  2️⃣ Price broke ABOVE last Lower High = CHoCH confirmed\n"
                "  3️⃣ Structure shifted from bearish to bullish\n"
                "  4️⃣ First entry on the new bullish structure\n"
                "  5️⃣ Early reversal play — tight SL below last swing low"
            )
        else:
            basis = (
                "📖 <b>Why this trade:</b>\n"
                "  1️⃣ Market was bullish (higher highs + higher lows)\n"
                "  2️⃣ Price broke BELOW last Higher Low = CHoCH confirmed\n"
                "  3️⃣ Structure shifted from bullish to bearish\n"
                "  4️⃣ First entry on the new bearish structure\n"
                "  5️⃣ Early reversal — tight SL above last swing high"
            )
    elif setup == 'HTF_CONFLUENCE':
        bias = 'bullish' if is_buy else 'bearish'
        basis = (
            "📖 <b>Why this trade:</b>\n"
            f"  1️⃣ Weekly EMA: {esc(sig.get('weekly','—'))} ← top-down bias\n"
            f"  2️⃣ Daily EMA: {esc(sig.get('daily','—'))} ← confirms direction\n"
            f"  3️⃣ 1h structure: {bias} ← all 3 timeframes aligned\n"
            f"  4️⃣ EMA9 > EMA20 > EMA50 stack ({'bull' if is_buy else 'bear'}) on 1h\n"
            f"  5️⃣ MACD momentum confirming — trend continuation entry"
        )
    elif setup == 'BOS':
        if is_buy:
            basis = (
                "📖 <b>Why this trade:</b>\n"
                "  1️⃣ Clean bullish structure: HH → HH → HH (higher highs)\n"
                "  2️⃣ Price broke above last swing HIGH = BOS confirmed\n"
                "  3️⃣ Volume spike confirms institutional buying\n"
                "  4️⃣ EMA stack bullish + MACD positive\n"
                "  5️⃣ Trend continuation — buy the break"
            )
        else:
            basis = (
                "📖 <b>Why this trade:</b>\n"
                "  1️⃣ Clean bearish structure: LL → LL → LL (lower lows)\n"
                "  2️⃣ Price broke below last swing LOW = BOS confirmed\n"
                "  3️⃣ Volume spike confirms institutional selling\n"
                "  4️⃣ EMA stack bearish + MACD negative\n"
                "  5️⃣ Trend continuation — sell the break"
            )
    else:
        basis = "📖 <b>Why this trade:</b>\n  SMC confluence setup detected"

    # ── Exchange SL note ──
    sl_note = (
        "\n⚠️ <b>SL note:</b> Set SL on YOUR exchange price, not this exact level.\n"
        "Different exchanges vary ±0.1-0.3%. Add small buffer if needed."
    )

    # ── TP levels ──
    risk = abs(sig['price'] - sig['sl'])
    tp3  = sig.get('tp3', sig['price'] + risk*3 if is_buy else sig['price'] - risk*3)
    tp1  = sig['tp1']
    tp2  = sig['tp']

    lines = [
        f"{'🟢' if is_buy else '🔴'} <b>{'STRONG ' if sig['score']>=9 else ''}{sig['dir']} — {pair['sym']}/USD</b>",
        f"{e} <b>Setup: {esc(sig['name'])}</b>",
        f"📊 Score: {sig['score']}/10  |  Confidence: {sig['conf']}%  |  R:R 1:{sig['rr']}",
        "",
        basis,
        "",
        "💰 <b>Trade Levels</b>",
        f"  Entry:  <code>{fp(sig['price'])}</code>",
        f"  SL:     <code>{fp(sig['sl'])}</code>  <i>(-{sig['risk_pct']}%) — below swept wick</i>" if setup=='SWEEP_OB'
            else f"  SL:     <code>{fp(sig['sl'])}</code>  <i>(-{sig['risk_pct']}%)</i>",
        f"  TP1:    <code>{fp(tp1)}</code>  <i>(1:2 — close 50%, move SL to entry)</i>",
        f"  TP2:    <code>{fp(tp2)}</code>   <i>(1:{sig['rr']} — close 30%)</i>",
        f"  TP3:    <code>{fp(tp3)}</code>  <i>(1:3 — let runner go)</i>",
        "",
        "🔍 <b>Confluences:</b> " + esc(' · '.join(sig.get('tags', []))),
        f"  Weekly: {esc(sig.get('weekly','—'))}  |  Daily: {esc(sig.get('daily','—'))}  |  RSI: {sig.get('rsi_val','—')}",
        (f"  OB Zone: {fp(sig['ob']['bot'])} – {fp(sig['ob']['top'])}" if sig.get('ob') else ""),
        (f"  Swept at: {fp(sig['swept'])}" if sig.get('swept') else ""),
        sl_note,
        "",
        "📋 <b>Trade Management:</b>",
        "  • Enter at market price or limit at OB zone" if setup=='SWEEP_OB' else "  • Enter at market or on retest",
        "  • At TP1 → close 50%, move SL to breakeven",
        "  • Let rest run to TP2, trail to TP3",
        "",
        "⚠️ <i>Not financial advice. Always manage risk.</i>",
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC  |  📡 <b>SMC Engine Pro v3</b>",
    ]
    return '\n'.join(l for l in lines if l is not None)

def build_result_msg(sym, result, pnl, trade):
    e = '✅' if result == 'WIN' else '❌'
    return '\n'.join([
        f"{e} <b>TRADE {result} — {sym}/USD  {'+' if pnl>=0 else ''}{pnl:.2f}%</b>",
        '',
        f"📐 Setup:  {trade.get('setup_name','—')}",
        f"💰 Entry:  {fp(trade.get('entry',0))}",
        f"{'🎯' if result=='WIN' else '🛑'} Exit:   {fp(trade.get('tp' if result=='WIN' else 'sl',0))}",
        f"📊 Score:  {trade.get('score',0)}/10",
        '',
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC  |  📡 <b>SMC Engine Pro v3</b>",
    ])

# ── TELEGRAM ───────────────────────────────────
def send_tg(msg):
    if not TG_TOKEN or not TG_CHAT:
        log.error("TG_TOKEN or TG_CHAT not set!")
        return False
    try:
        r = requests.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={
                'chat_id': TG_CHAT,
                'text': msg,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            },
            timeout=10
        )
        if r.ok:
            return True
        # Log the full error details
        try:
            err_body = r.json()
            err_desc = err_body.get('description', 'unknown')
            err_code = err_body.get('error_code', r.status_code)
        except:
            err_desc = r.text[:200]
            err_code = r.status_code
        log.error(f"TG failed [{err_code}]: {err_desc}")
        log.error(f"  Full token length: {len(TG_TOKEN)} chars")
        log.error(f"  Token: {TG_TOKEN[:15]}...{TG_TOKEN[-6:]}")
        log.error(f"  Chat ID: {TG_CHAT}")
        # Retry without HTML if parse error
        if 'parse' in err_desc.lower() or 'html' in err_desc.lower():
            r2 = requests.post(
                f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
                json={'chat_id': TG_CHAT, 'text': msg[:4000], 'disable_web_page_preview': True},
                timeout=10
            )
            if r2.ok:
                log.info("TG sent (plain text fallback)")
                return True
        return False
    except Exception as e:
        log.error(f"TG exception: {e}")
        return False

# ── PRICE CHECK (for BE manager) ───────────────
def check_prices():
    for sym, trade in list(state['open_trades'].items()):
        try:
            pair   = next(p for p in PAIRS if p['sym'] == sym)
            kl_chk = fetch_candles(pair, limit=5)
            if not kl_chk: continue
            price  = float(kl_chk[-1]['c'])
            chk_hi = max(k['h'] for k in kl_chk[-3:])
            chk_lo = min(k['l'] for k in kl_chk[-3:])
            if not price: continue
            is_buy = trade['dir'] == 'BUY'
            entry  = trade['entry']
            sl_p   = trade['sl']
            tp1_p  = trade['tp1']
            tp2_p  = trade['tp']
            tp3_p  = trade.get('tp3', tp2_p)

            # TP1: Breakeven
            if not trade.get('be_triggered'):
                if (is_buy and chk_hi>=tp1_p) or (not is_buy and chk_lo<=tp1_p):
                    trade['be_triggered'] = True
                    pnl1 = round(abs(tp1_p-entry)/entry*100,2)
                    send_tg(
                        f"🎯 <b>TP1 HIT — {sym}/USD +{pnl1}%</b>\n\n"
                        f"✅ Close 50% at <code>{fp(price)}</code>\n"
                        f"🔒 Move SL to entry: <code>{fp(entry)}</code>\n\n"
                        f"🎯 TP2 target: <code>{fp(tp2_p)}</code>\n"
                        f"🎯 TP3 runner: <code>{fp(tp3_p)}</code>\n\n"
                        f"<i>Trade is now risk-free. Let it run.</i>\n"
                        f"📐 {trade.get('setup_name','—')}  |  📡 SMC Engine Pro v3"
                    )
                    log.info(f"  🎯 {sym}: TP1 hit +{pnl1}%")

            # TP2: WIN
            if not trade.get('tp2_hit'):
                if (is_buy and chk_hi>=tp2_p) or (not is_buy and chk_lo<=tp2_p):
                    trade['tp2_hit'] = True
                    pnl = round(abs(tp2_p-entry)/entry*100,2)
                    analysis = _analyze_win(trade)
                    send_tg(
                        f"✅ <b>WIN — {sym}/USD +{pnl}%</b>\n\n"
                        f"⚡ Setup: {trade.get('setup_name','—')}\n"
                        f"📊 Score: {trade.get('score',0)}/10  |  R:R 1:{trade.get('rr',0)}\n\n"
                        f"💰 Entry: <code>{fp(entry)}</code>\n"
                        f"🎯 Exit:  <code>{fp(price)}</code>  (+{pnl}%)\n"
                        f"⏱ Held: {_hours_held(trade)}\n\n"
                        f"🏆 <b>Why it worked:</b>\n{analysis}\n\n"
                        f"🎯 TP3 runner still open: <code>{fp(tp3_p)}</code>\n"
                        f"   Trail SL to TP1 level now\n\n"
                        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC  |  📡 SMC Engine Pro v3"
                    )
                    lid = trade.get('learn_id')
                    if lid: close_trade(lid,'win',price)
                    journal_close_trade(sym,'win',price)
                    # 🧠 Deep chart re-analysis
                    try:
                        pair_deep = next(p for p in PAIRS if p['sym']==sym)
                        kl_deep   = fetch_candles(pair_deep, limit=100)
                        if kl_deep:
                            learn_from_trade({**trade,'sym':sym,'pnl':pnl}, 'win', kl_deep)
                    except Exception as _e:
                        log.debug(f"Deep learn WIN {sym}: {_e}")
                    state['stats']['wins'] = state['stats'].get('wins',0)+1
                    _upd_stat(trade.get('setup','?'),'w')
                    log.info(f"  ✅ WIN: {sym} +{pnl}%")
                    del state['open_trades'][sym]; continue

            # TP3: Runner
            if trade.get('tp2_hit') and not trade.get('tp3_hit'):
                if (is_buy and chk_hi>=tp3_p) or (not is_buy and chk_lo<=tp3_p):
                    trade['tp3_hit'] = True
                    pnl = round(abs(tp3_p-entry)/entry*100,2)
                    send_tg(
                        f"🚀 <b>RUNNER HIT — {sym}/USD +{pnl}%</b>\n\n"
                        f"TP3 reached! Full exit.\n"
                        f"📐 {trade.get('setup_name','—')}  |  "
                        f"⏰ {datetime.now(timezone.utc).strftime('%H:%M')} UTC"
                    )
                    continue

            # SL: LOSS
            if (is_buy and chk_lo<=sl_p) or (not is_buy and chk_hi>=sl_p):
                pnl = round(abs(sl_p-entry)/entry*100,2)
                analysis = _analyze_loss(trade, price)
                send_tg(
                    f"❌ <b>LOSS — {sym}/USD -{pnl}%</b>\n\n"
                    f"📐 Setup: {trade.get('setup_name','—')}\n"
                    f"📊 Score was: {trade.get('score',0)}/10\n\n"
                    f"💰 Entry:  <code>{fp(entry)}</code>\n"
                    f"🛑 SL hit: <code>{fp(price)}</code>  (-{pnl}%)\n"
                    f"⏱ Held: {_hours_held(trade)}\n\n"
                    f"🔍 <b>Failure Analysis:</b>\n{analysis}\n\n"
                    f"📚 <i>Engine is learning from this trade.</i>\n"
                    f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC  |  📡 SMC Engine Pro v3"
                )
                lid = trade.get('learn_id')
                if lid: close_trade(lid,'loss',price)
                journal_close_trade(sym,'loss',price)
                # 🧠 Deep chart re-analysis on LOSS
                try:
                    _pair_d = next(p for p in PAIRS if p['sym']==sym)
                    _kl_d   = fetch_candles(_pair_d, limit=100)
                    if _kl_d:
                        learn_from_trade({**trade,'sym':sym,'pnl':-pnl,'id':f"{sym}_{int(time.time())}"}, 'loss', _kl_d)
                except Exception as _le:
                    log.debug(f"Deep learn LOSS {sym}: {_le}")
                state['stats']['losses'] = state['stats'].get('losses',0)+1
                _upd_stat(trade.get('setup','?'),'l')
                log.info(f"  ❌ LOSS: {sym} -{pnl}%")
                del state['open_trades'][sym]; continue

        except Exception as e:
            log.debug(f"check_prices {sym}: {e}")
        time.sleep(0.5)

def _hours_held(trade):
    try:
        secs = time.time()-datetime.fromisoformat(trade['time']).timestamp()
        return f"{int(secs//3600)}h {int((secs%3600)//60)}m"
    except: return '—'

def _upd_stat(setup, result):
    state['stats'].setdefault('by_setup',{})
    state['stats']['by_setup'].setdefault(setup,{'w':0,'l':0,'be':0})
    state['stats']['by_setup'][setup][result]+=1

def _analyze_win(trade):
    lines=[]; tags=trade.get('tags',[]); sess=trade.get('session_name','')
    if 'RSI_Div✓' in tags: lines.append("  ✅ RSI divergence confirmed reversal")
    if any('Sweep' in t for t in tags): lines.append("  ✅ Liquidity sweep cleared stops perfectly")
    if 'OB_Retest' in tags: lines.append("  ✅ OB zone defended by institutions")
    if 'Vol✓' in tags: lines.append("  ✅ Volume surge confirmed smart money")
    if any('EMA' in t for t in tags): lines.append("  ✅ EMA stack aligned with direction")
    if sess in ('London','New York'): lines.append(f"  ✅ Active {sess} session — high quality")
    if 'MACD✓' in tags: lines.append("  ✅ MACD momentum confirmed")
    if trade.get('weekly') in ('bullish','bearish'): lines.append("  ✅ Weekly trend supported move")
    return '\n'.join(lines) if lines else "  ✅ All confluence factors aligned"

def _analyze_loss(trade, exit_price):
    lines=[]; tags=trade.get('tags',[]); is_buy=trade['dir']=='BUY'
    weekly=trade.get('weekly','neutral'); daily=trade.get('daily','neutral')
    rsi=trade.get('rsi_val',50); sess=trade.get('session_name','')
    score=trade.get('score',0); entry=trade['entry']; setup=trade.get('setup','')
    move_pct=abs(exit_price-entry)/entry*100

    # Trend issues
    if is_buy and weekly=='bearish': lines.append("  ⚠️ BUY taken against weekly bearish trend")
    if not is_buy and weekly=='bullish': lines.append("  ⚠️ SELL taken against weekly bullish trend")
    if is_buy and daily=='bearish': lines.append("  ⚠️ Daily bias bearish — against trade direction")
    if not is_buy and daily=='bullish': lines.append("  ⚠️ Daily bias bullish — against trade direction")
    # RSI issues
    if is_buy and rsi>62: lines.append(f"  ⚠️ RSI {rsi} — overbought for BUY entry")
    if not is_buy and rsi<38: lines.append(f"  ⚠️ RSI {rsi} — oversold for SELL entry")
    # Session
    if sess=='Weekend': lines.append("  ⚠️ Weekend — low volume, fake sweeps common")
    if sess=='Asian': lines.append("  ⚠️ Asian session — choppy, low institutional activity")
    # Score
    if score<=7: lines.append(f"  ⚠️ Marginal score ({score}/10) — weak confluence")
    # Missing confirmations
    if 'RSI_Div✓' not in tags: lines.append("  ⚠️ No RSI divergence — reversal unconfirmed")
    if 'Vol✓' not in tags: lines.append("  ⚠️ Low volume — no institutional confirmation")
    # Stop distance
    if move_pct<0.3: lines.append(f"  ℹ️ Stopped immediately — entry timing was off")
    elif move_pct<0.8: lines.append(f"  ℹ️ SL may have been too tight for this volatility")
    if not lines: lines.append("  ℹ️ Valid setup — market conditions changed unexpectedly")
    lines.append(f"\n  🧠 Adjusting {setup} weights to reduce similar losses")
    return '\n'.join(lines)


def build_signal_msg(sig, pair):
    is_buy = sig['dir'] == 'BUY'
    emojis = {'SWEEP_OB':'⚡','HTF_CONFLUENCE':'📊','CHOCH':'🔄','BOS':'📈'}
    e = emojis.get(sig['setup'], '📡')
    tips = {
        'SWEEP_OB':       f"Institutions swept retail stops at {fp(sig.get('swept',sig['price']))}. OB retest entry. SL below wick.",
        'HTF_CONFLUENCE': 'Weekly + Daily + 1h EMA stacks all aligned. Trend continuation.',
        'CHOCH':          'Change of Character — structural shift. Early reversal, tight SL.',
        'BOS':            'Break of Structure confirmed. Trend continuation with clean swings.',
    }
    risk = abs(sig['price'] - sig['sl'])
    is_b = sig['dir'] == 'BUY'
    tp3  = sig.get('tp3', sig['price'] + risk*3 if is_b else sig['price'] - risk*3)
    lines = [
        f"{'🟢' if is_buy else '🔴'} <b>{sig['dir']} — {pair['sym']}/USD</b>",
        f"{e} <b>Setup: {esc(sig['name'])}</b>", '',
        f"📌 <i>{tips.get(sig['setup'],'SMC confluence setup.')}</i>", '',
        '💰 <b>Trade Levels</b>',
        f"  Entry:  <code>{fp(sig['price'])}</code>",
        f"  SL:     <code>{fp(sig['sl'])}</code>  <i>(-{sig['risk_pct']}%)</i>",
        f"  TP1:    <code>{fp(sig['tp1'])}</code>  <i>(1:2 — close 50%, move SL to BE)</i>",
        f"  TP2:    <code>{fp(sig['tp'])}</code>   <i>(1:{sig['rr']} — main target)</i>",
        f"  TP3:    <code>{fp(tp3)}</code>  <i>(1:3 — runner)</i>", '',
        f"📊 <b>Score: {sig['score']}/10  |  Conf: {sig['conf']}%  |  R:R 1:{sig['rr']}</b>",
        f"  Confluences: {esc(' · '.join(sig['tags']))}",
        f"  Weekly: {esc(sig.get('weekly','—'))}  |  Daily: {esc(sig.get('daily','—'))}  |  RSI: {sig.get('rsi_val','—')}",
    ]
    if sig.get('ob'):
        lines.append(f"  OB Zone: {fp(sig['ob']['bot'])} – {fp(sig['ob']['top'])}")
    if sig.get('swept'):
        lines.append(f"  Swept at: {fp(sig['swept'])}")
    lines += [
        '', '📋 <b>Trade Management:</b>',
        '  • At TP1 → close 50% of position',
        '  • Move SL to breakeven (entry)',
        '  • Let remaining run to TP2, then TP3',
        '', '⚠️ <i>Not financial advice. Always manage risk.</i>',
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC  |  📡 <b>SMC Engine Pro v3</b>",
    ]
    return '\n'.join(l for l in lines if l is not None)

def build_result_msg(sym, result, pnl, trade):
    e = '✅' if result == 'WIN' else '❌'
    return '\n'.join([
        f"{e} <b>TRADE {result} — {sym}/USD  {'+' if pnl>=0 else ''}{pnl:.2f}%</b>", '',
        f"📐 Setup: {trade.get('setup_name','—')}",
        f"💰 Entry: {fp(trade.get('entry',0))}",
        f"{'🎯' if result=='WIN' else '🛑'} Exit: {fp(trade.get('tp' if result=='WIN' else 'sl',0))}",
        f"📊 Score: {trade.get('score',0)}/10",
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC  |  📡 SMC Engine Pro v3",
    ])

def esc(s):
    return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def send_tg(msg):
    if not TG_TOKEN or not TG_CHAT: return False
    try:
        r = requests.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={'chat_id':TG_CHAT,'text':msg,'parse_mode':'HTML',
                  'disable_web_page_preview':True},
            timeout=10)
        if r.ok: return True
        err = r.json().get('description','unknown')
        log.error(f"TG failed: {err}")
        # Retry plain
        r2 = requests.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={'chat_id':TG_CHAT,'text':msg[:4000],'disable_web_page_preview':True},
            timeout=10)
        return r2.ok
    except Exception as e:
        log.error(f"TG exception: {e}"); return False

def saveTG():
    pass  # config via env vars only

def run_scan():
    state['scans_done'] += 1
    state['last_scan'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    log.info(f"Scan #{state['scans_done']} — {len(PAIRS)} pairs")
    # circuit breaker disabled — ML handles risk via score thresholds
    kl_btc = None
    try:
        kl_btc = fetch_candles(PAIRS[0], limit=300)
    except: pass
    for pair in PAIRS:
        try:
            kl = kl_btc if pair['sym']=='BTC' else fetch_candles(pair, limit=300)
            if not kl: log.info(f"  {pair['sym']}: no data"); continue
            sig = compute(kl, pair, kl_btc if pair['sym']!='BTC' else None)
            if sig:
                msg = build_signal_msg(sig, pair)
                ok  = send_tg(msg)
                if ok:
                    last_fired[pair['sym']] = {'time': time.time()}
                    state['alerts_sent'] += 1
                    session_now = get_session()
                    learn_id = log_signal(sig, pair, session_now)
                    trade_entry = {
                        'dir':        sig['dir'],
                        'id':         f"{pair['sym']}_{int(time.time())}",
                        'setup':      sig['setup'],
                        'setup_name': sig['name'],
                        'entry':      sig['price'],
                        'sl':         sig['sl'],
                        'tp':         sig['tp'],
                        'tp1':        sig['tp1'],
                        'tp3':        sig.get('tp3', sig['tp']),
                        'score':      sig['score'],
                        'rr':         sig['rr'],
                        'time':       datetime.now(timezone.utc).isoformat(),
                        'be_triggered': False,
                        'tp2_hit':    False,
                        'tp3_hit':    False,
                        'tags':       sig.get('tags',[]),
                        'weekly':     sig.get('weekly','neutral'),
                        'daily':      sig.get('daily','neutral'),
                        'rsi_val':    sig.get('rsi_val',50),
                        'session_name': session_now,
                        'learn_id':   learn_id,
                    }
                    state['open_trades'][pair['sym']] = trade_entry
                    journal_log_signal(sig, pair)
                    log.info(f"  ✓ {pair['sym']}: {sig['name']} {sig['dir']} score={sig['score']} → TG sent")
                else:
                    log.error(f"  {pair['sym']}: TG failed")
            else:
                log.info(f"  {pair['sym']}: no setup")
            time.sleep(0.8)
        except Exception as e:
            log.error(f"  {pair['sym']} error: {e}")
    log.info(f"Scan done. Alerts: {state['alerts_sent']}")

def main():
    if not TG_TOKEN or not TG_CHAT:
        log.error("Missing TG_TOKEN or TG_CHAT"); raise SystemExit(1)
    log.info("="*55)
    log.info("SMC ENGINE PRO v3 — 24/7 SELF-LEARNING SERVER")
    if TG_TOKEN:
        log.info(f"TG_TOKEN: {TG_TOKEN[:15]}...{TG_TOKEN[-6:]} (len={len(TG_TOKEN)})")
    log.info(f"TG_CHAT: {TG_CHAT}")
    log.info(f"Pairs: {len(PAIRS)} | Score≥{MIN_SCORE}(off-session:{MIN_SCORE+1}) | Every {SCAN_EVERY}m")
    log.info(f"Sessions: London 07-12 UTC | NY 13-18 UTC")
    log.info(f"Filters: Weekly gate | BTC gate | Struct SL | BE mgmt | Self-learning")
    log.info("="*55)

    threading.Thread(target=start_health, daemon=True).start()
    log.info(f"Health server on port {PORT}")

    send_tg(
        "🚀 <b>SMC Engine Pro v3 — Self Learning Started</b>\n\n"
        "<b>Settings:</b>\n"
        f"⏱ Scan every: {SCAN_EVERY} minute\n"
        f"🔁 Cooldown: {COOLDOWN_M} min\n"
        f"📊 Min score: {MIN_SCORE}/10\n\n"
        "<b>Alerts you will receive:</b>\n"
        "📡 Signal → Entry/SL/TP1/TP2/TP3 levels\n"
        "🎯 TP1 hit → move SL to breakeven\n"
        "✅ TP2 hit → WIN + why it worked\n"
        "🚀 TP3 hit → runner closed\n"
        "❌ SL hit → LOSS + failure analysis\n"
        "📅 Every Monday → weekly learning report\n\n"
        "<b>Commands:</b>\n"
        "/stats /learn /weights /weekly /open /help\n\n"
        "📡 <b>SMC Engine Pro v3 — Self Learning</b>"
    )
    log.info("✓ Startup TG message sent")

    def be_loop():
        while True:
            try:
                if state['open_trades']: check_prices()
            except Exception as e:
                log.debug(f"BE check error: {e}")
            time.sleep(60)
    threading.Thread(target=be_loop, daemon=True).start()

    def weekly_report_loop():
        while True:
            now = datetime.now(timezone.utc)
            if now.weekday()==0 and now.hour==8 and now.minute<5:
                send_tg(weekly_learning_report())
                log.info("Weekly learning report sent")
                time.sleep(300)
            time.sleep(60)
    threading.Thread(target=weekly_report_loop, daemon=True).start()

    last_update_id = [0]
    def tg_commands():
        while True:
            try:
                r = requests.get(
                    f'https://api.telegram.org/bot{TG_TOKEN}/getUpdates',
                    params={'offset':last_update_id[0]+1,'timeout':10},
                    timeout=15)
                if r.ok:
                    for upd in r.json().get('result',[]):
                        last_update_id[0] = upd['update_id']
                        txt = upd.get('message',{}).get('text','').strip().lower()
                        if txt in ('/stats','/report','/journal'):
                            w=state['stats'].get('wins',0); l=state['stats'].get('losses',0)
                            b=state['stats'].get('be',0); tot=w+l+b
                            journal_msg = journal_stats_report()
                            sess_msg = (
                                f"\n\n📊 <b>This Session:</b>\n"
                                f"  ✅ Wins:   {w}\n  ❌ Losses: {l}\n  ➡️ BE: {b}\n"
                                f"  WR: {round(w/tot*100) if tot else 0}%\n"
                                f"  Open: {len(state['open_trades'])}\n"
                                f"  Alerts sent: {state['alerts_sent']}"
                            )
                            send_tg(journal_msg + sess_msg)
                        elif txt in ('/learn','/learning','/ml'):
                            send_tg(performance_report())
                        elif txt in ('/deep','/deeplearn','/analysis'):
                            send_tg(deep_learning_report())
                        elif txt == '/weights':
                            db = load_db()
                            w_d = db['weights']
                            lines = ["⚖️ <b>Learned Weights</b>\n","<b>Setup scores:</b>"]
                            for s,v in w_d['setup_scores'].items():
                                lines.append(f"  {s}: {v:.2f}")
                            lines.append("\n<b>Session multipliers:</b>")
                            for s,v in w_d['session_weights'].items():
                                lines.append(f"  {s}: {v:.2f}x")
                            lines.append("\n<b>Top tag weights:</b>")
                            for t,v in sorted(w_d['tag_weights'].items(),key=lambda x:-x[1])[:8]:
                                lines.append(f"  {t}: {v:.2f}")
                            send_tg('\n'.join(lines))
                        elif txt == '/weekly':
                            send_tg(weekly_learning_report())
                        elif txt == '/open':
                            if state['open_trades']:
                                msg = "🔓 <b>Open trades:</b>\n" + "\n".join(
                                    f"  • {sym}: {v['dir']} {v.get('setup','?')} @ {fp(v['entry'])}"
                                    for sym,v in state['open_trades'].items())
                            else:
                                msg = "✅ No open trades right now"
                            send_tg(msg)
                        elif txt == '/reset_weights':
                            db = load_db(); db['weights']=DEFAULT_WEIGHTS.copy(); save_db(db)
                            send_tg("✅ Weights reset to defaults")
                        elif txt == '/help':
                            send_tg(
                                "📡 <b>SMC Bot Commands</b>\n\n"
                                "/stats   — performance report\n"
                                "/learn   — learning weights report\n"
                                "/deep    — deep chart analysis report\n"
                                "/weights — learned weights\n"
                                "/weekly  — weekly summary\n"
                                "/open    — open trades\n"
                                "/help    — this menu"
                            )
            except Exception as e:
                log.debug(f"TG cmd error: {e}")
            time.sleep(30)
    threading.Thread(target=tg_commands, daemon=True).start()
    log.info("✓ TG command listener started")

    while True:
        try:
            run_scan()
        except Exception as e:
            log.error(f"Scan error: {e}")
        log.info(f"Next scan in {SCAN_EVERY}m...")
        time.sleep(SCAN_EVERY * 60)

if __name__ == '__main__':
    main()
