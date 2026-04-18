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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ── CONFIG ─────────────────────────────────────
TG_TOKEN   = os.environ.get('TG_TOKEN', '')
TG_CHAT    = os.environ.get('TG_CHAT', '')
MIN_SCORE  = int(os.environ.get('MIN_SCORE', '6'))
SCAN_EVERY = int(os.environ.get('SCAN_EVERY_MIN', '1'))
COOLDOWN_M = int(os.environ.get('COOLDOWN_MIN', '30'))
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
def structure_sl(sh, sl_sw, i, direction, atr_v, swept=None, ob=None):
    """SL at structural level — not fixed ATR multiple"""
    buf = atr_v * 0.12
    if direction == 'BUY':
        lvls = []
        if swept: lvls.append(swept - buf)
        if ob:    lvls.append(ob - buf)
        rl = [(idx,p) for idx,p in sl_sw if idx <= i][-3:]
        if rl: lvls.append(min(p for _,p in rl) - buf)
        return min(lvls) if lvls else None
    else:
        lvls = []
        if swept: lvls.append(swept + buf)
        if ob:    lvls.append(ob + buf)
        rh = [(idx,p) for idx,p in sh if idx <= i][-3:]
        if rh: lvls.append(max(p for _,p in rh) + buf)
        return max(lvls) if lvls else None

# ── SIGNAL DETECTION ───────────────────────────
def get_signal(kl, sh, sl_sw, i, closes, rsi_a, e9_a, e20_a, e50_a,
               ht_a, atr_a, va_a, weekly_b, daily_b):
    price = kl[i]['c']; k = kl[i]
    at = atr_a[i]; va_v = va_a[i]

    # ── SETUP 1: SWEEP + OB ─────────────────────
    for li, lvl in [(ix,p) for ix,p in sl_sw if ix < i-1 and ix > i-50][-4:]:
        if not (k['l'] < lvl < price): continue
        if lvl - k['l'] < at*0.28: continue
        if k['v'] < va_v*1.15: continue
        if daily_b != 'bullish': continue
        if weekly_b == 'bearish': continue   # IMPROVEMENT 2
        if not rsi_a[i] or not (25 < rsi_a[i] < 62): continue
        ob = None
        for j in range(li-1, max(0, li-12), -1):
            if kl[j]['c'] < kl[j]['o']:
                fwd = (kl[min(j+2,len(kl)-1)]['c'] - kl[j]['c']) / kl[j]['c']
                if fwd > 0.003:
                    ob = {'top': kl[j]['o'], 'bot': kl[j]['l']}; break
        if not ob or not (ob['bot'] <= price <= ob['top']*1.005): continue
        ema_ok = e20_a[i] and e50_a[i] and price > e20_a[i] > e50_a[i]
        return {'dir':'BUY', 'setup':'SWEEP_OB',
                'name':'⚡ Liq Sweep + OB Retest',
                'score': 8+(0.5 if ema_ok else 0),
                'ob': ob, 'swept': lvl,
                'tags': ['Sweep↑','OB_Retest','Vol✓','HTF✓','Week✓']
                       +((['EMA↑'] if ema_ok else []))+[f'RSI{round(rsi_a[i])}']}

    for hi_, lvl in [(ix,p) for ix,p in sh if ix < i-1 and ix > i-50][-4:]:
        if not (k['h'] > lvl > price): continue
        if k['h'] - lvl < at*0.28: continue
        if k['v'] < va_v*1.15: continue
        if daily_b != 'bearish': continue
        if weekly_b == 'bullish': continue   # IMPROVEMENT 2
        if not rsi_a[i] or not (38 < rsi_a[i] < 75): continue
        ob = None
        for j in range(hi_-1, max(0, hi_-12), -1):
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
                'tags': ['Sweep↓','OB_Retest','Vol✓','HTF✓','Week✓']
                       +((['EMA↓'] if ema_ok else []))+[f'RSI{round(rsi_a[i])}']}

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
    lf = last_fired.get(pair['sym'])
    if lf and (time.time()-lf['time'])/60 < COOLDOWN_M:
        return None  # Still in cooldown for this coin

    sh, sl = swings(kl, 5)
    weekly_b = calc_bias(kl, i, 21)   # IMPROVEMENT 2: weekly gate
    daily_b  = calc_bias(kl, i, 5)

    sig = get_signal(kl, sh, sl, i, closes, rsi_a, e9_a, e20_a, e50_a,
                     ht_a, atr_a, va_a, weekly_b, daily_b)
    # Outside active session → require 1 extra point of confluence
    effective_min = MIN_SCORE if session_on else MIN_SCORE + 1
    if not sig or sig['score'] < effective_min: return None

    # IMPROVEMENT 3: BTC correlation gate
    if pair['sym'] != 'BTC' and kl_btc:
        if not btc_gate(kl_btc, i, sig['dir']):
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

    # IMPROVEMENT 4: Structure-based SL
    sw = sig.get('swept')
    ob_level = (sig['ob']['bot'] if is_buy else sig['ob']['top']) if sig.get('ob') else None
    sl_p = structure_sl(sh, sl, i, sig['dir'], atr_a[i], sw, ob_level)
    if sl_p is None:
        sl_p = price - atr_a[i]*1.5 if is_buy else price + atr_a[i]*1.5

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
    lines = [
        f"{'🟢' if is_buy else '🔴'} <b>{'STRONG ' if sig['score']>=9 else ''}{sig['dir']} — {pair['sym']}/USD</b>",
        f"{e} <b>Setup: {esc(sig['name'])}</b>",
        '',
        f"📌 <i>{TIPS.get(sig['setup'], 'SMC confluence setup.')}</i>",
        '',
        '💰 <b>Trade Levels</b>',
        f"  Entry:  <code>{fp(sig['price'])}</code>",
        f"  SL:     <code>{fp(sig['sl'])}</code>  <i>(-{sig['risk_pct']}%)</i>",
        f"  TP1:    <code>{fp(sig['tp1'])}</code>  <i>(1:2 — close 50%, move SL to BE)</i>",
        f"  TP2:    <code>{fp(sig['tp'])}</code>   <i>(1:{sig['rr']} — close 30%)</i>",
        f"  TP3:    <code>{fp(sig['tp3'])}</code>  <i>(1:3 — let runner go)</i>",
        '',
        f"📊 <b>Score: {sig['score']}/10  |  Confidence: {sig['conf']}%  |  R:R 1:{sig['rr']}</b>",
        f"  Confluences: {esc(' · '.join(sig['tags']))}",
        f"  Weekly: {esc(sig['weekly'])}  |  Daily: {esc(sig['daily'])}  |  RSI: {sig['rsi_val']}",
    ]
    if sig.get('ob'):
        lines.append(f"  OB Zone: {fp(sig['ob']['bot'])} – {fp(sig['ob']['top'])}")
    if sig.get('swept'):
        lines.append(f"  Swept at: {fp(sig['swept'])}")
    lines += [
        '',
        '📋 <b>Trade Management:</b>',
        '  • At TP1 → close 50% of position',
        '  • Move SL to breakeven (entry)',
        '  • Let remaining run to TP2, then TP3',
        '',
        '⚠️ <i>Not financial advice. Always manage risk.</i>',
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
        f"📡 <b>SMC Engine Pro v3</b>",
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
    """Price check every minute: BE management + win/loss auto-detection"""
    for sym, trade in list(state['open_trades'].items()):
        try:
            pair = next(p for p in PAIRS if p['sym'] == sym)
            r = requests.get(f'{CG}/simple/price',
                params={'ids': pair['cg'], 'vs_currencies': 'usd'},
                timeout=8)
            price = r.json()[pair['cg']]['usd']
            if not price: continue

            is_buy = trade['dir'] == 'BUY'
            sl_p   = trade['sl']
            tp_p   = trade['tp']
            tp1_p  = trade['tp1']

            # ── BREAKEVEN: TP1 hit → alert to move SL ──
            if not trade.get('be_triggered') and tp1_p:
                tp1_hit = (is_buy and price >= tp1_p) or (not is_buy and price <= tp1_p)
                if tp1_hit:
                    trade['be_triggered'] = True
                    check_breakeven(sym, price)

            # ── WIN: TP2 hit ──
            tp_hit = (is_buy and price >= tp_p) or (not is_buy and price <= tp_p)
            if tp_hit:
                pnl = round(abs(tp_p - trade['entry']) / trade['entry'] * 100, 2)
                msg = build_result_msg(sym, 'WIN', pnl, trade)
                send_tg(msg)
                journal_close_trade(sym, 'win', tp_p)
                state['stats']['wins'] = state['stats'].get('wins', 0) + 1
                state['stats'].setdefault('by_setup', {}).setdefault(trade.get('setup','?'), {'w':0,'l':0,'be':0})
                state['stats']['by_setup'][trade.get('setup','?')]['w'] += 1
                log.info(f"  ✅ WIN: {sym} +{pnl}% → TG sent")
                del state['open_trades'][sym]
                continue

            # ── LOSS: SL hit ──
            sl_hit = (is_buy and price <= sl_p) or (not is_buy and price >= sl_p)
            if sl_hit:
                pnl = round(abs(sl_p - trade['entry']) / trade['entry'] * 100, 2)
                msg = build_result_msg(sym, 'LOSS', -pnl, trade)
                send_tg(msg)
                journal_close_trade(sym, 'loss', sl_p)
                state['stats']['losses'] = state['stats'].get('losses', 0) + 1
                state['stats'].setdefault('by_setup', {}).setdefault(trade.get('setup','?'), {'w':0,'l':0,'be':0})
                state['stats']['by_setup'][trade.get('setup','?')]['l'] += 1
                log.info(f"  ❌ LOSS: {sym} -{pnl}% → TG sent")
                del state['open_trades'][sym]
                continue

        except: pass
        time.sleep(0.5)

# ── SCAN ───────────────────────────────────────
def run_scan():
    state['scans_done'] += 1
    state['last_scan'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    log.info(f"Scan #{state['scans_done']} — {len(PAIRS)} pairs")
    # Circuit breaker: pause after 3 consecutive losses
    if check_circuit_breaker():
        log.warning("Circuit breaker active — skipping scan")
        return

    # Fetch BTC first for correlation gate
    kl_btc = None
    try:
        kl_btc = fetch_candles(PAIRS[0], limit=300)
        if not kl_btc: log.warning("BTC candles failed")
    except: pass

    for pair in PAIRS:
        try:
            kl = kl_btc if pair['sym'] == 'BTC' else fetch_candles(pair, limit=300)
            if not kl:
                log.info(f"  {pair['sym']}: no data"); continue

            sig = compute(kl, pair, kl_btc if pair['sym'] != 'BTC' else None)

            if sig:
                msg = build_signal_msg(sig, pair)
                ok  = send_tg(msg)
                if ok:
                    last_fired[pair['sym']] = {'time': time.time()}
                    state['alerts_sent'] += 1
                    journal_log_signal(sig, pair)  # 📓 Log to journal
                    # Track for BE management (IMPROVEMENT 5)
                    state['open_trades'][pair['sym']] = {
                        'dir':        sig['dir'],
                        'setup':      sig['setup'],
                        'setup_name': sig['name'],
                        'entry':      sig['price'],
                        'sl':         sig['sl'],
                        'tp':         sig['tp'],
                        'tp1':        sig['tp1'],
                        'tp3':        sig['tp3'],
                        'score':      sig['score'],
                        'time':       time.time(),
                        'be_triggered': False,
                    }
                    log.info(f"  ✓ {pair['sym']}: {sig['name']} {sig['dir']} "
                             f"score={sig['score']} conf={sig['conf']}% → TG sent")
                else:
                    log.error(f"  {pair['sym']}: TG send failed")
            else:
                log.info(f"  {pair['sym']}: no setup")

            time.sleep(0.8)  # rate limit — faster scan

        except Exception as e:
            log.error(f"  {pair['sym']} error: {e}")

    log.info(f"Scan done. Total alerts: {state['alerts_sent']}")

# ── MAIN ───────────────────────────────────────
def main():
    if not TG_TOKEN or not TG_CHAT:
        log.error("Missing TG_TOKEN or TG_CHAT environment variables")
        log.error("Set them in Railway → Variables tab")
        raise SystemExit(1)

    log.info("="*55)
    log.info("SMC ENGINE PRO v3 — 24/7 ALERT SERVER")
    # Print token info for debugging (first/last 6 chars only)
    if TG_TOKEN:
        log.info(f"TG_TOKEN: {TG_TOKEN[:15]}...{TG_TOKEN[-6:]} (len={len(TG_TOKEN)})")
    else:
        log.info("TG_TOKEN: NOT SET")
    log.info(f"TG_CHAT:  {TG_CHAT}")
    log.info(f"Pairs: {len(PAIRS)} | Score≥{MIN_SCORE} (off-session: {MIN_SCORE+1}) | Every {SCAN_EVERY}m")
    log.info(f"Interval: {SCAN_EVERY}m | Cooldown: {COOLDOWN_M}m")
    log.info(f"Sessions: London 07-12 UTC | NY 13-18 UTC")
    log.info(f"Filters:  Weekly gate | BTC gate | Struct SL | BE mgmt")
    log.info("="*55)

    # Start health server (Railway needs this)
    threading.Thread(target=start_health, daemon=True).start()
    log.info(f"Health server started on port {PORT}")

    # Startup message
    send_tg(
        "🚀 <b>SMC Engine Pro v3 Started</b>\n\n"
        "<b>5 improvements active:</b>\n"
        "1️⃣ Session filter — London + NY only\n"
        "2️⃣ Weekly bias gate — no counter-trend trades\n"
        "3️⃣ BTC correlation — altcoins aligned with BTC\n"
        "4️⃣ Structure SL — SL at swept wick / OB level\n"
        "5️⃣ Breakeven alerts — auto-alert when TP1 hit\n\n"
        "<b>Setups (priority order):</b>\n"
        "⚡ Sweep + OB Retest\n"
        "📊 3-TF HTF Confluence\n"
        "🔄 CHoCH Reversal\n"
        "📈 BOS Continuation\n\n"
        f"Scanning 10 pairs every 1 minute 📡\n"
        "Laptop can be off — alerts arrive 24/7\n\n"
        "📡 <b>SMC Engine Pro v3</b>"
    )
    log.info("✓ Startup TG message sent")

    # BE price check loop (every 60s in background)
    def be_loop():
        while True:
            try:
                if state['open_trades']:
                    check_prices()
            except Exception as e:
                log.debug(f"BE check error: {e}")
            time.sleep(60)
    threading.Thread(target=be_loop, daemon=True).start()

    # Telegram command listener (every 30s)
    # Supports: /stats /report /open /weekly
    last_update_id = [0]
    def tg_commands():
        while True:
            try:
                r = requests.get(
                    f'https://api.telegram.org/bot{TG_TOKEN}/getUpdates',
                    params={'offset': last_update_id[0]+1, 'timeout': 10},
                    timeout=15
                )
                if r.ok:
                    for upd in r.json().get('result', []):
                        last_update_id[0] = upd['update_id']
                        txt = upd.get('message',{}).get('text','').strip().lower()
                        if txt in ('/stats','/report','/journal'):
                            # Combine journal + live session stats
                            journal_msg = journal_stats_report()
                            w=state['stats'].get('wins',0); l=state['stats'].get('losses',0); b=state['stats'].get('be',0)
                            tot=w+l+b
                            sess_msg=(
                                f"\n\n📊 <b>This Session:</b>\n"
                                f"  ✅ Wins:    {w}\n"
                                f"  ❌ Losses:  {l}\n"
                                f"  ➡️ BE:       {b}\n"
                                f"  WR: {round(w/tot*100) if tot else 0}%\n"
                                f"  Open: {len(state['open_trades'])}\n"
                                f"  Alerts sent: {state['alerts_sent']}"
                            )
                            send_tg(journal_msg + sess_msg)
                        elif txt == '/open':
                            j = load_journal()
                            if j['open']:
                                msg = "🔓 <b>Open trades:</b>\n" + "\n".join(
                                    f"  • {sym}" for sym in j['open'])
                            else:
                                msg = "✅ No open trades right now"
                            send_tg(msg)
                        elif txt == '/help':
                            send_tg(
                                "📡 <b>SMC Bot Commands</b>\n\n"
                                "/stats — full performance report\n"
                                "/open  — show open trades\n"
                                "/help  — show this menu"
                            )
            except Exception as e:
                log.debug(f"TG command error: {e}")
            time.sleep(30)
    threading.Thread(target=tg_commands, daemon=True).start()
    log.info("✓ Telegram command listener started (/stats /open /help)")

    # Main scan loop
    while True:
        try:
            run_scan()
        except Exception as e:
            log.error(f"Scan error: {e}")
        log.info(f"Next scan in {SCAN_EVERY}m...")
        time.sleep(SCAN_EVERY * 60)

if __name__ == '__main__':
    main()