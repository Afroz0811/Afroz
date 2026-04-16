"""
SMC Engine — 24/7 Alert Server (Railway-compatible)
Uses environment variables — no command line args needed for Railway

Railway setup:
  1. Add these environment variables in Railway dashboard:
     TG_TOKEN = your bot token
     TG_CHAT  = your chat ID
  2. Deploy — done. Alerts arrive on Telegram 24/7.

Local run:
  pip install requests numpy
  TG_TOKEN=xxx TG_CHAT=yyy python smc_alert_server.py
"""

import os, time, logging, threading
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

# ── CONFIG (from env vars) ─────────────────────
TG_TOKEN    = os.environ.get('TG_TOKEN', '')
TG_CHAT     = os.environ.get('TG_CHAT', '')
MIN_SCORE   = int(os.environ.get('MIN_SCORE', '7'))
SCAN_EVERY  = int(os.environ.get('SCAN_EVERY_MIN', '5'))   # minutes
COOLDOWN_M  = int(os.environ.get('COOLDOWN_MIN', '60'))    # minutes
PORT        = int(os.environ.get('PORT', '8080'))           # Railway injects PORT

PAIRS = [
    {'sym':'BTC',  'name':'Bitcoin',   'kr':'XXBTZUSD', 'cg':'bitcoin'},
    {'sym':'ETH',  'name':'Ethereum',  'kr':'XETHZUSD', 'cg':'ethereum'},
    {'sym':'SOL',  'name':'Solana',    'kr':'SOLUSD',   'cg':'solana'},
    {'sym':'XRP',  'name':'XRP',       'kr':'XXRPZUSD', 'cg':'ripple'},
    {'sym':'ADA',  'name':'Cardano',   'kr':'ADAUSD',   'cg':'cardano'},
    {'sym':'DOGE', 'name':'Dogecoin',  'kr':'XDGUSD',   'cg':'dogecoin'},
    {'sym':'AVAX', 'name':'Avalanche', 'kr':'AVAXUSD',  'cg':'avalanche-2'},
    {'sym':'DOT',  'name':'Polkadot',  'kr':'DOTUSD',   'cg':'polkadot'},
    {'sym':'LINK', 'name':'Chainlink', 'kr':'LINKUSD',  'cg':'chainlink'},
    {'sym':'MATIC','name':'Polygon',   'kr':'MATICUSD', 'cg':'matic-network'},
]

KR_BASE = 'https://api.kraken.com/0/public'
CG_BASE = 'https://api.coingecko.com/api/v3'

# ── HEALTH CHECK SERVER (Railway requires this) ─
last_scan_time = 'Not yet'
signals_sent   = 0

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = f"""SMC Engine Running
Pairs:        {len(PAIRS)}
Min Score:    {MIN_SCORE}
Scan Every:   {SCAN_EVERY}m
Last Scan:    {last_scan_time}
Alerts Sent:  {signals_sent}
Time (UTC):   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
""".encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress noisy HTTP logs

def run_health_server():
    server = HTTPServer(('0.0.0.0', PORT), HealthHandler)
    log.info(f"Health server on port {PORT}")
    server.serve_forever()

# ── DATA FETCH ─────────────────────────────────
def fetch_candles(pair, limit=300):
    # Try Kraken first
    try:
        r = requests.get(f"{KR_BASE}/OHLC",
            params={'pair': pair['kr'], 'interval': 60},
            timeout=15)
        d = r.json()
        if not d.get('error'):
            key = next((k for k in d['result'] if k != 'last'), None)
            if key:
                raw = d['result'][key]
                if len(raw) > 20:
                    return [{'t':i,'o':float(k[1]),'h':float(k[2]),
                             'l':float(k[3]),'c':float(k[4]),'v':float(k[6])}
                            for i, k in enumerate(raw[-limit:])]
    except Exception as e:
        log.debug(f"Kraken error {pair['sym']}: {e}")

    # CoinGecko fallback
    try:
        r = requests.get(f"{CG_BASE}/coins/{pair['cg']}/ohlc",
            params={'vs_currency': 'usd', 'days': 7},
            timeout=15)
        raw = r.json()
        if isinstance(raw, list) and len(raw) > 5:
            return [{'t':i,'o':float(k[1]),'h':float(k[2]),
                     'l':float(k[3]),'c':float(k[4]),'v':50.0}
                    for i, k in enumerate(raw[-limit:])]
    except Exception as e:
        log.debug(f"CoinGecko error {pair['sym']}: {e}")

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
    r.append(100 if al == 0 else 100-100/(1+ag/al))
    for i in range(p+1, len(c)):
        d = c[i]-c[i-1]
        gi = d if d > 0 else 0
        li = abs(d) if d < 0 else 0
        ag = (ag*(p-1)+gi)/p
        al = (al*(p-1)+li)/p
        r.append(100 if al == 0 else 100-100/(1+ag/al))
    return r

def macd_hist(c):
    e12 = ema(c, 12); e26 = ema(c, 26)
    ln = [e12[i]-e26[i] if e12[i] and e26[i] else None for i in range(len(c))]
    vl = [v for v in ln if v is not None]
    if len(vl) < 9: return [None]*len(c)
    sr = ema(vl, 9)
    sg = [None]*len(c); si = 0
    for i in range(len(c)):
        if ln[i] is not None:
            sg[i] = sr[si] if si < len(sr) else None
            si += 1
    return [ln[i]-sg[i] if ln[i] is not None and sg[i] is not None else None
            for i in range(len(c))]

def calc_atr(kl, p=14):
    tr = [None]
    for i in range(1, len(kl)):
        t = max(kl[i]['h']-kl[i]['l'],
                abs(kl[i]['h']-kl[i-1]['c']),
                abs(kl[i]['l']-kl[i-1]['c']))
        tr.append(t)
    if len(tr) < p+1: return [None]*len(kl)
    r = [None]*p
    s = sum(tr[1:p+1])/p; r.append(s); pv = s
    for i in range(p+1, len(tr)):
        pv = (pv*(p-1)+tr[i])/p; r.append(pv)
    return r

def vol_avg(v, p=20):
    r = [None]*p
    for i in range(p, len(v)):
        r.append(sum(v[i-p:i])/p)
    return r

# ── SMC CORE ───────────────────────────────────
def find_swings(kl, lb=5):
    sh = []; sl = []
    for i in range(lb, len(kl)-lb):
        if all(kl[i]['h'] >= kl[j]['h'] for j in range(i-lb, i+lb+1) if j != i):
            sh.append((i, kl[i]['h']))
        if all(kl[i]['l'] <= kl[j]['l'] for j in range(i-lb, i+lb+1) if j != i):
            sl.append((i, kl[i]['l']))
    return sh, sl

def htf_bias(kl, factor=5):
    if len(kl) < factor*25: return 'neutral'
    htf = [kl[i*factor+factor-1]['c'] for i in range(len(kl)//factor)]
    e20 = ema(htf, 20); e50 = ema(htf, 50); n = len(htf)-1
    if not e20[n] or not e50[n]: return 'neutral'
    if htf[n] > e20[n] > e50[n]: return 'bullish'
    if htf[n] < e20[n] < e50[n]: return 'bearish'
    return 'neutral'

def is_choppy(atr_a, i, thresh=0.40):
    recent = [a for a in atr_a[max(0,i-20):i] if a is not None]
    if len(recent) < 5: return True
    return atr_a[i] < np.mean(recent)*thresh if atr_a[i] else True

# ── SETUP 1: SWEEP + OB ────────────────────────
def detect_sweep_ob(kl, sh, sl, i, atr_v, va_v, rsi_v, e20_v, e50_v, htf_b):
    if i < 15 or not atr_v or not va_v: return None
    k = kl[i]; price = k['c']

    # Bullish sweep
    r_lows = [(idx,p) for idx,p in sl if idx < i-1 and idx > i-50][-5:]
    for li, lvl in r_lows:
        if not (k['l'] < lvl < price): continue
        if lvl - k['l'] < atr_v*0.30: continue
        if k['v'] < va_v*1.20: continue
        if htf_b != 'bullish': continue
        if not rsi_v or not (25 < rsi_v < 62): continue
        ob = None
        for j in range(li-1, max(0, li-12), -1):
            if kl[j]['c'] < kl[j]['o']:
                fwd = (kl[min(j+2, len(kl)-1)]['c'] - kl[j]['c']) / kl[j]['c']
                if fwd > 0.003:
                    ob = {'top': kl[j]['o'], 'bot': kl[j]['l']}
                    break
        if not ob or not (ob['bot'] <= price <= ob['top']*1.005): continue
        ema_ok = e20_v and e50_v and price > e20_v > e50_v
        return {'dir':'BUY', 'setup':'SWEEP_OB', 'name':'⚡ Liq Sweep + OB Retest',
                'score': 8+(0.5 if ema_ok else 0), 'ob': ob, 'sweep_lvl': lvl,
                'tags': ['Sweep↑','OB_Retest','Vol✓','HTF✓']+(
                    ['EMA↑'] if ema_ok else [])+[f'RSI{round(rsi_v)}']}

    # Bearish sweep
    r_highs = [(idx,p) for idx,p in sh if idx < i-1 and idx > i-50][-5:]
    for hi_, lvl in r_highs:
        if not (k['h'] > lvl > price): continue
        if k['h'] - lvl < atr_v*0.30: continue
        if k['v'] < va_v*1.20: continue
        if htf_b != 'bearish': continue
        if not rsi_v or not (38 < rsi_v < 75): continue
        ob = None
        for j in range(hi_-1, max(0, hi_-12), -1):
            if kl[j]['c'] > kl[j]['o']:
                fwd = (kl[min(j+2, len(kl)-1)]['c'] - kl[j]['c']) / kl[j]['c']
                if fwd < -0.003:
                    ob = {'top': kl[j]['h'], 'bot': kl[j]['c']}
                    break
        if not ob or not (ob['bot']*0.995 <= price <= ob['top']): continue
        ema_ok = e20_v and e50_v and price < e20_v < e50_v
        return {'dir':'SELL', 'setup':'SWEEP_OB', 'name':'⚡ Liq Sweep + OB Retest',
                'score': 8+(0.5 if ema_ok else 0), 'ob': ob, 'sweep_lvl': lvl,
                'tags': ['Sweep↓','OB_Retest','Vol✓','HTF✓']+(
                    ['EMA↓'] if ema_ok else [])+[f'RSI{round(rsi_v)}']}
    return None

# ── SETUP 2: HTF CONFLUENCE ────────────────────
def detect_htf_confluence(kl, sh, sl, i, closes, rsi_v, e9_v, e20_v, e50_v, ht_v, va_v):
    if i < 50 or not ht_v: return None
    price = closes[i]
    wb = htf_bias(kl[:i+1], 21)
    db = htf_bias(kl[:i+1], 5)
    if wb == 'neutral' or db == 'neutral' or wb != db: return None
    rh = [(idx,p) for idx,p in sh if idx <= i][-4:]
    rl = [(idx,p) for idx,p in sl if idx <= i][-4:]
    h1 = 'neutral'
    if len(rh) >= 2 and len(rl) >= 2:
        if rh[-1][1] > rh[-2][1] and rl[-1][1] > rl[-2][1]: h1 = 'bullish'
        elif rh[-1][1] < rh[-2][1] and rl[-1][1] < rl[-2][1]: h1 = 'bearish'
    if h1 != wb: return None
    is_buy = (h1 == 'bullish')
    if is_buy and not (e9_v and e20_v and e50_v and e9_v > e20_v > e50_v): return None
    if not is_buy and not (e9_v and e20_v and e50_v and e9_v < e20_v < e50_v): return None
    if is_buy and ht_v <= 0: return None
    if not is_buy and ht_v >= 0: return None
    if is_buy and not (rsi_v and 25 < rsi_v < 62): return None
    if not is_buy and not (rsi_v and 38 < rsi_v < 75): return None
    vol_ok = va_v and kl[i]['v'] > va_v*1.1
    return {'dir': 'BUY' if is_buy else 'SELL',
            'setup': 'HTF_CONFLUENCE', 'name': '📊 3-TF HTF Confluence',
            'score': 8+(0.5 if vol_ok else 0),
            'tags': [f'W:{wb[:4]}', f'D:{db[:4]}', f'1h:{h1[:4]}',
                     'EMA_stack','MACD✓']+(['Vol✓'] if vol_ok else [])+[f'RSI{round(rsi_v)}']}

# ── SETUP 3: CHOCH ────────────────────────────
def detect_choch(kl, sh, sl, i, closes, rsi_v, e20_v, e50_v, ht_v, va_v):
    if i < 20 or not ht_v or not va_v: return None
    price = closes[i]
    rh = [(idx,p) for idx,p in sh if idx <= i][-5:]
    rl = [(idx,p) for idx,p in sl if idx <= i][-5:]
    if len(rh) < 3 or len(rl) < 3: return None
    h_gaps = [abs(rh[j+1][1]-rh[j][1])/rh[j][1] for j in range(len(rh)-1)]
    l_gaps = [abs(rl[j+1][1]-rl[j][1])/rl[j][1] for j in range(len(rl)-1)]
    if any(g < 0.003 for g in h_gaps[-2:]) or any(g < 0.003 for g in l_gaps[-2:]): return None
    h2, h1p = rh[-2][1], rh[-3][1]
    l2, l1p = rl[-2][1], rl[-3][1]
    vol_ok = kl[i]['v'] > va_v*1.05
    if (h2 < h1p and l2 < l1p and price > h2 and
            e20_v and price > e20_v and ht_v > 0 and
            rsi_v and 28 < rsi_v < 65 and vol_ok):
        return {'dir':'BUY', 'setup':'CHOCH', 'name':'🔄 CHoCH Reversal (Bear→Bull)',
                'score':8, 'tags':['CHoCH↑','CleanStr','Vol✓','MACD✓',f'RSI{round(rsi_v)}']}
    if (h2 > h1p and l2 > l1p and price < l2 and
            e20_v and price < e20_v and ht_v < 0 and
            rsi_v and 35 < rsi_v < 72 and vol_ok):
        return {'dir':'SELL', 'setup':'CHOCH', 'name':'🔄 CHoCH Reversal (Bull→Bear)',
                'score':8, 'tags':['CHoCH↓','CleanStr','Vol✓','MACD✓',f'RSI{round(rsi_v)}']}
    return None

# ── SETUP 4: BOS ───────────────────────────────
def detect_bos(kl, sh, sl, i, closes, rsi_v, e20_v, e50_v, ht_v, va_v):
    if i < 20 or not ht_v or not va_v: return None
    price = closes[i]
    rh = [(idx,p) for idx,p in sh if idx <= i][-4:]
    rl = [(idx,p) for idx,p in sl if idx <= i][-4:]
    if len(rh) < 3 or len(rl) < 3: return None
    h_gaps = [abs(rh[j+1][1]-rh[j][1])/rh[j][1] for j in range(len(rh)-1)]
    l_gaps = [abs(rl[j+1][1]-rl[j][1])/rl[j][1] for j in range(len(rl)-1)]
    if any(g < 0.003 for g in h_gaps[-2:]) or any(g < 0.003 for g in l_gaps[-2:]): return None
    h1p, h2p, h3p = rh[-3][1], rh[-2][1], rh[-1][1]
    l1p, l2p, l3p = rl[-3][1], rl[-2][1], rl[-1][1]
    vol_ok = kl[i]['v'] > va_v*1.1
    if (h3p > h2p > h1p and l3p > l2p and price > h2p and ht_v > 0 and
            e20_v and e50_v and price > e20_v > e50_v and
            rsi_v and 28 < rsi_v < 68 and vol_ok):
        return {'dir':'BUY', 'setup':'BOS', 'name':'📈 BOS Continuation (Bullish)',
                'score':7, 'tags':['BOS↑','HH+HL','Vol✓','MACD✓',f'RSI{round(rsi_v)}']}
    if (h3p < h2p < h1p and l3p < l2p and price < l2p and ht_v < 0 and
            e20_v and e50_v and price < e20_v < e50_v and
            rsi_v and 32 < rsi_v < 72 and vol_ok):
        return {'dir':'SELL', 'setup':'BOS', 'name':'📉 BOS Continuation (Bearish)',
                'score':7, 'tags':['BOS↓','LH+LL','Vol✓','MACD✓',f'RSI{round(rsi_v)}']}
    return None

# ── MAIN SIGNAL ENGINE ─────────────────────────
last_fired = {}

def compute_signal(kl, pair):
    if len(kl) < 80: return None
    i = len(kl)-1
    closes = [k['c'] for k in kl]
    vols   = [k['v'] for k in kl]
    rsi_a  = rsi(closes)
    e9_a   = ema(closes, 9)
    e20_a  = ema(closes, 20)
    e50_a  = ema(closes, 50)
    ht_a   = macd_hist(closes)
    atr_a  = calc_atr(kl)
    va_a   = vol_avg(vols)
    price  = closes[i]

    if any(x is None for x in [rsi_a[i], e9_a[i], e20_a[i], e50_a[i], atr_a[i], va_a[i]]):
        return None
    if is_choppy(atr_a, i): return None
    if atr_a[i]/price < 0.002: return None

    # Cooldown
    last = last_fired.get(pair['sym'])
    if last and (time.time()-last['time'])/60 < COOLDOWN_M:
        return None

    sh, sl = find_swings(kl, 5)
    htf_b  = htf_bias(kl, 5)

    sig = (detect_sweep_ob(kl,sh,sl,i,atr_a[i],va_a[i],rsi_a[i],e20_a[i],e50_a[i],htf_b) or
           detect_htf_confluence(kl,sh,sl,i,closes,rsi_a[i],e9_a[i],e20_a[i],e50_a[i],ht_a[i],va_a[i]) or
           detect_choch(kl,sh,sl,i,closes,rsi_a[i],e20_a[i],e50_a[i],ht_a[i],va_a[i]) or
           detect_bos(kl,sh,sl,i,closes,rsi_a[i],e20_a[i],e50_a[i],ht_a[i],va_a[i]))

    if not sig or sig['score'] < MIN_SCORE: return None

    is_buy = sig['dir'] == 'BUY'
    rh_ = [(idx,p) for idx,p in sh if idx <= i]
    rl_ = [(idx,p) for idx,p in sl if idx <= i]
    last_l = rl_[-1][1] if rl_ else price-atr_a[i]*2
    last_h = rh_[-1][1] if rh_ else price+atr_a[i]*2

    sl_p = (min(last_l, price-atr_a[i]*1.5)*0.997 if is_buy
            else max(last_h, price+atr_a[i]*1.5)*1.003)
    if sig.get('ob'):
        if is_buy:  sl_p = min(sl_p, sig['ob']['bot']*0.997)
        else:       sl_p = max(sl_p, sig['ob']['top']*1.003)

    risk = abs(price - sl_p)
    if risk <= 0: return None

    rr_mult = 3.0 if sig['setup'] == 'SWEEP_OB' else 2.5
    tp_p    = price + risk*rr_mult if is_buy else price - risk*rr_mult
    rr      = abs(tp_p - price) / risk
    if rr < 2.0: return None

    tp1  = price + risk*2 if is_buy else price - risk*2
    tp3  = price + risk*3 if is_buy else price - risk*3
    conf = min(97, int(sig['score']*8.5 + min(rr,3)*2.5))

    return {**sig, 'pair':pair, 'price':price, 'sl':sl_p, 'tp':tp_p,
            'tp1':tp1, 'tp3':tp3, 'rr':round(rr,2), 'conf':conf,
            'risk_pct': round(abs(price-sl_p)/price*100, 2),
            'rew_pct':  round(abs(tp_p-price)/price*100, 2),
            'htf':htf_b, 'rsi_val':round(rsi_a[i])}

# ── FORMATTING ─────────────────────────────────
def fp(p):
    if not p: return '—'
    if p >= 10000: return f'${p:,.0f}'
    if p >= 100:   return f'${p:.2f}'
    if p >= 1:     return f'${p:.3f}'
    return f'${p:.5f}'

def build_alert(sig):
    s = sig
    is_buy = s['dir'] == 'BUY'
    tips = {
        'SWEEP_OB':        f"Liquidity swept at {fp(s.get('sweep_lvl', s['price']))} — institutions filled, now reversing. OB retest entry.",
        'HTF_CONFLUENCE':  'Weekly + Daily + 1h EMA stacks all aligned. High-conviction trend continuation.',
        'CHOCH':           'Change of Character — structural shift. Early reversal, tight SL.',
        'BOS':             'Break of Structure confirmed. Trend continuation with clean structure.',
    }
    setup_emojis = {'SWEEP_OB':'⚡','HTF_CONFLUENCE':'📊','CHOCH':'🔄','BOS':'📈'}
    e = setup_emojis.get(s['setup'], '📡')
    lines = [
        f"{'🟢' if is_buy else '🔴'} <b>{s['dir']} — {s['pair']['sym']}/USD</b>",
        f"{e} <b>Setup: {s['name']}</b>",
        '',
        f"📌 <i>{tips.get(s['setup'], 'SMC confluence setup.')}</i>",
        '',
        '💰 <b>Trade Levels</b>',
        f"  Entry:  <code>{fp(s['price'])}</code>",
        f"  SL:     <code>{fp(s['sl'])}</code>  <i>(-{s['risk_pct']}%)</i>",
        f"  TP1:    <code>{fp(s['tp1'])}</code>  <i>(1:2 — partial close)</i>",
        f"  TP2:    <code>{fp(s['tp'])}</code>   <i>(1:{s['rr']} — main)</i>",
        f"  TP3:    <code>{fp(s['tp3'])}</code>  <i>(1:3 — runner)</i>",
        '',
        f"📊 <b>Score: {s['score']}/10  |  Conf: {s['conf']}%  |  R:R 1:{s['rr']}</b>",
        f"  Tags:  {' · '.join(s['tags'])}",
        f"  HTF:   {s['htf']}  |  RSI: {s['rsi_val']}",
    ]
    if s.get('ob'):
        lines.append(f"  OB:    {fp(s['ob']['bot'])} – {fp(s['ob']['top'])}")
    lines += [
        '',
        '⚠️ <i>Not financial advice. Always manage risk.</i>',
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC  |  📡 <b>SMC Engine Pro</b>",
    ]
    return '\n'.join(lines)

# ── TELEGRAM ───────────────────────────────────
def send_tg(msg):
    if not TG_TOKEN or not TG_CHAT:
        log.warning("TG_TOKEN or TG_CHAT not set")
        return False
    try:
        r = requests.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={'chat_id': TG_CHAT, 'text': msg,
                  'parse_mode': 'HTML', 'disable_web_page_preview': True},
            timeout=10
        )
        return r.ok
    except Exception as e:
        log.error(f"Telegram error: {e}")
        return False

# ── SCAN ───────────────────────────────────────
def run_scan():
    global last_scan_time, signals_sent
    last_scan_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    log.info(f"Scanning {len(PAIRS)} pairs...")
    fired = 0
    for pair in PAIRS:
        try:
            kl = fetch_candles(pair, limit=300)
            if not kl:
                log.info(f"  {pair['sym']}: no data")
                continue
            sig = compute_signal(kl, pair)
            if sig:
                msg = build_alert(sig)
                ok  = send_tg(msg)
                if ok:
                    last_fired[pair['sym']] = {'setup': sig['setup'], 'time': time.time()}
                    signals_sent += 1
                    fired += 1
                    log.info(f"  ✓ {pair['sym']}: {sig['name']} {sig['dir']} score={sig['score']} → TG sent")
                else:
                    log.error(f"  {pair['sym']}: TG failed")
            else:
                log.info(f"  {pair['sym']}: no setup")
            time.sleep(1.5)  # rate limit between pairs
        except Exception as e:
            log.error(f"  {pair['sym']} error: {e}")
    log.info(f"Done. {fired} alerts sent this scan.")

# ── MAIN ───────────────────────────────────────
# (ONLY CHANGE: removed invalid global line — everything else same)

# ── MAIN ───────────────────────────────────────
def main():
    if not TG_TOKEN or not TG_CHAT:
        log.error("ERROR: Set TG_TOKEN and TG_CHAT environment variables")
        log.error("  Railway: add them in Variables tab")
        log.error("  Local:   TG_TOKEN=xxx TG_CHAT=yyy python smc_alert_server.py")
        raise SystemExit(1)

    log.info("="*55)
    log.info("SMC ENGINE 24/7 ALERT SERVER")
    log.info(f"Pairs: {len(PAIRS)} | Score ≥ {MIN_SCORE} | Every {SCAN_EVERY}m")
    log.info(f"Cooldown: {COOLDOWN_M}m | Health: port {PORT}")
    log.info("="*55)

    # Start health check server
    t = threading.Thread(target=run_health_server, daemon=True)
    t.start()

    send_tg(
        "✅ <b>SMC Engine Pro — Server Started</b>\n\n"
        f"🔍 Scanning 10 pairs every {SCAN_EVERY} minutes\n"
        "⚡ Setup 1: Liq Sweep + OB Retest\n"
        "📊 Setup 2: 3-TF HTF Confluence\n"
        "🔄 Setup 3: CHoCH Reversal\n"
        "📈 Setup 4: BOS Continuation\n\n"
        f"Minimum score: {MIN_SCORE}/10\n"
        "Alerts arrive here automatically 🚀"
    )

    while True:
        try:
            run_scan()
        except Exception as e:
            log.error(f"Scan loop error: {e}")
        log.info(f"Next scan in {SCAN_EVERY} minutes...")
        time.sleep(SCAN_EVERY * 60)


if __name__ == '__main__':
    main()