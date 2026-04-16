"""
SMC Engine Pro v2 — Improved Strategy Server
Improvements:
  1. Session filter (London + NY only)
  2. Weekly bias hard gate
  3. BTC correlation gate
  4. Confirmation candle for Sweep+OB
  5. SL to breakeven + trail alerts
  6. Pullback entry (50% retracement)
  7. Structure-based SL
"""
import os, time, logging, threading, json
import requests
import numpy as np
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────
TG_TOKEN   = os.environ.get('TG_TOKEN', '')
TG_CHAT    = os.environ.get('TG_CHAT', '')
MIN_SCORE  = int(os.environ.get('MIN_SCORE', '7'))
SCAN_EVERY = int(os.environ.get('SCAN_EVERY_MIN', '5'))
COOLDOWN_M = int(os.environ.get('COOLDOWN_MIN', '60'))
PORT       = int(os.environ.get('PORT', '8080'))

# Session windows (UTC hours) — London + NY only
LONDON_OPEN  = (7,  12)
NY_OPEN      = (13, 18)
USE_SESSIONS = os.environ.get('USE_SESSIONS', 'true').lower() == 'true'

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

# ── STATE ─────────────────────────────────────
last_fired   = {}       # sym -> {setup, time, price}
open_trades  = {}       # sym -> trade dict (for SL management)
btc_cache    = {'bias':'neutral','time':0}
server_stats = {'scans':0,'alerts':0,'start':time.time(),'last_scan':'never'}

# ── HEALTH SERVER ─────────────────────────────
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        up = int(time.time()-server_stats['start'])
        body = (
            f"SMC Engine Pro v2\n"
            f"Uptime:       {up//3600}h {(up%3600)//60}m\n"
            f"Scans:        {server_stats['scans']}\n"
            f"Alerts sent:  {server_stats['alerts']}\n"
            f"Last scan:    {server_stats['last_scan']}\n"
            f"Open trades:  {len(open_trades)}\n"
            f"BTC bias:     {btc_cache['bias']}\n"
            f"Session:      {'ACTIVE' if in_session() else 'CLOSED'}\n"
            f"Time UTC:     {datetime.now(timezone.utc).strftime('%H:%M:%S')}\n"
        ).encode()
        self.send_response(200)
        self.send_header('Content-Type','text/plain')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self,*a): pass

def run_health():
    HTTPServer(('0.0.0.0',PORT),Health).serve_forever()

# ── IMPROVEMENT 1: SESSION FILTER ────────────
def in_session():
    """Only trade London (07-12 UTC) and NY (13-18 UTC)"""
    if not USE_SESSIONS:
        return True
    h = datetime.now(timezone.utc).hour
    return (LONDON_OPEN[0] <= h < LONDON_OPEN[1] or
            NY_OPEN[0]     <= h < NY_OPEN[1])

def session_name():
    h = datetime.now(timezone.utc).hour
    if LONDON_OPEN[0] <= h < LONDON_OPEN[1]: return 'London'
    if NY_OPEN[0]     <= h < NY_OPEN[1]:     return 'New York'
    return None

# ── DATA FETCH ────────────────────────────────
def fetch_candles(pair, limit=300):
    try:
        r = requests.get(f"{KR_BASE}/OHLC",
            params={'pair':pair['kr'],'interval':60}, timeout=15)
        d = r.json()
        if not d.get('error'):
            key = next((k for k in d['result'] if k!='last'),None)
            if key:
                raw = d['result'][key]
                if len(raw)>20:
                    return [{'t':i,'o':float(k[1]),'h':float(k[2]),
                             'l':float(k[3]),'c':float(k[4]),'v':float(k[6])}
                            for i,k in enumerate(raw[-limit:])]
    except: pass
    try:
        r = requests.get(f"{CG_BASE}/coins/{pair['cg']}/ohlc",
            params={'vs_currency':'usd','days':7}, timeout=15)
        raw = r.json()
        if isinstance(raw,list) and len(raw)>5:
            return [{'t':i,'o':float(k[1]),'h':float(k[2]),
                     'l':float(k[3]),'c':float(k[4]),'v':50.0}
                    for i,k in enumerate(raw[-limit:])]
    except: pass
    return []

# ── INDICATORS ────────────────────────────────
def ema(c,p):
    if len(c)<p: return [None]*len(c)
    k=2/(p+1); r=[None]*(p-1); s=sum(c[:p])/p; r.append(s); pv=s
    for i in range(p,len(c)): pv=c[i]*k+pv*(1-k); r.append(pv)
    return r

def rsi(c,p=14):
    if len(c)<p+1: return [None]*len(c)
    r=[None]*p; g=l=0.0
    for i in range(1,p+1):
        d=c[i]-c[i-1]
        if d>0: g+=d
        else: l+=abs(d)
    ag,al=g/p,l/p; r.append(100 if al==0 else 100-100/(1+ag/al))
    for i in range(p+1,len(c)):
        d=c[i]-c[i-1]; gi=d if d>0 else 0; li=abs(d) if d<0 else 0
        ag=(ag*(p-1)+gi)/p; al=(al*(p-1)+li)/p
        r.append(100 if al==0 else 100-100/(1+ag/al))
    return r

def macd_hist(c):
    e12=ema(c,12); e26=ema(c,26)
    ln=[e12[i]-e26[i] if e12[i] and e26[i] else None for i in range(len(c))]
    vl=[v for v in ln if v is not None]
    if len(vl)<9: return [None]*len(c)
    sr=ema(vl,9); sg=[None]*len(c); si=0
    for i in range(len(c)):
        if ln[i] is not None: sg[i]=sr[si] if si<len(sr) else None; si+=1
    return [ln[i]-sg[i] if ln[i] is not None and sg[i] is not None else None
            for i in range(len(c))]

def calc_atr(kl,p=14):
    tr=[None]
    for i in range(1,len(kl)):
        t=max(kl[i]['h']-kl[i]['l'],abs(kl[i]['h']-kl[i-1]['c']),
              abs(kl[i]['l']-kl[i-1]['c']))
        tr.append(t)
    if len(tr)<p+1: return [None]*len(kl)
    r=[None]*p; s=sum(tr[1:p+1])/p; r.append(s); pv=s
    for i in range(p+1,len(tr)): pv=(pv*(p-1)+tr[i])/p; r.append(pv)
    return r

def vol_avg(v,p=20):
    r=[None]*p
    for i in range(p,len(v)): r.append(sum(v[i-p:i])/p)
    return r

# ── SMC CORE ──────────────────────────────────
def find_swings(kl,lb=5):
    sh=[]; sl=[]
    for i in range(lb,len(kl)-lb):
        if all(kl[i]['h']>=kl[j]['h'] for j in range(i-lb,i+lb+1) if j!=i):
            sh.append((i,kl[i]['h']))
        if all(kl[i]['l']<=kl[j]['l'] for j in range(i-lb,i+lb+1) if j!=i):
            sl.append((i,kl[i]['l']))
    return sh,sl

def htf_bias_calc(kl,factor=5):
    if len(kl)<factor*25: return 'neutral'
    htf=[kl[i*factor+factor-1]['c'] for i in range(len(kl)//factor)]
    e20=ema(htf,20); e50=ema(htf,50); n=len(htf)-1
    if not e20[n] or not e50[n]: return 'neutral'
    if htf[n]>e20[n]>e50[n]: return 'bullish'
    if htf[n]<e20[n]<e50[n]: return 'bearish'
    return 'neutral'

def weekly_bias_calc(kl):
    """IMPROVEMENT 2: Weekly bias via 21x resampling"""
    return htf_bias_calc(kl, 21)

def is_choppy(atr_a,i,thresh=0.40):
    recent=[a for a in atr_a[max(0,i-20):i] if a is not None]
    if len(recent)<5: return True
    return atr_a[i]<np.mean(recent)*thresh if atr_a[i] else True

# ── IMPROVEMENT 2: WEEKLY BIAS GATE ──────────
def passes_weekly_gate(kl, direction):
    """
    Hard gate: weekly trend must agree with signal direction.
    Weekly bullish → only BUY. Weekly bearish → only SELL.
    Weekly neutral → skip.
    """
    wb = weekly_bias_calc(kl)
    if wb == 'neutral':
        return False, wb
    if direction == 'BUY' and wb != 'bullish':
        return False, wb
    if direction == 'SELL' and wb != 'bearish':
        return False, wb
    return True, wb

# ── IMPROVEMENT 3: BTC CORRELATION GATE ──────
def get_btc_bias(force_refresh=False):
    """
    Cache BTC bias for 15 mins.
    Altcoin signals must align with BTC direction.
    """
    if force_refresh or time.time()-btc_cache['time'] > 900:
        btc_pair = next(p for p in PAIRS if p['sym']=='BTC')
        kl = fetch_candles(btc_pair, 300)
        if kl:
            bias = htf_bias_calc(kl, 5)
            btc_cache['bias'] = bias
            btc_cache['time'] = time.time()
            log.info(f"BTC bias refreshed: {bias}")
    return btc_cache['bias']

def passes_btc_gate(sym, direction):
    """
    If BTC is strongly trending, altcoins must follow.
    BTC strongly bullish → block altcoin SELL signals
    BTC strongly bearish → block altcoin BUY signals
    """
    if sym == 'BTC':
        return True  # BTC doesn't need its own gate
    btc_b = btc_cache['bias']
    if btc_b == 'bullish' and direction == 'SELL':
        return False
    if btc_b == 'bearish' and direction == 'BUY':
        return False
    return True

# ── IMPROVEMENT 4: CONFIRMATION CANDLE ────────
def has_confirmation_candle(kl, i, direction):
    """
    After price enters OB zone, wait for a confirmation candle.
    Bullish confirmation: green candle with body > 50% of range,
                         closing in upper 40% of its range.
    Bearish confirmation: red candle closing in lower 40%.
    Prevents entering on a knife-catch.
    """
    if i < 1: return False
    k = kl[i]
    body = abs(k['c'] - k['o'])
    rng  = k['h'] - k['l']
    if rng < 0.0001: return False
    body_ratio = body / rng

    if direction == 'BUY':
        # Green candle, body > 50% of range, close in upper 40%
        is_green = k['c'] > k['o']
        close_position = (k['c'] - k['l']) / rng
        return is_green and body_ratio > 0.50 and close_position > 0.60

    else:  # SELL
        # Red candle, body > 50% of range, close in lower 40%
        is_red = k['c'] < k['o']
        close_position = (k['c'] - k['l']) / rng
        return is_red and body_ratio > 0.50 and close_position < 0.40

# ── IMPROVEMENT 6: PULLBACK ENTRY ─────────────
def get_pullback_entry(kl, i, direction, signal_candle_idx):
    """
    Instead of entering at close, use 50% retracement of signal candle.
    This gives better RR and more precise entry.
    For live trading — this becomes a limit order level.
    """
    k = kl[signal_candle_idx]
    mid = (k['h'] + k['l']) / 2
    # If price has already pulled back past mid, entry at current price
    current = kl[i]['c']
    if direction == 'BUY':
        return min(current, mid * 1.001)   # slightly above 50%
    else:
        return max(current, mid * 0.999)

# ── IMPROVEMENT 7: STRUCTURE-BASED SL ────────
def get_structure_sl(kl, sh, sl, i, direction, ob=None, sweep_lvl=None, atr_v=0):
    """
    SL based purely on structure, not ATR multiples:
    BUY:  SL = below swept wick low (if sweep) OR below OB bottom
    SELL: SL = above swept wick high (if sweep) OR above OB top
    Fallback to ATR only if no structural level found.
    """
    price = kl[i]['c']
    if direction == 'BUY':
        candidates = []
        if sweep_lvl:
            # Below the sweep wick
            sweep_low = min(k['l'] for k in kl[max(0,i-3):i+1])
            candidates.append(sweep_low * 0.997)
        if ob:
            candidates.append(ob['bot'] * 0.997)
        rl = [(idx,p) for idx,p in sl if idx<=i]
        if rl:
            candidates.append(rl[-1][1] * 0.997)
        if candidates:
            sl_p = max(candidates)  # take the highest (closest to price = tightest)
            # Sanity check: SL must be below price and risk must be reasonable
            if sl_p < price and (price-sl_p)/price < 0.08:
                return sl_p
        return price - atr_v * 1.5  # fallback
    else:
        candidates = []
        if sweep_lvl:
            sweep_high = max(k['h'] for k in kl[max(0,i-3):i+1])
            candidates.append(sweep_high * 1.003)
        if ob:
            candidates.append(ob['top'] * 1.003)
        rh = [(idx,p) for idx,p in sh if idx<=i]
        if rh:
            candidates.append(rh[-1][1] * 1.003)
        if candidates:
            sl_p = min(candidates)
            if sl_p > price and (sl_p-price)/price < 0.08:
                return sl_p
        return price + atr_v * 1.5

# ── SETUPS ────────────────────────────────────
def detect_sweep_ob(kl,sh,sl,i,atr_v,va_v,rsi_v,e20_v,e50_v,htf_b):
    if i<15 or not atr_v or not va_v: return None
    k=kl[i]; price=k['c']
    # Bullish
    r_lows=[(idx,p) for idx,p in sl if idx<i-1 and idx>i-50][-5:]
    for li,lvl in r_lows:
        if not(k['l']<lvl<price): continue
        if lvl-k['l']<atr_v*0.30: continue
        if k['v']<va_v*1.20: continue
        if htf_b!='bullish': continue
        if not rsi_v or not(25<rsi_v<62): continue
        ob=None
        for j in range(li-1,max(0,li-12),-1):
            if kl[j]['c']<kl[j]['o']:
                fwd=(kl[min(j+2,len(kl)-1)]['c']-kl[j]['c'])/kl[j]['c']
                if fwd>0.003: ob={'top':kl[j]['o'],'bot':kl[j]['l']}; break
        if not ob or not(ob['bot']<=price<=ob['top']*1.005): continue
        # IMPROVEMENT 4: require confirmation candle
        if not has_confirmation_candle(kl,i,'BUY'): continue
        ema_ok=e20_v and e50_v and price>e20_v>e50_v
        return{'dir':'BUY','setup':'SWEEP_OB','name':'⚡ Liq Sweep + OB Retest',
               'score':8+(0.5 if ema_ok else 0),'ob':ob,'sweep_lvl':lvl,
               'tags':['Sweep↑','OB✓','Conf✓','HTF✓']+(
                   ['EMA↑'] if ema_ok else [])+[f'RSI{round(rsi_v)}']}
    # Bearish
    r_highs=[(idx,p) for idx,p in sh if idx<i-1 and idx>i-50][-5:]
    for hi_,lvl in r_highs:
        if not(k['h']>lvl>price): continue
        if k['h']-lvl<atr_v*0.30: continue
        if k['v']<va_v*1.20: continue
        if htf_b!='bearish': continue
        if not rsi_v or not(38<rsi_v<75): continue
        ob=None
        for j in range(hi_-1,max(0,hi_-12),-1):
            if kl[j]['c']>kl[j]['o']:
                fwd=(kl[min(j+2,len(kl)-1)]['c']-kl[j]['c'])/kl[j]['c']
                if fwd<-0.003: ob={'top':kl[j]['h'],'bot':kl[j]['c']}; break
        if not ob or not(ob['bot']*0.995<=price<=ob['top']): continue
        if not has_confirmation_candle(kl,i,'SELL'): continue
        ema_ok=e20_v and e50_v and price<e20_v<e50_v
        return{'dir':'SELL','setup':'SWEEP_OB','name':'⚡ Liq Sweep + OB Retest',
               'score':8+(0.5 if ema_ok else 0),'ob':ob,'sweep_lvl':lvl,
               'tags':['Sweep↓','OB✓','Conf✓','HTF✓']+(
                   ['EMA↓'] if ema_ok else [])+[f'RSI{round(rsi_v)}']}
    return None

def detect_htf_confluence(kl,sh,sl,i,closes,rsi_v,e9_v,e20_v,e50_v,ht_v,va_v):
    if i<50 or not ht_v: return None
    price=closes[i]
    wb=htf_bias_calc(kl[:i+1],21); db=htf_bias_calc(kl[:i+1],5)
    if wb=='neutral' or db=='neutral' or wb!=db: return None
    rh_=[(idx,p) for idx,p in sh if idx<=i][-4:]
    rl_=[(idx,p) for idx,p in sl if idx<=i][-4:]
    h1='neutral'
    if len(rh_)>=2 and len(rl_)>=2:
        if rh_[-1][1]>rh_[-2][1] and rl_[-1][1]>rl_[-2][1]: h1='bullish'
        elif rh_[-1][1]<rh_[-2][1] and rl_[-1][1]<rl_[-2][1]: h1='bearish'
    if h1!=wb: return None
    is_buy=(h1=='bullish')
    if is_buy and not(e9_v and e20_v and e50_v and e9_v>e20_v>e50_v): return None
    if not is_buy and not(e9_v and e20_v and e50_v and e9_v<e20_v<e50_v): return None
    if is_buy and ht_v<=0: return None
    if not is_buy and ht_v>=0: return None
    if is_buy and not(rsi_v and 25<rsi_v<62): return None
    if not is_buy and not(rsi_v and 38<rsi_v<75): return None
    vol_ok=va_v and kl[i]['v']>va_v*1.1
    return{'dir':'BUY' if is_buy else 'SELL','setup':'HTF_CONFLUENCE',
           'name':'📊 3-TF HTF Confluence','score':8+(0.5 if vol_ok else 0),
           'tags':[f'W:{wb[:4]}',f'D:{db[:4]}',f'1h:{h1[:4]}',
                   'EMA_stack','MACD✓']+(['Vol✓'] if vol_ok else [])+[f'RSI{round(rsi_v)}']}

def detect_choch(kl,sh,sl,i,closes,rsi_v,e20_v,e50_v,ht_v,va_v):
    if i<20 or not ht_v or not va_v: return None
    price=closes[i]
    rh=[(idx,p) for idx,p in sh if idx<=i][-5:]
    rl=[(idx,p) for idx,p in sl if idx<=i][-5:]
    if len(rh)<3 or len(rl)<3: return None
    h_gaps=[abs(rh[j+1][1]-rh[j][1])/rh[j][1] for j in range(len(rh)-1)]
    l_gaps=[abs(rl[j+1][1]-rl[j][1])/rl[j][1] for j in range(len(rl)-1)]
    if any(g<0.003 for g in h_gaps[-2:]) or any(g<0.003 for g in l_gaps[-2:]): return None
    h2,h1p=rh[-2][1],rh[-3][1]; l2,l1p=rl[-2][1],rl[-3][1]
    vol_ok=kl[i]['v']>va_v*1.05
    if(h2<h1p and l2<l1p and price>h2 and e20_v and price>e20_v
       and ht_v>0 and rsi_v and 28<rsi_v<65 and vol_ok):
        return{'dir':'BUY','setup':'CHOCH','name':'🔄 CHoCH Reversal (Bear→Bull)',
               'score':8,'tags':['CHoCH↑','CleanStr','Vol✓','MACD✓',f'RSI{round(rsi_v)}']}
    if(h2>h1p and l2>l1p and price<l2 and e20_v and price<e20_v
       and ht_v<0 and rsi_v and 35<rsi_v<72 and vol_ok):
        return{'dir':'SELL','setup':'CHOCH','name':'🔄 CHoCH Reversal (Bull→Bear)',
               'score':8,'tags':['CHoCH↓','CleanStr','Vol✓','MACD✓',f'RSI{round(rsi_v)}']}
    return None

def detect_bos(kl,sh,sl,i,closes,rsi_v,e20_v,e50_v,ht_v,va_v):
    if i<20 or not ht_v or not va_v: return None
    price=closes[i]
    rh=[(idx,p) for idx,p in sh if idx<=i][-4:]
    rl=[(idx,p) for idx,p in sl if idx<=i][-4:]
    if len(rh)<3 or len(rl)<3: return None
    h_gaps=[abs(rh[j+1][1]-rh[j][1])/rh[j][1] for j in range(len(rh)-1)]
    l_gaps=[abs(rl[j+1][1]-rl[j][1])/rl[j][1] for j in range(len(rl)-1)]
    if any(g<0.003 for g in h_gaps[-2:]) or any(g<0.003 for g in l_gaps[-2:]): return None
    h1p,h2p,h3p=rh[-3][1],rh[-2][1],rh[-1][1]
    l1p,l2p,l3p=rl[-3][1],rl[-2][1],rl[-1][1]
    vol_ok=kl[i]['v']>va_v*1.1
    if(h3p>h2p>h1p and l3p>l2p and price>h2p and ht_v>0
       and e20_v and e50_v and price>e20_v>e50_v
       and rsi_v and 28<rsi_v<68 and vol_ok):
        return{'dir':'BUY','setup':'BOS','name':'📈 BOS Continuation (Bullish)',
               'score':7,'tags':['BOS↑','HH+HL','Vol✓','MACD✓',f'RSI{round(rsi_v)}']}
    if(h3p<h2p<h1p and l3p<l2p and price<l2p and ht_v<0
       and e20_v and e50_v and price<e20_v<e50_v
       and rsi_v and 32<rsi_v<72 and vol_ok):
        return{'dir':'SELL','setup':'BOS','name':'📉 BOS Continuation (Bearish)',
               'score':7,'tags':['BOS↓','LH+LL','Vol✓','MACD✓',f'RSI{round(rsi_v)}']}
    return None

# ── MAIN SIGNAL ENGINE ─────────────────────────
def compute_signal(kl, pair):
    if len(kl)<80: return None
    i=len(kl)-1
    closes=[k['c'] for k in kl]; vols=[k['v'] for k in kl]
    rsi_a=rsi(closes); e9_a=ema(closes,9)
    e20_a=ema(closes,20); e50_a=ema(closes,50)
    ht_a=macd_hist(closes); atr_a=calc_atr(kl); va_a=vol_avg(vols)
    price=closes[i]
    if any(x is None for x in [rsi_a[i],e9_a[i],e20_a[i],e50_a[i],atr_a[i],va_a[i]]):
        return None
    if is_choppy(atr_a,i): return None
    if atr_a[i]/price<0.002: return None
    last=last_fired.get(pair['sym'])
    if last and (time.time()-last['time'])/60<COOLDOWN_M: return None

    sh,sl=find_swings(kl,5)
    htf_b=htf_bias_calc(kl,5)

    sig=(detect_sweep_ob(kl,sh,sl,i,atr_a[i],va_a[i],rsi_a[i],e20_a[i],e50_a[i],htf_b) or
         detect_htf_confluence(kl,sh,sl,i,closes,rsi_a[i],e9_a[i],e20_a[i],e50_a[i],ht_a[i],va_a[i]) or
         detect_choch(kl,sh,sl,i,closes,rsi_a[i],e20_a[i],e50_a[i],ht_a[i],va_a[i]) or
         detect_bos(kl,sh,sl,i,closes,rsi_a[i],e20_a[i],e50_a[i],ht_a[i],va_a[i]))

    if not sig or sig['score']<MIN_SCORE: return None

    # IMPROVEMENT 2: Weekly bias gate
    ok,wb=passes_weekly_gate(kl,sig['dir'])
    if not ok:
        log.info(f"  {pair['sym']}: blocked by weekly gate (weekly={wb}, signal={sig['dir']})")
        return None

    # IMPROVEMENT 3: BTC correlation gate
    if not passes_btc_gate(pair['sym'],sig['dir']):
        log.info(f"  {pair['sym']}: blocked by BTC gate (btc={btc_cache['bias']}, signal={sig['dir']})")
        return None

    is_buy=sig['dir']=='BUY'

    # IMPROVEMENT 7: Structure-based SL
    sl_p=get_structure_sl(kl,sh,sl,i,sig['dir'],
                          ob=sig.get('ob'),
                          sweep_lvl=sig.get('sweep_lvl'),
                          atr_v=atr_a[i])

    risk=abs(price-sl_p)
    if risk<=0 or risk/price>0.08: return None

    # IMPROVEMENT 6: Pullback entry level (limit order suggestion)
    entry_limit=get_pullback_entry(kl,i,sig['dir'],i)

    rr_mult=3.0 if sig['setup']=='SWEEP_OB' else 2.5
    tp_p=price+risk*rr_mult if is_buy else price-risk*rr_mult
    tp1=price+risk*2 if is_buy else price-risk*2
    tp3=price+risk*3 if is_buy else price-risk*3
    rr=abs(tp_p-price)/risk
    if rr<2.0: return None

    conf=min(97,int(sig['score']*8.5+min(rr,3)*2.5))
    sess=session_name()

    return{**sig,'pair':pair,'price':price,'entry_limit':entry_limit,
           'sl':sl_p,'tp':tp_p,'tp1':tp1,'tp3':tp3,
           'rr':round(rr,2),'conf':conf,
           'risk_pct':round(abs(price-sl_p)/price*100,2),
           'rew_pct':round(abs(tp_p-price)/price*100,2),
           'htf':htf_b,'weekly':wb,'rsi_val':round(rsi_a[i]),
           'session':sess}

# ── IMPROVEMENT 5: TRADE MANAGEMENT ──────────
def check_open_trades(kl_map):
    """
    Monitor open trades and send SL management alerts:
    - TP1 hit → alert to move SL to breakeven
    - 75% to TP2 hit → alert to trail SL to TP1
    """
    for sym, trade in list(open_trades.items()):
        pair = next((p for p in PAIRS if p['sym']==sym), None)
        if not pair or sym not in kl_map: continue
        kl = kl_map[sym]
        if not kl: continue
        price = kl[-1]['c']
        is_buy = trade['dir']=='BUY'
        entry  = trade['entry']
        sl_p   = trade['sl']
        tp1    = trade['tp1']
        tp2    = trade['tp']
        tp3    = trade['tp3']

        # Check TP1 hit
        if not trade.get('tp1_hit'):
            if (is_buy and price>=tp1) or (not is_buy and price<=tp1):
                trade['tp1_hit'] = True
                trade['sl'] = entry  # move SL to breakeven
                send_tg(
                    f"🎯 <b>TP1 HIT — {sym}/USD</b>\n\n"
                    f"Setup: {trade.get('name','SMC')}\n"
                    f"✅ Close 50% of position at <code>{fp(price)}</code>\n"
                    f"🔒 <b>Move SL to breakeven: <code>{fp(entry)}</code></b>\n"
                    f"🎯 TP2 still open: <code>{fp(tp2)}</code>\n"
                    f"🏃 TP3 runner: <code>{fp(tp3)}</code>\n\n"
                    f"📡 <b>SMC Engine Pro</b>"
                )
                log.info(f"  {sym}: TP1 hit @ {fp(price)} — SL moved to BE")

        # Check 75% to TP2 (trail SL to TP1)
        elif not trade.get('tp2_trail'):
            dist_to_tp2 = abs(tp2-entry)
            dist_moved  = abs(price-entry)
            if dist_to_tp2>0 and dist_moved/dist_to_tp2>=0.75:
                trade['tp2_trail'] = True
                send_tg(
                    f"📈 <b>75% TO TP2 — {sym}/USD</b>\n\n"
                    f"Setup: {trade.get('name','SMC')}\n"
                    f"Price: <code>{fp(price)}</code>\n"
                    f"🔒 <b>Trail SL to TP1: <code>{fp(tp1)}</code></b>\n"
                    f"🎯 TP2 target: <code>{fp(tp2)}</code>\n\n"
                    f"📡 <b>SMC Engine Pro</b>"
                )
                log.info(f"  {sym}: 75% to TP2 — SL trailed to TP1")

        # Check TP2 hit (close remaining, let runner go)
        elif not trade.get('tp2_hit'):
            if (is_buy and price>=tp2) or (not is_buy and price<=tp2):
                trade['tp2_hit'] = True
                send_tg(
                    f"🏆 <b>TP2 HIT — {sym}/USD</b>\n\n"
                    f"Setup: {trade.get('name','SMC')}\n"
                    f"✅ Close another 25–40% at <code>{fp(price)}</code>\n"
                    f"🏃 <b>Let runner go to TP3: <code>{fp(tp3)}</code></b>\n"
                    f"🔒 Trail SL to TP2: <code>{fp(tp2)}</code>\n\n"
                    f"📡 <b>SMC Engine Pro</b>"
                )
                log.info(f"  {sym}: TP2 hit @ {fp(price)}")

        # Check SL hit (stop out)
        elif not trade.get('sl_hit'):
            if (is_buy and price<=trade['sl']) or (not is_buy and price>=trade['sl']):
                trade['sl_hit'] = True
                pnl = (price-entry)/entry*100 if is_buy else (entry-price)/entry*100
                send_tg(
                    f"🛑 <b>STOPPED OUT — {sym}/USD</b>\n\n"
                    f"Setup: {trade.get('name','SMC')}\n"
                    f"Exit: <code>{fp(price)}</code>  |  "
                    f"P&L: <b>{'+'if pnl>=0 else ''}{pnl:.2f}%</b>\n\n"
                    f"{'Breakeven — no loss 👍' if abs(pnl)<0.2 else 'Risk was managed ✓'}\n\n"
                    f"📡 <b>SMC Engine Pro</b>"
                )
                del open_trades[sym]
                log.info(f"  {sym}: stopped out @ {fp(price)} {pnl:+.2f}%")

# ── FORMATTING ────────────────────────────────
def fp(p):
    if not p: return '—'
    if p>=10000: return f'${p:,.0f}'
    if p>=100:   return f'${p:.2f}'
    if p>=1:     return f'${p:.3f}'
    return f'${p:.5f}'

def build_alert(sig):
    s=sig; is_buy=s['dir']=='BUY'
    tips={
        'SWEEP_OB':       f"Liq swept at {fp(s.get('sweep_lvl',s['price']))} — institutions filled. Confirmation candle confirmed reversal.",
        'HTF_CONFLUENCE': 'Weekly + Daily + 1h all aligned. Highest conviction setup.',
        'CHOCH':          'Structural shift confirmed. Clean structure, volume surge.',
        'BOS':            'Trend continuation confirmed. Clean HH/HL structure break.',
    }
    e={'SWEEP_OB':'⚡','HTF_CONFLUENCE':'📊','CHOCH':'🔄','BOS':'📈'}.get(s['setup'],'📡')
    sess = s.get('session','')
    limit_note = (f"\n  <i>Limit order at <code>{fp(s['entry_limit'])}</code> "
                  f"(50% pullback) for better entry</i>" if abs(s['entry_limit']-s['price'])/s['price']>0.001 else '')
    lines=[
        f"{'🟢' if is_buy else '🔴'} <b>{s['dir']} — {s['pair']['sym']}/USD</b>  "
        f"{'🇬🇧 London' if sess=='London' else '🇺🇸 New York' if sess=='New York' else ''}",
        f"{e} <b>Setup: {s['name']}</b>",
        f"📅 Weekly: {s.get('weekly','—')} | HTF: {s['htf']}",
        '',
        f"📌 <i>{tips.get(s['setup'],'SMC confluence.')}</i>",
        '',
        '💰 <b>Trade Levels</b>',
        f"  Market:  <code>{fp(s['price'])}</code>{limit_note}",
        f"  SL:      <code>{fp(s['sl'])}</code>  <i>(-{s['risk_pct']}%) ← structure based</i>",
        f"  TP1:     <code>{fp(s['tp1'])}</code>  <i>(1:2 — close 50%, move SL to BE)</i>",
        f"  TP2:     <code>{fp(s['tp'])}</code>   <i>(1:{s['rr']} — close 40%)</i>",
        f"  TP3:     <code>{fp(s['tp3'])}</code>  <i>(1:3 — runner, trail SL)</i>",
        '',
        f"📊 <b>Score: {s['score']}/10  |  Conf: {s['conf']}%  |  R:R 1:{s['rr']}</b>",
        f"  Tags:   {' · '.join(s['tags'])}",
        f"  RSI:    {s['rsi_val']}",
    ]
    if s.get('ob'):
        lines.append(f"  OB zone: {fp(s['ob']['bot'])} – {fp(s['ob']['top'])}")
    lines+=[
        '',
        '📋 <b>Trade Plan</b>',
        f"  1. Enter at market <code>{fp(s['price'])}</code>",
        f"  2. SL at <code>{fp(s['sl'])}</code>",
        f"  3. TP1 hit → close 50% + move SL to breakeven",
        f"  4. TP2 hit → close 40% + trail SL",
        f"  5. Let 10% runner go to TP3",
        '',
        '⚠️ <i>Not financial advice. Always manage risk.</i>',
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC  |  📡 <b>SMC Engine Pro v2</b>",
    ]
    return '\n'.join(lines)

# ── TELEGRAM ──────────────────────────────────
def send_tg(msg):
    if not TG_TOKEN or not TG_CHAT: return False
    try:
        r=requests.post(f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={'chat_id':TG_CHAT,'text':msg,'parse_mode':'HTML',
                  'disable_web_page_preview':True}, timeout=10)
        return r.ok
    except Exception as e:
        log.error(f"TG error: {e}"); return False

# ── SCAN LOOP ─────────────────────────────────
def run_scan():
    server_stats['scans'] += 1
    server_stats['last_scan'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    # IMPROVEMENT 1: Session filter
    if not in_session():
        h=datetime.now(timezone.utc).hour
        log.info(f"Outside session (UTC hour {h}) — skipping scan. Next: London 07:00 or NY 13:00")
        return

    sess = session_name()
    log.info(f"{'='*50}")
    log.info(f"Scan #{server_stats['scans']} | {sess} session | {server_stats['last_scan']}")

    # IMPROVEMENT 3: Refresh BTC bias first
    get_btc_bias(force_refresh=True)
    log.info(f"BTC bias: {btc_cache['bias']}")

    kl_map = {}
    for pair in PAIRS:
        try:
            kl = fetch_candles(pair, 300)
            kl_map[pair['sym']] = kl
            if not kl:
                log.info(f"  {pair['sym']}: no data"); continue

            sig = compute_signal(kl, pair)
            if sig:
                msg = build_alert(sig)
                ok  = send_tg(msg)
                if ok:
                    last_fired[pair['sym']] = {'setup':sig['setup'],'time':time.time()}
                    # Register as open trade for management
                    open_trades[pair['sym']] = {
                        'dir':sig['dir'],'entry':sig['price'],
                        'sl':sig['sl'],'tp':sig['tp'],
                        'tp1':sig['tp1'],'tp3':sig['tp3'],
                        'name':sig['name'],'time':time.time()
                    }
                    server_stats['alerts'] += 1
                    log.info(f"  ✓ {pair['sym']}: {sig['name']} {sig['dir']} "
                             f"score={sig['score']} → TG sent")
            else:
                log.info(f"  {pair['sym']}: no setup")
            time.sleep(1.5)
        except Exception as e:
            log.error(f"  {pair['sym']} error: {e}")

    # IMPROVEMENT 5: Check open trades for SL management
    if open_trades:
        log.info(f"Checking {len(open_trades)} open trades...")
        check_open_trades(kl_map)

    log.info(f"Scan done. Alerts this scan: {sum(1 for p in PAIRS if p['sym'] in last_fired and time.time()-last_fired[p['sym']]['time']<SCAN_EVERY*60)}")

# ── MAIN ──────────────────────────────────────
def main():
    if not TG_TOKEN or not TG_CHAT:
        log.error("Set TG_TOKEN and TG_CHAT environment variables")
        raise SystemExit(1)

    log.info("="*55)
    log.info("SMC ENGINE PRO v2 — IMPROVED STRATEGY")
    log.info(f"Improvements active:")
    log.info(f"  ✓ Session filter (London 07-12 + NY 13-18 UTC)")
    log.info(f"  ✓ Weekly bias hard gate")
    log.info(f"  ✓ BTC correlation gate")
    log.info(f"  ✓ Confirmation candle (Sweep+OB)")
    log.info(f"  ✓ Structure-based SL")
    log.info(f"  ✓ Pullback entry levels")
    log.info(f"  ✓ TP1/TP2 SL management alerts")
    log.info(f"Pairs: {len(PAIRS)} | Score≥{MIN_SCORE} | Every {SCAN_EVERY}m")
    log.info("="*55)

    # Health server
    threading.Thread(target=run_health,daemon=True).start()

    # Startup TG message
    send_tg(
        "✅ <b>SMC Engine Pro v2 — Started</b>\n\n"
        "<b>7 improvements active:</b>\n"
        "🕐 Session filter (London + NY only)\n"
        "📅 Weekly bias hard gate\n"
        "₿  BTC correlation gate\n"
        "🕯 Confirmation candle for Sweep+OB\n"
        "📐 Structure-based SL (not fixed ATR)\n"
        "🎯 Pullback entry levels\n"
        "🔒 TP1/TP2 SL management alerts\n\n"
        f"Scanning every {SCAN_EVERY} minutes during London + NY sessions\n"
        "You get alerted at entry AND at each TP level 📡\n\n"
        "📡 <b>SMC Engine Pro v2</b>"
    )
    log.info("✓ Startup message sent")

    while True:
        try:
            run_scan()
        except Exception as e:
            log.error(f"Scan error: {e}")
        log.info(f"Sleeping {SCAN_EVERY} minutes...")
        time.sleep(SCAN_EVERY*60)

if __name__=='__main__':
    main()