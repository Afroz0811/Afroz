"""
SMC Engine — 24/7 Alert Server
Run this on any cloud server (free tier works fine)
Sends Telegram alerts when setups fire — no browser needed

Setup:
  pip install requests schedule
  python3 smc_alert_server.py --token YOUR_BOT_TOKEN --chat YOUR_CHAT_ID

Free hosting options:
  - Railway.app     (free tier, always on)
  - Render.com      (free tier, always on)
  - PythonAnywhere  (free tier)
  - Oracle Cloud    (always free VM)
  - Replit          (free with always-on)
"""

import requests, time, json, argparse, logging, os
from datetime import datetime, timezone
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────
PAIRS = [
    {'sym':'BTC',  'name':'Bitcoin',    'kr':'XXBTZUSD', 'cg':'bitcoin'},
    {'sym':'ETH',  'name':'Ethereum',   'kr':'XETHZUSD', 'cg':'ethereum'},
    {'sym':'SOL',  'name':'Solana',     'kr':'SOLUSD',   'cg':'solana'},
    {'sym':'XRP',  'name':'XRP',        'kr':'XXRPZUSD', 'cg':'ripple'},
    {'sym':'ADA',  'name':'Cardano',    'kr':'ADAUSD',   'cg':'cardano'},
    {'sym':'DOGE', 'name':'Dogecoin',   'kr':'XDGUSD',   'cg':'dogecoin'},
    {'sym':'AVAX', 'name':'Avalanche',  'kr':'AVAXUSD',  'cg':'avalanche-2'},
    {'sym':'DOT',  'name':'Polkadot',   'kr':'DOTUSD',   'cg':'polkadot'},
    {'sym':'LINK', 'name':'Chainlink',  'kr':'LINKUSD',  'cg':'chainlink'},
    {'sym':'MATIC','name':'Polygon',    'kr':'MATICUSD', 'cg':'matic-network'},
]
INTERVAL    = 60    # minutes (1h candles)
SCAN_EVERY  = 5     # minutes between scans
MIN_SCORE   = 7
COOLDOWN_M  = 60    # minutes before same coin can re-alert

KR_BASE = 'https://api.kraken.com/0/public'
CG_BASE = 'https://api.coingecko.com/api/v3'

# ── FETCH CANDLES ──────────────────────────────
def fetch_candles(pair: dict, limit: int = 300) -> list:
    """Try Kraken first, fall back to CoinGecko"""
    # Kraken
    try:
        r = requests.get(
            f"{KR_BASE}/OHLC",
            params={'pair': pair['kr'], 'interval': INTERVAL},
            timeout=15
        )
        d = r.json()
        if not d.get('error'):
            key = next(k for k in d['result'] if k != 'last')
            raw = d['result'][key]
            if len(raw) > 20:
                return [{'t':i,'o':float(k[1]),'h':float(k[2]),
                         'l':float(k[3]),'c':float(k[4]),'v':float(k[6])}
                        for i, k in enumerate(raw[-limit:])]
    except Exception as e:
        log.debug(f"Kraken failed for {pair['sym']}: {e}")

    # CoinGecko fallback
    try:
        r = requests.get(
            f"{CG_BASE}/coins/{pair['cg']}/ohlc",
            params={'vs_currency': 'usd', 'days': 7},
            timeout=15
        )
        raw = r.json()
        if isinstance(raw, list) and len(raw) > 5:
            return [{'t':i,'o':float(k[1]),'h':float(k[2]),
                     'l':float(k[3]),'c':float(k[4]),'v':50.0}
                    for i, k in enumerate(raw[-limit:])]
    except Exception as e:
        log.debug(f"CoinGecko failed for {pair['sym']}: {e}")

    return []

def fetch_price(pair: dict) -> float:
    try:
        r = requests.get(
            f"{CG_BASE}/simple/price",
            params={'ids': pair['cg'], 'vs_currencies': 'usd'},
            timeout=10
        )
        return r.json()[pair['cg']]['usd']
    except:
        return 0.0

# ── INDICATORS ─────────────────────────────────
import numpy as np

def ema(c, p):
    k = 2/(p+1); r = [None]*(p-1); s = sum(c[:p])/p; r.append(s); pv = s
    for i in range(p, len(c)): pv = c[i]*k + pv*(1-k); r.append(pv)
    return r

def rsi(c, p=14):
    if len(c) < p+1: return [None]*len(c)
    r = [None]*p; g = l = 0
    for i in range(1, p+1):
        d = c[i]-c[i-1]
        if d > 0: g += d
        else: l += abs(d)
    ag, al = g/p, l/p
    r.append(100 if al==0 else 100-100/(1+ag/al))
    for i in range(p+1, len(c)):
        d = c[i]-c[i-1]; gi = d if d>0 else 0; li = abs(d) if d<0 else 0
        ag = (ag*(p-1)+gi)/p; al = (al*(p-1)+li)/p
        r.append(100 if al==0 else 100-100/(1+ag/al))
    return r

def macd_hist(c):
    e12=ema(c,12); e26=ema(c,26)
    ln=[e12[i]-e26[i] if e12[i] and e26[i] else None for i in range(len(c))]
    vl=[v for v in ln if v]; sr=ema(vl,9)
    sg=[None]*len(c); si=0
    for i in range(len(c)):
        if ln[i] is not None: sg[i]=sr[si] if si<len(sr) else None; si+=1
    return [ln[i]-sg[i] if ln[i] and sg[i] else None for i in range(len(c))]

def atr(kl, p=14):
    tr=[None]
    for i in range(1,len(kl)):
        t=max(kl[i]['h']-kl[i]['l'],abs(kl[i]['h']-kl[i-1]['c']),abs(kl[i]['l']-kl[i-1]['c']))
        tr.append(t)
    r=[None]*p; s=sum(tr[1:p+1])/p; r.append(s); pv=s
    for i in range(p+1,len(tr)): pv=(pv*(p-1)+tr[i])/p; r.append(pv)
    return r

def vavg(v, p=20):
    r=[None]*p
    for i in range(p,len(v)): r.append(sum(v[i-p:i])/p)
    return r

# ── SMC SETUPS ─────────────────────────────────
def find_swings(kl, lb=5):
    sh=[]; sl=[]
    for i in range(lb, len(kl)-lb):
        if all(kl[i]['h']>=kl[j]['h'] for j in range(i-lb,i+lb+1) if j!=i): sh.append((i,kl[i]['h']))
        if all(kl[i]['l']<=kl[j]['l'] for j in range(i-lb,i+lb+1) if j!=i): sl.append((i,kl[i]['l']))
    return sh, sl

def htf_bias(kl, factor=5):
    if len(kl) < factor*25: return 'neutral'
    htf = [kl[i*factor+factor-1]['c'] for i in range(len(kl)//factor)]
    e20=ema(htf,20); e50=ema(htf,50); n=len(htf)-1
    if not e20[n] or not e50[n]: return 'neutral'
    if htf[n]>e20[n]>e50[n]: return 'bullish'
    if htf[n]<e20[n]<e50[n]: return 'bearish'
    return 'neutral'

def is_choppy(atr_a, i, thresh=0.40):
    recent = [a for a in atr_a[max(0,i-20):i] if a]
    if len(recent) < 5: return True
    return atr_a[i] < np.mean(recent)*thresh

def detect_sweep_ob(kl, sh, sl, i, atr_v, va_v, rsi_v, e20_v, e50_v, htf_b):
    if i<15 or not atr_v or not va_v: return None
    k=kl[i]; price=k['c']
    # Bullish
    r_lows=[(idx,p) for idx,p in sl if idx<i-1 and idx>i-50][-5:]
    for li, lvl in r_lows:
        if not(k['l']<lvl<price): continue
        if lvl-k['l']<atr_v*0.30: continue
        if k['v']<va_v*1.20: continue
        if htf_b!='bullish': continue
        if not rsi_v or not(25<rsi_v<62): continue
        ob=None
        for j in range(li-1, max(0,li-12), -1):
            if kl[j]['c']<kl[j]['o']:
                fwd=(kl[min(j+2,len(kl)-1)]['c']-kl[j]['c'])/kl[j]['c']
                if fwd>0.003: ob={'top':kl[j]['o'],'bot':kl[j]['l']}; break
        if not ob or not(ob['bot']<=price<=ob['top']*1.005): continue
        ema_ok=e20_v and e50_v and price>e20_v>e50_v
        return{'dir':'BUY','setup':'SWEEP_OB','name':'⚡ Liq Sweep + OB Retest',
               'score':8+(0.5 if ema_ok else 0),'ob':ob,'sweep':lvl,
               'tags':['Sweep↑','OB_Retest','Vol✓','HTF✓']+(['EMA↑'] if ema_ok else [])+[f'RSI{round(rsi_v)}']}
    # Bearish
    r_highs=[(idx,p) for idx,p in sh if idx<i-1 and idx>i-50][-5:]
    for hi_, lvl in r_highs:
        if not(k['h']>lvl>price): continue
        if k['h']-lvl<atr_v*0.30: continue
        if k['v']<va_v*1.20: continue
        if htf_b!='bearish': continue
        if not rsi_v or not(38<rsi_v<75): continue
        ob=None
        for j in range(hi_-1, max(0,hi_-12), -1):
            if kl[j]['c']>kl[j]['o']:
                fwd=(kl[min(j+2,len(kl)-1)]['c']-kl[j]['c'])/kl[j]['c']
                if fwd<-0.003: ob={'top':kl[j]['h'],'bot':kl[j]['c']}; break
        if not ob or not(ob['bot']*0.995<=price<=ob['top']): continue
        ema_ok=e20_v and e50_v and price<e20_v<e50_v
        return{'dir':'SELL','setup':'SWEEP_OB','name':'⚡ Liq Sweep + OB Retest',
               'score':8+(0.5 if ema_ok else 0),'ob':ob,'sweep':lvl,
               'tags':['Sweep↓','OB_Retest','Vol✓','HTF✓']+(['EMA↓'] if ema_ok else [])+[f'RSI{round(rsi_v)}']}
    return None

def detect_htf_confluence(kl, sh, sl, i, closes, rsi_v, e9_v, e20_v, e50_v, ht_v, va_v):
    if i<50 or not ht_v: return None
    price=closes[i]
    wb=htf_bias(kl[:i+1],21); db=htf_bias(kl[:i+1],5)
    if wb=='neutral' or db=='neutral' or wb!=db: return None
    rh=[(idx,p) for idx,p in sh if idx<=i][-4:]
    rl=[(idx,p) for idx,p in sl if idx<=i][-4:]
    h1='neutral'
    if len(rh)>=2 and len(rl)>=2:
        if rh[-1][1]>rh[-2][1] and rl[-1][1]>rl[-2][1]: h1='bullish'
        elif rh[-1][1]<rh[-2][1] and rl[-1][1]<rl[-2][1]: h1='bearish'
    if h1!=wb: return None
    is_buy=h1=='bullish'
    if is_buy and not(e9_v and e20_v and e50_v and e9_v>e20_v>e50_v): return None
    if not is_buy and not(e9_v and e20_v and e50_v and e9_v<e20_v<e50_v): return None
    if is_buy and ht_v<=0: return None
    if not is_buy and ht_v>=0: return None
    if is_buy and not(rsi_v and 25<rsi_v<62): return None
    if not is_buy and not(rsi_v and 38<rsi_v<75): return None
    vol_ok=va_v and kl[i]['v']>va_v*1.1
    return{'dir':'BUY' if is_buy else 'SELL','setup':'HTF_CONFLUENCE',
           'name':'📊 3-TF HTF Confluence','score':8+(0.5 if vol_ok else 0),
           'tags':[f'W:{wb[:4]}',f'D:{db[:4]}',f'1h:{h1[:4]}','EMA_stack','MACD✓']+(['Vol✓'] if vol_ok else [])+[f'RSI{round(rsi_v)}']}

def detect_choch(kl, sh, sl, i, closes, rsi_v, e20_v, e50_v, ht_v, va_v):
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
    if h2<h1p and l2<l1p and price>h2 and e20_v and price>e20_v and ht_v>0 and rsi_v and 28<rsi_v<65 and vol_ok:
        return{'dir':'BUY','setup':'CHOCH','name':'🔄 CHoCH Reversal (Bear→Bull)',
               'score':8,'tags':['CHoCH↑','CleanStr','Vol✓','MACD✓',f'RSI{round(rsi_v)}']}
    if h2>h1p and l2>l1p and price<l2 and e20_v and price<e20_v and ht_v<0 and rsi_v and 35<rsi_v<72 and vol_ok:
        return{'dir':'SELL','setup':'CHOCH','name':'🔄 CHoCH Reversal (Bull→Bear)',
               'score':8,'tags':['CHoCH↓','CleanStr','Vol✓','MACD✓',f'RSI{round(rsi_v)}']}
    return None

def detect_bos(kl, sh, sl, i, closes, rsi_v, e20_v, e50_v, ht_v, va_v):
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
    if h3p>h2p>h1p and l3p>l2p and price>h2p and ht_v>0 and e20_v and e50_v and price>e20_v>e50_v and rsi_v and 28<rsi_v<68 and vol_ok:
        return{'dir':'BUY','setup':'BOS','name':'📈 BOS Continuation (Bullish)',
               'score':7,'tags':['BOS↑','HH+HL','Vol✓','MACD✓',f'RSI{round(rsi_v)}']}
    if h3p<h2p<h1p and l3p<l2p and price<l2p and ht_v<0 and e20_v and e50_v and price<e20_v<e50_v and rsi_v and 32<rsi_v<72 and vol_ok:
        return{'dir':'SELL','setup':'BOS','name':'📉 BOS Continuation (Bearish)',
               'score':7,'tags':['BOS↓','LH+LL','Vol✓','MACD✓',f'RSI{round(rsi_v)}']}
    return None

# ── MAIN SIGNAL ENGINE ─────────────────────────
last_fired = {}  # sym -> {setup, time}

def compute_signal(kl: list, pair: dict) -> Optional[dict]:
    if len(kl) < 80: return None
    n=len(kl); i=n-1
    closes=[k['c'] for k in kl]; vols=[k['v'] for k in kl]
    rsi_a=rsi(closes); e9_a=ema(closes,9); e20_a=ema(closes,20); e50_a=ema(closes,50)
    ht_a=macd_hist(closes); atr_a=atr(kl); va_a=vavg(vols)
    price=closes[i]
    if any(x is None for x in [rsi_a[i],e9_a[i],e20_a[i],e50_a[i],atr_a[i],va_a[i]]): return None
    if is_choppy(atr_a, i): return None
    if atr_a[i]/price < 0.002: return None
    # Cooldown check
    last=last_fired.get(pair['sym'])
    if last:
        mins_since=(time.time()-last['time'])/60
        if mins_since < COOLDOWN_M: return None
    sh, sl = find_swings(kl, 5)
    htf_b = htf_bias(kl, 5)
    sig = (detect_sweep_ob(kl,sh,sl,i,atr_a[i],va_a[i],rsi_a[i],e20_a[i],e50_a[i],htf_b) or
           detect_htf_confluence(kl,sh,sl,i,closes,rsi_a[i],e9_a[i],e20_a[i],e50_a[i],ht_a[i],va_a[i]) or
           detect_choch(kl,sh,sl,i,closes,rsi_a[i],e20_a[i],e50_a[i],ht_a[i],va_a[i]) or
           detect_bos(kl,sh,sl,i,closes,rsi_a[i],e20_a[i],e50_a[i],ht_a[i],va_a[i]))
    if not sig or sig['score'] < MIN_SCORE: return None
    is_buy = sig['dir']=='BUY'
    rh=[(idx,p) for idx,p in sh if idx<=i]; rl=[(idx,p) for idx,p in sl if idx<=i]
    last_l=rl[-1][1] if rl else price-atr_a[i]*2
    last_h=rh[-1][1] if rh else price+atr_a[i]*2
    sl_p=(min(last_l,price-atr_a[i]*1.5)*0.997) if is_buy else (max(last_h,price+atr_a[i]*1.5)*1.003)
    if sig.get('ob'):
        if is_buy: sl_p=min(sl_p,sig['ob']['bot']*0.997)
        else: sl_p=max(sl_p,sig['ob']['top']*1.003)
    risk=abs(price-sl_p)
    if risk<=0: return None
    rr_mult=3.0 if sig['setup']=='SWEEP_OB' else 2.5
    tp_p=price+risk*rr_mult if is_buy else price-risk*rr_mult
    rr=abs(tp_p-price)/risk
    if rr<2.0: return None
    tp1=price+risk*2 if is_buy else price-risk*2
    conf=min(97,int(sig['score']*8.5+min(rr,3)*2.5))
    return{**sig,'pair':pair,'price':price,'sl':sl_p,'tp':tp_p,'tp1':tp1,
           'rr':round(rr,2),'conf':conf,
           'risk_pct':round(abs(price-sl_p)/price*100,2),
           'rew_pct':round(abs(tp_p-price)/price*100,2),
           'htf':htf_b,'rsi':round(rsi_a[i])}

# ── FORMAT ─────────────────────────────────────
def fp(p):
    if not p: return '—'
    if p>=10000: return f'${p:,.0f}'
    if p>=100:   return f'${p:.2f}'
    if p>=1:     return f'${p:.3f}'
    return f'${p:.5f}'

# ── TELEGRAM ───────────────────────────────────
def send_tg(token: str, chat_id: str, msg: str) -> bool:
    try:
        r = requests.post(
            f'https://api.telegram.org/bot{token}/sendMessage',
            json={'chat_id':chat_id,'text':msg,'parse_mode':'HTML',
                  'disable_web_page_preview':True},
            timeout=10
        )
        return r.ok
    except Exception as e:
        log.error(f"Telegram error: {e}")
        return False

def build_alert(sig: dict) -> str:
    s=sig; is_buy=s['dir']=='BUY'
    p=s['pair']; price=s['price']
    risk=abs(price-s['sl']); tp3=price+risk*3 if is_buy else price-risk*3
    tips={'SWEEP_OB':f"Liq swept at {fp(s.get('sweep',price))} — institutions filled. OB retest entry.",
          'HTF_CONFLUENCE':'Weekly+Daily+1h all aligned. High-conviction trend continuation.',
          'CHOCH':'Structural shift detected. Early reversal, tight SL.',
          'BOS':'Clean structure break. Trend continuation confirmed.'}
    return '\n'.join(filter(None,[
        f"{'🟢' if is_buy else '🔴'} <b>{s['dir']} — {p['sym']}/USD</b>",
        f"{'⚡📊🔄📈'.split()[['SWEEP_OB','HTF_CONFLUENCE','CHOCH','BOS'].index(s['setup'])]} <b>Setup: {s['name']}</b>",
        '',
        f"📌 <i>{tips.get(s['setup'],'SMC confluence setup.')}</i>",
        '',
        '💰 <b>Trade Levels</b>',
        f"  Entry:  <code>{fp(price)}</code>",
        f"  SL:     <code>{fp(s['sl'])}</code>  <i>(-{s['risk_pct']}%)</i>",
        f"  TP1:    <code>{fp(s['tp1'])}</code>  <i>(1:2 — partial close)</i>",
        f"  TP2:    <code>{fp(s['tp'])}</code>  <i>(1:{s['rr']} — main target)</i>",
        f"  TP3:    <code>{fp(tp3)}</code>  <i>(1:3 — runner)</i>",
        '',
        f"📊 <b>Score: {s['score']}/10  |  Conf: {s['conf']}%  |  R:R 1:{s['rr']}</b>",
        f"  Tags:  {' · '.join(s['tags'])}",
        f"  HTF:   {s['htf']}  |  RSI: {s['rsi']}",
        f"  OB Zone: {fp(s['ob']['bot'])} – {fp(s['ob']['top'])}" if s.get('ob') else '',
        '',
        '⚠️ <i>Not financial advice. Always manage risk.</i>',
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC  |  📡 <b>SMC Engine Pro</b>",
    ]))

# ── SCAN LOOP ──────────────────────────────────
def scan(token: str, chat_id: str):
    log.info(f"Scanning {len(PAIRS)} pairs...")
    fired = 0
    for pair in PAIRS:
        try:
            kl = fetch_candles(pair, limit=300)
            if not kl:
                log.warning(f"  {pair['sym']}: no candles")
                continue
            sig = compute_signal(kl, pair)
            if sig:
                msg = build_alert(sig)
                ok = send_tg(token, chat_id, msg)
                if ok:
                    last_fired[pair['sym']] = {'setup':sig['setup'],'time':time.time()}
                    log.info(f"  ✓ {pair['sym']}: {sig['name']} {sig['dir']} score={sig['score']} sent to TG")
                    fired += 1
                else:
                    log.error(f"  {pair['sym']}: TG send failed")
            else:
                log.info(f"  {pair['sym']}: no setup")
            time.sleep(1.2)  # rate limit
        except Exception as e:
            log.error(f"  {pair['sym']}: error — {e}")
    log.info(f"Scan complete. {fired} alerts sent.")
    return fired

def main():
    parser = argparse.ArgumentParser(description='SMC Engine 24/7 Alert Server')
    parser.add_argument('--token',    required=True,  help='Telegram bot token from @BotFather')
    parser.add_argument('--chat',     required=True,  help='Telegram chat ID from @userinfobot')
    parser.add_argument('--interval', type=int, default=SCAN_EVERY, help='Scan every N minutes (default 5)')
    parser.add_argument('--score',    type=int, default=MIN_SCORE,  help='Min score to alert (default 7)')
    parser.add_argument('--once',     action='store_true', help='Run once and exit')
    args = parser.parse_args()

    global MIN_SCORE, SCAN_EVERY
    MIN_SCORE  = args.score
    SCAN_EVERY = args.interval

    log.info("="*55)
    log.info("SMC ENGINE 24/7 ALERT SERVER")
    log.info(f"Pairs: {len(PAIRS)} | Score≥{MIN_SCORE} | Scan every {SCAN_EVERY}m")
    log.info(f"Cooldown: {COOLDOWN_M}m per coin | RR≥2.0")
    log.info("="*55)

    # Test TG connection
    ok = send_tg(args.token, args.chat,
        "✅ <b>SMC Engine Server Started</b>\n\n"
        "🔍 Scanning 10 pairs every 5 minutes\n"
        "⚡ Setups: Sweep+OB · HTF · CHoCH · BOS\n"
        "📊 Alerts fire when Score ≥ 7\n\n"
        "You'll get alerts here 24/7 — no browser needed.\n"
        "📡 <b>SMC Engine Pro</b>")
    if ok:
        log.info("✓ Telegram connected — startup message sent")
    else:
        log.error("✗ Telegram connection failed — check token and chat ID")
        return

    if args.once:
        scan(args.token, args.chat)
        return

    # Run forever
    log.info(f"Starting scan loop (every {SCAN_EVERY} minutes)...")
    while True:
        try:
            scan(args.token, args.chat)
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break
        except Exception as e:
            log.error(f"Scan error: {e}")
        log.info(f"Next scan in {SCAN_EVERY} minutes...")
        time.sleep(SCAN_EVERY * 60)

if __name__ == '__main__':
    main()
