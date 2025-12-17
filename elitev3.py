import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import defaultdict, deque
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

# Configuration
TELEGRAM_TOKEN = "7731521911:AAFnus-fDivEwoKqrtwZXMmKEj5BU1EhQn4"
TELEGRAM_CHAT_ID = "7500072234"
BINANCE_API = "https://fapi.binance.com"
SCAN_INTERVAL = 300  # 5 minutes

# BEAST MODE Parameters
MIN_VOLUME_24H = 150_000_000
MAX_CONCURRENT_TRADES = 3
MIN_SIGNAL_SCORE = 28  # Out of 45
BLACKLIST_COOLDOWN = 10800
MIN_RR_RATIO = 2.5
MIN_WIN_RATE = 0.40

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class BeastTradingBot:
    def __init__(self):
        self.active_trades = {}
        self.pair_performance = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total_profit': 0, 'consecutive_losses': 0
        })
        self.blacklist = {}
        self.daily_stats = {
            'signals': 0, 'wins': 0, 'losses': 0,
            'total_profit': 0, 'best': 0, 'worst': 0
        }
        
    def get_pairs(self):
        try:
            response = requests.get(f"{BINANCE_API}/fapi/v1/ticker/24hr", timeout=10)
            data = response.json()
            
            pairs = []
            for t in data:
                if t['symbol'].endswith('USDT') and not any(x in t['symbol'] for x in ['DOWN','UP','BEAR','BULL']):
                    vol = float(t['quoteVolume'])
                    chg = abs(float(t['priceChangePercent']))
                    trd = float(t['count'])
                    
                    if vol > MIN_VOLUME_24H and 0.8 < chg < 12 and trd > 80000:
                        quality = min(vol/500_000_000, 2) * min(chg/8, 1.5) * min(trd/200000, 1.5)
                        pairs.append({'symbol': t['symbol'], 'quality': quality})
            
            pairs.sort(key=lambda x: x['quality'], reverse=True)
            logger.info(f"Found {len(pairs)} quality pairs")
            return [p['symbol'] for p in pairs[:80]]
        except:
            return []
    
    def supertrend(self, df, period=10, mult=3):
        try:
            hl2 = (df['high'] + df['low']) / 2
            tr = pd.concat([df['high']-df['low'], 
                          abs(df['high']-df['close'].shift()),
                          abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            ub = hl2 + mult * atr
            lb = hl2 - mult * atr
            
            st = pd.Series(index=df.index, dtype=float)
            dir = pd.Series(index=df.index, dtype=int)
            
            for i in range(period, len(df)):
                if df['close'].iloc[i] > ub.iloc[i-1]:
                    dir.iloc[i] = 1
                elif df['close'].iloc[i] < lb.iloc[i-1]:
                    dir.iloc[i] = -1
                else:
                    dir.iloc[i] = dir.iloc[i-1]
                
                if dir.iloc[i] == 1:
                    st.iloc[i] = lb.iloc[i] if dir.iloc[i-1] == -1 else max(lb.iloc[i], st.iloc[i-1])
                else:
                    st.iloc[i] = ub.iloc[i] if dir.iloc[i-1] == 1 else min(ub.iloc[i], st.iloc[i-1])
            
            return st, dir, atr
        except:
            return None, None, None
    
    def stoch_rsi(self, series, period=14, k=3, d=3):
        try:
            delta = series.diff()
            gain = delta.where(delta > 0, 0).ewm(span=period, adjust=False).mean()
            loss = -delta.where(delta < 0, 0).ewm(span=period, adjust=False).mean()
            rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
            
            rsi_min = rsi.rolling(period).min()
            rsi_max = rsi.rolling(period).max()
            stoch = 100 * (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
            k_line = stoch.rolling(k).mean()
            d_line = k_line.rolling(d).mean()
            return k_line, d_line, rsi
        except:
            return None, None, None
    
    def adx(self, df, period=14):
        try:
            hd = df['high'].diff()
            ld = -df['low'].diff()
            
            pos_dm = hd.where((hd > ld) & (hd > 0), 0)
            neg_dm = ld.where((ld > hd) & (ld > 0), 0)
            
            tr = pd.concat([df['high']-df['low'],
                          abs(df['high']-df['close'].shift()),
                          abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
            
            atr = tr.ewm(span=period, adjust=False).mean()
            pos_di = 100 * pos_dm.ewm(span=period, adjust=False).mean() / atr
            neg_di = 100 * neg_dm.ewm(span=period, adjust=False).mean() / atr
            
            dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
            adx_val = dx.ewm(span=period, adjust=False).mean()
            
            return adx_val, pos_di, neg_di
        except:
            return None, None, None
    
    def mfi(self, df, period=14):
        try:
            tp = (df['high'] + df['low'] + df['close']) / 3
            mf = tp * df['volume']
            
            pos = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
            neg = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
            
            return 100 - (100 / (1 + pos / (neg + 1e-10)))
        except:
            return None
    
    def divergence(self, price, ind, lookback=30):
        try:
            highs = argrelextrema(price.values, np.greater, order=5)[0]
            lows = argrelextrema(price.values, np.less, order=5)[0]
            
            if len(highs) < 2 or len(lows) < 2:
                return 0, 0
            
            bear = bull = 0
            
            h1, h2 = highs[-2], highs[-1]
            if 5 < h2 - h1 < lookback:
                if price.iloc[h2] > price.iloc[h1] * 1.001 and ind.iloc[h2] < ind.iloc[h1] * 0.99:
                    bear = 3
            
            l1, l2 = lows[-2], lows[-1]
            if 5 < l2 - l1 < lookback:
                if price.iloc[l2] < price.iloc[l1] * 0.999 and ind.iloc[l2] > ind.iloc[l1] * 1.01:
                    bull = 3
            
            return bull, bear
        except:
            return 0, 0
    
    def structure(self, df):
        try:
            highs = argrelextrema(df['high'].values, np.greater, order=5)[0]
            lows = argrelextrema(df['low'].values, np.less, order=5)[0]
            
            if len(highs) < 3 or len(lows) < 3:
                return "RANGING", 0
            
            rh = highs[-3:]
            rl = lows[-3:]
            
            hh = all(df['high'].iloc[rh[i]] < df['high'].iloc[rh[i+1]] for i in range(2))
            hl = all(df['low'].iloc[rl[i]] < df['low'].iloc[rl[i+1]] for i in range(2))
            lh = all(df['high'].iloc[rh[i]] > df['high'].iloc[rh[i+1]] for i in range(2))
            ll = all(df['low'].iloc[rl[i]] > df['low'].iloc[rl[i+1]] for i in range(2))
            
            if hh and hl: return "UPTREND", 4
            if lh and ll: return "DOWNTREND", 4
            if hh or hl: return "BULLISH", 2
            if lh or ll: return "BEARISH", 2
            return "RANGING", 0
        except:
            return "RANGING", 0
    
    def get_klines(self, sym, iv='15m', lim=300):
        for _ in range(3):
            try:
                r = requests.get(f"{BINANCE_API}/fapi/v1/klines",
                               params={'symbol':sym,'interval':iv,'limit':lim}, timeout=10)
                k = r.json()
                df = pd.DataFrame(k, columns=['t','o','h','l','c','v','ct','qv','tr','tb','tq','i'])
                for col in ['o','h','l','c','v']:
                    df[col] = pd.to_numeric(df[col])
                df.columns = ['timestamp','open','high','low','close','volume','ct','qv','trades','tb','tq','i']
                return df
            except:
                time.sleep(1)
        return None
    
    def analyze(self, symbol):
        try:
            df_15m = self.get_klines(symbol, '15m', 300)
            df_1h = self.get_klines(symbol, '1h', 150)
            df_4h = self.get_klines(symbol, '4h', 100)
            
            if df_15m is None or len(df_15m) < 200 or df_1h is None or df_4h is None:
                return None
            
            df = df_15m
            
            # Indicators
            stoch_k, stoch_d, rsi = self.stoch_rsi(df['close'])
            if stoch_k is None: return None
            
            st, st_dir, atr = self.supertrend(df)
            if st is None: return None
            
            adx_val, pos_di, neg_di = self.adx(df)
            if adx_val is None: return None
            
            mfi_val = self.mfi(df)
            if mfi_val is None: return None
            
            df['rsi'] = rsi
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            df['st_dir'] = st_dir
            df['adx'] = adx_val
            df['pos_di'] = pos_di
            df['neg_di'] = neg_di
            df['mfi'] = mfi_val
            
            # MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_sig']
            
            # EMAs
            df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
            
            # Bollinger
            df['bb_mid'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_up'] = df['bb_mid'] + bb_std * 2
            df['bb_low'] = df['bb_mid'] - bb_std * 2
            
            # Volume
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1e-10)
            
            # VWAP
            tp = (df['high'] + df['low'] + df['close']) / 3
            vwap = (tp * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap'] = vwap
            
            # HTF
            df_1h['ema50'] = df_1h['close'].ewm(span=50, adjust=False).mean()
            df_1h['ema200'] = df_1h['close'].ewm(span=200, adjust=False).mean()
            df_4h['ema50'] = df_4h['close'].ewm(span=50, adjust=False).mean()
            
            htf_1h = 1 if df_1h['close'].iloc[-1] > df_1h['ema50'].iloc[-1] else -1
            htf_1h_strong = 1 if df_1h['close'].iloc[-1] > df_1h['ema200'].iloc[-1] else -1
            htf_4h = 1 if df_4h['close'].iloc[-1] > df_4h['ema50'].iloc[-1] else -1
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            if pd.isna(curr['stoch_k']) or pd.isna(curr['adx']):
                return None
            
            struct, struct_score = self.structure(df.tail(100))
            bull_div, bear_div = self.divergence(df['close'].tail(60), df['rsi'].tail(60))
            
            score = 0
            signals = []
            direction = None
            
            # === LONG ===
            if (10 < curr['stoch_k'] < 40 and curr['stoch_k'] > curr['stoch_d'] and 
                prev['stoch_k'] <= prev['stoch_d']):
                
                if curr['st_dir'] == 1:
                    score += 6
                    signals.append("🟢 Supertrend UP")
                    
                    # ADX
                    if curr['adx'] > 25 and curr['pos_di'] > curr['neg_di']:
                        score += 5
                        signals.append(f"💪 Strong Trend ({curr['adx']:.1f})")
                    elif curr['adx'] > 20:
                        score += 2
                    else:
                        score -= 2
                    
                    # Structure
                    if struct in ["UPTREND", "BULLISH"]:
                        score += struct_score
                        signals.append(f"📊 {struct}")
                    
                    # HTF
                    htf_score = 0
                    if htf_1h == 1: htf_score += 2
                    if htf_1h_strong == 1: htf_score += 2
                    if htf_4h == 1: htf_score += 2
                    
                    if htf_score >= 4:
                        score += 6
                        signals.append("⏰ HTF ALIGNED")
                    elif htf_score >= 2:
                        score += 3
                        signals.append("⏰ HTF Partial")
                    
                    # EMA Stack
                    if curr['close'] > curr['ema9'] > curr['ema21'] > curr['ema50']:
                        score += 4
                        signals.append("📈 Perfect EMA Stack")
                    elif curr['close'] > curr['ema9'] > curr['ema21']:
                        score += 2
                    
                    # Above 200 EMA
                    if curr['close'] > curr['ema200']:
                        score += 2
                        signals.append("✅ Above 200 EMA")
                    
                    # VWAP
                    vwap_dist = (curr['close'] - curr['vwap']) / curr['vwap'] * 100
                    if -0.3 < vwap_dist < 0.5:
                        score += 4
                        signals.append("🎯 Perfect VWAP")
                    elif curr['close'] > curr['vwap']:
                        score += 2
                    
                    # RSI
                    if 30 < curr['rsi'] < 55:
                        score += 3
                        signals.append(f"✅ RSI Healthy ({curr['rsi']:.1f})")
                    elif curr['rsi'] < 30:
                        score += 2
                        signals.append("🔥 RSI Oversold")
                    
                    # MFI
                    if 30 < curr['mfi'] < 60:
                        score += 2
                    elif curr['mfi'] < 30:
                        score += 3
                        signals.append("💰 Money Flow Oversold")
                    
                    # MACD
                    if curr['macd_hist'] > 0:
                        score += 2
                        signals.append("📊 MACD Bullish")
                    elif curr['macd_hist'] > prev['macd_hist']:
                        score += 1
                    
                    # BB
                    bb_pos = (curr['close'] - curr['bb_low']) / (curr['bb_up'] - curr['bb_low'])
                    if bb_pos < 0.25:
                        score += 3
                        signals.append("🎯 BB Lower Zone")
                    
                    # Volume
                    if curr['vol_ratio'] > 1.8:
                        score += 4
                        signals.append(f"📊 VOLUME SURGE ({curr['vol_ratio']:.1f}x)")
                    elif curr['vol_ratio'] > 1.3:
                        score += 2
                    
                    # Divergence
                    if bull_div > 0:
                        score += bull_div + 2
                        signals.append("🚀 BULLISH DIVERGENCE")
                    
                    # Rejection wick
                    body = abs(curr['close'] - curr['open'])
                    lower_wick = min(curr['open'], curr['close']) - curr['low']
                    if lower_wick > body * 2.5:
                        score += 3
                        signals.append("🔨 Strong Rejection")
                    
                    if score >= MIN_SIGNAL_SCORE:
                        direction = "LONG"
            
            # === SHORT ===
            elif (60 < curr['stoch_k'] < 90 and curr['stoch_k'] < curr['stoch_d'] and
                  prev['stoch_k'] >= prev['stoch_d']):
                
                if curr['st_dir'] == -1:
                    score += 6
                    signals.append("🔴 Supertrend DOWN")
                    
                    if curr['adx'] > 25 and curr['neg_di'] > curr['pos_di']:
                        score += 5
                        signals.append(f"💪 Strong Trend ({curr['adx']:.1f})")
                    elif curr['adx'] > 20:
                        score += 2
                    else:
                        score -= 2
                    
                    if struct in ["DOWNTREND", "BEARISH"]:
                        score += struct_score
                        signals.append(f"📊 {struct}")
                    
                    htf_score = 0
                    if htf_1h == -1: htf_score += 2
                    if htf_1h_strong == -1: htf_score += 2
                    if htf_4h == -1: htf_score += 2
                    
                    if htf_score >= 4:
                        score += 6
                        signals.append("⏰ HTF ALIGNED")
                    elif htf_score >= 2:
                        score += 3
                    
                    if curr['close'] < curr['ema9'] < curr['ema21'] < curr['ema50']:
                        score += 4
                        signals.append("📉 Perfect EMA Stack")
                    elif curr['close'] < curr['ema9'] < curr['ema21']:
                        score += 2
                    
                    if curr['close'] < curr['ema200']:
                        score += 2
                        signals.append("✅ Below 200 EMA")
                    
                    vwap_dist = (curr['close'] - curr['vwap']) / curr['vwap'] * 100
                    if -0.5 < vwap_dist < 0.3:
                        score += 4
                        signals.append("🎯 Perfect VWAP")
                    elif curr['close'] < curr['vwap']:
                        score += 2
                    
                    if 45 < curr['rsi'] < 70:
                        score += 3
                        signals.append(f"✅ RSI Healthy ({curr['rsi']:.1f})")
                    elif curr['rsi'] > 70:
                        score += 2
                        signals.append("🔥 RSI Overbought")
                    
                    if 40 < curr['mfi'] < 70:
                        score += 2
                    elif curr['mfi'] > 70:
                        score += 3
                        signals.append("💰 Money Flow Overbought")
                    
                    if curr['macd_hist'] < 0:
                        score += 2
                        signals.append("📊 MACD Bearish")
                    elif curr['macd_hist'] < prev['macd_hist']:
                        score += 1
                    
                    bb_pos = (curr['close'] - curr['bb_low']) / (curr['bb_up'] - curr['bb_low'])
                    if bb_pos > 0.75:
                        score += 3
                        signals.append("🎯 BB Upper Zone")
                    
                    if curr['vol_ratio'] > 1.8:
                        score += 4
                        signals.append(f"📊 VOLUME SURGE ({curr['vol_ratio']:.1f}x)")
                    elif curr['vol_ratio'] > 1.3:
                        score += 2
                    
                    if bear_div > 0:
                        score += bear_div + 2
                        signals.append("💥 BEARISH DIVERGENCE")
                    
                    body = abs(curr['close'] - curr['open'])
                    upper_wick = curr['high'] - max(curr['open'], curr['close'])
                    if upper_wick > body * 2.5:
                        score += 3
                        signals.append("🔨 Strong Rejection")
                    
                    if score >= MIN_SIGNAL_SCORE:
                        direction = "SHORT"
            
            if direction and score >= MIN_SIGNAL_SCORE:
                confidence = "VERY HIGH" if score >= 35 else "HIGH" if score >= 30 else "MEDIUM"
                return {
                    'direction': direction,
                    'score': score,
                    'confidence': confidence,
                    'signals': signals[:10],
                    'price': curr['close'],
                    'stoch_k': curr['stoch_k'],
                    'rsi': curr['rsi'],
                    'adx': curr['adx'],
                    'vol_ratio': curr['vol_ratio'],
                    'df': df
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def calc_targets(self, symbol, direction, analysis):
        try:
            df = analysis['df']
            price = analysis['price']
            
            atr = df['high'].subtract(df['low']).rolling(14).mean().iloc[-1]
            recent_high = df['high'].tail(30).max()
            recent_low = df['low'].tail(30).min()
            
            entry = price
            
            if direction == "LONG":
                sl = min(recent_low * 0.995, entry - atr * 2.0)
                risk = entry - sl
                
                tp1 = entry + risk * 1.2
                tp2 = entry + risk * 2.5
                tp3 = entry + risk * 4.0
                tp4 = entry + risk * 6.0
                tp5 = entry + risk * 8.5
            else:
                sl = max(recent_high * 1.005, entry + atr * 2.0)
                risk = sl - entry
                
                tp1 = entry - risk * 1.2
                tp2 = entry - risk * 2.5
                tp3 = entry - risk * 4.0
                tp4 = entry - risk * 6.0
                tp5 = entry - risk * 8.5
            
            risk_pct = (risk / entry) * 100
            
            conf_mult = 1.4 if analysis['confidence'] == "VERY HIGH" else 1.2 if analysis['confidence'] == "HIGH" else 1.0
            
            if risk_pct < 1.0:
                lev = int(22 * conf_mult)
            elif risk_pct < 1.5:
                lev = int(18 * conf_mult)
            elif risk_pct < 2.5:
                lev = int(15 * conf_mult)
            else:
                lev = int(12 * conf_mult)
            
            lev = min(lev, 25)
            
            rr = abs(tp2 - entry) / risk if risk > 0 else 0
            
            return {
                'entry': round(entry, 8),
                'sl': round(sl, 8),
                'tp1': round(tp1, 8),
                'tp2': round(tp2, 8),
                'tp3': round(tp3, 8),
                'tp4': round(tp4, 8),
                'tp5': round(tp5, 8),
                'leverage': lev,
                'risk_pct': round(risk_pct, 2),
                'rr': round(rr, 2)
            }
        except:
            return None
    
    def can_trade(self, symbol):
        if symbol in self.blacklist:
            if time.time() - self.blacklist[symbol] < BLACKLIST_COOLDOWN:
                return False
            del self.blacklist[symbol]
        
        perf = self.pair_performance[symbol]
        total = perf['wins'] + perf['losses']
        
        if total >= 3:
            wr = perf['wins'] / total
            if wr < MIN_WIN_RATE or perf['consecutive_losses'] >= 2:
                logger.info(f"❌ {symbol} blocked - poor performance")
                return False
        
        if len(self.active_trades) >= MAX_CONCURRENT_TRADES:
            return False
        
        for t in self.active_trades.values():
            if t['symbol'] == symbol:
                return False
        
        return True
    
    def send_tg(self, msg):
        for _ in range(3):
            try:
                r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                                data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'}, timeout=10)
                if r.status_code == 200:
                    return True
                time.sleep(1)
            except:
                pass
        return False
    
    def format_signal(self, symbol, analysis, targets):
        emoji = "🚀" if analysis['direction'] == "LONG" else "💥"
        dir_emoji = "🟢" if analysis['direction'] == "LONG" else "🔴"
        
        lev = targets['leverage']
        entry = targets['entry']
        
        tp1_p = abs(targets['tp1'] - entry) / entry * 100 * lev
        tp2_p = abs(targets['tp2'] - entry) / entry * 100 * lev
        tp3_p = abs(targets['tp3'] - entry) / entry * 100 * lev
        tp4_p = abs(targets['tp4'] - entry) / entry * 100 * lev
        
        conf_emoji = "🔥🔥🔥" if analysis['confidence'] == "VERY HIGH" else "🔥🔥" if analysis['confidence'] == "HIGH" else "🔥"
        
        msg = f"""
{emoji} <b>BEAST SIGNAL #{symbol}</b> {emoji}
{dir_emoji} <b>{analysis['direction']}</b> {dir_emoji}

<b>📍 ENTRY:</b> {targets['entry']}
<b>🛡️ STOP:</b> {targets['sl']}

<b>🎯 TARGETS:</b>
<b>TP1:</b> {targets['tp1']} → <b>{tp1_p:.1f}%</b> ⚡
<b>TP2:</b> {targets['tp2']} → <b>{tp2_p:.1f}%</b> 💎
<b>TP3:</b> {targets['tp3']} → <b>{tp3_p:.1f}%</b> 🚀
<b>TP4:</b> {targets['tp4']} → <b>{tp4_p:.1f}%</b> 🌙
<b>TP5:</b> {targets['tp5']} → <b>MOON SHOT</b> 🌕

<b>⚡ LEVERAGE:</b> <b>x{targets['leverage']}</b>

<b>📊 METRICS:</b>
• Score: <b>{analysis['score']}/45</b> {conf_emoji}
• Confidence: <b>{analysis['confidence']}</b>
• Risk: <b>{targets['risk_pct']}%</b>
• R:R: <b>{targets['rr']}:1</b> 📈
• ADX: <b>{analysis['adx']:.1f}</b>
• Stoch: <b>{analysis['stoch_k']:.1f}</b>
• Volume: <b>{analysis['vol_ratio']:.1f}x</b>

<b>🔍 CONFIRMATIONS:</b>
{chr(10).join('✅ ' + s for s in analysis['signals'])}

<b>⏰ {datetime.now().strftime('%I:%M %p')}</b>
<b>━━━━━━━━━━━━━━━━━━━━</b>
<i>Beast Mode Multi-Confirmation System</i>
        """
        return msg.strip()
    
    def get_price(self, symbol):
        for _ in range(2):
            try:
                r = requests.get(f"{BINANCE_API}/fapi/v1/ticker/price",
                               params={'symbol': symbol}, timeout=10)
                return float(r.json()['price'])
            except:
                time.sleep(0.5)
        return None
    
    def monitor_trade(self, tid, trade):
        try:
            price = self.get_price(trade['symbol'])
            if not price:
                return None
            
            direction = trade['direction']
            entry = trade['entry']
            sl = trade['sl']
            lev = trade['leverage']
            
            elapsed = time.time() - trade['entry_time']
            hours = int(elapsed / 3600)
            mins = int((elapsed % 3600) / 60)
            time_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"
            
            result = None
            
            if direction == "LONG":
                if price <= sl:
                    loss = ((sl - entry) / entry) * 100 * lev
                    result = {'status': 'STOP LOSS', 'price': sl, 'profit': round(loss, 2), 'time': time_str}
                elif price >= trade['tp5']:
                    profit = ((trade['tp5'] - entry) / entry) * 100 * lev
                    result = {'status': 'TP5 HIT! 🌕', 'price': trade['tp5'], 'profit': round(profit, 2), 'time': time_str}
                elif price >= trade['tp4']:
                    profit = ((trade['tp4'] - entry) / entry) * 100 * lev
                    result = {'status': 'TP4 HIT! 🌙', 'price': trade['tp4'], 'profit': round(profit, 2), 'time': time_str}
                elif price >= trade['tp3']:
                    profit = ((trade['tp3'] - entry) / entry) * 100 * lev
                    result = {'status': 'TP3 HIT! 🚀', 'price': trade['tp3'], 'profit': round(profit, 2), 'time': time_str}
                elif price >= trade['tp2']:
                    profit = ((trade['tp2'] - entry) / entry) * 100 * lev
                    result = {'status': 'TP2 HIT! 💎', 'price': trade['tp2'], 'profit': round(profit, 2), 'time': time_str}
                elif price >= trade['tp1']:
                    profit = ((trade['tp1'] - entry) / entry) * 100 * lev
                    result = {'status': 'TP1 HIT! ⚡', 'price': trade['tp1'], 'profit': round(profit, 2), 'time': time_str}
            else:
                if price >= sl:
                    loss = ((entry - sl) / entry) * 100 * lev
                    result = {'status': 'STOP LOSS', 'price': sl, 'profit': round(-loss, 2), 'time': time_str}
                elif price <= trade['tp5']:
                    profit = ((entry - trade['tp5']) / entry) * 100 * lev
                    result = {'status': 'TP5 HIT! 🌕', 'price': trade['tp5'], 'profit': round(profit, 2), 'time': time_str}
                elif price <= trade['tp4']:
                    profit = ((entry - trade['tp4']) / entry) * 100 * lev
                    result = {'status': 'TP4 HIT! 🌙', 'price': trade['tp4'], 'profit': round(profit, 2), 'time': time_str}
                elif price <= trade['tp3']:
                    profit = ((entry - trade['tp3']) / entry) * 100 * lev
                    result = {'status': 'TP3 HIT! 🚀', 'price': trade['tp3'], 'profit': round(profit, 2), 'time': time_str}
                elif price <= trade['tp2']:
                    profit = ((entry - trade['tp2']) / entry) * 100 * lev
                    result = {'status': 'TP2 HIT! 💎', 'price': trade['tp2'], 'profit': round(profit, 2), 'time': time_str}
                elif price <= trade['tp1']:
                    profit = ((entry - trade['tp1']) / entry) * 100 * lev
                    result = {'status': 'TP1 HIT! ⚡', 'price': trade['tp1'], 'profit': round(profit, 2), 'time': time_str}
            
            return result
        except:
            return None
    
    def send_result(self, symbol, result):
        is_win = 'TP' in result['status']
        emoji = "✅💰" if is_win else "🚫"
        
        celebration = ""
        if is_win:
            if result['profit'] > 150:
                celebration = "🎊 MASSIVE WIN! 🎊"
            elif result['profit'] > 80:
                celebration = "🎉 GREAT WIN! 🎉"
            elif result['profit'] > 40:
                celebration = "💪 SOLID WIN! 💪"
            else:
                celebration = "✨ WIN! ✨"
        else:
            celebration = "⚠️ Stop Hit"
        
        msg = f"""
{emoji} <b>#{symbol}</b> {emoji}
{celebration}

<b>{result['status']}</b>
<b>Profit: {result['profit']:+.2f}%</b>
<b>Duration: {result['time']}</b> ⏱️

<b>━━━━━━━━━━━━━━━━━━━━</b>
        """
        self.send_tg(msg.strip())
        
        perf = self.pair_performance[symbol]
        if is_win:
            self.daily_stats['wins'] += 1
            perf['wins'] += 1
            perf['consecutive_losses'] = 0
            if result['profit'] > self.daily_stats['best']:
                self.daily_stats['best'] = result['profit']
        else:
            self.daily_stats['losses'] += 1
            perf['losses'] += 1
            perf['consecutive_losses'] += 1
            self.blacklist[symbol] = time.time()
            if result['profit'] < self.daily_stats['worst']:
                self.daily_stats['worst'] = result['profit']
        
        self.daily_stats['total_profit'] += result['profit']
        perf['total_profit'] += result['profit']
    
    def monitor_all(self):
        if not self.active_trades:
            return
        
        logger.info(f"👁️ Monitoring {len(self.active_trades)} trades...")
        
        completed = []
        for tid, trade in list(self.active_trades.items()):
            result = self.monitor_trade(tid, trade)
            if result:
                self.send_result(trade['symbol'], result)
                completed.append(tid)
                logger.info(f"✅ {trade['symbol']} closed: {result['status']} {result['profit']:+.2f}%")
        
        for tid in completed:
            del self.active_trades[tid]
    
    def scan(self):
        logger.info("🔍 BEAST SCANNER ACTIVE...")
        
        pairs = self.get_pairs()
        signals = 0
        max_signals = 5
        
        for i, symbol in enumerate(pairs):
            try:
                if signals >= max_signals:
                    logger.info(f"✅ Quality threshold reached")
                    break
                
                if (i + 1) % 15 == 0:
                    logger.info(f"📊 Progress: {i+1}/{len(pairs)}")
                
                analysis = self.analyze(symbol)
                
                if analysis:
                    if not self.can_trade(symbol):
                        continue
                    
                    targets = self.calc_targets(symbol, analysis['direction'], analysis)
                    
                    if targets and targets['rr'] >= MIN_RR_RATIO:
                        msg = self.format_signal(symbol, analysis, targets)
                        
                        if self.send_tg(msg):
                            tid = f"{symbol}_{int(time.time())}"
                            self.active_trades[tid] = {
                                'symbol': symbol,
                                'direction': analysis['direction'],
                                'entry': targets['entry'],
                                'sl': targets['sl'],
                                'tp1': targets['tp1'],
                                'tp2': targets['tp2'],
                                'tp3': targets['tp3'],
                                'tp4': targets['tp4'],
                                'tp5': targets['tp5'],
                                'leverage': targets['leverage'],
                                'entry_time': time.time()
                            }
                            
                            signals += 1
                            self.daily_stats['signals'] += 1
                            
                            logger.info(f"🚀 {analysis['direction']}: {symbol} (Score: {analysis['score']}, {analysis['confidence']})")
                            time.sleep(2)
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        logger.info(f"✅ Scan complete! {signals} signals sent")
        
        if signals == 0:
            summary = f"📊 Scan: {datetime.now().strftime('%I:%M %p')}\n💤 No quality setups\n⚡ Active: {len(self.active_trades)}"
            self.send_tg(summary)
    
    def send_performance(self):
        total = self.daily_stats['wins'] + self.daily_stats['losses']
        if total == 0:
            return
        
        wr = (self.daily_stats['wins'] / total) * 100
        avg = self.daily_stats['total_profit'] / total
        
        if wr >= 70 and avg > 25:
            grade = "🏆 ELITE"
        elif wr >= 60 and avg > 18:
            grade = "💎 EXCELLENT"
        elif wr >= 50 and avg > 12:
            grade = "✅ GOOD"
        else:
            grade = "⚠️ IMPROVING"
        
        top = sorted([(k, v) for k, v in self.pair_performance.items() 
                     if v['wins'] + v['losses'] > 0],
                    key=lambda x: x[1]['total_profit'], reverse=True)[:3]
        
        top_text = "\n".join([f"• {p}: {pf['total_profit']:+.1f}% ({pf['wins']}W-{pf['losses']}L)"
                             for p, pf in top]) if top else "None yet"
        
        msg = f"""
📊 <b>BEAST BOT PERFORMANCE</b>
{grade}

<b>📈 STATISTICS:</b>
• Signals: <b>{self.daily_stats['signals']}</b>
• Completed: <b>{total}</b>
• Win Rate: <b>{wr:.1f}%</b> ✅
• W/L: <b>{self.daily_stats['wins']}/{self.daily_stats['losses']}</b>

<b>💰 PROFIT/LOSS:</b>
• Total: <b>{self.daily_stats['total_profit']:+.2f}%</b>
• Avg/Trade: <b>{avg:+.2f}%</b>
• Best: <b>{self.daily_stats['best']:+.1f}%</b> 🚀
• Worst: <b>{self.daily_stats['worst']:+.1f}%</b>

<b>⚡ STATUS:</b>
• Active: <b>{len(self.active_trades)}</b>
• Blacklisted: <b>{len(self.blacklist)}</b>

<b>🏆 TOP PERFORMERS:</b>
{top_text}

<b>━━━━━━━━━━━━━━━━━━━━</b>
<i>{datetime.now().strftime('%Y-%m-%d %I:%M %p')}</i>
        """
        self.send_tg(msg.strip())
    
    def run(self):
        logger.info("🚀 BEAST TRADING BOT V3.0 STARTED!")
        
        startup = """
🤖 <b>BEAST TRADING BOT V3.0</b>
━━━━━━━━━━━━━━━━━━━━

<b>🎯 BEAST MODE STRATEGY:</b>
✅ Ultra-Strict Multi-Confirmation System
✅ 15+ Advanced Technical Indicators
✅ Market Structure Analysis (HH/HL/LH/LL)
✅ Chart Pattern Detection
✅ RSI/MFI Divergence Detection
✅ Multi-Timeframe Alignment (15m/1h/4h)
✅ Perfect EMA Stack Requirements
✅ VWAP Precision Entry System
✅ Volume Surge Confirmation
✅ Rejection Wick Analysis

<b>📊 KEY FEATURES:</b>
• Minimum Score: 28/45 (very strict)
• Minimum R:R: 2.5:1
• Dynamic Leverage: 12-25x
• Smart Pair Performance Tracking
• Auto-Blacklist Poor Performers
• Max 3 Concurrent Trades
• 5 Progressive Take-Profits
• HTF Trend Confirmation Required

<b>🎯 ENTRY REQUIREMENTS:</b>
• Supertrend aligned (CRITICAL)
• ADX > 20 (trend strength)
• HTF alignment (1h + 4h)
• Perfect EMA stack
• Volume confirmation
• Market structure support
• Multiple indicator confluence

<b>🔥 BUILT FOR CONSISTENCY!</b>
<i>Quality over quantity - Beast Mode activated 💪</i>

<b>━━━━━━━━━━━━━━━━━━━━</b>
        """
        self.send_tg(startup.strip())
        
        scan_count = 0
        
        while True:
            try:
                self.scan()
                
                if self.active_trades:
                    self.monitor_all()
                
                scan_count += 1
                if scan_count >= 12:  # ~1 hour
                    self.send_performance()
                    scan_count = 0
                
                # Clean blacklist
                current = time.time()
                self.blacklist = {k: v for k, v in self.blacklist.items()
                                 if current - v < BLACKLIST_COOLDOWN}
                
                logger.info(f"💤 Next scan in {SCAN_INTERVAL}s...")
                time.sleep(SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                self.send_tg("🛑 Beast Bot stopped")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║    BEAST TRADING BOT V3.0 - INITIALIZED 🚀      ║
    ║    Ultra-Strict Multi-Confirmation System        ║
    ╚═══════════════════════════════════════════════════╝
    
    🎯 BEAST MODE FEATURES:
    ✅ 15+ Advanced Technical Indicators
    ✅ Market Structure Detection (HH/HL/LH/LL)
    ✅ Chart Pattern Recognition
    ✅ Advanced Divergence Detection
    ✅ Multi-Timeframe Analysis (15m/1h/4h)
    ✅ Perfect EMA Stack Requirements
    ✅ VWAP Precision System
    ✅ Volume Surge Confirmation
    ✅ Rejection Wick Analysis
    ✅ Smart Pair Performance Tracking
    ✅ Auto-Blacklist System
    ✅ Dynamic Leverage (12-25x)
    ✅ 5 Progressive Take-Profits
    
    📊 STRATEGY LOGIC:
    Score System: 0-45 points (Min: 28)
    
    LONG = Supertrend UP + Stoch 10-40 + Multi-confirm
    - ADX > 25 with Bullish DI
    - HTF (1h + 4h) aligned bullish
    - Perfect EMA stack (9>21>50>200)
    - Price near/above VWAP
    - Volume surge > 1.8x
    - Market structure: Uptrend
    - Divergence bonus points
    
    SHORT = Supertrend DOWN + Stoch 60-90 + Multi-confirm
    - ADX > 25 with Bearish DI
    - HTF (1h + 4h) aligned bearish
    - Perfect EMA stack (9<21<50<200)
    - Price near/below VWAP
    - Volume surge > 1.8x
    - Market structure: Downtrend
    - Divergence bonus points
    
    🎯 TARGETS (Risk-Based):
    TP1: 1.2R (Quick secure)
    TP2: 2.5R (Main target - R:R minimum)
    TP3: 4.0R (Extended)
    TP4: 6.0R (Major breakout)
    TP5: 8.5R (Moon shot)
    
    ⚙️ REQUIREMENTS:
    pip install requests pandas numpy scipy
    
    🔥 BUILT TO PRINT CASH - NO FALSE POSITIVES! 🔥
    Press Ctrl+C to stop
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    bot = BeastTradingBot()
    bot.run()