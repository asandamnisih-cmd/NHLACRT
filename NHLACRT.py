import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import os
import nest_asyncio
from telegram import Bot
from datetime import datetime

nest_asyncio.apply()

# ===== 1. CONFIGURATION =====
TG_TOKEN = os.getenv("TG_TOKEN") 
CHAT_ID = "7053671715"                

INSTRUMENTS = {
    "^DJI":     "US30",
    "^IXIC":    "Nasdaq",
    "GC=F":     "Gold",
    "^GSPC":    "US500",
    "USDJPY=X": "USDJPY",
    "USDCAD=X": "USDCAD",
    "GBPUSD=X": "GBPUSD",
    "EURUSD=X": "EURUSD",
    "BTC-USD":  "BTCUSD",
    "ETH-USD":  "ETHUSD",
    "^FTSE":    "UK100"
}

TIMEFRAME = "4h"      # CRT Reference Timeframe
ENTRY_TF = "15m"      # <--- UPDATED TO 15 MINUTES

# ===== 2. DATA FETCHING & CALCULATIONS =====
def fetch_ohlc(symbol, interval, period):
    try:
        # Note: We only pull 5 days for 15m data to avoid rate limits
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if df.empty:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[["Open", "High", "Low", "Close"]].copy()
        df.columns = ["open", "high", "low", "close"]
        
        # Calculate ATR for OB
        df['tr'] = np.maximum(df['high'] - df['low'], 
                              np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                         abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        return df
    except Exception as e:
        print(f"Error fetching {symbol} {interval}: {e}")
        return None

# ===== 3. FVG & OB DETECTION (On 15m) =====
def check_recent_fvg(df, is_bull):
    lookback = min(10, len(df) - 3)
    for i in range(1, lookback):
        idx = -i
        if is_bull:
            if df['low'].iloc[idx] > df['high'].iloc[idx-2]:
                return True
        else:
            if df['high'].iloc[idx] < df['low'].iloc[idx-2]:
                return True
    return False

def has_recent_ob(df, is_bull):
    lookback = 3
    if len(df) < lookback + 5:
        return False
        
    for i in range(1, lookback + 1):
        idx = -i
        candle = df.iloc[idx]
        body_size = abs(candle['close'] - candle['open'])
        
        if body_size > candle['atr'] * 1.5:
            if is_bull and candle['close'] > candle['open']:
                recent_high = df['high'].iloc[idx-5:idx].max()
                if candle['close'] > recent_high:
                    return True
            elif not is_bull and candle['close'] < candle['open']:
                recent_low = df['low'].iloc[idx-5:idx].min()
                if candle['close'] < recent_low:
                    return True
    return False

# ===== 4. MODEL LOGIC =====
def check_crt_signals(df_4h, df_15m):
    if df_4h is None or len(df_4h) < 3 or df_15m is None or len(df_15m) < 10:
        return None

    # 1. Get CRT Level from 4H
    c1 = df_4h.iloc[-2]
    c1_high, c1_low = c1['high'], c1['low']
    c1_eq = (c1_high + c1_low) / 2

    # 2. Check 15m data for sweep & entry
    prev_candle = df_15m.iloc[-2]
    curr_candle = df_15m.iloc[-1]
    current_close = curr_candle['close']

    recent_bull_fvg = check_recent_fvg(df_15m, True)
    recent_bear_fvg = check_recent_fvg(df_15m, False)
    has_bull_ob = has_recent_ob(df_15m, True)
    has_bear_ob = has_recent_ob(df_15m, False)

    # --- MODEL 1 (15m Sweep & Close) ---
    m1_sell = curr_candle['high'] > c1_high and curr_candle['close'] < c1_high and (recent_bear_fvg or has_bear_ob)
    m1_buy = curr_candle['low'] < c1_low and curr_candle['close'] > c1_low and (recent_bull_fvg or has_bull_ob)

    # --- MODEL 2 (15m Breakout & Retest) ---
    m2_buy = False
    m2_sell = False
    
    if prev_candle['close'] > c1_high and curr_candle['low'] <= c1_high and curr_candle['close'] > c1_high and curr_candle['close'] > curr_candle['open']:
         m2_buy = (recent_bull_fvg or has_bull_ob)
         
    if prev_candle['close'] < c1_low and curr_candle['high'] >= c1_low and curr_candle['close'] < c1_low and curr_candle['close'] < curr_candle['open']:
         m2_sell = (recent_bear_fvg or has_bear_ob)

    # --- CONFLUENCE SCORE ---
    bull_score = 0
    if curr_candle['low'] < c1_low: bull_score += 1
    if recent_bull_fvg: bull_score += 1
    if has_bull_ob: bull_score += 1
    if current_close < c1_eq: bull_score += 1 
    if m1_buy or m2_buy: bull_score += 1

    bear_score = 0
    if curr_candle['high'] > c1_high: bear_score += 1
    if recent_bear_fvg: bear_score += 1
    if has_bear_ob: bear_score += 1
    if current_close > c1_eq: bear_score += 1 
    if m1_sell or m2_sell: bear_score += 1

    signal = None
    if m1_buy:
        signal = {"dir": "BULLISH", "model": "Model 1 (15m Sweep)", "score": bull_score, "eq": c1_eq}
    elif m1_sell:
        signal = {"dir": "BEARISH", "model": "Model 1 (15m Sweep)", "score": bear_score, "eq": c1_eq}
    elif m2_buy:
        signal = {"dir": "BULLISH", "model": "Model 2 (15m MSS)", "score": bull_score, "eq": c1_eq}
    elif m2_sell:
        signal = {"dir": "BEARISH", "model": "Model 2 (15m MSS)", "score": bear_score, "eq": c1_eq}

    return signal

# ===== 5. ASYNC SCANNER & TELEGRAM =====
async def send_telegram_message(msg):
    if not TG_TOKEN:
        print("ERROR: TG_TOKEN is missing!")
        return
    bot = Bot(token=TG_TOKEN)
    
    for attempt in range(3):
        try:
            await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
            print("--> Message successfully sent to Telegram!")
            return
        except Exception as e:
            print(f"Telegram attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                await asyncio.sleep(3)
            else:
                print("--> Failed to send message after 3 attempts.")

async def run_scan():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting CRT Master scan (15m Entry)...")
    alerts = []

    for symbol, name in INSTRUMENTS.items():
        print(f"Scanning {name}...")
        
        # Fetch 4H data (for CRT range)
        df_4h = fetch_ohlc(symbol, TIMEFRAME, "60d")
        # Fetch 15m data (for entry confirmation - only 5 days needed)
        df_15m = fetch_ohlc(symbol, ENTRY_TF, "5d")
        
        if df_4h is None or df_15m is None:
            continue
            
        signal = check_crt_signals(df_4h, df_15m)
        
        if signal:
            if signal['score'] >= 3:
                alerts.append({
                    "instrument": name,
                    "direction": signal['dir'],
                    "model": signal['model'],
                    "score": signal['score'],
                    "price": df_15m.iloc[-1]['close'],
                    "eq": signal['eq']
                })

    if alerts:
        msg = "🚨 *CRT MASTER SIGNAL* 🚨\n\n"
        for a in alerts:
            msg += (
                f"*{a['instrument']}* | {a['direction']}\n"
                f"Price: `{a['price']:.4f}` | EQ: `{a['eq']:.4f}`\n"
                f"Confluence Score: *{a['score']}/5*\n"
                f"Trigger: {a['model']}\n\n"
            )
        await send_telegram_message(msg)
        print(f"--> Found {len(alerts)} signal(s). Sent to Telegram!")
    else:
        msg = (
            "🔍 *CRT Master Scanner Update (15m Entry)*\n\n"
            "No valid high-confluence signals detected.\n\n"
            f"*Scanned:* US30, Nasdaq, Gold, US500, USDJPY, USDCAD, GBPUSD, EURUSD, BTCUSD, ETHUSD, UK100\n"
            f"*Time:* {datetime.now().strftime('%H:%M')} UTC\n\n"
            "Waiting for the next 15m candle close."
        )
        await send_telegram_message(msg)
        print("--> No signals found. Heartbeat message processed.")

# ===== 6. EXECUTION =====
if __name__ == "__main__":
    print("🤖 Running scheduled CRT Master scan...")
    asyncio.run(run_scan())
    print("Scan complete. Exiting.")
