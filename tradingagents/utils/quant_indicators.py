import pandas as pd
import numpy as np
from stockstats import wrap

def calculate_volatility_scaled_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculates Volatility-Scaled RSI (RSI Z-Score).
    
    Instead of raw 0-100 values, this returns the Z-Score of the RSI 
    relative to its own history. This adjusts for changing volatility regimes.
    
    Values > 2.0 indicate statistically significant overbought conditions.
    Values < -2.0 indicate statistically significant oversold conditions.
    """
    # Ensure we have a StockDataFrame
    stock = wrap(df.copy())
    
    # Calculate base RSI
    rsi = stock[f'rsi_{window}']
    
    # Calculate Z-Score over a longer window (e.g., 3 months approx 60 days)
    # to establish a statistical baseline
    baseline_window = window * 4
    
    rsi_mean = rsi.rolling(window=baseline_window).mean()
    rsi_std = rsi.rolling(window=baseline_window).std()
    
    # Avoid division by zero
    rsi_std = rsi_std.replace(0, 1e-9)
    
    rsi_z = (rsi - rsi_mean) / rsi_std
    
    return rsi_z

def calculate_impulse_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Impulse MACD.
    
    Augments standard MACD with a volatility filter on the Histogram.
    Identifies "Impulse" moves where momentum is statistically significant
    (Histogram breaks its own Bollinger Bands).
    
    Returns DataFrame with:
    - macd: The MACD line
    - signal: The Signal line
    - hist: The MACD Histogram
    - impulse: 1 (Bullish Impulse), -1 (Bearish Impulse), 0 (Neutral)
    """
    stock = wrap(df.copy())
    
    macd = stock['macd']
    macds = stock['macds'] # Signal line
    macdh = stock['macdh'] # Histogram
    
    # Calculate Bollinger Bands of the MACD Histogram to detect significant momentum shifts
    window = 20
    macdh_mean = macdh.rolling(window=window).mean()
    macdh_std = macdh.rolling(window=window).std()
    
    ub = macdh_mean + (2 * macdh_std)
    lb = macdh_mean - (2 * macdh_std)
    
    # Impulse Signal: 
    # Bullish Impulse: Histogram > Upper Band
    # Bearish Impulse: Histogram < Lower Band
    impulse = pd.Series(0, index=df.index)
    impulse[macdh > ub] = 1
    impulse[macdh < lb] = -1
    
    return pd.DataFrame({
        'macd': macd,
        'signal': macds,
        'hist': macdh,
        'impulse': impulse
    })

def calculate_market_regime(df: pd.DataFrame) -> pd.Series:
    """
    Classifies market regime into 4 states:
    - Trending Bullish (ADX > 25 & Price > 50SMA)
    - Trending Bearish (ADX > 25 & Price < 50SMA)
    - Mean Reverting / Ranging (ADX < 20)
    - Neutral / Transition (ADX between 20-25)
    """
    stock = wrap(df.copy())
    
    # Ensure ADX and SMA are calculated
    adx = stock['adx']
    sma50 = stock['close_50_sma']
    close = stock['close']
    
    regime = pd.Series("Neutral / Transition", index=df.index)
    
    # Trending
    is_trending = adx > 25
    bullish = is_trending & (close > sma50)
    bearish = is_trending & (close < sma50)
    
    regime[bullish] = "Trending Bullish"
    regime[bearish] = "Trending Bearish"
    
    # Ranging
    is_ranging = adx < 20
    regime[is_ranging] = "Mean Reverting / Ranging"
    
    return regime

def calculate_vpt(df: pd.DataFrame) -> pd.Series:
    """
    Calculates Volume Price Trend (VPT).
    
    VPT combines price and volume to measure the strength of a price trend.
    It is similar to On Balance Volume (OBV) but takes into account the 
    percentage increase or decrease in price, not just the direction.
    """
    # Manual calculation as stockstats might not have VPT
    # VPT = Previous VPT + Volume * (Today's Price Change / Previous Price)
    # VPT = Cumulative Sum of (Volume * % Change in Price)
    
    # Ensure standard pandas series
    if 'Close' in df.columns:
        close = df['Close']
    elif 'close' in df.columns:
        close = df['close']
    else:
        # Fallback
        return pd.Series(0, index=df.index)

    if 'Volume' in df.columns:
        volume = df['Volume']
    elif 'volume' in df.columns:
        volume = df['volume']
    else:
        # Fallback if volume is missing
        return pd.Series(0, index=df.index)
        
    pct_change = close.pct_change()
    vpt = (volume * pct_change).cumsum()
    
    # Fill NaN at start
    vpt = vpt.fillna(0)
    
    return vpt
