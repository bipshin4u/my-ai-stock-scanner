import streamlit as st
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf
from datetime import datetime, timedelta
import io

# Set up beautiful web layout
st.set_page_config(page_title="AI Market Confluence Scanner", layout="wide")
st.title("📊 AI Market Confluence Live Dashboard")
st.write("Calculates 8/21 EMA triggers, RSI divergence, and automated SuperTrend regime bypass filters.")

# Sidebar for Watchlist management
st.sidebar.header("📋 Watchlist Configuration")
default_watchlist = "AAPL, MSFT, NVDA, AMD, META, AMZN, GOOGL, TSLA, NFLX, AVGO, PLTR, SMCI, COIN, ORCL, CRM, QCOM, INTC, MU, PANW, MRVL"
user_input = st.sidebar.text_area("Edit your 20 stocks (comma separated):", default_watchlist, height=200)

# Process tickers
watchlist = [t.strip().upper() for t in user_input.split(",") if t.strip()]

def run_scanner(tickers):
    fast_ema_len, slow_ema_len, trend_ema_len = 8, 21, 200
    rsi_len, rsi_lower, rsi_upper = 14, 35, 65
    st_period, st_multiplier = 10, 2.5
    regime_len, regime_threshold = 14, 1.2

    results = []
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    progress_bar = st.progress(0)
    for idx, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty or len(df) < trend_ema_len:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            close_series = pd.Series(df['Close'])
            fast_ema = ta.ema(close_series, length=fast_ema_len).to_numpy()
            slow_ema = ta.ema(close_series, length=slow_ema_len).to_numpy()
            trend_ema = ta.ema(close_series, length=trend_ema_len).to_numpy()
            rsi = ta.rsi(close_series, length=rsi_len).to_numpy()
            
            st_df = ta.supertrend(df['High'], df['Low'], df['Close'], length=st_period, multiplier=st_multiplier)
            st_dir = st_df.iloc[:, 1].fillna(0).to_numpy()
            
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=regime_len)
            std_dev = close_series.rolling(regime_len).std()
            market_volatility = (std_dev / atr).fillna(1.0).to_numpy()

            is_trending = market_volatility[-1] > regime_threshold
            ema_buy_trigger = fast_ema[-1] > slow_ema[-1] and fast_ema[-2] <= slow_ema[-2]
            ema_sell_trigger = fast_ema[-1] < slow_ema[-1] and fast_ema[-2] >= slow_ema[-2]
            above_macro_trend = close_series.iloc[-1] > trend_ema[-1]
            
            st_bullish = st_dir[-1] == 1 if is_trending else True
            st_bearish = st_dir[-1] == -1 if is_trending else True

            rsi_bullish_div = rsi[-1] > rsi[-2] and close_series.iloc[-1] <= close_series.iloc[-2] and rsi[-1] < rsi_upper
            rsi_bearish_div = rsi[-1] < rsi[-2] and close_series.iloc[-1] >= close_series.iloc[-2] and rsi[-1] > rsi_lower

            buy_score = 0
            sell_score = 0
            if above_macro_trend: buy_score += 1
            else: sell_score += 1
            if ema_buy_trigger: buy_score += 2
            if ema_sell_trigger: sell_score += 2
            if st_bullish: buy_score += 1
            if st_bearish: sell_score += 1
            if is_trending and rsi_bullish_div: buy_score += 2
            if is_trending and rsi_bearish_div: sell_score += 2

            regime_label = "TRENDING 📈" if is_trending else "RANGING ↔️ (Bypassed)"
            signal = "🔥 STRONG BUY" if buy_score >= 5 else "🚨 STRONG SELL" if sell_score >= 5 else "⏳ HOLD / NEUTRAL"

            results.append({
                "Ticker": ticker,
                "Last Close": f"${close_series.iloc[-1]:.2f}",
                "Market Regime": regime_label,
                "Buy Score": f"{buy_score}/6",
                "Sell Score": f"{sell_score}/6",
                "Action Signal": signal
            })
        except:
            pass
        progress_bar.progress((idx + 1) / len(tickers))
    return pd.DataFrame(results)

# Run Scanner button
if st.sidebar.button("🚀 Run Live Market Scan"):
    with st.spinner("Fetching latest Wall Street prices..."):
        scanner_df = run_scanner(watchlist)
        
        if not scanner_df.empty:
            scanner_df['SortOrder'] = scanner_df['Action Signal'].map({'🔥 STRONG BUY': 0, '🚨 STRONG SELL': 1, '⏳ HOLD / NEUTRAL': 2})
            scanner_df = scanner_df.sort_values(by='SortOrder').drop(columns=['SortOrder'])
            
            st.subheader(f"📊 Market Leaderboard — {datetime.now().strftime('%Y-%m-%d %H:%M')} EST")
            
            def color_signals(val):
                if val == "🔥 STRONG BUY": return "background-color: #d4edda; color: #155724; font-weight: bold;"
                if val == "🚨 STRONG SELL": return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
                return ""
            
            st.dataframe(scanner_df.style.map(color_signals, subset=['Action Signal']), use_container_width=True, hide_index=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                scanner_df.to_excel(writer, index=False, sheet_name="Daily Signals")
            
            st.download_button(
                label="📥 Download Results as Excel Spreadsheet",
                data=buffer.getvalue(),
                file_name=f"Market_Signals_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("No stock data could be gathered. Please check your ticker spelling formatting.")
