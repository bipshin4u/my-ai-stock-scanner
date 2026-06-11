import streamlit as st
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf
from datetime import datetime, timedelta
import io
import time

# Page Setup
st.set_page_config(page_title="AI Market Confluence Scanner", layout="wide")
st.title("📊 AI Market Confluence Live Dashboard")
st.write("Calculates 8/21 EMA triggers, RSI divergence, and automated SuperTrend regime bypass filters.")

# Initialize Session State Memory (Temporary browser database)
if "stacked_results" not in st.session_state:
    st.session_state.stacked_results = pd.DataFrame()
if "scanned_batches" not in st.session_state:
    st.session_state.scanned_batches = set()

# Sidebar Watchlist Input
st.sidebar.header("📋 Custom Watchlist Mode")
default_watchlist = "AAPL, MSFT, NVDA, AMD, META, AMZN, GOOGL, TSLA, NFLX, AVGO"
user_input = st.sidebar.text_area("Edit your custom stocks (comma separated):", default_watchlist, height=100)
watchlist = [t.strip().upper() for t in user_input.split(",") if t.strip()]

# BATCH 1: Ranks 1 to 50 Mega-Cap Giants
BATCH_1_STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "PLTR", "NFLX",
    "AVGO", "SMCI", "COIN", "ORCL", "CRM", "QCOM", "INTC", "MU", "PANW", "MRVL",
    "JPM", "BAC", "WMT", "COST", "DIS", "XOM", "CVX", "LLY", "UNH", "V",
    "MA", "HD", "PG", "PFE", "MRK", "ABBV", "KO", "PEP", "NKE", "SBUX",
    "GE", "CAT", "BA", "HON", "IBM", "ACN", "TXN", "AMGN", "GILD", "BABA"
]

# BATCH 2: Ranks 51 to 100 High-Volume Large-Caps
BATCH_2_STOCKS = [
    "UBER", "ABNB", "SQ", "PYPL", "SOFI", "DKNG", "HOOD", "AFRM", "RIVN", "LCID",
    "NIO", "XPEV", "LI", "F", "GM", "TM", "T", "VZ", "CMCSA", "WBD", 
    "PARA", "SPOT", "RBLX", "U", "AI", "PATH", "SNOW", "NET", "CRWD", "OKTA", 
    "DDOG", "ZS", "FTNT", "CHKP", "MDB", "DOCU", "TWLO", "PINS", "SNAP", "SHOP", 
    "SE", "MELI", "PDD", "JD", "BIDU", "GME", "AMC"
]

# BATCH 3: Ranks 101 to 150 Mid-Cap & Small-Cap Alpha Tickers
BATCH_3_STOCKS = [
    "CELH", "ELF", "DUOL", "IOT", "MSTR", "UPST", "HIMS", "ASTS", "OKLO", "NNE", 
    "CLSK", "MARA", "RIOT", "WULF", "IREN", "CIFR", "CORZ", "APPS", "PTON", "ROKU", 
    "CVNA", "CHWY", "FSLR", "ENPH", "RUN", "CSIQ", "GTLB", "ALGM", "ALTR", "LSCC", 
    "POWI", "SLAB", "CRUS", "COHR", "LITE", "FN", "VRT", "MOD", "SYM", "AOUT", 
    "JOBY", "ACHR", "LUNR", "RKLB", "BBAI", "SOUN"
]

def run_scanner(tickers, is_discovery=False):
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

            # Scoring Math Logic
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
            numeric_score = buy_score if buy_score >= sell_score else -sell_score

            results.append({
                "Ticker": ticker,
                "Last Close": f"${close_series.iloc[-1]:.2f}",
                "Market Regime": regime_label,
                "Buy Score": f"{buy_score}/6",
                "Sell Score": f"{sell_score}/6",
                "Action Signal": signal,
                "RawScore": numeric_score
            })
            
            if is_discovery:
                time.sleep(0.12) # Anti-throttling server delay
                
        except:
            pass
        progress_bar.progress((idx + 1) / len(tickers))
        
    return pd.DataFrame(results)

def display_master_leaderboard():
    df = st.session_state.stacked_results
    
    if not df.empty:
        # Prevent row duplication during multiple sequential clicks
        df = df.drop_duplicates(subset=["Ticker"], keep="last")
        
        # Sort matrix perfectly to prioritize strongest setups across all batches combined
        df['SortOrder'] = df['Action Signal'].map({'🔥 STRONG BUY': 0, '🚨 STRONG SELL': 1, '⏳ HOLD / NEUTRAL': 2})
        df = df.sort_values(by=["SortOrder", "RawScore"], ascending=[True, False]).drop(columns=['SortOrder'])
        
        total_buys = int(sum(df['Action Signal'] == "🔥 STRONG BUY"))
        total_sells = int(sum(df['Action Signal'] == "🚨 STRONG SELL"))
        
        # Upper KPI Metrics Panel
        st.subheader("🎯 Aggregated Master Analysis Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Stacked Strong Buys Found", f"{total_buys} Stocks")
        m_col2.metric("Stacked Strong Sells Found", f"{total_sells} Stocks")
        m_col3.metric("Total Cumulative Assets Scanned", f"{len(df)} Tickers")
        
        # Status Tracker Info Banner
        loaded_batches = ", ".join(sorted(list(st.session_state.scanned_batches)))
        st.info(f"📁 Current data stacked from: **{loaded_batches}**")
        
        st.markdown("---")
        st.subheader(f"📊 Global Master Leaderboard — {datetime.now().strftime('%Y-%m-%d %H:%M')} EST")
        
        # Row Highlighter Engine using Pandas modern .map structure
        def color_whole_rows(row):
            if row['Action Signal'] == "🔥 STRONG BUY":
                return ['background-color: #1e4620; color: #ffffff; font-weight: bold;'] * len(row)
            elif row['Action Signal'] == "🚨 STRONG SELL":
                return ['background-color: #611f1d; color: #ffffff; font-weight: bold;'] * len(row)
            else:
                return ['background-color: #1a1c23; color: #a3a8b4;'] * len(row)
        
        display_df = df.drop(columns=['RawScore']) if 'RawScore' in df.columns else df
        styled_df = display_df.style.map(color_whole_rows, axis=1)
        
        # Patched responsive width handling using stretch layout specification
        st.dataframe(styled_df, width="stretch", hide_index=True)
        
        # Spreadsheet compiler memory bridge
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            display_df.to_excel(writer, index=False, sheet_name="Master Signals")
        
        st.download_button(
            label="📥 Download Stacked Master Report as Excel",
            data=buffer.getvalue(),
            file_name=f"Master_Stacked_Signals_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("Dashboard empty. Click a scanning routine on the sidebar menu to populate the live tables.")

# --- UI SIDEBAR INTERACTION BUTTONS ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Discovery Matrix Engine")
st.sidebar.write("Scan market layers. Results automatically append into the global master list view below.")

# 1. Watchlist Scan Trigger
if st.sidebar.button("🚀 Run Custom Watchlist Scan"):
    with st.spinner("Analyzing custom watchlist..."):
        res_df = run_scanner(watchlist, is_discovery=False)
        st.session_state.stacked_results = res_df
        st.session_state.scanned_batches = {"Custom Watchlist"}
        st.rerun()

# 2. Batch 1 Trigger
if st.sidebar.button("🔍 Scan Batch 1: Ranks 1-50 (Mega-Caps)"):
    with st.spinner("Processing Ranks 1-50..."):
        df1 = run_scanner(BATCH_1_STOCKS, is_discovery=True)
        st.session_state.stacked_results = pd.concat([st.session_state.stacked_results, df1], ignore_index=True)
        st.session_state.scanned_batches.add("Batch 1 (1-50)")
        st.rerun()

# 3. Batch 2 Trigger
if st.sidebar.button("⏭️ Scan Batch 2: Ranks 51-100 (Large-Caps)"):
    with st.spinner("Processing Ranks 51-100..."):
        df2 = run_scanner(BATCH_2_STOCKS, is_discovery=True)
        st.session_state.stacked_results = pd.concat([st.session_state.stacked_results, df2], ignore_index=True)
        st.session_state.scanned_batches.add("Batch 2 (51-100)")
        st.rerun()
        
# 4. Batch 3 Trigger
if st.sidebar.button("🔬 Scan Batch 3: Ranks 101-150 (Mid/Small-Caps)"):
    with st.spinner("Processing Ranks 101-150 High-Growth tickers..."):
        df3 = run_scanner(BATCH_3_STOCKS, is_discovery=True)
        st.session_state.stacked_results = pd.concat([st.session_state.stacked_results, df3], ignore_index=True)
        st.session_state.scanned_batches.add("Batch 3 (101-150)")
        st.rerun()
        
# 5. Clear Memory Reset Button
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Screen & Reset Scanner", type="primary"):
    st.session_state.stacked_results = pd.DataFrame()
    st.session_state.scanned_batches = set()
    st.toast("Dashboard cache completely wiped clean!")
    st.rerun()
    
# Execute Drawing Routine
display_master_leaderboard()
