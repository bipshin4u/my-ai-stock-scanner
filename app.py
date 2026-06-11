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
st.write("Calculates 8/21 EMA triggers, RSI divergence, and automated SuperTrend regime bypass filters using live indices.")

# Initialize Session State Memory (Temporary browser database)
if "stacked_results" not in st.session_state:
    st.session_state.stacked_results = pd.DataFrame()
if "scanned_batches" not in st.session_state:
    st.session_state.scanned_batches = set()
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "None"

# Sidebar Watchlist Input
st.sidebar.header("📋 Custom Watchlist Mode")
default_watchlist = "AAPL, MSFT, NVDA, AMD, META, AMZN, GOOGL, TSLA, NFLX, AVGO"
user_input = st.sidebar.text_area("Edit your custom stocks (comma separated):", default_watchlist, height=100)
watchlist = [t.strip().upper() for t in user_input.split(",") if t.strip()]

@st.cache_data(ttl=86400)
def fetch_live_index_tickers():
    """
    Scrapes highly isolated, dedicated Wikipedia tables 
    to completely avoid HTML text/script scraping bugs.
    """
    import urllib.request
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        # 1. Scrape S&P 500 cleanly using the definitive index link
        sp500_url = "https://wikipedia.org"
        req_sp = urllib.request.Request(sp500_url, headers=headers)
        with urllib.request.urlopen(req_sp) as response:
            sp500_table = pd.read_html(response.read())[0] # Explicitly lock table index 0
        sp500_tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()

        # 2. Scrape Nasdaq 100 via its cleaner dedicated component page URL
        ndx_url = "https://wikipedia.org"
        req_ndx = urllib.request.Request(ndx_url, headers=headers)
        with urllib.request.urlopen(req_ndx) as response:
            ndx_tables = pd.read_html(response.read())
            
        # Target the explicit components matrix list cleanly
        ndx_df = None
        for table in ndx_tables:
            if 'Ticker' in table.columns:
                ndx_df = table
                break
            elif 'Symbol' in table.columns:
                ndx_df = table
                break
        
        # If no table caught, use index 4 which is the structural default block
        if ndx_df is None:
            ndx_df = ndx_tables[4]
            
        ticker_col = 'Ticker' if 'Ticker' in ndx_df.columns else 'Symbol'
        ndx_tickers = ndx_df[ticker_col].str.replace('.', '-', regex=False).tolist()
        
        return sp500_tickers, ndx_tickers
    except Exception as e:
        st.sidebar.error(f"Live fetch error: {str(e)}. Using baseline matrix.")
        fallback = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "AVGO"]
        return fallback, fallback
        
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
                time.sleep(0.12) # Safeguard API pacing delay
                
        except:
            pass
        progress_bar.progress((idx + 1) / len(tickers))
        
    return pd.DataFrame(results)

def display_master_leaderboard():
    df = st.session_state.stacked_results
    
    if not df.empty:
        df = df.drop_duplicates(subset=["Ticker"], keep="last")
        
        df['SortOrder'] = df['Action Signal'].map({'🔥 STRONG BUY': 0, '🚨 STRONG SELL': 1, '⏳ HOLD / NEUTRAL': 2})
        df = df.sort_values(by=["SortOrder", "RawScore"], ascending=[True, False]).drop(columns=['SortOrder'])
        
        total_buys = int(sum(df['Action Signal'] == "🔥 STRONG BUY"))
        total_sells = int(sum(df['Action Signal'] == "🚨 STRONG SELL"))
        
        st.subheader(f"🎯 Aggregated {st.session_state.active_mode} Analysis Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Strong Buy Setups Found", f"{total_buys} Stocks")
        m_col2.metric("Strong Sell Setups Found", f"{total_sells} Stocks")
        m_col3.metric("Total Active Table Tickers", f"{len(df)} Tickers")
        
        loaded_batches = ", ".join(sorted(list(st.session_state.scanned_batches)))
        st.info(f"📁 Current visible data layer: **{loaded_batches}**")
        
        st.markdown("---")
        st.subheader(f"📊 Live Signal Matrix Leaderboard — {datetime.now().strftime('%Y-%m-%d %H:%M')} EST")
        
        def color_whole_rows(row):
            if row['Action Signal'] == "🔥 STRONG BUY":
                return ['background-color: #1e4620; color: #ffffff; font-weight: bold;'] * len(row)
            elif row['Action Signal'] == "🚨 STRONG SELL":
                return ['background-color: #611f1d; color: #ffffff; font-weight: bold;'] * len(row)
            else:
                return ['background-color: #1a1c23; color: #a3a8b4;'] * len(row)
        
        display_df = df.drop(columns=['RawScore']) if 'RawScore' in df.columns else df
        styled_df = display_df.style.apply(color_whole_rows, axis=1)
        st.dataframe(styled_df, width="stretch", hide_index=True)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            display_df.to_excel(writer, index=False, sheet_name="Market Signals")
        
        st.download_button(
            label="📥 Download Active Report Layout as Excel",
            data=buffer.getvalue(),
            file_name=f"Trading_Signals_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("Dashboard view screen empty. Click any scanning trigger on the sidebar panel menu to initiate analysis.")

# Load dynamic web lists on initial script compilation execution
sp500_live, ndx_live = fetch_live_index_tickers()

# --- UI SIDEBAR INTERACTION BUTTONS ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Execution Control Center")
st.sidebar.write("Launch specific market scanning pipelines below.")

# 1. Isolated Custom Watchlist Mode
if st.sidebar.button("🚀 Run Custom Watchlist Scan"):
    with st.spinner("Analyzing custom watchlist tickers exclusively..."):
        st.session_state.stacked_results = pd.DataFrame()
        st.session_state.scanned_batches = {"Custom Watchlist"}
        st.session_state.active_mode = "Custom Watchlist"
        
        res_df = run_scanner(watchlist, is_discovery=False)
        st.session_state.stacked_results = res_df
        st.rerun()

# 2. Dynamic Batch 1 Trigger (First 50 of the live Nasdaq-100 index)
if st.sidebar.button("🔍 Scan Batch 1: Nasdaq Core (1-50)"):
    with st.spinner("Processing dynamic Nasdaq-100 Ranks 1-50..."):
        if st.session_state.active_mode != "Discovery Batches":
            st.session_state.stacked_results = pd.DataFrame()
            st.session_state.scanned_batches = set()
            st.session_state.active_mode = "Discovery Batches"
            
        target_tickers = ndx_live[:50]
        df1 = run_scanner(target_tickers, is_discovery=True)
        st.session_state.stacked_results = pd.concat([st.session_state.stacked_results, df1], ignore_index=True)
        st.session_state.scanned_batches.add("Nasdaq Batch 1")
        st.rerun()
        
# 3. Dynamic Batch 2 Trigger (Next 50 of the live Nasdaq-100 index)
if st.sidebar.button("⏭️ Scan Batch 2: Nasdaq Expansion (51-100)"):
    with st.spinner("Processing dynamic Nasdaq-100 Ranks 51-100..."):
        if st.session_state.active_mode != "Discovery Batches":
            st.session_state.stacked_results = pd.DataFrame()
            st.session_state.scanned_batches = set()
            st.session_state.active_mode = "Discovery Batches"
            
        target_tickers = ndx_live[50:100]
        df2 = run_scanner(target_tickers, is_discovery=True)
        st.session_state.stacked_results = pd.concat([st.session_state.stacked_results, df2], ignore_index=True)
        st.session_state.scanned_batches.add("Nasdaq Batch 2")
        st.rerun()
        
# 4. Dynamic Batch 3 Trigger (Top 50 Alpha Large/Mid Caps from S&P 500)
if st.sidebar.button("🔬 Scan Batch 3: S&P Alpha Layer (Top 50)"):
    with st.spinner("Processing dynamic S&P 500 Alpha Layer..."):
        if st.session_state.active_mode != "Discovery Batches":
            st.session_state.stacked_results = pd.DataFrame()
            st.session_state.scanned_batches = set()
            st.session_state.active_mode = "Discovery Batches"
            
        target_tickers = sp500_live[:50]
        df3 = run_scanner(target_tickers, is_discovery=True)
        st.session_state.stacked_results = pd.concat([st.session_state.stacked_results, df3], ignore_index=True)
        st.session_state.scanned_batches.add("S&P Alpha Batch 3")
        st.rerun()
        
# 5. Manual Clear System
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Screen & Reset Scanner", type="primary"):
    st.session_state.stacked_results = pd.DataFrame()
    st.session_state.scanned_batches = set()
    st.session_state.active_mode = "None"
    st.toast("Dashboard cache completely wiped clean!")
    st.rerun()
    
# Execute Drawing Routine
display_master_leaderboard()
