import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import io
import time
import requests
import numpy as np
import plotly.graph_objects as go

# Page Layout Configuration
st.set_page_config(page_title="AI Market Confluence Scanner", layout="wide")
st.title("📊 AI Market Confluence Live Dashboard")
st.write("Calculates 8/21 EMA triggers, RSI divergence, and automated SuperTrend regime bypass filters.")

# Initialize Session State Memory (Temporary browser database)
if "stacked_results" not in st.session_state:
    st.session_state.stacked_results = pd.DataFrame()
if "scanned_batches" not in st.session_state:
    st.session_state.scanned_batches = set()
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "None"

# --- GLOBAL MARKET SWITCHER & TICKER CONFIGURATIONS ---
st.sidebar.header("🗺️ Market Selection & Control Panel")

# Automated Web Scraper for Live Index Tickers
@st.cache_data(ttl=86400)  # Cache the data for 24 hours
def fetch_live_index(url, column_name):
    """Scrapes Wikipedia for live index constituents."""
    try:
        # Polite User-Agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        # Pandas 2.0+ requires text to be wrapped in StringIO
        import io
        tables = pd.read_html(io.StringIO(response.text))
        
        for df in tables:
            if column_name in df.columns:
                symbols = df[column_name].astype(str).str.replace('.', '-', regex=False).tolist()
                return symbols
        
        st.sidebar.error(f"Could not find a column named '{column_name}' on Wikipedia.")
        return []
    except Exception as e:
        # If it fails, print the EXACT error to the sidebar so we can see it!
        st.sidebar.error(f"Scraper Error: {e}")
        return []

market_mode = st.sidebar.selectbox("Select Target Market", ["US Equities (NASDAQ/NYSE)", "Indian Equities (NSE)"])

if market_mode == "US Equities (NASDAQ/NYSE)":
    currency_char = "$"
    # Dynamically fetch the S&P 500 and split it into batches
    sp500_live = fetch_live_index('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 'Symbol')
    
    if sp500_live:
        b1_default = ", ".join(sp500_live[:50])       # Top 50
        b2_default = ", ".join(sp500_live[50:100])    # Next 50
        b3_default = ", ".join(sp500_live[100:150])   # Next 50
    else:
        # Fallback just in case Wikipedia is down
        b1_default = "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA" 
        b2_default = ""
        b3_default = ""
        
    default_watchlist = "AAPL, MSFT, NVDA, AMD, META, AMZN, GOOGL, TSLA"

else:
    currency_char = "₹"
    # Dynamically fetch Nifty 50 and Nifty Next 50 (These work perfectly!)
    nifty50_live = fetch_live_index('https://en.wikipedia.org/wiki/NIFTY_50', 'Symbol')
    niftynext50_live = fetch_live_index('https://en.wikipedia.org/wiki/NIFTY_Next_50', 'Symbol')
    
    # Wikipedia doesn't have a Midcap 50 table, so we supply a robust, static list of top mid-caps
    midcap_50_static = [
        "AUBANK", "ASHOKLEY", "ASTRAL", "AUROPHARMA", "BALKRISIND", "BANDHANBNK", "BANKINDIA", 
        "BATAINDIA", "BHARATFORG", "BHEL", "BIOCON", "CANBK", "CHOLAFIN", "COFORGE", "CONCOR", 
        "COROMANDEL", "CROMPTON", "CUMMINSIND", "DEEPAKNTR", "DIXON", "ESCORTS", "FEDERALBNK", 
        "FORTIS", "GODREJPROP", "HAL", "HINDPETRO", "IDEA", "IDFCFIRSTB", "IGL", "INDHOTEL", 
        "INDUSTOWER", "JSWENERGY", "JUBLFOOD", "LALPATHLAB", "LAURUSLABS", "LICHSGFIN", 
        "MAXHEALTH", "MFSL", "MPHASIS", "MRF", "MUTHOOTFIN", "NAVINFLUOR", "PAGEIND", "PEL", 
        "PERSISTENT", "PETRONET", "PIIND", "POLYCAB", "PRESTIGE", "REC"
    ]
    
    if nifty50_live:
        b1_default = ", ".join(nifty50_live)              # All 50 of Nifty 50
        b2_default = ", ".join(niftynext50_live)          # All 50 of Nifty Next 50
        b3_default = ", ".join(midcap_50_static)          # 50 Institutional Midcaps
    else:
        b1_default = "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK"
        b2_default = ""
        b3_default = ""

    default_watchlist = "RELIANCE, TCS, INFY, HDFCBANK, TATAMOTORS, ZOMATO"

# Automated formatting function with a built-in Yahoo Translator
def format_tickers(ticker_string, mode):
    raw_list = [t.strip().upper() for t in ticker_string.split(",") if t.strip()]
    
    # 💡 The "Translation Dictionary" for Yahoo's weird database quirks
    yahoo_corrections = {
        "UNIONBK": "UNIONBANK",
        "REC": "RECLTD",
        "PEL": "PEL-EQ", # Yahoo occasionally alters Piramal's mapping
        "CAPLINPOINT": "CAPLIPOINT"
    }
    
    corrected_list = []
    for t in raw_list:
        # Temporarily remove .NS if it was already typed, so we can check the base name
        clean_name = t.replace(".NS", "")
        
        # If the base name is in our dictionary, swap it out for the Yahoo version
        if clean_name in yahoo_corrections:
            corrected_list.append(yahoo_corrections[clean_name])
        else:
            corrected_list.append(clean_name)
            
    # Re-attach the appropriate market suffix
    if mode == "Indian Equities (NSE)":
        return [f"{t}.NS" if not t.endswith(".BO") else t for t in corrected_list]
        
    return corrected_list

# ==============================================================================
# --- SIDEBAR UI: WATCHLISTS & CALCULATORS ---
# ==============================================================================

st.sidebar.markdown("---")

# 1. The Custom Watchlist Text Area (Always visible for quick access)
user_input = st.sidebar.text_area(f"✍️ Edit Custom Watchlist Tickers ({market_mode}):", default_watchlist, height=100)
watchlist = format_tickers(user_input, market_mode)

st.sidebar.markdown("---")

# 2. The Collapsible Discovery Batches
with st.sidebar.expander("⚙️ View / Edit Discovery Ticker Blocks", expanded=False):
    st.caption("💡 These lists are automatically pulled from live index data!")
    b1_input = st.text_area("Batch 1 Tickers:", b1_default, height=100)
    b2_input = st.text_area("Batch 2 Tickers:", b2_default, height=100)
    b3_input = st.text_area("Batch 3 Tickers:", b3_default, height=100)

batch_1_list = format_tickers(b1_input, market_mode)
batch_2_list = format_tickers(b2_input, market_mode)
batch_3_list = format_tickers(b3_input, market_mode)

st.sidebar.markdown("---")

# 3. The Collapsible Risk & Sizing Calculator
with st.sidebar.expander("💰 Risk & Sizing Calculator", expanded=False):
    base_currency = st.radio("Base Currency", ["USD ($)", "INR (₹)"], horizontal=True)
    account_equity = st.number_input("Total Account Equity", value=100000.00, step=1000.0)
    risk_pct = st.slider("Max Risk Per Trade (%)", 0.1, 5.0, 1.0, 0.1)
    
    max_risk_amount = account_equity * (risk_pct / 100)
    curr_symbol = "$" if "USD" in base_currency else "₹"
    
    st.info(f"**Max Capital Risked Per Trade:**\n\n{curr_symbol}{max_risk_amount:,.2f}")

#st.sidebar.markdown("---")
# ==============================================================================

# --- NATIVE MATHEMATICAL MATH ENGINES (NO EXTERNAL LIBRARIES REQUIRED) ---
def compute_native_ema(prices, length):
    """Calculates Exponential Moving Average mathematically."""
    alpha = 2 / (length + 1)
    ema = np.zeros(len(prices))
    if len(prices) > 0:
        ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def compute_native_rsi(prices, length=14):
    """Calculates Relative Strength Index mathematically."""
    if len(prices) <= length:
        return np.zeros(len(prices))
    deltas = np.diff(prices)
    seed = deltas[:length]
    up = seed[seed >= 0].sum() / length
    down = -seed[seed < 0].sum() / length
    rsi = np.zeros(len(prices))
    
    if down == 0:
        rsi[length] = 100
    else:
        rs = up / down
        rsi[length] = 100 - (100 / (1 + rs))
        
    for i in range(length + 1, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            up_chg = delta
            down_chg = 0.0
        else:
            up_chg = 0.0
            down_chg = -delta
        up = (up * (length - 1) + up_chg) / length
        down = (down * (length - 1) + down_chg) / length
        if down == 0:
            rsi[i] = 100
        else:
            rs = up / down
            rsi[i] = 100 - (100 / (1 + rs))
    return rsi

def compute_native_atr(high, low, close, length=14):
    """Calculates Average True Range mathematically."""
    tr = np.zeros(len(close))
    if len(close) > 0:
        tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i-1])
        lpc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hpc, lpc)
    
    atr = np.zeros(len(close))
    if len(close) >= length:
        atr[length-1] = np.mean(tr[:length])
        for i in range(length, len(close)):
            atr[i] = (atr[i-1] * (length - 1) + tr[i]) / length
    return atr

def run_scanner(tickers, is_discovery=False):
    if not tickers:
        return pd.DataFrame()
        
    fast_ema_len, slow_ema_len, trend_ema_len = 8, 21, 200
    rsi_len, rsi_lower, rsi_upper = 14, 35, 65
    st_multiplier = 2.5
    regime_len, regime_threshold = 14, 1.2

    results = []
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    progress_bar = st.progress(0)
    st.sidebar.info(f"⏳ Downloading 365-day history for {len(tickers)} stocks in ONE batch request...")

    try:
        # --- THE MAGIC FIX: ONE SINGLE NETWORK CALL ---
        # This completely bypasses the loop rate-limit blocks
        batch_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    except Exception as e:
        st.sidebar.error(f"🛑 Master Batch Download Failed: {e}")
        return pd.DataFrame()

    for idx, ticker in enumerate(tickers):
        try:
            # Extract the specific ticker's arrays from the master batch dataframe
            if len(tickers) > 1:
                if 'Close' not in batch_data or ticker not in batch_data['Close']:
                    continue
                raw_close = batch_data['Close'][ticker].dropna().values.astype('float64')
                raw_high = batch_data['High'][ticker].dropna().values.astype('float64')
                raw_low = batch_data['Low'][ticker].dropna().values.astype('float64')
                # Add Volume extraction
                raw_volume = batch_data['Volume'][ticker].dropna().values.astype('float64')
            else:
                raw_close = batch_data['Close'].dropna().values.astype('float64')
                raw_high = batch_data['High'].dropna().values.astype('float64')
                raw_low = batch_data['Low'].dropna().values.astype('float64')
                # Add Volume extraction
                raw_volume = batch_data['Volume'].dropna().values.astype('float64')
                
            # Ensure we have enough data to calculate the 200-Day EMA
            if len(raw_close) < trend_ema_len:
                continue

            # Calculate Native indicators securely
            fast_ema = compute_native_ema(raw_close.copy(), fast_ema_len)
            slow_ema = compute_native_ema(raw_close.copy(), slow_ema_len)
            trend_ema = compute_native_ema(raw_close.copy(), trend_ema_len)
            rsi = compute_native_rsi(raw_close.copy(), rsi_len)
            atr_np = compute_native_atr(raw_high, raw_low, raw_close, length=regime_len)

            # Native SuperTrend Calculation
            src = (raw_high + raw_low) / 2
            basic_ub = src + (st_multiplier * atr_np)
            basic_lb = src - (st_multiplier * atr_np)
            
            final_ub = np.zeros(len(raw_close))
            final_lb = np.zeros(len(raw_close))
            st_dir_array = np.zeros(len(raw_close))
            
            for i in range(1, len(raw_close)):
                if basic_ub[i] < final_ub[i-1] or raw_close[i-1] > final_ub[i-1]:
                    final_ub[i] = basic_ub[i]
                else:
                    final_ub[i] = final_ub[i-1]
                    
                if basic_lb[i] > final_lb[i-1] or raw_close[i-1] < final_lb[i-1]:
                    final_lb[i] = basic_lb[i]
                else:
                    final_lb[i] = final_lb[i-1]
                    
                if raw_close[i] > final_ub[i]:
                    st_dir_array[i] = 1
                elif raw_close[i] < final_lb[i]:
                    st_dir_array[i] = -1
                else:
                    st_dir_array[i] = st_dir_array[i-1]
                    if st_dir_array[i] == 1 and final_lb[i] < final_lb[i-1]:
                        final_lb[i] = final_lb[i-1]
                    if st_dir_array[i] == -1 and final_ub[i] > final_ub[i-1]:
                        final_ub[i] = final_ub[i-1]
        
            # Regime calculation
            rolling_std = pd.Series(raw_close).rolling(regime_len).std().to_numpy()
            market_volatility = np.where(atr_np > 0, rolling_std / atr_np, 1.0)
            is_trending = market_volatility[-1] > 1.0  # Volatility Threshold

            # Brain 1 Indicators (Trend)
            ema_buy_trigger = fast_ema[-1] > slow_ema[-1]
            ema_sell_trigger = fast_ema[-1] < slow_ema[-1]
            above_macro_trend = raw_close[-1] > trend_ema[-1]
            st_bullish = st_dir_array[-1] == 1 
            st_bearish = st_dir_array[-1] == -1 
            rsi_bullish_div = rsi[-1] > rsi[-2] and raw_close[-1] <= raw_close[-2] and rsi[-1] < rsi_upper
            rsi_bearish_div = rsi[-1] < rsi[-2] and raw_close[-1] >= raw_close[-2] and rsi[-1] > rsi_lower

            # Brain 2 Indicators (Mean Reversion / Bollinger Bands)
            bb_window = 20
            bb_sma = pd.Series(raw_close).rolling(window=bb_window).mean().to_numpy()
            bb_std = pd.Series(raw_close).rolling(window=bb_window).std().to_numpy()
            bb_upper = bb_sma + (2 * bb_std)
            bb_lower = bb_sma - (2 * bb_std)
            
            hitting_lower_bb = raw_close[-1] <= bb_lower[-1] or raw_close[-2] <= bb_lower[-2]
            hitting_upper_bb = raw_close[-1] >= bb_upper[-1] or raw_close[-2] >= bb_upper[-2]
            is_oversold = rsi[-1] < 35
            is_overbought = rsi[-1] > 65

            # --- The Volume Lie-Detector ---
            vol_sma = pd.Series(raw_volume).rolling(window=20).mean().to_numpy()
            # Require today's volume to be at least 10% higher than the 20-day average
            strong_volume_confirmed = raw_volume[-1] > (vol_sma[-1] * 1.1)
            
            # ==========================================
            # THE DUAL-BRAIN SCORING ENGINE
            # ==========================================
            buy_score = 0
            sell_score = 0

            if is_trending:
                regime_label = "TRENDING"
                if above_macro_trend: buy_score += 1
                else: sell_score += 1
                if ema_buy_trigger: buy_score += 2
                if ema_sell_trigger: sell_score += 2
                if st_bullish: buy_score += 1
                if st_bearish: sell_score += 1
                if rsi_bullish_div: buy_score += 2
                if rsi_bearish_div: sell_score += 2
                
                # Tiered Grading + Volume Risk Manager
                if buy_score >= 5:
                    signal = "🔥 SUPER BUY"
                elif buy_score == 4:
                    signal = "BUY" if strong_volume_confirmed else "⚠️ LOW VOL (Bypass)"
                elif sell_score >= 5:
                    signal = "🩸 SUPER SELL"
                elif sell_score == 4:
                    signal = "SELL" if strong_volume_confirmed else "⚠️ LOW VOL (Bypass)"
                else:
                    signal = "HOLD"
                
            else:
                regime_label = "RANGING (Mean-Reversion)"
                if hitting_lower_bb and is_oversold:
                    buy_score = 5 
                    sell_score = 0
                    # Bounces need volume too!
                    signal = "🔥 SUPER BUY" if strong_volume_confirmed else "⚠️ LOW VOL (Bypass)"
                elif hitting_upper_bb and is_overbought:
                    sell_score = 5 
                    buy_score = 0
                    signal = "🩸 SUPER SELL" if strong_volume_confirmed else "⚠️ LOW VOL (Bypass)"
                else:
                    signal = "HOLD"

            numeric_score = buy_score if buy_score >= sell_score else -sell_score

            results.append({
                "Ticker": ticker,
                "Last Close": f"${raw_close[-1]:.2f}",
                "Market Regime": regime_label,
                "Buy Score": f"{buy_score}/6",
                "Sell Score": f"{sell_score}/6",
                "Action Signal": signal,
                "RawScore": numeric_score
            })

            # The visual progress bar updates instantly now because the data is already downloaded
            if is_discovery:
                time.sleep(0.01) 

        except Exception as e:
            # If a single stock mathematically fails, quietly skip it without crashing the app
            pass
            
        progress_bar.progress((idx + 1) / len(tickers))
        
    return pd.DataFrame(results)
    
def display_master_leaderboard():
    df = st.session_state.stacked_results
    
    if not df.empty:
        df = df.drop_duplicates(subset=["Ticker"], keep="last")
        df['SortOrder'] = df['Action Signal'].map({'BUY': 0, 'SELL': 1, 'HOLD': 2})
        df = df.sort_values(by=["SortOrder", "RawScore"], ascending=[True, False]).drop(columns=['SortOrder'])
        
        total_buys = int(sum(df['Action Signal'] == "BUY"))
        total_sells = int(sum(df['Action Signal'] == "SELL"))
        
        st.subheader(f"🎯 Aggregated {st.session_state.active_mode} Analysis Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Strong Buy Setups Found", f"{total_buys} Stocks")
        m_col2.metric("Strong Sell Setups Found", f"{total_sells} Stocks")
        m_col3.metric("Total Active Table Tickers", f"{len(df)} Tickers")
        
        loaded_batches = ", ".join(sorted(list(st.session_state.scanned_batches)))
        st.info(f"📁 Current visible data layer: **{loaded_batches}**")
        
        st.markdown("---")
        # Inside display_master_leaderboard():
        time_suffix = "IST" if market_mode == "Indian Equities (NSE)" else "EST"
        st.subheader(f"📊 Live Signal Matrix Leaderboard — {datetime.now().strftime('%Y-%m-%d %H:%M')} {time_suffix}")
        st.caption("💡 Tip: Click on any column header name below to instantly re-sort the rows dynamically.")
        
        def color_whole_rows(row):
            if row['Action Signal'] == "BUY":
                return ['background-color: #1e4620; color: #ffffff; font-weight: bold;'] * len(row)
            elif row['Action Signal'] == "SELL":
                return ['background-color: #611f1d; color: #ffffff; font-weight: bold;'] * len(row)
            else:
                return ['background-color: #1a1c23; color: #a3a8b4;'] * len(row)
        
        display_df = df.drop(columns=['RawScore']) if 'RawScore' in df.columns else df
        styled_df = display_df.style.apply(color_whole_rows, axis=1)
        
        st.dataframe(
            styled_df, 
            width="stretch", 
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Buy Score": st.column_config.TextColumn("Buy Score"),
                "Sell Score": st.column_config.TextColumn("Sell Score"),
                "Action Signal": st.column_config.TextColumn("Action Signal")
            }
        )
        
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
        
# --- UI SIDEBAR INTERACTION BUTTONS ---
st.sidebar.markdown("---")

if st.sidebar.button("🚀 Run Custom Watchlist Scan"):
    with st.spinner("Analyzing custom watchlist tickers exclusively..."):
        st.session_state.stacked_results = pd.DataFrame()
        st.session_state.scanned_batches = {"Custom Watchlist"}
        st.session_state.active_mode = "Custom Watchlist"
        res_df = run_scanner(watchlist, is_discovery=False)
        st.session_state.stacked_results = res_df
        st.rerun()

if st.sidebar.button("🔍 Scan Batch 1: Mega-Caps (1-50)"):
    with st.spinner("Processing Ranks 1-50..."):
        if st.session_state.active_mode != "Discovery Batches":
            st.session_state.stacked_results = pd.DataFrame()
            st.session_state.scanned_batches = set()
            st.session_state.active_mode = "Discovery Batches"
        df1 = run_scanner(batch_1_list, is_discovery=True)
        st.session_state.stacked_results = pd.concat([st.session_state.stacked_results, df1], ignore_index=True)
        st.session_state.scanned_batches.add("Batch 1 (Mega)")
        st.rerun()

if st.sidebar.button("⏭️ Scan Batch 2: Large-Caps (51-100)"):
    with st.spinner("Processing Ranks 51-100..."):
        if st.session_state.active_mode != "Discovery Batches":
            st.session_state.stacked_results = pd.DataFrame()
            st.session_state.scanned_batches = set()
            st.session_state.active_mode = "Discovery Batches"
        df2 = run_scanner(batch_2_list, is_discovery=True)
        st.session_state.stacked_results = pd.concat([st.session_state.stacked_results, df2], ignore_index=True)
        st.session_state.scanned_batches.add("Batch 2 (Large)")
        st.rerun()

if st.sidebar.button("🔬 Scan Batch 3: Mid/Small-Caps (101-150)"):
    with st.spinner("Processing Ranks 101-150..."):
        if st.session_state.active_mode != "Discovery Batches":
            st.session_state.stacked_results = pd.DataFrame()
            st.session_state.scanned_batches = set()
            st.session_state.active_mode = "Discovery Batches"
        df3 = run_scanner(batch_3_list, is_discovery=True)
        st.session_state.stacked_results = pd.concat([st.session_state.stacked_results, df3], ignore_index=True)
        st.session_state.scanned_batches.add("Batch 3 (Mid/Small)")
        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Screen & Reset Scanner", type="primary"):
    st.session_state.stacked_results = pd.DataFrame()
    st.session_state.scanned_batches = set()
    st.session_state.active_mode = "None"
    st.toast("Dashboard cache completely wiped clean!")
    st.rerun()

# --- ADVANCED CHARTING VISUALIZATION ENGINE ---
def render_charting_layout():
    st.subheader("📈 Interactive Advanced Charting Workspace")
    
    # Check if we have successfully scanned any data yet
    if "stacked_results" not in st.session_state or st.session_state.stacked_results.empty:
        st.warning("Please execute a market scan from the sidebar to initialize the charting environment.")
        return

    # Automatically grab the list of valid tickers that were just scanned
    available_tickers = st.session_state.stacked_results["Ticker"].tolist()
    
    selected_ticker = st.selectbox("🎯 Select an analyzed stock to visualize:", available_tickers)
    
    # Download clean historical data for just this single selected ticker
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    df = yf.download(selected_ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        st.error("Unable to render chart: Data feed timed out.")
        return

    # Flat array transformations for native calculations
    raw_close = df['Close'].dropna().values.astype('float64')
    raw_high = df['High'].dropna().values.astype('float64')
    raw_low = df['Low'].dropna().values.astype('float64')
    raw_volume = df['Volume'].dropna().values.astype('float64')
    dates = df.index

    # 1. Compute Indicators for the entire historical dataset
    fast_ema = compute_native_ema(raw_close.copy(), 8)
    slow_ema = compute_native_ema(raw_close.copy(), 21)
    trend_ema = compute_native_ema(raw_close.copy(), 200)
    rsi = compute_native_rsi(raw_close.copy(), 14)
    atr = compute_native_atr(raw_high, raw_low, raw_close, length=14)
    
    # Bollinger Bands
    bb_sma = pd.Series(raw_close).rolling(window=20).mean().to_numpy()
    bb_std = pd.Series(raw_close).rolling(window=20).std().to_numpy()
    bb_upper = bb_sma + (2 * bb_std)
    bb_lower = bb_sma - (2 * bb_std)
    
    # Volume Baseline
    vol_sma = pd.Series(raw_volume).rolling(window=20).mean().to_numpy()

    # SuperTrend Arrays
    src = (raw_high + raw_low) / 2
    b_ub = src + (2.5 * atr)
    b_lb = src - (2.5 * atr)
    f_ub, f_lb = np.zeros(len(df)), np.zeros(len(df))
    st_dir = np.zeros(len(df))
    
    for i in range(1, len(df)):
        f_ub[i] = b_ub[i] if b_ub[i] < f_ub[i-1] or raw_close[i-1] > f_ub[i-1] else f_ub[i-1]
        f_lb[i] = b_lb[i] if b_lb[i] > f_lb[i-1] or raw_close[i-1] < f_lb[i-1] else f_lb[i-1]
        st_dir[i] = 1 if raw_close[i] > f_ub[i] else -1 if raw_close[i] < f_lb[i] else st_dir[i-1]
        if st_dir[i] == 1 and f_lb[i] < f_lb[i-1]: f_lb[i] = f_lb[i-1]
        if st_dir[i] == -1 and f_ub[i] > f_ub[i-1]: f_ub[i] = f_ub[i-1]

    # 2. Historical Signal Scanner Engine (Backtrack calculation for visual markers)
    buy_arrow_x, buy_arrow_y = [], []
    sell_arrow_x, sell_arrow_y = [], []

    for i in range(200, len(df)):
        rolling_std = pd.Series(raw_close[:i+1]).rolling(14).std().to_numpy()
        volatility = rolling_std[-1] / atr[i] if atr[i] > 0 else 1.0
        is_trending = volatility > 1.0
        strong_vol = raw_volume[i] > (vol_sma[i] * 1.1)

        b_score, s_score = 0, 0
        if raw_close[i] > trend_ema[i]: b_score += 1
        else: s_score += 1
        if fast_ema[i] > slow_ema[i]: b_score += 2
        else: s_score += 2
        if st_dir[i] == 1: b_score += 1
        else: s_score += 1

        if is_trending:
            if b_score >= 4 and strong_vol:
                buy_arrow_x.append(dates[i])
                buy_arrow_y.append(raw_low[i] * 0.98) # Place slightly below candle low
            elif s_score >= 4 and strong_vol:
                sell_arrow_x.append(dates[i])
                sell_arrow_y.append(raw_high[i] * 1.02) # Place slightly above candle high
        else:
            hit_lower = raw_close[i] <= bb_lower[i] or raw_close[i-1] <= bb_lower[i-1]
            hit_upper = raw_close[i] >= bb_upper[i] or raw_close[i-1] >= bb_upper[i-1]
            if hit_lower and rsi[i] < 35 and strong_vol:
                buy_arrow_x.append(dates[i])
                buy_arrow_y.append(raw_low[i] * 0.98)
            elif hit_upper and rsi[i] > 65 and strong_vol:
                sell_arrow_x.append(dates[i])
                sell_arrow_y.append(raw_high[i] * 1.02)

    # 3. Render Advanced Candlestick Charts using Plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)

    # Base Candlesticks
    fig.add_trace(go.Candlestick(x=dates, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price Action"), row=1, col=1)
    
    # Overlays
    fig.add_trace(go.Scatter(x=dates, y=fast_ema, line=dict(width=1.5), name="8 EMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=slow_ema, line=dict(width=1.5), name="21 EMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=trend_ema, line=dict(dash='dash'), name="200 Macro EMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=bb_upper, line=dict(dash='dot', width=1), name="Upper BB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=bb_lower, line=dict(dash='dot', width=1), name="Lower BB"), row=1, col=1)

    # 🎯 VISUAL TRIGGER MARKERS (The Upgrade)
    fig.add_trace(go.Scatter(
        x=buy_arrow_x, y=buy_arrow_y, mode='markers',
        marker=dict(symbol='triangle-up', size=12, line=dict(width=1)), name="Engine BUY Signal"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sell_arrow_x, y=sell_arrow_y, mode='markers',
        marker=dict(symbol='triangle-down', size=12, line=dict(width=1)), name="Engine SELL Signal"
    ), row=1, col=1)

    # Volume Subplot Layout
    fig.add_trace(go.Bar(x=dates, y=df['Volume'], name="Volume Feed"), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=vol_sma, line=dict(width=1.2), name="20 Vol SMA"), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=650, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # -- DYNAMIC POSITION SIZING EXECUTION --
    current_price = chart_df['Close'].iloc[-1]
    current_st_stop = chart_df['Active SuperTrend'].iloc[-1]
    current_dir = chart_df['Direction'].iloc[-1]
                    
                    st.markdown("### 🧮 Live Position Sizing")
                    risk_dist = abs(current_price - current_st_stop)
                    
                    if risk_dist > 0:
                        position_size_shares = int(max_risk_amount / risk_dist)
                        capital_required = position_size_shares * current_price
                        
                        sz_col1, sz_col2, sz_col3, sz_col4 = st.columns(4)
                        # Inside the position sizing metric card layout section of render_charting_layout():
                        sz_col1.metric("Current Entry Price", f"{currency_char}{current_price:.2f}")
                        sz_col2.metric("SuperTrend Stop Loss", f"{currency_char}{current_st_stop:.2f}")
                        sz_col3.metric("Recommended Shares", f"{position_size_shares:,}")
                        
                        # Capital allocation warning
                        if capital_required > account_equity:
                            sz_col4.error(f"⚠️ Insufficient Buying Power")
                        else:
                            sz_col4.metric("Capital Allocated", f"{currency_char}{capital_required:,.2f}")
                    else:
                        st.warning("Risk distance is zero. Wait for valid volatility expansion.")
                        
                else:
                    st.error("Insufficient historical trading volume data found to map structural trend chart.")
                    
# ==============================================================================
# --- MAIN WORKSPACE TERMINAL TABS ---
# ==============================================================================
# Create professional terminal tabs to separate data from visualization
tab_leaderboard, tab_charting = st.tabs([
    "📊 Live Signal Leaderboard", 
    "📈 Advanced Charting & Risk"
])

# Route the matrix to Tab 1
with tab_leaderboard:
    display_master_leaderboard()

# Route the interactive Plotly workspace to Tab 2
with tab_charting:
    render_charting_layout()
