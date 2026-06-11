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

# Automated formatting function to guarantee proper exchange suffixes
def format_tickers(ticker_string, mode):
    raw_list = [t.strip().upper() for t in ticker_string.split(",") if t.strip()]
    if mode == "Indian Equities (NSE)":
        return [f"{t}.NS" if not t.endswith(".NS") and not t.endswith(".BO") else t for t in raw_list]
    return raw_list

user_input = st.sidebar.text_area(f"✍️ Edit Custom Watchlist Tickers ({market_mode}):", default_watchlist, height=100)
watchlist = format_tickers(user_input, market_mode)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Edit Discovery Groups")
with st.sidebar.expander("View / Edit Ticker Blocks"):
    st.caption("💡 These lists are now automatically pulled from live S&P 500 and Nifty 50 index data!")
    b1_input = st.text_area("Batch 1 Tickers:", b1_default, height=100)
    b2_input = st.text_area("Batch 2 Tickers:", b2_default, height=100)
    b3_input = st.text_area("Batch 3 Tickers:", b3_default, height=100)

batch_1_list = format_tickers(b1_input, market_mode)
batch_2_list = format_tickers(b2_input, market_mode)
batch_3_list = format_tickers(b3_input, market_mode)

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
    fast_ema_len, slow_ema_len, trend_ema_len = 8, 21, 200
    rsi_len, rsi_lower, rsi_upper = 14, 35, 65
    st_multiplier = 2.5
    regime_len, regime_threshold = 14, 1.2

    results = []
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Instead of a normal requests session, use a cached database
    import requests_cache

    # This creates a tiny database file that remembers stock prices for 12 hours
    session = requests_cache.CachedSession('yfinance.cache', expire_after=43200)
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    # Now when you call yf.download(..., session=session), 
    # it won't trigger Yahoo's alarms if you recently downloaded it!

    progress_bar = st.progress(0)
    
    for idx, ticker in enumerate(tickers):
        try:
            # --- Smart Retry Protocol for Yahoo Rate Limits ---
            max_attempts = 3
            df = pd.DataFrame()
            
            for attempt in range(max_attempts):
                df = yf.download(ticker, start=start_date, end=end_date, progress=False, session=session)
                
                # If we successfully got data, break out of the retry loop
                if not df.empty and len(df) > 0:
                    break
                
                # If it failed (rate limited), pause for 2.5 seconds before trying again
                time.sleep(2.5)
            # --------------------------------------------------
            
            if df.empty:
                st.sidebar.warning(f"⚠️ Yahoo blocked {ticker} - No data returned.")
                continue
            if len(df) < trend_ema_len:
                st.sidebar.warning(f"⚠️ Skipped {ticker} - Less than 200 days history.")
                continue
                
            # --- Bulletproof Row Extract: Grab data columns regardless of multiindex nesting labels ---
            col_strings = [str(c).lower() for c in df.columns]
            close_idx, high_idx, low_idx = -1, -1, -1
            
            for c_i, c_str in enumerate(col_strings):
                if 'close' in c_str: close_idx = c_i
                if 'high' in c_str: high_idx = c_i
                if 'low' in c_str: low_idx = c_i
                
            if close_idx == -1 or high_idx == -1 or low_idx == -1:
                continue

            raw_close = df.iloc[:, close_idx].values.flatten().astype('float64')
            raw_high = df.iloc[:, high_idx].values.flatten().astype('float64')
            raw_low = df.iloc[:, low_idx].values.flatten().astype('float64')

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
            
            final_ub = np.zeros(len(df))
            final_lb = np.zeros(len(df))
            st_dir_array = np.zeros(len(df))
            
            for i in range(1, len(df)):
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

            is_trending = market_volatility[-1] > regime_threshold
            ema_buy_trigger = fast_ema[-1] > slow_ema[-1] and fast_ema[-2] <= slow_ema[-2]
            ema_sell_trigger = fast_ema[-1] < slow_ema[-1] and fast_ema[-2] >= slow_ema[-2]
            above_macro_trend = raw_close[-1] > trend_ema[-1]
            
            st_bullish = st_dir_array[-1] == 1 if is_trending else True
            st_bearish = st_dir_array[-1] == -1 if is_trending else True

            rsi_bullish_div = rsi[-1] > rsi[-2] and raw_close[-1] <= raw_close[-2] and rsi[-1] < rsi_upper
            rsi_bearish_div = rsi[-1] < rsi[-2] and raw_close[-1] >= raw_close[-2] and rsi[-1] > rsi_lower

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

            regime_label = "TRENDING" if is_trending else "RANGING (Bypassed)"
            signal = "BUY" if buy_score >= 5 else "SELL" if sell_score >= 5 else "HOLD"
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

            if is_discovery:
                time.sleep(1.5)

        except Exception as e:
            st.sidebar.error(f"🛑 Code Crash on {ticker}: {e}")
            
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
        
# --- DYNAMIC POSITION SIZING & RISK DASHBOARD ---
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Risk & Sizing Calculator")

# Currency selector for multi-market flexibility
currency_symbol = st.sidebar.radio("Base Currency", ["USD ($)", "INR (₹)"], horizontal=True)
currency_char = "$" if "USD" in currency_symbol else "₹"

account_size = st.sidebar.number_input(f"Total Account Equity ({currency_char})", value=100000.0, step=1000.0)
risk_pct = st.sidebar.slider("Max Risk Per Trade (%)", 0.1, 5.0, 1.0, 0.1)

risk_allowance = account_size * (risk_pct / 100)
st.sidebar.info(f"**Max Capital Risked Per Trade:** {currency_char}{risk_allowance:,.2f}")

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
    """
    Renders an interactive Plotly layout chart for a selected ticker 
    and calculates dynamic position sizing based on native ATR.
    """
    df_results = st.session_state.stacked_results
    
    if not df_results.empty:
        st.markdown("---")
        st.subheader("📈 Institutional Charting & Risk Workspace")
        
        ticker_options = sorted(df_results['Ticker'].unique())
        selected_ticker = st.selectbox("🎯 Select an analyzed stock to visualize:", ticker_options)
        
        if selected_ticker:
            with st.spinner(f"Generating indicator chart for {selected_ticker}..."):
                end_date = datetime.today().strftime('%Y-%m-%d')
                start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
                
                session = requests.Session()
                session.headers.update({'User-Agent': 'Mozilla/5.0'})
                df_stock = yf.download(selected_ticker, start=start_date, end=end_date, progress=False, session=session)
                
                if not df_stock.empty and len(df_stock) >= 200:
                    col_strings = [str(c).lower() for c in df_stock.columns]
                    open_idx, close_idx, high_idx, low_idx = -1, -1, -1, -1
                    
                    for c_i, c_str in enumerate(col_strings):
                        if 'open' in c_str: open_idx = c_i
                        if 'close' in c_str: close_idx = c_i
                        if 'high' in c_str: high_idx = c_i
                        if 'low' in c_str: low_idx = c_i
                    
                    raw_open = df_stock.iloc[:, open_idx].values.flatten().astype('float64')
                    raw_close = df_stock.iloc[:, close_idx].values.flatten().astype('float64')
                    raw_high = df_stock.iloc[:, high_idx].values.flatten().astype('float64')
                    raw_low = df_stock.iloc[:, low_idx].values.flatten().astype('float64')
                    
                    fast_ema = compute_native_ema(raw_close.copy(), 8)
                    slow_ema = compute_native_ema(raw_close.copy(), 21)
                    trend_ema = compute_native_ema(raw_close.copy(), 200)
                    atr_np = compute_native_atr(raw_high, raw_low, raw_close, length=14)
                    
                    # Native SuperTrend logic
                    src = (raw_high + raw_low) / 2
                    basic_ub = src + (2.5 * atr_np)
                    basic_lb = src - (2.5 * atr_np)
                    final_ub = np.zeros(len(df_stock))
                    final_lb = np.zeros(len(df_stock))
                    st_dir = np.zeros(len(df_stock))
                    
                    for i in range(1, len(df_stock)):
                        if basic_ub[i] < final_ub[i-1] or raw_close[i-1] > final_ub[i-1]: final_ub[i] = basic_ub[i]
                        else: final_ub[i] = final_ub[i-1]
                        if basic_lb[i] > final_lb[i-1] or raw_close[i-1] < final_lb[i-1]: final_lb[i] = basic_lb[i]
                        else: final_lb[i] = final_lb[i-1]
                        
                        if raw_close[i] > final_ub[i]: st_dir[i] = 1
                        elif raw_close[i] < final_lb[i]: st_dir[i] = -1
                        else:
                            st_dir[i] = st_dir[i-1]
                            if st_dir[i] == 1 and final_lb[i] < final_lb[i-1]: final_lb[i] = final_lb[i-1]
                            if st_dir[i] == -1 and final_ub[i] > final_ub[i-1]: final_ub[i] = final_ub[i-1]
                    
                    chart_df = pd.DataFrame({
                        'Open': raw_open, 'High': raw_high, 'Low': raw_low, 'Close': raw_close,
                        '8 EMA': fast_ema, '21 EMA': slow_ema, '200 EMA': trend_ema,
                        'SuperTrend Upper': final_ub, 'SuperTrend Lower': final_lb, 'Direction': st_dir
                    }, index=df_stock.index).tail(90)
                    
                    chart_df['Active SuperTrend'] = np.where(chart_df['Direction'] == 1, chart_df['SuperTrend Lower'], chart_df['SuperTrend Upper'])
                    
                    # -- PLOTLY CANDLESTICK INTEGRATION --
                    fig = go.Figure()
                    
                    # Candlesticks
                    fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], 
                                                 low=chart_df['Low'], close=chart_df['Close'], name='Price',
                                                 increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
                    
                    # EMAs
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['8 EMA'], mode='lines', name='8 EMA', line=dict(color='#00d1ff', width=1.5)))
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['21 EMA'], mode='lines', name='21 EMA', line=dict(color='#ffb800', width=1.5)))
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['200 EMA'], mode='lines', name='200 EMA', line=dict(color='#ff0055', width=2)))
                    
                    # Dynamic SuperTrend Color Split
                    bullish_st = chart_df['Active SuperTrend'].where(chart_df['Direction'] == 1)
                    bearish_st = chart_df['Active SuperTrend'].where(chart_df['Direction'] == -1)
                    fig.add_trace(go.Scatter(x=chart_df.index, y=bullish_st, mode='lines', name='ST Bull Support', line=dict(color='#00ff66', width=2, dash='dot')))
                    fig.add_trace(go.Scatter(x=chart_df.index, y=bearish_st, mode='lines', name='ST Bear Res', line=dict(color='#ff3333', width=2, dash='dot')))

                    fig.update_layout(
                        template='plotly_dark',
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=550,
                        xaxis_rangeslider_visible=False,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, width="stretch")

                    # -- DYNAMIC POSITION SIZING EXECUTION --
                    current_price = chart_df['Close'].iloc[-1]
                    current_st_stop = chart_df['Active SuperTrend'].iloc[-1]
                    current_dir = chart_df['Direction'].iloc[-1]
                    
                    st.markdown("### 🧮 Live Position Sizing")
                    risk_dist = abs(current_price - current_st_stop)
                    
                    if risk_dist > 0:
                        position_size_shares = int(risk_allowance / risk_dist)
                        capital_required = position_size_shares * current_price
                        
                        sz_col1, sz_col2, sz_col3, sz_col4 = st.columns(4)
                        # Inside the position sizing metric card layout section of render_charting_layout():
                        sz_col1.metric("Current Entry Price", f"{currency_char}{current_price:.2f}")
                        sz_col2.metric("SuperTrend Stop Loss", f"{currency_char}{current_st_stop:.2f}")
                        sz_col3.metric("Recommended Shares", f"{position_size_shares:,}")
                        
                        # Capital allocation warning
                        if capital_required > account_size:
                            sz_col4.error(f"⚠️ Insufficient Buying Power")
                        else:
                            sz_col4.metric("Capital Allocated", f"{currency_char}{capital_required:,.2f}")
                    else:
                        st.warning("Risk distance is zero. Wait for valid volatility expansion.")
                        
                else:
                    st.error("Insufficient historical trading volume data found to map structural trend chart.")
                    
# --- APPLICATION FOOTPRINT MAP CHANGER ---
# Append the chart draw call to sit directly under your dashboard leaderboard drawer 
display_master_leaderboard()
render_charting_layout()
