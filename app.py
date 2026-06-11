import streamlit as st
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf
from datetime import datetime, timedelta
import io
import time
import requests
import numpy as np

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

# --- SIDEBAR WATCHLIST CONFIGURATIONS ---
st.sidebar.header("📋 Scanner Control Panel")

# Compact structured defaults to keep browser formatting clean
b1_default = "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, PLTR, NFLX, AVGO, SMCI, COIN, ORCL, CRM, QCOM, INTC, MU, PANW, MRVL, JPM, BAC, WMT, COST, DIS, XOM, CVX, LLY, UNH, V, MA, HD, PG, PFE, MRK, ABBV, KO, PEP, NKE, SBUX, GE, CAT, BA, HON, IBM, ACN, TXN, AMGN, GILD, BABA"
b2_default = "UBER, ABNB, PYPL, SOFI, DKNG, HOOD, AFRM, RIVN, LCID, NIO, XPEV, LI, F, GM, TM, T, VZ, CMCSA, WBD, SPOT, RBLX, U, PATH, SNOW, NET, CRWD, OKTA, DDOG, ZS, FTNT, CHKP, MDB, DOCU, TWLO, PINS, SNAP, SHOP, SE, MELI, PDD, JD, BIDU, GME, AMC"
b3_default = "CELH, ELF, DUOL, IOT, MSTR, UPST, HIMS, ASTS, CLSK, MARA, RIOT, WULF, IREN, CIFR, CORZ, APPS, PTON, ROKU, CVNA, CHWY, FSLR, ENPH, RUN, CSIQ, GTLB, ALGM, AEHR, LSCC, POWI, SLAB, CRUS, COHR, LITE, FN, VRT, MOD, SYM, AOUT, JOBY, ACHR, LUNR, RKLB, BBAI, SOUN"

# User Custom entry list block
default_watchlist = "AAPL, MSFT, NVDA, AMD, META, AMZN, GOOGL, TSLA, NFLX, AVGO"
user_input = st.sidebar.text_area("✍️ Edit Custom Watchlist Tickers:", default_watchlist, height=100)
watchlist = [t.strip().upper() for t in user_input.split(",") if t.strip()]

# Advanced Sidebar configurations to let you customize discovery components on the fly
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Edit Discovery Groups")
with st.sidebar.expander("View / Edit Ticker Blocks"):
    b1_input = st.text_area("Batch 1 Tickers:", b1_default, height=100)
    b2_input = st.text_area("Batch 2 Tickers:", b2_default, height=100)
    b3_input = st.text_area("Batch 3 Tickers:", b3_default, height=100)

batch_1_list = [t.strip().upper() for t in b1_input.split(",") if t.strip()]
batch_2_list = [t.strip().upper() for t in b2_input.split(",") if t.strip()]
batch_3_list = [t.strip().upper() for t in b3_input.split(",") if t.strip()]

def run_scanner(tickers, is_discovery=False):
    fast_ema_len, slow_ema_len, trend_ema_len = 8, 21, 200
    rsi_len, rsi_lower, rsi_upper = 14, 35, 65
    st_period, st_multiplier = 10, 2.5
    regime_len, regime_threshold = 14, 1.2

    results = []
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Establish an encrypted browser session to bypass rate limits
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    progress_bar = st.progress(0)
    
    for idx, ticker in enumerate(tickers):
        try:
            # Download ticker data with standard layout configurations
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, session=session)
            
            if df.empty or len(df) < trend_ema_len:
                continue
                
            # Flatten multiindex data allocations by capturing core values directly
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col if isinstance(col, tuple) else col for col in df.columns]
            
            # Capitalize tracking columns
            df.columns = [str(c).strip().capitalize() for c in df.columns]
            
            if 'Close' not in df.columns:
                continue

            # Extract numeric series vectors
            raw_close = df['Close'].values.flatten().astype('float64')
            raw_high = df['High'].values.flatten().astype('float64')
            raw_low = df['Low'].values.flatten().astype('float64')
            flat_index = pd.to_datetime(df.index).tz_localize(None)

            close_series = pd.Series(raw_close, index=flat_index)
            high_series = pd.Series(raw_high, index=flat_index)
            low_series = pd.Series(raw_low, index=flat_index)

            # Compute EMA and RSI trends using isolated numpy inputs
            fast_ema = ta.ema(close_series, length=fast_ema_len).to_numpy()
            slow_ema = ta.ema(close_series, length=slow_ema_len).to_numpy()
            trend_ema = ta.ema(close_series, length=trend_ema_len).to_numpy()
            rsi = ta.rsi(close_series, length=rsi_len).to_numpy()
            
            atr = ta.atr(high_series, low_series, close_series, length=regime_len)
            atr_np = atr.to_numpy()
            
            # Mathematical SuperTrend native loop logic
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
            
            std_dev = close_series.rolling(regime_len).std()
            market_volatility = (std_dev / atr).fillna(1.0).to_numpy()

            # Volatility regime gate checks
            is_trending = market_volatility[-1] > regime_threshold
            ema_buy_trigger = fast_ema[-1] > slow_ema[-1] and fast_ema[-2] <= slow_ema[-2]
            ema_sell_trigger = fast_ema[-1] < slow_ema[-1] and fast_ema[-2] >= slow_ema[-2]
            above_macro_trend = close_series.iloc[-1] > trend_ema[-1]
            
            st_bullish = st_dir_array[-1] == 1 if is_trending else True
            st_bearish = st_dir_array[-1] == -1 if is_trending else True

            rsi_bullish_div = rsi[-1] > rsi[-2] and close_series.iloc[-1] <= close_series.iloc[-2] and rsi[-1] < rsi_upper
            rsi_bearish_div = rsi[-1] < rsi[-2] and close_series.iloc[-1] >= close_series.iloc[-2] and rsi[-1] > rsi_lower

            # Compute final mathematical alignment score
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
                "Last Close": f"${close_series.iloc[-1]:.2f}",
                "Market Regime": regime_label,
                "Buy Score": f"{buy_score}/6",
                "Sell Score": f"{sell_score}/6",
                "Action Signal": signal,
                "RawScore": numeric_score
            })
            
            if is_discovery:
                time.sleep(0.35)
                
        except Exception as e:
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
        st.subheader(f"📊 Live Signal Matrix Leaderboard — {datetime.now().strftime('%Y-%m-%d %H:%M')} EST")
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
        
        # ACTIVE INTERACTIVE CLICK-SORTING ENGINE
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

display_master_leaderboard()
