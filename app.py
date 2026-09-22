import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="台股 5MA 主力選股與回測", layout="wide")
st.title("🎯 台股 5MA 技術面與主力進場選股系統")

# 預設觀察名單 (使用者可自行增刪)
default_tickers = "2330.TW, 2317.TW, 2454.TW, 2382.TW, 2308.TW, 2603.TW, 3034.TW, 3231.TW, 2376.TW, 3081.TWO"

st.sidebar.header("選股池與時間設定")
ticker_input = st.sidebar.text_area("輸入股票代號 (逗號分隔，上市加 .TW，上櫃加 .TWO)", value=default_tickers)
lookback_days = st.sidebar.slider("分析天數 (用於計算均線)", 30, 365, 180)
volume_multiplier = st.sidebar.number_input("主力爆量倍數 (成交量大於均量幾倍)", value=2.0, step=0.5)

@st.cache_data
def get_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker.strip(), start=start, end=end, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df['5MA'] = df['Close'].rolling(window=5).mean()
        df['5VMA'] = df['Volume'].rolling(window=5).mean()
        df['Prev_Close'] = df['Close'].shift(1)
        return df.dropna()
    except:
        return None

# --- 第一階段：盤後選股 ---
st.header("1. 今日策略選股結果")
if st.button("執行選股掃描"):
    with st.spinner("正在掃描股票池..."):
        tickers = [t.strip() for t in ticker_input.split(",")]
        end_date = datetime.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        selected_stocks = []
        
        for ticker in tickers:
            df = get_stock_data(ticker, start_date, end_date)
            if df is not None and not df.empty:
                last_day = df.iloc[-1]
                
                # 條件 1: 收盤價站上 5MA
                cond_5ma = last_day['Close'] > last_day['5MA']
                # 條件 2: 主力進場特徵 (成交量爆出 5日均量的設定倍數，且收紅K)
                cond_volume = last_day['Volume'] > (last_day['5VMA'] * volume_multiplier)
                cond_bullish = last_day['Close'] > last_day['Open']
                
                if cond_5ma and cond_volume and cond_bullish:
                    selected_stocks.append({
                        "股票代號": ticker,
                        "最新收盤價": round(last_day['Close'], 2),
                        "成交量": int(last_day['Volume']),
                        "5日均量": int(last_day['5VMA'])
                    })
        
        if selected_stocks:
            st.success(f"掃描完成！共挑出 {len(selected_stocks)} 檔符合「站上5MA且主力爆量」的股票。")
            st.dataframe(pd.DataFrame(selected_stocks), use_container_width=True)
        else:
            st.warning("今日無符合條件的股票。")

st.divider()

# --- 第二階段：個股策略回測 ---
st.header("2. 策略歷史回測")
st.markdown("測試「爆量站上5MA買進，跌破5MA賣出」的歷史績效")

test_ticker = st.text_input("輸入要回測的股票代號", value="2330.TW")
if st.button("執行回測"):
    with st.spinner("計算歷史回測數據..."):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365) # 回測近一年
        df = get_stock_data(test_ticker, start_date, end_date)
        
        if df is not None:
            # 產生進場訊號 (爆量且站上5MA)
            df['Buy_Signal'] = (df['Close'] > df['5MA']) & (df['Volume'] > df['5VMA'] * volume_multiplier) & (df['Close'] > df['Open'])
            
            capital = 1000000
            position = 0
            trades_count = 0
            equity_records = []

            for date, row in df.iterrows():
                # 出場邏輯：跌破 5MA
                if position > 0 and row['Close'] < row['5MA']:
                    capital = position * row['Close']
                    position = 0
                    trades_count += 1
                # 進場邏輯：訊號出現且空手
                elif position == 0 and row['Buy_Signal']:
                    position = capital / row['Close']
                    capital = 0

                current_equity = capital if position == 0 else position * row['Close']
                buy_and_hold = (1000000 / df.iloc[0]['Close']) * row['Close']
                
                equity_records.append({'Date': date, 'Strategy': round(current_equity), 'Benchmark': round(buy_and_hold)})

            res_df = pd.DataFrame(equity_records).set_index('Date')
            final_value = res_df['Strategy'].iloc[-1]
            return_pct = ((final_value - 1000000) / 1000000) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("策略總報酬率", f"{return_pct:.2f}%")
            col2.metric("期末總資產", f"${final_value:,.0f}")
            col3.metric("交易次數", f"{trades_count} 次")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Strategy'], name='主力爆量 5MA 策略', line=dict(color='#ef4444', width=2)))
            fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Benchmark'], name='大盤/買入持有', line=dict(color='#94a3b8', dash='dash')))
            fig.update_layout(title=f"{test_ticker} 資金曲線圖", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("無法取得該股票資料。")
