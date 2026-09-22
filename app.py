import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="台股多策略選股與操作規劃", layout="wide")
st.title("📡 台股多策略技術面雷達與操作規劃")
st.markdown("掃描技術面進場訊號，並自動計算建議的進場、停損與停利參考價位。")

# 內建常見股名對應字典 (包含您的持股與熱門權值股，提升讀取速度)
STOCK_NAMES = {
    "0050.TW": "元大台灣50", "006208.TW": "富邦台50", "00631L.TW": "元大台灣50正2",
    "2059.TW": "川湖", "2327.TW": "國巨", "2330.TW": "台積電", 
    "2481.TW": "強茂", "2882.TW": "國泰金", "2891.TW": "中信金", 
    "3081.TWO": "聯亞", "4979.TWO": "華星光", "2317.TW": "鴻海",
    "2454.TW": "聯發科", "2603.TW": "長榮", "3231.TW": "緯創"
}

# 預設選股池
default_tickers = "0050.TW, 00631L.TW, 2330.TW, 2059.TW, 2327.TW, 2481.TW, 2882.TW, 2891.TW, 3081.TWO, 4979.TWO, 2317.TW"

st.sidebar.header("選股池與參數設定")
ticker_input = st.sidebar.text_area("輸入股票代號 (逗號分隔)", value=default_tickers, height=120)
lookback_days = st.sidebar.slider("歷史資料天數 (用於計算均線)", 100, 365, 180)

def get_stock_name(ticker):
    """取得中文股名，若字典沒有則嘗試從 Yahoo 抓取，再沒有則顯示代號"""
    if ticker in STOCK_NAMES:
        return STOCK_NAMES[ticker]
    try:
        info = yf.Ticker(ticker).info
        return info.get('shortName', ticker)
    except:
        return ticker

@st.cache_data
def get_and_calc_indicators(ticker, start, end):
    try:
        df = yf.download(ticker.strip(), start=start, end=end, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.dropna()
        # 均線
        df['5MA'] = df['Close'].rolling(window=5).mean()
        df['20MA'] = df['Close'].rolling(window=20).mean()
        df['5VMA'] = df['Volume'].rolling(window=5).mean()
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        # 布林通道
        df['BB_Mid'] = df['20MA']
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        
        return df.dropna()
    except:
        return None

if st.button("啟動策略掃描", type="primary"):
    with st.spinner("資料運算中，請稍候..."):
        tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
        end_date = datetime.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        results = []
        
        for ticker in tickers:
            df = get_and_calc_indicators(ticker, start_date, end_date)
            if df is not None and len(df) >= 5:
                last = df.iloc[-1]
                prev = df.iloc[-2]
                current_price = round(last['Close'], 2)
                recent_high = df['High'].tail(10).max()
                recent_low = df['Low'].tail(5).min()
                
                strategies = []
                # 預設操作規劃
                entry_point = f"{current_price} 附近"
                stop_loss = round(recent_low * 0.97, 2) # 預設跌破近5日低點再多抓3%緩衝
                take_profit = round(recent_high * 1.05, 2) # 預設過近期高點5%
                
                # --- 策略 1：5MA 爆量突破 ---
                if last['Close'] > last['5MA'] and prev['Close'] <= prev['5MA'] and last['Volume'] > (last['5VMA'] * 1.5):
                    strategies.append("🔥 5MA爆量突破")
                    stop_loss = round(last['5MA'] * 0.98, 2) # 跌破5日線2%停損
                    take_profit = round(current_price * 1.15, 2) # 強勢股抓15%停利
                    
                # --- 策略 2：5MA 量縮回測 ---
                elif last['Low'] <= last['5MA'] and last['Close'] > last['5MA'] and last['Volume'] < last['5VMA']:
                    strategies.append("📉 5MA量縮回測")
                    entry_point = f"{round(last['5MA'], 2)} (貼近5日線)"
                    stop_loss = round(last['5MA'] * 0.97, 2)
                    take_profit = round(current_price * 1.10, 2)
                    
                # --- 策略 3：MACD 黃金交叉 ---
                elif last['MACD'] > last['Signal'] and prev['MACD'] <= prev['Signal'] and last['MACD'] < 0:
                    strategies.append("📈 MACD 低檔黃金交叉")
                    stop_loss = round(recent_low, 2) # 守前波低點
                    take_profit = round(last['20MA'] * 1.05, 2) # 以月線之上為目標
                    
                # --- 策略 4：布林通道下軌支撐 ---
                elif last['Low'] <= last['BB_Lower'] and last['Close'] > last['BB_Lower']:
                    strategies.append("🛡️ 布林通道下軌反彈")
                    stop_loss = round(last['BB_Lower'] * 0.98, 2) # 跌破下軌2%停損
                    take_profit = round(last['BB_Mid'], 2) # 停利先看中軌(月線)

                if strategies:
                    results.append({
                        "股票代號": ticker.replace(".TW", "").replace(".TWO", ""),
                        "股票名稱": get_stock_name(ticker),
                        "最新收盤價": current_price,
                        "符合策略": " + ".join(strategies),
                        "建議進場區間": entry_point,
                        "嚴格停損價": stop_loss,
                        "波段停利價": take_profit
                    })
        
        if results:
            st.success(f"掃描完成！共挑出 {len(results)} 檔符合策略的標的。")
            
            # 使用 Streamlit 內建的 DataFrame 顯示，並自訂欄位樣式
            df_display = pd.DataFrame(results)
            st.dataframe(
                df_display, 
                use_container_width=True,
                hide_index=True,
                column_config={
                    "最新收盤價": st.column_config.NumberColumn(format="%.2f"),
                    "嚴格停損價": st.column_config.NumberColumn(format="%.2f"),
                    "波段停利價": st.column_config.NumberColumn(format="%.2f"),
                }
            )
        else:
            st.info("今日無股票觸發技術面策略。量化交易的優勢在於：沒有訊號就空手觀望，避免非理性交易。")

st.divider()
st.markdown("""
### 💡 操作規劃說明
* **嚴格停損價**：依據策略特性計算（如跌破均線或跌破布林下軌）。**觸價應無條件執行，以保護本金。**
* **波段停利價**：根據近期高點或下一個壓力位（如月線）計算的預期目標。若股價強勢可改採「跌破5日線」的移動停利法來放大獲利。
""")
