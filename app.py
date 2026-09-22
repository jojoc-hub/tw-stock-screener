import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="台股多策略選股雷達", layout="wide")
st.title("📡 台股多策略技術面選股雷達")
st.markdown("自動掃描多種常見技術分析策略，只要符合任何一項特徵即列入觀察名單。")

# 預設觀察名單 (已加入台積電、川湖、國巨、強茂、國泰金、中信金、聯亞、華星光等)
default_tickers = "2330.TW, 2059.TW, 2327.TW, 2481.TW, 2882.TW, 2891.TW, 3081.TWO, 4979.TWO, 2317.TW, 2454.TW, 2603.TW"

st.sidebar.header("選股池設定")
ticker_input = st.sidebar.text_area("輸入股票代號 (逗號分隔)", value=default_tickers, height=150)
lookback_days = st.sidebar.slider("載入歷史天數 (用於計算長天期指標)", 100, 365, 180)

@st.cache_data
def get_and_calc_indicators(ticker, start, end):
    try:
        df = yf.download(ticker.strip(), start=start, end=end, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.dropna()
        
        # 1. 計算均線 (Moving Averages)
        df['5MA'] = df['Close'].rolling(window=5).mean()
        df['20MA'] = df['Close'].rolling(window=20).mean()
        df['60MA'] = df['Close'].rolling(window=60).mean()
        df['5VMA'] = df['Volume'].rolling(window=5).mean()
        
        # 2. 計算 MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 3. 計算 RSI (14日)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. 計算布林通道 (Bollinger Bands 20, 2)
        df['BB_Mid'] = df['20MA']
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        
        return df.dropna()
    except Exception as e:
        return None

if st.button("啟動全策略掃描", type="primary"):
    with st.spinner("正在下載資料並計算技術指標..."):
        tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
        end_date = datetime.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        results = []
        
        for ticker in tickers:
            df = get_and_calc_indicators(ticker, start_date, end_date)
            if df is not None and len(df) >= 2:
                # 取得最近兩日的資料來判斷「交叉」或「突破」動作
                last = df.iloc[-1]
                prev = df.iloc[-2]
                
                triggered_strategies = []
                
                # --- 策略 1：5MA 爆量突破 ---
                if last['Close'] > last['5MA'] and prev['Close'] <= prev['5MA'] and last['Volume'] > (last['5VMA'] * 1.5):
                    triggered_strategies.append("🔥 5MA爆量突破 (成交量>均量1.5倍)")
                    
                # --- 策略 2：5MA 量縮回測支撐 ---
                if last['Low'] <= last['5MA'] and last['Close'] > last['5MA'] and last['Volume'] < last['5VMA']:
                    triggered_strategies.append("📉 5MA量縮回測 (測底後收上)")
                    
                # --- 策略 3：均線黃金交叉 (5MA 上穿 20MA) ---
                if last['5MA'] > last['20MA'] and prev['5MA'] <= prev['20MA']:
                    triggered_strategies.append("✨ 均線黃金交叉 (5日線突破月線)")
                    
                # --- 策略 4：MACD 黃金交叉 ---
                if last['MACD'] > last['Signal'] and prev['MACD'] <= prev['Signal'] and last['MACD'] < 0:
                    triggered_strategies.append("📈 MACD 低檔黃金交叉")
                    
                # --- 策略 5：RSI 超賣區反彈 ---
                if last['RSI'] > 30 and prev['RSI'] <= 30:
                    triggered_strategies.append("🔄 RSI 超賣反彈 (跌深反彈訊號)")
                    
                # --- 策略 6：布林通道下軌支撐 ---
                if last['Low'] <= last['BB_Lower'] and last['Close'] > last['BB_Lower']:
                    triggered_strategies.append("🛡️ 布林通道下軌支撐 (收盤站回通道內)")

                # 如果有觸發任何策略，就加入結果清單
                if triggered_strategies:
                    results.append({
                        "股票代號": ticker,
                        "最新收盤價": round(last['Close'], 2),
                        "符合的策略清單": " \n+ ".join(triggered_strategies),
                        "今日成交量": int(last['Volume']),
                        "RSI 值": round(last['RSI'], 1)
                    })
        
        # 顯示結果
        if results:
            st.success(f"掃描完成！共發現 {len(results)} 檔股票出現技術面進場訊號。")
            # 轉換為 DataFrame 以表格呈現
            result_df = pd.DataFrame(results)
            st.dataframe(
                result_df, 
                use_container_width=True,
                column_config={
                    "符合的策略清單": st.column_config.TextColumn("符合的策略清單 (多項代表訊號越強)"),
                }
            )
        else:
            st.info("今日大盤/個股走勢平淡，目前觀察名單中無任何股票觸發上述技術面策略。建議擴大選股池或耐心等待。")
            
st.divider()
st.markdown("""
### 💡 策略說明：
1. **5MA爆量突破**：短線主力發動攻擊的常見訊號，適合動能交易。
2. **5MA量縮回測**：多頭趨勢中，短線洗盤結束的切入點。
3. **均線黃金交叉**：短天期成本突破長天期成本，波段翻多訊號。
4. **MACD 低檔黃金交叉**：趨勢由空轉多的早期動能指標。
5. **RSI 超賣反彈**：短線跌幅過深（小於30）後出現的技術性買盤。
6. **布林通道下軌支撐**：統計學上的極端偏離值，股價容易向均值回歸。
""")
