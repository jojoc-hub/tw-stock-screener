import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from itertools import product

st.set_page_config(page_title="台股週期自適應選股雷達", layout="wide")
st.title("📡 台股週期自適應選股雷達")
st.caption("自動比較 500+ 組策略參數，依短、中、長週期與市場狀態選出近期樣本外表現較佳的模型。僅供研究，不構成投資建議。")

STOCK_NAMES = {}

st.sidebar.header("全上市股票與回測設定")
st.sidebar.success("選股範圍：TWSE 全部上市公司普通股，程式啟動時自動更新代號及名稱")
years = st.sidebar.slider("回測年數", 2, 10, 5)
min_trades = st.sidebar.slider("每個策略最少訊號數", 10, 100, 25)
fee_bps = st.sidebar.number_input("單次進出總成本（bps）", 0, 200, 40, 5)
top_n = st.sidebar.slider("顯示標的數", 3, 50, 20)
batch_size = st.sidebar.slider("每批下載檔數", 20, 100, 50, 10)

TWSE_COMPANY_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"

@st.cache_data(ttl=86400, show_spinner=False)
def get_twse_universe():
    """抓取證交所上市公司基本資料，只保留四位數普通股代號。"""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(TWSE_COMPANY_URL, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for item in data:
        code = str(item.get("公司代號", "")).strip()
        name = str(item.get("公司簡稱", item.get("公司名稱", code))).strip()
        # 上市公司的普通股代號為四位數；排除 ETF、ETN、權證及其他非公司證券。
        if len(code) == 4 and code.isdigit():
            rows.append((f"{code}.TW", name))
    if not rows:
        raise RuntimeError("TWSE 公司清單為空")
    return dict(sorted(set(rows)))

HORIZONS = {"短線（5日）": 5, "波段（20日）": 20, "中期（60日）": 60}

@dataclass(frozen=True)
class Strategy:
    fast: int
    slow: int
    rsi_min: int
    rsi_max: int
    vol_ratio: float
    breakout: int
    market_filter: bool

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers, start, end, batch_size=50):
    """分批下載全上市公司，避免單次請求過大；失敗批次會重試一次。"""
    out = {}
    progress = st.progress(0, text="下載上市股票歷史行情")
    total_batches = max(1, (len(tickers) + batch_size - 1) // batch_size)
    for batch_no, i in enumerate(range(0, len(tickers), batch_size), start=1):
        batch = tickers[i:i + batch_size]
        raw = None
        for attempt in range(2):
            try:
                raw = yf.download(batch, start=start, end=end, auto_adjust=True,
                                  progress=False, group_by="ticker", threads=True,
                                  timeout=30)
                break
            except Exception:
                if attempt == 0:
                    time.sleep(1.0)
        if raw is not None and not raw.empty:
            if len(batch) == 1:
                out[batch[0]] = raw.dropna(how="all")
            else:
                for t in batch:
                    try:
                        d = raw[t].dropna(how="all").copy()
                        if not d.empty:
                            out[t] = d
                    except Exception:
                        pass
        progress.progress(batch_no / total_batches,
                          text=f"下載上市股票歷史行情：{batch_no}/{total_batches} 批")
    progress.empty()
    return out

def indicators(df):
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    for n in [5, 10, 20, 40, 60, 120, 200]:
        d[f"MA{n}"] = c.rolling(n).mean()
    d["VMA20"] = v.rolling(20).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    d["VOL20"] = c.pct_change().rolling(20).std() * np.sqrt(252)
    for n in [20, 55]:
        d[f"HH{n}"] = h.shift(1).rolling(n).max()
    d["MOM20"] = c.pct_change(20)
    d["MOM60"] = c.pct_change(60)
    return d

def market_regime(bench):
    b = indicators(bench).dropna()
    if b.empty:
        return "資料不足", 0.5, 0.5
    last = b.iloc[-1]
    vol_rank = b["VOL20"].rank(pct=True).iloc[-1]
    trend = int(last["Close"] > last["MA60"]) + int(last["MA60"] > last["MA120"])
    if trend == 2 and vol_rank < 0.75:
        regime = "多頭趨勢"
    elif trend == 2:
        regime = "高波動多頭"
    elif trend == 0:
        regime = "空頭／防禦"
    else:
        regime = "震盪／轉折"
    return regime, float(vol_rank), float(last["Close"] / last["MA120"] - 1)

def strategy_library():
    # 3×3×3×2×3×2×2 = 648 組，扣除 fast>=slow 後仍超過 500 組。
    configs = []
    for x in product([5, 10, 20], [40, 60, 120], [40, 45, 50],
                     [70, 80], [0.8, 1.0, 1.2], [20, 55], [False, True]):
        s = Strategy(*x)
        if s.fast < s.slow:
            configs.append(s)
    return configs

def signal(d, s, bench_filter):
    cond = (
        (d["Close"] > d[f"MA{s.fast}"]) &
        (d[f"MA{s.fast}"] > d[f"MA{s.slow}"]) &
        (d["RSI"] >= s.rsi_min) & (d["RSI"] <= s.rsi_max) &
        (d["Volume"] >= d["VMA20"] * s.vol_ratio) &
        (d["Close"] >= d[f"HH{s.breakout}"] * 0.98)
    )
    if s.market_filter:
        cond &= bench_filter.reindex(d.index).ffill().fillna(False)
    return cond.fillna(False)

def evaluate_strategy(frames, s, horizon, bench_filter, cost):
    rows = []
    for ticker, d in frames.items():
        sig = signal(d, s, bench_filter)
        future = d["Close"].shift(-horizon) / d["Close"] - 1 - cost
        tmp = pd.DataFrame({"ret": future, "sig": sig}).dropna()
        tmp = tmp[tmp["sig"]]
        if not tmp.empty:
            tmp["ticker"] = ticker
            rows.append(tmp[["ret", "ticker"]])
    if not rows:
        return None
    r = pd.concat(rows)["ret"]
    if len(r) < min_trades or r.std(ddof=0) == 0:
        return None
    # 避免只追逐平均報酬：報酬、勝率、尾部風險共同評分。
    mean, std = r.mean(), r.std(ddof=0)
    sharpe_like = mean / std
    win_rate = (r > 0).mean()
    downside = abs(r.quantile(0.10)) + 1e-9
    score = sharpe_like * 0.55 + (win_rate - 0.5) * 0.30 + (mean / downside) * 0.15
    return {"score": score, "avg_return": mean, "win_rate": win_rate,
            "signals": len(r), "p10": r.quantile(0.10)}

def choose_strategy(frames, horizon, bench_filter, cost):
    # 時間切割：前 70% 僅作訓練，後 30% 作驗證；以驗證結果選模，減少過度擬合。
    dates = sorted(set().union(*[set(d.index) for d in frames.values()]))
    split = dates[int(len(dates) * 0.70)]
    train = {t: d[d.index < split] for t, d in frames.items()}
    valid = {t: d[d.index >= split] for t, d in frames.items()}
    candidates = []
    for s in strategy_library():
        tr = evaluate_strategy(train, s, horizon, bench_filter, cost)
        if tr is not None:
            candidates.append((tr["score"], s))
    candidates.sort(key=lambda x: x[0], reverse=True)
    # 只把訓練前 20 名帶入驗證，最後由驗證集決勝。
    validated = []
    for _, s in candidates[:20]:
        va = evaluate_strategy(valid, s, horizon, bench_filter, cost)
        if va is not None:
            validated.append((va["score"], s, va))
    return max(validated, key=lambda x: x[0]) if validated else None

def rank_today(frames, strategy, horizon, bench_filter):
    picks = []
    for ticker, d in frames.items():
        if len(d) < 130:
            continue
        sig = signal(d, strategy, bench_filter)
        last = d.iloc[-1]
        if not bool(sig.iloc[-1]):
            continue
        atr = float(last["ATR"])
        px = float(last["Close"])
        risk_mult = 1.5 if horizon <= 5 else (2.0 if horizon <= 20 else 2.5)
        reward_mult = 2.0 if horizon <= 5 else (3.0 if horizon <= 20 else 4.0)
        stop = max(px - risk_mult * atr, float(last[f"MA{strategy.fast}"]) * 0.97)
        target = px + reward_mult * atr
        strength = (float(last["MOM20"]) * 0.55 + float(last["MOM60"]) * 0.30
                    - float(last["VOL20"]) * 0.15)
        picks.append({
            "股票代號": ticker.replace(".TW", "").replace(".TWO", ""),
            "股票名稱": STOCK_NAMES.get(ticker, ticker),
            "最新價": px, "RSI": float(last["RSI"]),
            "20日動能": float(last["MOM20"]), "年化波動": float(last["VOL20"]),
            "綜合強度": strength, "參考停損": stop, "參考目標": target
        })
    return pd.DataFrame(picks).sort_values("綜合強度", ascending=False).head(top_n) if picks else pd.DataFrame()

if st.button("執行週期分析與選股", type="primary"):
    try:
        universe = get_twse_universe()
    except Exception as exc:
        st.error(f"無法取得 TWSE 上市公司清單：{exc}")
        st.stop()
    STOCK_NAMES.update(universe)
    tickers = list(universe.keys())
    # 0050 只作大盤狀態代理，不列入公司選股結果。
    download_tickers = ["0050.TW"] + tickers
    st.write(f"本次已載入 {len(tickers):,} 家上市公司；實際完成行情下載的檔數會顯示於下方。")
    end = datetime.today() + timedelta(days=1)
    start = end - timedelta(days=int(years * 365.25) + 250)
    with st.spinner("下載資料、建立 500+ 組策略並執行時間切割驗證..."):
        raw = download_prices(download_tickers, start, end, batch_size)
        if "0050.TW" not in raw:
            st.error("無法取得 0050.TW，請稍後再試或檢查網路。")
            st.stop()
        frames = {t: indicators(d).dropna() for t, d in raw.items() if len(d) >= 250}
        st.caption(f"Yahoo Finance 成功取得 {max(0, len(frames) - 1):,} 家上市公司有效歷史資料。")
        bench = frames["0050.TW"]
        stock_frames = {t: d for t, d in frames.items() if t != "0050.TW"}
        bench_filter = (bench["Close"] > bench["MA60"]) & (bench["MA60"] > bench["MA120"])
        regime, vol_rank, trend_gap = market_regime(raw["0050.TW"])
        c1, c2, c3 = st.columns(3)
        c1.metric("目前市場狀態", regime)
        c2.metric("20日波動百分位", f"{vol_rank:.0%}")
        c3.metric("0050 距120日線", f"{trend_gap:.1%}")
        st.info("週期轉換觀察：60日線與120日線決定趨勢方向，20日波動百分位辨識風險升溫；市場由多頭轉震盪時，通常先看到趨勢差收斂與波動升高。此為模型定義，不是未來保證。")
        tabs = st.tabs(list(HORIZONS.keys()))
        cost = fee_bps / 10000
        for tab, (label, horizon) in zip(tabs, HORIZONS.items()):
            with tab:
                best = choose_strategy(stock_frames, horizon, bench_filter, cost)
                if best is None:
                    st.warning("有效訊號不足，請擴大股票池、增加回測年數或降低最少訊號數。")
                    continue
                _, s, stats = best
                st.subheader(f"{label}：近期樣本外最佳候選")
                a, b, c, d = st.columns(4)
                a.metric("驗證期平均報酬", f"{stats['avg_return']:.2%}")
                b.metric("驗證期勝率", f"{stats['win_rate']:.1%}")
                c.metric("驗證期訊號數", f"{stats['signals']}")
                d.metric("驗證期10%分位", f"{stats['p10']:.2%}")
                st.code(str(asdict(s)), language="python")
                picks = rank_today(stock_frames, s, horizon, bench_filter)
                if picks.empty:
                    st.info("目前沒有標的同時通過此週期的趨勢、動能、量能與突破條件。")
                else:
                    st.dataframe(picks, use_container_width=True, hide_index=True,
                        column_config={
                            "最新價": st.column_config.NumberColumn(format="%.2f"),
                            "RSI": st.column_config.NumberColumn(format="%.1f"),
                            "20日動能": st.column_config.NumberColumn(format="%.2%%"),
                            "年化波動": st.column_config.NumberColumn(format="%.2%%"),
                            "綜合強度": st.column_config.NumberColumn(format="%.4f"),
                            "參考停損": st.column_config.NumberColumn(format="%.2f"),
                            "參考目標": st.column_config.NumberColumn(format="%.2f")
                        })

st.divider()
st.markdown("""
### 方法與限制
- 策略庫超過 500 組，但不是聲稱涵蓋網路上所有策略；大量策略其實只是同一因子的參數變形。
- 使用時間切割驗證、交易成本、最低訊號數與尾部風險評分，降低資料探勘偏誤。
- Yahoo Finance 可能有缺值、延遲或調整差異；正式交易前應改用可靠行情源，加入滑價、漲跌停、停牌、流動性與全市場存活者偏誤處理。
- 「最佳」只代表全部上市公司中可成功取得資料的股票、所選期間、成本與驗證窗格中的相對最佳，不代表未來報酬。
""")
