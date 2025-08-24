import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# ---------------------------------------------------------------------
@st.cache_resource
def carregar_universo(path="dados/ativos_totais.xlsx"):
    """Carrega o universo de ativos a partir de um arquivo Excel."""
    try:
        return pd.read_excel(path)
    except Exception as e:
        st.error(f"Erro ao carregar universo: {e}")
        return pd.DataFrame()
# ---------------------------------------------------------------------

def mostrar_kpis_preco(dados: pd.DataFrame, ticker: str, info: dict | None):
    """Exibe 8 KPIs em caixinhas (st.metric) com labels em inglês, 2 linhas x 4 colunas."""
    st.subheader("📌 Visão Geral do Preço")

    if dados.empty:
        st.info("Sem dados para mostrar KPIs.")
        return

    # --- Cálculos básicos ---
    ultimo = float(dados["Close"].iloc[-1])
    primeiro = float(dados["Close"].iloc[0])
    ret_total = ultimo / primeiro - 1.0

    # YTD Return
    hoje = pd.Timestamp.today()
    ytd_inicio = pd.Timestamp(year=hoje.year, month=1, day=1)
    if ytd_inicio in dados.index:
        ytd_close0 = float(dados.loc[ytd_inicio, "Close"])
    else:
        mask = dados.index >= ytd_inicio
        ytd_close0 = float(dados.loc[mask, "Close"].iloc[0]) if mask.any() else primeiro
    ret_ytd = ultimo / ytd_close0 - 1.0

    # Volume médio 20d (se houver)
    vol_medio = np.nan
    if "Volume" in dados.columns:
        vol_medio = float(dados["Volume"].rolling(20).mean().iloc[-1])

    # Faixa 52 semanas
    hi52 = float(dados["High"].rolling(252).max().iloc[-1])
    lo52 = float(dados["Low"].rolling(252).min().iloc[-1])

    # 52-Week Change (usa info se existir; senão, aproxima pelo histórico)
    change_52_info = (info or {}).get("52WeekChange", None)
    if change_52_info is not None and not (isinstance(change_52_info, float) and np.isnan(change_52_info)):
        change_52 = float(change_52_info)
    else:
        if len(dados) >= 252:
            change_52 = ultimo / float(dados["Close"].iloc[-252]) - 1.0
        else:
            change_52 = np.nan

    # Market Cap (se disponível em info)
    market_cap = (info or {}).get("marketCap", None)

    # --- Linha 1 ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", f"${ultimo:,.2f}")
    c2.metric("Period Return", f"{ret_total:.2%}")
    c3.metric("YTD Return", f"{ret_ytd:.2%}")
    c4.metric("20d Avg Volume", f"{vol_medio:,.0f}" if not np.isnan(vol_medio) else "N/A")

    # --- Linha 2 ---
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("52-Week High", f"${hi52:,.2f}")
    c6.metric("52-Week Low", f"${lo52:,.2f}")
    c7.metric("52-Week Change", f"{change_52:.2%}" if not np.isnan(change_52) else "N/A")
    c8.metric("Market Cap", formatar_valor(market_cap, "moeda") if market_cap is not None else "N/A")


# ---------------------------------------------------------------------

def formatar_valor(valor, tipo="moeda"):
    if pd.isna(valor) or valor is None:
        return "N/A"
    try:
        if tipo == "moeda":
            return f"${valor:,.2f}"
        elif tipo == "porcentagem":
            return f"{valor * 100:.2f}%"
        elif tipo == "inteiro":
            return f"{int(valor):,}"
        else:
            return str(valor)
    except:
        return "N/A"

def mostrar_detalhes_fundamentalistas(
    ticker: str,
    tipo_ativo: str | None = None,
    dados_preco: pd.DataFrame | None = None
):
    """
    Exibe '📊 Fundamental Details' sem expander.
    - STOCK: blocos fundamentalistas clássicos.
    - ETF: perfil de ETF (categoria, AUM, NAV, expense ratio, retornos).
    - INDEX: snapshot quantitativo a partir de preços (1M/3M/6M/1Y, vol 1Y, 52W range, MDD).
    Labels em inglês.
    """
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import streamlit as st

    # --- Helpers locais ---
    def _epoch_to_date_txt(v):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return pd.to_datetime(v, unit="s").date().isoformat()
        except Exception:
            return "N/A"

    def _inferir_tipo(info_dict: dict) -> str:
        qt = (info_dict.get("quoteType") or "").upper()
        if tipo_ativo is not None:
            return tipo_ativo.upper()
        if "ETF" in qt or "FUND" in qt or "MUTUALFUND" in qt:
            return "ETF"
        if "INDEX" in qt:
            return "INDEX"
        return "STOCK"

    def _ret_n_dias(df, n):
        if df is None or df.empty or len(df) <= n:
            return np.nan
        c = df["Close"]
        return float(c.iloc[-1] / c.iloc[-n] - 1.0)

    def _vol_anual_1y(df):
        if df is None or df.empty:
            return np.nan
        r = df["Close"].pct_change().dropna()
        r = r.iloc[-252:] if len(r) > 252 else r
        if r.empty:
            return np.nan
        return float(r.std() * np.sqrt(252))

    def _mdd_1y(df):
        if df is None or df.empty:
            return np.nan
        c = (1 + df["Close"].pct_change()).dropna().cumprod()
        c = c.iloc[-252:] if len(c) > 252 else c
        if c.empty:
            return np.nan
        pico = np.maximum.accumulate(c)
        dd = c / pico - 1.0
        return float(dd.min())

    # --- Carregar info (quando aplicável) ---
    try:
        tk = yf.Ticker(ticker)
        info = getattr(tk, "info", {}) or {}
    except Exception as e:
        st.error(f"❌ Error loading fundamentals: {e}")
        return

    tipo = _inferir_tipo(info)

    st.subheader("📊 Fundamental Details")

    if tipo == "STOCK":
        # ======================= STOCK =======================
        st.markdown("### 🏢 Company / 💹 Valuation / 📈 Performance")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🏢 Company")
            st.markdown(f"**Name:** {info.get('longName', '-')}")
            st.markdown(f"**Ticker:** {info.get('symbol', ticker)}")
            st.markdown(f"**Sector:** {info.get('sector', '-')}")
            st.markdown(f"**Industry:** {info.get('industry', '-')}")
            st.markdown(f"**Employees:** {formatar_valor(info.get('fullTimeEmployees'), 'inteiro')}")
            st.markdown(f"**Country:** {info.get('country', '-')}")

        with col2:
            st.markdown("#### 💹 Valuation")
            st.markdown(f"**P/E (TTM):** {formatar_valor(info.get('trailingPE'), 'moeda')}")
            st.markdown(f"**Forward P/E:** {formatar_valor(info.get('forwardPE'), 'moeda')}")
            st.markdown(f"**PEG Ratio:** {info.get('pegRatio', 'N/A')}")
            st.markdown(f"**P/B Ratio:** {formatar_valor(info.get('priceToBook'), 'moeda')}")
            st.markdown(f"**EV/EBITDA:** {formatar_valor(info.get('enterpriseToEbitda'), 'moeda')}")
            st.markdown(f"**Beta:** {formatar_valor(info.get('beta'), 'moeda')}")

        with col3:
            st.markdown("#### 📈 Performance & Yield")
            st.markdown(f"**Dividend Yield:** {formatar_valor(info.get('dividendYield'), 'porcentagem')}")
            st.markdown(f"**Last Dividend:** {formatar_valor(info.get('lastDividendValue'), 'moeda')}")
            st.markdown(f"**Ex-Dividend Date:** {_epoch_to_date_txt(info.get('exDividendDate'))}")
            st.markdown(f"**52W Change:** {formatar_valor(info.get('52WeekChange'), 'porcentagem')}")
            st.markdown(f"**52W High:** {formatar_valor(info.get('fiftyTwoWeekHigh'), 'moeda')}")
            st.markdown(f"**52W Low:** {formatar_valor(info.get('fiftyTwoWeekLow'), 'moeda')}")

        st.markdown("---")
        col4, col5 = st.columns(2)
        with col4:
            st.markdown("#### 💰 Financials")
            st.markdown(f"**Market Cap:** {formatar_valor(info.get('marketCap'), 'moeda')}")
            st.markdown(f"**Revenue (TTM):** {formatar_valor(info.get('totalRevenue'), 'moeda')}")
            st.markdown(f"**Gross Profit:** {formatar_valor(info.get('grossProfits'), 'moeda')}")
            st.markdown(f"**Net Income:** {formatar_valor(info.get('netIncomeToCommon'), 'moeda')}")
            st.markdown(f"**Operating Margin:** {formatar_valor(info.get('operatingMargins'), 'porcentagem')}")
            st.markdown(f"**Net Margin:** {formatar_valor(info.get('profitMargins'), 'porcentagem')}")

        with col5:
            st.markdown("#### 🧾 Debt & Cash")
            st.markdown(f"**Total Debt:** {formatar_valor(info.get('totalDebt'), 'moeda')}")
            st.markdown(f"**Total Cash:** {formatar_valor(info.get('totalCash'), 'moeda')}")
            st.markdown(f"**Free Cash Flow:** {formatar_valor(info.get('freeCashflow'), 'moeda')}")
            st.markdown(f"**Debt/Equity Ratio:** {formatar_valor(info.get('debtToEquity'), 'porcentagem')}")
            st.markdown(f"**ROE:** {formatar_valor(info.get('returnOnEquity'), 'porcentagem')}")
            st.markdown(f"**ROA:** {formatar_valor(info.get('returnOnAssets'), 'porcentagem')}")

    elif tipo == "ETF":
        # ======================== ETF ========================
        st.markdown("### 🧺 ETF Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Overview")
            st.markdown(f"**Name:** {info.get('longName', '-')}")
            st.markdown(f"**Ticker:** {info.get('symbol', ticker)}")
            st.markdown(f"**Category:** {info.get('category', 'N/A')}")
            st.markdown(f"**Fund Family:** {info.get('fundFamily', 'N/A')}")
            st.markdown(f"**Legal Type:** {info.get('legalType', 'N/A')}")
            st.markdown(f"**Inception Date:** {_epoch_to_date_txt(info.get('fundInceptionDate'))}")

        with col2:
            st.markdown("#### Size & Fees")
            st.markdown(f"**AUM / Total Assets:** {formatar_valor(info.get('totalAssets'), 'moeda')}")
            st.markdown(f"**NAV Price:** {formatar_valor(info.get('navPrice'), 'moeda')}")
            st.markdown(f"**Expense Ratio:** {formatar_valor(info.get('annualReportExpenseRatio'), 'porcentagem')}")
            st.markdown(f"**Yield:** {formatar_valor(info.get('yield'), 'porcentagem')}")

        with col3:
            st.markdown("#### Historical Stats")
            st.markdown(f"**YTD Return:** {formatar_valor(info.get('ytdReturn'), 'porcentagem')}")
            st.markdown(f"**3Y Avg Return:** {formatar_valor(info.get('threeYearAverageReturn'), 'porcentagem')}")
            st.markdown(f"**5Y Avg Return:** {formatar_valor(info.get('fiveYearAverageReturn'), 'porcentagem')}")
            st.markdown(f"**Beta (3Y):** {formatar_valor(info.get('beta3Year'), 'moeda')}")
            st.markdown(f"**Holdings Turnover:** {formatar_valor(info.get('annualHoldingsTurnover'), 'porcentagem')}")

        st.caption("Nota: muitos ETFs no Yahoo não expõem a composição (holdings) pela API; para ver holdings pode ser necessário outra fonte.")

    else:
        # ======================= INDEX =======================
        st.markdown("### 🧭 Index Snapshot (price-based)")
        if dados_preco is None or dados_preco.empty:
            st.info("Sem dados de preço disponíveis para calcular métricas do índice.")
            return

        r1m  = _ret_n_dias(dados_preco, 21)
        r3m  = _ret_n_dias(dados_preco, 63)
        r6m  = _ret_n_dias(dados_preco, 126)
        r1y  = _ret_n_dias(dados_preco, 252)
        vol1y = _vol_anual_1y(dados_preco)
        mdd1y = _mdd_1y(dados_preco)

        hi52 = float(dados_preco["High"].rolling(252).max().iloc[-1])
        lo52 = float(dados_preco["Low"].rolling(252).min().iloc[-1])
        last = float(dados_preco["Close"].iloc[-1])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1M Return", f"{r1m:.2%}" if not np.isnan(r1m) else "N/A")
        c2.metric("3M Return", f"{r3m:.2%}" if not np.isnan(r3m) else "N/A")
        c3.metric("6M Return", f"{r6m:.2%}" if not np.isnan(r6m) else "N/A")
        c4.metric("1Y Return", f"{r1y:.2%}" if not np.isnan(r1y) else "N/A")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Last Close", f"${last:,.2f}")
        c6.metric("52-Week High", f"${hi52:,.2f}")
        c7.metric("52-Week Low", f"${lo52:,.2f}")
        c8.metric("Ann. Vol (1Y)", f"{vol1y:.2%}" if not np.isnan(vol1y) else "N/A")

        st.caption(f"Max Drawdown (1Y): **{mdd1y:.2%}**" if not np.isnan(mdd1y) else "Max Drawdown (1Y): N/A")





# ---------------------------------------------------------------------
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_combined_chart(df, symbol, sma_values=None, macd=None, signal=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=symbol
    ), row=1, col=1)

    if sma_values:
        colors = ['blue', 'orange', 'green', 'red']
        for i, (window, sma) in enumerate(sma_values.items()):
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=sma,
                mode='lines',
                name=f'SMA {window}',
                line=dict(width=2, color=colors[i % len(colors)])
            ), row=1, col=1)

    if macd is not None and signal is not None:
        histogram = macd - signal
        fig.add_trace(go.Scatter(x=df['date'], y=macd, mode='lines', name='MACD Line', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=signal, mode='lines', name='Signal Line', line=dict(color='orange')), row=2, col=1)
        fig.add_trace(go.Bar(x=df['date'], y=histogram.where(histogram > 0), name='MACD Histogram +', marker_color='green', opacity=0.5), row=2, col=1)
        fig.add_trace(go.Bar(x=df['date'], y=histogram.where(histogram < 0), name='MACD Histogram -', marker_color='red', opacity=0.5), row=2, col=1)

    fig.update_layout(
        title=f'{symbol} - Candlestick + SMA + MACD',
        xaxis_title='Data',
        yaxis_title='Preço',
        xaxis2_title='Data',
        yaxis2_title='MACD',
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig

# ---------------------------------------------------------------------
def mostrar_grafico_tecnico(ticker: str, dados: pd.DataFrame):
    """Calcula indicadores técnicos e mostra o gráfico do ativo."""
    st.subheader("📊 Gráfico Técnico")

    df = dados.copy()

    # 🛠️ Corrigir MultiIndex se necessário
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    df['date'] = df.index

    if df.empty or df[['open', 'high', 'low', 'close']].isna().all().any():
        st.warning("❗ Dados incompletos para este ativo.")
        return

    # Calcular SMA
    sma_values = {
        20: df['close'].rolling(window=20).mean(),
        50: df['close'].rolling(window=50).mean()
    }

    # Calcular MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    fig = plot_combined_chart(df, ticker, sma_values=sma_values, macd=macd, signal=signal)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
def calcular_metricas_performance(df: pd.DataFrame, taxa_livre_risco=0.01):
    """Calcula métricas de performance com base em preços de fechamento."""
    df = df.copy()
    df["Returns"] = df["Close"].pct_change()
    df["Cumulative"] = (1 + df["Returns"]).cumprod()

    total_return = df["Cumulative"].iloc[-1] - 1
    volatilidade = df["Returns"].std() * np.sqrt(252)
    sharpe = ((df["Returns"].mean() * 252) - taxa_livre_risco) / (df["Returns"].std() * np.sqrt(252))

    max_acumulado = df["Cumulative"].cummax()
    drawdown = (df["Cumulative"] - max_acumulado) / max_acumulado
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "volatility": volatilidade,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "cumulative": df["Cumulative"]
    }

import plotly.graph_objects as go

def mostrar_metricas_performance(metricas: dict):
    """Exibe métricas de performance e gráfico de retorno acumulado com Plotly."""
    st.subheader("📊 Performance Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", f"{metricas['total_return']:.2%}")

    with col2:
        st.metric("Volatility (Ann.)", f"{metricas['volatility']:.2%}")

    with col3:
        st.metric("Sharpe Ratio", f"{metricas['sharpe_ratio']:.2f}")

    with col4:
        st.metric("Max Drawdown", f"{metricas['max_drawdown']:.2%}")

    st.markdown("#### 📈 Cumulative Return (%)")

    # Preparar dados para o gráfico
    df_plot = metricas["cumulative"].copy()
    df_plot = (df_plot - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot,
        mode='lines',
        name='Cumulative Return',
        line=dict(width=2)
    ))

    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Retorno Acumulado (%)",
        height=400,
        margin=dict(t=20, b=40),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------

# ===================== RF: last N=150 -> next 30 (retornos log) =====================
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
)

# ---------- 1) Preparar série ----------
def preparar_close(obj) -> pd.DataFrame:
    """
    Retorna DataFrame com coluna única 'Close' (float) e índice datetime limpo.
    Aceita DataFrame normal, MultiIndex (yfinance) ou Series.
    """
    if isinstance(obj, pd.Series):
        s = obj.dropna().copy()
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return pd.DataFrame({"Close": pd.to_numeric(s, errors="coerce")}).dropna()

    if not isinstance(obj, pd.DataFrame):
        raise ValueError("Esperado DataFrame ou Series.")

    df = obj.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # MultiIndex: ('TICKER','Close')
    if isinstance(df.columns, pd.MultiIndex):
        cols_close = [c for c in df.columns if str(c[-1]).lower() == "close"]
        if cols_close:
            s = df[cols_close[0]].rename("Close")
            return pd.DataFrame({"Close": pd.to_numeric(s, errors="coerce")}).dropna()
        cols_adj = [c for c in df.columns if str(c[-1]).lower() in ("adj close","adjclose","adj_close")]
        if cols_adj:
            s = df[cols_adj[0]].rename("Close")
            return pd.DataFrame({"Close": pd.to_numeric(s, errors="coerce")}).dropna()

    # Coluna 'Close' “plana”
    if "Close" in df.columns:
        col = df["Close"]
        if isinstance(col, pd.DataFrame):
            num = col.select_dtypes(include=[np.number])
            s = num.iloc[:, 0] if not num.empty else pd.to_numeric(col.iloc[:, 0], errors="coerce")
        else:
            s = pd.to_numeric(col, errors="coerce")
        return pd.DataFrame({"Close": s}).dropna()

    # Fallbacks
    for alt in ("Adj Close","AdjClose","adj_close","adjclose"):
        if alt in df.columns:
            return pd.DataFrame({"Close": pd.to_numeric(df[alt], errors="coerce")}).dropna()

    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        return pd.DataFrame({"Close": num.iloc[:, 0]}).dropna()

    raise ValueError("Não encontrei 'Close' nem coluna numérica utilizável.")


# ---------- 2) Features de retornos (com clipping de outliers) ----------
def construir_features_retorno(df_close: pd.DataFrame, n_lags: int = 10, clip_pct: int = 99):
    """
    y(t)  = retorno log em t
    X(t)  = lags de retorno (1..n_lags) + médias e desvios (5,10,21) defasados.
    Retorna X, y, idx, p_t e p_{t-1} alinhados.
    """
    close = df_close["Close"].astype(float)
    r = np.log(close).diff()

    # clipping robusto de outliers
    try:
        cap = float(np.nanpercentile(np.abs(r.dropna()), clip_pct))
        if np.isfinite(cap) and cap > 0:
            r = r.clip(-cap, cap)
    except Exception:
        pass

    df_feat = pd.DataFrame({"ret": r}, index=close.index)

    for k in range(1, n_lags + 1):
        df_feat[f"ret_lag_{k}"] = r.shift(k)

    for w in (5, 10, 21):
        df_feat[f"ret_mean_{w}"] = r.rolling(w).mean().shift(1)
        df_feat[f"ret_std_{w}"]  = r.rolling(w).std().shift(1)

    df_feat = df_feat.dropna()

    y   = df_feat["ret"].values
    X   = df_feat.drop(columns=["ret"]).values
    idx = df_feat.index

    p_t   = close.reindex(idx)
    p_tm1 = close.shift(1).reindex(idx)   # evita NaN no primeiro ponto

    return X, y, idx, p_t, p_tm1


# ---------- 3) Treinar, avaliar e calibrar ----------
def treinar_e_avaliar_modelo(df_close: pd.DataFrame, n_lags: int = 10,
                              n_estimators: int = 600, seed: int = 42, fracao_teste: float = 0.2):
    """
    Split temporal; treina RF em retornos log; mede erro em PREÇO (one-step) vs baseline constante.
    Aplica calibração de amplitude r̂_cal = α * r̂ (α ∈ [0,1]) aprendida no treino.
    Retorna (modelo_full, ultimo_x, métricas, aux).
    """
    X, y, idx, p_t, p_tm1 = construir_features_retorno(df_close, n_lags)
    n = len(X)
    if n < 50:
        raise ValueError("Poucos dados após features (≥50 linhas).")

    n_test  = max(10, int(n * fracao_teste))
    n_train = n - n_test

    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]
    p_t_tr, p_tm1_tr = p_t[:n_train].values,  p_tm1[:n_train].values
    p_t_te, p_tm1_te = p_t[n_train:].values,  p_tm1[n_train:].values

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)

    # previsões de retorno
    yhat_tr = rf.predict(X_tr)
    yhat_te = rf.predict(X_te)

    # calibração α (sem intercepto)
    denom = float(np.dot(yhat_tr, yhat_tr))
    alpha = float(np.dot(yhat_tr, y_tr) / denom) if denom > 0 else 0.0
    alpha = max(0.0, min(1.0, alpha))

    # reconstrução de preço one-step
    p_hat_tr = p_tm1_tr * np.exp(alpha * yhat_tr)
    p_hat_te = p_tm1_te * np.exp(alpha * yhat_te)
    p_naive_tr = p_tm1_tr
    p_naive_te = p_tm1_te

    # métricas em preço (com máscara segura)
    def _metrica_preco(y_true, y_pred, y_nv):
        y_true = np.asarray(y_true, float)
        y_pred = np.asarray(y_pred, float)
        y_nv   = np.asarray(y_nv, float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_nv)
        if mask.sum() < 2:
            return {"MAE": np.nan, "RMSE": np.nan, "MAPE_%": np.nan, "RMSE_naive": np.nan}
        mae  = mean_absolute_error(y_true[mask], y_pred[mask])
        rmse = sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100.0
        rmse_nv = sqrt(mean_squared_error(y_true[mask], y_nv[mask]))
        return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "RMSE_naive": rmse_nv}

    m_tr = _metrica_preco(p_t_tr, p_hat_tr, p_naive_tr)
    m_te = _metrica_preco(p_t_te, p_hat_te, p_naive_te)

    # R² em retornos
    mask_tr = np.isfinite(y_tr) & np.isfinite(yhat_tr)
    mask_te = np.isfinite(y_te) & np.isfinite(yhat_te)
    r2_tr = r2_score(y_tr[mask_tr], yhat_tr[mask_tr]) if mask_tr.any() else np.nan
    r2_te = r2_score(y_te[mask_te], yhat_te[mask_te]) if mask_te.any() else np.nan

    # confiança vs baseline
    if np.isfinite(m_te["RMSE"]) and np.isfinite(m_te["RMSE_naive"]) and m_te["RMSE_naive"] > 0:
        conf = max(0.0, min(100.0, 100.0 * (1.0 - (m_te["RMSE"]/m_te["RMSE_naive"]))))
    else:
        conf = 0.0

    metricas = {
        "train": {"MAE": m_tr["MAE"], "RMSE": m_tr["RMSE"], "MAPE_%": m_tr["MAPE_%"], "R2_returns": r2_tr},
        "test":  {"MAE": m_te["MAE"], "RMSE": m_te["RMSE"], "MAPE_%": m_te["MAPE_%"], "R2_returns": r2_te},
        "model_confidence_%": conf,
        "alpha_calibration": alpha
    }

    # re-treina em TODO para projetar futuro
    rf_full = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=6, min_samples_leaf=5,
        random_state=seed, n_jobs=-1
    )
    rf_full.fit(X, y)

    ultimo_x = X[-1:].copy()
    aux = {
        "n_lags": n_lags,
        "returns_series": np.log(df_close["Close"].astype(float)).diff().dropna().values,
        "last_price": float(df_close["Close"].astype(float).iloc[-1]),
        "alpha": alpha
    }
    return rf_full, ultimo_x, metricas, aux


# ---------- 4) Previsão multi-passos (em preço) ----------
def prever_modelo_futuro(modelo, aux, passos: int = 30) -> np.ndarray:
    """
    Gera trajetória futura de PREÇOS usando retornos previstos iterativamente.
    Aplica calibração α nas previsões.
    """
    n_lags = int(aux["n_lags"])
    r_hist = aux["returns_series"].copy()
    p_last = float(aux["last_price"])
    alpha  = float(aux.get("alpha", 1.0))

    preds = []
    for _ in range(passos):
        # lags + estatísticas defasadas
        lags = r_hist[-n_lags:] if len(r_hist) >= n_lags else np.r_[np.zeros(n_lags - len(r_hist)), r_hist]
        feats = list(lags[::-1])
        for w in (5, 10, 21):
            ref = r_hist if len(r_hist) >= w else np.r_[np.zeros(w - len(r_hist)), r_hist]
            feats.append(ref[-w:].mean())
            feats.append(ref[-w:].std(ddof=0))
        x = np.array(feats, float).reshape(1, -1)

        yhat = float(modelo.predict(x)[0]) * alpha
        p_last = p_last * np.exp(yhat)
        preds.append(p_last)
        r_hist = np.r_[r_hist, yhat]

    return np.array(preds, float)


# ---------- 5) Wrapper: treinar nos últimos 150 e prever próximos 30 ----------
def prever_dias(df_raw: pd.DataFrame, dias: int = 1, n_lags: int = 10,
                n_estimators: int = 600, seed: int = 42,
                janela: int | None = None, usar_ultimos: int = 150):
    """Treina apenas nos últimos 'usar_ultimos' e retorna (df_prev, métricas, base_usada)."""
    if janela is not None:
        n_lags = janela

    base_full = preparar_close(df_raw)
    usar_ultimos = int(min(max(usar_ultimos, 60), len(base_full)))
    base_usada = base_full.tail(usar_ultimos).copy()

    modelo, ultimo_x, metricas, aux = treinar_e_avaliar_modelo(
        base_usada, n_lags=n_lags, n_estimators=n_estimators, seed=seed
    )

    preds_price = prever_modelo_futuro(modelo, aux, passos=dias)

    freq = pd.infer_freq(base_full.index) or "B"
    ultima = base_full.index[-1]
    try:
        datas_fut = pd.date_range(start=ultima, periods=dias + 1, freq=freq)[1:]
    except Exception:
        datas_fut = pd.date_range(start=ultima + pd.Timedelta(days=1), periods=dias, freq="D")

    df_prev = pd.DataFrame({"RF Predicted Close": preds_price}, index=datas_fut)
    return df_prev, metricas, base_usada


# ---------- 6) Mostrar no Streamlit (labels em inglês) ----------
def mostrar_predicao_rf_min(df_raw: pd.DataFrame, dias: int = 30, n_lags: int = 10,
                            n_estimators: int = 600, seed: int = 42,
                            janela: int | None = None, usar_ultimos: int = 150):
    """Mostra histórico (janela usada) + previsão e métricas."""
    import streamlit as st
    import plotly.graph_objects as go

    if janela is not None:
        n_lags = janela

    df_prev, metricas, base_usada = prever_dias(
        df_raw, dias=dias, n_lags=n_lags, n_estimators=n_estimators,
        seed=seed, usar_ultimos=usar_ultimos
    )

    hist = base_usada

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Historical Close"))
    fig.add_trace(go.Scatter(x=df_prev.index, y=df_prev["RF Predicted Close"],
                             mode="lines+markers", name="Model Prediction", line=dict(dash="dot")))
    fig.update_xaxes(range=[hist.index[0], df_prev.index[-1]])
    fig.update_layout(hovermode="x unified")
    st.subheader(f"Prediction (Close) — last {usar_ultimos} → next {dias}  |  model: RF")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("🎲 Model Confidence", f"{metricas['model_confidence_%']:.1f}%")
    c2.metric("📈 Training R² (returns)", f"{metricas['train']['R2_returns']:.3f}")
    c3.metric("🧪 Test R² (returns)",     f"{metricas['test']['R2_returns']:.3f}")

    tabela = pd.DataFrame({
        "Set": ["Train", "Test"],
        "MAE (price)":  [metricas["train"]["MAE"],  metricas["test"]["MAE"]],
        "RMSE (price)": [metricas["train"]["RMSE"], metricas["test"]["RMSE"]],
        "MAPE (%)":     [metricas["train"]["MAPE_%"], metricas["test"]["MAPE_%"]],
        "R² (returns)": [metricas["train"]["R2_returns"], metricas["test"]["R2_returns"]],
    }).round({"MAE (price)": 4, "RMSE (price)": 4, "MAPE (%)": 2, "R² (returns)": 3})
    st.dataframe(tabela, use_container_width=True)

    with st.expander("🔍 View raw data"):
        st.write("Base used (tail):", hist.tail())
        st.write("Predictions (head):", df_prev.head())
        st.write("α (amplitude calibration):", f"{metricas['alpha_calibration']:.3f}")

    return df_prev








    
    
    
# ---------------------------------------------------------------------
# Página principal
st.set_page_config(page_title="Análise de Ativo", layout="wide")
st.title("🔎 Análise Individual de Ativo")

# Sidebar de filtros
st.sidebar.header("Configurações do Ativo")
universo = carregar_universo()

tipos = sorted(universo.get("Categoria Original", pd.Series()).dropna().unique().tolist())
tipo_escolhido = st.sidebar.selectbox("Categoria Original", tipos)
dados_filtrados = universo[universo["Categoria Original"] == tipo_escolhido]

for coluna in ["País", "Setor", "Indústria"]:
    if coluna in dados_filtrados.columns:
        opcoes = sorted(dados_filtrados[coluna].dropna().unique())
        if len(opcoes) > 1:
            escolha = st.sidebar.selectbox(coluna, ["Todos"] + opcoes, key=f"filtro_{coluna}")
            if escolha != "Todos":
                dados_filtrados = dados_filtrados[dados_filtrados[coluna] == escolha]

nomes_para_tickers = dados_filtrados.set_index("Nome Curto")["Ticker"].dropna().to_dict()
nome_escolhido = st.sidebar.selectbox("Ativo", list(nomes_para_tickers.keys()))
ticker = nomes_para_tickers[nome_escolhido]

anos = st.sidebar.slider("Horizonte (anos)", 1, 20, 10)
frequencia = st.sidebar.selectbox("Frequência", ["1d", "1wk", "1mo"])
btn = st.sidebar.button("🔍 Analisar Ativo")

# ---------------------------------------------------------------------
# Execução ao clicar no botão
if btn and ticker:
    st.markdown(f"## 📌 {nome_escolhido} ({ticker})")

    data_inicio = pd.Timestamp.today() - pd.DateOffset(years=anos)
    data_fim = pd.Timestamp.today()

    dados = yf.download(ticker, start=data_inicio, end=data_fim, interval=frequencia, auto_adjust=False)
    df_precos = dados[["Close"]].dropna()

    if dados.empty:
        st.warning("⚠️ Nenhum dado encontrado para esse ativo e período.")
    else:
        tk = yf.Ticker(ticker)
        info = getattr(tk, "info", {}) or {}
        mostrar_kpis_preco(dados, ticker, info)
        mostrar_detalhes_fundamentalistas(ticker, tipo_ativo=tipo_escolhido, dados_preco=dados)
        mostrar_grafico_tecnico(ticker, dados)
        metricas = calcular_metricas_performance(dados)
        mostrar_metricas_performance(metricas)
        df_prev_rf = mostrar_predicao_rf_min(df_precos, dias=1, janela=10, usar_ultimos=150)


    with st.expander("🔍 Ver dados brutos"):
            st.dataframe(dados.tail())

else:
    st.info("Escolha um ativo e clique em **Analisar Ativo**.")
