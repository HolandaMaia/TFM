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

       

    with st.expander("🔍 Ver dados brutos"):
            st.dataframe(dados.tail())

else:
    st.info("Escolha um ativo e clique em **Analisar Ativo**.")
