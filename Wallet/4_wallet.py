

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wallet Dashboard", layout="wide")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Função para carregar o universo de ativos diretamente do GitHub
@st.cache_resource
def carregar_universo():
    url = "https://raw.githubusercontent.com/HolandaMaia/TFM/master/dados/ativos_totais.xlsx"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Verifica se houve erro na requisição
        df = pd.read_excel(response.content)
        return df
    except Exception as e:
        st.error(f"Error loading asset database from GitHub: {e}")
        return pd.DataFrame()

universo = carregar_universo()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def obter_dados(tickers, start, end, interval="1d"):
    data = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True)["Close"]
    return data.dropna()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def otimizar_carteira(dados):
    returns = np.log(dados / dados.shift(1)).dropna()
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(len(mu))
    w = inv_cov @ mu
    w = np.clip(w, 0, None)
    if w.sum() == 0:
        # 🇪🇸 Si todo es 0 (caso extremo), repartir uniforme
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    return pd.Series(w, index=dados.columns)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def calcular_metricas(dados, pesos):
    returns = np.log(dados / dados.shift(1)).dropna()
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    port_return = (pesos * mu).sum()
    port_vol = np.sqrt(pesos.values.T @ cov.values @ pesos.values)
    sharpe = port_return / port_vol if port_vol else 0
    cum_returns = (1 + returns.dot(pesos)).cumprod()
    running_max = cum_returns.cummax()
    drawdown = ((cum_returns - running_max) / running_max).min()
    num_assets = len(pesos)
    std = returns.std() * np.sqrt(252)
    acumulado_1y = cum_returns.iloc[-1] - 1
    return port_return, port_vol, sharpe, drawdown, num_assets, std.mean(), acumulado_1y

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def mostrar_kpis(metrics):
    retorno, vol, sharpe, drawdown, _n, desvio, acumulado = metrics

    st.subheader("📊 Key Performance Indicators (KPIs)", divider='blue')
    st.caption(
        "These metrics provide insights into the performance and risk of your portfolio. Hover over the info icons for detailed explanations."
    )

    # Organize KPIs into two rows for better visual balance
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Expected Return",
        f"{retorno:.2%}",
        help="This is the average annual profit you might earn from your portfolio. It is based on historical data and helps you understand potential growth."
    )
    col2.metric(
        "Volatility",
        f"{vol:.2%}",
        help="Volatility measures how much the portfolio's value goes up and down. Higher volatility means more risk, but also more potential reward."
    )
    col3.metric(
        "Sharpe Ratio",
        f"{sharpe:.2f}",
        help="The Sharpe Ratio shows how much return you get for the risk you take. A higher ratio means better risk-adjusted performance."
    )

    col4, col5, col6 = st.columns(3)
    col4.metric(
        "Max Drawdown",
        f"{drawdown:.2%}",
        help="This is the largest drop in your portfolio's value from its highest point. It shows the worst loss you could have experienced."
    )
    col5.metric(
        "Standard Deviation",
        f"{desvio:.2%}",
        help="Standard deviation measures how much the returns vary from the average. It helps you understand the consistency of returns."
    )
    col6.metric(
        "Cumulative Return",
        f"{acumulado:.2%}",
        help="Cumulative return is the total profit or loss over the entire period. It shows how much your portfolio has grown overall."
    )



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def mostrar_performance(dados, pesos):
    st.subheader("Historical Performance", divider='blue')
    returns = np.log(dados / dados.shift(1)).dropna()
    carteira = (1 + returns.dot(pesos)).cumprod()
    drawdown = carteira / carteira.cummax() - 1

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(
            "📈 Portfolio Cumulative Return",
            help="This chart displays the cumulative return of your portfolio over time. The x-axis represents the time period, and the y-axis shows the cumulative return as a percentage. It helps you track the overall growth of your portfolio."
        )
        fig_ret = px.area(
            x=carteira.index,
            y=((carteira - 1) * 100),  # transforma em %
            labels={"x": "Date", "y": "Cumulative Return (%)"},
            title="Portfolio Cumulative Return"
        )
        fig_ret.update_layout(
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    with col2:
        st.subheader(
            "📉 Portfolio Drawdown",
            help="This chart shows the drawdown of your portfolio over time. The x-axis represents the time period, and the y-axis shows the drawdown as a percentage. It helps you understand the maximum loss from the portfolio's peak value."
        )
        fig_dd = px.area(
            x=drawdown.index,
            y=drawdown,
            labels={"x": "Date", "y": "Drawdown (%)"},
            title="Portfolio Drawdown"
        )
        fig_dd.update_layout(
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig_dd, use_container_width=True)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def mostrar_tabela_ativos(dados, pesos, nomes_para_tickers):
    st.subheader("Asset Breakdown", divider='blue')
    st.caption(
        "This section provides a detailed breakdown of your portfolio's assets. "
        "The table includes asset names, weights, current prices, units held, invested values, "
        "and percentage changes over different time periods (24h, 7d, 1m). These numbers are "
        "calculated based on the latest available data and the weights assigned to each asset."
    )

    # Mapear nomes dos ativos usando os nomes completos selecionados no menu
    nomes_ativos = [next((nome for nome, tck in nomes_para_tickers.items() if tck == ticker), ticker) for ticker in dados.columns]
    ultimo_preco = dados.ffill().iloc[-1]
    investimento = pesos * 1000  # hipotético
    unidades = investimento / ultimo_preco
    var_1d = dados.pct_change().iloc[-1]
    var_7d = dados.pct_change(7).iloc[-1]
    var_1m = dados.pct_change(21).iloc[-1]

    # Adicionar setas para variações
    def format_variation(value):
        if value > 0:
            return f"↑ {value:.2f}%"
        elif value < 0:
            return f"↓ {value:.2f}%"
        else:
            return f"→ {value:.2f}%"

    tabela = pd.DataFrame({
        "Asset Name": nomes_ativos,
        "Weight (%)": pesos * 100,
        "Current Price": ultimo_preco,
        "Units": unidades,
        "Invested Value": investimento,
        "24h Change (%)": var_1d.apply(format_variation),
        "7d Change (%)": var_7d.apply(format_variation),
        "1m Change (%)": var_1m.apply(format_variation),
    })

    # Usar st.dataframe com formatação personalizada
    st.dataframe(tabela.style.format({
        "Weight (%)": "{:.2f}",
        "Current Price": "{:.2f}",
        "Units": "{:.2f}",
        "Invested Value": "{:.2f}",
    }), use_container_width=True)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis2_title='Date',
        yaxis2_title='MACD',
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig

def mostrar_graficos_ativos(pesos, anos, frequencia, nomes_para_tickers):
    st.subheader("📊 Technical Analysis by Asset", divider='blue')
    st.caption(
        "Explore detailed technical analysis for each asset in your portfolio. "
        "The charts include candlestick patterns, moving averages (SMA20, SMA50), and MACD indicators. "
        "These tools help you identify trends, momentum, and potential buy/sell signals."
    )

    # Definir datas
    data_fim = pd.Timestamp.today()
    data_inicio = data_fim - pd.DateOffset(years=anos)

    # Baixar dados OHLC
    tickers = list(pesos.index)
    dados_ohlc = yf.download(
        tickers,
        start=data_inicio,
        end=data_fim,
        interval=frequencia,
        auto_adjust=False,
        group_by='ticker'
    )

    for ticker in tickers:
        asset_name = next((nome for nome, tck in nomes_para_tickers.items() if tck == ticker), ticker)

        if ticker not in dados_ohlc.columns.levels[0]:
            st.warning(f"❗ Data not available for {asset_name}")
            continue

        df = dados_ohlc[ticker].dropna().copy()
        df = df.rename(columns=str.lower)
        df['date'] = df.index

        if df.empty or df[['open', 'high', 'low', 'close']].isna().all().any():
            st.warning(f"❗ Data not available for {asset_name}")
            continue

        # Indicadores
        sma_values = {
            20: df['close'].rolling(window=20).mean(),
            50: df['close'].rolling(window=50).mean()
        }
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        # Gráfico
        with st.expander(f"📈 {asset_name} - Technical Analysis"):
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])

            fig.add_trace(go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=f"{asset_name}"
            ), row=1, col=1)

            colors = ['blue', 'orange']
            for i, (window, sma) in enumerate(sma_values.items()):
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=sma,
                    mode='lines',
                    name=f'SMA {window}',
                    line=dict(width=2, color=colors[i % len(colors)])
                ), row=1, col=1)

            histogram = macd - signal
            fig.add_trace(go.Scatter(x=df['date'], y=macd, mode='lines', name='MACD Line', line=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=signal, mode='lines', name='Signal Line', line=dict(color='orange')), row=2, col=1)
            fig.add_trace(go.Bar(x=df['date'], y=histogram.where(histogram > 0), name='MACD Histogram +', marker_color='green', opacity=0.5), row=2, col=1)
            fig.add_trace(go.Bar(x=df['date'], y=histogram.where(histogram < 0), name='MACD Histogram -', marker_color='red', opacity=0.5), row=2, col=1)

            fig.update_layout(
                title=f'{asset_name} - Candlestick + SMA + MACD',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis2_title='Date',
                yaxis2_title='MACD',
                xaxis_rangeslider_visible=False,
                height=600,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Adicionar legenda explicativa
            st.caption(
                "- **Candlestick**: Shows the open, high, low, and close prices for each time period.\n"
                "- **SMA20**: 20-day simple moving average, useful for identifying short-term trends.\n"
                "- **SMA50**: 50-day simple moving average, useful for identifying medium-term trends.\n"
                "- **MACD**: Moving Average Convergence Divergence, highlights momentum and trend direction.\n"
                "- **Signal Line**: A 9-day EMA of the MACD, used to generate buy/sell signals.\n"
                "- **Histogram**: Difference between MACD and Signal Line, shows momentum strength.\n"
                "- **SMA Crossovers**: When SMA20 crosses above SMA50, it may indicate a bullish trend; crossing below may indicate a bearish trend."
            )
        
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def mostrar_benchmark_simples(dados, pesos, benchmark_ticker, nomes_para_tickers, anos=10):
    st.subheader("📊 Benchmark Historical Analysis", divider='blue')
    st.caption(
        "Compare the historical performance of your portfolio against a selected benchmark. "
        "This analysis helps you understand how your portfolio performs relative to a market index ou other reference asset."
    )

    if not benchmark_ticker:
        st.warning("⚠️ No benchmark selected.")
        return

    # Obter o nome completo do benchmark
    benchmark_name = next((nome for nome, tck in nomes_para_tickers.items() if tck == benchmark_ticker), benchmark_ticker)

    data_fim = pd.Timestamp.today()
    data_inicio = data_fim - pd.DateOffset(years=anos)

    try:
        benchmark_df = yf.download(
            benchmark_ticker,
            start=data_inicio,
            end=data_fim,
            auto_adjust=True,
            progress=False
        )
    except Exception as e:
        st.error(f"❌ Error downloading benchmark data: {e}")
        return

    if benchmark_df.empty or "Close" not in benchmark_df.columns:
        st.warning("⚠️ Benchmark closing data unavailable.")
        return

    # Série de preços do benchmark
    close_series = benchmark_df["Close"].dropna()
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze()

    # Retorno acumulado do benchmark em %
    benchmark_ret = (1 + close_series.pct_change().dropna()).cumprod()
    benchmark_ret_pct = (benchmark_ret - 1) * 100

    # Retorno acumulado da carteira em %
    returns = np.log(dados / dados.shift(1)).dropna()
    carteira = (1 + returns.dot(pesos)).cumprod()
    carteira_pct = (carteira - 1) * 100

    # Alinhar datas
    datas_comuns = benchmark_ret_pct.index.intersection(carteira_pct.index)
    benchmark_ret_pct = benchmark_ret_pct.loc[datas_comuns]
    carteira_pct = carteira_pct.loc[datas_comuns]

    # 📉 Gráfico 1: Preço de Fechamento do Benchmark
    st.markdown(
        f"### 💵 Price Evolution - {benchmark_name}",
        help="This chart shows the historical closing prices of the selected benchmark asset. It helps you understand the price trends over time."
    )
    fig_close = px.line(
        x=close_series.index,
        y=close_series.values,
        title=f"{benchmark_name} - Closing Price",
        labels={"x": "Date", "y": "Price"}
    )
    fig_close.update_yaxes(title="Price")  # Removido o prefixo 'US$'
    fig_close.update_layout(height=400, showlegend=False)
    fig_close.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x unified"
    )
    st.plotly_chart(fig_close, use_container_width=True, key="grafico_preco_benchmark")

    # 📈 Gráfico 2: Retorno Acumulado (%)
    st.markdown(
        "### 📈 Cumulative Return (%) - Benchmark vs Portfolio",
        help="This chart compares the cumulative returns of your portfolio and the selected benchmark. It shows how much each has grown over time."
    )
    fig_ret = go.Figure()

    fig_ret.add_trace(go.Scatter(
        x=benchmark_ret_pct.index,
        y=benchmark_ret_pct.values,
        mode="lines",
        name=benchmark_name,
        line=dict(color="royalblue")
    ))

    fig_ret.add_trace(go.Scatter(
        x=carteira_pct.index,
        y=carteira_pct.values,
        mode="lines",
        name="Portfolio",
        line=dict(color="seagreen")
    ))

    fig_ret.update_yaxes(title="Return (%)", ticksuffix="%")
    fig_ret.update_layout(
        height=400,
        xaxis_title="Date",
        title="Cumulative Return (%)",
        legend_title="Series",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    # Adicionado anotações para destacar valores finais
    fig_ret.add_annotation(
        x=carteira_pct.index[-1],
        y=carteira_pct.values[-1],
        text="Portfolio End Value",
        showarrow=True,
        arrowhead=2
    )
    fig_ret.add_annotation(
        x=benchmark_ret_pct.index[-1],
        y=benchmark_ret_pct.values[-1],
        text="Benchmark End Value",
        showarrow=True,
        arrowhead=2
    )
    st.plotly_chart(fig_ret, use_container_width=True, key="grafico_retorno_comparado")



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def mostrar_fronteira(dados, pesos):
    returns = np.log(dados / dados.shift(1)).dropna()
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    n_port = 300
    results = np.zeros((3, n_port))
    for i in range(n_port):
        w = np.random.random(len(mu))
        w /= np.sum(w)
        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(w.T @ cov @ w)
        results[0, i] = port_vol
        results[1, i] = port_ret
        results[2, i] = port_ret / port_vol if port_vol else 0
    fig = px.scatter(x=results[0], y=results[1], color=results[2],
                    labels={"x": "Volatility", "y": "Return"},
                    title="Markowitz Frontier")
    carteira_ret = np.dot(pesos, mu)
    carteira_vol = np.sqrt(pesos.values.T @ cov.values @ pesos.values)
    fig.add_trace(go.Scatter(x=[carteira_vol], y=[carteira_ret], mode="markers",
                            marker=dict(color="red", size=10), name="Wallet"))
    return fig

def mostrar_heatmap(dados):
    corr = dados.pct_change().corr()
    # Estilo para valores legíveis
    sns.set(font_scale=1.1)
    # Figura proporcional à altura da fronteira
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(corr, ax=ax, annot=True, cmap="RdBu", center=0)
    return fig


def mostrar_fronteira_heatmap(dados, pesos):
    st.subheader(
        "Efficient Frontier and Correlation",divider='blue'
    )
    st.caption(
        "The Efficient Frontier helps you visualize the best possible returns for a given level of risk, based on historical data. "
        "The Correlation Matrix provides insights into how asset returns move in relation to each other, which is crucial for diversification."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        fig_front = mostrar_fronteira(dados, pesos)
        st.plotly_chart(
            fig_front,
            use_container_width=True,
            help="The Markowitz Frontier visualizes the trade-off between risk (volatility) and return for different portfolio allocations."
        )
    with col2:
        st.markdown(
            "**Correlation Matrix**",
            help="The Correlation Matrix shows how the returns of different assets are related. Values close to 1 indicate strong positive correlation, while values close to -1 indicate strong negative correlation."
        )
        fig_heat = mostrar_heatmap(dados)
        st.pyplot(fig_heat, use_container_width=True)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --- Monte Carlo da carteira: cálculo (corrigido para passo diário) ---


# --- Lê parâmetros da calculadora atuarial (se existirem) ---
def obter_objetivo_atuarial():
    """
    Retorna (n_anos, objetivo_reserva, info).
    Se não houver dados atuariais, usa n_anos=10 e objetivo_reserva=None.
    """
    n_anos = 10
    objetivo_reserva = None
    info = {"tem_atuarial": False}

    if "actuarial_result" in st.session_state and "user_inputs" in st.session_state:
        res = st.session_state["actuarial_result"]
        ui = st.session_state["user_inputs"]
        objetivo_reserva = float(res.get("reserva_aposentadoria", None))
        idade_atual = int(ui.get("idade_atual"))
        idade_apos = int(ui.get("idade_apos"))
        n_anos = max(1, idade_apos - idade_atual)
        info.update({
            "tem_atuarial": True,
            "regiao": ui.get("regiao"),
            "idade_atual": idade_atual,
            "idade_apos": idade_apos,
            "renda_mensal": float(ui.get("renda_mensal", 0.0)),
            "taxa_juros": float(ui.get("taxa_juros", 0.0)),
        })
    return n_anos, objetivo_reserva, info



def simular_monte_carlo_carteira(
    dados_close: pd.DataFrame,
    pesos: pd.Series,
    capital_inicial: float = 1000.0,
    n_anos: int = 10,
    simulacoes: int = 5000,
    passos_por_ano: int = 252,
    usar_bootstrap: bool = False,
    seed: int | None = 42,
    debug: bool = False
):
    """
    Simula o valor da carteira via Monte Carlo a partir dos retornos históricos da CARTEIRA.
    Estima μ_d e σ_d em base diária (log-retornos) e usa PASSO DIÁRIO (dt=1).
    """
    # 1) Alinhar colunas/pesos e normalizar
    cols = [c for c in dados_close.columns if c in pesos.index]
    if not cols:
        raise ValueError("No intersection between data columns and weight indices.")
    dados_close = dados_close[cols].ffill().dropna()
    pesos = pesos.loc[cols]
    pesos = pesos / pesos.sum()

    # 2) Retornos da carteira
    ret_ativos = dados_close.pct_change().dropna()
    ret_carteira = (ret_ativos @ pesos).dropna()

    # 3) Parâmetros diários (log-retornos)
    log_ret = np.log1p(ret_carteira)
    mu_d = float(log_ret.mean())                 # média diária
    sigma_d = float(log_ret.std(ddof=1))         # desvio diário

    if debug:
        # Diagnóstico para comparar com teus KPIs
        mu_a = (np.exp(mu_d) - 1) * passos_por_ano      # aprox. anual simples a partir de log-mean
        vol_a = sigma_d * np.sqrt(passos_por_ano)
        # Sharpe “histórico” aproximado (sem RF)
        sharpe_a = mu_a / vol_a if vol_a > 0 else np.nan
        st.write({
            "daily_log_mean": mu_d,
            "daily_log_std": sigma_d,
            "annual_return_approx(%)": mu_a * 100,
            "annual_volatility(%)": vol_a * 100,
            "sharpe_approx": sharpe_a
        })

    # 4) Simulação com passo diário (dt = 1 dia)
    n_passos = int(n_anos * passos_por_ano)
    rng = np.random.default_rng(seed)

    if usar_bootstrap:
        hist = ret_carteira.values
        idx = rng.integers(0, len(hist), size=(n_passos, simulacoes))
        fatores = 1.0 + hist[idx]  # retorno simples reamostrado por dia
    else:
        # GBM diário: S_{t+1} = S_t * exp((μ_d - 0.5σ_d^2) + σ_d * Z_t)
        Z = rng.standard_normal(size=(n_passos, simulacoes))
        fatores = np.exp((mu_d - 0.5 * sigma_d**2) + sigma_d * Z)

    fatores_diarios = fatores
      
    # 5) Trajetórias
    niveis = np.vstack([np.ones((1, simulacoes)), fatores]).cumprod(axis=0)
    valores = capital_inicial * niveis  # shape = (tempo, simulações)

    # 6) Percentis ao longo do tempo
    pcts = np.percentile(valores, [5, 25, 50, 75, 95], axis=1)
    t_anos = np.arange(valores.shape[0]) / passos_por_ano
    percentiles_df = pd.DataFrame({
        "years": t_anos,
        "p05": pcts[0], "p25": pcts[1], "p50": pcts[2], "p75": pcts[3], "p95": pcts[4]
    })

    # 7) Fatores anuais (para cálculo de aportes)
    total_completo = n_anos * passos_por_ano
    fd = fatores_diarios[:total_completo, :]
    fatores_anuais = fd.reshape(n_anos, passos_por_ano, simulacoes).prod(axis=1)

    # 8) Distribuição terminal e CAGR
    vt = valores[-1, :]
    cagr = (vt / capital_inicial) ** (1.0 / n_anos) - 1.0

    # 8) Métricas
    var5 = np.percentile(vt - capital_inicial, 5)
    es5 = (vt - capital_inicial)[(vt - capital_inicial) <= var5].mean()
    resumo = {
        "mu_log_diario": mu_d,
        "sigma_log_diario": sigma_d,
        "mediana_final": float(np.median(vt)),
        "p5_final": float(np.percentile(vt, 5)),
        "p95_final": float(np.percentile(vt, 95)),
        "prob_perda": float(np.mean(vt < capital_inicial)),
        "var5_eur_10y": float(var5),
        "es5_eur_10y": float(es5) if not np.isnan(es5) else None,
        "cagr_mediana": float(np.median(cagr)),
        "cagr_p5": float(np.percentile(cagr, 5)),
        "cagr_p95": float(np.percentile(cagr, 95)),
        "capital_inicial": float(capital_inicial),
        "n_anos": int(n_anos),
    }

    return {
        "percentiles_df": percentiles_df,
        "valores_terminais": vt,
        "cagr_sim": cagr,
        "resumo": resumo,
        "fatores_anuais": fatores_anuais,
    }

# --- Cálculo do aporte anual necessário (IGNORANDO capital inicial) ---
def calcular_aporte_anual_sem_capital(fatores_anuais: np.ndarray, FV_objetivo: float) -> np.ndarray:
    """
    Valor futuro com aportes no fim de cada ano, desconsiderando o capital inicial:
      FV = A * Σ Π(G_{t+1..N})
      => A = FV_objetivo / S
    """
    N, sims = fatores_anuais.shape

    suf_prod = np.ones((N + 1, sims))
    for k in range(N - 1, -1, -1):
        suf_prod[k, :] = suf_prod[k + 1, :] * fatores_anuais[k, :]
    S = np.sum(suf_prod[1:, :], axis=0)

    S_seguro = np.where(S <= 1e-12, np.nan, S)
    A_raw = FV_objetivo / S_seguro
    return A_raw



def mostrar_simulacao_carteira(
    resultado_mc: dict,
    titulo: str = None,
    objetivo_reserva: float | None = None
):
    """
    Mostra:
      - KPIs (median, P5, P95, prob. loss, CAGR quantis)
      - Fan chart (P5–P95) do valor simulado
      - PDF do CAGR
      - Histograma e CDF do retorno final
      - (Opcional) Distribuição do aporte anual necessário (se objetivo_reserva for passado)
    Labels em inglês; código e comentários em PT-BR.
    """


    # 1) Extrair dados da simulação
    p = resultado_mc["percentiles_df"]
    vt = resultado_mc["valores_terminais"]
    cagr = resultado_mc["cagr_sim"]
    r = resultado_mc["resumo"]
    fatores_anuais = resultado_mc.get("fatores_anuais", None)

    n_anos = int(r.get("n_anos", 10))
    capital_inicial = float(r.get("capital_inicial", 1000.0))

    # 2) Título dinâmico
    if not titulo:
        titulo = f"🔮 Monte Carlo Simulation ({n_anos} years)"

    # Subheader for Monte Carlo Simulation
    st.subheader("🔮 Monte Carlo Simulation", help="Monte Carlo simulation helps estimate the potential future performance of your portfolio by simulating thousands of possible outcomes based on historical data.", divider='blue')

    # Add caption below the subheader
    st.caption(
        "Monte Carlo Simulation provides a probabilistic approach to understanding potential portfolio outcomes. "
        "By simulating thousands of scenarios with an initial investment of 1000 units, it helps investors assess risks, "
        "identify potential returns, and make informed decisions based on a range of possible future states."
    )

    # 3) KPIs principais
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median terminal ()", f"{r['mediana_final']:,.0f}", help="The median terminal value represents the middle value of all simulated portfolio outcomes at the end of the investment horizon.")
    c2.metric("P5 terminal ()",     f"{r['p5_final']:,.0f}", help="The 5th percentile terminal value indicates the value below which only 5% of the simulated outcomes fall, representing a pessimistic scenario.")
    c3.metric("P95 terminal ()",    f"{r['p95_final']:,.0f}", help="The 95th percentile terminal value indicates the value above which only 5% of the simulated outcomes fall, representing an optimistic scenario.")
    c4.metric("Loss probability",     f"{100*r['prob_perda']:.1f}%", help="The probability of loss shows the likelihood of the portfolio ending with a value lower than the initial capital.")

    c5, c6, c7 = st.columns(3)
    c5.metric("Median CAGR", f"{100*r['cagr_mediana']:.2f}%", help="The median Compound Annual Growth Rate (CAGR) represents the typical annualized return of the portfolio over the investment horizon.")
    c6.metric("CAGR P5",     f"{100*r['cagr_p5']:.2f}%", help="The 5th percentile CAGR represents a pessimistic annualized return scenario.")
    c7.metric("CAGR P95",    f"{100*r['cagr_p95']:.2f}%", help="The 95th percentile CAGR represents an optimistic annualized return scenario.")

    # Fan chart
    st.markdown(
        "### Projected Portfolio Value (Fan Chart)",
        help="The fan chart illustrates the range of possible portfolio values over time, with percentiles showing the uncertainty in projections."
    )
    fig_fan = go.Figure()
    fig_fan.add_trace(go.Scatter(x=p["years"], y=p["p50"], mode="lines", name="Median"))
    fig_fan.add_trace(go.Scatter(x=p["years"], y=p["p25"], mode="lines", name="P25", line=dict(dash="dash")))
    fig_fan.add_trace(go.Scatter(x=p["years"], y=p["p75"], mode="lines", name="P75", line=dict(dash="dash")))
    fig_fan.add_trace(go.Scatter(
        x=np.concatenate([p["years"].values, p["years"].values[::-1]]),
        y=np.concatenate([p["p95"].values, p["p05"].values[::-1]]),
        fill="toself", name="P5–P95", mode="lines", line=dict(width=0), opacity=0.3
    ))
    fig_fan.update_layout(
        xaxis_title="Years", yaxis_title="Value ()", hovermode="x unified"
    )
    st.plotly_chart(fig_fan, use_container_width=True)
    st.caption("This chart provides a probabilistic view of your portfolio's future value. The shaded area represents the range between the 5th and 95th percentiles, helping you understand the potential risks and rewards.")

    # PDF do CAGR
    st.markdown(
        "### CAGR Distribution (Annualized)",
        help="This chart shows the distribution of annualized returns (CAGR) from the simulation, helping you assess the likelihood of different growth rates."
    )
    hist_vals, bins = np.histogram(cagr, bins=60, density=True)
    xmid = (bins[:-1] + bins[1:]) / 2.0
    fig_pdf = go.Figure()
    fig_pdf.add_trace(go.Bar(x=xmid, y=hist_vals, opacity=0.7, name="Density"))
    fig_pdf.update_layout(
        xaxis_title="CAGR", yaxis_title="Density"
    )
    st.plotly_chart(fig_pdf, use_container_width=True)
    st.caption("The CAGR distribution chart helps you understand the range and likelihood of annualized returns for your portfolio, based on the simulation.")

    # 6) Histograma do retorno terminal
    st.markdown(
        "### Terminal Return Distribution",
        help="This chart shows the distribution of terminal returns from the simulation, helping you assess the likelihood of different outcomes."
    )
    ret_final = vt / capital_inicial - 1.0
    h2, b2 = np.histogram(ret_final, bins=60, density=True)
    x2 = (b2[:-1] + b2[1:]) / 2.0
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(x=x2, y=h2, opacity=0.7, name="Density"))
    fig_hist.update_layout(
        xaxis_title="Terminal Return (%)",
        yaxis_title="Density"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("The terminal return distribution chart provides insights into the range of possible returns at the end of the investment period. It highlights the most likely outcomes and helps you understand the variability of potential returns.")

    # 7) CDF empírica
    st.markdown(
        "### Cumulative Distribution (Terminal Return)",
        help="This chart illustrates the cumulative probability of terminal returns from the simulation. It helps you understand the likelihood of achieving or exceeding specific return levels."
    )
    orden = np.sort(ret_final)
    cdf = np.linspace(0, 1, len(orden))
    fig_cdf = go.Figure()
    fig_cdf.add_trace(go.Scatter(x=orden, y=cdf, mode="lines", name="CDF"))
    fig_cdf.update_layout(
        xaxis_title="Terminal Return", yaxis_title="Cumulative Probability"
    )
    st.plotly_chart(fig_cdf, use_container_width=True)
    st.caption("The cumulative distribution chart provides a probabilistic view of terminal returns, showing the likelihood of achieving or exceeding specific outcomes at the end of the investment period.")


    # 8) Aporte anual (modelo 2: ignorar capital inicial)
    if objetivo_reserva is not None and np.isfinite(objetivo_reserva) and fatores_anuais is not None:
        A_sem = calcular_aporte_anual_sem_capital(
            fatores_anuais=fatores_anuais,
            FV_objetivo=float(objetivo_reserva)
        )
        A_valid = A_sem[np.isfinite(A_sem)]

        st.markdown("### 💶 Required Annual Contribution")
        if A_valid.size > 0:
            A_mediana = float(np.median(A_valid))
            A_p5      = float(np.percentile(A_valid, 5))
            A_p95     = float(np.percentile(A_valid, 95))

            # --- parâmetros atuariais exibidos abaixo ---
            res = st.session_state.get("actuarial_result", {}) or {}
            ui  = st.session_state.get("user_inputs", {}) or {}
            objetivo_res_calc = float(res.get("reserva_aposentadoria", float("nan")))
            taxa_juros_calc   = float(ui.get("taxa_juros", 0.0))
            idade_atual       = ui.get("idade_atual", None)
            idade_apos        = ui.get("idade_apos", None)

            st.caption(
                f"On average, the 'Annual Contribution (Median)' value of {A_mediana:,.2f}  represents the amount you need to save annually and invest in this portfolio, with the given weights, to potentially achieve your retirement goal of {objetivo_res_calc:,.2f}."
            )

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Annual Contribution (Median)", f"{A_mediana:,.2f} ", help="The median annual contribution represents the typical amount you need to save each year to reach your retirement goal, based on the portfolio's performance.")
            d2.metric("Annual Contribution (P5)",     f"{A_p5:,.2f} ", help="The 5th percentile annual contribution indicates a more optimistic scenario where you need to save less annually to achieve your goal.")
            d3.metric("Annual Contribution (P95)",    f"{A_p95:,.2f} ", help="The 95th percentile annual contribution represents a more conservative scenario where you need to save more annually to achieve your goal.")
            d4.metric("Horizon",                      f"{n_anos} years", help="The investment horizon is the number of years you plan to save and invest to reach your retirement goal.")

            st.markdown("### 📊 Actuarial Parameters", help="This section provides key actuarial parameters used to calculate your retirement goal and required contributions.")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Actuarial target (retirement)",
                      f"{objetivo_res_calc:,.2f} " if np.isfinite(objetivo_res_calc) else "—",
                      help="The actuarial target is the total amount you aim to accumulate by the time you retire.")
            a2.metric("Discount/interest rate",
                      f"{taxa_juros_calc*100:.2f}%" if np.isfinite(taxa_juros_calc) else "—",
                      help="The discount or interest rate is the assumed annual return on your investments.")
            if idade_atual is not None:
                a3.metric("Current age", f"{int(idade_atual)}", help="Your current age, used to calculate the time remaining until retirement.")
            if idade_apos is not None:
                a4.metric("Retirement age", f"{int(idade_apos)}", help="Your planned retirement age, used to determine the investment horizon.")


            # --- distribuição do aporte ---
            st.markdown(
            "### Required annual contribution (distribution)",
            help="This chart shows the distribution of required annual contributions to reach your retirement goal, helping you understand the range of possible savings amounts.")            
            ha, ba = np.histogram(A_valid, bins=50, density=True)
            xa = (ba[:-1] + ba[1:]) / 2.0
            fig_A = go.Figure()
            fig_A.add_trace(go.Bar(x=xa, y=ha, opacity=0.7, name="Density"))
            fig_A.update_layout(
                xaxis_title="Annual contribution", yaxis_title="Density"
            )
            st.plotly_chart(fig_A, use_container_width=True)
            st.caption("The distribution of required annual contributions helps you understand the variability in the amount you need to save each year to reach your retirement goal, based on different market scenarios.")


            # Evolution of reserve until retirement
            G_med = np.median(fatores_anuais, axis=1)
            V_path = [capital_inicial]
            for k in range(n_anos):
                V_next = V_path[-1] * G_med[k] + A_mediana
                V_path.append(V_next)
            anos_axis = np.arange(0, n_anos + 1, 1)

            p_years = p["years"].values
            p50_vals = p["p50"].values
            p50_anual = []
            for k in range(n_anos + 1):
                idx = int(np.argmin(np.abs(p_years - k)))
                p50_anual.append(p50_vals[idx])

            fig_ev = go.Figure()
            fig_ev.add_trace(go.Bar(x=anos_axis, y=V_path,
                                    name="Reserve with median annual contribution"))
            fig_ev.add_trace(go.Scatter(x=anos_axis, y=p50_anual,
                                        mode="lines+markers",
                                        name="Median (no contributions)"))
            fig_ev.update_layout(
                title="Evolution of reserve until retirement",
                xaxis_title="Years", yaxis_title="Value ()", barmode="overlay", hovermode="x unified"
            )
            st.plotly_chart(fig_ev, use_container_width=True)
            st.caption("This chart illustrates how your portfolio reserve is expected to grow over time with the median annual contribution, compared to the median growth without additional contributions.")


            # Cheque determinístico com CAGR mediano
            r_med = float(r["cagr_mediana"])
            if r_med > -0.9999:
                if abs(r_med) < 1e-12:
                    A_det = (objetivo_reserva - capital_inicial) / max(1, n_anos)
                else:
                    fator = (1 + r_med) ** n_anos
                    A_det = (objetivo_reserva - capital_inicial * fator) / ((fator - 1) / r_med)
                st.caption(
                    f"✅ Deterministic check (median CAGR ≈ {100*r_med:.2f}%): "
                    f"≈ **{A_det:,.2f} /year**"
                )
            else:
                st.caption("⚠️ Median CAGR invalid for deterministic check.")

        # 9) Nota
        st.caption("Notes: parameters estimated from daily log-returns of the portfolio. "
                   "Enable 'usar_bootstrap=True' to preserve historical tails.")







#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 📊 Optimized Portfolio Dashboard
st.title("Optimized Portfolio Dashboard")

st.sidebar.header("Portfolio Configuration")
st.sidebar.markdown("Use the options below to configure your portfolio and optimize it.")

# Section: Asset Types
st.sidebar.subheader("Asset Types")
st.sidebar.markdown("Select one or more asset types to filter the available assets.")
tipos = sorted(universo.get("Tipo de Ativo", pd.Series()).dropna().unique().tolist())
tipos_escolhidos = st.sidebar.multiselect("Asset Types", tipos, help="Choose one or more asset types to include in your portfolio.")

# Filter data based on selected asset types
dados_filtrados = universo[universo["Tipo de Ativo"].isin(tipos_escolhidos)] if tipos_escolhidos else universo

# Section: Asset Selection
st.sidebar.subheader("Asset Selection")
st.sidebar.markdown("Search and select the assets to include in your portfolio.")

# Group assets by categories for easier navigation
categories = ["Tipo de Ativo"]
selected_assets = []

for category in categories:
    if category in dados_filtrados.columns:
        unique_values = sorted(dados_filtrados[category].dropna().unique())
        for value in unique_values:
            with st.sidebar.expander(f"{value}"):
                filtered_assets = dados_filtrados[dados_filtrados[category] == value]

                # Add Equity-specific filters inside the Equity expander
                if value.lower() == "equity":
                    st.markdown("Refine your equity selection using the filters below.")
                    for coluna, label in zip(["País", "Setor", "Indústria"], ["Country", "Sector", "Industry"]):
                        if coluna in filtered_assets.columns:
                            opcoes = sorted(filtered_assets[coluna].dropna().unique())
                            if len(opcoes) > 1:
                                escolha = st.selectbox(label, ["All"] + opcoes, help=f"Filter equities by {label.lower()}.")
                                if escolha != "All":
                                    filtered_assets = filtered_assets[filtered_assets[coluna] == escolha]

                # Multiselect for assets
                asset_names = filtered_assets["Nome Curto"].tolist()
                selected = st.multiselect(f"Select assets in {value}", asset_names, key=f"{category}_{value}")
                selected_assets.extend(selected)

# Map selected asset names to tickers
nomes_para_tickers = dados_filtrados.set_index("Nome Curto")["Ticker"].dropna().to_dict()
selecionados = [nomes_para_tickers[nome] for nome in selected_assets]

# Section: Investment Parameters
st.sidebar.subheader("Investment Parameters")
st.sidebar.markdown("Define the parameters for your investment.")
anos = st.sidebar.slider("Investment Horizon (years)", 1, 20, 10, help="Set the number of years for your investment horizon.")
frequencia = st.sidebar.selectbox("Frequency", ["1d", "1wk", "1mo"], help="Choose the frequency of data for analysis.")

# Section: Benchmark
st.sidebar.subheader("Benchmark")
st.sidebar.markdown("Select a benchmark to compare your portfolio's performance.")
benchmarks_df = universo[universo["Tipo de Ativo"] == "INDEX"]
benchmarks_opcoes = benchmarks_df.set_index("Nome Curto")["Ticker"].dropna().to_dict()
benchmark_escolhido_nome = st.sidebar.selectbox("Benchmark", ["None"] + list(benchmarks_opcoes.keys()), help="Choose a benchmark index for comparison.")
benchmark_ticker = benchmarks_opcoes.get(benchmark_escolhido_nome) if benchmark_escolhido_nome != "None" else None

# Optimize Portfolio Button
if st.sidebar.button("Optimize Portfolio"):
    if selecionados:
        data_inicio = pd.Timestamp.today() - pd.DateOffset(years=anos)
        data_fim = pd.Timestamp.today()
        dados = obter_dados(selecionados, data_inicio, data_fim, frequencia)
        pesos = otimizar_carteira(dados)
        metrics = calcular_metricas(dados, pesos)

        mostrar_kpis(metrics)
        mostrar_tabela_ativos(dados, pesos, nomes_para_tickers)
        mostrar_graficos_ativos(pesos, anos, frequencia, nomes_para_tickers)
        mostrar_performance(dados, pesos)
        mostrar_fronteira_heatmap(dados, pesos)
        mostrar_benchmark_simples(dados, pesos, benchmark_ticker=benchmark_ticker, nomes_para_tickers=nomes_para_tickers, anos=anos)

        n_anos, objetivo_reserva, info = obter_objetivo_atuarial()
        resultado_mc = simular_monte_carlo_carteira(dados, pesos, capital_inicial=1_000.0, n_anos=n_anos, simulacoes=10_000)
        mostrar_simulacao_carteira(resultado_mc, objetivo_reserva=objetivo_reserva)

        st.write("Preview of loaded data:")
        st.dataframe(dados.head())
    else:
        st.warning("Please select at least one asset to optimize the portfolio.")

st.caption("Disclaimer: This is a simulation tool. Past performance is not indicative of future results. The outcomes presented are based on historical data and assumptions, and actual results may vary.")
