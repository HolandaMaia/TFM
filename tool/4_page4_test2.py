import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard da Carteira", layout="wide")


def carregar_universo(path="dados/ativos_totais.xlsx"):
    """Carrega o universo de ativos a partir de um arquivo Excel."""
    try:
        df = pd.read_excel(path)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar base de ativos: {e}")
        return pd.DataFrame()


def obter_dados(tickers, start, end, interval="1d"):
    data = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True)["Close"]
    return data.dropna()


def otimizar_carteira(dados):
    returns = np.log(dados / dados.shift(1)).dropna()
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(len(mu))
    w = inv_cov @ mu
    w = np.maximum(w, 0)
    w = w / np.sum(w)
    return pd.Series(w, index=dados.columns)


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


def mostrar_kpis(metrics):
    retorno, vol, sharpe, drawdown, _n, desvio, acumulado = metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Retorno Esperado", f"{retorno:.2%}")
    col2.metric("Volatilidade", f"{vol:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col4, col5, col6 = st.columns(3)
    col4.metric("Drawdown Máximo", f"{drawdown:.2%}")
    col5.metric("Desvio Padrão", f"{desvio:.2%}")
    col6.metric("Retorno Acumulado", f"{acumulado:.2%}")




def mostrar_performance(dados, pesos):
    st.subheader("Performance Histórica")
    returns = np.log(dados / dados.shift(1)).dropna()
    carteira = (1 + returns.dot(pesos)).cumprod()
    drawdown = carteira / carteira.cummax() - 1

    col1, col2 = st.columns([3, 1])
    fig_ret = px.area(x=carteira.index, y=carteira, title="Retorno Acumulado")
    col1.plotly_chart(fig_ret, use_container_width=True)

    fig_dd = px.area(x=drawdown.index, y=drawdown, title="Drawdown")
    fig_dd.update_yaxes(tickformat=".0%")
    col2.plotly_chart(fig_dd, use_container_width=True)


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
                    labels={"x": "Volatilidade", "y": "Retorno"},
                    title="Fronteira de Markowitz")
    carteira_ret = np.dot(pesos, mu)
    carteira_vol = np.sqrt(pesos.values.T @ cov.values @ pesos.values)
    fig.add_trace(go.Scatter(x=[carteira_vol], y=[carteira_ret], mode="markers",
                            marker=dict(color="red", size=10), name="Carteira"))
    return fig


def mostrar_tabela_ativos(dados, pesos):
    st.subheader("Detalhamento por Ativo")
    ultimo_preco = dados.ffill().iloc[-1]
    investimento = pesos * 1000  # hipotético
    unidades = investimento / ultimo_preco
    var_1d = dados.pct_change().iloc[-1]
    var_7d = dados.pct_change(7).iloc[-1]
    var_1m = dados.pct_change(21).iloc[-1]
    tabela = pd.DataFrame({
        "Ticker": pesos.index,
        "Peso": (pesos * 100).round(2),
        "Preço Atual": ultimo_preco.round(2),
        "Unidades": unidades.round(2),
        "Valor Investido": investimento.round(2),
        "Variação 24h": (var_1d * 100).round(2),
        "7d": (var_7d * 100).round(2),
        "1m": (var_1m * 100).round(2),
    })
    st.dataframe(tabela, use_container_width=True)



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


def mostrar_graficos_ativos(pesos, anos, frequencia):
    st.subheader("📊 Análise Técnica por Ativo")

    # Define datas com base no parâmetro "anos"
    data_fim = pd.Timestamp.today()
    data_inicio = data_fim - pd.DateOffset(years=anos)

    # Baixa os dados OHLC completos apenas para os tickers da carteira
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
        if ticker not in dados_ohlc.columns.levels[0]:
            st.warning(f"❗ Dados não disponíveis para {ticker}")
            continue

        df = dados_ohlc[ticker].dropna().copy()
        df = df.rename(columns=str.lower)
        df['date'] = df.index

        if df.empty or df[['open', 'high', 'low', 'close']].isna().all().any():
            st.warning(f"❗ Dados incompletos para {ticker}")
            continue

        # Calcular indicadores
        sma_values = {
            20: df['close'].rolling(window=20).mean(),
            50: df['close'].rolling(window=50).mean()
        }

        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        # Plota gráfico técnico
        fig = plot_combined_chart(df, ticker, sma_values=sma_values, macd=macd, signal=signal)
        st.plotly_chart(fig, use_container_width=True)



def mostrar_backtest(dados, pesos, benchmark_ticker=None):
    st.subheader("📈 Backtest da Carteira")

    if dados.empty:
        st.warning("Dados da carteira não disponíveis.")
        return

    data_inicio = dados.index.min()
    data_fim = dados.index.max()

    if benchmark_ticker is None:
        benchmark_ticker = "^GSPC"

    try:
        benchmark_df = yf.download(
            benchmark_ticker,
            start=data_inicio,
            end=data_fim,
            auto_adjust=True,
            progress=False
        )
        st.write(f"Prévia dos dados brutos de `{benchmark_ticker}`:")
        st.dataframe(benchmark_df.head())

    except Exception as e:
        st.error(f"Erro ao baixar benchmark: {e}")
        return

    if benchmark_df.empty or "Close" not in benchmark_df.columns:
        st.warning("Benchmark inválido ou sem dados de fechamento.")
        return

    benchmark = benchmark_df["Close"].dropna()

    # 👉 Retornos da carteira
    returns = np.log(dados / dados.shift(1)).dropna()
    carteira = (1 + returns.dot(pesos)).cumprod()

    # 👉 Retornos acumulados do benchmark (base 1)
    benchmark_ret = np.log(benchmark / benchmark.shift(1)).dropna()
    benchmark_cum = (1 + benchmark_ret).cumprod()

    # 👉 Alinhamento de datas
    benchmark_cum = benchmark_cum.reindex(carteira.index, method="ffill").dropna()
    carteira = carteira.loc[benchmark_cum.index]

    # 📊 Gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=carteira.index, y=carteira, name="Carteira"))
    fig.add_trace(go.Scatter(x=benchmark_cum.index, y=benchmark_cum, name=benchmark_ticker))
    fig.update_layout(
        title="Evolução Acumulada da Carteira vs Benchmark",
        xaxis_title="Data",
        yaxis_title="Valor Normalizado",
        legend_title="Série",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)







def mostrar_heatmap(dados):
    corr = dados.pct_change().corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, ax=ax, annot=True, cmap="RdBu", center=0)
    return fig


def mostrar_fronteira_heatmap(dados, pesos):
    st.subheader("Fronteira Eficiente e Correlação")
    col1, col2 = st.columns([3, 1])
    with col1:
        fig_front = mostrar_fronteira(dados, pesos)
        st.plotly_chart(fig_front, use_container_width=True)
    with col2:
        fig_heat = mostrar_heatmap(dados)
        st.pyplot(fig_heat)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
st.title("📊 Dashboard da Carteira Otimizada")

st.sidebar.header("Configurações da Carteira")
universo = carregar_universo()

# Filtro obrigatório: Tipo de Ativo
tipos = sorted(universo.get("Tipo de Ativo", pd.Series()).dropna().unique().tolist())
tipo_escolhido = st.sidebar.selectbox("Tipo de Ativo", tipos)
dados_filtrados = universo[universo["Tipo de Ativo"] == tipo_escolhido]

# Filtros dinâmicos opcionais (País, Setor, Indústria)
for coluna in ["País", "Setor", "Indústria"]:
    if coluna in dados_filtrados.columns:
        opcoes = sorted(dados_filtrados[coluna].dropna().unique())
        if len(opcoes) > 1:
            escolha = st.sidebar.selectbox(coluna, ["Todos"] + opcoes)
            if escolha != "Todos":
                dados_filtrados = dados_filtrados[dados_filtrados[coluna] == escolha]

# Multiselect: Nome Curto visível, Ticker interno
nomes_para_tickers = dados_filtrados.set_index("Nome Curto")["Ticker"].dropna().to_dict()
selecionados_nomes = st.sidebar.multiselect("Ativos", list(nomes_para_tickers.keys()))
selecionados = [nomes_para_tickers[nome] for nome in selecionados_nomes]

# Parâmetros adicionais
anos = st.sidebar.slider("Horizonte (anos)", 1, 20, 10)
frequencia = st.sidebar.selectbox("Frequência", ["1d", "1wk", "1mo"])
# 📌 Caixa para seleção de Benchmark (apenas ativos tipo "Index")
benchmarks_df = universo[universo["Tipo de Ativo"] == "INDEX"]
benchmarks_opcoes = benchmarks_df.set_index("Nome Curto")["Ticker"].dropna().to_dict()
benchmark_escolhido_nome = st.sidebar.selectbox("Benchmark", ["Nenhum"] + list(benchmarks_opcoes.keys()))
benchmark_ticker = benchmarks_opcoes.get(benchmark_escolhido_nome) if benchmark_escolhido_nome != "Nenhum" else None
btn = st.sidebar.button("Otimizar Carteira")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
if btn and selecionados:
    data_inicio = pd.Timestamp.today() - pd.DateOffset(years=anos)
    data_fim = pd.Timestamp.today()
    dados = obter_dados(selecionados, data_inicio, data_fim, frequencia)
    pesos = otimizar_carteira(dados)
    metrics = calcular_metricas(dados, pesos)

    mostrar_kpis(metrics)
    mostrar_tabela_ativos(dados, pesos)
    mostrar_graficos_ativos(pesos, anos, frequencia)
    mostrar_performance(dados, pesos)
    mostrar_fronteira_heatmap(dados, pesos)
    mostrar_backtest(dados, pesos)
    st.write("Prévia dos dados carregados:")
    st.dataframe(dados.head())
else:
    st.info("Escolha os ativos e clique em Otimizar Carteira")
