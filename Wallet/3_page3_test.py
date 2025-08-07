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

def mostrar_detalhes_fundamentalistas(ticker: str):
    """Displays complete fundamental information about a stock."""
    try:
        ativo = yf.Ticker(ticker)
        info = ativo.info

        st.subheader("📊 Fundamental Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🏢 Company")
            st.markdown(f"**Name:** {info.get('longName', '-')}")
            st.markdown(f"**Ticker:** {info.get('symbol', '-')}")
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
            st.markdown(f"**Ex-Dividend Date:** {info.get('exDividendDate', 'N/A')}")
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

    except Exception as e:
        st.error(f"❌ Error loading fundamentals: {e}")



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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


def modelo_ml_prever_futuro(df: pd.DataFrame):
    """Permite prever os próximos 30 dias com ML, escolhendo janela e visualizando incertezas."""

    df = df.copy()

    # ✅ Verifica se a coluna 'Close' existe
    if 'Close' not in df.columns:
        st.error("❌ A coluna 'Close' não está presente no DataFrame fornecido.")
        st.write("🔎 Colunas disponíveis no DataFrame:", list(df.columns))
        return

    # ✅ Dropna só se 'Close' existe
    df = df.dropna(subset=['Close'])

    # 📏 Parâmetro: tamanho da janela deslizante
    window_size = st.slider("⏳ Select window size (past days used to predict the future)", 10, 90, 30, step=5)

    close_series = df['Close'].values
    X, y = [], []

    for i in range(len(close_series) - window_size):
        X.append(close_series[i:i+window_size])
        y.append(close_series[i+window_size])

    X = np.array(X)
    y = np.array(y)

    if len(X) < 30:
        st.warning("⚠️ Dados insuficientes para treinar o modelo.")
        return

    # Separação treino/teste
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Métricas
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    confidence = np.mean([model.score(X_train, y_train), model.score(X_test, y_test)])

    # 🎯 Métricas visuais
    st.subheader("🤖 Machine Learning Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎲 Model Confidence", f"{confidence:.2%}")
    col2.metric("📈 Train R²", f"{r2_train:.2%}")
    col3.metric("✅ Test R²", f"{r2_test:.2%}")
    col4.metric("📉 MAE (Test)", f"{mae_test:.2f}")

    # Previsão iterativa para os próximos 30 dias
    ultimos_dias = list(close_series[-window_size:])
    previsoes = []
    previsoes_std = []

    for _ in range(30):
        entrada = np.array(ultimos_dias[-window_size:]).reshape(1, -1)
        all_preds = np.stack([tree.predict(entrada) for tree in model.estimators_])
        media = all_preds.mean()
        desvio = all_preds.std()

        previsoes.append(media)
        previsoes_std.append(desvio)
        ultimos_dias.append(media)

    # Datas futuras
    datas_futuras = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
    df_previsao = pd.DataFrame({
        'Date': datas_futuras,
        'Predicted Close': previsoes,
        'Lower Bound': np.array(previsoes) - np.array(previsoes_std),
        'Upper Bound': np.array(previsoes) + np.array(previsoes_std),
    }).set_index('Date')

    # 📊 Gráfico com intervalo de confiança
    st.subheader("📊 Forecast: Next 30 Days (with confidence intervals)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df_previsao.index,
        y=df_previsao['Predicted Close'],
        mode='lines',
        name='Predicted Close',
        line=dict(color='orange', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df_previsao.index.tolist() + df_previsao.index[::-1].tolist(),
        y=(df_previsao['Upper Bound'].tolist() + df_previsao['Lower Bound'][::-1].tolist()),
        fill='toself',
        fillcolor='rgba(255,165,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Confidence Interval'
    ))

    fig.update_layout(
        title=f"30-Day Forecast (Window: {window_size} days)",
        xaxis_title="Date",
        yaxis_title="Close Price (€)",
        height=500,
        margin=dict(t=30, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    return df_previsao


    
    
    
# ---------------------------------------------------------------------
# Página principal
st.set_page_config(page_title="Análise de Ativo", layout="wide")
st.title("🔎 Análise Individual de Ativo")

# Sidebar de filtros
st.sidebar.header("Configurações do Ativo")
universo = carregar_universo()

tipos = sorted(universo.get("Tipo de Ativo", pd.Series()).dropna().unique().tolist())
tipo_escolhido = st.sidebar.selectbox("Tipo de Ativo", tipos)
dados_filtrados = universo[universo["Tipo de Ativo"] == tipo_escolhido]

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

    if dados.empty:
        st.warning("⚠️ Nenhum dado encontrado para esse ativo e período.")
    else:
        
        
        mostrar_detalhes_fundamentalistas(ticker)
        mostrar_grafico_tecnico(ticker, dados)
        metricas = calcular_metricas_performance(dados)
        mostrar_metricas_performance(metricas)
        modelo_ml_prever_futuro(dados)



        

        with st.expander("🔍 Ver dados brutos"):
            st.dataframe(dados.tail())

else:
    st.info("Escolha um ativo e clique em **Analisar Ativo**.")
