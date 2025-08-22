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

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_resource
def carregar_universo(path="dados/ativos_totais.xlsx"):
    """Carrega o universo de ativos a partir de um arquivo Excel."""
    try:
        df = pd.read_excel(path)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar base de ativos: {e}")
        return pd.DataFrame()

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
    w = np.maximum(w, 0)
    w = w / np.sum(w)
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
    col1, col2, col3 = st.columns(3)
    col1.metric("Retorno Esperado", f"{retorno:.2%}")
    col2.metric("Volatilidade", f"{vol:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col4, col5, col6 = st.columns(3)
    col4.metric("Drawdown Máximo", f"{drawdown:.2%}")
    col5.metric("Desvio Padrão", f"{desvio:.2%}")
    col6.metric("Retorno Acumulado", f"{acumulado:.2%}")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def mostrar_performance(dados, pesos):
    st.subheader("Performance Histórica")
    returns = np.log(dados / dados.shift(1)).dropna()
    carteira = (1 + returns.dot(pesos)).cumprod()
    drawdown = carteira / carteira.cummax() - 1

    col1, col2 = st.columns([3, 1])
    fig_ret = px.area(
    x=carteira.index,
    y=((carteira - 1) * 100),  # transforma em %
    title="Retorno Acumulado")
    fig_ret.update_layout(
    xaxis_title=None,
    yaxis_title=None,
    height=400)
    fig_ret.update_yaxes(ticksuffix="%")
    col1.plotly_chart(fig_ret, use_container_width=True)

    fig_dd = px.area(
    x=drawdown.index,
    y=drawdown,
    title="Drawdown")

    fig_dd.update_yaxes(tickformat=".0%", title=None)
    fig_dd.update_layout(xaxis_title=None)

    col2.plotly_chart(fig_dd, use_container_width=True)


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
                    labels={"x": "Volatilidade", "y": "Retorno"},
                    title="Fronteira de Markowitz")
    carteira_ret = np.dot(pesos, mu)
    carteira_vol = np.sqrt(pesos.values.T @ cov.values @ pesos.values)
    fig.add_trace(go.Scatter(x=[carteira_vol], y=[carteira_ret], mode="markers",
                            marker=dict(color="red", size=10), name="Carteira"))
    return fig

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
    with st.expander(f"📊 Technical Analysis by Asset"):
        st.subheader("📊 Technical Analysis by Asset")

        # Definir fechas
        fecha_fin = pd.Timestamp.today()
        fecha_inicio = fecha_fin - pd.DateOffset(years=anos)

        # Descargar datos OHLC
        tickers = list(pesos.index)
        datos_ohlc = yf.download(
            tickers,
            start=fecha_inicio,
            end=fecha_fin,
            interval=frequencia,
            auto_adjust=False,
            group_by='ticker'
        )

        for ticker in tickers:
            if ticker not in datos_ohlc.columns.levels[0]:
                st.warning(f"❗ Datos no disponibles para {ticker}")
                continue

            df = datos_ohlc[ticker].dropna().copy()
            df = df.rename(columns=str.lower)
            df['date'] = df.index

            if df.empty or df[['open', 'high', 'low', 'close']].isna().all().any():
                st.warning(f"❗ Datos incompletos para {ticker}")
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
            fig = plot_combined_chart(df, ticker, sma_values=sma_values, macd=macd, signal=signal)
            st.plotly_chart(fig, use_container_width=True)

        
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def mostrar_benchmark_simples(dados, pesos, benchmark_ticker, anos=10):
    st.subheader("📊 Análise Histórica do Benchmark")

    if not benchmark_ticker:
        st.warning("⚠️ Nenhum benchmark selecionado.")
        return

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
        st.error(f"❌ Erro ao baixar dados do benchmark: {e}")
        return

    if benchmark_df.empty or "Close" not in benchmark_df.columns:
        st.warning("⚠️ Dados de fechamento do benchmark indisponíveis.")
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
    st.markdown(f"### 💵 Evolução do Preço - `{benchmark_ticker}`")
    fig_close = px.line(
        x=close_series.index,
        y=close_series.values,
        title="Preço de Fechamento",
        labels={"x": "Data", "y": "Preço"}
    )
    fig_close.update_yaxes(tickprefix="US$ ", title="Preço")
    fig_close.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_close, use_container_width=True, key="grafico_preco_benchmark")

    # 📈 Gráfico 2: Retorno Acumulado (%)
    st.markdown("### 📈 Retorno Acumulado (%) - Benchmark vs Carteira")
    fig_ret = go.Figure()

    fig_ret.add_trace(go.Scatter(
        x=benchmark_ret_pct.index,
        y=benchmark_ret_pct.values,
        mode="lines",
        name=benchmark_ticker,
        line=dict(color="royalblue")
    ))

    fig_ret.add_trace(go.Scatter(
        x=carteira_pct.index,
        y=carteira_pct.values,
        mode="lines",
        name="Carteira",
        line=dict(color="seagreen")
    ))

    fig_ret.update_yaxes(title="Retorno (%)", ticksuffix="%")
    fig_ret.update_layout(
        height=400,
        xaxis_title="Data",
        title="Retorno Acumulado em %",
        legend_title="Série"
    )
    st.plotly_chart(fig_ret, use_container_width=True, key="grafico_retorno_comparado")



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def mostrar_heatmap(dados):
    corr = dados.pct_change().corr()
    # Estilo para valores legíveis
    sns.set(font_scale=1.1)
    # Figura proporcional à altura da fronteira
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(corr, ax=ax, annot=True, cmap="RdBu", center=0)
    return fig


def mostrar_fronteira_heatmap(dados, pesos):
    st.subheader("Fronteira Eficiente e Correlação")
    col1, col2 = st.columns([3, 1])
    with col1:
        fig_front = mostrar_fronteira(dados, pesos)
        st.plotly_chart(fig_front, use_container_width=True)
    with col2:
        st.markdown("** Matriz de Correlação**")  
        fig_heat = mostrar_heatmap(dados)
        st.pyplot(fig_heat, use_container_width=True)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --- Monte Carlo da carteira: cálculo (corrigido para passo diário) ---
import numpy as np
import pandas as pd

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
        raise ValueError("Não há interseção entre colunas de dados e índices de pesos.")
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
        import streamlit as st
        st.write({
            "mu_log_diario": mu_d,
            "sigma_log_diario": sigma_d,
            "retorno_anual_aprox(%)": mu_a * 100,
            "vol_anual(%)": vol_a * 100,
            "sharpe_aprox": sharpe_a
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

# --- Cálculo do aporte anual necessário ---
def calcular_aporte_anual(fatores_anuais: np.ndarray, V0: float, FV_objetivo: float) -> np.ndarray:
    """
    Fórmula de valor futuro com aportes no final de cada ano:
    FV = V0*ΠG + A * Σ Π(G_{t+1..N})
    => A = (FV_objetivo - V0*ΠG) / S
    """
    N, sims = fatores_anuais.shape
    prod_total = fatores_anuais.prod(axis=0)

    suf_prod = np.ones((N + 1, sims))
    for k in range(N - 1, -1, -1):
        suf_prod[k, :] = suf_prod[k + 1, :] * fatores_anuais[k, :]
    S = np.sum(suf_prod[1:, :], axis=0)

    numerador = FV_objetivo - V0 * prod_total
    S_seguro = np.where(S <= 1e-12, np.nan, S)
    A = numerador / S_seguro
    A = np.where(np.isnan(A), np.nan, np.maximum(0.0, A))
    return A


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
    Labels em inglês; textos em PT-BR.
    """
    import numpy as np
    import plotly.graph_objects as go
    import streamlit as st

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

    with st.expander(titulo, expanded=True):

        # 3) KPIs principais
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median terminal (€)", f"{r['mediana_final']:,.0f}")
        c2.metric("P5 terminal (€)",     f"{r['p5_final']:,.0f}")
        c3.metric("P95 terminal (€)",    f"{r['p95_final']:,.0f}")
        c4.metric("Loss probability",     f"{100*r['prob_perda']:.1f}%")

        c5, c6, c7 = st.columns(3)
        c5.metric("Median CAGR", f"{100*r['cagr_mediana']:.2f}%")
        c6.metric("CAGR P5",     f"{100*r['cagr_p5']:.2f}%")
        c7.metric("CAGR P95",    f"{100*r['cagr_p95']:.2f}%")

        # 4) Fan chart
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
            title="Projected portfolio value (fan chart)",
            xaxis_title="Years", yaxis_title="Value (€)", hovermode="x unified"
        )
        st.plotly_chart(fig_fan, use_container_width=True)

        # 5) PDF do CAGR
        hist_vals, bins = np.histogram(cagr, bins=60, density=True)
        xmid = (bins[:-1] + bins[1:]) / 2.0
        fig_pdf = go.Figure()
        fig_pdf.add_trace(go.Bar(x=xmid, y=hist_vals, opacity=0.7, name="Density"))
        fig_pdf.update_layout(
            title="CAGR distribution (annualized)",
            xaxis_title="CAGR", yaxis_title="Density"
        )
        st.plotly_chart(fig_pdf, use_container_width=True)

        # 6) Histograma do retorno terminal
        ret_final = vt / capital_inicial - 1.0
        h2, b2 = np.histogram(ret_final, bins=60, density=True)
        x2 = (b2[:-1] + b2[1:]) / 2.0
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(x=x2, y=h2, opacity=0.7, name="Density"))
        fig_hist.update_layout(
            title="Terminal return distribution (V_T / V_0 − 1)",
            xaxis_title="Terminal return", yaxis_title="Density"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # 7) CDF empírica
        orden = np.sort(ret_final)
        cdf = np.linspace(0, 1, len(orden))
        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(x=orden, y=cdf, mode="lines", name="CDF"))
        fig_cdf.update_layout(
            title="Cumulative distribution (terminal return)",
            xaxis_title="Terminal return", yaxis_title="Cumulative prob."
        )
        st.plotly_chart(fig_cdf, use_container_width=True)

        # 8) (Opcional) Aporte anual necessário se houver objetivo
        if objetivo_reserva is not None and np.isfinite(objetivo_reserva) and fatores_anuais is not None:
            # 8.1) Calcular distribuição de A
            A_vect = calcular_aporte_anual(
                fatores_anuais=fatores_anuais,
                V0=capital_inicial,
                FV_objetivo=float(objetivo_reserva)
            )
            A_valid = A_vect[np.isfinite(A_vect)]

            st.markdown("### 💶 Required Annual Contribution")
            if A_valid.size > 0:
                A_mediana = float(np.median(A_valid))
                A_p5      = float(np.percentile(A_valid, 5))
                A_p95     = float(np.percentile(A_valid, 95))

                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Annual Contribution (Median)", f"{A_mediana:,.2f} €")
                d2.metric("Annual Contribution (P5)",     f"{A_p5:,.2f} €")
                d3.metric("Annual Contribution (P95)",    f"{A_p95:,.2f} €")
                d4.metric("Horizon",                      f"{n_anos} years")

                # 8.2) Histograma de A
                ha, ba = np.histogram(A_valid, bins=50, density=True)
                xa = (ba[:-1] + ba[1:]) / 2.0
                fig_A = go.Figure()
                fig_A.add_trace(go.Bar(x=xa, y=ha, opacity=0.7, name="Density"))
                fig_A.update_layout(
                    title="Required annual contribution (distribution)",
                    xaxis_title="Annual contribution (€)", yaxis_title="Density"
                )
                st.plotly_chart(fig_A, use_container_width=True)

                # 8.3) Evolution of reserve until retirement (barras com A_mediana) + linha p50 sem aportes
                # 8.3.1) Fatores anuais medianos por ano (G_med)
                G_med = np.median(fatores_anuais, axis=1)  # shape = (n_anos,)

                # 8.3.2) Caminho da reserva com aportes anuais A_mediana (no fim do ano)
                V_path = [capital_inicial]
                for k in range(n_anos):
                    V_next = V_path[-1] * G_med[k] + A_mediana
                    V_path.append(V_next)
                anos_axis = np.arange(0, n_anos + 1, 1)

                # 8.3.3) Linha de referência: mediana do fan chart (sem aportes) nos anos inteiros
                p_years = p["years"].values
                p50_vals = p["p50"].values
                p50_anual = []
                for k in range(n_anos + 1):
                    idx = int(np.argmin(np.abs(p_years - k)))
                    p50_anual.append(p50_vals[idx])

                # 8.3.4) Plotar barras + linha
                fig_ev = go.Figure()
                fig_ev.add_trace(go.Bar(x=anos_axis, y=V_path, name="Reserve with median annual contribution"))
                fig_ev.add_trace(go.Scatter(x=anos_axis, y=p50_anual, mode="lines+markers", name="Median (no contributions)"))
                fig_ev.update_layout(
                    title="Evolution of reserve until retirement",
                    xaxis_title="Years", yaxis_title="Value", barmode="overlay", hovermode="x unified"
                )
                st.plotly_chart(fig_ev, use_container_width=True)

                # 8.4) Cheque determinístico com CAGR mediano
                r_med = float(r["cagr_mediana"])
                if r_med > -0.9999:
                    if abs(r_med) < 1e-12:
                        A_det = (objetivo_reserva - capital_inicial) / max(1, n_anos)
                    else:
                        fator = (1 + r_med) ** n_anos
                        A_det = (objetivo_reserva - capital_inicial * fator) / ((fator - 1) / r_med)
                    st.caption(
                        f"✅ Deterministic check (median CAGR ≈ {100*r_med:.2f}%): "
                        f"≈ **{A_det:,.2f} $/year**"
                    )
                else:
                    st.caption("⚠️ Median CAGR invalid for deterministic check.")

        # 9) Nota
        st.caption("Notes: parameters estimated from daily log-returns of the portfolio. "
                   "Enable 'usar_bootstrap=True' to preserve historical tails.")






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
    mostrar_benchmark_simples(dados, pesos, benchmark_ticker=benchmark_ticker, anos=anos)

    n_anos, objetivo_reserva, info = obter_objetivo_atuarial()
    resultado_mc = simular_monte_carlo_carteira(dados, pesos, capital_inicial=1_000.0, n_anos=n_anos, simulacoes=10_000)
    mostrar_simulacao_carteira(resultado_mc, objetivo_reserva=objetivo_reserva)

    st.write("Prévia dos dados carregados:")
    st.dataframe(dados.head())
else:
    st.info("Escolha os ativos e clique em Otimizar Carteira")
