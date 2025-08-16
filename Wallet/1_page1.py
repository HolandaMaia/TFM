# app_streamlit_tfm_demo.py
# -----------------------------------------------------------
# Página única demonstrando:
# - Carteira Markowitz (histórica) vs Markowitz + ML (µ previsto)
# - KPIs comparativos no topo
# - Curva de retorno acumulado e drawdown comparados
# - Fronteira eficiente (amostragem) com os dois pontos destacados
# - Heatmap de correlação e tabela detalhada por ativo
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="📊 Comparativo Markowitz vs ML", layout="wide")
np.random.seed(42)

# ===========================================================
# 1) DADOS
# ===========================================================
@st.cache_resource
def carregar_universo(path="dados/ativos_totais.xlsx"):
    try:
        return pd.read_excel(path)
    except Exception as e:
        st.error(f"Erro ao carregar base de ativos: {e}")
        return pd.DataFrame()

@st.cache_resource
def obter_dados(tickers, start, end, interval="1d"):
    df = yf.download(
        tickers, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False
    )["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(how="all").ffill().dropna(axis=1, how="any")
    return df

def limpar_series_ruins(precos: pd.DataFrame, tol_const=1e-12):
    # Remove colunas com todos zeros/NaN ou séries quase constantes
    ok_cols = []
    for c in precos.columns:
        s = precos[c].astype(float)
        if s.isna().all(): 
            continue
        if (s == 0).all():
            continue
        if s.max() - s.min() < tol_const:
            continue
        ok_cols.append(c)
    return precos[ok_cols]

def stats_mu_cov(precos: pd.DataFrame, tipo="log"):
    if tipo == "log":
        rets = np.log(precos / precos.shift(1)).dropna()
    else:
        rets = precos.pct_change().dropna()
    mu = rets.mean() * 252.0
    cov = rets.cov() * 252.0
    return rets, mu, cov

# ===========================================================
# 2) PREVISÃO ML — µ_ml (anual)
# ===========================================================
def prever_mu_ml(precos: pd.DataFrame, janela=180):
    """
    Para cada ativo:
      - cria features em retornos diários (lag1, média 5/21, vol 5/21)
      - treina RF nos últimos 'janela' pontos
      - prevê retorno diário "amanhã" e anualiza (×252)
    Retorna: pd.Series mu_ml (anual) e df_previsoes (tabela para UI)
    """
    mu_ml = {}
    linhas = []
    for ativo in precos.columns:
        p = precos[ativo].dropna()
        if len(p) < (janela + 50):  # mínimo para features + janela
            continue

        r = p.pct_change().dropna()
        df = pd.DataFrame({
            "ret": r,
            "lag1": r.shift(1),
            "m5": r.rolling(5).mean(),
            "m21": r.rolling(21).mean(),
            "vol5": r.rolling(5).std(),
            "vol21": r.rolling(21).std(),
        }).dropna()

        # recorte final da janela para treino
        df_win = df.iloc[-janela:].copy()
        if len(df_win) < 60:
            continue

        X = df_win.drop(columns=["ret"])
        y = df_win["ret"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        # Previsão do próximo dia: usa a última linha de features
        x_last = X_scaled[-1].reshape(1, -1)
        r_hat_diario = float(model.predict(x_last)[0])
        mu_anual = r_hat_diario * 252.0

        mu_ml[ativo] = mu_anual
        linhas.append({
            "Ativo": ativo,
            "Previsão diária (%)": r_hat_diario * 100,
            "Previsão anual (%)": mu_anual * 100
        })

    mu_ml = pd.Series(mu_ml, name="mu_ml")
    df_prev = pd.DataFrame(linhas).sort_values("Ativo")
    return mu_ml, df_prev

# ===========================================================
# 3) OTIMIZAÇÃO (amostragem com restrições)
# ===========================================================
def amostrar_pesos(n, wmax=0.3, n_amostras=10000):
    """
    Gera amostras com pesos >=0, soma=1 e cada peso <= wmax.
    Estratégia: Dirichlet + rejeição simples.
    """
    amostras = []
    tentativas = 0
    while len(amostras) < n_amostras and tentativas < n_amostras * 20:
        w = np.random.dirichlet(np.ones(n))
        if (w <= wmax + 1e-9).all():
            amostras.append(w)
        tentativas += 1
    if len(amostras) == 0:
        # fallback: relaxa limite
        for _ in range(n_amostras):
            w = np.random.dirichlet(np.ones(n))
            amostras.append(w)
    return np.array(amostras)

def melhor_sharpe(mu: pd.Series, cov: pd.DataFrame, rf=0.0, wmax=0.3, n_amostras=10000):
    """
    Maximiza Sharpe por busca aleatória com restrições simples.
    """
    ativos = list(mu.index)
    n = len(ativos)
    W = amostrar_pesos(n, wmax=wmax, n_amostras=n_amostras)
    mu_vec = mu.values
    cov_mat = cov.values

    # retorno e vol para todas as amostras
    rets = W @ mu_vec
    vols = np.sqrt(np.einsum('ij,jk,ik->i', W, cov_mat, W))
    sharpe = (rets - rf) / np.where(vols == 0, np.nan, vols)
    idx = np.nanargmax(sharpe)
    w_best = pd.Series(W[idx], index=ativos, name="pesos")
    return w_best, float(rets[idx]), float(vols[idx]), float(sharpe[idx])

# ===========================================================
# 4) AVALIAÇÃO
# ===========================================================
def curva_carteira(precos: pd.DataFrame, pesos: pd.Series, tipo="log"):
    if tipo == "log":
        r = np.log(precos / precos.shift(1)).dropna()
    else:
        r = precos.pct_change().dropna()
    curva = (1 + r.dot(pesos)).cumprod()
    return curva

def metricas(precos: pd.DataFrame, pesos: pd.Series):
    r = np.log(precos / precos.shift(1)).dropna()
    mu = r.mean() * 252
    cov = r.cov() * 252
    port_ret = float((pesos * mu).sum())
    port_vol = float(np.sqrt(pesos.values.T @ cov.values @ pesos.values))
    sharpe = port_ret / port_vol if port_vol > 0 else 0.0
    curva = (1 + r.dot(pesos)).cumprod()
    running_max = curva.cummax()
    dd = float(((curva - running_max) / running_max).min())
    acum_1y = float(curva.iloc[-1] - 1.0)
    std_ann = float(r.std().mean() * np.sqrt(252))
    return {
        "ret": port_ret, "vol": port_vol, "sharpe": sharpe,
        "dd": dd, "std": std_ann, "acum_1y": acum_1y
    }

# ===========================================================
# 5) VISUAIS
# ===========================================================
def kpis_duas_carteiras(mtz, ml):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Retorno (Hist.)", f"{mtz['ret']:.2%}")
    c2.metric("Vol (Hist.)", f"{mtz['vol']:.2%}")
    c3.metric("Sharpe (Hist.)", f"{mtz['sharpe']:.2f}")
    c4.metric("Retorno (ML)", f"{ml['ret']:.2%}")
    c5.metric("Vol (ML)", f"{ml['vol']:.2%}")
    c6.metric("Sharpe (ML)", f"{ml['sharpe']:.2f}")
    c7, c8 = st.columns(2)
    c7.metric("Drawdown (Hist.)", f"{mtz['dd']:.2%}")
    c8.metric("Drawdown (ML)", f"{ml['dd']:.2%}")

def graf_ret_acumulado(curva_hist, curva_ml, curva_bench=None, bench_name=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curva_hist.index, y=(curva_hist-1)*100,
                             mode="lines", name="Carteira (Hist.)"))
    fig.add_trace(go.Scatter(x=curva_ml.index, y=(curva_ml-1)*100,
                             mode="lines", name="Carteira (ML)"))
    if curva_bench is not None:
        fig.add_trace(go.Scatter(x=curva_bench.index, y=(curva_bench-1)*100,
                                 mode="lines", name=bench_name or "Benchmark"))
    fig.update_yaxes(ticksuffix="%", title=None)
    fig.update_layout(title="Retorno Acumulado (%)", xaxis_title=None, height=420)
    st.plotly_chart(fig, use_container_width=True)

def graf_drawdown(curva_hist, curva_ml):
    dd_hist = curva_hist / curva_hist.cummax() - 1
    dd_ml = curva_ml / curva_ml.cummax() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd_hist.index, y=dd_hist, mode="lines", name="DD Hist."))
    fig.add_trace(go.Scatter(x=dd_ml.index, y=dd_ml, mode="lines", name="DD ML"))
    fig.update_yaxes(tickformat=".0%", title=None)
    fig.update_layout(title="Drawdown", xaxis_title=None, height=320)
    st.plotly_chart(fig, use_container_width=True)

def fronteira_com_pontos(mu, cov, pesos_hist, pesos_ml, titulo="Fronteira (amostragem)"):
    n_port = 800
    W = amostrar_pesos(len(mu), wmax=0.3, n_amostras=n_port)
    mu_vec = mu.values
    cov_mat = cov.values
    vols = np.sqrt(np.einsum('ij,jk,ik->i', W, cov_mat, W))
    rets = W @ mu_vec
    fig = px.scatter(x=vols, y=rets, labels={"x": "Volatilidade", "y": "Retorno"}, title=titulo)
    # pontos das carteiras
    ret_h = float((pesos_hist * mu).sum())
    vol_h = float(np.sqrt(pesos_hist.values.T @ cov_mat @ pesos_hist.values))
    ret_m = float((pesos_ml * mu).sum())  # usa mu histórico para posicionar
    vol_m = float(np.sqrt(pesos_ml.values.T @ cov_mat @ pesos_ml.values))
    fig.add_trace(go.Scatter(x=[vol_h], y=[ret_h], mode="markers",
                             marker=dict(color="red", size=10), name="Carteira Hist."))
    fig.add_trace(go.Scatter(x=[vol_m], y=[ret_m], mode="markers",
                             marker=dict(color="green", size=10), name="Carteira ML"))
    st.plotly_chart(fig, use_container_width=True)

def heatmap_corr(precos):
    corr = precos.pct_change().dropna().corr()
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(corr, ax=ax, annot=True, cmap="RdBu", center=0)
    st.pyplot(fig, use_container_width=True)

def tabela_ativos(precos, pesos):
    ultimo = precos.ffill().iloc[-1]
    investimento = pesos * 1000.0
    unidades = investimento / ultimo
    var_1d = precos.pct_change().iloc[-1]
    var_7d = precos.pct_change(7).iloc[-1]
    var_1m = precos.pct_change(21).iloc[-1]
    tab = pd.DataFrame({
        "Ticker": pesos.index,
        "Peso (%))": (pesos * 100).round(2),
        "Preço": ultimo.round(2),
        "Unidades": unidades.round(2),
        "Valor (€)": investimento.round(2),
        "24h (%)": (var_1d * 100).round(2),
        "7d (%)": (var_7d * 100).round(2),
        "1m (%)": (var_1m * 100).round(2),
    })
    st.dataframe(tab, use_container_width=True)

# ===========================================================
# 6) SIDEBAR / CONTROLES
# ===========================================================
st.title("📊 Otimizador de Carteira — Markowitz vs ML")

universo = carregar_universo()

st.sidebar.header("Configurações")
anos = st.sidebar.slider("Horizonte (anos)", 1, 20, 5)
frequencia = st.sidebar.selectbox("Frequência", ["1d", "1wk", "1mo"])
rf = st.sidebar.number_input("Taxa livre de risco (a.a.)", min_value=0.0, max_value=0.2, value=0.0, step=0.005, format="%.3f")
wmax = st.sidebar.slider("Limite por ativo", 0.05, 1.0, 0.3, step=0.05)
n_amostras = st.sidebar.slider("Amostras Fronteira/Busca", 500, 20000, 5000, step=500)

# Filtro obrigatório: Tipo de Ativo
tipos = sorted(universo.get("Tipo de Ativo", pd.Series()).dropna().unique().tolist()) if not universo.empty else []
tipo_escolhido = st.sidebar.selectbox("Tipo de Ativo", tipos) if tipos else None
dados_filtrados = universo[universo["Tipo de Ativo"] == tipo_escolhido] if tipo_escolhido else pd.DataFrame()

# Filtros opcionais
for coluna in ["País", "Setor", "Indústria"]:
    if not dados_filtrados.empty and coluna in dados_filtrados.columns:
        opcoes = sorted([x for x in dados_filtrados[coluna].dropna().unique()])
        if len(opcoes) > 1:
            escolha = st.sidebar.selectbox(coluna, ["Todos"] + opcoes)
            if escolha != "Todos":
                dados_filtrados = dados_filtrados[dados_filtrados[coluna] == escolha]

# Multiselect: Nome Curto → Ticker
nomes_para_tickers = dados_filtrados.set_index("Nome Curto")["Ticker"].dropna().to_dict() if not dados_filtrados.empty else {}
selecionados_nomes = st.sidebar.multiselect("Ativos", list(nomes_para_tickers.keys()))
selecionados = [nomes_para_tickers[n] for n in selecionados_nomes]

# Benchmark (INDEX)
benchmarks_df = universo[universo["Tipo de Ativo"] == "INDEX"] if not universo.empty else pd.DataFrame()
benchmarks_op = benchmarks_df.set_index("Nome Curto")["Ticker"].dropna().to_dict() if not benchmarks_df.empty else {}
bench_name = st.sidebar.selectbox("Benchmark", ["Nenhum"] + list(benchmarks_op.keys()))
benchmark_ticker = benchmarks_op.get(bench_name) if bench_name != "Nenhum" else None

btn = st.sidebar.button("Otimizar & Comparar")

# ===========================================================
# 7) EXECUÇÃO
# ===========================================================
if btn and selecionados:
    ini = pd.Timestamp.today().normalize() - pd.DateOffset(years=anos)
    fim = pd.Timestamp.today().normalize()

    precos = obter_dados(selecionados, ini, fim, frequencia)
    precos = limpar_series_ruins(precos)

    if precos.shape[1] < 2:
        st.warning("Preciso de pelo menos 2 ativos com dados válidos.")
        st.stop()

    # µ e Σ históricos
    rets, mu_hist, cov_hist = stats_mu_cov(precos, tipo="log")

    # -------- Carteira HISTÓRICA (Markowitz tradicional, via busca) ----------
    pesos_hist, ret_h, vol_h, sharpe_h = melhor_sharpe(mu_hist, cov_hist, rf=rf, wmax=wmax, n_amostras=n_amostras)

    # -------- Carteira ML (µ previsto com janela 180d) ----------
    mu_ml, df_prev = prever_mu_ml(precos, janela=180)
    # alinhar ativos
    comuns = mu_hist.index.intersection(mu_ml.index)
    mu_ml = mu_ml.loc[comuns]
    mu_hist2 = mu_hist.loc[comuns]
    cov_hist2 = cov_hist.loc[comuns, comuns]
    precos2 = precos[comuns]

    # se Carteira Hist. tinha ativos fora de "comuns", reamostra com comuns
    pesos_hist = pesos_hist.reindex(comuns).fillna(0.0)
    if not np.isclose(pesos_hist.sum(), 1.0) or (pesos_hist < 0).any():
        # normaliza apenas para exibir métricas corretas
        s = pesos_hist.clip(lower=0)
        s = s / s.sum()
        pesos_hist = s

    pesos_ml, ret_m, vol_m, sharpe_m = melhor_sharpe(mu_ml, cov_hist2, rf=rf, wmax=wmax, n_amostras=n_amostras)

    # -------- Curvas & KPIs ----------
    curva_hist = curva_carteira(precos2, pesos_hist, tipo="log")
    curva_ml = curva_carteira(precos2, pesos_ml, tipo="log")

    kpi_hist = metricas(precos2, pesos_hist)
    kpi_ml = metricas(precos2, pesos_ml)
    st.subheader("🧮 KPIs — Markowitz (Histórico) vs Markowitz + ML")
    kpis_duas_carteiras(kpi_hist, kpi_ml)

    # -------- Benchmark ----------
    curva_bench = None
    if benchmark_ticker:
        bench = yf.download(benchmark_ticker, start=ini, end=fim, auto_adjust=True, progress=False)
        if not bench.empty and "Close" in bench.columns:
            b = (1 + bench["Close"].pct_change().dropna()).cumprod()
            # alinhar datas
            idx = curva_hist.index.intersection(b.index)
            curva_bench = b.loc[idx]
            curva_hist = curva_hist.loc[idx]
            curva_ml = curva_ml.loc[idx]

    st.subheader("📈 Retorno Acumulado (%)")
    graf_ret_acumulado(curva_hist, curva_ml, curva_bench, bench_name if benchmark_ticker else None)

    st.subheader("📉 Drawdown Comparado")
    graf_drawdown(curva_hist, curva_ml)

    st.subheader("🗺️ Fronteira Eficiente (amostragem) + Carteiras")
    fronteira_com_pontos(mu_hist2, cov_hist2, pesos_hist, pesos_ml)

    colA, colB = st.columns([2, 1])
    with colA:
        st.subheader("📋 Tabela por Ativo (Carteira ML)")
        tabela_ativos(precos2, pesos_ml)
    with colB:
        st.subheader("🔗 Correlação")
        heatmap_corr(precos2)

    st.subheader("🔮 Previsões de Retorno (ML) — janela 180 dias")
    if not df_prev.empty:
        st.dataframe(df_prev.style.format({
            "Previsão diária (%)": "{:.3f}",
            "Previsão anual (%)": "{:.2f}"
        }), use_container_width=True)
    else:
        st.info("Sem previsões ML suficientes (dados insuficientes para alguns ativos).")

    st.caption("Nota: otimização por amostragem aleatória com restrições simples (wmax). µ(ML) baseado em RF com lags e médias móveis nos últimos 180 dias.")

else:
    st.info("Selecione ativos no menu lateral e clique em **Otimizar & Comparar**.")
