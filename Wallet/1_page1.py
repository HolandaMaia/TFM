import streamlit as st
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
from datetime import datetime
import plotly.graph_objects as go
import plotly.graph_objects as go

# 📥 Carregar os dados
df = pd.read_excel("C:/Users/mathe/OneDrive/Área de Trabalho/MASTER/FLI/FMI/dados/ativos_totais.xlsx")

st.set_page_config(page_title="Carteira Ótima Markowitz", layout="wide")
st.title("📊 Otimizador de Carteira Global (Markowitz)")

# 📌 FILTROS
st.sidebar.header("🎯 Selecione os ativos")
tipo = st.sidebar.multiselect("Tipo de Ativo", df["Tipo de Ativo"].dropna().unique())
filtro = df[df["Tipo de Ativo"].isin(tipo)] if tipo else df

pais = st.sidebar.multiselect("País", filtro["País"].dropna().unique())
filtro = filtro[filtro["País"].isin(pais)] if pais else filtro

setor = st.sidebar.multiselect("Setor", filtro["Setor"].dropna().unique())
filtro = filtro[filtro["Setor"].isin(setor)] if setor else filtro

opcoes = filtro["Nome Curto"] + " (" + filtro["Ticker"] + ")"
selecionados = st.sidebar.multiselect("Ativos disponíveis", opcoes.tolist(), max_selections=8)

data_inicio = st.sidebar.date_input("Data inicial", datetime(2010, 1, 1))
data_fim = st.sidebar.date_input("Data final", datetime.today())

if selecionados:
    tickers = [s.split("(")[-1].replace(")", "").strip() for s in selecionados]

    try:
        st.info("⏳ Baixando dados de mercado...")
        dados_brutos = yf.download(
            tickers,
            start=data_inicio,
            end=data_fim,
            progress=False,
            auto_adjust=True
        )

        dados = dados_brutos["Close"] if "Close" in dados_brutos else dados_brutos
        dados = dados.dropna(axis=1)

        if dados.empty:
            st.error("❌ Nenhum dado de preço válido foi encontrado para os ativos selecionados.")
        else:
            st.success(f"✅ Dados baixados com sucesso para {len(dados.columns)} ativos.")

            mu = expected_returns.mean_historical_return(dados)
            S = risk_models.sample_cov(dados)
            ef = EfficientFrontier(mu, S)
            ef.max_sharpe()
            pesos_limpos = ef.clean_weights()

            st.subheader("💼 Pesos ótimos da carteira (Max Sharpe)")
            colunas = st.columns(len(pesos_limpos))
            for i, (ticker, peso) in enumerate(pesos_limpos.items()):
                with colunas[i]:
                    st.metric(label=ticker, value=f"{peso*100:.2f} %")

            st.subheader("📈 Indicadores de Performance")
            ret, vol, sharpe = ef.portfolio_performance()
            c1, c2, c3 = st.columns(3)
            c1.metric("Retorno Esperado", f"{ret*100:.2f}%")
            c2.metric("Volatilidade", f"{vol*100:.2f}%")
            c3.metric("Sharpe Ratio", f"{sharpe:.2f}")

            st.subheader("📊 Gráfico de Evolução dos Ativos")
            for ticker in dados.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dados.index, y=dados[ticker], mode='lines', name=ticker))
                fig.update_layout(title=f"Evolução: {ticker}", xaxis_title="Data", yaxis_title="Preço")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("📉 Evolução da Carteira Ponderada")
            dados_norm = dados / dados.iloc[0]
            pesos_series = pd.Series(pesos_limpos)
            carteira_valor = dados_norm.dot(pesos_series)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=carteira_valor.index, y=carteira_valor, mode='lines', name='Carteira'))
            fig.update_layout(title="Valor Normalizado da Carteira", xaxis_title="Data", yaxis_title="Valor")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao calcular a carteira: {e}")

else:
    st.warning("Selecione ao menos um ativo para montar a carteira.")