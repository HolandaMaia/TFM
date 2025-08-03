import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Função para carregar e limpar a tábua ---
def carregar_tabela_mortalidade(caminho_arquivo: str, aba: str = "Estimates 2016-2023") -> pd.DataFrame:
    df_raw = pd.read_excel(
        caminho_arquivo,
        sheet_name=aba,
        skiprows=16
    )

    df_raw.columns = [col.strip() for col in df_raw.columns]
    df = df_raw.rename(columns={
        'Region, subregion, country or area *': 'local',
        'Year': 'ano',
        'Age (x)': 'idade',
        'Number of survivors l(x)': 'lx',
        'Expectation of life e(x)': 'ex',
        'Average number of years lived a(x,n)': 'a_xn'
    })

    df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
    df['ano'] = pd.to_numeric(df['ano'], errors='coerce')
    df['lx'] = df['lx'].astype(str).str.replace(" ", "").str.replace(",", ".")
    df['lx'] = pd.to_numeric(df['lx'], errors='coerce')
    df['ex'] = pd.to_numeric(df['ex'].astype(str).str.replace(",", "."), errors='coerce')
    df['local'] = df['local'].astype(str).str.strip()

    df = df.dropna(subset=['idade', 'lx', 'ex', 'ano'])
    return df

# --- Função de cálculo ---
def calcular_reserva(df_mortalidade, local, idade_atual, idade_apos, renda_mensal, taxa_juros):
    renda_anual = renda_mensal * 12
    anos_ate_aposentadoria = idade_apos - idade_atual
    v = 1 / (1 + taxa_juros)

    tabua = df_mortalidade[df_mortalidade['local'] == local].sort_values(by='idade')
    try:
        expectativa_vida = tabua[tabua['idade'] == idade_apos]['ex'].values[0]
    except IndexError:
        return None

    anos_de_renda = int(round(expectativa_vida))
    valores_futuros = [renda_anual * (v ** (t + 1)) for t in range(anos_de_renda)]
    reserva_aposentadoria = sum(valores_futuros)
    valor_presente_hoje = reserva_aposentadoria * (v ** anos_ate_aposentadoria)

    return {
        "renda_anual": renda_anual,
        "expectativa_vida": expectativa_vida,
        "anos_de_renda": anos_de_renda,
        "reserva_aposentadoria": reserva_aposentadoria,
        "reserva_hoje": valor_presente_hoje
    }

# --- Função de visualização ---
def mostrar_resultado(res, idade_atual, idade_apos, taxa_juros):
    st.markdown("### 📌 Estatísticas da Região")
    col4, col5, col6 = st.columns(3)
    idade_mais_sobreviventes = df_regiao.loc[df_regiao['idade'] != 0, :].sort_values(by='lx', ascending=False).iloc[0]['idade']
    col4.metric("Idade com mais sobreviventes", f"{int(idade_mais_sobreviventes)}")
    col5.metric("Expectativa de vida ao nascer", f"{df_regiao.loc[df_regiao['idade'] == 0, 'ex'].values[0]:.1f} anos")
    prob_sobrevivencia_65 = df_regiao.loc[df_regiao['idade'] == 65, 'lx'].values[0] / df_regiao.loc[df_regiao['idade'] == 0, 'lx'].values[0] * 100
    col6.metric("Sobrevivência aos 65 anos", f"{prob_sobrevivencia_65:.1f}%")

    st.subheader(f"📊 Simulação com Gráficos - {regiao}")
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        st.markdown("**📈 Expectativa de Vida - Linha**")
        df_plot = df_regiao[['idade', 'ex']].set_index('idade')
        st.line_chart(df_plot)

    with col_g2:
        st.markdown("**🏛️ Pirâmide Etária de Sobrevivência**")
        df_piramide = df_regiao[['idade', 'lx']].copy()
        df_piramide['lx'] = df_piramide['lx'] / 1000
        st.bar_chart(df_piramide.set_index('idade'))

    st.markdown("### 🧮 Resultados da Simulação")
    col1, col2, col3 = st.columns(3)
    col1.metric("Reserva HOJE (€)", f"{res['reserva_hoje']:,.2f}")
    col2.metric("Reserva NA APOSENTADORIA (€)", f"{res['reserva_aposentadoria']:,.2f}")
    col3.metric("Expectativa de vida aos {idade_apos}", f"{res['expectativa_vida']:.1f} anos")

    st.markdown("### 📉 Evolução da Reserva Até a Aposentadoria")
    anos_ate_apos = idade_apos - idade_atual
    idades = list(range(idade_atual, idade_apos + 1))
    valores_ano_a_ano = [
        res['reserva_aposentadoria'] / ((1 + taxa_juros) ** (anos_ate_apos - t))
        for t in range(anos_ate_apos + 1)
    ]
    df_reserva = pd.DataFrame({"Idade": idades, "Reserva acumulada (€)": valores_ano_a_ano}).set_index("Idade")
    st.line_chart(df_reserva)

    st.markdown("### 📉 Evolução do Capital Pós-Aposentadoria")
    idades_apos = list(range(idade_apos, idade_apos + res['anos_de_renda']))
    capital_com_juros = [res['reserva_aposentadoria']]
    capital_sem_juros = [res['reserva_aposentadoria']]
    renda_anual = res['renda_anual']

    for _ in range(1, res['anos_de_renda']):
        capital_com_juros.append(capital_com_juros[-1] * (1 + taxa_juros) - renda_anual)
        capital_sem_juros.append(capital_sem_juros[-1] - renda_anual)

    df_apos = pd.DataFrame({
        "Idade": idades_apos,
        "Com Juros (€)": capital_com_juros,
        "Sem Juros (€)": capital_sem_juros
    }).set_index("Idade")
    st.line_chart(df_apos)

# --- Interface principal ---
st.set_page_config(page_title="Calculadora Atuarial", layout="wide")
st.title("🧮 Calculadora Atuarial de Reserva para Aposentadoria")

@st.cache_resource
def carregar_dados():
    caminho = r"C:\\Users\\mathe\\OneDrive\\Área de Trabalho\\MASTER\\FLI\\FMI\\dados\\WPP2024_MORT_F06_1_SINGLE_AGE_LIFE_TABLE_ESTIMATES_BOTH_SEXES.xlsx"
    return carregar_tabela_mortalidade(caminho)

df_mortalidade = carregar_dados()

st.sidebar.header("Parâmetros do Usuário")

locais_disponiveis = df_mortalidade['local'].dropna().unique()
regiao = st.sidebar.selectbox("Região", sorted(locais_disponiveis))
idade_atual = st.sidebar.number_input("Idade atual", min_value=18, max_value=100, value=35)
renda_mensal = st.sidebar.number_input("Renda mensal desejada (€)", min_value=0.0, value=1000.0)
idade_apos = st.sidebar.number_input("Idade de aposentadoria", min_value=idade_atual + 1, max_value=100, value=67)
taxa_juros = st.sidebar.number_input("Taxa de juros anual (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100

if st.sidebar.button("Calcular Reserva"):
    res = calcular_reserva(df_mortalidade, regiao, idade_atual, idade_apos, renda_mensal, taxa_juros)
    df_regiao = df_mortalidade[df_mortalidade['local'] == regiao].sort_values(by='idade')
    if res:
        mostrar_resultado(res, idade_atual, idade_apos, taxa_juros)
    else:
        st.warning("Não foi possível encontrar a idade na tábua de mortalidade.")