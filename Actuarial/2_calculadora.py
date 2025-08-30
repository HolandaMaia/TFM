import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

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
        'Type': 'tipo',
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

    df = df.dropna(subset=['idade', 'lx', 'ex', 'ano', 'local', 'tipo'])
    df = df[df['ano'] == ano_alvo]
    df = df.sort_values(['local', 'idade']).reset_index(drop=True)
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
    st.markdown(f"### 📌 Region Statistics - {regiao}")
    col4, col5, col6 = st.columns(3)

    # idade onde idade e expectativa de vida se cruzam
    df_regiao['diff'] = (df_regiao['idade'] - df_regiao['ex']).abs()
    idade_ex_proximos = df_regiao.loc[df_regiao['diff'].idxmin(), 'idade']
    col4.metric("Age ≈ Expected Remaining Years", f"{int(idade_ex_proximos)}")

    col5.metric("Life Expectancy", f"{df_regiao.loc[df_regiao['idade'] == 0, 'ex'].values[0]:.1f} years")

    # Calcular probabilidade de sobreviver até a idade da aposentadoria
    prob_sobrevivencia_apos = (
        (df_regiao.loc[df_regiao['idade'] == idade_apos, 'lx'].values[0] /
        df_regiao.loc[df_regiao['idade'] == idade_atual, 'lx'].values[0]) * 100
    )

    col6.metric(
    f"Probability of surviving from {idade_atual} to {idade_apos} years",
        f"{prob_sobrevivencia_apos:.1f}%"
    )

    st.subheader(f"📊 Simulation with Charts - {regiao}")
    col_g1, col_g2 = st.columns([2, 1])
    
    with col_g1:
        st.markdown("**📈 Life Expectancy**")
        df_plot = df_regiao[['idade', 'ex']].rename(columns={
            'idade': 'Age',
            'ex': 'Life Expectancy'
        })
        fig1 = px.line(
        df_plot,
        x="Age",
        y="Life Expectancy",
        labels={"Age": "Age", "Life Expectancy": "Expected Remaining Years"}
    )
        st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        st.markdown("**🏛️ Age Pyramid of Survival**")
        df_piramide = df_regiao[['idade', 'lx']].rename(columns={
        'idade': 'Age',
        'lx': 'Survivors'
        })
        df_piramide['Survivors (thousands)'] = df_piramide['Survivors'] / 1000
        fig2 = px.bar(
        df_piramide,
        x="Age",
        y="Survivors (thousands)",
        orientation="v",
        labels={"Age": "Age", "Survivors (thousands)": "Survivors (thousands)"}
    )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🧮 Simulation Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Reserve", f"{res['reserva_hoje']:,.2f}")
    col2.metric("Retirement Reserve", f"{res['reserva_aposentadoria']:,.2f}")
    col3.metric(f"Life Expectancy at {idade_apos}", f"{res['expectativa_vida']:.1f} years")

    st.markdown("### 📉 Evolution of Reserve Until Retirement")
    anos_ate_apos = idade_apos - idade_atual
    idades = list(range(idade_atual, idade_apos + 1))
    valores_ano_a_ano = [
        res['reserva_aposentadoria'] / ((1 + taxa_juros) ** (anos_ate_apos - t))
        for t in range(anos_ate_apos + 1)
    ]
    df_reserva = pd.DataFrame({"Age": idades, "Accumulated Reserve": valores_ano_a_ano})
    # Gráfico com labels em inglês
    fig_reserva = px.bar(
        df_reserva,
        x="Age",
        y="Accumulated Reserve",
        labels={
            "Age": "Age (years)",
            "Accumulated Reserve": "Reserve"
        },
        title="Evolution of Reserve Until Retirement"
    )
    fig_reserva.update_yaxes(separatethousands=True)
    st.plotly_chart(fig_reserva, use_container_width=True)


    st.markdown("### 📉 Evolution of Capital After Retirement")
    idades_apos = list(range(idade_apos, idade_apos + res['anos_de_renda']))
    capital_com_juros = [res['reserva_aposentadoria']]
    capital_sem_juros = [res['reserva_aposentadoria']]
    renda_anual = res['renda_anual']

    for _ in range(1, res['anos_de_renda']):
        capital_com_juros.append(capital_com_juros[-1] * (1 + taxa_juros) - renda_anual)
        capital_sem_juros.append(capital_sem_juros[-1] - renda_anual)

    df_apos = pd.DataFrame({
        "Idade": idades_apos,
        "with Interest": capital_com_juros,
        "Without Interest": capital_sem_juros
    })
    fig_apos = px.line(
        df_apos,
        x="Idade",
        y=["with Interest", "Without Interest"],
        labels={
            "Idade": "Age",
            "value": "Capital",
            "variable": "Scenario"            
        },
        title="Evolution of Capital After Retirement"
    )
    fig_apos.update_yaxes(separatethousands=True)
    st.plotly_chart(fig_apos, use_container_width=True)

# --- Interface principal ---
st.set_page_config(page_title="Calculadora Atuarial", layout="wide")
st.title("🧮 Actuarial Retirement Reserve Calculator")

@st.cache_resource
def carregar_dados():
    caminho = r"C:\\Users\\mathe\\OneDrive\\Área de Trabalho\\MASTER\\FLI\\FMI\\dados\\WPP2024_MORT_F06_1_SINGLE_AGE_LIFE_TABLE_ESTIMATES_BOTH_SEXES.xlsx"
    return carregar_tabela_mortalidade(caminho)

df_mortalidade = carregar_dados()

st.sidebar.header("User Parameters")

tipos_locais_disponiveis = df_mortalidade['Type'].dropna().unique()
tipo_local = st.sidebar.selectbox("Local Type", sorted(tipos_locais_disponiveis))
locais_filtrados = df_mortalidade[df_mortalidade['Type'] == tipo_local]['local'].dropna().unique()
regiao = st.sidebar.selectbox("Region", sorted(locais_filtrados))
idade_atual = st.sidebar.number_input("Current Age", min_value=18, max_value=100, value=35)
idade_apos = st.sidebar.number_input("Retirement Age", min_value=idade_atual + 1, max_value=100, value=67)
renda_mensal = st.sidebar.number_input("Desired Monthly Income (€)", min_value=0.0, value=1000.0)
taxa_juros = st.sidebar.number_input("Annual Interest Rate (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100

if st.sidebar.button("Calculate Reserve"):
    res = calcular_reserva(df_mortalidade, regiao, idade_atual, idade_apos, renda_mensal, taxa_juros)
    df_regiao = df_mortalidade[df_mortalidade['local'] == regiao].sort_values(by='idade')
    if res:
        # Salvar no session_state para usar em outras páginas
        st.session_state["actuarial_result"] = res
        st.session_state["user_inputs"] = {
            "regiao": regiao,
            "idade_atual": idade_atual,
            "idade_apos": idade_apos,
            "renda_mensal": renda_mensal,
            "taxa_juros": taxa_juros,
        }

        mostrar_resultado(res, idade_atual, idade_apos, taxa_juros)
    else:
        st.warning("Could not find the age in the mortality table.")