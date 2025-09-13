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
    df['tipo'] = df['tipo'].astype(str).str.strip()
    
    df = df.dropna(subset=['idade', 'lx', 'ex', 'ano', 'local', 'tipo'])
    df = df[df['ano'] == 2023]
    df = df.sort_values(['local', 'idade']).reset_index(drop=True)
    return df

# --- Função de cálculo ---
def calcular_reserva(df_mortalidade, local, idade_atual, idade_apos, renda_mensal, taxa_juros):
    renda_anual = renda_mensal * 12
    anos_ate_aposentadoria = idade_apos - idade_atual
    v = 1 / (1 + taxa_juros)

    tabua = df_mortalidade[df_mortalidade['local'] == local].sort_values(by='idade')
    if not (tabua['idade'] == idade_apos).any():
        return None
    
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
def mostrar_resultado(res, idade_atual, idade_apos, taxa_juros, regiao, df_regiao):
    st.subheader(f"🌍 Region Statistics - {regiao}", divider='blue')
    st.caption(
        "This section provides insights into the selected region's life expectancy and survival probabilities. "
        "Use this information to better understand the demographic context of your simulation."
    )

    col4, col5, col6 = st.columns(3)

    # Age where age and life expectancy intersect
    df_regiao['diff'] = (df_regiao['idade'] - df_regiao['ex']).abs()
    idade_ex_proximos = df_regiao.loc[df_regiao['diff'].idxmin(), 'idade']
    col4.metric(
        "Age ≈ Expected Remaining Years", 
        f"{int(idade_ex_proximos)}",
        help="This is the age where the remaining life expectancy is approximately equal to the current age."
    )

    # Life expectancy at birth
    if (df_regiao['idade'] == 0).any():
        le_0 = df_regiao.loc[df_regiao['idade'] == 0, 'ex'].values[0]
    else:
        le_0 = df_regiao['ex'].iloc[0]

    col5.metric(
        "Life Expectancy at Birth", 
        f"{le_0:.1f} years",
        help="This represents the average number of years a newborn is expected to live in this region."
    )

    # Probability of surviving until retirement age
    if (df_regiao['idade'] == idade_apos).any() and (df_regiao['idade'] == idade_atual).any():
        prob_sobrevivencia_apos = (
            (df_regiao.loc[df_regiao['idade'] == idade_apos, 'lx'].values[0] /
             df_regiao.loc[df_regiao['idade'] == idade_atual, 'lx'].values[0]) * 100
        )
    else:
        prob_sobrevivencia_apos = float('nan')

    col6.metric(
        f"Survival Probability ({idade_atual} to {idade_apos} years)",
        f"{prob_sobrevivencia_apos:.1f}%",
        help="This is the probability of surviving from your current age to your planned retirement age."
    )

    st.subheader(f"📊 Simulation with Charts - {regiao}")
    col_g1, col_g2 = st.columns([2, 1])
    
    with col_g1:
        st.subheader("📈 Life Expectancy", help="This chart shows the expected remaining years of life for each age in the selected region. The x-axis represents the age, and the y-axis shows the remaining years of life expected for that age. It helps you understand how life expectancy changes with age and plan accordingly.")
        df_plot = df_regiao[['idade', 'ex']].rename(columns={
            'idade': 'Age',
            'ex': 'Life Expectancy'
        })
        fig1 = px.line(
            df_plot,
            x="Age",
            y="Life Expectancy",
            labels={"Age": "Age", "Life Expectancy": "Expected Remaining Years"},
            title="Life Expectancy by Age"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        st.subheader("🏛️ Age Pyramid of Survival", help="This bar chart represents the number of survivors at each age in the selected region. The x-axis represents the age, and the y-axis shows the number of survivors (in thousands). It provides a visual representation of the population's survival distribution, helping you assess demographic trends.")
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
            labels={"Age": "Age", "Survivors (thousands)": "Survivors (thousands)"},
            title="Age Pyramid of Survival"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🧮 Simulation Results", divider='blue')
    st.caption(
        "This section provides a summary of the simulation results, including the current reserve, retirement reserve, and life expectancy at the planned retirement age. "
        "Use this information to evaluate your financial planning and retirement goals."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Current Reserve", 
        f"{res['reserva_hoje']:,.2f}",
        help="This is the amount of reserve you currently have accumulated."
    )
    col2.metric(
        "Retirement Reserve", 
        f"{res['reserva_aposentadoria']:,.2f}",
        help="This is the total reserve required to sustain your desired retirement income."
    )
    col3.metric(
        f"Life Expectancy at {idade_apos}", 
        f"{res['expectativa_vida']:.1f} years",
        help="This represents the expected number of years you will live after reaching your planned retirement age."
    )

    st.subheader("📉 Evolution of Reserve Until Retirement", help="This chart illustrates how your reserve is expected to grow annually until your planned retirement age. It helps you visualize the accumulation of your financial resources over time.")
    anos_ate_apos = idade_apos - idade_atual
    idades = list(range(idade_atual, idade_apos + 1))
    valores_ano_a_ano = [
        res['reserva_aposentadoria'] / ((1 + taxa_juros) ** (anos_ate_apos - t))
        for t in range(anos_ate_apos + 1)
    ]
    df_reserva = pd.DataFrame({"Age": idades, "Accumulated Reserve": valores_ano_a_ano})
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

    st.subheader("📉 Evolution of Capital After Retirement", help="This chart shows the evolution of your capital after retirement, with and without interest. It helps you understand how your reserve will deplete over time based on your annual income needs.")
    idades_apos = list(range(idade_apos, idade_apos + res['anos_de_renda']))
    capital_com_juros = [res['reserva_aposentadoria']]
    capital_sem_juros = [res['reserva_aposentadoria']]
    renda_anual = res['renda_anual']

    if res['anos_de_renda'] > 0:
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

@st.cache_data(show_spinner=False)
def carregar_dados():
    caminho = "https://raw.githubusercontent.com/HolandaMaia/TFM/master/dados/WPP2024_MORT_F06_1_SINGLE_AGE_LIFE_TABLE_ESTIMATES_BOTH_SEXES.xlsx"
    return carregar_tabela_mortalidade(caminho)

df_mortalidade = carregar_dados()

# --- Melhorias no menu para UX ---
st.sidebar.header("User Parameters")
st.sidebar.markdown("Configure the parameters below to simulate your retirement reserve.")

# Section: Location
tipos_locais_disponiveis = df_mortalidade['tipo'].dropna().unique()
st.sidebar.subheader("🌍 Location")
st.sidebar.markdown("Select the type of location and region for the simulation.")
tipo_local = st.sidebar.selectbox("Location Type", sorted(tipos_locais_disponiveis), help="The type of location (e.g., country, region) affects the mortality data used in the simulation.")
locais_filtrados = df_mortalidade[df_mortalidade['tipo'] == tipo_local]['local'].dropna().unique()
regiao = st.sidebar.selectbox("Region", sorted(locais_filtrados), help="The selected region determines the mortality table used.")

# Section: Age
st.sidebar.subheader("📅 Age")
st.sidebar.markdown("Enter your current age and planned retirement age.")
idade_atual = st.sidebar.number_input(
    "Current Age", 
    min_value=18, 
    max_value=100, 
    value=35, 
    help="Your current age is used to calculate the time until retirement."
)
idade_apos = st.sidebar.number_input(
    "Retirement Age", 
    min_value=idade_atual + 1, 
    max_value=100, 
    value=67, 
    help="The retirement age affects the accumulation period and the payout period."
)

# Section: Financial Parameters
st.sidebar.subheader("💰 Financial Parameters")
st.sidebar.markdown("Define the financial values for the simulation.")
renda_mensal = st.sidebar.number_input(
    "Desired Monthly Income at Retirement", 
    min_value=0.0, 
    value=1000.0, 
    help="The desired monthly income is used to calculate the total amount needed for retirement."
)
taxa_juros = st.sidebar.number_input(
    "Annual Interest Rate (%)", 
    min_value=0.0, 
    max_value=10.0, 
    value=1.0, 
    step=0.1, 
    help="The annual interest rate affects the growth of the accumulated capital."
) / 100

# Calculate button
if st.sidebar.button("Calculate Reserve", type="primary", use_container_width=True):
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

        mostrar_resultado(res, idade_atual, idade_apos, taxa_juros, regiao, df_regiao)
    else:
        st.warning("Could not find the age in the mortality table.")