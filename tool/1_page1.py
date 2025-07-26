import streamlit as st
from pyliferisk import Actuarial
from pyliferisk.mortalitytables import GKM95
import matplotlib.pyplot as plt

# Título da aplicação
st.set_page_config(page_title="Calculadora Atuarial de Aposentadoria", layout="centered")
st.title("🧮 Calculadora Atuarial de Reserva para Aposentadoria")

st.markdown("""
Esta aplicação estima o valor da **reserva necessária** para garantir uma **renda constante na aposentadoria**.
Os cálculos são feitos com base em dados atuariais da tábua de mortalidade **GKM95** e assumem uma taxa de juros real.
""")

st.header("📝 Parâmetros do Usuário")

col1, col2 = st.columns(2)
with col1:
    
    idade_atual = st.number_input("📌 Idade atual", min_value=18, max_value=100, value=35)
    renda_mensal = st.number_input("💶 Renda mensal desejada (€)", min_value=0.0, value=1000.0)

with col2:
    idade_aposentadoria = st.number_input("🎯 Idade de aposentadoria", min_value=idade_atual + 1, max_value=100, value=67)
    taxa_juros = st.number_input("📉 Taxa de juros anual (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100


# Cálculos
renda_anual = renda_mensal * 12
anos_ate_aposentadoria = idade_aposentadoria - idade_atual

mt = Actuarial(nt=GKM95, i=taxa_juros)
expectativa_vida = mt.ex[idade_aposentadoria]  # expectativa de vida futura aos 67 anos
anos_de_renda = int(round(expectativa_vida))

v = 1 / (1 + mt.i)
valores_por_ano = [renda_anual * (v ** (ano + 1)) for ano in range(anos_de_renda)]
valor_na_aposentadoria = sum(valores_por_ano)
desconto_total = v ** anos_ate_aposentadoria
valor_presente_hoje = valor_na_aposentadoria * desconto_total

# Resultados
st.subheader("📊 Resultados:")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Reserva HOJE (€)", value=f"{valor_presente_hoje:,.2f}")

with col2:
    st.metric(label="Reserva NA APOSENTADORIA (€)", value=f"{valor_na_aposentadoria:,.2f}")

with col3:
    st.metric(label="Expectativa de vida aos 67", value=f"{expectativa_vida:.1f} anos")


# Gráfico da reserva acumulada ao longo do tempo
idades_acumuladas = list(range(idade_atual, idade_aposentadoria + 1))
valores_acumulados_ano_a_ano = [
    valor_na_aposentadoria / ((1 + taxa_juros) ** (anos_ate_aposentadoria - t))
    for t in range(anos_ate_aposentadoria + 1)
]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(idades_acumuladas, valores_acumulados_ano_a_ano, marker='o', color='teal')
ax.set_title("Evolução da Reserva Acumulada Até a Aposentadoria")
ax.set_xlabel("Idade")
ax.set_ylabel("Reserva acumulada (€)")
ax.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig)


# Faixa de idades após aposentadoria
idades_apos = list(range(idade_aposentadoria, idade_aposentadoria + anos_de_renda))

# Capital acumulado com e sem juros, ano a ano após aposentadoria
capital_com_juros = [valor_na_aposentadoria]
capital_sem_juros = [valor_na_aposentadoria]

for _ in range(1, anos_de_renda):
    capital_com_juros.append(capital_com_juros[-1] * (1 + taxa_juros) - renda_anual)
    capital_sem_juros.append(capital_sem_juros[-1] - renda_anual)

# Gráfico de comparação
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(idades_apos, capital_com_juros, label="Com juros", marker='o')
ax3.plot(idades_apos, capital_sem_juros, label="Sem juros", linestyle='--', marker='o')
ax3.axhline(0, color='gray', linestyle=':', linewidth=1)
ax3.set_title("Evolução do Capital Pós-Aposentadoria")
ax3.set_xlabel("Idade")
ax3.set_ylabel("Capital acumulado (€)")
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend()
st.pyplot(fig3)