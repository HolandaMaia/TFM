# nav.py (simples e direto)
import streamlit as st

# Configuração básica
st.set_page_config(page_title="Retirement & Investment App", page_icon="💹", layout="wide")

# Páginas
home        = st.Page("Homepage/1_Homepage.py",     title="Homepage",             icon="🏠")
act_calc    = st.Page("Actuarial/2_calculadora.py", title="Actuarial Calculator", icon="🔢")
wallet_opt  = st.Page("Wallet/4_wallet.py",         title="Portfolio Optimizer",  icon="📊")
stock_an    = st.Page("Wallet/3_ativo.py",          title="Stock Analysis",       icon="📈")

# Menu enxuto (grupos curtos)
pg = st.navigation({
    "Start":    [home],
    "Plan":     [act_calc],
    "Invest":   [wallet_opt],
    "Research": [stock_an],
})

# Sidebar minimalista com o fluxo sugerido
with st.sidebar:
    st.markdown("**Suggested flow**")
    st.caption("1) Set retirement target\n2) Optimize portfolio\n3) Analyze stocks")

pg.run()
