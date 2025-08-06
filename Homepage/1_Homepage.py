import streamlit as st

# Exibe apenas uma vez no início
st.set_page_config(page_title="Meu App", layout="wide")

# Redirecionamento simples (simulando homepage)
st.title("🏠 Bem-vindo à Homepage")

if st.button("Ir para Calculadora Atuarial"):
    st.switch_page("Actuarial/2_calculadora.py")
elif st.button("Ir para Wallet"):
    st.switch_page("Wallet/4_page4_test2.py")
