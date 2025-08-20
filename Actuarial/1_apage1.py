import streamlit as st

st.title("💼 Investment Wallet")

# Verificar se há resultados salvos da calculadora atuarial
if "actuarial_result" in st.session_state and "user_inputs" in st.session_state:
    res = st.session_state["actuarial_result"]
    user_inputs = st.session_state["user_inputs"]

    st.subheader("🔗 Dados Importados da Calculadora Atuarial")
    st.write(f"📍 Região: {user_inputs['regiao']}")
    st.write(f"👤 Idade atual: {user_inputs['idade_atual']}")
    st.write(f"🎯 Idade aposentadoria: {user_inputs['idade_apos']}")
    st.write(f"💶 Renda mensal desejada: {user_inputs['renda_mensal']:.2f} €")
    st.write(f"📉 Taxa de juros: {user_inputs['taxa_juros']*100:.2f}%")

    st.metric("Reserva necessária hoje (€)", f"{res['reserva_hoje']:,.2f}")
    st.metric("Reserva na aposentadoria (€)", f"{res['reserva_aposentadoria']:,.2f}")

else:
    st.warning("⚠️ Nenhuma simulação da Calculadora Atuarial foi encontrada. Vá até a página 'Calculadora Atuarial' e execute uma simulação.")
