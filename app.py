import streamlit as st

print("oi")


homepage = st.Page("Homepage/1_Homepage.py", title="Homepage")
calculadora = st.Page("Actuarial/2_calculadora.py", title="Actuarial calculator")
page2 = st.Page("Wallet/2_page2.py", title="Stock")
page1 = st.Page("Wallet/4_wallet.py", title="Wallet")
page3 = st.Page("Wallet/3_page3_test.py", title="ativo")

pg = st.navigation({
                    "Homepage": [homepage],
                    "Atuarial": [calculadora],
                    "Wallet": [page1, page2, page3]
        })

pg.run()