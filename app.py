import streamlit as st
        


homepage = st.Page("Homepage/1_Homepage.py", title="Homepage")
calculadora = st.Page("Actuarial/2_calculadora.py", title="Actuarial calculator")
page1 = st.Page("Wallet/4_wallet.py", title="Wallet")
page2 = st.Page("Wallet/3_ativo.py", title="ativo")
page3 = st.Page("Actuarial/1_apage1.py", title="page1")

pg = st.navigation({
                    "Homepage": [homepage],
                    "Atuarial": [calculadora],
                    "Wallet": [page1, page2]
        })

pg.run()