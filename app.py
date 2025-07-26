import streamlit as st

print("oi")

st.title("Welcome to My Streamlit App")
st.write("This is a simple Streamlit application.")


bugs = st.Page("report/1_apage1.py", title="Bug reports")
page1 = st.Page("tool/1_page1.py", title="Page 1")
page2 = st.Page("tool/2_page2.py", title="Page 2")

pg = st.navigation({            
                    "Reports": [bugs],
                    "Pages": [page1, page2]
        })

pg.run()