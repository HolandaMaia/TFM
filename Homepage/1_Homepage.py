import streamlit as st

# Page config
st.set_page_config(page_title="Home | Retirement & Investment App", layout="wide")

# Title and introduction
st.title("🏠 Welcome to the Retirement & Investment Dashboard")
st.markdown("""
This application combines **actuarial modeling**, **portfolio optimization**, and **AI-driven stock analysis** to help you plan for retirement and make smarter investment decisions.

Below you can learn about each tool available in the platform:
""")

# --- ACTUARIAL CALCULATOR SECTION ---
with st.expander("🧮 Actuarial Retirement Calculator"):
    st.markdown("""
    This tool helps you answer a very important question:

> **How much money do I need to save by the time I retire, so I can receive a fixed annual income for the rest of my life?**

To calculate that, we consider:

- Your **current age**
- The age at which you plan to **retire**
- How much money you want to receive **every year after retirement**
- A small **interest rate** (to account for investment returns)

---

### 👵 Life Expectancy & Mortality Tables

We use **mortality tables** provided by the United Nations (UN), which tell us the **probability of a person surviving to each future age**, based on their current age.

🔗 You can check the source here:  
[UN Mortality Projections – World Population Prospects](https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Mortality)

In this app, we use the **combined table for both sexes**.

---

### 📐 How the calculation works

We use a formula that estimates how much money you need **today** to guarantee a fixed income after retirement, taking into account:

- the **interest rate** applied to savings
- your **life expectancy**
- and the probability that you will still be alive each year to receive the income.

#### The formula:
\\[
PV = \\sum_{t=1}^{n} \\frac{R}{(1+i)^t} \\cdot {}_tp_x
\\]

Where:
- \\( PV \\): Present Value – the total amount you need to save  
- \\( R \\): Annual income you want to receive during retirement  
- \\( i \\): Interest rate (e.g. 1% = 0.01)  
- \\( {}_tp_x \\): Probability of being alive at age \\( x + t \\)  
- \\( n \\): Number of years after retirement you're expected to live

---

### 💡 Why this is useful

This calculator gives you a **realistic estimate** of how much capital you’ll need to accumulate before retiring, based on international life expectancy data.

It’s a simple yet powerful way to plan your financial future with confidence.

""")
if st.button("🔢 Go to Actuarial Calculator"):
    st.switch_page("Actuarial/2_calculadora.py")

# --- WALLET OPTIMIZER SECTION ---
with st.expander("💼 Portfolio Optimizer"):
    st.markdown("""
    This tool helps you answer another important question:

    > **How can I invest my money in a smart and balanced way to grow it over time without taking too much risk?**

    The goal of this tool is to help you **build a diversified investment portfolio**. That means distributing your money across different assets (such as stocks, ETFs, or funds) in a way that **maximizes your expected return** while **minimizing risk**.

    ---

    ### 📊 What does this optimizer do?

    This module uses two techniques to build your portfolio:

    - **Modern Portfolio Theory (Markowitz)**  
      A classic mathematical method that finds the best combination of investments to get the highest possible return for a given level of risk.

    - **Machine Learning Predictions**  
      A more advanced technique that uses historical data to try to **predict future returns** for each asset and make better decisions based on those predictions.

    The user can choose which approach to use, or even compare both.

    ---

    ### 📐 How does it work?

    The optimizer calculates the **best percentage (weight)** of your money to invest in each asset based on:

    - the asset's **past performance**
    - how much the asset tends to **fluctuate (volatility)**
    - how assets are **related** to each other (some go up while others go down)
    - and your selected strategy: traditional or AI-based

    In the case of the Markowitz method, we use a formula that maximizes the **Sharpe Ratio** – a measure that tells you **how much return you're getting for each unit of risk**.

    #### The formula:
    \\[
    \\max_w \\frac{w^T \\mu - r_f}{\\sqrt{w^T \\Sigma w}}
    \\]

    Where:
    - \\( w \\): Weight (percentage) of your money allocated to each asset  
    - \\( \\mu \\): Expected return of each asset  
    - \\( \\Sigma \\): Risk between assets (covariance matrix)  
    - \\( r_f \\): Return from a risk-free asset (like government bonds)  
    - The result is the **Sharpe Ratio**, which we try to maximize

    This way, you can invest more confidently — not just based on intuition or emotion, but on solid math and data.

    ---

    ### 🌍 Global diversification

    The optimizer includes **assets from different countries and sectors**, allowing you to build a portfolio that is not limited to one region.  
    This improves stability by **spreading your risk across the world**.

    ---

    ### 💡 Why this is useful

    This optimizer gives you a **personalized and data-driven investment strategy**.

    Whether you're a beginner or an experienced investor, it helps you:

    - Decide **how much to invest in each asset**
    - See the **expected return and risk** of your portfolio
    - Explore **different combinations** and approaches (with or without AI)
    - Take control of your financial future using clear numbers and visualizations

""")
if st.button("📊 Go to Portfolio Optimizer"):
    st.switch_page("Wallet/4_wallet.py")

# --- STOCK ANALYSIS SECTION ---
with st.expander("📈 Stock Analysis & Forecasting"):
    st.markdown("""
   This tool helps you answer a practical question:

    > **What might happen to a stock’s price in the near future?**

    We don’t try to guess randomly. Instead, we use **Machine Learning (ML)** — a type of artificial intelligence that learns from past data to identify patterns and make predictions.

    ---

    ### 🧠 What is Machine Learning (ML)?

    Machine Learning is like training a smart assistant:  
    You show it lots of examples (in this case, **historical stock prices**) and it learns how prices usually move over time.

    Once trained, it can **predict the next price movements**, helping investors make more informed decisions.

    ---

    ### ⚙️ How this forecasting works

    In this module, we use a model called **Random Forest Regressor**, which is very good at handling noisy and complex data — like the stock market.

    Here's how it works step by step:

    1. We collect historical data for each stock.
    2. We calculate features such as:
        - daily return  
        - 5-day and 21-day average return  
        - short-term volatility  
    3. We train the machine learning model using this data.
    4. The model then **predicts the next return or price** of the stock.

    ---

    ### 🔎 Why is this useful?

    With this tool, you can:

    - Analyze the **expected short-term performance** of each stock  
    - Compare different stocks based on **AI predictions**  
    - Use the forecasts to **refine your investment strategy**  
    - Experiment with Machine Learning without needing coding skills

    This module is especially helpful for investors who want to:

    - Combine traditional analysis with modern AI techniques  
    - Get an **extra layer of insight** before making a decision  
    - Explore how ML can support smarter investing

    ---

    ### 🚧 Important note

    While Machine Learning can help identify patterns, it’s not a crystal ball.  
    The market is influenced by many unpredictable events.

    So always use forecasts as **one more tool** — not the only one — in your decision-making process.

""")
if st.button("📉 Go to Stock Forecasting"):
    st.switch_page("Wallet/3_ativo.py")
