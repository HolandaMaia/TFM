Ordenando el Ordenado
Master’s Thesis (TFM) for the Master in Data Science, Big Data, and Business Analytics at Universidad Complutense de Madrid.

This project integrates actuarial modeling with investment portfolio optimization, addressing a key question:

What periodic contribution flow is required in an investment portfolio to reach the actuarial reserve needed at retirement?

🎯 Project Objectives
Calculate the actuarial present value of the reserve required to sustain a periodic income after retirement.
Project the future value of this reserve across different time horizons.
Translate the actuarial goal into annual or monthly contribution requirements under various investment strategies.
Build and evaluate an optimal investment portfolio using Modern Portfolio Theory (Markowitz).
Assess the probability of achieving the actuarial goal through Monte Carlo simulations, analyzing metrics such as Sharpe ratio, drawdown, and CAGR.
🛠️ Approach
The project is implemented as an interactive Python application (Streamlit) that includes:

Actuarial Calculator: Estimates the required reserve using UN mortality tables.
Portfolio Optimizer: Builds efficient portfolios using historical data from Yahoo Finance.
Contribution Simulation: Projects future scenarios using Monte Carlo simulations to estimate required contributions.
📂 Repository Structure
├── app/                # Streamlit application code
├── data/               # Input data (mortality tables, etc.)
├── notebooks/          # Exploratory analysis and tests
├── requirements.txt    # Project dependencies
└── README.md           # This file
📊 Data & Tools
Mortality Tables: United Nations (World Population Prospects).
Financial Data: Yahoo Finance (stocks, ETFs, global indices).
Technologies:
Python (pandas, numpy, scikit-learn)
Portfolio Optimization: PyPortfolioOpt
Visualization: Plotly
Actuarial Modeling: pyliferisk
UI: Streamlit
🚀 How to Run the Project
Clone the repository:

Create a virtual environment and install dependencies:

Run the application:

⚠️ Disclaimer
This project is for academic and simulation purposes only. It does not constitute financial advice and should not be interpreted as an investment recommendation.

📜 License
This project is distributed under the MIT License. See the LICENSE file for details.
