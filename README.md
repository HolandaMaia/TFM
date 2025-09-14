
# Modelo Integrado de Cálculo Actuarial y Optimización de Carteras para la Planificación de la Jubilación

🔗 **Live Demo**: [Can I Retire Yet?](https://caniretireyet.streamlit.app/)  

This project is a practical simulation tool developed by **Matheus Holanda Maia**.  
It integrates **actuarial modeling** with **investment portfolio optimization** to answer a key financial planning question:

> What periodic contributions are needed in an investment portfolio to reach the actuarial reserve required at retirement?

Unlike traditional approaches that separately analyze pension sufficiency or investment returns, this project combines both perspectives into a **unified framework**. The actuarial reserve becomes the financial target, and the investment strategy is optimized to achieve it.

---

## 🎯 Key Features

- Actuarial calculator based on UN mortality tables
- Financial data retrieval via Yahoo Finance
- Time series econometric modeling (ARMA-GARCH)
- Interactive visualizations with Plotly
- Portfolio optimization using Modern Portfolio Theory (Markowitz)
- Monte Carlo simulation of retirement outcomes
- User-friendly interface built with Streamlit

---

## 🔍 Data Sources

- **Mortality Tables**: United Nations (World Population Prospects, 2016–2023)  
- **Financial Data**: Yahoo Finance API (stocks, ETFs, indices)  

---

## 🛠️ Technologies and Libraries Used

- **Language**: Python  
- **Core Libraries**:  
  - Data Manipulation: `pandas`, `numpy`  
  - Visualization: `plotly`, `seaborn`, `matplotlib`  
  - User Interface: `streamlit`  
  - Time Series Modeling: `statsmodels` (ARMA), `arch` (GARCH)  
  - Evaluation Metrics: `sklearn.metrics` (MAPE)  
  - Portfolio Optimization: `PyPortfolioOpt`  
  - Actuarial Modeling: `pyliferisk`  

---

## 📂 Repository Structure

```
├── Actuarial/                          
│   ├── 1_apage1.py
│   └── 2_calculadora.py
│
├── Homepage/                           
│   └── 1_Homepage.py
│
├── Wallet/                             
│   ├── 3_ativo.py
│   └── 4_wallet.py
│
├── dados/                             
│   ├── ativos_totais.xlsx
│   └── WPP2024_MORT_F06_1_SINGLE_AGE_LIFE_TABLE_ESTIMATES_BOTH_SEXES.xlsx
│
├── app.py                              
├── requirements.txt                    
├── README.md                           
├── LICENSE                             
└── .gitignore                         
```

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/HolandaMaia/TFM.git
   cd TFM
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app/main.py
   ```

---

## 📈 Workflow Diagram

```mermaid
flowchart TD
    A[User Input] --> B[Actuarial Calculator]
    B --> C[Portfolio Optimizer]
    C --> D[Monte Carlo Simulation]
    D --> E[Contribution Estimation]
    E --> F[Results & Visualization]
```

---

## ⚠️ Disclaimer

This project is for **educational and simulation purposes only**. It does not constitute financial advice and should not be interpreted as an investment recommendation.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
