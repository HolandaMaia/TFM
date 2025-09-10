# home.py
import streamlit as st

# =========================================================
# Configuração da página
# =========================================================
st.set_page_config(
    page_title="Home | Retirement & Investment App",
    page_icon="💹",
    layout="wide"
)

# =========================================================
# Estilos básicos (CSS leve; compatível com tema claro/escuro)
# =========================================================
st.markdown("""
<style>
/* Reduz margens verticais padrão */
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

/* Título (hero) com peso e espaçamento melhor */
h1.hero-title {
  font-size: 2.1rem;
  line-height: 1.25;
  margin-bottom: .25rem;
}
p.hero-subtitle {
  font-size: 1.05rem;
  color: var(--text-color-secondary, #6b7280);
  margin-top: 0;
}

/* Cards */
.card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  padding: 18px 18px 14px 18px;
  background: rgba(255,255,255,0.55);
  backdrop-filter: blur(6px);
  transition: all .15s ease;
}
.card:hover { box-shadow: 0 10px 28px rgba(0,0,0,0.08); transform: translateY(-2px); }

/* Badges */
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid #d4d4d8;
  color: #52525b;
  background: #fafafa;
  margin-right: 6px;
  margin-bottom: 6px;
}

/* Botões mais “pill” */
button[kind="primary"] {
  border-radius: 999px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Utilitários de UI (código em PT-BR; texto visível em EN)
# =========================================================
def hero():
    """Seção inicial com headline + CTAs diretos para as ferramentas."""
    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1:
        st.markdown("🏠", help="Home")
        st.markdown("<h1 class='hero-title'>Retirement & Investment Dashboard</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p class='hero-subtitle'>Actuarial modeling, portfolio optimization, and AI-driven stock analysis — "
            "a clear path from retirement goals to actionable allocations.</p>",
            unsafe_allow_html=True
        )
        a, b, c = st.columns(3)
        with a:
            if st.button("🔢 Actuarial Calculator", type="primary", use_container_width=True):
                st.switch_page("Actuarial/2_calculadora.py")
        with b:
            if st.button("📊 Portfolio Optimizer", type="primary", use_container_width=True):
                st.switch_page("Wallet/4_wallet.py")
        with c:
            if st.button("📈 Stock Analysis", type="primary", use_container_width=True):
                st.switch_page("Wallet/3_ativo.py")

        st.write("")
        st.markdown(
            "<span class='badge'>Data-driven</span>"
            "<span class='badge'>Actuarial</span>"
            "<span class='badge'>Machine Learning</span>"
            "<span class='badge'>Risk-aware</span>",
            unsafe_allow_html=True
        )
    with c2:
        # Três métricas-resumo simples e autoexplicativas
        st.metric("Approach", "Actuarial + ML")
        st.metric("Coverage", "Global assets")
        st.metric("Focus", "Clarity & Action")

def tool_card(emoji: str, title: str, desc_md: str, bullets: list[str], button_label: str, route: str):
    """Card compacto e objetivo para cada ferramenta."""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"### {emoji} {title}")
    st.markdown(desc_md)
    for b in bullets:
        st.markdown(f"- {b}")
    st.write("")
    if st.button(button_label, use_container_width=True):
        st.switch_page(route)
    st.markdown("</div>", unsafe_allow_html=True)

def quick_start():
    """Pequena trilha de onboarding (5 passos)."""
    st.subheader("🚀 Quick Start")
    st.markdown("""
1. **Define your target**: retirement income and horizon.
2. **Set assumptions**: interest/discount rate and mortality table.
3. **Explore portfolios**: select assets and run the optimizer.
4. **Drill into assets**: fundamentals, technicals, and forecasts.
5. **Iterate & compare**: adjust inputs, validate risk/return, and refine.
""")

def highlights():
    """Três pontos de valor (curtos, sem poluição visual)."""
    a, b, c = st.columns(3)
    with a:
        st.info("**Transparent Inputs**  \nAssumptions and methods are explicit.")
    with b:
        st.success("**Actionable Outputs**  \nKPIs, charts, and clear next steps.")
    with c:
        st.warning("**Balanced View**  \nActuarial rigor meets ML insights.")

def footer():
    """Rodapé simples e discreto."""
    st.caption("This dashboard is for educational purposes and does not constitute financial advice.")

# =========================================================
# Layout da Home
# =========================================================
hero()
st.divider()

# Seção: Overview das ferramentas (cards)
st.subheader("🧰 Tools Overview")
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    tool_card(
        "🧮",
        "Actuarial Retirement Calculator",
        "Estimate the present value required to secure a fixed post-retirement income using UN life tables.",
        bullets=[
            "Life expectancy–weighted flows",
            "Transparent interest/discount rate",
            "Outputs in clear currency units"
        ],
        button_label="Open Actuarial Calculator",
        route="Actuarial/2_calculadora.py"
    )

with col2:
    tool_card(
        "💼",
        "Portfolio Optimizer",
        "Construct diversified portfolios via Modern Portfolio Theory and ML-based expectations.",
        bullets=[
            "Weights, risk/return, Sharpe",
            "Efficient frontier & correlations",
            "Benchmark comparison built-in"
        ],
        button_label="Open Portfolio Optimizer",
        route="Wallet/4_wallet.py"
    )

with col3:
    tool_card(
        "📈",
        "Stock Analysis & Forecasting",
        "Inspect fundamentals and technicals, and run ARMA-GARCH forecasts with accuracy tracking.",
        bullets=[
            "KPIs and technical charting",
            "In-sample vs. out-of-sample",
            "MAPE-based model feedback"
        ],
        button_label="Open Stock Analysis",
        route="Wallet/3_ativo.py"
    )

st.divider()

# Seção: Quick Start + Highlights
c_left, c_right = st.columns([1.05, 1], gap="large")
with c_left:
    quick_start()
with c_right:
    st.subheader("✨ Why this dashboard?")
    highlights()

st.divider()
footer()
