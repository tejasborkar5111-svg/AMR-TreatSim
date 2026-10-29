"""
app.py - AMR-TreatSim web interface (Streamlit)

Run with:   python -m streamlit run app.py

Tab 1: simulator - predicts ARB log removal for UV, Ozone, AOP, MBR and a
       Hybrid (MBR + Ozone) train and recommends the cheapest option that
       meets the target.
Tab 2: real-case check - compares the ozone model with a published
       hospital-wastewater ozonation study (Baghal Asghari et al. 2021).
"""

import math

import pandas as pd
import streamlit as st

from kinetic_models import (
    uv_log_removal,
    first_order_log10_removal,
    mbr_log_removal,
    hybrid_log_removal,
    effluent_concentration,
    log_removal_to_percent,
)

st.set_page_config(page_title="AMR-TreatSim", layout="wide")
st.title("AMR-TreatSim")
st.caption("Predicting ARB removal across hospital wastewater treatment technologies")

# ---------------------------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------------------------
st.sidebar.header("1. Hospital & wastewater")
beds = st.sidebar.number_input("Number of beds", min_value=10, max_value=3000, value=300, step=10)
lpcd = st.sidebar.number_input(
    "Wastewater per bed (L/bed/day)", min_value=100, max_value=1200, value=450, step=25,
    help="BIS (India) suggests 450 L/bed/day for hospitals with more than 100 beds.",
)
flow_m3 = beds * lpcd / 1000.0

influent_log = st.sidebar.slider(
    "Influent ARB level (log10 CFU/mL)", 3.0, 9.0, 8.0, 0.5,
    help="8.0 means 10^8 CFU/mL, the starting level used in Baghal Asghari et al. 2021.",
)
target_log = st.sidebar.slider(
    "Target effluent ARB level (log10 CFU/mL)", 0.0, 6.0, 3.0, 0.5,
    help="3.0 means 10^3 CFU/mL.",
)
required_removal = max(influent_log - target_log, 0.0)

st.sidebar.header("2. Operating conditions")
uv_dose = st.sidebar.slider("UV dose (mJ/cm2)", 5, 100, 20, 5)
ozone_time = st.sidebar.slider("Ozone contact time (min)", 1, 30, 10, 1)
aop_time = st.sidebar.slider("AOP contact time (min)", 5, 120, 30, 5)
mbr_removal = st.sidebar.slider(
    "MBR log removal", 1.5, 7.3, 4.4, 0.1,
    help="Literature range 1.5-7.3 log (full-scale MBR ARG study). Default is the midpoint.",
)

with st.sidebar.expander("Rate constants (calibrated from literature)"):
    uv_slope = st.number_input(
        "UV: log10 removal per mJ/cm2", value=0.185, format="%.3f",
        help="This is the SLOPE of your lab graph (log removal vs dose). "
             "Start value 3.7 log / 20 mJ/cm2 = 0.185 (PubMed 24946583). "
             "Type your own lab slope here directly.",
    )
    ozone_k = st.number_input("Ozone k (1/min, natural-log basis)", value=1.0, format="%.2f",
                              help="Azuma et al. 2022: mean about 1.0 /min.")
    aop_k = st.number_input("AOP k (1/min, natural-log basis)",
                            value=round(2.9 * math.log(10) / 30, 4), format="%.4f",
                            help="Adamou et al. 2025, rNAu-1 variant: 2.9-log in 30 min.")
uv_k = uv_slope * math.log(10)  # model works in natural-log units internally

with st.sidebar.expander("Relative cost index (assumption)"):
    st.caption("Illustrative ranking only, NOT measured data. Replace with real cost figures when you have them.")
    cost_uv = st.number_input("UV", value=1.0, step=0.5)
    cost_ozone = st.number_input("Ozone", value=2.0, step=0.5)
    cost_aop = st.number_input("AOP", value=3.0, step=0.5)
    cost_mbr = st.number_input("MBR", value=4.0, step=0.5)
    cost_hybrid = st.number_input("Hybrid (MBR + Ozone)", value=6.0, step=0.5)

# ---------------------------------------------------------------------------
# Compute predictions
# ---------------------------------------------------------------------------
uv_lr = uv_log_removal(uv_k, uv_dose)
ozone_lr = first_order_log10_removal(ozone_k, ozone_time)
aop_lr = first_order_log10_removal(aop_k, aop_time)
mbr_lr = mbr_log_removal(mbr_removal)
hybrid_lr = hybrid_log_removal([mbr_lr, ozone_lr])

influent_conc = 10 ** influent_log
results = [
    ("UV", uv_lr, cost_uv),
    ("Ozone", ozone_lr, cost_ozone),
    ("AOP", aop_lr, cost_aop),
    ("MBR", mbr_lr, cost_mbr),
    ("Hybrid (MBR + Ozone, modeled)", hybrid_lr, cost_hybrid),
]

rows = []
for name, lr, cost in results:
    rows.append({
        "Technology": name,
        "Log removal": round(lr, 2),
        "Removal (%)": round(log_removal_to_percent(lr), 4),
        "Effluent (CFU/mL)": f"{effluent_concentration(influent_conc, lr):.3g}",
        "Meets target?": "Yes" if lr >= required_removal else "No",
        "Relative cost": cost,
    })
df = pd.DataFrame(rows)

tab_sim, tab_val = st.tabs(["Simulator", "Real-case check"])

# ---------------------------------------------------------------------------
# Tab 1: simulator
# ---------------------------------------------------------------------------
with tab_sim:
    c1, c2, c3 = st.columns(3)
    c1.metric("Hospital flow", f"{flow_m3:,.0f} m3/day")
    c2.metric("Influent ARB", f"10^{influent_log:g} CFU/mL")
    c3.metric("Required removal", f"{required_removal:.1f} log")

    st.subheader("Predicted performance")
    st.dataframe(df, width="stretch", hide_index=True)

    chart_df = pd.DataFrame({"Log removal": [r[1] for r in results]},
                            index=[r[0] for r in results])
    st.bar_chart(chart_df)
    st.caption(f"Required removal to hit the target: {required_removal:.1f} log. "
               "Any bar at or above this value meets the target.")

    st.subheader("Recommendation")
    meeting = [(n, lr, c) for n, lr, c in results if lr >= required_removal]
    if required_removal == 0:
        st.info("Target is not stricter than the influent, so no treatment is required.")
    elif meeting:
        best = min(meeting, key=lambda x: (x[2], -x[1]))
        st.success(
            f"Recommended: {best[0]} - predicted {best[1]:.2f}-log removal against "
            f"{required_removal:.1f}-log required, at the lowest relative cost among the options that meet the target."
        )
    else:
        closest = max(results, key=lambda x: x[1])
        st.error(
            f"No single option meets the {required_removal:.1f}-log target. "
            f"Closest is {closest[0]} at {closest[1]:.2f}-log. Consider a longer contact time/higher dose, "
            "or a combined treatment train."
        )

    st.warning(
        "Limitations: the model is calibrated on ARB (bacteria). ARG removal is usually much lower "
        "(e.g. UV gave only 0.2-1.4 log for genes in Azuma et al. 2024), and rate constants are organism-specific. "
        "The Hybrid value is modeled by adding MBR and Ozone log removals, not taken from a single study. "
        "Relative cost values are illustrative assumptions."
    )

# ---------------------------------------------------------------------------
# Tab 2: real-case check against a published study
# ---------------------------------------------------------------------------
with tab_val:
    st.subheader("Does the ozone model agree with a real hospital-wastewater study?")
    st.markdown(
        "**Study:** Baghal Asghari et al. (2021), *Scientific Reports* 11:24519. "
        "Batch ozonation of hospital wastewater from a Tabriz university hospital. The wastewater was "
        "sterilised and then spiked with *E. coli* at 10^8 CFU/mL (an experimental set-up, not "
        "untouched raw sewage). Contact time 5 min, transferred ozone dose 11-45 mg/L."
    )

    observed = pd.DataFrame({
        "Transferred ozone dose (mg/L)": [11, 25, 37, 45],
        "Observed E. coli log removal (approx.)": [1, 2, 4, 5],
    })
    model_5min = first_order_log10_removal(ozone_k, 5)
    observed["Our model at 5 min"] = round(model_5min, 2)

    st.dataframe(observed, width="stretch", hide_index=True)
    st.caption("Observed values are approximate, read from the paper's description "
               "('almost one, two, four and five orders of magnitude'). Check them against the paper's Figure 2 before quoting exact numbers.")
    st.bar_chart(observed.set_index("Transferred ozone dose (mg/L)"))

    lo, hi = observed.iloc[:, 1].min(), observed.iloc[:, 1].max()
    if lo <= model_5min <= hi:
        st.success(
            f"Our model predicts {model_5min:.2f}-log at 5 min, inside the 1-5 log range the study observed. "
            "So the order of magnitude is consistent."
        )
    else:
        st.error(f"Our model predicts {model_5min:.2f}-log at 5 min, outside the 1-5 log range the study observed.")

    st.info(
        "Honest reading: this is a consistency check, not a full validation. The study shows removal rising "
        "strongly with ozone DOSE (about 1-log at 11 mg/L up to about 5-log at 45 mg/L), but our ozone model "
        "uses contact time only, with a rate constant taken from a different study (Azuma et al. 2022), so it "
        "gives one value whatever the dose. Also, P. aeruginosa was harder to kill than E. coli in the same "
        "study (about 2-log at 37 mg/L), which again shows rate constants are organism-specific."
    )
