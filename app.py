import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pulp import *
import base64

st.set_page_config(page_title="VoltIQ Optimizer", layout="wide")

st.title("⚡ VoltIQ – Smart Microgrid & EV Optimizer")

# -------- SIDEBAR INPUTS --------
st.sidebar.header("System Parameters")

battery_capacity = st.sidebar.number_input("Battery Capacity (kWh)", value=10)
diesel_cost = st.sidebar.number_input("Diesel Price (Rs/kWh)", value=20)

st.sidebar.subheader("EV Details")
ev_energy = st.sidebar.number_input("EV Energy Required (kWh)", value=6)
ev_arrival = st.sidebar.number_input("EV Arrival Hour (1-24)", min_value=1, max_value=24, value=10)
ev_depart = st.sidebar.number_input("EV Departure Hour (1-24)", min_value=1, max_value=24, value=20)
ev_rate = st.sidebar.number_input("Max EV Charging Rate (kW)", value=2)

# -------- FILE UPLOAD --------
st.sidebar.subheader("Upload Custom Data (Optional)")

uploaded = st.sidebar.file_uploader(
    "Upload CSV with columns: hour, load, solar",
    type=["csv"]
)

tariff_upload = st.sidebar.file_uploader(
    "Upload Tariff CSV (hour, price)",
    type=["csv"]
)

# -------- DEFAULT DATA --------
T = 24

default_load = [3,2.8,2.5,2.3,2.2,2.5,3.5,4.2,4.8,5.2,5.5,5.8,6,5.9,5.7,5.3,5,4.6,4.2,4,3.8,3.5,3.3,3]
default_solar = [0,0,0,0,0.5,1.5,3,4.5,5.5,6,6.2,6.5,6.3,5.8,5,4,3,2,1,0.5,0,0,0,0]

# -------- HANDLE FILE INPUT --------
if uploaded is not None:
    df = pd.read_csv(uploaded)
    load = df["load"].tolist()
    solar = df["solar"].tolist()
    st.success("Custom load & solar data uploaded!")
else:
    load = default_load
    solar = default_solar
    st.info("Using default load & solar profiles")

# -------- TARIFF SETTINGS --------
st.sidebar.subheader("Dynamic Tariff Options")

tariff_choice = st.sidebar.selectbox(
    "Tariff Mode",
    ["Default Flat", "Peak Evening", "Upload Tariff File"]
)

if tariff_choice == "Default Flat":
    grid_price = [8] * 24

elif tariff_choice == "Peak Evening":
    grid_price = [6]*8 + [10]*8 + [7]*8

else:
    if tariff_upload is not None:
        tdf = pd.read_csv(tariff_upload)
        grid_price = tdf["price"].tolist()
        st.success("Custom tariff uploaded!")
    else:
        grid_price = [8] * 24
        st.warning("No tariff uploaded – using default prices")

# -------- DISPLAY INPUT GRAPHS --------
st.subheader("Input Profiles")

c1, c2 = st.columns(2)

with c1:
    fig, ax = plt.subplots()
    ax.plot(load, label="Load", marker='o')
    ax.plot(solar, label="Solar", marker='x')
    ax.set_title("Load vs Solar")
    ax.legend()
    st.pyplot(fig)

with c2:
    fig2, ax2 = plt.subplots()
    ax2.plot(grid_price, marker='o')
    ax2.set_title("Hourly Tariff")
    ax2.set_ylabel("Price (Rs/kWh)")
    st.pyplot(fig2)

# -------- DOWNLOAD HELPER --------
def create_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV Report</a>'
    return href

# -------- OPTIMIZATION BUTTON --------
if st.button("Run Optimization"):

    if ev_arrival >= ev_depart:
        st.error("EV Arrival hour must be less than Departure hour")
        st.stop()

    st.subheader("Optimization Process")
    st.write("• Creating optimization model")
    st.write("• Adding constraints")
    st.write("• Solving for minimum cost schedule")

    model = LpProblem("Microgrid", LpMinimize)

    G = LpVariable.dicts("Grid", range(T), lowBound=0)
    D = LpVariable.dicts("Diesel", range(T), lowBound=0)
    C = LpVariable.dicts("Charge", range(T), lowBound=0, upBound=2)
    B = LpVariable.dicts("Discharge", range(T), lowBound=0, upBound=2)
    EV = LpVariable.dicts("EV", range(T), lowBound=0, upBound=ev_rate)
    SOC = LpVariable.dicts("SOC", range(T), lowBound=0.2*battery_capacity, upBound=0.9*battery_capacity)

    model += lpSum([G[t]*grid_price[t] + D[t]*diesel_cost for t in range(T)])

    for t in range(T):
        model += solar[t] + B[t] + G[t] + D[t] == load[t] + C[t] + EV[t]

    for t in range(1,T):
        model += SOC[t] == SOC[t-1] + 0.9*C[t] - B[t]/0.9

    model += SOC[0] == 0.5*battery_capacity

    model += lpSum([EV[t] for t in range(ev_arrival-1, ev_depart)]) == ev_energy

    model.solve()

    st.write("Solver Status:", LpStatus[model.status])

    Gv = [value(G[t]) for t in range(T)]
    Dv = [value(D[t]) for t in range(T)]
    SOCv = [value(SOC[t]) for t in range(T)]
    EVv = [value(EV[t]) for t in range(T)]

    baseline_cost = sum(max(load[t]-solar[t],0) * grid_price[t] for t in range(T))
    opt_cost = value(model.objective)

    st.subheader("Results Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Baseline Cost", f"Rs {round(baseline_cost,2)}")
    col2.metric("Optimized Cost", f"Rs {round(opt_cost,2)}")
    col3.metric("Savings", f"Rs {round(baseline_cost-opt_cost,2)}")

    st.subheader("Graphs")

    g1, g2 = st.columns(2)

    with g1:
        fig, ax = plt.subplots()
        ax.plot(SOCv, marker='o')
        ax.set_title("Battery SOC")
        st.pyplot(fig)

    with g2:
        fig2, ax2 = plt.subplots()
        ax2.bar(range(T), Gv, label="Grid")
        ax2.bar(range(T), Dv, bottom=Gv, label="Diesel")
        ax2.legend()
        ax2.set_title("Grid & Diesel Usage")
        st.pyplot(fig2)

    st.subheader("EV Charging Schedule")

    fig3, ax3 = plt.subplots()
    ax3.bar(range(1,T+1), EVv)
    ax3.set_title("EV Charging Plan")
    st.pyplot(fig3)

    st.subheader("Hourly Decision Table")

    result = pd.DataFrame({
        "Hour": range(1,25),
        "Load": load,
        "Solar": solar,
        "Tariff": grid_price,
        "Grid Used": np.round(Gv,2),
        "Diesel Used": np.round(Dv,2),
        "EV Charge": np.round(EVv,2),
        "SOC": np.round(SOCv,2)
    })

    st.dataframe(result)

    st.subheader("Download Reports")

    st.markdown(create_download_link(result, "optimization_results.csv"), unsafe_allow_html=True)

    report = f"""
VOLT-IQ OPTIMIZATION REPORT

Baseline Cost: Rs {round(baseline_cost,2)}
Optimized Cost: Rs {round(opt_cost,2)}
Total Savings: Rs {round(baseline_cost-opt_cost,2)}

Total Grid Energy Used: {round(sum(Gv),2)} kWh
Total Diesel Energy Used: {round(sum(Dv),2)} kWh
Total EV Charging Energy: {round(sum(EVv),2)} kWh

Solver Status: {LpStatus[model.status]}
"""

    st.download_button(
        label="Download Summary Report (TXT)",
        data=report,
        file_name="VoltIQ_Summary_Report.txt",
        mime="text/plain"
    )

    st.success("Optimization complete!")

