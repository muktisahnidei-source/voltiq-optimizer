import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, PULP_CBC_CMD, LpStatus, LpStatusOptimal

# ================= PAGE SETUP =================
st.set_page_config(page_title="VoltIQ- Microgrid Optimization Dashboard", layout="wide")
st.title("⚡ VoltIQ- Microgrid Optimization Dashboard")
st.caption(
    "A user-driven optimization tool that schedules energy resources optimally "
    "for any load–solar scenario and visually demonstrates system behavior."
)

# ================= USER INPUT ZONE =================
st.header("🔧 User Inputs")

col1, col2 = st.columns(2)

with col1:
    data_mode = st.radio(
        "Load & Solar Input Method",
        ["Use Sample Data", "Manual Profile Generator", "Upload CSV"]
    )

with col2:
    st.markdown(
        """
        **CSV format (if uploaded):**  
        Columns: `hour, load, solar`  
        Rows: 24 (hours 1–24)  
        Units: kW (hourly average)
        """
    )

hours = np.arange(1, 25)

# -------- Load & Solar Input --------
if data_mode == "Use Sample Data":
    load = np.array([
        3,2.8,2.5,2.3,2.2,2.5,3.5,4.2,4.8,5.2,5.5,5.8,
        6,5.9,5.7,5.3,5,4.6,4.2,4,3.8,3.5,3.3,3
    ])
    solar = np.array([
        0,0,0,0,0.5,1.5,3,4.5,5.5,6,6.2,6.5,
        6.3,5.8,5,4,3,2,1,0.5,0,0,0,0
    ])

elif data_mode == "Manual Profile Generator":
    peak_load = st.slider("Peak Load (kW)", 3.0, 10.0, 6.0)
    peak_solar = st.slider("Peak Solar (kW)", 0.0, 8.0, 6.0)

    load = peak_load * np.exp(-0.5 * ((hours - 14) / 5)**2)
    solar = peak_solar * np.exp(-0.5 * ((hours - 12) / 4)**2)

else:
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file is None:
        st.warning("Upload a CSV file to proceed.")
        st.stop()

    df = pd.read_csv(file)

    if list(df.columns) != ["hour", "load", "solar"] or len(df) != 24:
        st.error("CSV must have columns: hour, load, solar and exactly 24 rows.")
        st.stop()

    load = df["load"].values
    solar = df["solar"].values

# -------- System Parameters --------
st.subheader("⚙️ System Parameters")

p1, p2, p3, p4 = st.columns(4)

with p1:
    battery_capacity = st.slider("Battery Capacity (kWh)", 5, 40, 10)
with p2:
    grid_price = st.slider("Grid Price (Rs/kWh)", 4, 15, 8)
with p3:
    diesel_price = st.slider("Diesel Price (Rs/kWh)", 15, 40, 20)
with p4:
    failure = st.selectbox(
        "Failure Scenario",
        ["None", "Solar Failure", "Grid Outage", "Battery Degradation"]
    )

# ================= FAILURE EFFECTS =================
capacity = battery_capacity
eta = 0.9
max_rate = 2
soc_min = 0.2 * capacity
soc_max = 0.9 * capacity

if failure == "Solar Failure":
    solar[:] = 0

if failure == "Grid Outage":
    grid_price = 100  # strong penalty

if failure == "Battery Degradation":
    capacity *= 0.5
    soc_min = 0.2 * capacity
    soc_max = 0.9 * capacity

# ================= BASELINE: RULE-BASED =================
def rule_based(load, solar):
    soc = 0.5 * capacity
    grid, diesel, soc_hist, cost_hist = [], [], [], []
    cost = 0

    for h in range(24):
        net = load[h] - solar[h]

        if net < 0:
            soc = min(soc + eta * min(-net, max_rate), soc_max)
            net = 0

        if net > 0:
            discharge = min(net, max_rate, soc - soc_min)
            soc -= discharge / eta
            net -= discharge

        if net > 0:
            if net <= 3:
                grid.append(net)
                diesel.append(0)
                cost += net * grid_price
            else:
                grid.append(3)
                diesel.append(net - 3)
                cost += 3 * grid_price + (net - 3) * diesel_price
        else:
            grid.append(0)
            diesel.append(0)

        soc_hist.append(soc)
        cost_hist.append(cost)

    return grid, diesel, soc_hist, cost_hist

# ================= OPTIMIZED =================
def optimized(load, solar):
    T = 24
    model = LpProblem("Microgrid_Optimization", LpMinimize)

    # Create decision variables explicitly. This avoids relying on
    # LpVariable.dicts(), which is the call that failed in the deployed app.
    G = [LpVariable(f"Grid_{t}", lowBound=0) for t in range(T)]
    D = [LpVariable(f"Diesel_{t}", lowBound=0) for t in range(T)]
    C = [LpVariable(f"Charge_{t}", lowBound=0, upBound=max_rate) for t in range(T)]
    B = [LpVariable(f"Discharge_{t}", lowBound=0, upBound=max_rate) for t in range(T)]

    # SOC has T+1 values: the initial state plus the state after each hour.
    SOC = [
        LpVariable(f"SOC_{t}", lowBound=soc_min, upBound=soc_max)
        for t in range(T + 1)
    ]

    model += lpSum(
        G[t] * grid_price + D[t] * diesel_price
        for t in range(T)
    )

    # Initial battery state.
    initial_soc = 0.5 * capacity
    model += SOC[0] == initial_soc

    for t in range(T):
        # Hourly power balance.
        model += solar[t] + G[t] + D[t] + B[t] == load[t] + C[t]

        # Battery state transition, including the first hour.
        model += SOC[t + 1] == SOC[t] + eta * C[t] - B[t] / eta

    # Preserve at least the starting charge at the end of the 24-hour cycle.
    model += SOC[T] >= initial_soc

    # Solve quietly and verify that the solver found an optimal solution.
    solver = PULP_CBC_CMD(msg=False)
    model.solve(solver)

    if model.status != LpStatusOptimal:
        raise RuntimeError(
            f"Optimization did not find an optimal solution. "
            f"Solver status: {LpStatus.get(model.status, model.status)}"
        )

    grid = [value(G[t]) for t in range(T)]
    diesel = [value(D[t]) for t in range(T)]
    soc = [value(SOC[t + 1]) for t in range(T)]

    cost = []
    c = 0
    for t in range(T):
        c += grid[t]*grid_price + diesel[t]*diesel_price
        cost.append(c)

    return grid, diesel, soc, cost

# ================= RUN OPTIMIZATION =================
st.header("▶ Run Optimization")

if st.button("Run Optimization"):

    rb_g, rb_d, rb_soc, rb_cost = rule_based(load, solar)

    try:
        op_g, op_d, op_soc, op_cost = optimized(load, solar)
    except Exception as exc:
        st.error(
            "VoltIQ could not complete the optimization. "
            "Please check the deployment logs for the technical details."
        )
        st.exception(exc)
        st.stop()

    # ================= RESULTS =================
    st.header("📊 Results & Visual Simulation")

    m1, m2, m3 = st.columns(3)
    m1.metric("Rule-Based Cost", f"Rs {rb_cost[-1]:.2f}")
    m2.metric("Optimized Cost", f"Rs {op_cost[-1]:.2f}")
    m3.metric("Savings", f"Rs {rb_cost[-1] - op_cost[-1]:.2f}")

    # -------- Graph 1: Load vs Solar --------
    st.subheader("Load vs Solar Profile")
    fig1, ax1 = plt.subplots()
    ax1.plot(hours, load, label="Load", color="orange")
    ax1.plot(hours, solar, label="Solar", color="green")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Power (kW)")
    ax1.legend()
    st.pyplot(fig1)

    # -------- Graph 2: Battery SOC --------
    st.subheader("Battery SOC (Rule-Based vs Optimized)")
    fig2, ax2 = plt.subplots()
    ax2.plot(rb_soc, label="Rule-Based SOC")
    ax2.plot(op_soc, label="Optimized SOC")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("SOC (kWh)")
    ax2.legend()
    st.pyplot(fig2)

    # -------- Graph 3: Grid & Diesel Usage --------
    st.subheader("Hourly Grid & Diesel Usage")
    fig3, ax3 = plt.subplots()
    ax3.bar(hours, rb_g, label="Grid (Rule-Based)")
    ax3.bar(hours, rb_d, bottom=rb_g, label="Diesel (Rule-Based)")
    ax3.plot(op_g, "--", label="Grid (Optimized)", color="black")
    ax3.plot(op_d, "--", label="Diesel (Optimized)", color="red")
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("Power (kW)")
    ax3.legend()
    st.pyplot(fig3)

    # -------- Graph 4: Cumulative Cost --------
    st.subheader("Cumulative Cost Comparison")
    fig4, ax4 = plt.subplots()
    ax4.plot(rb_cost, label="Rule-Based Cost")
    ax4.plot(op_cost, label="Optimized Cost")
    ax4.set_xlabel("Hour")
    ax4.set_ylabel("Cost (Rs)")
    ax4.legend()
    st.pyplot(fig4)

    # -------- Decision Reasoning (Explainable Logic) --------
    st.subheader("🧠 Decision Reasoning (Explainable Logic)")

    logic_points = []

    for h in range(24):
        if solar[h] > load[h] and op_soc[h] > rb_soc[h]:
            logic_points.append(
                f"Hour {h+1}: Excess solar available → battery charged for future demand."
            )
        elif op_d[h] == 0 and rb_d[h] > 0:
            logic_points.append(
                f"Hour {h+1}: Battery discharged to avoid diesel usage."
            )
        elif op_d[h] > 0:
            logic_points.append(
                f"Hour {h+1}: Diesel used as last resort due to high demand and low battery."
            )
        elif op_g[h] > 0 and op_d[h] == 0:
            logic_points.append(
                f"Hour {h+1}: Grid used instead of diesel due to lower operating cost."
            )

    logic_points = logic_points[:6]

    for point in logic_points:
        st.write("• " + point)

    st.info(
        "This logic explanation is automatically generated from optimization results, "
        "showing why specific operational decisions were taken at key hours."
    )
