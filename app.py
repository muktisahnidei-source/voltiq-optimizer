import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum, value,
    PULP_CBC_CMD, LpStatus, LpStatusOptimal
)

# ================= PAGE CONFIGURATION =================
st.set_page_config(
    page_title="VoltIQ | Microgrid Energy Management",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================= PROFESSIONAL DARK NAVY THEME =================
st.markdown("""
<style>
:root { --navy:#081525; --panel:#10243A; --blue:#38BDF8; --green:#34D399; --orange:#FB923C; }
.stApp { background:linear-gradient(145deg,#071321 0%,#0B1B2D 55%,#10243A 100%); color:#E8F1F8; }
[data-testid="stSidebar"] { background:#0A1929; border-right:1px solid #203B56; }
[data-testid="stHeader"] { background:rgba(7,19,33,.85); }
h1,h2,h3 { color:#F2F7FC !important; }
p,label,.stMarkdown { color:#D5E2EE; }
.hero { background:linear-gradient(110deg,#102A43,#123654); border:1px solid #245071; border-radius:18px; padding:24px 28px; margin-bottom:20px; }
.hero h1 { margin:4px 0; font-size:2.35rem; }
.hero p { color:#B9CDDF; margin:0; }
.kicker { color:#38BDF8; font-size:.78rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#112A42,#0E2237); border:1px solid #24445F; padding:14px 16px; border-radius:14px; }
div[data-testid="stMetricLabel"] { color:#AFC3D6; }
div[data-testid="stMetricValue"] { color:#F2F7FC; }
div.stButton > button[kind="primary"] { background:linear-gradient(90deg,#0284C7,#0EA5E9); border:0; border-radius:10px; color:white; font-weight:700; min-height:2.8rem; }
div[data-testid="stExpander"] { background:#10243A; border:1px solid #24445F; border-radius:12px; }
.small-note { color:#9FB3C8; font-size:.84rem; }
hr { border-color:#244058; }
</style>
<div class="hero">
  <div class="kicker">Microgrid Energy Management System</div>
  <h1>⚡ VoltIQ</h1>
  <p>Intelligent energy scheduling • IoT-ready monitoring • Cost and carbon analysis</p>
</div>
""", unsafe_allow_html=True)

# ================= SAMPLE PROFILES =================
hours = np.arange(1, 25)
sample_load = np.array([3,2.8,2.5,2.3,2.2,2.5,3.5,4.2,4.8,5.2,5.5,5.8,6,5.9,5.7,5.3,5,4.6,4.2,4,3.8,3.5,3.3,3], dtype=float)
sample_solar = np.array([0,0,0,0,0.5,1.5,3,4.5,5.5,6,6.2,6.5,6.3,5.8,5,4,3,2,1,0.5,0,0,0,0], dtype=float)

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## ⚙️ System configuration")
    data_mode = st.radio(
        "Load & solar input",
        ["Use Sample Data", "Manual Profile Generator", "Upload CSV", "IoT Hardware Mode"],
    )
    st.divider()
    st.markdown("### Energy system parameters")
    battery_capacity = st.slider("Battery capacity (kWh)", 5, 40, 10)
    grid_price = st.slider("Grid tariff (₹/kWh)", 4, 15, 8)
    diesel_price = st.slider("Diesel cost (₹/kWh)", 15, 40, 20)
    eta = st.slider("Battery one-way efficiency", 0.70, 1.00, 0.90, 0.01)
    max_rate = st.slider("Battery charge/discharge limit (kW)", 1, 6, 2)
    st.divider()
    failure = st.selectbox("Operating scenario", ["None", "Solar Failure", "Grid Outage", "Battery Degradation"])
    st.caption("Grid outage disables grid imports. Battery degradation reduces usable capacity by 50%.")

# ================= INPUT DATA =================
st.markdown('<div class="kicker">01 / Energy input</div>', unsafe_allow_html=True)
input_col, profile_col = st.columns([1, 1.35], gap="large")
iot_simulation = False
sensor_df = None

with input_col:
    if data_mode == "Use Sample Data":
        load, solar = sample_load.copy(), sample_solar.copy()
        st.info("Using the built-in 24-hour sample load and solar profiles.")
    elif data_mode == "Manual Profile Generator":
        peak_load = st.slider("Peak load (kW)", 3.0, 10.0, 6.0, 0.5)
        peak_solar = st.slider("Peak solar generation (kW)", 0.0, 8.0, 6.0, 0.5)
        load = peak_load * np.exp(-0.5 * ((hours - 14) / 5) ** 2)
        solar = peak_solar * np.exp(-0.5 * ((hours - 12) / 4) ** 2)
        st.caption("Hourly profiles are generated from the selected peak values.")
    elif data_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="Required columns: hour, load, solar; exactly 24 rows.")
        if uploaded_file is None:
            st.warning("Upload a CSV file or select another input method.")
            st.stop()
        try:
            df = pd.read_csv(uploaded_file)
            if list(df.columns) != ["hour", "load", "solar"] or len(df) != 24:
                st.error("CSV must contain columns hour, load, solar in that order and exactly 24 rows.")
                st.stop()
            csv_hours = pd.to_numeric(df["hour"], errors="coerce").to_numpy()
            load = pd.to_numeric(df["load"], errors="coerce").to_numpy(dtype=float)
            solar = pd.to_numeric(df["solar"], errors="coerce").to_numpy(dtype=float)
            if not np.array_equal(csv_hours, hours):
                st.error("The hour column must contain integers 1 through 24 in order.")
                st.stop()
            if not np.all(np.isfinite(load)) or not np.all(np.isfinite(solar)) or np.any(load < 0) or np.any(solar < 0):
                st.error("Load and solar values must be finite, non-negative numbers.")
                st.stop()
            st.success("CSV profile loaded and validated.")
        except Exception as exc:
            st.error(f"Unable to read the CSV: {exc}")
            st.stop()
    else:
        st.markdown("### IoT hardware input")
        iot_choice = st.radio("Connection type", ["Simulation Mode", "ESP32 / Blynk connection"], help="Simulation works without physical components.")
        if iot_choice == "Simulation Mode":
            iot_simulation = True
            st.success("Simulation active — no physical ESP32 is connected.")
            st.caption("VoltIQ generates illustrative sensor telemetry using the sample microgrid profile.")
            load, solar = sample_load.copy(), sample_solar.copy()
            bus_voltage = st.slider("Illustrative DC bus voltage (V)", 12, 48, 24)
            gas_level = st.slider("Simulated MQ-2 sensor reading (ppm-equivalent)", 100, 1000, 250, 25)
        else:
            st.warning("Hardware not connected")
            st.write("This is a future integration placeholder. No live sensor data is being received.")
            st.caption("A future ESP32 integration can publish voltage, current, power and sensor status through Blynk API or MQTT.")
            st.code("ESP32 → Wi-Fi → Blynk / MQTT → VoltIQ", language="text")
            st.info("Select Simulation Mode to explore the IoT dashboard without hardware.")
            load, solar = sample_load.copy(), sample_solar.copy()

with profile_col:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.set_facecolor("#10243A"); fig.patch.set_facecolor("#10243A")
    ax.plot(hours, load, color="#38BDF8", linewidth=2.4, label="Load")
    ax.plot(hours, solar, color="#FBBF24", linewidth=2.4, label="Solar")
    ax.set(xlabel="Hour", ylabel="Power (kW)")
    ax.tick_params(colors="#C8D7E5")
    ax.xaxis.label.set_color("#C8D7E5"); ax.yaxis.label.set_color("#C8D7E5")
    ax.grid(alpha=.15, color="#AFC3D6")
    for spine in ax.spines.values(): spine.set_color("#36536D")
    leg = ax.legend(facecolor="#142C45", edgecolor="#36536D")
    for item in leg.get_texts(): item.set_color("#E8F1F8")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

if iot_simulation:
    # Simulated values only; these are not readings from physical sensors.
    solar_power_w = solar * 1000.0
    simulated_voltage = np.full(24, float(bus_voltage))
    simulated_current = np.divide(solar_power_w, simulated_voltage, out=np.zeros(24), where=simulated_voltage > 0)
    sensor_df = pd.DataFrame({
        "Hour": hours,
        "Simulated DC voltage (V)": np.round(simulated_voltage, 2),
        "Simulated current (A)": np.round(simulated_current, 2),
        "Calculated solar power (W)": np.round(solar_power_w, 2),
        "MQ-2 demonstration reading": [gas_level] * 24,
    })

# Preserve the selected input profile for fair scenario comparisons.
base_load = load.copy()
base_solar = solar.copy()

# ================= ACTIVE SCENARIO =================
capacity = float(battery_capacity)
if failure == "Solar Failure":
    solar = np.zeros_like(solar)
elif failure == "Battery Degradation":
    capacity *= 0.5
soc_min, soc_max = 0.20 * capacity, 0.90 * capacity
initial_soc = 0.50 * capacity
grid_available = failure != "Grid Outage"
if failure != "None":
    st.warning(f"Active scenario: **{failure}**. Results reflect this operating condition.")

# ================= DISPATCH MODELS =================
def dispatch_rule(load_values, solar_values, cap, min_soc, max_soc, initial, grid_ok):
    soc = initial
    grid_values, diesel_values, soc_values, cost_values = [], [], [], []
    running_cost = 0.0
    for t in range(24):
        net = float(load_values[t] - solar_values[t])
        if net < 0:
            charge = min(-net, max_rate, max(0.0, (max_soc - soc) / eta))
            soc += eta * charge
            net += charge
        if net > 0:
            discharge = min(net, max_rate, max(0.0, (soc - min_soc) * eta))
            soc -= discharge / eta
            net -= discharge
        # Any remaining negative net power is surplus solar that is curtailed.
        grid_use = max(0.0, net) if grid_ok else 0.0
        diesel_use = max(0.0, net) if not grid_ok else 0.0
        running_cost += grid_use * grid_price + diesel_use * diesel_price
        grid_values.append(grid_use); diesel_values.append(diesel_use)
        soc_values.append(soc); cost_values.append(running_cost)
    return grid_values, diesel_values, soc_values, cost_values

def dispatch_optimized(load_values, solar_values, cap, min_soc, max_soc, initial, grid_ok):
    T = 24
    model = LpProblem("VoltIQ_Microgrid_Optimization", LpMinimize)
    G = [LpVariable(f"Grid_{t}", lowBound=0) for t in range(T)]
    D = [LpVariable(f"Diesel_{t}", lowBound=0) for t in range(T)]
    C = [LpVariable(f"Charge_{t}", lowBound=0, upBound=max_rate) for t in range(T)]
    B = [LpVariable(f"Discharge_{t}", lowBound=0, upBound=max_rate) for t in range(T)]
    Curtail = [LpVariable(f"SolarCurtail_{t}", lowBound=0) for t in range(T)]
    SOC = [LpVariable(f"SOC_{t}", lowBound=min_soc, upBound=max_soc) for t in range(T + 1)]
    if not grid_ok:
        for t in range(T):
            model += G[t] == 0
    model += lpSum(G[t] * grid_price + D[t] * diesel_price for t in range(T))
    model += SOC[0] == initial
    for t in range(T):
        model += solar_values[t] + G[t] + D[t] + B[t] == load_values[t] + C[t] + Curtail[t]
        # Correct PuLP formulation: multiply by reciprocal; do not divide a variable.
        model += SOC[t + 1] == SOC[t] + eta * C[t] - (1.0 / eta) * B[t]
    model += SOC[T] >= initial
    model.solve(PULP_CBC_CMD(msg=False, timeLimit=30))
    if model.status != LpStatusOptimal:
        raise RuntimeError(f"Solver status: {LpStatus.get(model.status, model.status)}")
    def extract(items):
        vals = [value(items[t]) for t in range(T)]
        if any(v is None or not np.isfinite(v) for v in vals):
            raise RuntimeError("Solver returned incomplete values.")
        return [float(v) for v in vals]
    grid_values, diesel_values = extract(G), extract(D)
    soc_values = [float(value(SOC[t + 1])) for t in range(T)]
    running_cost, cost_values = 0.0, []
    for t in range(T):
        running_cost += grid_values[t] * grid_price + diesel_values[t] * diesel_price
        cost_values.append(running_cost)
    return grid_values, diesel_values, soc_values, cost_values

# ================= MAIN TABS =================
tab_overview, tab_results, tab_iot, tab_data = st.tabs(
    ["◈ Overview", "⚡ Optimization & scenarios", "⌁ IoT monitoring", "▤ Hourly data"]
)

with tab_overview:
    st.markdown('<div class="kicker">02 / System overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Daily energy demand", f"{np.sum(load):.1f} kWh")
    k2.metric("Solar generation", f"{np.sum(solar):.1f} kWh")
    k3.metric("Usable battery capacity", f"{capacity:.1f} kWh")
    k4.metric("Operating condition", "Grid outage" if not grid_available else "Normal / scenario")
    a, b, c = st.columns(3)
    a.metric("Grid tariff", f"₹{grid_price}/kWh")
    b.metric("Diesel cost", f"₹{diesel_price}/kWh")
    c.metric("Battery efficiency", f"{eta*100:.0f}%")
    st.markdown("### Energy flow overview")
    solar_used = np.minimum(solar, load)
    solar_surplus = np.maximum(solar - load, 0)
    residual_demand = np.maximum(load - solar, 0)
    flow1, flow2, flow3 = st.columns(3)
    flow1.metric("Solar used directly", f"{solar_used.sum():.1f} kWh")
    flow2.metric("Solar surplus", f"{solar_surplus.sum():.1f} kWh")
    flow3.metric("Demand after solar", f"{residual_demand.sum():.1f} kWh")
    flow_df = pd.DataFrame({"Hour": hours, "Load": load, "Solar": solar,
                            "Solar used directly": solar_used, "Residual demand": residual_demand})
    st.dataframe(flow_df.round(2), use_container_width=True, hide_index=True)
    st.caption("Hourly average power is treated as energy over one-hour intervals.")

with tab_results:
    st.markdown('<div class="kicker">03 / Optimization and decision support</div>', unsafe_allow_html=True)
    carbon_col1, carbon_col2 = st.columns(2)
    with carbon_col1:
        grid_ef = st.number_input("Grid emission factor (kg CO₂/kWh)", min_value=0.0, max_value=2.0, value=0.82, step=0.01,
                                  help="Editable illustrative factor. Replace with the applicable regional factor for formal reporting.")
    with carbon_col2:
        diesel_ef = st.number_input("Diesel generation emission factor (kg CO₂/kWh)", min_value=0.0, max_value=2.0, value=0.70, step=0.01,
                                    help="Editable illustrative operational factor.")
    compare_scenarios = st.checkbox("Compare all operating scenarios (None, Solar Failure, Grid Outage, Battery Degradation)", value=True)
    run_clicked = st.button("⚡ Run VoltIQ optimization", type="primary", use_container_width=True)
    if run_clicked:
        try:
            rb_g, rb_d, rb_soc, rb_cost = dispatch_rule(load, solar, capacity, soc_min, soc_max, initial_soc, grid_available)
            op_g, op_d, op_soc, op_cost = dispatch_optimized(load, solar, capacity, soc_min, soc_max, initial_soc, grid_available)
            rule_total, opt_total = rb_cost[-1], op_cost[-1]
            savings = rule_total - opt_total
            savings_pct = savings / rule_total * 100 if rule_total > 0 else 0.0
            rb_emissions = sum(rb_g) * grid_ef + sum(rb_d) * diesel_ef
            op_emissions = sum(op_g) * grid_ef + sum(op_d) * diesel_ef
            st.success("Optimization completed.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rule-based cost", f"₹{rule_total:,.2f}")
            m2.metric("Optimized cost", f"₹{opt_total:,.2f}")
            m3.metric("Cost difference", f"₹{savings:,.2f}", delta=f"{savings_pct:.1f}% vs rule-based")
            m4.metric("Optimized emissions", f"{op_emissions:,.2f} kg CO₂")
            st.markdown("### Energy dispatch")
            dispatch_fig, dispatch_ax = plt.subplots(figsize=(10, 3.8))
            dispatch_ax.stackplot(hours, solar, op_g, op_d, labels=["Solar profile", "Grid import", "Diesel generation"],
                                  colors=["#FBBF24", "#38BDF8", "#FB923C"], alpha=.8)
            dispatch_ax.plot(hours, load, color="#F8FAFC", linewidth=2, label="Load")
            dispatch_ax.set(xlabel="Hour", ylabel="Power (kW)", xticks=hours)
            dispatch_ax.grid(alpha=.18); dispatch_ax.legend(ncol=4)
            st.pyplot(dispatch_fig, use_container_width=True); plt.close(dispatch_fig)
            st.markdown("### Battery state of charge")
            soc_fig, soc_ax = plt.subplots(figsize=(10, 3.4))
            soc_ax.plot(hours, rb_soc, label="Rule-based", color="#38BDF8", linewidth=2)
            soc_ax.plot(hours, op_soc, label="Optimized", color="#34D399", linewidth=2)
            soc_ax.axhline(soc_min, color="#FB923C", linestyle=":", label="Minimum SOC")
            soc_ax.axhline(soc_max, color="#A78BFA", linestyle=":", label="Maximum SOC")
            soc_ax.set(xlabel="Hour", ylabel="Stored energy (kWh)", xticks=hours); soc_ax.grid(alpha=.2); soc_ax.legend()
            st.pyplot(soc_fig, use_container_width=True); plt.close(soc_fig)
            st.markdown("### Cost and carbon comparison")
            carbon_df = pd.DataFrame({
                "Strategy": ["Rule-based", "Optimized"],
                "Operating cost (₹)": [rule_total, opt_total],
                "Estimated emissions (kg CO₂)": [rb_emissions, op_emissions],
            })
            st.dataframe(carbon_df.round(2), use_container_width=True, hide_index=True)
            st.caption("Emissions are estimates based on the emission factors entered above; they are not direct sensor measurements.")
            if compare_scenarios:
                st.markdown("### Scenario comparison")
                scenario_rows = []
                for scenario_name in ["None", "Solar Failure", "Grid Outage", "Battery Degradation"]:
                    scenario_solar = base_solar.copy()
                    scenario_load = base_load.copy()
                    scenario_capacity = float(battery_capacity) * (0.5 if scenario_name == "Battery Degradation" else 1.0)
                    if scenario_name == "Solar Failure":
                        scenario_solar[:] = 0
                    scenario_grid_ok = scenario_name != "Grid Outage"
                    scenario_min, scenario_max = .2 * scenario_capacity, .9 * scenario_capacity
                    scenario_initial = .5 * scenario_capacity
                    sg, sd, ss, sc = dispatch_optimized(scenario_load, scenario_solar, scenario_capacity,
                                                         scenario_min, scenario_max, scenario_initial, scenario_grid_ok)
                    scenario_emissions = sum(sg) * grid_ef + sum(sd) * diesel_ef
                    scenario_rows.append({"Scenario": scenario_name, "Optimized cost (₹)": sc[-1],
                                          "Grid energy (kWh)": sum(sg), "Diesel energy (kWh)": sum(sd),
                                          "Estimated CO₂ (kg)": scenario_emissions})
                scenario_df = pd.DataFrame(scenario_rows)
                st.dataframe(scenario_df.round(2), use_container_width=True, hide_index=True)
                st.download_button("⬇ Download scenario comparison (CSV)", scenario_df.to_csv(index=False).encode("utf-8"),
                                   "voltiq_scenario_comparison.csv", "text/csv")
            st.markdown("### Explainable operating decisions")
            reason_rows = []
            for t in range(24):
                prev_soc = op_soc[t-1] if t > 0 else initial_soc
                if solar[t] > load[t] and op_soc[t] > prev_soc + 1e-5:
                    reason = "Surplus solar is stored in the battery."
                elif op_d[t] > 1e-5:
                    reason = "Diesel generation supplies remaining demand."
                elif op_g[t] > 1e-5:
                    reason = "Grid import supplies remaining demand."
                elif op_soc[t] < prev_soc - 1e-5:
                    reason = "Battery energy is discharged to meet demand."
                else:
                    reason = "Solar and available stored energy meet demand."
                reason_rows.append({"Hour": t+1, "Load (kW)": load[t], "Solar (kW)": solar[t],
                                    "Grid (kW)": op_g[t], "Diesel (kW)": op_d[t],
                                    "Battery SOC (kWh)": op_soc[t], "Decision": reason})
            results_df = pd.DataFrame(reason_rows)
            st.dataframe(results_df.round(2), use_container_width=True, hide_index=True)
            st.download_button("⬇ Download hourly optimization report (CSV)", results_df.to_csv(index=False).encode("utf-8"),
                               "voltiq_optimization_report.csv", "text/csv")
            st.session_state["voltiq_last_results"] = {"rule_cost": rule_total, "optimized_cost": opt_total,
                                                       "savings": savings, "estimated_emissions": op_emissions}
        except Exception as exc:
            st.error("VoltIQ could not complete the optimization.")
            st.exception(exc)
            st.info("If the solver fails, check the Streamlit deployment logs and the installed PuLP version.")

with tab_iot:
    st.markdown('<div class="kicker">04 / IoT monitoring</div>', unsafe_allow_html=True)
    if data_mode == "IoT Hardware Mode" and iot_simulation:
        st.success("SIMULATION MODE — illustrative values only; no physical ESP32 or MQ sensor is connected.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DC bus voltage", f"{bus_voltage:.1f} V")
        c2.metric("Solar current (peak)", f"{np.max(simulated_current):.2f} A")
        c3.metric("Solar power (peak)", f"{np.max(solar_power_w):.0f} W")
        c4.metric("MQ-2 status", "Normal" if gas_level < 500 else "Check level")
        st.line_chart(pd.DataFrame({"Solar power (W)": solar_power_w, "Solar current (A)": simulated_current}, index=hours))
        st.dataframe(sensor_df, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download simulated sensor data (CSV)", sensor_df.to_csv(index=False).encode("utf-8"),
                           "voltiq_simulated_iot_data.csv", "text/csv")
    elif data_mode == "IoT Hardware Mode":
        st.warning("Hardware not connected. This page is a connection placeholder, not a live feed.")
        st.markdown("""
**Future connection architecture**

1. ESP32 reads voltage/current sensors (for example, INA219).
2. ESP32 publishes telemetry through Wi-Fi using Blynk or MQTT.
3. VoltIQ receives validated readings through a configured endpoint.
4. Live measurements can replace simulated solar/load inputs after integration and calibration.
        """)
        st.code("ESP32 + INA219 / MQ sensor  →  Wi-Fi  →  Blynk / MQTT  →  VoltIQ", language="text")
        st.info("No credentials, API token, device connection or live data retrieval are implemented in this version.")
    else:
        st.info("Select **IoT Hardware Mode** in the sidebar to view simulated sensor telemetry or the future hardware connection plan.")
        st.write("VoltIQ can be used without any IoT hardware. The existing sample, manual and CSV input modes remain available.")

with tab_data:
    st.markdown('<div class="kicker">05 / Hourly data</div>', unsafe_allow_html=True)
    input_df = pd.DataFrame({"Hour": hours, "Load (kW)": load, "Solar generation (kW)": solar})
    st.dataframe(input_df.round(3), use_container_width=True, hide_index=True)
    st.download_button("⬇ Download input profile (CSV)", input_df.to_csv(index=False).encode("utf-8"),
                       "voltiq_input_profile.csv", "text/csv")
    st.caption("CSV upload format: hour, load, solar — exactly 24 rows, hours 1 to 24, values in kW.")

st.divider()
st.markdown('<p class="small-note">VoltIQ is an educational microgrid scheduling and decision-support model. IoT readings in Simulation Mode are illustrative, not physical measurements. The ESP32/Blynk connection is a future integration placeholder. Results depend on input profiles, tariffs, emission factors and simplified operating assumptions; this application does not directly control physical microgrid equipment.</p>', unsafe_allow_html=True)
