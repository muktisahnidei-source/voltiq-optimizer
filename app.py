import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum, value,
    PULP_CBC_CMD, LpStatus, LpStatusOptimal
)

# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(
    page_title="VoltIQ | Microgrid Energy Management",
    page_icon="⚡",
    layout="wide",
)

# A restrained, conventional dashboard style: light background,
# dark navy headings, simple borders and minimal decoration.
st.markdown("""
<style>
    .stApp { background: #F5F7FA; color: #172B3A; }
    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #DCE3EA; }
    [data-testid="stHeader"] { background: #F5F7FA; }
    h1, h2, h3 { color: #17324D !important; }
    p, label, .stMarkdown { color: #334155; }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE3EA;
        border-radius: 6px;
        padding: 12px 14px;
    }
    div[data-testid="stMetricLabel"] { color: #526579; }
    div[data-testid="stMetricValue"] { color: #17324D; }
    div.stButton > button[kind="primary"] {
        background: #174A70;
        border: 1px solid #174A70;
        border-radius: 5px;
        color: white;
        font-weight: 600;
    }
    div.stButton > button {
        border-radius: 5px;
    }
    div[data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #DCE3EA;
        border-radius: 6px;
    }
    hr { border-color: #DCE3EA; }
    .app-subtitle { color: #64748B; margin-top: -12px; }
    .section-note { color: #64748B; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("VoltIQ")
st.markdown(
    '<p class="app-subtitle">Microgrid energy management and scheduling</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="background:#FFFFFF;border:1px solid #DCE3EA;border-left:4px solid #174A70;'
    'padding:12px 16px;margin:16px 0 22px 0;border-radius:4px;">'
    '<strong style="color:#17324D;">Energy management at a glance</strong>'
    '<span style="color:#64748B;"> &nbsp; Profile monitoring &nbsp;|&nbsp; '
    'Cost optimization &nbsp;|&nbsp; Scenario analysis &nbsp;|&nbsp; IoT-ready monitoring</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# SAMPLE DATA
# ------------------------------------------------------------
hours = np.arange(1, 25)
sample_load = np.array([
    3.0, 2.8, 2.5, 2.3, 2.2, 2.5, 3.5, 4.2,
    4.8, 5.2, 5.5, 5.8, 6.0, 5.9, 5.7, 5.3,
    5.0, 4.6, 4.2, 4.0, 3.8, 3.5, 3.3, 3.0
], dtype=float)
sample_solar = np.array([
    0.0, 0.0, 0.0, 0.0, 0.5, 1.5, 3.0, 4.5,
    5.5, 6.0, 6.2, 6.5, 6.3, 5.8, 5.0, 4.0,
    3.0, 2.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0
], dtype=float)

# ------------------------------------------------------------
# SIDEBAR: INPUT AND SYSTEM SETTINGS
# ------------------------------------------------------------
with st.sidebar:
    st.header("Project settings")
    input_method = st.selectbox(
        "Energy data source",
        ["Sample data", "Manual profile", "Upload CSV", "IoT mode"],
    )

    st.divider()
    st.subheader("System parameters")
    battery_capacity = st.number_input(
        "Battery capacity (kWh)", min_value=1.0, max_value=500.0,
        value=10.0, step=1.0
    )
    grid_price = st.number_input(
        "Grid tariff (₹/kWh)", min_value=0.0, max_value=100.0,
        value=8.0, step=0.5
    )
    diesel_price = st.number_input(
        "Diesel generation cost (₹/kWh)", min_value=0.0, max_value=200.0,
        value=20.0, step=1.0
    )
    battery_efficiency = st.slider(
        "Battery one-way efficiency", 0.70, 1.00, 0.90, 0.01
    )
    max_battery_rate = st.number_input(
        "Maximum battery charge/discharge (kW)",
        min_value=0.1, max_value=100.0, value=2.0, step=0.5
    )

    st.divider()
    operating_scenario = st.selectbox(
        "Operating condition",
        ["Normal operation", "Solar failure", "Grid outage", "Battery degradation"],
    )
    st.caption(
        "Grid outage disables grid imports. Battery degradation reduces "
        "available battery capacity by 50%."
    )

# ------------------------------------------------------------
# INPUT DATA
# ------------------------------------------------------------
st.header("1. Energy profile")
st.markdown(
    '<p class="section-note">Choose a data source from the sidebar. '
    'VoltIQ uses hourly demand and solar generation to evaluate the microgrid.</p>',
    unsafe_allow_html=True,
)

iot_mode = None
iot_voltage = 24
iot_gas = 250
uploaded_name = None

if input_method == "Sample data":
    load = sample_load.copy()
    solar = sample_solar.copy()
    st.info("The built-in 24-hour sample profile is selected.")

elif input_method == "Manual profile":
    manual_col1, manual_col2 = st.columns(2)
    with manual_col1:
        peak_load = st.number_input(
            "Peak demand (kW)", min_value=0.1, max_value=100.0,
            value=6.0, step=0.5
        )
    with manual_col2:
        peak_solar = st.number_input(
            "Peak solar generation (kW)", min_value=0.0, max_value=100.0,
            value=6.5, step=0.5
        )
    load = peak_load * np.exp(-0.5 * ((hours - 14) / 5) ** 2)
    solar = peak_solar * np.exp(-0.5 * ((hours - 12) / 4) ** 2)
    st.caption("VoltIQ creates a representative 24-hour profile from the two peak values.")

elif input_method == "Upload CSV":
    st.markdown(
        "Upload a CSV containing exactly 24 hourly records and the columns "
        "`hour`, `load`, `solar`, in that order."
    )
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded_file is None:
        st.warning("Please upload your 24-hour CSV file to continue.")
        st.stop()

    uploaded_name = uploaded_file.name
    try:
        data = pd.read_csv(uploaded_file)
        if list(data.columns) != ["hour", "load", "solar"]:
            st.error("Column names and order must be exactly: hour, load, solar.")
            st.stop()
        if len(data) != 24:
            st.error(f"The file contains {len(data)} data rows. It must contain exactly 24.")
            st.stop()

        csv_hours = pd.to_numeric(data["hour"], errors="coerce").to_numpy()
        load = pd.to_numeric(data["load"], errors="coerce").to_numpy(dtype=float)
        solar = pd.to_numeric(data["solar"], errors="coerce").to_numpy(dtype=float)

        if not np.array_equal(csv_hours, hours):
            st.error("The hour column must contain 1, 2, 3, ... 24 in this order.")
            st.stop()
        if (not np.all(np.isfinite(load)) or not np.all(np.isfinite(solar))
                or np.any(load < 0) or np.any(solar < 0)):
            st.error("Load and solar values must be numeric, finite and non-negative.")
            st.stop()

        st.success(f"Successfully loaded: {uploaded_name}")
    except Exception as exc:
        st.error(f"Could not read this CSV file: {exc}")
        st.stop()

else:  # IoT mode
    st.markdown("Choose how you want to use the IoT section.")
    iot_mode = st.radio(
        "IoT data option",
        ["Simulated sensor data", "ESP32 / Blynk (not connected)"],
        horizontal=True,
    )
    load = sample_load.copy()
    solar = sample_solar.copy()

    if iot_mode == "Simulated sensor data":
        iot_col1, iot_col2 = st.columns(2)
        with iot_col1:
            iot_voltage = st.number_input(
                "Illustrative DC voltage (V)", min_value=1.0,
                max_value=100.0, value=24.0, step=1.0
            )
        with iot_col2:
            iot_gas = st.number_input(
                "Illustrative MQ-2 reading", min_value=0,
                max_value=2000, value=250, step=25
            )
        st.info("These are simulated values. No physical ESP32 or MQ sensor is connected.")
    else:
        st.warning(
            "Hardware is not connected. Live ESP32/Blynk data is not available "
            "in this version. The sample energy profile is used for the calculation."
        )

# Keep the original selected profile for scenario comparison.
base_load = load.copy()
base_solar = solar.copy()

# Apply the selected operating condition.
capacity = float(battery_capacity)
if operating_scenario == "Solar failure":
    solar = np.zeros(24, dtype=float)
elif operating_scenario == "Battery degradation":
    capacity *= 0.5

minimum_soc = 0.20 * capacity
maximum_soc = 0.90 * capacity
initial_soc = 0.50 * capacity
grid_available = operating_scenario != "Grid outage"

if operating_scenario != "Normal operation":
    st.warning(f"Current operating condition: {operating_scenario}.")

# ------------------------------------------------------------
# PROFILE PREVIEW
# ------------------------------------------------------------
profile_df = pd.DataFrame({
    "Hour": hours,
    "Load (kW)": load,
    "Solar generation (kW)": solar,
})
st.subheader("24-hour energy profile")
profile_chart, profile_axis = plt.subplots(figsize=(11, 3.6))
profile_chart.patch.set_facecolor("#FFFFFF")
profile_axis.set_facecolor("#FFFFFF")
profile_axis.plot(hours, load, color="#174A70", linewidth=2.6, label="Electricity demand")
profile_axis.plot(hours, solar, color="#D99A16", linewidth=2.6, label="Solar generation")
profile_axis.fill_between(hours, load, color="#174A70", alpha=0.07)
profile_axis.set_xlabel("Hour of day")
profile_axis.set_ylabel("Power (kW)")
profile_axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
profile_axis.grid(axis="y", color="#E5EAF0", linewidth=0.8)
profile_axis.spines[["top", "right"]].set_visible(False)
profile_axis.spines[["left", "bottom"]].set_color("#CBD5E1")
profile_axis.tick_params(colors="#526579")
profile_axis.xaxis.label.set_color("#526579")
profile_axis.yaxis.label.set_color("#526579")
profile_axis.legend(frameon=False, ncol=2, loc="upper left")
profile_chart.tight_layout()
st.pyplot(profile_chart, use_container_width=True)
plt.close(profile_chart)

with st.expander("View hourly input data"):
    st.dataframe(profile_df.round(2), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# OPTIMIZATION FUNCTIONS
# ------------------------------------------------------------
def run_rule_based(load_values, solar_values, min_soc, max_soc, start_soc, grid_ok):
    soc = float(start_soc)
    grid_values, diesel_values, soc_values = [], [], []
    total_cost = 0.0
    cumulative_cost = []

    for t in range(24):
        net_demand = float(load_values[t] - solar_values[t])

        if net_demand < 0:
            charge = min(
                -net_demand,
                max_battery_rate,
                max(0.0, (max_soc - soc) / battery_efficiency),
            )
            soc += battery_efficiency * charge
            net_demand += charge

        if net_demand > 0:
            discharge = min(
                net_demand,
                max_battery_rate,
                max(0.0, (soc - min_soc) * battery_efficiency),
            )
            soc -= discharge / battery_efficiency
            net_demand -= discharge

        grid_import = max(0.0, net_demand) if grid_ok else 0.0
        diesel_generation = max(0.0, net_demand) if not grid_ok else 0.0

        total_cost += grid_import * grid_price + diesel_generation * diesel_price
        grid_values.append(grid_import)
        diesel_values.append(diesel_generation)
        soc_values.append(soc)
        cumulative_cost.append(total_cost)

    return grid_values, diesel_values, soc_values, cumulative_cost


def run_optimized(load_values, solar_values, min_soc, max_soc, start_soc, grid_ok):
    model = LpProblem("VoltIQ_Microgrid_Optimization", LpMinimize)

    grid = [LpVariable(f"Grid_{t}", lowBound=0) for t in range(24)]
    diesel = [LpVariable(f"Diesel_{t}", lowBound=0) for t in range(24)]
    charge = [
        LpVariable(f"Charge_{t}", lowBound=0, upBound=max_battery_rate)
        for t in range(24)
    ]
    discharge = [
        LpVariable(f"Discharge_{t}", lowBound=0, upBound=max_battery_rate)
        for t in range(24)
    ]
    curtailment = [LpVariable(f"Curtail_{t}", lowBound=0) for t in range(24)]
    soc = [
        LpVariable(f"SOC_{t}", lowBound=min_soc, upBound=max_soc)
        for t in range(25)
    ]

    if not grid_ok:
        for t in range(24):
            model += grid[t] == 0

    model += lpSum(
        grid[t] * grid_price + diesel[t] * diesel_price
        for t in range(24)
    )
    model += soc[0] == start_soc

    for t in range(24):
        model += (
            solar_values[t] + grid[t] + diesel[t] + discharge[t]
            == load_values[t] + charge[t] + curtailment[t]
        )
        # Retains the corrected PuLP equation: never divide a PuLP variable.
        model += (
            soc[t + 1]
            == soc[t] + battery_efficiency * charge[t]
            - (1.0 / battery_efficiency) * discharge[t]
        )

    model += soc[24] >= start_soc
    model.solve(PULP_CBC_CMD(msg=False, timeLimit=30))

    if model.status != LpStatusOptimal:
        raise RuntimeError(f"Optimization status: {LpStatus.get(model.status, model.status)}")

    def values(variable_list):
        result = [value(variable_list[t]) for t in range(24)]
        if any(v is None or not np.isfinite(v) for v in result):
            raise RuntimeError("The solver returned incomplete results.")
        return [float(v) for v in result]

    grid_values = values(grid)
    diesel_values = values(diesel)
    soc_values = [float(value(soc[t + 1])) for t in range(24)]

    running_cost = 0.0
    cumulative_cost = []
    for t in range(24):
        running_cost += grid_values[t] * grid_price + diesel_values[t] * diesel_price
        cumulative_cost.append(running_cost)

    return grid_values, diesel_values, soc_values, cumulative_cost


# ------------------------------------------------------------
# RUN OPTIMIZATION
# ------------------------------------------------------------
st.divider()
st.divider()
st.header("2. Optimization")
st.markdown(
    '<p class="section-note">Compare conventional rule-based dispatch with '
    'VoltIQ cost-optimized scheduling.</p>',
    unsafe_allow_html=True,
)
run_col, explain_col = st.columns([1, 2])
with run_col:
    run_button = st.button("Run optimization", type="primary", use_container_width=True)
with explain_col:
    st.markdown(
        '<p class="section-note" style="padding-top:8px;">The optimizer schedules grid imports, '
        'battery operation and diesel generation to meet hourly demand at minimum modeled cost.</p>',
        unsafe_allow_html=True,
    )

if run_button:
    try:
        rb_grid, rb_diesel, rb_soc, rb_cost = run_rule_based(
            load, solar, minimum_soc, maximum_soc, initial_soc, grid_available
        )
        opt_grid, opt_diesel, opt_soc, opt_cost = run_optimized(
            load, solar, minimum_soc, maximum_soc, initial_soc, grid_available
        )

        rule_cost = rb_cost[-1]
        optimized_cost = opt_cost[-1]
        cost_difference = rule_cost - optimized_cost
        savings_percent = (
            cost_difference / rule_cost * 100 if rule_cost > 0 else 0.0
        )

        grid_emission_factor = 0.82
        diesel_emission_factor = 0.70
        rule_emissions = (
            sum(rb_grid) * grid_emission_factor
            + sum(rb_diesel) * diesel_emission_factor
        )
        optimized_emissions = (
            sum(opt_grid) * grid_emission_factor
            + sum(opt_diesel) * diesel_emission_factor
        )

        st.session_state["voltiq_results"] = {
            "rule_grid": rb_grid,
            "rule_diesel": rb_diesel,
            "rule_soc": rb_soc,
            "rule_cost": rb_cost,
            "opt_grid": opt_grid,
            "opt_diesel": opt_diesel,
            "opt_soc": opt_soc,
            "opt_cost": opt_cost,
            "rule_total": rule_cost,
            "optimized_total": optimized_cost,
            "rule_emissions": rule_emissions,
            "optimized_emissions": optimized_emissions,
            "savings_percent": savings_percent,
            "load": load.tolist(),
            "solar": solar.tolist(),
            "initial_soc": initial_soc,
            "minimum_soc": minimum_soc,
            "maximum_soc": maximum_soc,
        }
    except Exception as exc:
        st.error("VoltIQ could not complete the optimization.")
        st.exception(exc)

# Display last completed run so controls elsewhere do not hide results.
if "voltiq_results" in st.session_state:
    result = st.session_state["voltiq_results"]

    st.subheader("Optimization results")
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Rule-based operating cost", f"₹{result['rule_total']:,.2f}")
    metric2.metric("Optimized operating cost", f"₹{result['optimized_total']:,.2f}")
    metric3.metric("Cost reduction", f"{result['savings_percent']:.1f}%")
    metric4.metric("Optimized emissions", f"{result['optimized_emissions']:.2f} kg CO₂")

    st.subheader("Optimization visualizations")
    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:
        st.markdown("**Hourly power balance**")
        dispatch_chart, dispatch_axis = plt.subplots(figsize=(7, 3.5))
        dispatch_chart.patch.set_facecolor("#FFFFFF")
        dispatch_axis.set_facecolor("#FFFFFF")
        dispatch_axis.plot(hours, result["load"], color="#17324D", linewidth=2.4, label="Demand")
        dispatch_axis.plot(hours, result["solar"], color="#D99A16", linewidth=2.2, label="Solar")
        dispatch_axis.plot(hours, result["opt_grid"], color="#2583B8", linewidth=1.9, label="Grid")
        dispatch_axis.plot(hours, result["opt_diesel"], color="#C66A36", linewidth=1.9, label="Diesel")
        dispatch_axis.set_xlabel("Hour")
        dispatch_axis.set_ylabel("Power (kW)")
        dispatch_axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
        dispatch_axis.grid(axis="y", color="#E5EAF0")
        dispatch_axis.spines[["top", "right"]].set_visible(False)
        dispatch_axis.legend(frameon=False, fontsize=8)
        dispatch_chart.tight_layout()
        st.pyplot(dispatch_chart, use_container_width=True)
        plt.close(dispatch_chart)

    with chart_right:
        st.markdown("**Battery state of charge**")
        soc_chart, soc_axis = plt.subplots(figsize=(7, 3.5))
        soc_chart.patch.set_facecolor("#FFFFFF")
        soc_axis.set_facecolor("#FFFFFF")
        soc_axis.plot(hours, result["rule_soc"], color="#7B8794", linewidth=2,
                      linestyle="--", label="Rule-based")
        soc_axis.plot(hours, result["opt_soc"], color="#25835A", linewidth=2.4,
                      label="Optimized")
        soc_axis.axhline(result["minimum_soc"], color="#C66A36",
                         linestyle=":", linewidth=1.4, label="Minimum SOC")
        soc_axis.axhline(result["maximum_soc"], color="#7C6AA6",
                         linestyle=":", linewidth=1.4, label="Maximum SOC")
        soc_axis.set_xlabel("Hour")
        soc_axis.set_ylabel("Stored energy (kWh)")
        soc_axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
        soc_axis.grid(axis="y", color="#E5EAF0")
        soc_axis.spines[["top", "right"]].set_visible(False)
        soc_axis.legend(frameon=False, fontsize=8)
        soc_chart.tight_layout()
        st.pyplot(soc_chart, use_container_width=True)
        plt.close(soc_chart)

    st.markdown("**Cumulative operating cost**")
    cost_chart, cost_axis = plt.subplots(figsize=(11, 3.2))
    cost_chart.patch.set_facecolor("#FFFFFF")
    cost_axis.set_facecolor("#FFFFFF")
    cost_axis.plot(hours, result["rule_cost"], color="#7B8794", linewidth=2.2,
                   linestyle="--", label="Rule-based")
    cost_axis.plot(hours, result["opt_cost"], color="#174A70", linewidth=2.6,
                   label="VoltIQ optimized")
    cost_axis.set_xlabel("Hour")
    cost_axis.set_ylabel("Cumulative cost (₹)")
    cost_axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
    cost_axis.grid(axis="y", color="#E5EAF0")
    cost_axis.spines[["top", "right"]].set_visible(False)
    cost_axis.legend(frameon=False, ncol=2)
    cost_chart.tight_layout()
    st.pyplot(cost_chart, use_container_width=True)
    plt.close(cost_chart)

    st.subheader("Carbon emission estimate")
    emission_table = pd.DataFrame({
        "Strategy": ["Rule-based", "Optimized"],
        "Estimated emissions (kg CO₂)": [
            result["rule_emissions"], result["optimized_emissions"]
        ],
    })
    st.dataframe(emission_table.round(2), use_container_width=True, hide_index=True)
    st.caption(
        "Estimates use illustrative factors of 0.82 kg CO₂/kWh for grid electricity "
        "and 0.70 kg CO₂/kWh for diesel generation. These are editable in the source code "
        "and should be replaced with applicable verified factors for formal reporting."
    )

    # Scenario comparison uses the originally selected input profile.
    with st.expander("Compare operating scenarios"):
        scenario_rows = []
        for scenario_name in [
            "Normal operation", "Solar failure", "Grid outage", "Battery degradation"
        ]:
            scenario_capacity = float(battery_capacity)
            scenario_solar = np.array(base_solar, dtype=float).copy()
            if scenario_name == "Solar failure":
                scenario_solar[:] = 0.0
            if scenario_name == "Battery degradation":
                scenario_capacity *= 0.5

            scenario_min = 0.20 * scenario_capacity
            scenario_max = 0.90 * scenario_capacity
            scenario_initial = 0.50 * scenario_capacity
            scenario_grid_ok = scenario_name != "Grid outage"

            sg, sd, _, sc = run_optimized(
                base_load, scenario_solar, scenario_min, scenario_max,
                scenario_initial, scenario_grid_ok
            )
            scenario_rows.append({
                "Scenario": scenario_name,
                "Operating cost (₹)": sc[-1],
                "Grid energy (kWh)": sum(sg),
                "Diesel energy (kWh)": sum(sd),
                "Estimated emissions (kg CO₂)": (
                    sum(sg) * 0.82 + sum(sd) * 0.70
                ),
            })

        scenario_table = pd.DataFrame(scenario_rows)
        st.dataframe(scenario_table.round(2), use_container_width=True, hide_index=True)
        st.download_button(
            "Download scenario comparison CSV",
            scenario_table.to_csv(index=False).encode("utf-8"),
            "voltiq_scenario_comparison.csv",
            "text/csv",
        )

    hourly_report = pd.DataFrame({
        "hour": hours,
        "load": result["load"],
        "solar": result["solar"],
        "grid_import": result["opt_grid"],
        "diesel_generation": result["opt_diesel"],
        "battery_soc": result["opt_soc"],
        "cumulative_cost": result["opt_cost"],
    })
    st.download_button(
        "Download hourly optimization report CSV",
        hourly_report.to_csv(index=False).encode("utf-8"),
        "voltiq_optimization_report.csv",
        "text/csv",
    )

# ------------------------------------------------------------
# OPTIONAL IOT SECTION
# ------------------------------------------------------------
with st.expander("Optional IoT monitoring (ESP32 / Blynk)"):
    st.write(
        "This section is optional. It can be explored without hardware, but the "
        "current version does not receive live ESP32 or Blynk data."
    )
    if input_method == "IoT mode" and iot_mode == "Simulated sensor data":
        solar_watts = solar * 1000.0
        simulated_current = solar_watts / max(float(iot_voltage), 0.1)
        gas_status = "Check sensor level" if iot_gas >= 500 else "Normal"
        iot_m1, iot_m2, iot_m3 = st.columns(3)
        iot_m1.metric("Illustrative DC voltage", f"{iot_voltage:.1f} V")
        iot_m2.metric("Calculated peak current", f"{np.max(simulated_current):.2f} A")
        iot_m3.metric("MQ-2 demonstration status", gas_status)
        sensor_table = pd.DataFrame({
            "Hour": hours,
            "Simulated voltage (V)": [iot_voltage] * 24,
            "Calculated current (A)": np.round(simulated_current, 2),
            "Calculated solar power (W)": np.round(solar_watts, 2),
            "Simulated MQ-2 reading": [iot_gas] * 24,
        })
        st.dataframe(sensor_table, use_container_width=True, hide_index=True)
        st.download_button(
            "Download simulated sensor data CSV",
            sensor_table.to_csv(index=False).encode("utf-8"),
            "voltiq_simulated_iot_data.csv",
            "text/csv",
        )
        st.caption("All values shown here are simulated or calculated, not measured by physical sensors.")
    else:
        st.info(
            "Hardware status: Not connected. A future implementation can connect "
            "ESP32 sensor readings through Blynk API or MQTT."
        )
        st.code("ESP32 + sensors  →  Wi-Fi  →  Blynk / MQTT  →  VoltIQ")

# ------------------------------------------------------------
# CSV FORMAT HELP
# ------------------------------------------------------------
with st.expander("CSV upload format and example"):
    st.write("The uploaded file must be a CSV with exactly these three columns, in this order:")
    st.code(
        "hour,load,solar\n"
        "1,3.0,0.0\n"
        "2,2.8,0.0\n"
        "3,2.5,0.0\n"
        "...\n"
        "24,3.0,0.0",
        language="csv",
    )
    st.markdown("""
- **hour:** Integer hour number from 1 to 24.
- **load:** Electricity demand during that hour, in kW.
- **solar:** Solar power generated during that hour, in kW.

The file must contain 24 data rows, no extra columns, no blank cells and no negative values.
""")

st.divider()
footer_left, footer_right = st.columns([3, 2])
with footer_left:
    st.caption(
        "VoltIQ is an educational microgrid scheduling and decision-support model. "
        "It does not directly control physical equipment. IoT values are simulated unless "
        "a real hardware connection is implemented."
    )
with footer_right:
    st.markdown(
        '<p style="text-align:right;color:#526579;font-size:0.88rem;'
        'padding-top:6px;"><strong>Developed by Mukti Sahni</strong></p>',
        unsafe_allow_html=True,
    )
