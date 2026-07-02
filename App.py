import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Economic Indicators Forecasting Platform",
    layout="wide"
)

# =====================================================================
# 1. LOAD ARTIFACTS & BASELINE DATA
# =====================================================================
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('Models/multi_output_lgbm.pkl')
    encoder = joblib.load('Models/country_encoder.pkl')
    return model, encoder


@st.cache_data
def load_historical_data():
    df = pd.read_csv('Datasets/cleaned_macro_data2.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


try:
    model, encoder = load_model_artifacts()
    master_df = load_historical_data()
except Exception as e:
    st.error(f"⚠️ Error loading model artifacts or data file: {e}")
    st.stop()



# =====================================================================
# 2. APP HEADER
# =====================================================================
st.title("Monetary Economics ML Simulation Engine")
st.markdown("""
    The ML Engine uses a multi-output LightGBM machine learning algorithm to simulate how global supply chain
    and economic shocks ripple through the economies of African nations. 
    By blending historical lags and log-return transformations, 
    the dashboard converts real-world policy delays into live, visual 
    forecasts for inflation, interest rates, and currency movements across the region.
""")
st.write("---")

# =====================================================================
# 3. MAIN AREA SIMULATION CONTROLS
# =====================================================================
st.header(" Model Inputs")

col1, col2 = st.columns(2)

with col1:
    st.image("Images/inflation.jpg",width=500)
with col2:
    st.image("Images/Exchange rates.jpg",width=480)

st.info(
        "Adjust your country parameters or global economic stress-testing scenarios above, then click 'Access Monetary Risk' to generate metrics.")

param_col1, param_col2 = st.columns(2)

with param_col1:
    available_countries = sorted(master_df['Country'].unique())
    selected_country = st.selectbox("**Select Country**", available_countries)


    country_history = master_df[master_df['Country'] == selected_country]
    latest_row = country_history.iloc[-1]
    latest_date_str = latest_row['Date'].strftime('%B %Y')
    st.caption(
        f" **Temporal Anchor:** **Automatically pulling historical baseline data points**.")

with param_col2:
  # Step 1: Choose the overarching Category using a Radio Button
    shock_category = st.radio(
        " **Select Shock Domain**",
        [
            "Trade & Supply Shocks",
            "Economic Shocks",
            "Commodity & Energy Shocks",
            "Monetary & Financial Shocks",
            "Geopolitical & External Shocks",
            "Capital Flow Shocks"
        ]
    )

    # Step 2: Dynamically change the sub-scenario dropdown options based on the radio selection
    if shock_category == "Trade & Supply Shocks":
        selected_scenario = st.selectbox("2. Select Scenario:", [
            "Baseline / Stable Global Trade",
            "Global Supply Chain Disruption",
            "Trade War & Tariff Escalation",
            "Global Export Restrictions"
        ])

    elif shock_category == "Economic Shocks":
        selected_scenario = st.selectbox("2. Select Scenario:", [
            "Global Economic Recession",
            "Global Economic Boom",
            "Financial Market Crisis",
            "Global Banking Crisis"
        ])

    elif shock_category == "Commodity & Energy Shocks":
        selected_scenario = st.selectbox("2. Select Scenario:", [
            "Energy & Oil Price Shock",
            "Fuel Price Surge",
            "Food Commodity Price Shock",
            "Commodity Price Collapse"
        ])

    elif shock_category == "Monetary & Financial Shocks":
        selected_scenario = st.selectbox("2. Select Scenario:", [
            "Major Central Bank Interest Rate Hike",
            "Global Inflation Surge",
            "Deflationary Shock",
            "Currency Market Volatility",
            "US Dollar Strengthening Shock"
        ])

    elif shock_category == "Geopolitical & External Shocks":
        selected_scenario = st.selectbox("2. Select Scenario:", [
            "Geopolitical Conflict",
            "International Sanctions",
            "Global Pandemic Resurgence",
            "Climate & Natural Disaster Shock"
        ])

    elif shock_category == "Capital Flow Shocks":
        selected_scenario = st.selectbox("2. Select Scenario:", [
            "Foreign Investment Surge",
            "Capital Flight",
            "External Debt Stress",
            "Decline in Foreign Direct Investment (FDI)"
        ])
  # 3. Comprehensive Mapping for All Scenario Paths
    if selected_scenario == "Baseline / Stable Global Trade":
        input_gpi, input_wop_vol = 0.0, 14.5
        severity = "warning"
        scenario_desc = "Balanced global distribution environment with normal maritime freight rates and baseline energy stability."

    elif selected_scenario == "Global Supply Chain Disruption":
        input_gpi, input_wop_vol = 2.4, 28.0
        severity = "warning"
        scenario_desc = "Severe maritime logistics gridlock and surging container freight indices, driving imported pricing shocks."

    elif selected_scenario == "Trade War & Tariff Escalation":
        input_gpi, input_wop_vol = 1.8, 22.0
        severity = "warning"
        scenario_desc = "Bilateral trade protectionism and retaliatory import levies, choking regional export volume growth."

    elif selected_scenario == "Global Export Restrictions":
        input_gpi, input_wop_vol = 2.1, 26.5
        severity = "warning"
        scenario_desc = "Protectionist export bans on essential raw industrial inputs, causing acute localized supply shocks."

    elif selected_scenario == "Global Economic Recession":
        input_gpi, input_wop_vol = -2.0, 12.0
        severity = "warning"
        scenario_desc = "Significant downturn in broad global economic activity, causing collapsing consumer demand and depressed transit metrics."

    elif selected_scenario == "Global Economic Boom":
        input_gpi, input_wop_vol = 1.8, 18.5
        severity = "Info"
        scenario_desc = "Strong worldwide growth and surging trade volumes, leading to moderate upward demand pressure on global markets."

    elif selected_scenario == "Financial Market Crisis":
        input_gpi, input_wop_vol = 2.5, 35.0
        severity = "warning"
        scenario_desc = "Sudden, severe drop in global asset values accompanied by high systemic uncertainty and fluctuating energy market risks."

    elif selected_scenario == "Global Banking Crisis":
        input_gpi, input_wop_vol = 3.2, 42.0
        severity = "warning"
        scenario_desc = "Widespread credit freezes and severe global liquidity strain, triggering extreme economic stress and maximum market volatility."

    elif selected_scenario == "Energy & Oil Price Shock":
        input_gpi, input_wop_vol = 1.2, 48.0
        severity = "warning"
        scenario_desc = "Supply-side crude disruptions spiking energy risk metrics, directly pressuring domestic transportation costs."

    elif selected_scenario == "Fuel Price Surge":
        input_gpi, input_wop_vol = 0.9, 40.0
        severity = "warning"
        scenario_desc = "Refinery capacity bottlenecks causing sharp increases in refined petroleum products despite baseline crude trends."

    elif selected_scenario == "Food Commodity Price Shock":
        input_gpi, input_wop_vol = 1.5, 25.0
        severity = "warning"
        scenario_desc = "Global agricultural supply failures inflating imported grain and fertilizer costs, driving severe local food CPI shocks."

    elif selected_scenario == "Commodity Price Collapse":
        input_gpi, input_wop_vol = -1.8, 30.0
        severity = "warning"
        scenario_desc = "Abrupt unwinding of global industrial demand, driving sharp drops in raw material prices and mining sector returns."

    elif selected_scenario == "Major Central Bank Interest Rate Hike":
        input_gpi, input_wop_vol = 1.5, 19.0
        severity = "warning"
        scenario_desc = "Aggressive monetary tightening by core central banks, draining liquidity and triggering capital flight from frontier markets."

    elif selected_scenario == "Global Inflation Surge":
        input_gpi, input_wop_vol = 2.0, 32.5
        severity = "warning"
        scenario_desc = "Systemic, worldwide inflationary pressures overshooting targets and raising the nominal baseline for all imported inputs."

    elif selected_scenario == "Deflationary Shock":
        input_gpi, input_wop_vol = -2.3, 11.0
        severity = "warning"
        scenario_desc = "Major collapse in broad aggregate global demand, leading to falling transit costs and structural domestic disinflation."

    elif selected_scenario == "Currency Market Volatility":
        input_gpi, input_wop_vol = 1.1, 29.0
        severity = "warning"
        scenario_desc = "Erratic shifts in major FX pairs increasing import billing uncertainty and destabilizing local exchange rate return models."

    elif selected_scenario == "US Dollar Strengthening Shock":
        input_gpi, input_wop_vol = 1.7, 21.0
        severity = "warning"
        scenario_desc = "Aggressive safe-haven flows into the USD, putting immediate structural depreciation pressure on regional currencies."

    elif selected_scenario == "Geopolitical Conflict":
        input_gpi, input_wop_vol = 2.2, 45.0
        severity = "warning"
        scenario_desc = "Outbreak of regional hostilities disrupting key international trade corridors and triggering immediate energy market panic."

    elif selected_scenario == "International Sanctions":
        input_gpi, input_wop_vol = 1.4, 31.0
        severity = "warning"
        scenario_desc = "Strict trade embargos altering global transaction tracking, forcing costly alternative trade routing structures."

    elif selected_scenario == "Global Pandemic Resurgence":
        input_gpi, input_wop_vol = -1.5, 38.0
        severity = "warning"
        scenario_desc = "Renewed international public health barriers halting physical border crossings and destabilizing energy demand structures."

    elif selected_scenario == "Climate & Natural Disaster Shock":
        input_gpi, input_wop_vol = 0.5, 24.0
        severity = "warning"
        scenario_desc = "Extreme environmental disruptions hitting regional infrastructure networks, creating localized logistics backlogs."

    elif selected_scenario == "Foreign Investment Surge":
        input_gpi, input_wop_vol = -0.5, 13.0
        severity = "info"
        scenario_desc = "Strong portfolio inflows driven by elevated global risk appetite, stabilizing frontier market external financing."

    elif selected_scenario == "Capital Flight":
        input_gpi, input_wop_vol = 2.0, 27.0
        severity = "warning"
        scenario_desc = "Massive risk-off migration of global capital toward developed markets, threatening local balance of payments stability."

    elif selected_scenario == "External Debt Stress":
        input_gpi, input_wop_vol = 1.6, 25.5
        severity = "warning"
        scenario_desc = "Rising global financing yields complicating domestic debt service burdens and straining reserves."

    elif selected_scenario == "Decline in Foreign Direct Investment (FDI)":
        input_gpi, input_wop_vol = 0.8, 17.0
        severity = "warning"
        scenario_desc = "Persistent cooling of long-term international capital allocations toward developing regional infrastructure projects."

# 1. Define the map strictly in UPPERCASE to match your exact dataset entries
currency_map = {
    "ANGOLA": "AOA",
    "BOTSWANA": "BWP",
    "BURUNDI": "BIF",
    "GHANA": "GHS",
    "KENYA": "KES",
    "NIGERIA": "NGN",
    "RWANDA": "RWF",
    "S. AFRICA": "ZAR",
    "SOUTH AFRICA": "ZAR",
    "UGANDA": "UGX"
}

# 2. Force the dropdown selection to uppercase during the lookup check
current_currency = currency_map.get(selected_country.upper(), "LOC")

st.write("")
run_forecast = st.button(" Access Monetary Risk ", use_container_width=True)
st.write("---")



# =====================================================================
# 4. BACKEND FEATURE ASSEMBLY
# =====================================================================
input_dict = {
    'Country': selected_country,
    'wop_lag1': latest_row['wop_lag1'],
    'wop_lag2': latest_row['wop_lag2'],
    'gea_lag1': latest_row['gea_lag1'],
    'wop_pct_change': latest_row['wop_pct_change'],
    'oil_moving_average': latest_row['oil_moving_average'],
    'gea_moving_average': latest_row['gea_moving_average'],
    'wop_volatility': input_wop_vol,  # Set by interactive dropdown selection
    'Global_Pressure_Index': input_gpi,  # Set by interactive dropdown selection
    'CPI Inflation_Lag1': latest_row['CPI Inflation_Lag1'],
    'CPI Inflation_Lag2': latest_row['CPI Inflation_Lag2'],
    ' Lending Interest Rate_Lag1': latest_row[' Lending Interest Rate_Lag1'],
    ' Lending Interest Rate_Lag2': latest_row[' Lending Interest Rate_Lag2'],
    'EXR_Log_Returns_Lag1': latest_row['EXR_Log_Returns_Lag1'],
    'EXR_Log_Returns_Lag2': latest_row['EXR_Log_Returns_Lag2']
}

input_df = pd.DataFrame([input_dict])
input_df['Country'] = encoder.transform(input_df[['Country']])

# =====================================================================
# 5. FORECAST RENDERING WORKSPACE
# =====================================================================
if run_forecast:

    # Trigger inference across the cloned model panel
    preds = model.predict(input_df)[0]

    pred_cpi = preds[0]
    pred_lending = preds[1]
    pred_exr_log_return = preds[2]

    # Reconstruct raw exchange rate parity from the stationary log return target
    current_raw_exr = latest_row['Exchange Rate']
    projected_raw_exr = current_raw_exr * np.exp(pred_exr_log_return)
    exr_pct_change = (np.exp(pred_exr_log_return) - 1) * 100

    # Display Predictions Panel
    st.subheader(f"Monetary Projections ({selected_country.lower()})")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label=" Predicted CPI Inflation",
            value=f"{pred_cpi:.2f}%",
            delta=f"{(pred_cpi - latest_row['CPI Inflation']):+.2f}% vs The Baseline"
        )

    with col2:
        st.metric(
            label=" Predicted Lending Interest Rate",
            value=f"{pred_lending:.2f}%",
            delta=f"{(pred_lending - latest_row[' Lending Interest Rate']):+.2f}% vs The Baseline"
        )

    with col3:
        # Determine movement and explicitly align the user interface color semantics
        if exr_pct_change > 0:
            direction = "Depreciation 🔻"
            color_mode = "inverse"  # Displays as standard red indicator for currency weakening
        else:
            direction = "Appreciation 🔺"
            color_mode = "normal"  # Displays as positive green indicator for currency strengthening

        st.metric(
            label=f" Projected Exchange Rate ({current_currency}/USD)",
            value=f"{projected_raw_exr:.2f}",
            delta=f"{exr_pct_change:+.2f}% {direction}",
            delta_color=color_mode
        )

    # Historical Baseline Summary Section at bottom
    st.write("---")
    st.subheader(f" Contextual Reference (Historical Baseline)")

    ref_col1, ref_col2, ref_col3 = st.columns(3)
    ref_col1.metric("Base Inflation Level", f"{latest_row['CPI Inflation']:.2f}%")
    ref_col2.metric("Base Lending Benchmark", f"{latest_row[' Lending Interest Rate']:.2f}%")
    ref_col3.metric("Base Spot Exchange Rate", f"{latest_row['Exchange Rate']:.2f} {current_currency}/USD")

    if severity == "warning":
        st.warning(f"**Scenario Dynamics (Risk Alert):** {scenario_desc}")
    elif severity == "success":
        st.success(f"**Scenario Dynamics (Favorable Inflows):** {scenario_desc}")
    else:
        st.info(f"**Scenario Dynamics:** {scenario_desc}")
else:
    # Default state before execution
    st.markdown("---")
