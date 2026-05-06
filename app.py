
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score


# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Shelter Capacity Decision Tool",
    layout="wide"
)

st.title("Shelter Capacity Decision Tool")

st.write("""
This prototype allows leadership to simulate shelter capacity scenarios, 
forecast future occupancy, compare projections against dynamic capacity, 
and receive a plain-language risk summary and recommendation.
""")


# =========================================================
# DEFAULT EXIT ASSUMPTIONS
# =========================================================
default_manual_exit_values = {
    "2026-05": 95,
    "2026-06": 95,
    "2026-07": 95,
    "2026-08": 95,
    "2026-09": 27,
    "2026-10": 79,
    "2026-11": 79,
    "2026-12": 79,
    "2027-01": 79,
    "2027-02": 79,
    "2027-03": 79,
    "2027-04": 79,
    "2027-05": 79,
    "2027-06": 79,
    "2027-07": 79,
    "2027-08": 79,
    "2027-09": 79
}


# =========================================================
# DYNAMIC CAPACITY
# =========================================================
def get_capacity(date):
    date = pd.to_datetime(date)

    if date < pd.to_datetime("2025-10-01"):
        return 354
    elif date < pd.to_datetime("2025-11-01"):
        return 367
    elif date < pd.to_datetime("2025-12-01"):
        return 387
    else:
        return 405


# =========================================================
# RULE-BASED RECOMMENDATION
# =========================================================
def get_rule_based_recommendation(risk_level):
    if risk_level == "Critical":
        return (
            "Immediate action is needed. Forecasted occupancy is at or above capacity. "
            "Leadership should consider opening additional capacity, increasing exits, "
            "or preparing an overflow response."
        )
    elif risk_level == "High":
        return (
            "Capacity risk is high. Leadership should monitor this period closely, "
            "review exit assumptions, and prepare a contingency plan."
        )
    elif risk_level == "Medium":
        return (
            "Capacity should continue to be monitored. Leadership may want to review "
            "whether current exit assumptions are realistic."
        )
    else:
        return (
            "No immediate capacity action is needed based on this scenario, "
            "but trends should continue to be monitored."
        )


# =========================================================
# MODEL TRAINING - CACHED
# =========================================================
@st.cache_resource
def train_model(X, y, sample_weights):
    model = CatBoostRegressor(
        iterations=300,
        depth=4,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=0,
        random_seed=42
    )

    model.fit(X, y, sample_weight=sample_weights)
    return model


# =========================================================
# AI-LIKE SUMMARY WITHOUT GOOGLE API
# =========================================================
def generate_ai_explanation(forecast_summary):
    worst_row = forecast_summary.loc[
        forecast_summary["Remaining_Capacity"].idxmin()
    ]

    return (
        f"The forecast shows the highest capacity risk in "
        f"{worst_row['Date'].strftime('%B %Y')}. "
        f"Forecasted occupancy is {worst_row['Forecasted_Occupancy']:.0f} "
        f"against a capacity of {worst_row['Capacity']:.0f}, leaving "
        f"{worst_row['Remaining_Capacity']:.0f} units available. "
        f"The risk level is {worst_row['Risk_Level']}. "
        f"Leadership should continue monitoring exits and prepare a response "
        f"if occupancy moves closer to capacity."
    )


# =========================================================
# FORECAST FUNCTION - CACHED
# =========================================================
@st.cache_data
def run_forecast(
    df_raw,
    start_date,
    end_date,
    exit_mode_value,
    same_exit,
    manual_exit_values,
    low_buffer,
    high_buffer
):

    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    df.rename(
        columns={
            "date": "Date",
            "Occupied_Units": "Occupancy"
        },
        inplace=True
    )

    if "Date" not in df.columns or "Occupancy" not in df.columns:
        raise ValueError(
            "Your CSV must include either 'date' and 'Occupied_Units' "
            "or 'Date' and 'Occupancy'."
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Occupancy"] = pd.to_numeric(df["Occupancy"], errors="coerce")

    df = (
        df[["Date", "Occupancy"]]
        .dropna()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    df["Exits"] = 0

    for lag in range(1, 7):
        df[f"lag_occ_{lag}"] = df["Occupancy"].shift(lag)

    df["roll_occ_3"] = df["Occupancy"].rolling(3).mean()
    df["roll_occ_6"] = df["Occupancy"].rolling(6).mean()

    df["month"] = df["Date"].dt.month
    df["trend"] = np.arange(len(df))

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.dropna().reset_index(drop=True)

    if len(df) <= 12:
        raise ValueError(
            "Not enough rows after feature engineering. You need more than 12 rows."
        )

    alpha = 0.03
    time_index = np.arange(len(df))
    sample_weights = np.exp(alpha * (time_index - time_index[-1]))

    feature_cols = [f"lag_occ_{i}" for i in range(1, 7)] + [
        "roll_occ_3",
        "roll_occ_6",
        "Exits",
        "trend",
        "sin_month",
        "cos_month"
    ]

    X = df[feature_cols]
    y = df["Occupancy"]

    model = train_model(X, y, sample_weights)

    X_train = X.iloc[:-12]
    y_train = y.iloc[:-12]
    X_test = X.iloc[-12:]
    y_test = y.iloc[-12:]

    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))

    history = df.copy()

    future_dates = pd.date_range(
        start=pd.to_datetime(start_date),
        end=pd.to_datetime(end_date),
        freq="MS"
    )

    forecast_results = []

    for date in future_dates:
        month_key = date.to_period("M").strftime("%Y-%m")

        if exit_mode_value == "Use same exits for all months":
            injected_exits = same_exit
        else:
            injected_exits = manual_exit_values.get(month_key, 0)

        row = {
            "Date": date,
            "month": date.month,
            "trend": len(history),
            "sin_month": np.sin(2 * np.pi * date.month / 12),
            "cos_month": np.cos(2 * np.pi * date.month / 12),
            "Exits": injected_exits
        }

        for lag in range(1, 7):
            row[f"lag_occ_{lag}"] = history["Occupancy"].iloc[-lag]

        row["roll_occ_3"] = history["Occupancy"].iloc[-3:].mean()
        row["roll_occ_6"] = history["Occupancy"].iloc[-6:].mean()

        X_future = pd.DataFrame([row])[feature_cols]
        forecast = float(model.predict(X_future)[0])

        forecast_results.append({
            "Date": date,
            "Injected_Exits": injected_exits,
            "Forecasted_Occupancy": round(forecast, 2)
        })

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [{
                        "Date": date,
                        "Occupancy": forecast,
                        "Exits": injected_exits
                    }]
                )
            ],
            ignore_index=True
        )

    forecast_summary = pd.DataFrame(forecast_results)

    forecast_summary["Capacity"] = forecast_summary["Date"].apply(get_capacity)

    forecast_summary["Remaining_Capacity"] = (
        forecast_summary["Capacity"] - forecast_summary["Forecasted_Occupancy"]
    ).round(2)

    forecast_summary["Risk_Level"] = np.select(
        [
            forecast_summary["Forecasted_Occupancy"] >= forecast_summary["Capacity"],
            forecast_summary["Remaining_Capacity"] < high_buffer,
            forecast_summary["Remaining_Capacity"] < low_buffer,
            forecast_summary["Remaining_Capacity"] >= low_buffer
        ],
        [
            "Critical",
            "High",
            "Medium",
            "Low"
        ],
        default="Unknown"
    )

    forecast_summary["Rule_Based_Recommendation"] = forecast_summary[
        "Risk_Level"
    ].apply(get_rule_based_recommendation)

    forecast_summary["Leadership_Summary"] = forecast_summary.apply(
        lambda r: (
            f"In {r['Date'].strftime('%B %Y')}, forecasted occupancy is "
            f"{r['Forecasted_Occupancy']:.0f}, capacity is {r['Capacity']:.0f}, "
            f"remaining capacity is {r['Remaining_Capacity']:.0f}, "
            f"and the risk level is {r['Risk_Level']}."
        ),
        axis=1
    )

    return df, forecast_summary, train_r2, test_r2


# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("Scenario Inputs")

forecast_start = st.sidebar.date_input(
    "Forecast Start Date",
    value=pd.to_datetime("2026-05-01")
)

forecast_end = st.sidebar.date_input(
    "Forecast End Date",
    value=pd.to_datetime("2027-09-01")
)

exit_mode = st.sidebar.radio(
    "Exit Scenario Mode",
    [
        "Use default monthly exits",
        "Use same exits for all months"
    ]
)

same_exit_value = st.sidebar.number_input(
    "Monthly Exits if using same value",
    min_value=0,
    value=79,
    step=1
)

low_risk_buffer = st.sidebar.number_input(
    "Medium Risk if remaining capacity is below:",
    min_value=0,
    value=50,
    step=1
)

high_risk_buffer = st.sidebar.number_input(
    "High Risk if remaining capacity is below:",
    min_value=0,
    value=20,
    step=1
)


# =========================================================
# LOAD CSV AUTOMATICALLY
# =========================================================
try:
    df_uploaded = pd.read_csv("monthly_occupancy.csv")
    data_loaded = True
except FileNotFoundError:
    st.error(
        "monthly_occupancy.csv was not found. "
        "Please place monthly_occupancy.csv in the same folder as app.py."
    )
    data_loaded = False


# =========================================================
# RUN APP
# =========================================================
if data_loaded:

    if st.sidebar.button("Run Agent"):

        try:
            historical_df, forecast_summary, train_r2, test_r2 = run_forecast(
                df_raw=df_uploaded,
                start_date=forecast_start,
                end_date=forecast_end,
                exit_mode_value=exit_mode,
                same_exit=same_exit_value,
                manual_exit_values=default_manual_exit_values,
                low_buffer=low_risk_buffer,
                high_buffer=high_risk_buffer
            )

            worst_row = forecast_summary.loc[
                forecast_summary["Remaining_Capacity"].idxmin()
            ]

            agent_summary = generate_ai_explanation(forecast_summary)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Train R²", round(train_r2, 4))
            col2.metric("Test R²", round(test_r2, 4))
            col3.metric("Highest Risk Month", worst_row["Date"].strftime("%b %Y"))
            col4.metric("Highest Risk Level", worst_row["Risk_Level"])

            st.subheader("Key Insight")

            st.write(f"""
            The highest-risk month in this scenario is **{worst_row['Date'].strftime('%B %Y')}**.  
            Forecasted occupancy is **{worst_row['Forecasted_Occupancy']:.0f}** against a capacity of **{worst_row['Capacity']:.0f}**.  
            Remaining capacity is **{worst_row['Remaining_Capacity']:.0f}**, and the risk level is **{worst_row['Risk_Level']}**.
            """)

            st.subheader("Agent Summary")
            st.write(agent_summary)

            st.subheader("Rule-Based Recommendation")
            st.write(worst_row["Rule_Based_Recommendation"])

            st.subheader("Occupancy Forecast vs. Dynamic Capacity")

            fig, ax = plt.subplots(figsize=(13, 6))

            ax.plot(
                historical_df["Date"],
                historical_df["Occupancy"],
                marker="o",
                label="Historical Occupancy"
            )

            ax.plot(
                forecast_summary["Date"],
                forecast_summary["Forecasted_Occupancy"],
                marker="o",
                linestyle="--",
                label="Forecasted Occupancy"
            )

            ax.plot(
                forecast_summary["Date"],
                forecast_summary["Capacity"],
                linestyle="-",
                label="Dynamic Capacity"
            )

            ax.set_title("Shelter Occupancy Forecast with Dynamic Capacity")
            ax.set_xlabel("Date")
            ax.set_ylabel("Occupied Units")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)

            st.subheader("Forecast Table")

            display_cols = [
                "Date",
                "Injected_Exits",
                "Forecasted_Occupancy",
                "Capacity",
                "Remaining_Capacity",
                "Risk_Level",
                "Rule_Based_Recommendation",
                "Leadership_Summary"
            ]

            st.dataframe(
                forecast_summary[display_cols],
                use_container_width=True
            )

            csv = forecast_summary[display_cols].to_csv(index=False)

            st.download_button(
                label="Download Forecast Results",
                data=csv,
                file_name="ai_shelter_capacity_agent_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
