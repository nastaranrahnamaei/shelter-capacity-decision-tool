%%writefile app.py
import pandas as pd
import numpy as np
import streamlit as st

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score


st.set_page_config(page_title="Shelter Capacity Decision Tool", layout="wide")

st.title("Shelter Capacity Decision Tool")

st.write(
    "Simulate shelter capacity scenarios, forecast occupancy, "
    "and open the Tableau dashboard in a separate tab."
)


TABLEAU_DASHBOARD_URL = (
    "https://tableau.dc.gov/#/site/DHS/views/"
    "DRAFT-New-Version-STFH-Projections-FY26-FY27/"
    "NicholeProjections-secondscenario?:iid=3"
)


DEFAULT_EXITS = {
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
    "2027-09": 79,
}


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


def get_recommendation(risk):
    if risk == "Critical":
        return "Immediate action is needed. Forecasted occupancy is at or above capacity."
    elif risk == "High":
        return "Capacity risk is high. Prepare a contingency plan."
    elif risk == "Medium":
        return "Monitor capacity and review exit assumptions."
    else:
        return "No immediate action needed. Continue monitoring."


def run_forecast(
    df_raw,
    start_date,
    end_date,
    exit_mode,
    same_exit,
    manual_exit_values,
    medium_buffer,
    high_buffer,
):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    df.rename(
        columns={
            "date": "Date",
            "Occupied_Units": "Occupancy",
        },
        inplace=True,
    )

    if "Date" not in df.columns or "Occupancy" not in df.columns:
        raise ValueError(
            "CSV must contain either 'Date' and 'Occupancy' "
            "or 'date' and 'Occupied_Units'."
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Occupancy"] = pd.to_numeric(df["Occupancy"], errors="coerce")

    df = (
        df[["Date", "Occupancy"]]
        .dropna()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if exit_mode == "Use same exits for all months":
        df["Exits"] = int(same_exit)
    else:
        df["Exits"] = (
            df["Date"]
            .dt.to_period("M")
            .astype(str)
            .map(manual_exit_values)
            .fillna(0)
        )

    for lag in range(1, 7):
        df[f"lag_occ_{lag}"] = df["Occupancy"].shift(lag)

    df["roll_occ_3"] = df["Occupancy"].rolling(3).mean()
    df["roll_occ_6"] = df["Occupancy"].rolling(6).mean()
    df["month"] = df["Date"].dt.month
    df["trend"] = np.arange(len(df))
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_cols = [f"lag_occ_{i}" for i in range(1, 7)] + [
        "roll_occ_3",
        "roll_occ_6",
        "Exits",
        "trend",
        "sin_month",
        "cos_month",
    ]

    X = df[feature_cols]
    y = df["Occupancy"]

    alpha = 0.03
    time_index = np.arange(len(df))
    sample_weights = np.exp(alpha * (time_index - time_index[-1]))

    model = CatBoostRegressor(
        iterations=300,
        depth=4,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=0,
        random_seed=42,
    )

    model.fit(X, y, sample_weight=sample_weights)

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
        freq="MS",
    )

    forecast_results = []

    for date in future_dates:
        month_key = date.to_period("M").strftime("%Y-%m")

        if exit_mode == "Use same exits for all months":
            injected_exits = int(same_exit)
        else:
            injected_exits = int(manual_exit_values.get(month_key, 0))

        row = {
            "Date": date,
            "month": date.month,
            "trend": len(history),
            "sin_month": np.sin(2 * np.pi * date.month / 12),
            "cos_month": np.cos(2 * np.pi * date.month / 12),
            "Exits": injected_exits,
        }

        for lag in range(1, 7):
            row[f"lag_occ_{lag}"] = history["Occupancy"].iloc[-lag]

        row["roll_occ_3"] = history["Occupancy"].iloc[-3:].mean()
        row["roll_occ_6"] = history["Occupancy"].iloc[-6:].mean()

        X_future = pd.DataFrame([row])[feature_cols]
        forecast = float(model.predict(X_future)[0])

        forecast_results.append(
            {
                "Date": date,
                "Forecasted_Occupancy": round(forecast, 2),
                "Injected_Exits": injected_exits,
            }
        )

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        {
                            "Date": date,
                            "Occupancy": forecast,
                            "Exits": injected_exits,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    forecast_summary = pd.DataFrame(forecast_results)

    forecast_summary["Capacity"] = forecast_summary["Date"].apply(get_capacity)

    forecast_summary["Remaining_Capacity"] = (
        forecast_summary["Capacity"]
        - forecast_summary["Forecasted_Occupancy"]
    ).round(2)

    forecast_summary["Risk_Level"] = np.select(
        [
            forecast_summary["Forecasted_Occupancy"] >= forecast_summary["Capacity"],
            forecast_summary["Remaining_Capacity"] < high_buffer,
            forecast_summary["Remaining_Capacity"] < medium_buffer,
        ],
        ["Critical", "High", "Medium"],
        default="Low",
    )

    forecast_summary["Recommendation"] = forecast_summary["Risk_Level"].apply(
        get_recommendation
    )

    return df, forecast_summary, train_r2, test_r2


st.sidebar.header("Scenario Inputs")

start = st.sidebar.date_input(
    "Forecast Start Date",
    value=pd.to_datetime("2026-05-01"),
)

end = st.sidebar.date_input(
    "Forecast End Date",
    value=pd.to_datetime("2027-09-01"),
)

mode = st.sidebar.radio(
    "Exit Scenario Mode",
    [
        "Use default monthly exits",
        "Use same exits for all months",
        "Enter different exits by month",
    ],
)

same_exit = st.sidebar.number_input(
    "Monthly Exits if using same value",
    min_value=0,
    value=79,
    step=1,
)

months = pd.date_range(
    start=pd.to_datetime(start),
    end=pd.to_datetime(end),
    freq="MS",
).strftime("%Y-%m")

manual = DEFAULT_EXITS.copy()

if mode == "Enter different exits by month":
    st.sidebar.subheader("Edit Monthly Exits")

    exit_df = pd.DataFrame(
        {
            "Month": months,
            "Exits": [DEFAULT_EXITS.get(month, 79) for month in months],
        }
    )

    edited = st.sidebar.data_editor(
        exit_df,
        width="stretch",
        hide_index=True,
        disabled=["Month"],
    )

    manual = dict(zip(edited["Month"], edited["Exits"].astype(int)))

elif mode == "Use default monthly exits":
    manual = DEFAULT_EXITS.copy()

else:
    manual = {month: int(same_exit) for month in months}


medium_buffer = st.sidebar.number_input(
    "Medium Risk if remaining capacity is below:",
    min_value=0,
    value=50,
    step=1,
)

high_buffer = st.sidebar.number_input(
    "High Risk if remaining capacity is below:",
    min_value=0,
    value=20,
    step=1,
)

run = st.sidebar.button("Run Agent")


try:
    df_data = pd.read_csv("monthly_occupancy.csv")
except FileNotFoundError:
    st.error("monthly_occupancy.csv was not found. Put it in the same folder as app.py.")
    st.stop()


st.subheader("Current Exit Assumptions")

if mode == "Use same exits for all months":
    assumption_df = pd.DataFrame(
        {
            "Month": months,
            "Exits": [same_exit] * len(months),
        }
    )
else:
    assumption_df = pd.DataFrame(
        {
            "Month": months,
            "Exits": [manual.get(month, 79) for month in months],
        }
    )

st.dataframe(assumption_df, width="stretch")


if run:
    historical_df, forecast_summary, train_r2, test_r2 = run_forecast(
        df_raw=df_data,
        start_date=start,
        end_date=end,
        exit_mode=mode,
        same_exit=same_exit,
        manual_exit_values=manual,
        medium_buffer=medium_buffer,
        high_buffer=high_buffer,
    )

    forecast_summary.to_csv("forecast_output.csv", index=False)

    worst = forecast_summary.loc[
        forecast_summary["Remaining_Capacity"].idxmin()
    ]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Train R²", round(train_r2, 4))
    col2.metric("Test R²", round(test_r2, 4))
    col3.metric("Highest Risk Month", worst["Date"].strftime("%b %Y"))
    col4.metric("Highest Risk Level", worst["Risk_Level"])

    st.subheader("Key Insight")

    st.write(
        f"""
        The highest-risk month in this scenario is **{worst['Date'].strftime('%B %Y')}**.  
        Forecasted occupancy is **{worst['Forecasted_Occupancy']:.0f}** against capacity **{worst['Capacity']:.0f}**.  
        Remaining capacity is **{worst['Remaining_Capacity']:.0f}**, and the risk level is **{worst['Risk_Level']}**.
        """
    )

    st.subheader("AI Capacity Agent")

    with st.container(border=True):
        st.markdown("### Capacity Risk Briefing")

        a1, a2, a3, a4 = st.columns(4)

        a1.metric("Highest Risk Month", worst["Date"].strftime("%b %Y"))
        a2.metric("Forecasted Occupancy", f"{worst['Forecasted_Occupancy']:.0f}")
        a3.metric("Capacity", f"{worst['Capacity']:.0f}")
        a4.metric("Risk Level", worst["Risk_Level"])

        st.markdown("#### Agent Summary")

        st.write(
            f"""
            The agent reviewed the selected forecast period, monthly exit assumptions, 
            projected occupancy, and available capacity.

            The highest-risk month is **{worst['Date'].strftime('%B %Y')}**.  
            Forecasted occupancy is **{worst['Forecasted_Occupancy']:.0f}** against capacity **{worst['Capacity']:.0f}**, 
            leaving **{worst['Remaining_Capacity']:.0f}** units available.
            """
        )

        st.markdown("#### Recommendation")

        if worst["Risk_Level"] in ["Critical", "High"]:
            st.error(worst["Recommendation"])
        elif worst["Risk_Level"] == "Medium":
            st.warning(worst["Recommendation"])
        else:
            st.success(worst["Recommendation"])

    st.subheader("Forecast Table")

    display_cols = [
        "Date",
        "Forecasted_Occupancy",
        "Injected_Exits",
        "Capacity",
        "Remaining_Capacity",
        "Risk_Level",
        "Recommendation",
    ]

    st.dataframe(forecast_summary[display_cols], width="stretch")

    csv = forecast_summary[display_cols].to_csv(index=False)

    st.download_button(
        label="Download Forecast Results",
        data=csv,
        file_name="shelter_capacity_forecast_results.csv",
        mime="text/csv",
    )

    st.subheader("Interactive Tableau Dashboard")

    st.info(
        "The DC Tableau server does not allow embedding inside Streamlit. "
        "Open the dashboard in a separate browser tab."
    )

    st.link_button(
        "Open Tableau Dashboard",
        TABLEAU_DASHBOARD_URL,
    )