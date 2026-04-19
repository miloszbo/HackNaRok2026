from __future__ import annotations

import pandas as pd
import streamlit as st
from pathlib import Path


SUMMARY_PATH = Path("experiments") / "summary.csv"


# -----------------------------
# Data loading
# -----------------------------

@st.cache_data
def load_summary():
    if not SUMMARY_PATH.exists():
        return pd.DataFrame(columns=[
            "Week", "Algorithm", "SuccessRate", "Encounters", "GlobalEncounters"
        ])

    df = pd.read_csv(
        SUMMARY_PATH,
        on_bad_lines="skip"  # 🔥 tolerate mixed schemas
    )

    if df.empty:
        return df

    # Ensure required columns exist
    for col in ["Week", "Algorithm", "SuccessRate", "Encounters"]:
        if col not in df.columns:
            df[col] = None

    # Ensure GlobalEncounters exists
    if "GlobalEncounters" not in df.columns:
        df["GlobalEncounters"] = df["Encounters"]

    # Convert to numeric safely
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce")
    df["SuccessRate"] = pd.to_numeric(df["SuccessRate"], errors="coerce")
    df["Encounters"] = pd.to_numeric(df["Encounters"], errors="coerce")
    df["GlobalEncounters"] = pd.to_numeric(df["GlobalEncounters"], errors="coerce")

    # Drop completely invalid rows
    df = df.dropna(subset=["Week", "Algorithm"])

    return df


# -----------------------------
# Processing
# -----------------------------

def total_real_encounters(df: pd.DataFrame):
    if df.empty:
        return 0

    return (
        df.drop_duplicates(subset=["Week"])["GlobalEncounters"]
        .fillna(0)
        .sum()
    )


def compute_cumulative(df: pd.DataFrame):
    if df.empty:
        return df

    df = df.sort_values(["Algorithm", "Week"]).copy()

    # 🔥 safer cumulative calculation (no groupby.apply)
    df["WeightedSuccess"] = df["SuccessRate"] * df["Encounters"]

    df["CumSuccess"] = (
        df.groupby("Algorithm")["WeightedSuccess"].cumsum()
        / df.groupby("Algorithm")["Encounters"].cumsum()
    )

    df.drop(columns=["WeightedSuccess"], inplace=True)

    return df


def latest_snapshot(df: pd.DataFrame):
    if df.empty:
        return df

    latest_week = df["Week"].max()
    snap = df[df["Week"] == latest_week].copy()

    snap["SuccessRate"] = (snap["SuccessRate"] * 100).round(2)

    return snap.sort_values("SuccessRate", ascending=False)


def total_encounters_by_algo(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(columns=["Algorithm", "TotalEncounters"])

    return (
        df.groupby("Algorithm")["Encounters"]
        .sum()
        .reset_index()
        .rename(columns={"Encounters": "TotalEncounters"})
        .sort_values("TotalEncounters", ascending=False)
    )


# -----------------------------
# UI
# -----------------------------

def render():
    st.set_page_config(page_title="Boar Algorithms Dashboard", layout="wide")

    st.title("🐗 Algorithm Comparison Dashboard")

    df = load_summary()

    if df.empty:
        st.warning("No data yet. Run simulation first.")
        return

    # Algorithm selector
    algorithms = sorted(df["Algorithm"].dropna().unique())

    selected_algos = st.multiselect(
        "Select algorithms",
        algorithms,
        default=algorithms
    )

    df = df[df["Algorithm"].isin(selected_algos)]

    df = compute_cumulative(df)

    # -----------------------------
    # Top Metrics
    # -----------------------------
    st.subheader("Overview")

    col1, col2 = st.columns(2)

    col1.metric("Weeks Simulated", int(df["Week"].max()))
    col2.metric("Total Records", len(df))

    st.divider()

    # -----------------------------
    # CUMULATIVE CHART
    # -----------------------------
    st.subheader("Cumulative Success Rate (%)")

    pivot_cum = df.pivot_table(
        index="Week",
        columns="Algorithm",
        values="CumSuccess",
        aggfunc="mean"
    )

    st.line_chart(pivot_cum, use_container_width=True)

    # -----------------------------
    # WEEKLY CHART
    # -----------------------------
    st.subheader("Weekly Success Rate (%)")

    pivot_week = df.pivot_table(
        index="Week",
        columns="Algorithm",
        values="SuccessRate",
        aggfunc="mean"
    )

    st.line_chart(pivot_week, use_container_width=True)

    # -----------------------------
    # Snapshot
    # -----------------------------
    st.subheader("Latest Week Snapshot")

    snap = latest_snapshot(df)
    st.dataframe(snap, use_container_width=True)

    # -----------------------------
    # Total encounters (per algo)
    # -----------------------------
    st.subheader("Total Encounters per Algorithm")

    totals = total_encounters_by_algo(df)
    st.dataframe(totals, use_container_width=True)

    # -----------------------------
    # Global real encounters
    # -----------------------------
    st.subheader("🌍 Real Total Encounters (Global)")

    real_total = total_real_encounters(df)
    st.metric("Actual Encounters", int(real_total))

    # -----------------------------
    # Raw data
    # -----------------------------
    with st.expander("Show raw data"):
        st.dataframe(df, use_container_width=True)


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":
    render()