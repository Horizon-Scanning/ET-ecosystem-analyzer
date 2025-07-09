import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np
import re
from streamlit_plotly_events import plotly_events  # NEW

st.set_page_config(page_title="Ecosystem Analyzer", layout="wide")

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    csv_path = Path(__file__).parent / "emerging_technologies_processed.csv"
    df = pd.read_csv(csv_path)
    # Ensure numeric columns are numeric
    df["Impact"] = pd.to_numeric(df["Impact"], errors="coerce")
    df["Tech_Count"] = pd.to_numeric(df["Tech_Count"], errors="coerce")
    df.dropna(subset=["Impact", "Tech_Count"], inplace=True)
    return df

df = load_data()

st.title("🕸️ Ecosystem Analyzer")

# ----- SIDEBAR FILTERS -----
with st.sidebar:
    st.header("Filter Settings")

    # OECD Research Area filter
    oecd_areas = sorted(df["OECD_Research_Area"].unique())
    selected_areas = st.multiselect("Research Area (OECD)", options=oecd_areas, default=[])

    # Year range filter (use the discrete ranges present)
    def _year_key(label: str) -> int:
        """Return the first integer found in the year-range label for sorting."""
        m = re.search(r"\d+", label)
        return int(m.group()) if m else 0

    all_year_ranges = sorted(df["Year_Range"].unique(), key=_year_key)
    selected_years = st.multiselect("Year Range", options=all_year_ranges, default=[])

    # Tech_Level filter
    all_tech_levels = sorted(df["Tech_Level"].dropna().unique())
    selected_tech_levels = st.multiselect("Tech Level", options=all_tech_levels, default=[])

    # Tech_Count filter slider
    min_tc, max_tc = int(df["Tech_Count"].min()), int(df["Tech_Count"].max())
    tc_range = st.slider("Tech_Count (size of icon)", min_value=min_tc, max_value=max_tc, value=(min_tc, max_tc))

# Apply filters
filtered_df = df[
    df["OECD_Research_Area"].isin(selected_areas)
    & df["Year_Range"].isin(selected_years)
    & df["Tech_Level"].isin(selected_tech_levels)
    & df["Tech_Count"].between(tc_range[0], tc_range[1])
].reset_index(drop=True)  # Reset index to ensure consecutive 0-based indexing

if filtered_df.empty:
    st.warning("No data matching the selected filters.")
    st.stop()

# Order TRL x Adoption categories for consistent angular placement
trl_order = [
    "TRL 1-2",
    "TRL 3-4",
    # "TRL 3-4 Innovators",
    "TRL 5-6",
    "TRL 5-6 Innovators",
    "TRL 5-6 Early Adopters",
    # "TRL 5-6 Early Majority",
    # "TRL 5-6 Late Majority",
    "TRL 7-8",
    "TRL 7-8 Innovators",
    "TRL 7-8 Early Adopters",
    "TRL 7-8 Early Majority",
    # "TRL 7-8 Late Majority",
    "TRL 9",
    "TRL 9 Innovators",
    "TRL 9 Early Adopters",
    "TRL 9 Early Majority",
    # "TRL 9 Late Majority",
]

# Build ordered list of TRL categories present, including any not in predefined list
present_cats = filtered_df["TRLxAdoption"].astype(str).unique()
trl_order_present = [c for c in trl_order if c in present_cats]
# Append any additional categories not captured (sorted for determinism)
trl_order_present += sorted([c for c in present_cats if c not in trl_order_present])

# ----- ANGLE (theta) WITH JITTER -----
step_deg = 360 / len(trl_order_present) if trl_order_present else 1
angle_map = {cat: i * step_deg for i, cat in enumerate(trl_order_present)}
np.random.seed(42)  # deterministic random jitter for repeatability
filtered_df = filtered_df.copy()
filtered_df["theta_deg"] = filtered_df["TRLxAdoption"].apply(
    lambda cat: angle_map[cat] + np.random.uniform(-step_deg * 0.3, step_deg * 0.3)
)

# Polar scatter plot with reversed radial axis (Impact 100 -> center)
fig = px.scatter_polar(
    filtered_df,
    r="Impact",
    theta="theta_deg",
    color="OECD_Research_Area",
    size="Tech_Count",
    # hover_name="Technology_Name",
    size_max=40,
    color_discrete_sequence=px.colors.qualitative.Safe,
)

# Enable selection on the chart
fig.update_layout(clickmode='event+select')

# Custom angular ticks to display TRL categories
fig.update_layout(
    polar=dict(
        radialaxis=dict(showticklabels=False, ticks=""),
        angularaxis=dict(
            direction="clockwise",
            tickmode="array",
            tickvals=list(angle_map.values()),
            ticktext=list(angle_map.keys()),
        ),
    ),
    legend_title_text="OECD Research Area",
    margin=dict(l=40, r=40, t=40, b=40),
    height=850,
)

st.caption("Lower Impact values are closer to the center of the chart (range 0-100).  \nIcon size reflects the number of technologies (Tech_Count).", unsafe_allow_html=False)

# ----- VISUALISATION & RESULTS -----
# Use streamlit-plotly-events to capture both single-click and lasso/box selections
selected_points = plotly_events(
    fig,
    click_event=True,   # enable click on a single marker
    select_event=True,  # enable box/lasso select
    override_height=850,  # keep the same height as the plot
    key="ecosystem_plot",
)

st.subheader("Results")

# Filter results based on selected points
results_df = filtered_df[["Technology_Name", "OECD_Research_Area", "Year_Range", "TRLxAdoption", "Impact", "Tech_Count"]].copy()

# Initialize session state for tracking selections
if "last_selected" not in st.session_state:
    st.session_state.last_selected = None
if "show_selected" not in st.session_state:
    st.session_state.show_selected = False

# Get the actual row indices from the plotly selection
selected_indices = []

if selected_points:
    for pt in selected_points:
        curve_num = pt.get("curveNumber")
        point_num = pt.get("pointNumber")
        
        # Only process if we have valid curve and point numbers
        if curve_num is not None and point_num is not None:
            # Get all unique OECD areas in the order they appear in the plot
            unique_areas = filtered_df["OECD_Research_Area"].unique()
            
            if curve_num < len(unique_areas):
                area = unique_areas[curve_num]
                # Get rows for this specific OECD area
                area_mask = filtered_df["OECD_Research_Area"] == area
                area_indices = filtered_df[area_mask].index.tolist()
                
                if point_num < len(area_indices):
                    selected_indices.append(area_indices[point_num])

# Handle toggle logic
current_selection = tuple(sorted(selected_indices)) if selected_indices else None

if current_selection:
    if st.session_state.last_selected == current_selection:
        # Same selection clicked again - toggle off
        st.session_state.show_selected = not st.session_state.show_selected
    else:
        # New selection - show it
        st.session_state.show_selected = True
        st.session_state.last_selected = current_selection
else:
    # No selection from plot
    st.session_state.show_selected = False
    st.session_state.last_selected = None

# Determine what to show in the table
if st.session_state.show_selected and selected_indices:
    # Use the exact selected rows
    results_df = filtered_df.loc[selected_indices, ["Technology_Name", "OECD_Research_Area", "Year_Range", "TRLxAdoption", "Impact", "Tech_Count"]]
    st.info(f"Showing {len(selected_indices)} selected technologies. Click again to clear or select others.")
else:
    # Show all filtered results
    results_df = filtered_df[["Technology_Name", "OECD_Research_Area", "Year_Range", "TRLxAdoption", "Impact", "Tech_Count"]]
    st.info("Click on chart bubbles to select specific technologies, or view all results below.")

st.dataframe(
    results_df.sort_values("Impact").reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)


