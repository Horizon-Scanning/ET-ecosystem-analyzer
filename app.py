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


# -----------------------------------------------------------------------
# Additional dataset – used for per-technology selection table
# -----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_separated_data() -> pd.DataFrame:
    """Load the separated technologies CSV (contains individual Technology_Name rows)."""
    csv_path = Path(__file__).parent / "emerging_technologies_separated.csv"
    return pd.read_csv(csv_path)


sep_df = load_separated_data()

df = load_data()

st.title("🕸️ Ecosystem Analyzer")

# ----- SIDEBAR FILTERS -----
with st.sidebar:
    st.header("Filter Settings")

    # OECD Research Area filter with color mapping
    oecd_areas = sorted(df["OECD_Research_Area"].unique())
    
    # Create color mapping for each research area
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d3", "#c7c7c7", "#dbdb8d", "#9edae5"
    ]
    
    area_color_map = {area: color_palette[i % len(color_palette)] for i, area in enumerate(oecd_areas)}
    
    # Generate CSS to color the selected items in multiselect
    css_rules = []
    for area in oecd_areas:
        color = area_color_map[area]
        # Create CSS rule for each area to color its selection tag
        css_rules.append(f"""
        div[data-testid="stMultiSelect"] span[title="{area}"] {{
            background-color: {color} !important;
        }}
        """)
    
    # Apply the CSS
    st.markdown(f"<style>{''.join(css_rules)}</style>", unsafe_allow_html=True)
    
    # Regular multiselect without colored squares
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
# Create color mapping for the filtered data
unique_areas_in_filtered = filtered_df["OECD_Research_Area"].unique()
color_sequence = [area_color_map[area] for area in unique_areas_in_filtered]

fig = px.scatter_polar(
    filtered_df,
    r="Impact",
    theta="theta_deg",
    color="OECD_Research_Area",
    size="Tech_Count",
    # hover_name="Technology_Name",
    size_max=40,
    color_discrete_sequence=color_sequence,
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
            tickfont=dict(size=14)  # Increased font size for angular axis labels
        ),
    ),
    legend_title_text="OECD Research Area",
    legend_title_font_size=14,  # Increased legend title font size
    legend_font_size=12,  # Increased legend item font size
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

# Filter results based on selected points
results_df = filtered_df[["OECD_Research_Area", "Year_Range", "TRLxAdoption", "Impact", "Tech_Count"]].copy()  # "Technology_Name",

# Initialize session state for tracking selections
if "last_selected" not in st.session_state:
    st.session_state.last_selected = None
if "show_selected" not in st.session_state:
    st.session_state.show_selected = False
if "persisted_selected_indices" not in st.session_state:
    st.session_state.persisted_selected_indices = []

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
        if not st.session_state.show_selected:
            st.session_state.persisted_selected_indices = []
        else:
            st.session_state.persisted_selected_indices = selected_indices
    else:
        # New selection - show it
        st.session_state.show_selected = True
        st.session_state.last_selected = current_selection
        st.session_state.persisted_selected_indices = selected_indices
else:
    # No selection from plot - but check if we should persist previous selection
    if not st.session_state.persisted_selected_indices:
        st.session_state.show_selected = False
        st.session_state.last_selected = None

# Use persisted indices if available, otherwise use current selection
display_indices = st.session_state.persisted_selected_indices if st.session_state.persisted_selected_indices else selected_indices

# Show results only when bubbles are selected
if st.session_state.show_selected and display_indices:
    st.subheader("Results")
    # Use the exact selected rows - ensure indices are valid
    valid_indices = [idx for idx in display_indices if idx in filtered_df.index]
    results_df = filtered_df.loc[
        valid_indices,
        ["OECD_Research_Area", "Year_Range", "TRLxAdoption", "Impact", "Tech_Count"],
    ]  # "Technology_Name",
    st.info(
        f"Showing {len(valid_indices)} selected technologies. Click again to clear or select others."
    )
    st.dataframe(
        results_df.sort_values("Impact").reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    # ---------------------------------------------------------------
    # Build the per-technology table by matching attributes for each
    # selected bubble, then show only the Technology_Name column with
    # a checkbox on the left.
    # ---------------------------------------------------------------

    # Collect attribute combinations for the selected bubbles
    bubble_attrs = filtered_df.loc[
        valid_indices, [
            "OECD_Research_Area",
            "Year_Range",
            "Tech_Level",
            "TRLxAdoption",
        ]
    ]

    # Create a mask for rows in sep_df that match ANY of the selected attribute sets
    mask = pd.Series(False, index=sep_df.index)
    for _, row in bubble_attrs.iterrows():
        cond = (
            (sep_df["OECD_Research_Area"] == row["OECD_Research_Area"]) &
            (sep_df["Year_Range"] == row["Year_Range"]) &
            (sep_df["Tech_Level"] == row["Tech_Level"]) &
            (sep_df["TRLxAdoption"] == row["TRLxAdoption"])
        )
        mask |= cond

    tech_df = sep_df.loc[mask, ["Technology_Name"]].drop_duplicates().reset_index(drop=True)

    # Prepend a checkbox column (default unchecked)
    tech_df.insert(0, "Select", False)

    if not tech_df.empty:
        st.subheader("Technologies")
        edited_tech_df = st.data_editor(
            tech_df,
            use_container_width=False,
            hide_index=True,
            key="tech_selector",
            disabled=["Technology_Name"],
            on_change=None,
            column_config={
                "Select": st.column_config.CheckboxColumn(required=False),
                "Technology_Name": st.column_config.TextColumn(width="medium")
            },
        )
        
        # Generate Report button (doesn't trigger anything)
        if st.button("Generate Report", key="generate_report", type="primary"):
            pass  # Button doesn't do anything yet
else:
    st.info("Click on chart bubbles to select specific technologies.")


