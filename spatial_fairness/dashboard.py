"""
Streamlit Dashboard for Spatial Fairness Term Validation.

This dashboard provides an interactive interface for:
- Configuring spatial fairness computation parameters
- Visualizing Gini coefficients and Lorenz curves
- Analyzing temporal patterns in spatial fairness
- Exploring service rate distributions across the grid

Usage:
    streamlit run dashboard.py

Requirements:
    pip install streamlit pandas plotly matplotlib seaborn
"""

import sys
import os
from pathlib import Path
import pickle
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent
OBJECTIVE_FUNCTION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = OBJECTIVE_FUNCTION_DIR.parent
sys.path.insert(0, str(OBJECTIVE_FUNCTION_DIR))

from config import SpatialFairnessConfig
from term import SpatialFairnessTerm
from utils import (
    compute_gini,
    compute_lorenz_curve,
    aggregate_counts_by_period,
    get_unique_periods,
    get_data_statistics,
    validate_pickup_dropoff_data,
)
# from spatial_fairness.config import SpatialFairnessConfig
# from spatial_fairness.term import SpatialFairnessTerm
# from spatial_fairness.utils import (
#     compute_gini,
#     compute_lorenz_curve,
#     aggregate_counts_by_period,
#     get_unique_periods,
#     get_data_statistics,
#     validate_pickup_dropoff_data,
# )


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Spatial Fairness Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data(filepath: str) -> Dict:
    """Load and cache pickup/dropoff counts data."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_default_data_path() -> Optional[str]:
    """Find the default data file path."""
    possible_paths = [
        PROJECT_ROOT / "source_data" / "pickup_dropoff_counts.pkl",
        Path("source_data/pickup_dropoff_counts.pkl"),
        Path("../source_data/pickup_dropoff_counts.pkl"),
        Path("../../source_data/pickup_dropoff_counts.pkl"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def plot_lorenz_curve(dsr_values: np.ndarray, asr_values: np.ndarray, title: str = "Lorenz Curves") -> go.Figure:
    """Create a Lorenz curve plot for DSR and ASR."""
    from spatial_fairness.utils import compute_lorenz_curve
    
    x_dsr, y_dsr = compute_lorenz_curve(dsr_values)
    x_asr, y_asr = compute_lorenz_curve(asr_values)
    
    fig = go.Figure()
    
    # Perfect equality line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Equality',
        line=dict(dash='dash', color='gray'),
    ))
    
    # Departure (Pickup) Lorenz curve
    fig.add_trace(go.Scatter(
        x=x_dsr, y=y_dsr,
        mode='lines',
        name=f'Pickups (Gini={compute_gini(dsr_values):.3f})',
        line=dict(color='blue'),
    ))
    
    # Arrival (Dropoff) Lorenz curve
    fig.add_trace(go.Scatter(
        x=x_asr, y=y_asr,
        mode='lines',
        name=f'Dropoffs (Gini={compute_gini(asr_values):.3f})',
        line=dict(color='red'),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Cumulative Share of Grid Cells",
        yaxis_title="Cumulative Share of Service",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=400,
    )
    
    return fig


def plot_gini_over_time(per_period_data: list, period_type: str) -> go.Figure:
    """Plot Gini coefficients over time periods."""
    df = pd.DataFrame(per_period_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df['gini_arrival'],
        mode='lines+markers',
        name='Gini (Dropoffs)',
        line=dict(color='red'),
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df['gini_departure'],
        mode='lines+markers',
        name='Gini (Pickups)',
        line=dict(color='blue'),
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df['fairness'],
        mode='lines+markers',
        name='Spatial Fairness',
        line=dict(color='green', width=3),
    ))
    
    fig.update_layout(
        title=f"Gini Coefficients and Fairness Over Time ({period_type})",
        xaxis_title="Period Index",
        yaxis_title="Value",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=400,
    )
    
    return fig


def plot_service_heatmap(grid_data: np.ndarray, title: str) -> go.Figure:
    """Create a heatmap of service distribution."""
    fig = px.imshow(
        grid_data.T,  # Transpose to show y on vertical axis
        labels=dict(x="X Grid", y="Y Grid", color="Count"),
        title=title,
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    
    fig.update_layout(height=500)
    return fig


def plot_gini_distribution(gini_values: list, title: str) -> go.Figure:
    """Plot histogram of Gini coefficients across periods."""
    fig = px.histogram(
        x=gini_values,
        nbins=30,
        labels={"x": "Gini Coefficient", "y": "Count"},
        title=title,
    )
    
    fig.add_vline(
        x=np.mean(gini_values),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {np.mean(gini_values):.3f}",
    )
    
    fig.update_layout(height=300)
    return fig


def plot_hourly_pattern(per_period_data: list) -> go.Figure:
    """Plot hourly pattern of spatial fairness (for hourly aggregation)."""
    df = pd.DataFrame(per_period_data)
    
    if 'period' not in df.columns:
        return None
    
    # Extract hour from period if it's hourly
    try:
        hours = [p[0] if isinstance(p, tuple) else p for p in df['period']]
        df['hour'] = hours
        
        # Group by hour across days
        hourly_avg = df.groupby('hour').agg({
            'fairness': 'mean',
            'gini_arrival': 'mean',
            'gini_departure': 'mean',
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hourly_avg['hour'],
            y=hourly_avg['fairness'],
            mode='lines+markers',
            name='Spatial Fairness',
            line=dict(color='green', width=3),
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_avg['hour'],
            y=hourly_avg['gini_arrival'],
            mode='lines',
            name='Gini (Dropoffs)',
            line=dict(color='red', dash='dot'),
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_avg['hour'],
            y=hourly_avg['gini_departure'],
            mode='lines',
            name='Gini (Pickups)',
            line=dict(color='blue', dash='dot'),
        ))
        
        fig.update_layout(
            title="Average Spatial Fairness by Hour of Day",
            xaxis_title="Hour",
            yaxis_title="Value",
            xaxis=dict(tickmode='linear', dtick=2),
            height=400,
        )
        
        return fig
    except Exception:
        return None


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.title("📊 Spatial Fairness Term Dashboard")
    st.markdown("""
    This dashboard helps validate and analyze the **Spatial Fairness Term** ($F_{\\text{spatial}}$)
    of the FAMAIL objective function. The term measures equality of taxi service distribution
    using the Gini coefficient.
    
    $$F_{\\text{spatial}} = 1 - \\frac{1}{2|P|} \\sum_{p \\in P} (G_a^p + G_d^p)$$
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Data file selection
    default_path = get_default_data_path()
    data_path = st.sidebar.text_input(
        "Data File Path",
        value=default_path or "source_data/pickup_dropoff_counts.pkl",
        help="Path to pickup_dropoff_counts.pkl file"
    )
    
    if not os.path.exists(data_path):
        st.error(f"Data file not found: {data_path}")
        st.info("Please provide a valid path to pickup_dropoff_counts.pkl")
        return
    
    # Load data
    try:
        data = load_data(data_path)
        st.sidebar.success(f"✅ Loaded {len(data):,} records")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Configuration options
    st.sidebar.subheader("Term Configuration")
    
    period_type = st.sidebar.selectbox(
        "Period Type",
        options=["hourly", "daily", "time_bucket", "all"],
        index=0,
        help="Temporal aggregation granularity"
    )
    
    include_zero_cells = st.sidebar.checkbox(
        "Include Zero Cells",
        value=False,
        help="Include cells with no activity in Gini calculation"
    )
    
    num_taxis = st.sidebar.number_input(
        "Number of Taxis",
        min_value=1,
        max_value=1000,
        value=50,
        help="Number of active taxis in dataset"
    )
    
    num_days = st.sidebar.number_input(
        "Number of Days",
        min_value=1.0,
        max_value=365.0,
        value=21.0,
        step=1.0,
        help="Number of days in dataset"
    )
    
    # Day filter
    st.sidebar.subheader("Filters")
    day_options = [1, 2, 3, 4, 5, 6]
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    selected_days = st.sidebar.multiselect(
        "Days to Include",
        options=day_options,
        default=day_options,
        format_func=lambda x: day_labels[x-1],
    )
    
    days_filter = selected_days if len(selected_days) < 6 else None
    
    # Create configuration
    config = SpatialFairnessConfig(
        period_type=period_type,
        grid_dims=(48, 90),
        num_taxis=num_taxis,
        num_days=num_days,
        include_zero_cells=include_zero_cells,
        days_filter=days_filter,
        verbose=False,
    )
    
    # Create term
    term = SpatialFairnessTerm(config)
    
    # Compute results
    with st.spinner("Computing spatial fairness..."):
        breakdown = term.compute_with_breakdown({}, {'pickup_dropoff_counts': data})
    
    # ==========================================================================
    # RESULTS DISPLAY
    # ==========================================================================
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Spatial Fairness",
            f"{breakdown['value']:.4f}",
            help="Higher is better (1.0 = perfect equality)"
        )
    
    with col2:
        st.metric(
            "Avg Gini (Pickups)",
            f"{breakdown['components']['avg_gini_departure']:.4f}",
            help="Inequality in pickup distribution"
        )
    
    with col3:
        st.metric(
            "Avg Gini (Dropoffs)",
            f"{breakdown['components']['avg_gini_arrival']:.4f}",
            help="Inequality in dropoff distribution"
        )
    
    with col4:
        st.metric(
            "Periods Analyzed",
            f"{breakdown['statistics']['n_periods']}",
            help="Number of time periods"
        )
    
    # Computation time
    st.caption(f"⏱️ Computation time: {breakdown['diagnostics']['computation_time_ms']:.1f} ms")
    
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Temporal Analysis",
        "🗺️ Spatial Distribution",
        "📊 Lorenz Curves",
        "📉 Statistics",
        "🔍 Raw Data",
    ])
    
    # Tab 1: Temporal Analysis
    with tab1:
        st.subheader("Temporal Patterns")
        
        per_period_data = breakdown['components']['per_period_data']
        
        if len(per_period_data) > 1:
            # Gini over time
            fig_time = plot_gini_over_time(per_period_data, period_type)
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Hourly pattern (if applicable)
            if period_type == "hourly":
                fig_hourly = plot_hourly_pattern(per_period_data)
                if fig_hourly:
                    st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Gini distributions
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist_pickup = plot_gini_distribution(
                    breakdown['components']['per_period_gini_departure'],
                    "Distribution of Pickup Gini Coefficients"
                )
                st.plotly_chart(fig_hist_pickup, use_container_width=True)
            
            with col2:
                fig_hist_dropoff = plot_gini_distribution(
                    breakdown['components']['per_period_gini_arrival'],
                    "Distribution of Dropoff Gini Coefficients"
                )
                st.plotly_chart(fig_hist_dropoff, use_container_width=True)
        else:
            st.info("Only one period available. Select finer temporal granularity for temporal analysis.")
    
    # Tab 2: Spatial Distribution
    with tab2:
        st.subheader("Spatial Distribution of Service")
        
        # Get heatmap data
        heatmap_data = term.get_spatial_heatmap_data(data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pickup = plot_service_heatmap(heatmap_data['pickups'], "Pickup Distribution")
            st.plotly_chart(fig_pickup, use_container_width=True)
        
        with col2:
            fig_dropoff = plot_service_heatmap(heatmap_data['dropoffs'], "Dropoff Distribution")
            st.plotly_chart(fig_dropoff, use_container_width=True)
        
        # Total service heatmap
        fig_total = plot_service_heatmap(heatmap_data['total'], "Total Service (Pickups + Dropoffs)")
        st.plotly_chart(fig_total, use_container_width=True)
        
        # Grid statistics
        st.subheader("Grid Statistics")
        active_cells = np.sum(heatmap_data['total'] > 0)
        total_cells = heatmap_data['total'].size
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Cells", f"{active_cells:,}")
        with col2:
            st.metric("Total Cells", f"{total_cells:,}")
        with col3:
            st.metric("Activity Rate", f"{100*active_cells/total_cells:.1f}%")
    
    # Tab 3: Lorenz Curves
    with tab3:
        st.subheader("Lorenz Curves")
        
        st.markdown("""
        The **Lorenz curve** shows the cumulative distribution of service across grid cells.
        The further the curve bows from the diagonal (perfect equality), the higher the inequality.
        The **Gini coefficient** equals twice the area between the curve and the diagonal.
        """)
        
        # Select period for Lorenz curve
        per_period_data = breakdown['components']['per_period_data']
        
        if len(per_period_data) > 1:
            period_idx = st.slider(
                "Select Period",
                0, len(per_period_data) - 1, 0,
                help="Choose a specific time period to visualize"
            )
            
            selected_period = per_period_data[period_idx]
        else:
            selected_period = per_period_data[0] if per_period_data else None
        
        if selected_period:
            # Get data for this period
            period_result = term.compute_for_single_period(data, selected_period['period'])
            
            # Show metrics for this period
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Period Fairness", f"{period_result['fairness']:.4f}")
            with col2:
                st.metric("Pickup Gini", f"{period_result['gini_departure']:.4f}")
            with col3:
                st.metric("Dropoff Gini", f"{period_result['gini_arrival']:.4f}")
            
            # Lorenz curves
            fig_lorenz = plot_lorenz_curve(
                period_result['dsr_values'],
                period_result['asr_values'],
                f"Lorenz Curves for Period {selected_period['period']}"
            )
            st.plotly_chart(fig_lorenz, use_container_width=True)
        
        # Overall Lorenz curve (aggregated)
        st.subheader("Overall Lorenz Curves (Aggregated)")
        
        # Aggregate all data
        pickups_agg, dropoffs_agg = aggregate_counts_by_period(data, period_type="all")
        periods = get_unique_periods(pickups_agg, dropoffs_agg)
        
        if periods:
            from spatial_fairness.utils import compute_service_rates_for_period, compute_period_duration_days
            
            period = periods[0]
            period_duration = compute_period_duration_days(period, "all", num_days)
            
            dsr_all, asr_all = compute_service_rates_for_period(
                pickups_agg, dropoffs_agg, period,
                config.grid_dims, num_taxis, period_duration,
                include_zero_cells, True, 0
            )
            
            fig_lorenz_all = plot_lorenz_curve(dsr_all, asr_all, "Overall Lorenz Curves (All Data)")
            st.plotly_chart(fig_lorenz_all, use_container_width=True)
    
    # Tab 4: Statistics
    with tab4:
        st.subheader("Detailed Statistics")
        
        # Gini statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Pickup (Departure) Gini Statistics**")
            pickup_stats = breakdown['statistics']['gini_departure_stats']
            st.json(pickup_stats)
        
        with col2:
            st.markdown("**Dropoff (Arrival) Gini Statistics**")
            dropoff_stats = breakdown['statistics']['gini_arrival_stats']
            st.json(dropoff_stats)
        
        # Data statistics
        st.markdown("**Data Statistics**")
        data_stats = breakdown['statistics']['data_stats']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Pickup Statistics**")
            st.json(data_stats.get('pickup_stats', {}))
        
        with col2:
            st.markdown("**Dropoff Statistics**")
            st.json(data_stats.get('dropoff_stats', {}))
        
        with col3:
            st.markdown("**Spatial Coverage**")
            st.json(data_stats.get('spatial', {}))
        
        # Configuration used
        st.markdown("**Configuration Used**")
        st.json(breakdown['diagnostics']['config'])
    
    # Tab 5: Raw Data
    with tab5:
        st.subheader("Per-Period Data")
        
        per_period_data = breakdown['components']['per_period_data']
        
        # Convert to DataFrame
        df = pd.DataFrame(per_period_data)
        
        # Format period column for display
        if 'period' in df.columns:
            df['period_str'] = df['period'].astype(str)
        
        # Select columns to display
        display_cols = ['period', 'fairness', 'gini_arrival', 'gini_departure', 'n_cells']
        available_cols = [c for c in display_cols if c in df.columns]
        
        st.dataframe(
            df[available_cols].style.format({
                'fairness': '{:.4f}',
                'gini_arrival': '{:.4f}',
                'gini_departure': '{:.4f}',
            }),
            use_container_width=True,
            height=400,
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="spatial_fairness_results.csv",
            mime="text/csv",
        )
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **FAMAIL Spatial Fairness Dashboard** | Version 1.0.0  
    Based on Su et al. (2018) "Uncovering Spatial Inequality in Taxi Services"
    """)


if __name__ == "__main__":
    main()
