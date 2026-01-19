# File 9: dashboard/app.py
"""
Streamlit Dashboard for Project Vande - Aadhaar Analytics.

Launch with: streamlit run dashboard/app.py
"""

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src import config
from src.metrics import calculate_asi, detect_inclusion_risk, calculate_saturation_status
from src.models import detect_anomalies_in_dataframe, EnrolmentForecaster
from src.viz import (
    plot_asi_choropleth, plot_timeseries, plot_anomaly_scatter,
    plot_forecast, plot_inclusion_risk_map, generate_summary_table
)
from src.preprocessing import run_preprocessing_pipeline, load_processed


# Page config
st.set_page_config(
    page_title="Project Vande - Aadhaar Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'


def get_theme_css(theme: str) -> str:
    """Generate CSS for light or dark theme."""
    base_kpi_css = """
        .kpi-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
        }
        .kpi-card {
            padding: 25px 40px;
            border-radius: 12px;
            text-align: center;
            color: white;
            font-weight: bold;
            min-width: 180px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .kpi-card .value {
            font-size: 2.5rem;
            margin-bottom: 5px;
        }
        .kpi-card .label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        .kpi-orange { background: linear-gradient(135deg, #f39c12 0%, #e74c3c 100%); }
        .kpi-cyan { background: linear-gradient(135deg, #00d2d3 0%, #54a0ff 100%); }
        .kpi-purple { background: linear-gradient(135deg, #a55eea 0%, #8854d0 100%); }
        .kpi-green { background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%); }
        .kpi-yellow { background: linear-gradient(135deg, #fed330 0%, #f7b731 100%); }
        .section-header {
            font-size: 1.3rem;
            font-weight: 600;
            margin: 20px 0 10px 0;
            padding-bottom: 5px;
            border-bottom: 2px solid #667eea;
        }
    """
    
    if theme == 'dark':
        return f"""
        <style>
            :root {{
                --bg-primary: #1e1e1e;
                --bg-secondary: #2d2d2d;
                --text-primary: #ffffff;
                --text-secondary: #b0b0b0;
                --accent: #667eea;
            }}
            .main-header {{font-size: 2.5rem; font-weight: bold; color: #667eea;}}
            .metric-card {{background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                          padding: 20px; border-radius: 10px; color: white;}}
            .stMetric {{background-color: #2d2d2d; padding: 10px; border-radius: 5px; color: white;}}
            .stApp {{background-color: #1e1e1e;}}
            .stSidebar {{background-color: #2d2d2d;}}
            .stDataFrame {{background-color: #2d2d2d;}}
            div[data-testid="stMetricValue"] {{color: #ffffff;}}
            div[data-testid="stMetricLabel"] {{color: #b0b0b0;}}
            .stMarkdown, .stText, p, span, label {{color: #ffffff !important;}}
            {base_kpi_css}
        </style>
        """
    else:
        return f"""
        <style>
            .main-header {{font-size: 2.5rem; font-weight: bold; color: #1f77b4;}}
            .metric-card {{background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                          padding: 20px; border-radius: 10px; color: white;}}
            .stMetric {{background-color: #f0f2f6; padding: 10px; border-radius: 5px;}}
            {base_kpi_css}
        </style>
        """


@st.cache_data(show_spinner="Loading data...")
def load_data_from_parquet(uploaded_file=None):
    """Load data from uploaded parquet or default path."""
    if uploaded_file is not None:
        return pd.read_parquet(uploaded_file)
    elif config.MERGED_DATA_FILE.exists():
        return pd.read_parquet(config.MERGED_DATA_FILE)
    return None


@st.cache_data(show_spinner="Processing CSV files from raw data...")
def load_data_from_raw_csvs():
    """Load and process data from default CSV directories."""
    try:
        # Check if CSV directories exist
        if not config.ENROLMENT_DIR.exists():
            st.error(f"Enrolment data folder not found: {config.ENROLMENT_DIR}")
            return None
        if not config.DEMOGRAPHIC_UPDATE_DIR.exists():
            st.error(f"Demographic data folder not found: {config.DEMOGRAPHIC_UPDATE_DIR}")
            return None
        if not config.BIOMETRIC_UPDATE_DIR.exists():
            st.error(f"Biometric data folder not found: {config.BIOMETRIC_UPDATE_DIR}")
            return None
        
        # Run the preprocessing pipeline
        df = run_preprocessing_pipeline(
            enrol_path=config.ENROLMENT_DIR,
            demo_path=config.DEMOGRAPHIC_UPDATE_DIR,
            bio_path=config.BIOMETRIC_UPDATE_DIR,
            output_path=config.MERGED_DATA_FILE
        )
        return df
    except Exception as e:
        st.error(f"Error processing CSV files: {str(e)}")
        return None


def ensure_asi_scores(df):
    """Calculate ASI if not present."""
    if 'asi_score' not in df.columns:
        st.info("Calculating ASI scores...")
        asi_scores = calculate_asi(df, district=None, include_national=False)
        asi_df = pd.DataFrame(list(asi_scores.items()), columns=['district', 'asi_score'])
        df = df.merge(asi_df, on='district', how='left')
        df['asi_score'] = df['asi_score'].fillna(50)
    return df


def ensure_anomalies(df):
    """Detect anomalies if not present."""
    if 'is_anomaly' not in df.columns:
        st.info("Detecting anomalies...")
        features = [f for f in config.ANOMALY_FEATURES if f in df.columns]
        df = detect_anomalies_in_dataframe(df, features)
    return df


def export_csv(df):
    """Export dataframe to CSV bytes."""
    return df.to_csv(index=False).encode('utf-8')


def export_pdf_report(df):
    """Generate PDF report and return bytes."""
    from src.viz import generate_pdf_report
    output_path = config.OUTPUTS_DIR / "dashboard_report.pdf"
    generate_pdf_report(df, output_path, "Project Vande Dashboard Report")
    with open(output_path, 'rb') as f:
        return f.read()


def main():
    # Apply theme CSS
    st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">📊 Project Vande - Aadhaar Analytics</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Theme Toggle
    st.sidebar.markdown("### 🎨 Theme")
    theme_toggle = st.sidebar.toggle(
        "🌙 Dark Mode",
        value=st.session_state.theme == 'dark',
        help="Toggle between light and dark themes"
    )
    if theme_toggle and st.session_state.theme != 'dark':
        st.session_state.theme = 'dark'
        st.rerun()
    elif not theme_toggle and st.session_state.theme != 'light':
        st.session_state.theme = 'light'
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Data Source Selection
    st.sidebar.markdown("### 📁 Data Source")
    data_source = st.sidebar.radio(
        "Select data source:",
        ["📂 Load from CSV Folders (Default)", "📤 Upload Parquet File"],
        help="Choose to load from default CSV folders in data/raw or upload a processed parquet file"
    )
    
    df = None
    
    if data_source == "📂 Load from CSV Folders (Default)":
        st.sidebar.info(f"Loading from:\n• {config.ENROLMENT_DIR.name}\n• {config.DEMOGRAPHIC_UPDATE_DIR.name}\n• {config.BIOMETRIC_UPDATE_DIR.name}")
        
        if st.sidebar.button("🔄 Load/Refresh Data", type="primary"):
            # Clear cache to force reload
            load_data_from_raw_csvs.clear()
            st.rerun()
        
        # Try to load processed data first, fall back to CSV processing
        if config.MERGED_DATA_FILE.exists():
            df = load_data_from_parquet()
            st.sidebar.success("✅ Loaded from cached parquet")
        else:
            df = load_data_from_raw_csvs()
            if df is not None:
                st.sidebar.success("✅ Processed CSVs successfully")
    else:
        # Parquet file upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload Parquet File",
            type=['parquet'],
            help="Upload a processed merged_data.parquet file"
        )
        if uploaded_file is not None:
            df = load_data_from_parquet(uploaded_file)
    
    st.sidebar.markdown("---")
    
    if df is None:
        st.warning("⚠️ No data loaded. Either:\n1. Click 'Load/Refresh Data' to process CSVs from data/raw\n2. Upload a parquet file")
        
        # Show directory status
        st.subheader("📊 Data Directory Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            enrol_exists = config.ENROLMENT_DIR.exists()
            csv_count = len(list(config.ENROLMENT_DIR.glob('*.csv'))) if enrol_exists else 0
            if enrol_exists and csv_count > 0:
                st.success(f"✅ Enrolment: {csv_count} CSVs")
            else:
                st.error(f"❌ Enrolment folder missing")
        with col2:
            demo_exists = config.DEMOGRAPHIC_UPDATE_DIR.exists()
            csv_count = len(list(config.DEMOGRAPHIC_UPDATE_DIR.glob('*.csv'))) if demo_exists else 0
            if demo_exists and csv_count > 0:
                st.success(f"✅ Demographic: {csv_count} CSVs")
            else:
                st.error(f"❌ Demographic folder missing")
        with col3:
            bio_exists = config.BIOMETRIC_UPDATE_DIR.exists()
            csv_count = len(list(config.BIOMETRIC_UPDATE_DIR.glob('*.csv'))) if bio_exists else 0
            if bio_exists and csv_count > 0:
                st.success(f"✅ Biometric: {csv_count} CSVs")
            else:
                st.error(f"❌ Biometric folder missing")
        
        st.stop()
    
    # Ensure required columns
    df['date'] = pd.to_datetime(df['date'])
    df = ensure_asi_scores(df)
    df = ensure_anomalies(df)
    
    # Sidebar filters
    st.sidebar.markdown("### 🔍 Filters")
    
    states = sorted(df['state'].unique())
    selected_state = st.sidebar.selectbox("State", ["All States"] + states)
    
    if selected_state != "All States":
        districts = sorted(df[df['state'] == selected_state]['district'].unique())
    else:
        districts = sorted(df['district'].unique())
    
    selected_district = st.sidebar.selectbox("District", ["All Districts"] + list(districts))
    
    # Date range
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Apply filters
    df_filtered = df.copy()
    if selected_state != "All States":
        df_filtered = df_filtered[df_filtered['state'] == selected_state]
    if selected_district != "All Districts":
        df_filtered = df_filtered[df_filtered['district'] == selected_district]
    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['date'].dt.date >= date_range[0]) &
            (df_filtered['date'].dt.date <= date_range[1])
        ]
    
    # Export buttons
    st.sidebar.markdown("### 📥 Export")
    
    csv_data = export_csv(df_filtered)
    st.sidebar.download_button(
        label="📄 Download CSV",
        data=csv_data,
        file_name="vande_data.csv",
        mime="text/csv"
    )
    
    if st.sidebar.button("📑 Generate PDF Report"):
        with st.spinner("Generating PDF..."):
            pdf_data = export_pdf_report(df_filtered)
            st.sidebar.download_button(
                label="📥 Download PDF",
                data=pdf_data,
                file_name="vande_report.pdf",
                mime="application/pdf"
            )
    
    # Main content
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_enrol = df_filtered[config.METRIC_ENROLMENT_TOTAL].sum()
        st.metric("Total Enrolments", f"{total_enrol:,.0f}")
    
    with col2:
        total_updates = df_filtered[config.METRIC_TOTAL_UPDATES].sum()
        st.metric("Total Updates", f"{total_updates:,.0f}")
    
    with col3:
        if selected_district != "All Districts":
            asi = df_filtered['asi_score'].mean()
        else:
            asi = df_filtered.groupby('district')['asi_score'].mean().mean()
        st.metric("Avg ASI Score", f"{asi:.1f}")
    
    with col4:
        anomaly_count = df_filtered['is_anomaly'].sum()
        st.metric("Anomalies Detected", f"{anomaly_count:,}")
    
    st.markdown("---")
    
    # Tabs - Data Overview first
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", "📈 Trends", "🗺️ ASI Map", "⚠️ Anomalies", "🎯 Risk Analysis", "🔮 Forecast"
    ])
    
    # ==================== DATA OVERVIEW TAB ====================
    with tab0:
        st.markdown('<p class="section-header">🏠 UIDAI Data Hackathon 2025 - Data Overview</p>', unsafe_allow_html=True)
        
        # Calculate totals
        total_enrol_sum = df_filtered[config.METRIC_ENROLMENT_TOTAL].sum()
        total_demo_sum = df_filtered.get(config.METRIC_DEMOGRAPHIC_UPDATES_TOTAL, pd.Series([0])).sum()
        total_bio_sum = df_filtered.get(config.METRIC_BIOMETRIC_UPDATES_TOTAL, pd.Series([0])).sum()
        
        # Format numbers
        def format_millions(n):
            if n >= 1_000_000_000:
                return f"{n/1_000_000_000:.1f}B"
            elif n >= 1_000_000:
                return f"{n/1_000_000:.1f}M"
            elif n >= 1_000:
                return f"{n/1_000:.1f}K"
            return f"{n:.0f}"
        
        # Hero KPI Cards
        st.markdown(f'''
        <div class="kpi-container">
            <div class="kpi-card kpi-orange">
                <div class="value">{format_millions(total_enrol_sum)}</div>
                <div class="label">Total Enrolments</div>
            </div>
            <div class="kpi-card kpi-cyan">
                <div class="value">{format_millions(total_demo_sum)}</div>
                <div class="label">Demographic Updates</div>
            </div>
            <div class="kpi-card kpi-purple">
                <div class="value">{format_millions(total_bio_sum)}</div>
                <div class="label">Biometric Updates</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Monthly data preparation
        df_monthly = df_filtered.copy()
        df_monthly['month'] = df_monthly['date'].dt.to_period('M').astype(str)
        
        # ---- Row 1: Enrollment Trend + Distribution Pie ----
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Month-wise Enrollment Trend**")
            # Aggregate by month for enrolment age groups
            enrol_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
            available_enrol = [c for c in enrol_cols if c in df_monthly.columns]
            
            if available_enrol:
                monthly_enrol = df_monthly.groupby('month')[available_enrol].sum().reset_index()
                fig_enrol = go.Figure()
                colors = ['#3498db', '#e74c3c', '#2ecc71']
                labels = ['Age 0-5', 'Age 5-17', 'Age 18+']
                for i, col in enumerate(available_enrol):
                    fig_enrol.add_trace(go.Scatter(
                        x=monthly_enrol['month'],
                        y=monthly_enrol[col],
                        mode='lines+markers',
                        name=labels[i] if i < len(labels) else col,
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=6)
                    ))
                fig_enrol.update_layout(
                    height=350,
                    template='plotly_white',
                    legend=dict(orientation='h', y=-0.2),
                    margin=dict(l=40, r=40, t=30, b=60),
                    xaxis_title='Month',
                    yaxis_title='Volume'
                )
                st.plotly_chart(fig_enrol, use_container_width=True)
            else:
                st.info("Enrollment age group columns not available")
        
        with col2:
            st.markdown("**Distribution of Total Activity**")
            # Distribution pie chart
            dist_data = {
                'Category': ['Enrolments', 'Demo Updates', 'Bio Updates'],
                'Value': [total_enrol_sum, total_demo_sum, total_bio_sum]
            }
            fig_pie = go.Figure(data=[go.Pie(
                labels=dist_data['Category'],
                values=dist_data['Value'],
                hole=0.5,
                marker=dict(colors=['#f39c12', '#00d2d3', '#a55eea']),
                textinfo='percent+label',
                textposition='outside'
            )])
            fig_pie.update_layout(
                height=350,
                showlegend=True,
                legend=dict(orientation='h', y=-0.1),
                margin=dict(l=20, r=20, t=30, b=60)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # ---- Row 2: Biometric Trend + Demographic Trend ----
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Month-wise Biometric Trend**")
            bio_cols = ['bio_age_5_17', 'bio_age_17_']
            available_bio = [c for c in bio_cols if c in df_monthly.columns]
            
            if available_bio:
                monthly_bio = df_monthly.groupby('month')[available_bio].sum().reset_index()
                fig_bio = go.Figure()
                bio_colors = ['#3498db', '#9b59b6']
                bio_labels = ['Bio Age 5-17', 'Bio Age 17+']
                for i, col in enumerate(available_bio):
                    fig_bio.add_trace(go.Scatter(
                        x=monthly_bio['month'],
                        y=monthly_bio[col],
                        mode='lines+markers',
                        name=bio_labels[i] if i < len(bio_labels) else col,
                        line=dict(color=bio_colors[i % len(bio_colors)], width=2),
                        marker=dict(size=6)
                    ))
                fig_bio.update_layout(
                    height=300,
                    template='plotly_white',
                    legend=dict(orientation='h', y=-0.25),
                    margin=dict(l=40, r=40, t=30, b=60),
                    xaxis_title='Month',
                    yaxis_title='Volume'
                )
                st.plotly_chart(fig_bio, use_container_width=True)
            else:
                st.info("Biometric age group columns not available")
        
        with col4:
            st.markdown("**Month-wise Demographic Trend**")
            demo_cols = ['demo_age_5_17', 'demo_age_17_']
            available_demo = [c for c in demo_cols if c in df_monthly.columns]
            
            if available_demo:
                monthly_demo = df_monthly.groupby('month')[available_demo].sum().reset_index()
                fig_demo = go.Figure()
                demo_colors = ['#3498db', '#e67e22']
                demo_labels = ['Demo Age 5-17', 'Demo Age 17+']
                for i, col in enumerate(available_demo):
                    fig_demo.add_trace(go.Scatter(
                        x=monthly_demo['month'],
                        y=monthly_demo[col],
                        mode='lines+markers',
                        name=demo_labels[i] if i < len(demo_labels) else col,
                        line=dict(color=demo_colors[i % len(demo_colors)], width=2),
                        marker=dict(size=6)
                    ))
                fig_demo.update_layout(
                    height=300,
                    template='plotly_white',
                    legend=dict(orientation='h', y=-0.25),
                    margin=dict(l=40, r=40, t=30, b=60),
                    xaxis_title='Month',
                    yaxis_title='Volume'
                )
                st.plotly_chart(fig_demo, use_container_width=True)
            else:
                st.info("Demographic age group columns not available")
        
        # ---- Row 3: Monthly Summary Table ----
        st.markdown("---")
        st.markdown("**📋 Monthly Aggregated Summary**")
        
        # Build summary columns
        agg_cols = {config.METRIC_ENROLMENT_TOTAL: 'sum', config.METRIC_TOTAL_UPDATES: 'sum'}
        if config.METRIC_DEMOGRAPHIC_UPDATES_TOTAL in df_monthly.columns:
            agg_cols[config.METRIC_DEMOGRAPHIC_UPDATES_TOTAL] = 'sum'
        if config.METRIC_BIOMETRIC_UPDATES_TOTAL in df_monthly.columns:
            agg_cols[config.METRIC_BIOMETRIC_UPDATES_TOTAL] = 'sum'
        agg_cols['district'] = 'nunique'
        
        monthly_summary = df_monthly.groupby('month').agg(agg_cols).reset_index()
        monthly_summary = monthly_summary.rename(columns={'district': 'Active Districts'})
        
        # Format large numbers
        for col in monthly_summary.columns:
            if col != 'month' and col != 'Active Districts':
                monthly_summary[col] = monthly_summary[col].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(monthly_summary, use_container_width=True, hide_index=True)
        
        # ---- Row 4: State-wise comparison ----
        st.markdown("---")
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("**🏆 Top 10 States by Enrolment**")
            state_enrol = df_filtered.groupby('state')[config.METRIC_ENROLMENT_TOTAL].sum().sort_values(ascending=False).head(10)
            fig_state = px.bar(
                x=state_enrol.values,
                y=state_enrol.index,
                orientation='h',
                color=state_enrol.values,
                color_continuous_scale='Oranges'
            )
            fig_state.update_layout(
                height=350,
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(categoryorder='total ascending'),
                xaxis_title='Total Enrolments',
                yaxis_title=''
            )
            st.plotly_chart(fig_state, use_container_width=True)
        
        with col6:
            st.markdown("**📊 Data Quality Overview**")
            # Data quality metrics
            total_records = len(df_filtered)
            unique_districts = df_filtered['district'].nunique()
            unique_states = df_filtered['state'].nunique()
            date_range_days = (df_filtered['date'].max() - df_filtered['date'].min()).days
            
            quality_metrics = {
                'Metric': ['Total Records', 'Unique Districts', 'Unique States', 'Date Range (Days)', 'Avg Records/District'],
                'Value': [
                    f"{total_records:,}",
                    f"{unique_districts:,}",
                    f"{unique_states:,}",
                    f"{date_range_days:,}",
                    f"{total_records/unique_districts:,.0f}" if unique_districts > 0 else "N/A"
                ]
            }
            st.dataframe(pd.DataFrame(quality_metrics), use_container_width=True, hide_index=True)
    
    with tab1:
        st.subheader("Enrolment & Update Trends")
        fig = plot_timeseries(df_filtered, config.METRIC_ENROLMENT_TOTAL, "Daily Enrolments")
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = plot_timeseries(df_filtered, config.METRIC_TOTAL_UPDATES, "Daily Updates")
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("ASI Score Distribution")
        fig = plot_asi_choropleth(df_filtered)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Anomaly Detection")
        fig = plot_anomaly_scatter(
            df_filtered,
            config.METRIC_ENROLMENT_TOTAL,
            config.METRIC_TOTAL_UPDATES
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly summary
        anomaly_summary = df_filtered[df_filtered['is_anomaly'] == 1].groupby(
            ['state', 'district']
        ).size().reset_index(name='count').sort_values('count', ascending=False)
        st.dataframe(anomaly_summary.head(20), use_container_width=True)
    
    with tab4:
        st.subheader("Inclusion Risk Analysis")
        
        if 'inclusion_risk' not in df_filtered.columns:
            df_filtered = detect_inclusion_risk(df_filtered)
        
        fig = plot_inclusion_risk_map(df_filtered)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.subheader("Top Districts Summary")
        summary = generate_summary_table(df_filtered, top_n=20)
        st.dataframe(summary, use_container_width=True)
    
    with tab5:
        st.subheader("30-Day Forecast")
        
        if st.button("Generate Forecast"):
            with st.spinner("Training Prophet model..."):
                daily = df_filtered.groupby('date').agg({
                    config.METRIC_ENROLMENT_TOTAL: 'sum'
                }).reset_index()
                
                if len(daily) >= 30:
                    forecaster = EnrolmentForecaster(horizon=30)
                    forecaster.fit(daily, target_col=config.METRIC_ENROLMENT_TOTAL)
                    forecast = forecaster.forecast(periods=30)
                    
                    # Rename for Prophet format compatibility
                    daily_prophet = daily.rename(columns={'date': 'ds', config.METRIC_ENROLMENT_TOTAL: 'actual'})
                    fig = plot_forecast(daily_prophet, forecast, date_col='ds', actual_col='actual', title="Enrolment Forecast (30 Days)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Backtest metrics
                    metrics = forecaster.backtest(daily, test_days=min(30, len(daily)//4))
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MAPE", f"{metrics['mape']:.1f}%")
                    col2.metric("RMSE", f"{metrics['rmse']:,.0f}")
                    col3.metric("MAE", f"{metrics['mae']:,.0f}")
                else:
                    st.warning("Insufficient data for forecasting (need at least 30 days)")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"*Data: {len(df_filtered):,} rows | "
        f"{df_filtered['district'].nunique()} districts | "
        f"{df_filtered['state'].nunique()} states*"
    )


if __name__ == "__main__":
    main()
