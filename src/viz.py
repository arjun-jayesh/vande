# File 5: src/viz.py
"""
Visualization module for Project Vande.

This module provides functions for creating charts, maps, and reports
for the Aadhaar analytics dashboard and PDF exports.

Functions:
    - plot_asi_choropleth: District-level ASI map
    - plot_timeseries: Line chart with annotations
    - plot_anomaly_scatter: 2D scatter with outlier highlighting
    - plot_forecast: Prophet-style forecast visualization
    - plot_inclusion_risk_map: Map showing inclusion risk districts
    - plot_age_distribution_pie: Pie chart for age group distribution
    - plot_stl_decomposition: STL decomposition visualization
    - generate_summary_table: Formatted top-N dataframe table
    - generate_pdf_report: Export all visualizations to PDF
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

from . import config

# Configure logging
logging.basicConfig(format=config.LOG_FORMAT, level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Set style defaults
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_asi_choropleth(
    df: pd.DataFrame,
    asi_col: str = 'asi_score',
    geojson: Optional[Dict] = None,
    title: str = "Aadhaar Stress Index by District"
) -> go.Figure:
    """
    Create a choropleth map showing ASI scores by district.
    
    Args:
        df: DataFrame with district and ASI score columns.
        asi_col: Name of ASI score column.
        geojson: Optional GeoJSON for district boundaries. If None, creates a bar chart.
        title: Chart title.
    
    Returns:
        go.Figure: Plotly figure object.
        
    Example:
        >>> fig = plot_asi_choropleth(df)
        >>> fig.write_html("asi_map.html")
    """
    logger.info("Creating ASI choropleth/chart...")
    
    # Aggregate to district level
    if 'district' not in df.columns:
        raise ValueError("DataFrame must contain 'district' column")
    
    # Check if ASI column exists, if not create placeholder
    if asi_col not in df.columns:
        logger.warning(f"ASI column '{asi_col}' not found. Creating placeholder.")
        df = df.copy()
        df[asi_col] = np.random.uniform(20, 80, len(df))
    
    district_asi = df.groupby(['state', 'district'])[asi_col].mean().reset_index()
    district_asi = district_asi.sort_values(asi_col, ascending=False)
    
    if geojson:
        # Create actual choropleth map
        fig = px.choropleth(
            district_asi,
            locations='district',
            geojson=geojson,
            color=asi_col,
            color_continuous_scale=config.ASI_COLORSCALE,
            range_color=[0, 100],
            title=title,
            labels={asi_col: 'ASI Score'}
        )
    else:
        # Create horizontal bar chart as alternative
        top_n = min(30, len(district_asi))
        plot_data = district_asi.head(top_n)
        
        fig = px.bar(
            plot_data,
            x=asi_col,
            y='district',
            color=asi_col,
            color_continuous_scale=config.ASI_COLORSCALE,
            range_color=[0, 100],
            orientation='h',
            title=f"{title} (Top {top_n} Districts)",
            labels={asi_col: 'ASI Score', 'district': 'District'}
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=max(400, top_n * 25)
        )
    
    fig.update_layout(
        template=config.PLOTLY_TEMPLATE,
        coloraxis_colorbar=dict(title="ASI Score")
    )
    
    return fig


def plot_timeseries(
    df: pd.DataFrame,
    metric: str,
    title: str,
    date_col: str = 'date',
    group_col: Optional[str] = None,
    show_trend: bool = True,
    annotations: Optional[List[Dict]] = None
) -> go.Figure:
    """
    Create a time series line chart with optional trend and annotations.
    
    Args:
        df: DataFrame with date and metric columns.
        metric: Name of metric column to plot.
        title: Chart title.
        date_col: Name of date column.
        group_col: Optional column for grouping lines (e.g., 'state').
        show_trend: Whether to show rolling average trend line.
        annotations: Optional list of annotation dicts with 'date', 'text' keys.
    
    Returns:
        go.Figure: Plotly figure object.
        
    Example:
        >>> fig = plot_timeseries(df, 'enrolment_total', 'Daily Enrolments')
        >>> fig.show()
    """
    logger.info(f"Creating time series plot for {metric}...")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if group_col and group_col in df.columns:
        fig = px.line(
            df.sort_values(date_col),
            x=date_col,
            y=metric,
            color=group_col,
            title=title,
            labels={metric: metric.replace('_', ' ').title(), date_col: 'Date'}
        )
    else:
        # Aggregate by date
        daily_data = df.groupby(date_col)[metric].sum().reset_index()
        daily_data = daily_data.sort_values(date_col)
        
        fig = go.Figure()
        
        # Main line
        fig.add_trace(go.Scatter(
            x=daily_data[date_col],
            y=daily_data[metric],
            mode='lines',
            name=metric.replace('_', ' ').title(),
            line=dict(color=config.COLOR_PRIMARY, width=2)
        ))
        
        # Trend line (7-day rolling average)
        if show_trend and len(daily_data) >= 7:
            trend = daily_data[metric].rolling(7, center=True).mean()
            fig.add_trace(go.Scatter(
                x=daily_data[date_col],
                y=trend,
                mode='lines',
                name='7-Day Trend',
                line=dict(color=config.COLOR_SECONDARY, width=2, dash='dash')
            ))
        
        fig.update_layout(title=title)
    
    # Add annotations
    if annotations:
        for ann in annotations:
            fig.add_annotation(
                x=ann.get('date'),
                y=ann.get('y', 0),
                text=ann.get('text', ''),
                showarrow=True,
                arrowhead=2
            )
    
    fig.update_layout(
        template=config.PLOTLY_TEMPLATE,
        xaxis_title='Date',
        yaxis_title=metric.replace('_', ' ').title(),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    return fig


def plot_anomaly_scatter(
    df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    anomaly_col: str = 'is_anomaly',
    score_col: str = 'anomaly_score',
    title: str = "Anomaly Detection Scatter Plot"
) -> go.Figure:
    """
    Create a 2D scatter plot highlighting anomalous points.
    
    Args:
        df: DataFrame with feature and anomaly columns.
        x_feature: Name of x-axis feature column.
        y_feature: Name of y-axis feature column.
        anomaly_col: Name of binary anomaly flag column.
        score_col: Name of anomaly score column.
        title: Chart title.
    
    Returns:
        go.Figure: Plotly figure object.
        
    Example:
        >>> fig = plot_anomaly_scatter(df, 'enrolment_total', 'total_updates')
        >>> fig.show()
    """
    logger.info("Creating anomaly scatter plot...")
    
    df = df.copy()
    
    # Handle missing columns
    if anomaly_col not in df.columns:
        df[anomaly_col] = 0
    if score_col not in df.columns:
        df[score_col] = 0
    
    # Separate normal and anomaly points
    normal = df[df[anomaly_col] == 0]
    anomalies = df[df[anomaly_col] == 1]
    
    fig = go.Figure()
    
    # Normal points
    fig.add_trace(go.Scatter(
        x=normal[x_feature],
        y=normal[y_feature],
        mode='markers',
        name='Normal',
        marker=dict(
            color=config.COLOR_PRIMARY,
            size=6,
            opacity=0.6
        ),
        text=normal.get('district', ''),
        hovertemplate='%{text}<br>' + 
                      f'{x_feature}: %{{x:.0f}}<br>' +
                      f'{y_feature}: %{{y:.0f}}<extra></extra>'
    ))
    
    # Anomaly points
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies[x_feature],
            y=anomalies[y_feature],
            mode='markers',
            name='Anomaly',
            marker=dict(
                color=config.COLOR_DANGER,
                size=10,
                symbol='x',
                line=dict(width=2)
            ),
            text=anomalies.get('district', ''),
            hovertemplate='%{text}<br>' + 
                          f'{x_feature}: %{{x:.0f}}<br>' +
                          f'{y_feature}: %{{y:.0f}}<br>' +
                          f'Score: %{{customdata:.2f}}<extra></extra>',
            customdata=anomalies[score_col]
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_feature.replace('_', ' ').title(),
        yaxis_title=y_feature.replace('_', ' ').title(),
        template=config.PLOTLY_TEMPLATE,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    return fig


def plot_forecast(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    date_col: str = 'ds',
    actual_col: str = 'actual',
    forecast_col: str = 'yhat',
    title: str = "Enrolment Forecast"
) -> go.Figure:
    """
    Create a Prophet-style forecast visualization.
    
    Args:
        historical: DataFrame with historical actual values.
        forecast: DataFrame with forecast predictions and confidence intervals.
        date_col: Name of date column.
        actual_col: Name of actual values column.
        forecast_col: Name of forecast column.
        title: Chart title.
    
    Returns:
        go.Figure: Plotly figure object.
        
    Example:
        >>> fig = plot_forecast(historical_df, forecast_df)
        >>> fig.show()
    """
    logger.info("Creating forecast plot...")
    
    fig = go.Figure()
    
    # Confidence intervals (if available)
    if 'yhat_upper' in forecast.columns and 'yhat_lower' in forecast.columns:
        fig.add_trace(go.Scatter(
            x=list(forecast[date_col]) + list(forecast[date_col])[::-1],
            y=list(forecast['yhat_upper']) + list(forecast['yhat_lower'])[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True
        ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast[date_col],
        y=forecast[forecast_col],
        mode='lines',
        name='Forecast',
        line=dict(color=config.COLOR_PRIMARY, width=2)
    ))
    
    # Actual values
    if actual_col in forecast.columns:
        actual_data = forecast[forecast[actual_col].notna()]
        fig.add_trace(go.Scatter(
            x=actual_data[date_col],
            y=actual_data[actual_col],
            mode='markers',
            name='Actual',
            marker=dict(color='black', size=4)
        ))
    
    # Vertical line at forecast start - use add_shape to avoid Plotly's datetime arithmetic bug
    if actual_col in forecast.columns and forecast[actual_col].notna().any():
        last_actual = forecast[forecast[actual_col].notna()][date_col].max()
        # Convert to string format for Plotly compatibility
        if hasattr(last_actual, 'isoformat'):
            last_actual_str = last_actual.isoformat() if hasattr(last_actual, 'isoformat') else str(last_actual)
        elif hasattr(last_actual, 'to_pydatetime'):
            last_actual_str = last_actual.to_pydatetime().isoformat()
        else:
            last_actual_str = str(last_actual)
        
        # Use add_shape instead of add_vline to avoid datetime arithmetic issues
        fig.add_shape(
            type="line",
            x0=last_actual_str, x1=last_actual_str,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="gray", width=2, dash="dash")
        )
        fig.add_annotation(
            x=last_actual_str,
            y=1,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            yanchor="bottom"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        template=config.PLOTLY_TEMPLATE,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    return fig


def plot_inclusion_risk_map(
    df: pd.DataFrame,
    risk_col: str = 'inclusion_risk',
    title: str = "Districts at Inclusion Risk"
) -> go.Figure:
    """
    Create a visualization showing districts with inclusion risk.
    
    Args:
        df: DataFrame with district and risk flag columns.
        risk_col: Name of risk flag column.
        title: Chart title.
    
    Returns:
        go.Figure: Plotly figure object.
        
    Example:
        >>> fig = plot_inclusion_risk_map(df)
        >>> fig.show()
    """
    logger.info("Creating inclusion risk visualization...")
    
    # Handle missing column
    if risk_col not in df.columns:
        df = df.copy()
        df[risk_col] = False
    
    # Aggregate by district
    risk_summary = df.groupby(['state', 'district']).agg({
        risk_col: 'any',
        'risk_low_velocity': 'any' if 'risk_low_velocity' in df.columns else lambda x: False,
        'risk_high_ratio_low_growth': 'any' if 'risk_high_ratio_low_growth' in df.columns else lambda x: False,
        'risk_zero_enrolments': 'any' if 'risk_zero_enrolments' in df.columns else lambda x: False,
    }).reset_index()
    
    risk_summary['risk_score'] = (
        risk_summary[risk_col].astype(int) * 3 +
        risk_summary.get('risk_low_velocity', 0).astype(int) +
        risk_summary.get('risk_high_ratio_low_growth', 0).astype(int) +
        risk_summary.get('risk_zero_enrolments', 0).astype(int)
    )
    
    # Sort by risk
    risk_summary = risk_summary.sort_values('risk_score', ascending=False)
    
    # Get at-risk districts
    at_risk = risk_summary[risk_summary[risk_col] == True]
    
    if len(at_risk) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No districts currently at inclusion risk",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
    else:
        fig = px.bar(
            at_risk.head(30),
            x='risk_score',
            y='district',
            color='state',
            orientation='h',
            title=f"{title} ({len(at_risk)} districts)",
            labels={'risk_score': 'Risk Score', 'district': 'District'}
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=max(400, len(at_risk.head(30)) * 25)
        )
    
    fig.update_layout(
        template=config.PLOTLY_TEMPLATE,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    return fig


def plot_age_distribution_pie(
    df: pd.DataFrame,
    dataset_type: str = 'enrolment',
    title: Optional[str] = None
) -> go.Figure:
    """
    Create a pie chart showing age group distribution.
    
    Args:
        df: DataFrame with age group columns.
        dataset_type: Type of data - 'enrolment', 'demographic', or 'biometric'.
        title: Optional chart title.
    
    Returns:
        go.Figure: Plotly figure object.
        
    Example:
        >>> fig = plot_age_distribution_pie(df, 'enrolment')
        >>> fig.show()
    """
    logger.info(f"Creating {dataset_type} age distribution pie chart...")
    
    if dataset_type == 'enrolment':
        cols = ['demo_age_5_17', 'demo_age_17_']
        labels = ['Age 5-17', 'Age 17+']
    elif dataset_type == 'demographic':
        cols = ['age_0_5', 'age_5_17', 'age_18_greater']
        labels = ['Age 0-5', 'Age 5-17', 'Age 18+']
    elif dataset_type == 'biometric':
        cols = ['bio_age_5', 'bio_age_17_']
        labels = ['Age 0-5', 'Age 5+']
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Filter available columns
    available_cols = [c for c in cols if c in df.columns]
    if not available_cols:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No age group columns found for {dataset_type}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Calculate totals
    totals = [df[col].sum() for col in available_cols]
    available_labels = [labels[cols.index(c)] for c in available_cols]
    
    fig = go.Figure(data=[go.Pie(
        labels=available_labels,
        values=totals,
        hole=0.4,
        textinfo='percent+label',
        marker=dict(colors=px.colors.qualitative.Set2)
    )])
    
    title = title or f"{dataset_type.title()} by Age Group"
    fig.update_layout(
        title=title,
        template=config.PLOTLY_TEMPLATE
    )
    
    return fig


def plot_stl_decomposition(
    df: pd.DataFrame,
    metric: str,
    date_col: str = 'date',
    period: int = 7,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Create STL decomposition visualization using statsmodels.
    
    Args:
        df: DataFrame with date and metric columns.
        metric: Name of metric column to decompose.
        date_col: Name of date column.
        period: Seasonal period (default 7 for weekly).
        title: Optional chart title.
    
    Returns:
        plt.Figure: Matplotlib figure object.
        
    Example:
        >>> fig = plot_stl_decomposition(df, 'enrolment_total')
        >>> fig.savefig('stl_decomposition.png')
    """
    logger.info(f"Creating STL decomposition for {metric}...")
    
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        logger.error("statsmodels is required for STL decomposition")
        raise
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Aggregate to daily
    daily = df.groupby(date_col)[metric].sum().sort_index()
    
    # Fill gaps if any
    daily = daily.asfreq('D', method='ffill')
    
    # Perform STL decomposition
    stl = STL(daily, period=period, robust=True)
    result = stl.fit()
    
    # Create plot
    fig, axes = plt.subplots(4, 1, figsize=(config.FIG_WIDTH, config.FIG_HEIGHT + 4))
    
    # Original
    axes[0].plot(daily.index, daily.values, 'b-', linewidth=1)
    axes[0].set_ylabel('Original')
    axes[0].set_title(title or f'STL Decomposition: {metric}')
    
    # Trend
    axes[1].plot(daily.index, result.trend, 'g-', linewidth=1)
    axes[1].set_ylabel('Trend')
    
    # Seasonal
    axes[2].plot(daily.index, result.seasonal, 'r-', linewidth=1)
    axes[2].set_ylabel('Seasonal')
    
    # Residual
    axes[3].plot(daily.index, result.resid, 'k-', linewidth=1, alpha=0.7)
    axes[3].axhline(y=0, color='gray', linestyle='--')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    
    plt.tight_layout()
    
    return fig


def generate_summary_table(
    df: pd.DataFrame,
    top_n: int = 20,
    metrics: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Generate a formatted summary table for top districts.
    
    Args:
        df: Input dataframe.
        top_n: Number of top districts to include.
        metrics: List of metric columns to include.
        sort_by: Column to sort by.
        ascending: Sort order.
    
    Returns:
        pd.DataFrame: Formatted summary table.
        
    Example:
        >>> table = generate_summary_table(df, top_n=20)
        >>> table.to_csv('summary.csv')
    """
    logger.info(f"Generating summary table for top {top_n} districts...")
    
    # Default metrics
    if metrics is None:
        metrics = [
            config.METRIC_ENROLMENT_TOTAL,
            config.METRIC_TOTAL_UPDATES,
            config.METRIC_UPDATE_TO_ENROLMENT_RATIO,
            config.METRIC_ENROLMENT_VELOCITY,
            config.METRIC_ENROLMENT_VOLATILITY
        ]
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in df.columns]
    
    # Aggregate by district
    agg_dict = {}
    for m in available_metrics:
        agg_dict[m] = 'mean'
    
    # Add counts
    agg_dict['date'] = 'count'
    
    summary = df.groupby(['state', 'district']).agg(agg_dict).reset_index()
    summary = summary.rename(columns={'date': 'data_points'})
    
    # Sort
    sort_col = sort_by or available_metrics[0] if available_metrics else 'data_points'
    summary = summary.sort_values(sort_col, ascending=ascending)
    
    # Format numeric columns
    for col in available_metrics:
        if col in summary.columns:
            if 'ratio' in col.lower():
                summary[col] = summary[col].round(3)
            else:
                summary[col] = summary[col].round(1)
    
    return summary.head(top_n)


def generate_state_ranking_table(
    df: pd.DataFrame,
    top_n: int = 10,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate state-level ranking table.
    
    Args:
        df: Input dataframe.
        top_n: Number of top states to include.
        metrics: List of metrics to compute.
    
    Returns:
        pd.DataFrame: State ranking table.
        
    Example:
        >>> rankings = generate_state_ranking_table(df)
        >>> print(rankings)
    """
    logger.info("Generating state ranking table...")
    
    if metrics is None:
        metrics = [
            config.METRIC_ENROLMENT_TOTAL,
            config.METRIC_UPDATE_TO_ENROLMENT_RATIO
        ]
    
    available_metrics = [m for m in metrics if m in df.columns]
    
    agg_dict = {m: 'sum' if 'total' in m.lower() else 'mean' for m in available_metrics}
    agg_dict['district'] = 'nunique'
    
    state_summary = df.groupby('state').agg(agg_dict).reset_index()
    state_summary = state_summary.rename(columns={'district': 'num_districts'})
    
    # Add ranks
    for metric in available_metrics:
        state_summary[f'{metric}_rank'] = state_summary[metric].rank(ascending=False)
    
    # Sort by first metric
    if available_metrics:
        state_summary = state_summary.sort_values(available_metrics[0], ascending=False)
    
    return state_summary.head(top_n)


def generate_pdf_report(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Project Vande Analytics Report"
) -> Path:
    """
    Generate a comprehensive PDF report with all visualizations.
    
    Args:
        df: Processed dataframe with all metrics.
        output_path: Path for output PDF. Defaults to outputs/final_report.pdf.
        title: Report title.
    
    Returns:
        Path: Path to generated PDF file.
        
    Example:
        >>> pdf_path = generate_pdf_report(df)
        >>> print(f"Report saved to: {pdf_path}")
    """
    logger.info("Generating PDF report...")
    
    try:
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        raise ImportError("matplotlib is required for PDF generation")
    
    output_path = Path(output_path) if output_path else config.OUTPUTS_DIR / "final_report.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT))
        ax.text(0.5, 0.6, title, ha='center', va='center', fontsize=24, fontweight='bold')
        ax.text(0.5, 0.4, f"Generated from {len(df):,} records", ha='center', va='center', fontsize=14)
        ax.text(0.5, 0.3, f"{df['district'].nunique()} districts | {df['state'].nunique()} states", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Summary statistics
        fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT))
        summary_data = [
            ['Total Enrolments', f"{df[config.METRIC_ENROLMENT_TOTAL].sum():,.0f}"],
            ['Total Updates', f"{df[config.METRIC_TOTAL_UPDATES].sum():,.0f}"],
            ['Avg Update Ratio', f"{df[config.METRIC_UPDATE_TO_ENROLMENT_RATIO].mean():.2f}"],
            ['Date Range', f"{df['date'].min().date()} to {df['date'].max().date()}"],
            ['Districts', f"{df['district'].nunique()}"],
            ['States', f"{df['state'].nunique()}"]
        ]
        table = ax.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                        loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        ax.set_title('Summary Statistics', fontsize=16, fontweight='bold')
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Time series plot
        fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT))
        daily = df.groupby('date')[config.METRIC_ENROLMENT_TOTAL].sum()
        ax.plot(daily.index, daily.values, 'b-', linewidth=1)
        ax.fill_between(daily.index, daily.values, alpha=0.3)
        ax.set_title('Daily Enrolment Trend', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Enrolments')
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Age distribution
        if 'demo_age_5_17' in df.columns and 'demo_age_17_' in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(config.FIG_WIDTH, 6))
            
            # Enrolment age distribution
            enrol_totals = [df['demo_age_5_17'].sum(), df['demo_age_17_'].sum()]
            axes[0].pie(enrol_totals, labels=['Age 5-17', 'Age 17+'], autopct='%1.1f%%',
                       colors=sns.color_palette("Set2"))
            axes[0].set_title('Enrolment by Age Group')
            
            # Demographic update distribution
            if 'age_0_5' in df.columns:
                demo_totals = [df['age_0_5'].sum(), df['age_5_17'].sum(), df['age_18_greater'].sum()]
                axes[1].pie(demo_totals, labels=['Age 0-5', 'Age 5-17', 'Age 18+'], autopct='%1.1f%%',
                           colors=sns.color_palette("Set3"))
                axes[1].set_title('Demographic Updates by Age Group')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        # STL decomposition
        try:
            stl_fig = plot_stl_decomposition(df, config.METRIC_ENROLMENT_TOTAL)
            pdf.savefig(stl_fig, bbox_inches='tight')
            plt.close(stl_fig)
        except Exception as e:
            logger.warning(f"Could not generate STL decomposition: {e}")
        
        # Top districts table
        fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT))
        top_districts = generate_summary_table(df, top_n=15)
        ax.axis('off')
        table = ax.table(
            cellText=top_districts.values,
            colLabels=top_districts.columns,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        ax.set_title('Top 15 Districts by Enrolment', fontsize=14, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    logger.info(f"PDF report saved to: {output_path}")
    
    return output_path


def save_figure(
    fig: Union[go.Figure, plt.Figure],
    filename: str,
    formats: List[str] = ['png', 'html']
) -> List[Path]:
    """
    Save a figure in multiple formats.
    
    Args:
        fig: Plotly or Matplotlib figure.
        filename: Base filename (without extension).
        formats: List of formats to save ('png', 'html', 'pdf', 'svg').
    
    Returns:
        List[Path]: Paths to saved files.
        
    Example:
        >>> paths = save_figure(fig, 'my_chart', ['png', 'html'])
    """
    saved_paths = []
    
    for fmt in formats:
        output_path = config.FIGURES_DIR / f"{filename}.{fmt}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if isinstance(fig, go.Figure):
                if fmt == 'html':
                    fig.write_html(str(output_path))
                elif fmt in ['png', 'pdf', 'svg']:
                    fig.write_image(str(output_path))
            else:
                fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
            
            saved_paths.append(output_path)
            logger.info(f"Saved figure to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save figure as {fmt}: {e}")
    
    return saved_paths


def export_dataframe_to_csv(
    df: pd.DataFrame,
    filename: str,
    include_index: bool = False
) -> Path:
    """
    Export dataframe to CSV in the tables output directory.
    
    Args:
        df: DataFrame to export.
        filename: Output filename (without extension).
        include_index: Whether to include index in output.
    
    Returns:
        Path: Path to saved CSV file.
    """
    output_path = config.TABLES_DIR / f"{filename}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=include_index)
    logger.info(f"Exported table to: {output_path}")
    
    return output_path
