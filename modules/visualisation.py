"""
Visualisation Module
Generates interactive charts and plots for traffic analysis and evaluation.
"""

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd


# Consistent colour scheme
COLOUR_SCHEME = px.colors.qualitative.Set2
PRIMARY_COLOUR = "#1f77b4"
SECONDARY_COLOUR = "#ff7f0e"
BENIGN_COLOUR = "#2ecc71"
MALICIOUS_COLOUR = "#e74c3c"


def plot_label_distribution(labels):
    """
    Bar chart showing distribution of traffic labels (benign vs attack types).
    """
    if hasattr(labels, 'value_counts'):
        counts = labels.value_counts()
    else:
        unique, cnt = np.unique(labels, return_counts=True)
        counts = pd.Series(cnt, index=unique)

    counts = counts.sort_values(ascending=True)

    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        labels={"x": "Count", "y": "Label"},
        title="Traffic Label Distribution",
        color=counts.index,
        color_discrete_sequence=COLOUR_SCHEME
    )
    fig.update_layout(showlegend=False, height=400)

    return fig


def plot_protocol_distribution(df, protocol_col):
    """
    Pie chart of protocol distribution.
    """
    counts = df[protocol_col].value_counts().head(10)

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title="Protocol Distribution",
        color_discrete_sequence=COLOUR_SCHEME
    )
    fig.update_layout(height=400)

    return fig


def plot_top_destination_ports(df, port_col, n=15):
    """
    Bar chart of top N destination ports by traffic volume.
    """
    # Map well-known ports to names
    port_services = {
        0: "0 (Reserved)", 20: "20 (FTP-Data)", 21: "21 (FTP)", 22: "22 (SSH)",
        23: "23 (Telnet)", 25: "25 (SMTP)", 53: "53 (DNS)", 67: "67 (DHCP)",
        80: "80 (HTTP)", 110: "110 (POP3)", 123: "123 (NTP)", 143: "143 (IMAP)",
        161: "161 (SNMP)", 443: "443 (HTTPS)", 445: "445 (SMB)", 465: "465 (SMTPS)",
        514: "514 (Syslog)", 587: "587 (SMTP)", 636: "636 (LDAPS)", 993: "993 (IMAPS)",
        995: "995 (POP3S)", 1433: "1433 (MSSQL)", 1521: "1521 (Oracle)",
        3306: "3306 (MySQL)", 3389: "3389 (RDP)", 5432: "5432 (Postgres)",
        5900: "5900 (VNC)", 6379: "6379 (Redis)", 8080: "8080 (HTTP-Alt)",
        8443: "8443 (HTTPS-Alt)", 27017: "27017 (MongoDB)",
    }

    counts = df[port_col].value_counts().head(n)
    port_labels = [
        port_services.get(int(p), str(int(p))) if pd.notna(p) else "Unknown"
        for p in counts.index
    ]
    counts.index = port_labels
    counts = counts.sort_values(ascending=True)

    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        labels={"x": "Number of Flows", "y": "Destination Port"},
        title=f"Top {n} Destination Ports",
        color=counts.index,
        color_discrete_sequence=COLOUR_SCHEME,
    )
    fig.update_layout(showlegend=False, height=450,
                      margin={"l": 160, "r": 20, "t": 50, "b": 40})

    return fig


def plot_flow_duration_by_label(df, duration_col, label_col):
    """
    Box plot of flow duration distribution grouped by label.
    """
    # Clip extreme outliers for readability
    df_plot = df[[duration_col, label_col]].copy()
    df_plot[duration_col] = df_plot[duration_col].clip(upper=df_plot[duration_col].quantile(0.95))

    fig = px.box(
        df_plot,
        x=label_col,
        y=duration_col,
        title="Flow Duration Distribution by Label",
        labels={duration_col: "Flow Duration (microseconds)", label_col: "Label"},
        color=label_col,
        color_discrete_sequence=COLOUR_SCHEME,
    )
    fig.update_layout(height=450, showlegend=False, xaxis={"tickangle": -30})

    return fig


def plot_packet_size_distribution(df, size_col):
    """
    Histogram of packet size distribution.
    """
    data = df[size_col].clip(upper=df[size_col].quantile(0.99))

    fig = px.histogram(
        x=data,
        nbins=50,
        title="Average Packet Size Distribution",
        labels={"x": "Average Packet Size (bytes)", "count": "Number of Flows"},
        color_discrete_sequence=[PRIMARY_COLOUR],
    )
    fig.update_layout(height=400, bargap=0.05, showlegend=False)

    return fig


def plot_feature_distributions(df, feature_name):
    """
    Histogram and box plot for a selected feature.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, subplot_titles=[
        f"Distribution of {feature_name}",
        f"Box Plot of {feature_name}"
    ], row_heights=[0.7, 0.3])

    # Histogram
    fig.add_trace(
        go.Histogram(x=df[feature_name], name="Histogram", marker_color=PRIMARY_COLOUR),
        row=1, col=1
    )

    # Box plot
    fig.add_trace(
        go.Box(x=df[feature_name], name="Box Plot", marker_color=PRIMARY_COLOUR),
        row=2, col=1
    )

    fig.update_layout(height=500, showlegend=False)

    return fig


def plot_correlation_heatmap(df):
    """
    Heatmap of feature correlations.
    """
    # Sample if too many rows for performance
    if len(df) > 10000:
        sample_df = df.sample(n=10000, random_state=42)
    else:
        sample_df = df

    corr = sample_df.select_dtypes(include=[np.number]).corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
    ))

    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=600,
        width=700,
    )

    return fig


def plot_confusion_matrix(cm, labels):
    """
    Heatmap of confusion matrix.
    """
    # Ensure labels is a list of strings
    labels = [str(l) for l in labels]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 14},
    ))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=450,
        width=450,
    )

    return fig


def plot_feature_importance(importances, feature_names, top_n=15):
    """
    Horizontal bar chart of feature importances.
    """
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True).tail(top_n)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Feature Importances (Random Forest)",
        color_discrete_sequence=[PRIMARY_COLOUR]
    )
    fig.update_layout(height=450)

    return fig


def plot_detection_comparison(rf_metrics, if_metrics):
    """
    Side-by-side bar chart comparing Random Forest and Isolation Forest metrics.
    """
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    rf_values = [rf_metrics.get(m, 0) for m in metrics]
    if_values = [if_metrics.get(m, 0) for m in metrics]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Random Forest",
        x=metric_labels,
        y=rf_values,
        marker_color=PRIMARY_COLOUR
    ))

    fig.add_trace(go.Bar(
        name="Isolation Forest",
        x=metric_labels,
        y=if_values,
        marker_color=SECONDARY_COLOUR
    ))

    fig.update_layout(
        title="Detection Method Comparison",
        yaxis_title="Score",
        barmode="group",
        height=400,
        yaxis=dict(range=[0, 1.05])
    )

    return fig


def plot_traffic_over_time(df, time_col=None):
    """
    Line or bar chart of traffic over time.
    """
    if time_col and time_col in df.columns:
        time_data = pd.to_datetime(df[time_col], errors="coerce")
        time_counts = time_data.dt.hour.value_counts().sort_index()

        fig = px.line(
            x=time_counts.index,
            y=time_counts.values,
            labels={"x": "Hour of Day", "y": "Number of Flows"},
            title="Traffic Volume Over Time",
        )
    else:
        fig = go.Figure(data=go.Bar(
            x=["Total Flows"],
            y=[len(df)],
            marker_color=PRIMARY_COLOUR
        ))
        fig.update_layout(title="Total Traffic Volume", yaxis_title="Number of Flows")

    fig.update_layout(height=400)
    return fig
