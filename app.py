"""
Network Traffic Monitoring and Detection System
Main Streamlit application entry point.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# Page Configuration
st.set_page_config(
    page_title="Network IDS",
    layout="wide",
    initial_sidebar_state="expanded"
)

from modules.data_loader import load_dataset, load_sample_dataset
from modules.preprocessing import preprocess_data
from modules.feature_extraction import extract_features
from modules.detection import prepare_data, run_random_forest, run_isolation_forest
from modules.evaluation import evaluate_supervised, evaluate_unsupervised, compare_methods
from modules import visualisation as viz


# Helper: find a column by candidate names
def _find_column(df, candidates):
    """Return the first column in df whose normalised name matches any candidate."""
    normalised = {c.lower().strip().replace(" ", "_").replace("/", "_"): c for c in df.columns}
    for cand in candidates:
        if cand in normalised:
            return normalised[cand]
    return None


# Sidebar Navigation
st.sidebar.title("Network IDS")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Data Upload",
        "Preprocessing",
        "Feature Extraction",
        "Traffic Dashboard",
        "Detection",
        "Evaluation",
        "Export"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.caption("Network Traffic Monitoring and Detection System")
st.sidebar.caption("Module 6Z0019 | MMU")


# Home Page
if page == "Home":
    st.title("Network Traffic Monitoring and Detection System")
    st.divider()

    st.markdown("""
    Welcome to the **Network Traffic Monitoring and Detection System**. This
    application analyses network traffic datasets to identify suspicious or
    malicious behaviour using machine learning techniques.

    ### System Capabilities

    - **Upload and preview** network traffic datasets (CSV format)
    - **Preprocess data** to handle missing values, duplicates, and data quality issues
    - **Extract relevant features** such as flow duration, packet counts, and byte counts
    - **Visualise traffic patterns** through an interactive monitoring dashboard
    - **Detect suspicious traffic** using Random Forest or Isolation Forest
    - **Evaluate detection performance** with accuracy, precision, recall, and F1-score
    - **Export results** as downloadable CSV files

    ### How to Use

    Follow the workflow using the sidebar navigation:

    1. **Data Upload** - Upload a CSV dataset or use the built-in sample data
    2. **Preprocessing** - Clean and prepare the data for analysis
    3. **Feature Extraction** - Select and engineer key traffic features
    4. **Traffic Dashboard** - Explore interactive visualisations of traffic patterns
    5. **Detection** - Run anomaly detection using Random Forest or Isolation Forest
    6. **Evaluation** - Review detection performance metrics and confusion matrix
    7. **Export** - Download the detection results
    """)

    st.info(
        "To get started, navigate to **Data Upload** in the sidebar and upload "
        "a CSV dataset or click **Use Sample Dataset** for a quick demonstration."
    )


# Data Upload Page
elif page == "Data Upload":
    st.title("Data Upload")
    st.divider()
    st.markdown("Upload a network traffic dataset in CSV format, or use the built-in sample dataset.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload a CSV dataset",
            type=["csv"],
            help="Upload a CSV file containing network traffic data (e.g., CIC-IDS-2017)"
        )

    with col2:
        st.markdown("##### Or use sample data")
        use_sample = st.button(
            "Use Sample Dataset",
            use_container_width=True,
            help="Load the built-in sample dataset for quick testing"
        )

    # Load data
    if uploaded_file is not None:
        with st.spinner("Loading dataset..."):
            df, message = load_dataset(uploaded_file)
            if df is not None:
                st.session_state["raw_data"] = df
                st.success(message)
            else:
                st.error(message)

    elif use_sample:
        with st.spinner("Loading sample dataset..."):
            df, message = load_sample_dataset()
            if df is not None:
                st.session_state["raw_data"] = df
                st.success(message)
            else:
                st.error(message)

    # Display preview if data is loaded
    if "raw_data" in st.session_state:
        df = st.session_state["raw_data"]
        st.divider()

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", f"{len(df.columns):,}")
        col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")

        # Data preview
        st.subheader("Dataset Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)

        # Column information
        with st.expander("Column Names and Data Types"):
            col_info = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Non-Null Count": df.notna().sum().values,
                "Null Count": df.isna().sum().values
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)

        # Basic statistics
        with st.expander("Basic Statistics"):
            st.dataframe(df.describe(), use_container_width=True)
    else:
        st.warning("No dataset loaded. Please upload a CSV file or use the sample dataset.")


# Preprocessing Page
elif page == "Preprocessing":
    st.title("Data Preprocessing")
    st.divider()

    if "raw_data" not in st.session_state:
        st.warning("Please upload a dataset first. Navigate to **Data Upload** in the sidebar.")
    else:
        df = st.session_state["raw_data"]

        st.markdown(
            "Preprocessing cleans the dataset by removing duplicates, handling missing "
            "and infinite values, cleaning column names, correcting data types, and "
            "removing constant columns."
        )

        # Before stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Before Preprocessing")
            st.metric("Rows", f"{len(df):,}")
            st.metric("Columns", f"{len(df.columns):,}")

        if st.button("Run Preprocessing", type="primary", use_container_width=True):
            with st.spinner("Preprocessing data..."):
                cleaned_df, summary = preprocess_data(df)
                st.session_state["preprocessed_data"] = cleaned_df
                st.session_state["preprocessing_summary"] = summary

        if "preprocessed_data" in st.session_state:
            cleaned_df = st.session_state["preprocessed_data"]
            summary = st.session_state["preprocessing_summary"]

            with col2:
                st.markdown("##### After Preprocessing")
                st.metric("Rows", f"{len(cleaned_df):,}")
                st.metric("Columns", f"{len(cleaned_df.columns):,}")

            st.divider()
            st.subheader("Preprocessing Summary")

            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            summary_col1.metric("Duplicates Removed", f"{summary.get('duplicates_removed', 0):,}")
            summary_col2.metric("Columns Dropped", f"{summary.get('columns_dropped', 0):,}")
            summary_col3.metric("Infinite Values", f"{summary.get('infinite_values_replaced', 0):,}")
            summary_col4.metric("Missing Values Filled", f"{summary.get('missing_values_filled', 0):,}")

            with st.expander("Detailed Preprocessing Log"):
                for action in summary.get("log", []):
                    st.write(f"- {action}")

            st.subheader("Cleaned Data Preview")
            st.dataframe(cleaned_df.head(10), use_container_width=True)

            st.success("Preprocessing complete. Proceed to **Feature Extraction**.")


# Feature Extraction Page
elif page == "Feature Extraction":
    st.title("Feature Extraction")
    st.divider()

    if "preprocessed_data" not in st.session_state:
        st.warning("Please preprocess the data first. Navigate to **Preprocessing** in the sidebar.")
    else:
        df = st.session_state["preprocessed_data"]

        st.markdown(
            "Feature extraction selects the most relevant network traffic features "
            "from the dataset and creates derived features for improved detection."
        )

        if st.button("Extract Features", type="primary", use_container_width=True):
            with st.spinner("Extracting features..."):
                features_df, labels, feature_names = extract_features(df)
                if features_df is not None:
                    st.session_state["features_df"] = features_df
                    st.session_state["labels"] = labels
                    st.session_state["feature_names"] = feature_names
                else:
                    st.error("Feature extraction failed. Please check your dataset.")

        if "features_df" in st.session_state:
            features_df = st.session_state["features_df"]
            labels = st.session_state["labels"]
            feature_names = st.session_state["feature_names"]

            col1, col2 = st.columns(2)
            col1.metric("Features Selected", len(feature_names))
            col2.metric("Samples", f"{len(features_df):,}")

            st.subheader("Selected Features")
            st.dataframe(
                pd.DataFrame({"Feature": feature_names}),
                use_container_width=True,
                hide_index=True
            )

            with st.expander("Feature Summary Statistics"):
                st.dataframe(features_df.describe(), use_container_width=True)

            st.subheader("Feature Correlation Heatmap")
            fig = viz.plot_correlation_heatmap(features_df)
            st.plotly_chart(fig, use_container_width=True)

            if labels is not None:
                st.subheader("Label Distribution")
                fig = viz.plot_label_distribution(labels)
                st.plotly_chart(fig, use_container_width=True)

            st.success("Feature extraction complete. Proceed to **Traffic Dashboard** or **Detection**.")


# Traffic Dashboard Page
elif page == "Traffic Dashboard":
    st.title("Traffic Monitoring Dashboard")
    st.divider()

    if "preprocessed_data" not in st.session_state:
        st.warning("Please preprocess the data first. Navigate to **Preprocessing** in the sidebar.")
    else:
        df = st.session_state["preprocessed_data"]
        features_df = st.session_state.get("features_df", None)
        labels = st.session_state.get("labels", None)

        # Locate useful columns that exist in CIC-IDS-2017 MachineLearning CSVs
        dst_port_col = _find_column(df, ["destination_port"])
        flow_dur_col = _find_column(df, ["flow_duration"])
        avg_pkt_col = _find_column(df, ["average_packet_size", "packet_length_mean"])
        label_col = _find_column(df, ["label"])

        # Summary metrics
        st.subheader("Traffic Summary")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Total Flows", f"{len(df):,}")

        if label_col is not None:
            benign_count = int((df[label_col].astype(str).str.upper() == "BENIGN").sum())
            malicious_count = len(df) - benign_count
            unique_attacks = df[df[label_col].astype(str).str.upper() != "BENIGN"][label_col].nunique()
            metric_cols[1].metric("Benign Flows", f"{benign_count:,}")
            metric_cols[2].metric("Malicious Flows", f"{malicious_count:,}")
            metric_cols[3].metric("Attack Types", f"{unique_attacks:,}")
        else:
            metric_cols[1].metric("Benign Flows", "N/A")
            metric_cols[2].metric("Malicious Flows", "N/A")
            metric_cols[3].metric("Attack Types", "N/A")

        st.divider()

        # Row 1: Label distribution and Top destination ports
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Traffic Label Distribution")
            labels_to_plot = labels if labels is not None else (df[label_col] if label_col else None)
            if labels_to_plot is not None:
                fig = viz.plot_label_distribution(labels_to_plot)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Label column not available.")

        with chart_col2:
            st.subheader("Top Destination Ports")
            if dst_port_col is not None:
                try:
                    fig = viz.plot_top_destination_ports(df, dst_port_col, n=15)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {e}")
                    counts = df[dst_port_col].value_counts().head(15)
                    st.bar_chart(counts)
            else:
                st.info("Destination Port column not available in this dataset.")

        # Row 2: Flow duration by label and packet size distribution
        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            st.subheader("Flow Duration by Label")
            if flow_dur_col and label_col:
                fig = viz.plot_flow_duration_by_label(df, flow_dur_col, label_col)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Flow duration or label column not available.")

        with chart_col4:
            st.subheader("Average Packet Size Distribution")
            if avg_pkt_col is not None:
                fig = viz.plot_packet_size_distribution(df, avg_pkt_col)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Packet size column not available.")

        # Feature distribution explorer
        if features_df is not None and len(features_df.columns) > 0:
            st.divider()
            st.subheader("Feature Distribution Explorer")
            st.caption("Select any extracted feature to view its distribution and box plot.")
            selected_feature = st.selectbox(
                "Feature",
                options=features_df.columns.tolist(),
                label_visibility="collapsed"
            )
            if selected_feature:
                fig = viz.plot_feature_distributions(features_df, selected_feature)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Extract features first to use the feature distribution explorer.")


# Detection Page
elif page == "Detection":
    st.title("Detection")
    st.divider()

    if "features_df" not in st.session_state or "labels" not in st.session_state:
        st.warning("Please extract features first. Navigate to **Feature Extraction** in the sidebar.")
    else:
        features_df = st.session_state["features_df"]
        labels = st.session_state["labels"]

        # Detection method selector
        method = st.radio(
            "Select Detection Method",
            ["Random Forest (Supervised)", "Isolation Forest (Unsupervised)"],
            horizontal=True
        )

        st.divider()

        if method == "Random Forest (Supervised)":
            st.markdown(
                "**Random Forest** is a supervised classifier that learns from labelled data "
                "to distinguish between benign and malicious traffic."
            )

            classification_mode = st.radio(
                "Classification Mode",
                ["Binary (Benign vs Malicious)", "Multi-class (All Attack Types)"],
                horizontal=True
            )

            if st.button("Run Random Forest Detection", type="primary", use_container_width=True):
                binary_mode = classification_mode.startswith("Binary")

                with st.spinner("Preparing data..."):
                    X_train, X_test, y_train, y_test, scaler = prepare_data(
                        features_df, labels, binary=binary_mode
                    )
                    st.session_state["scaler"] = scaler

                progress_bar = st.progress(0, text="Training Random Forest model...")
                progress_bar.progress(20, text="Training Random Forest model...")

                predictions, probabilities, model, importances = run_random_forest(
                    X_train, X_test, y_train, y_test
                )

                progress_bar.progress(80, text="Generating results...")

                st.session_state["rf_predictions"] = predictions
                st.session_state["rf_model"] = model
                st.session_state["rf_importances"] = importances
                st.session_state["rf_y_test"] = y_test
                st.session_state["rf_X_test"] = X_test
                st.session_state["rf_feature_names"] = st.session_state["feature_names"]
                st.session_state["detection_method"] = "Random Forest"
                st.session_state["rf_binary_mode"] = binary_mode

                progress_bar.progress(100, text="Detection complete.")

            # Display RF results
            if "rf_predictions" in st.session_state and st.session_state.get("detection_method") == "Random Forest":
                predictions = st.session_state["rf_predictions"]
                y_test = st.session_state["rf_y_test"]
                binary_mode = st.session_state.get("rf_binary_mode", True)

                st.divider()
                st.subheader("Detection Summary")

                total = len(predictions)
                if binary_mode:
                    # Binary: 3 clean metric cards
                    benign = int(np.sum(predictions == "BENIGN"))
                    malicious = int(np.sum(predictions == "MALICIOUS"))
                    sum_cols = st.columns(3)
                    sum_cols[0].metric("Total Records", f"{total:,}")
                    sum_cols[1].metric("Benign", f"{benign:,}", f"{benign/total*100:.2f}%")
                    sum_cols[2].metric("Malicious", f"{malicious:,}", f"{malicious/total*100:.2f}%")
                else:
                    # Multi-class: compact summary + detailed table (not 16 metric cards)
                    benign = int(np.sum(predictions == "BENIGN"))
                    malicious = total - benign
                    num_classes = len(np.unique(predictions))
                    sum_cols = st.columns(4)
                    sum_cols[0].metric("Total Records", f"{total:,}")
                    sum_cols[1].metric("Benign", f"{benign:,}", f"{benign/total*100:.2f}%")
                    sum_cols[2].metric("Malicious", f"{malicious:,}", f"{malicious/total*100:.2f}%")
                    sum_cols[3].metric("Predicted Classes", f"{num_classes}")

                    st.markdown("##### Prediction Breakdown by Class")
                    unique, counts = np.unique(predictions, return_counts=True)
                    breakdown = pd.DataFrame({
                        "Predicted Label": unique,
                        "Count": counts,
                        "Percentage": [f"{c/total*100:.2f}%" for c in counts]
                    }).sort_values("Count", ascending=False).reset_index(drop=True)
                    st.dataframe(breakdown, use_container_width=True, hide_index=True)

                # Results table
                results_df = pd.DataFrame(st.session_state["rf_X_test"], columns=st.session_state["rf_feature_names"])
                results_df["Actual_Label"] = y_test.values if hasattr(y_test, 'values') else y_test
                results_df["Prediction"] = predictions
                st.session_state["results_df"] = results_df

                st.subheader("Results Table (First 100 Rows)")
                st.dataframe(results_df.head(100), use_container_width=True)

                st.success("Random Forest detection complete. View **Evaluation** for detailed metrics.")

        else:
            st.markdown(
                "**Isolation Forest** is an unsupervised anomaly detection algorithm that "
                "identifies suspicious traffic without requiring labelled training data."
            )

            contamination = st.slider(
                "Contamination Parameter",
                min_value=0.01,
                max_value=0.50,
                value=0.10,
                step=0.01,
                help="Expected proportion of anomalies in the dataset (0.01 to 0.50)"
            )

            if st.button("Run Isolation Forest Detection", type="primary", use_container_width=True):
                with st.spinner("Preparing data..."):
                    X_train, X_test, y_train, y_test, scaler = prepare_data(
                        features_df, labels, binary=True
                    )
                    st.session_state["scaler"] = scaler

                progress_bar = st.progress(0, text="Training Isolation Forest model...")
                progress_bar.progress(20, text="Training Isolation Forest model...")

                predictions, anomaly_scores = run_isolation_forest(
                    X_train, X_test, contamination=contamination
                )

                progress_bar.progress(80, text="Generating results...")

                st.session_state["if_predictions"] = predictions
                st.session_state["if_scores"] = anomaly_scores
                st.session_state["if_y_test"] = y_test
                st.session_state["if_X_test"] = X_test
                st.session_state["if_feature_names"] = st.session_state["feature_names"]
                st.session_state["detection_method"] = "Isolation Forest"

                progress_bar.progress(100, text="Detection complete.")

            # Display IF results
            if "if_predictions" in st.session_state and st.session_state.get("detection_method") == "Isolation Forest":
                predictions = st.session_state["if_predictions"]
                y_test = st.session_state["if_y_test"]

                st.divider()
                st.subheader("Detection Summary")

                total = len(predictions)
                normal = int(np.sum(predictions == "Normal"))
                suspicious = int(np.sum(predictions == "Suspicious"))

                sum_cols = st.columns(3)
                sum_cols[0].metric("Total Records", f"{total:,}")
                sum_cols[1].metric("Normal", f"{normal:,}", f"{normal/total*100:.2f}%")
                sum_cols[2].metric("Suspicious", f"{suspicious:,}", f"{suspicious/total*100:.2f}%")

                # Results table
                results_df = pd.DataFrame(st.session_state["if_X_test"], columns=st.session_state["if_feature_names"])
                results_df["Actual_Label"] = y_test.values if hasattr(y_test, 'values') else y_test
                results_df["Prediction"] = predictions
                results_df["Anomaly_Score"] = st.session_state["if_scores"]
                st.session_state["results_df"] = results_df

                st.subheader("Results Table (First 100 Rows)")
                st.dataframe(results_df.head(100), use_container_width=True)

                st.success("Isolation Forest detection complete. View **Evaluation** for detailed metrics.")


# Evaluation Page
elif page == "Evaluation":
    st.title("Evaluation")
    st.divider()

    has_rf = "rf_predictions" in st.session_state
    has_if = "if_predictions" in st.session_state

    if not has_rf and not has_if:
        st.warning("Please run detection first. Navigate to **Detection** in the sidebar.")
    else:
        # Random Forest evaluation
        if has_rf:
            st.subheader("Random Forest Evaluation")

            y_true = st.session_state["rf_y_test"]
            y_pred = st.session_state["rf_predictions"]

            metrics, cm, report_str = evaluate_supervised(y_true, y_pred)

            metric_cols = st.columns(4)
            metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
            metric_cols[1].metric("Precision", f"{metrics['precision']:.4f}")
            metric_cols[2].metric("Recall", f"{metrics['recall']:.4f}")
            metric_cols[3].metric("F1-Score", f"{metrics['f1_score']:.4f}")

            st.session_state["rf_metrics"] = metrics

            eval_col1, eval_col2 = st.columns(2)

            with eval_col1:
                st.markdown("##### Confusion Matrix")
                unique_labels = sorted(set(list(y_true) + list(y_pred)))
                fig = viz.plot_confusion_matrix(cm, labels=unique_labels)
                st.plotly_chart(fig, use_container_width=True)

            with eval_col2:
                st.markdown("##### Feature Importance (Top 15)")
                if "rf_importances" in st.session_state and "rf_feature_names" in st.session_state:
                    fig = viz.plot_feature_importance(
                        st.session_state["rf_importances"],
                        st.session_state["rf_feature_names"]
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Classification report as a proper table
            st.markdown("##### Classification Report")
            from sklearn.metrics import classification_report as cls_report
            report_dict = cls_report(y_true, y_pred, zero_division=0, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            # Round numeric columns
            for col in ["precision", "recall", "f1-score"]:
                if col in report_df.columns:
                    report_df[col] = report_df[col].astype(float).round(4)
            if "support" in report_df.columns:
                report_df["support"] = report_df["support"].astype(int)
            st.dataframe(report_df, use_container_width=True)

            st.divider()

        # Isolation Forest evaluation
        if has_if:
            st.subheader("Isolation Forest Evaluation")

            y_true = st.session_state["if_y_test"]
            y_pred = st.session_state["if_predictions"]

            metrics, cm = evaluate_unsupervised(y_true, y_pred)

            metric_cols = st.columns(4)
            metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
            metric_cols[1].metric("Precision", f"{metrics['precision']:.4f}")
            metric_cols[2].metric("Recall", f"{metrics['recall']:.4f}")
            metric_cols[3].metric("F1-Score", f"{metrics['f1_score']:.4f}")

            st.session_state["if_metrics"] = metrics

            eval_col1, eval_col2 = st.columns(2)

            with eval_col1:
                st.markdown("##### Confusion Matrix")
                fig = viz.plot_confusion_matrix(cm, labels=["Normal", "Suspicious"])
                st.plotly_chart(fig, use_container_width=True)

            with eval_col2:
                st.markdown("##### Classification Report")
                from sklearn.metrics import classification_report as cls_report
                y_true_mapped = np.array([
                    "Normal" if str(lbl).upper() == "BENIGN" else "Suspicious"
                    for lbl in (y_true.values if hasattr(y_true, 'values') else y_true)
                ])
                report_dict = cls_report(y_true_mapped, y_pred, zero_division=0, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                for col in ["precision", "recall", "f1-score"]:
                    if col in report_df.columns:
                        report_df[col] = report_df[col].astype(float).round(4)
                if "support" in report_df.columns:
                    report_df["support"] = report_df["support"].astype(int)
                st.dataframe(report_df, use_container_width=True)

            st.divider()

        # Comparison if both methods have been run
        if has_rf and has_if:
            st.subheader("Method Comparison")
            rf_metrics = st.session_state.get("rf_metrics", {})
            if_metrics = st.session_state.get("if_metrics", {})

            comparison_df = compare_methods(rf_metrics, if_metrics)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            fig = viz.plot_detection_comparison(rf_metrics, if_metrics)
            st.plotly_chart(fig, use_container_width=True)


# Export Page
elif page == "Export":
    st.title("Export Results")
    st.divider()

    if "results_df" not in st.session_state:
        st.warning("No detection results to export. Please run detection first.")
    else:
        results_df = st.session_state["results_df"]

        st.markdown(
            f"Detection results are ready for download. "
            f"The file contains **{len(results_df):,} records** with features, "
            f"actual labels, and predictions."
        )

        col1, col2 = st.columns(2)
        col1.metric("Total Records", f"{len(results_df):,}")

        if "Prediction" in results_df.columns:
            suspicious_mask = results_df["Prediction"].astype(str).str.upper().isin(
                ["SUSPICIOUS", "MALICIOUS"]
            ) | ~results_df["Prediction"].astype(str).str.upper().isin(["NORMAL", "BENIGN"])
            col2.metric("Suspicious Records", f"{int(suspicious_mask.sum()):,}")

        st.subheader("Results Preview")
        st.dataframe(results_df.head(20), use_container_width=True)

        csv_data = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name="detection_results.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

        st.success("Click the button above to download the detection results.")
