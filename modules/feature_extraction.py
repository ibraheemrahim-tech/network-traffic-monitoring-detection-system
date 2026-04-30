"""
Feature Extraction Module
Selects key traffic features from CIC-IDS-2017 and creates derived features.
"""

import pandas as pd
import numpy as np


# CIC-IDS-2017 key feature names (normalised: lowercase, underscores)
TARGET_FEATURES = [
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "total_length_of_fwd_packets",
    "total_length_of_bwd_packets",
    "flow_bytes_s",
    "flow_packets_s",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_packet_length_mean",
    "bwd_packet_length_mean",
    "packet_length_mean",
    "packet_length_std",
    "average_packet_size",
    "down_up_ratio",
]

# Common label column names (normalised)
LABEL_COLUMNS = ["label", "labels", "class", "attack", "category"]


def _find_matching_columns(df_columns, target_names):
    """
    Find columns in the DataFrame that match target feature names.
    Uses fuzzy matching to handle slight naming variations.
    """
    matched = []
    df_cols_lower = {col: col for col in df_columns}

    for target in target_names:
        # Exact match
        if target in df_cols_lower:
            matched.append(target)
            continue

        # Try replacing common variations
        for col in df_columns:
            col_clean = col.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            if col_clean == target:
                matched.append(col)
                break

    return matched


def _find_label_column(df):
    """Find the label column in the DataFrame."""
    for col in df.columns:
        if col.lower().strip() in LABEL_COLUMNS:
            return col
    return None


def extract_features(df):
    """
    Extract key traffic features from the dataset.
    """
    try:
        df = df.copy()

        # Find label column
        label_col = _find_label_column(df)
        labels = None
        if label_col:
            labels = df[label_col].copy()

        # Find matching feature columns
        matched_features = _find_matching_columns(df.columns, TARGET_FEATURES)

        # If few matches found, try to use all numeric columns (excluding label)
        if len(matched_features) < 3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude columns that are likely identifiers
            exclude_patterns = ["id", "timestamp", "time", "date", "index"]
            matched_features = [
                col for col in numeric_cols
                if not any(pat in col.lower() for pat in exclude_patterns)
            ]

        if len(matched_features) == 0:
            return None, None, []

        features_df = df[matched_features].copy()

        # Create derived features
        # packets_per_second = total packets / flow duration
        total_fwd = _get_column(df, ["total_fwd_packets"])
        total_bwd = _get_column(df, ["total_backward_packets"])
        flow_dur = _get_column(df, ["flow_duration"])

        if total_fwd is not None and total_bwd is not None and flow_dur is not None:
            total_packets = total_fwd + total_bwd
            safe_duration = flow_dur.replace(0, np.nan)
            packets_per_second = total_packets / safe_duration
            packets_per_second = packets_per_second.fillna(0)
            features_df["packets_per_second"] = packets_per_second

        # bytes_per_packet = total bytes / total packets
        total_fwd_len = _get_column(df, ["total_length_of_fwd_packets"])
        total_bwd_len = _get_column(df, ["total_length_of_bwd_packets"])

        if total_fwd_len is not None and total_bwd_len is not None:
            total_bytes = total_fwd_len + total_bwd_len
            if total_fwd is not None and total_bwd is not None:
                total_pkts = total_fwd + total_bwd
                safe_pkts = total_pkts.replace(0, np.nan)
                bytes_per_packet = total_bytes / safe_pkts
                bytes_per_packet = bytes_per_packet.fillna(0)
                features_df["bytes_per_packet"] = bytes_per_packet

        # Replace any remaining inf/nan values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)

        feature_names = features_df.columns.tolist()

        return features_df, labels, feature_names

    except Exception as e:
        return None, None, []


def _get_column(df, possible_names):
    """
    Get a column from the DataFrame by trying multiple possible names.
    """
    for name in possible_names:
        if name in df.columns:
            return df[name]
        # Try with common variations
        for col in df.columns:
            if col.lower().replace(" ", "_") == name:
                return df[col]
    return None
