import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("Data.csv")  # Replace with your dataframe or model predictions

# 1. Compute Ratios
# Handle missing values automatically (pandas skips NaN)
df["Rt"] = df["Ts_Mpa"] / df["Cs_Mpa"]  # Tensile Ratio
df["Rf"] = df["Fs_Mpa"] / df["Cs_Mpa"]  # Flexural Ratio

# 2. Define ACI Bounds and Absolute Physical Bounds
ACI_RT_MIN, ACI_RT_MAX = 0.06, 0.18
ACI_RF_MIN, ACI_RF_MAX = 0.08, 0.25

# 3. Identify Violations
# Tensile Violations
ts_valid_count = df["Ts_Mpa"].notna().sum()
ts_violations_aci = df[
    (df["Ts_Mpa"].notna()) & ((df["Rt"] < ACI_RT_MIN) | (df["Rt"] > ACI_RT_MAX))
]
ts_impossible = df[(df["Ts_Mpa"].notna()) & (df["Ts_Mpa"] >= df["Cs_Mpa"])]

# Flexural Violations
fs_valid_count = df["Fs_Mpa"].notna().sum()
fs_violations_aci = df[
    (df["Fs_Mpa"].notna()) & ((df["Rf"] < ACI_RF_MIN) | (df["Rf"] > ACI_RF_MAX))
]
fs_impossible = df[(df["Fs_Mpa"].notna()) & (df["Rf"] > 0.35)]

# 4. Compute Physical Violation Rates (PVR)
pvr_ts_aci = (len(ts_violations_aci) / ts_valid_count) * 100
pvr_fs_aci = (len(fs_violations_aci) / fs_valid_count) * 100

pvr_ts_imp = (len(ts_impossible) / ts_valid_count) * 100
pvr_fs_imp = (len(fs_impossible) / fs_valid_count) * 100

# 5. Output Results
print("=" * 60)
print("PHYSICAL & ACI CODE RATIO AUDIT REPORT")
print("=" * 60)

print(
    f"Tensile Strength (Ts) Valid Samples: {ts_valid_count} / {len(df)}[cite: 6]"
)
print(
    f" - Mean Rt (Ts/Cs): {df['Rt'].mean():.4f} (Min: {df['Rt'].min():.4f}, Max: {df['Rt'].max():.4f})"
)
print(
    f" - ACI Code Violations ({ACI_RT_MIN} <= Rt <= {ACI_RT_MAX}): {len(ts_violations_aci)} ({pvr_ts_aci:.2f}%)"
)
print(
    f" - Physically Impossible (Ts >= Cs): {len(ts_impossible)} ({pvr_ts_imp:.2f}%)"
)

print("\n" + "-" * 60 + "\n")

print(
    f"Flexural Strength (Fs) Valid Samples: {fs_valid_count} / {len(df)}[cite: 6]"
)
print(
    f" - Mean Rf (Fs/Cs): {df['Rf'].mean():.4f} (Min: {df['Rf'].min():.4f}, Max: {df['Rf'].max():.4f})"
)
print(
    f" - ACI Code Violations ({ACI_RF_MIN} <= Rf <= {ACI_RF_MAX}): {len(fs_violations_aci)} ({pvr_fs_aci:.2f}%)"
)
print(
    f" - Physically Impossible (Rf > 0.35): {len(fs_impossible)} ({pvr_fs_imp:.2f}%)"
)

print("=" * 60)