import os
import pandas as pd

# Main MLPerf Tiny folder.
BASE_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

# Device folders and their summary files.
DEVICE_SUMMARIES = {

    "MLPerf_Tiny_PC_CPU":
        "mlperf_tiny_PC_CPU_summary.csv",

    "MLPerf_Tiny_PC_GPU":
        "mlperf_tiny_PC_GPU_summary.csv",

    "MLPerf_Tiny_RaspberryPi5":
        "mlperf_tiny_RaspberryPi5_summary.csv",

    "MLPerf_Tiny_Android":
        "mlperf_tiny_Android_summary.csv"
}

device_results = []

print("\n" + "=" * 70)
print("MLPerf Tiny All Devices Results Compiler")
print("=" * 70)

# Read the summary produced for each device.
for folder, summary_file in DEVICE_SUMMARIES.items():

    csv_file = os.path.join(
        BASE_FOLDER,
        folder,
        summary_file
    )

    if not os.path.isfile(csv_file):

        print(
            f"\nWARNING: Summary not found for {folder}"
        )

        continue

    print(
        f"\nReading: {summary_file}"
    )

    try:

        df = pd.read_csv(csv_file)

    except Exception as error:

        print(
            f"WARNING: Could not read {summary_file}: {error}"
        )

        continue

    # The device summary contains an Average row.
    # We only want the individual benchmark results here.
    df = df[df["Device"] != "Average"].copy()

    print(
        f"  Benchmark results: {len(df)}"
    )

    device_results.append(df)

# Stop if none of the device summaries could be found.
if not device_results:

    print("\nNo benchmark summaries were found.")

    raise SystemExit(1)

# Combine all device benchmark results.
summary = pd.concat(
    device_results,
    ignore_index=True
)

# Keep the workloads in the same order used by compile_results.py.
workload_order = {

    "Image Classification": 1,

    "Keyword Spotting": 2,

    "Visual Wake Words": 3
}

summary["_workload_order"] = (
    summary["Workload"]
    .map(workload_order)
    .fillna(99)
)

# Sort by device, workload and model.
summary = summary.sort_values(
    by=[
        "Device",
        "_workload_order",
        "Model"
    ]
)

summary = summary.drop(
    columns=["_workload_order"]
)

# Keep carbon emissions in scientific notation.
summary["Carbon Emissions (kgCO2e)"] = (
    summary["Carbon Emissions (kgCO2e)"]
    .astype(float)
    .map(lambda value: f"{value:.3e}")
)

# Save the combined results.
output_file = os.path.join(
    BASE_FOLDER,
    "mlperf_tiny_all_devices_summary.csv"
)


summary.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 70)
print("Compilation complete")
print("=" * 70)

print(f"Devices combined: {len(device_results)}")

print(f"Benchmark results: {len(summary) - 1}")

print(f"Output file: {output_file}")

print("=" * 70 + "\n")