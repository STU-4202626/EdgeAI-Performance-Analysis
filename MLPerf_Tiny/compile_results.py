import os
import sys
import pandas as pd

# Device folder is supplied when running the script.
#
# Examples:
#   python compile_results.py MLPerf_Tiny_PC_CPU
#   python compile_results.py MLPerf_Tiny_PC_GPU
#   python compile_results.py MLPerf_Tiny_RaspberryPi5
#   python compile_results.py MLPerf_Tiny_Android

if len(sys.argv) != 2:
    print("Usage:")
    print("python compile_results.py <device_folder>")
    sys.exit(1)


DEVICE = sys.argv[1]

# Main MLPerf Tiny directory.
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Location of the selected device folder.
DEVICE_DIR = os.path.join(
    BASE_DIR,
    DEVICE
)

if not os.path.isdir(DEVICE_DIR):
    print(f"\nERROR: Device folder not found: {DEVICE}")
    sys.exit(1)


# Benchmark workloads used by the project.
BENCHMARKS = [
    "image_classification",
    "keyword_spotting",
    "visual_wake_words"
]


# Filename prefix used by each device.
DEVICE_PREFIXES = {
    "MLPerf_Tiny_PC_CPU": "CPU",
    "MLPerf_Tiny_PC_GPU": "GPU",
    "MLPerf_Tiny_RaspberryPi5": "Pi5",
    "MLPerf_Tiny_Android": "Android"
}


if DEVICE not in DEVICE_PREFIXES:
    print(f"\nERROR: No filename prefix configured for: {DEVICE}")
    sys.exit(1)


DEVICE_PREFIX = DEVICE_PREFIXES[DEVICE]


# Output filename for each device.
SUMMARY_FILENAMES = {
    "MLPerf_Tiny_PC_CPU":
        "mlperf_tiny_PC_CPU_summary.csv",

    "MLPerf_Tiny_PC_GPU":
        "mlperf_tiny_PC_GPU_summary.csv",

    "MLPerf_Tiny_RaspberryPi5":
        "mlperf_tiny_RaspberryPi5_summary.csv",

    "MLPerf_Tiny_Android":
        "mlperf_tiny_Android_summary.csv"
}


# Calculate the average result for one benchmark CSV.
def calculate_average(df):

    # Remove the warm-up run before calculating the averages.
    df = df[df["Run"] != "Warm-up"].copy()

    if df.empty:
        return None

    # Average the values from Runs 1-3.
    samples_evaluated = int(
        df["Samples Evaluated"].mean()
    )

    average_inference_time = (
        df["Inference Time (s)"].mean()
    )

    # Calculate throughput from the average inference time.
    average_throughput = (
        samples_evaluated / average_inference_time
        if average_inference_time > 0
        else 0
    )

    # Build the summary row.
    result = {

        "Device":
            df["Device"].iloc[0],

        "Workload":
            df["Workload"].iloc[0],

        "Dataset":
            df["Dataset"].iloc[0],

        "Model":
            df["Model"].iloc[0],

        "Samples Evaluated":
            samples_evaluated,

        "Repetitions":
            round(df["Repetitions"].mean(), 2),

        "Target Measurement Time (s)":
            round(
                df["Target Measurement Time (s)"].mean(),
                3
            ),

        "Total Measurement Time (s)":
            round(
                df["Total Measurement Time (s)"].mean(),
                3
            ),

        "Accuracy":
            round(
                df["Accuracy"].mean(),
                4
            ),

        "Precision":
            round(
                df["Precision"].mean(),
                4
            ),

        "Recall":
            round(
                df["Recall"].mean(),
                4
            ),

        "F1-Score":
            round(
                df["F1-Score"].mean(),
                4
            ),

        "ROC-AUC":
            round(
                df["ROC-AUC"].mean(),
                4
            ),

        "Inference Time (s)":
            round(
                average_inference_time,
                3
            ),

        "Throughput":
            round(
                average_throughput,
                2
            ),

        # Keep carbon emissions in scientific notation.
        "Carbon Emissions (kgCO2e)":
            f"{df['Carbon Emissions (kgCO2e)'].mean():.3e}",

        "MLPerf Target":
            df["MLPerf Target"].iloc[0],

        "Benchmark Status":
            (
                "PASSED"
                if (df["Benchmark Status"] == "PASSED").all()
                else "FAILED"
            )
    }

    return result


def main():

    print("\n" + "=" * 70)
    print("MLPerf Tiny Benchmark Results Compiler")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    results = []

    # Process each workload.
    for benchmark in BENCHMARKS:

        print(f"\nScanning: {benchmark}")

        # Pi 5 keeps the full and quantized results in separate folders.
        if DEVICE == "MLPerf_Tiny_RaspberryPi5":

            results_folders = [
                "results_full",
                "results_quant"
            ]

        else:

            # The other devices have one results folder.
            results_folders = [
                "results"
            ]

        # Check each results folder for the benchmark CSV.
        for results_folder in results_folders:

            results_dir = os.path.join(
                DEVICE_DIR,
                "training",
                benchmark,
                results_folder
            )

            if not os.path.isdir(results_dir):
                print(
                    f"  Results folder not found: {results_folder}"
                )
                continue

            # Pi 5 has _full and _quant at the end of the filename.
            if DEVICE == "MLPerf_Tiny_RaspberryPi5":

                expected_filename = (
                    f"{DEVICE_PREFIX}_{benchmark}_"
                    f"benchmark_results_"
                    f"{'full' if results_folder == 'results_full' else 'quant'}"
                    ".csv"
                )

            else:

                expected_filename = (
                    f"{DEVICE_PREFIX}_{benchmark}_"
                    "benchmark_results.csv"
                )

            csv_path = os.path.join(
                results_dir,
                expected_filename
            )

            if not os.path.isfile(csv_path):
                print(
                    f"  Result file not found: {expected_filename}"
                )
                continue

            print(f"  Processing: {expected_filename}")

            try:
                df = pd.read_csv(csv_path)

            except Exception as error:
                print(
                    f"  WARNING: Could not read CSV: {error}"
                )
                continue

            # Make sure the CSV contains the expected columns.
            required_columns = {
                "Run",
                "Device",
                "Workload",
                "Dataset",
                "Model",
                "Samples Evaluated",
                "Repetitions",
                "Target Measurement Time (s)",
                "Total Measurement Time (s)",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "ROC-AUC",
                "Inference Time (s)",
                "Throughput",
                "Carbon Emissions (kgCO2e)",
                "MLPerf Target",
                "Benchmark Status"
            }

            missing_columns = (
                required_columns -
                set(df.columns)
            )

            if missing_columns:
                print(
                    "  WARNING: Missing columns: "
                    + ", ".join(sorted(missing_columns))
                )
                continue

            # Calculate the average using Runs 1-3.
            result = calculate_average(df)

            if result is None:
                print(
                    "  WARNING: No measured runs found."
                )
                continue

            results.append(result)

            measured_runs = len(
                df[df["Run"] != "Warm-up"]
            )

            print(
                f"  Measured runs: {measured_runs}"
            )

    # Stop if no valid benchmark results were found.
    if not results:
        print("\nNo benchmark results were found.")
        sys.exit(1)

    # Create the device summary.
    summary = pd.DataFrame(results)

    # Keep workloads in a consistent order.
    workload_order = {
        "Image Classification": 1,
        "Keyword Spotting": 2,
        "Visual Wake Words": 3
    }

    summary["_order"] = summary["Workload"].map(
        workload_order
    ).fillna(99)

    summary = summary.sort_values(
        by=["_order", "Model"]
    )

    summary = summary.drop(
        columns=["_order"]
    )

    print(
        f"\nBenchmark results compiled: {len(summary)}"
    )

    # Save the device summary.
    output_file = os.path.join(
        DEVICE_DIR,
        SUMMARY_FILENAMES[DEVICE]
    )

    summary.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("Compilation complete")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(
        f"Benchmark results: {len(results)}"
    )
    print(
        f"Output file: {output_file}"
    )

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()