import os
import pandas as pd

def save_benchmark_results(
    results_folder,
    results_filename,
    benchmark_runs,
):

    # Store the benchmark results from all four executions.
    rows = []

    for result in benchmark_runs:

        rows.append({

            "Run": result["run"],

            "Device": result["device"],

            "Workload": result["workload"],

            "Dataset": result["dataset"],

            "Model": result["model"],

            "Samples Evaluated": result["samples"],

            "Repetitions": result["repetitions"],

            "Target Measurement Time (s)": round(result["target_measurement_time"], 3),

            "Total Measurement Time (s)": round(result["total_measurement_time"], 3),

            "Accuracy": round(result["accuracy"], 4),

            "Precision": round(result["precision"], 4),

            "Recall": round(result["recall"], 4),

            "F1-Score": round(result["f1"], 4),

            "ROC-AUC": round(result["roc_auc"], 4),

            "Inference Time (s)": round(result["inference_time"], 3),

            "Throughput": round(result["throughput"], 2),

            "Carbon Emissions (kgCO2e)": f"{result['carbon']:.3e}",

            "MLPerf Target": round(result["target"], 2),

            "Benchmark Status": result["status"],
        })

    df = pd.DataFrame(rows)

    # Save the benchmark results to a CSV file inside the benchmark results folder.
    csv_path = os.path.join(
        results_folder,
        results_filename,
    )

    df.to_csv(
        csv_path,
        index=False
    )