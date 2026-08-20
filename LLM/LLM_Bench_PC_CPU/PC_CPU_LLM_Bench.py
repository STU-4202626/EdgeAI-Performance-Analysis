import subprocess   # Run cmd prompts within python
import csv
import re
from codecarbon import EmissionsTracker
import logging

logging.getLogger("codecarbon").disabled = True


device_name = "AMD Ryzen 7 6800H"
model_name = "gemma3:1b"

# Benchmark Prompts
knowledge_prompt = """
Explain the main causes of climate change and discuss three strategies that governments can implement to reduce greenhouse gas emissions. Provide a structured response of approximately 250 words.
"""

summarisation_prompt = """
Article:
Edge AI refers to the deployment of artificial intelligence directly on local devices where data is generated, rather than relying entirely on remote cloud servers. Instead of sending information across the internet for processing, AI models operate on the device itself. Examples include smartphone voice assistants, smart cameras that detect motion, and wearable health monitors that provide immediate feedback.
One of the main reasons for using Edge AI is speed. Because data does not need to travel to a remote server and back, decisions can be made almost instantly. This low latency is particularly important in applications such as autonomous vehicles, industrial automation, and healthcare systems where delays could reduce effectiveness or even create safety risks.
Edge AI also improves privacy and security. Since sensitive data can remain on the device, organisations and users reduce the risks associated with transmitting personal information across networks. This makes Edge AI attractive for applications involving financial, medical, or personal data.
Another advantage is reduced dependence on internet connectivity. Devices can continue operating even when network access is limited or unavailable. In addition, transmitting less data to cloud services can lower operating costs and reduce bandwidth usage.
To enable AI on resource-constrained devices, machine learning models are often compressed and optimised. These models can run on hardware such as smartphones, smart cameras, drones, robots, industrial sensors, and single-board computers like the Raspberry Pi. More powerful edge systems may also use specialised hardware accelerators to improve AI performance while maintaining energy efficiency.
Common applications of Edge AI include facial recognition on smartphones, inventory monitoring in retail stores, traffic analysis in smart cities, predictive maintenance in industrial environments, and wearable healthcare devices that monitor patient conditions in real time.
Although Edge AI provides many benefits, it also presents challenges. Edge devices typically have less processing power, memory, and storage than cloud servers. Developers must therefore balance model accuracy, response speed, and resource consumption when deploying AI applications at the edge.
As artificial intelligence continues to advance, Edge AI is expected to play an increasingly important role in enabling fast, private, and reliable intelligent systems. Future applications are likely to include advanced robotics, smart transportation networks, intelligent manufacturing systems, and large-scale Internet of Things ecosystems.

Summarise the provided article in approximately 150 words while retaining the main ideas, key findings, and conclusions.
"""

logical_reasoning_prompt = """
A farmer needs to transport a fox, a chicken, and a bag of grain across a river. His boat can carry only himself and one item at a time. If left unattended, the fox will eat the chicken and the chicken will eat the grain. Describe the sequence of crossings required to transport all three safely and explain your reasoning step by step.
"""

mathematical_reasoning_prompt = """
A company sells laptops for R12,000 each. The production cost per laptop is R8,000 and monthly fixed costs are R200,000. Determine how many laptops must be sold to break even. Show all calculations and explain your reasoning.
"""

code_generation_prompt = """
Write a Python program that reads a text file, counts the frequency of each word, and displays the ten most common words in descending order. Include comments explaining the logic and provide an example of how the program can be executed.
"""


# Extract benchmark metrics from Ollama output
def extract_metrics(output):

    metrics = {}

    lines = output.splitlines()

    for line in lines:

        if ":" in line:
            parts = line.split(":", 1)
            key = parts[0].strip()
            value = parts[1].strip()
            metrics[key] = value

    return metrics


# Convert Ollama time strings into seconds for averaging
def convert_to_seconds(time_string):

    time_string = time_string.strip()

    # Milliseconds
    if "ms" in time_string:
        return float(time_string.replace("ms", "").strip()) / 1000

    # Minutes + seconds
    elif "m" in time_string and "s" in time_string:
        minutes_part = time_string.split("m")[0]
        seconds_part = time_string.split("m")[1].replace("s", "")

        return (float(minutes_part) * 60) + float(seconds_part)

    # Seconds only
    elif "s" in time_string:

        return float(time_string.replace("s", "").strip())

    return 0

# Run Ollama benchmark
def run_prompt(prompt_text):

    command = ["ollama", "run", model_name, "--verbose", prompt_text]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")

    # Capture everything as shown in CMD
    output = result.stdout

    # Remove ANSI characters
    output = re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', output)

    # Remove spinning characters from Ollama's streaming output
    output = re.sub(r'[⠁-⣿]', '', output)

    return output


# Run workload 4 times (1 warmup + 3 measured runs)
def benchmark_workload(workload_category, prompt_text):

    print(f"\n==========================================")
    print(f"RUNNING {workload_category} workload...")
    print(f"==========================================\n")
    outputs = []
    emissions_list = []

    for run in range(1, 5):
        if run == 1:
            run_type = "Warmup"
        else:
            run_type = f"Measured_{run-1}"

        print(f"\n-----------------{run_type}--------------------\n")

        tracker = EmissionsTracker(
            project_name=f"{device_name}_{workload_category}_{run_type}",
            measure_power_secs=1,
            log_level="error",
            gpu_ids=[]
        )

        tracker.start()

        output = run_prompt(prompt_text)

        emissions = tracker.stop()

        outputs.append(output)
        emissions_list.append(emissions)

        metrics = extract_metrics(output)

        print(
            f"total duration:       {metrics['total duration']}\n"
            f"load duration:        {metrics['load duration']}\n"
            f"prompt eval count:    {metrics['prompt eval count']}\n"
            f"prompt eval duration: {metrics['prompt eval duration']}\n"
            f"prompt eval rate:     {metrics['prompt eval rate']}\n"
            f"eval count:           {metrics['eval count']}\n"
            f"eval duration:        {metrics['eval duration']}\n"
            f"eval rate:            {metrics['eval rate']}\n"
            f"Carbon Emissions:     {emissions:.3e} kgCO2e\n"
        )

    split_text = outputs[0].split("total duration:")
    model_response = split_text[0]
    response = model_response.replace("\n", " ")

    metrics2 = extract_metrics(outputs[1])   # Measured_1
    metrics3 = extract_metrics(outputs[2])   # Measured_2
    metrics4 = extract_metrics(outputs[3])   # Measured_3

    avg_total_duration = (convert_to_seconds(metrics2["total duration"]) + convert_to_seconds(metrics3["total duration"]) + convert_to_seconds(metrics4["total duration"])) / 3

    avg_load_duration = (convert_to_seconds(metrics2["load duration"]) + convert_to_seconds(metrics3["load duration"]) + convert_to_seconds(metrics4["load duration"])) / 3

    avg_prompt_eval_count = (float(metrics2["prompt eval count"].split()[0]) + float(metrics3["prompt eval count"].split()[0]) + float(metrics4["prompt eval count"].split()[0])) / 3

    avg_prompt_eval_duration = (convert_to_seconds(metrics2["prompt eval duration"]) + convert_to_seconds(metrics3["prompt eval duration"]) + convert_to_seconds(metrics4["prompt eval duration"])) / 3

    avg_prompt_eval_rate = (float(metrics2["prompt eval rate"].split()[0]) + float(metrics3["prompt eval rate"].split()[0]) + float(metrics4["prompt eval rate"].split()[0])) / 3

    avg_eval_count = (float(metrics2["eval count"].split()[0]) + float(metrics3["eval count"].split()[0]) + float(metrics4["eval count"].split()[0])) / 3

    avg_eval_duration = (convert_to_seconds(metrics2["eval duration"]) + convert_to_seconds(metrics3["eval duration"]) + convert_to_seconds(metrics4["eval duration"])) / 3

    avg_eval_rate = (float(metrics2["eval rate"].split()[0]) + float(metrics3["eval rate"].split()[0]) + float(metrics4["eval rate"].split()[0])) / 3

    avg_emissions = (emissions_list[1] + emissions_list[2] + emissions_list[3]) / 3

    warmup_metrics = ("total duration:" + outputs[0].split("total duration:")[1] + f"Carbon Emissions: {emissions_list[0]:.3e} kgCO2e")
    
    verbose_metrics1 = ("total duration:" + outputs[1].split("total duration:")[1] + f"Carbon Emissions: {emissions_list[1]:.3e} kgCO2e")

    verbose_metrics2 = ("total duration:" + outputs[2].split("total duration:")[1] + f"Carbon Emissions: {emissions_list[2]:.3e} kgCO2e")

    verbose_metrics3 = ("total duration:" + outputs[3].split("total duration:")[1] + f"Carbon Emissions: {emissions_list[3]:.3e} kgCO2e")


    avg_metrics = (
        f"Avg Total Duration: {avg_total_duration:.3f}s\n"
        f"Avg Load Duration: {avg_load_duration:.3f}s\n"
        f"Avg Prompt Eval Count: {avg_prompt_eval_count:.2f} token(s)\n"
        f"Avg Prompt Eval Duration: {avg_prompt_eval_duration:.3f}s\n"
        f"Avg Prompt Eval Rate: {avg_prompt_eval_rate:.2f} tokens/s\n"
        f"Avg Eval Count: {avg_eval_count:.2f} token(s)\n"
        f"Avg Eval Duration: {avg_eval_duration:.3f}s\n"
        f"Avg Eval Rate: {avg_eval_rate:.2f} tokens/s\n"
        f"Avg Carbon Emissions: {avg_emissions:.3e} kgCO2e"
    )

    return [device_name, model_name, workload_category, prompt_text, response, warmup_metrics, verbose_metrics1, verbose_metrics2, verbose_metrics3, avg_metrics]

# Run benchmarks
results = []

results.append(benchmark_workload("Knowledge Retrieval", knowledge_prompt))
results.append(benchmark_workload("Summarisation", summarisation_prompt))
results.append(benchmark_workload("Logical Reasoning", logical_reasoning_prompt))
results.append(benchmark_workload("Mathematical Reasoning", mathematical_reasoning_prompt))
results.append(benchmark_workload("Code Generation", code_generation_prompt))

# Save results to CSV
with open("CPU_LLM_bench.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["device_name", "model_name", "workload_category", "prompt_text", "model_response", "warmup_metrics", "verbose_metrics1", "verbose_metrics2", "verbose_metrics3", "avg_metrics"])
    writer.writerows(results)

print("\nBenchmark completed.")
print("Results saved to CPU_LLM_bench.csv\n")