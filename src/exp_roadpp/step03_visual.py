import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from src.exp_roadpp import utils_data

def visualize_baseline_results(baseline_results, output_dir):
    if not baseline_results:
        print("No baseline results to visualize.")
        return

    # Example visualization: test accuracy per class
    test_accuracy_per_class = baseline_results.get("test_accuracy_per_class", {})
    if not test_accuracy_per_class:
        print("No test accuracy per class to visualize.")
        return

    labels = utils_data.load_json(Path(output_dir).parent.parent / "gt" / "label.json")
    av_action_labels = labels["all_av_action_labels"]
    class_ids = list(test_accuracy_per_class.keys())
    classes = [av_action_labels[int(cid)] for cid in class_ids if cid != -1]

    accuracies = [test_accuracy_per_class[cid] for cid in class_ids if cid != -1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=accuracies)
    plt.xlabel("Class Label")
    plt.ylabel("Test Accuracy")
    plt.title("Baseline Test Accuracy per Class")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "baseline_test_accuracy_per_class.png")
    plt.close()
def visualize_rule_aggregation_results(dataset_path, dataset_summary, output_dir, suffix):
    if not dataset_summary:
        print("No dataset summary to visualize.")
        return

    # Example visualization: test accuracy per class
    test_accuracy_per_class = dataset_summary.get("test_accuracy_per_class", {})
    if not test_accuracy_per_class:
        print("No test accuracy per class to visualize.")
        return

    labels = utils_data.load_json(Path(dataset_path)/ "gt" / "label.json")
    av_action_labels = labels["all_av_action_labels"]
    class_ids = list(test_accuracy_per_class.keys())
    classes = [av_action_labels[int(cid)] for cid in class_ids if cid != -1]

    accuracies = [test_accuracy_per_class[cid] for cid in class_ids if cid != -1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=accuracies)
    plt.xlabel("Class Label")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy per Class")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"test_accuracy_per_class_{suffix}.png")
    plt.close()



def visual_bar(clauses, output_dir, filename):
    if not clauses:
        print("No clauses to visualize.")
        return

    # two subplots, left side shows the support frequencies, and right side shows the coverage frequencies
    # use bar charts to show
    # visualize the number of clauses by their frequency, 
    # x axis represents the frequencies, y axis represents the number of clauses, 
    # use 10 bins to group the frequencies
    # use log y ticks
    # the bin range should be increasing by frequency increasing, so it start from 1, then 2, 4, 8, 16,...
    # each bin width should be the same, and has its own tick label

    support_frequencies = [] 
    coverage_frequencies = [] 
    for _, clause_data in clauses.items():
        total_support = sum(clause_data["support_coverage"][vid]["support"] for vid in clause_data["support_coverage"])
        total_coverage = sum(clause_data["support_coverage"][vid]["coverage"] for vid in clause_data["support_coverage"])
        support_frequencies.append(total_support)
        coverage_frequencies.append(total_coverage)
    
    bins = [2**i for i in range(int(np.log2(max(support_frequencies))) + 2)] if support_frequencies else [1, 2] 

    coverage_bins = [2**i for i in range(int(np.log2(max(coverage_frequencies))) + 2)] if coverage_frequencies else [1, 2] 

    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    counts, _ = np.histogram(support_frequencies, bins=bins)
    positions = np.arange(len(counts))  # equal-width, evenly spaced bars

    plt.figure(figsize=(12,6))
    plt.bar(positions, counts, width=0.8, color="wheat")
    
    for x, count in zip(positions, counts):
        if count > 0:
            plt.text(x, count, f"{int(count)}", ha="center", va="bottom", fontsize=20)
    plt.xlabel("Clause Frequency", fontdict={"size": 26})
    plt.xticks(positions, bin_labels, rotation=45, ha="right")
    plt.ylabel("Number of Clauses", fontdict={"size": 26})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.yscale("log")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.title("Clause Support Frequency Histogram", fontdict={"size": 30})
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{filename}_support.png")
    plt.close()

    # Coverage frequency histogram
    bin_labels = [f"{coverage_bins[i]}-{coverage_bins[i+1]}" for i in range(len(coverage_bins) - 1)]
    counts, _ = np.histogram(coverage_frequencies, bins=coverage_bins)
    positions = np.arange(len(counts))

    plt.figure(figsize=(12,6))
    plt.bar(positions, counts, width=0.8, color="lightblue")
    
    for x, count in zip(positions, counts):
        if count > 0:
            plt.text(x, count, f"{int(count)}", ha="center", va="bottom", fontsize=20)
    plt.xlabel("Clause Frequency", fontdict={"size": 26})
    plt.xticks(positions, bin_labels, rotation=45, ha="right")
    plt.ylabel("Number of Clauses", fontdict={"size": 26})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.yscale("log")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.title("Clause Coverage Frequency Histogram", fontdict={"size": 30})
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{filename}_coverage.png")
    plt.close()