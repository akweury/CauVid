import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
def visualize_rule_aggregation_results(dataset_path, dataset_summary, output_dir):
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
    plt.savefig(Path(output_dir) / "test_accuracy_per_class.png")
    plt.close()