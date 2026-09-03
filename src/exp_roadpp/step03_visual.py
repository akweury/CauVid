import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.exp_roadpp import utils_data


def visualize_rule_aggregation_results(dataset_path, dataset_summary, output_dir):
    if not dataset_summary:
        print("No dataset summary to visualize.")
        return

    # Example visualization: test accuracy per class
    test_accuracy_per_class = dataset_summary.get("test_accuracy_per_class", {})
    if not test_accuracy_per_class:
        print("No test accuracy per class to visualize.")
        return

    labels = utils_data.load_json(Path(dataset_path)/ "gt" / "label.json")["all_input_labels"]
    class_ids = list(test_accuracy_per_class.keys())
    classes = [labels[int(cid)] for cid in class_ids]

    accuracies = list(test_accuracy_per_class.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=accuracies)
    plt.xlabel("Class Label")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy per Class")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "test_accuracy_per_class.png")
    plt.close()