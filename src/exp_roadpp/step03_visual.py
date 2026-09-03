import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def visualize_rule_aggregation_results(dataset_summary, output_dir):
    if not dataset_summary:
        print("No dataset summary to visualize.")
        return

    # Example visualization: test accuracy per class
    test_accuracy_per_class = dataset_summary.get("test_accuracy_per_class", {})
    if not test_accuracy_per_class:
        print("No test accuracy per class to visualize.")
        return

    classes = list(test_accuracy_per_class.keys())
    accuracies = list(test_accuracy_per_class.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=accuracies, y=classes)
    plt.xlabel("Test Accuracy")
    plt.ylabel("Class Label")
    plt.title("Test Accuracy per Class")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "test_accuracy_per_class.png")
    plt.close()