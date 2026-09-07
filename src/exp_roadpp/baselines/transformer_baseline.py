
from src.exp_roadpp import utils_data

"""
The transformer baseline model is implemented in this file. 
The model takes the facts of the objects of each video segment as input and predicts the av_action_id, 
i.e. the action of the autonomous vehicle in the segment.

The input includes the object class, action_ids, and loc_ids of the objects in the segment. 
The model is trained on the training set and evaluated on the test set.

"""

class TransformerModel:



    def __init__(self):
        # Initialize your Transformer model here
        pass

    def train(self, train_facts, val_facts):
        """
        Train the Transformer model on the training facts and validate on the validation facts.

        Args:
            train_facts (list): List of training facts.
            val_facts (list): List of validation facts.
        """
        # Implement the training logic here
        pass

    def evaluate(self, test_facts):
        """
        Evaluate the Transformer model on the test facts.

        Args:
            test_facts (list): List of test facts.

        Returns:
            dict: Evaluation results.
        """
        # Implement the evaluation logic here
        return {}

def prepare_dataset(train_facts, val_facts, test_facts):
    """
    Prepare the dataset for the Transformer model.

    Args:
        train_facts (list): List of training facts.
        val_facts (list): List of validation facts.
        test_facts (list): List of test facts.
    """
    # Implement the dataset preparation logic here
    pass



def run(train_facts, val_facts, test_facts, track_dir, output_dir, result_summary):
    """
    Run the Transformer baseline on the given train, validation, and test sets.

    Args:
        train_facts (list): List of training facts.
        val_facts (list): List of validation facts.
        test_facts (list): List of test facts.
        track_dir (Path): Directory containing the trajectory data.
        output_dir (Path): Directory to save the results.
        result_summary (dict): Dictionary to store the results summary.
    """
    print("\n--------- Running Transformer Baseline ----------------------\n")

    # TODO: Implement the Transformer baseline here
    transformer_results = {}  # Placeholder for actual results
    transformer_model_file = output_dir / "transformer_model.pth"  # Path to save the trained model
    transformer_result_file = output_dir / "transformer_results.json"  # Path to save the results
    if transformer_model_file.exists() and transformer_result_file.exists():
        print("Transformer model and results already exist. Loading them...")
        # Load the model and results if they already exist
        transformer_results = utils_data.load_json(transformer_result_file)
    else:
        dataset = prepare_dataset(train_facts, val_facts, test_facts)
        transformer_model = TransformerModel()  # Initialize your Transformer model here
        transformer_model.train(train_facts, val_facts)  # Train the model
        transformer_results = transformer_model.evaluate(test_facts)  # Evaluate on test set
        utils_data.save_json(transformer_results, transformer_result_file)  # Save the results
    result_summary["transformer"] = transformer_results
    print("\n--------- Transformer Baseline Done! ------------------------\n")