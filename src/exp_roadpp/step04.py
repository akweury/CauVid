

from pathlib import Path
from src.exp_roadpp import utils_data



def load_fact_file(fact_file):
    import json
    with open(fact_file, "r") as f:
        data = json.load(f)
    return data


def main(input_data):
    output_dir = input_data["output_dir"]
    step03_output_dir = input_data["step03_output_dir"]
    device = input_data["device"]

    # Implement the causal reasoning logic here
    # For now, just print the input data
    print(f"Running step 04 with output_dir: {output_dir}, step03_output_dir: {step03_output_dir}, device: {device}")

    fact_files = list(Path(step03_output_dir).glob("*_facts.json"))
    for fact_file in fact_files:
        print(f"Processing fact file: {fact_file}")
        fact_data = load_fact_file(fact_file)
        # Here you can implement the causal reasoning logic using the loaded fact_data
        # For now, just print the fact data
        print(f"Fact data: {fact_data}")
        
    return
