import os 
from pathlib import Path


from src.exp_roadpp import utils_data
from src.exp_roadpp.step_dataset_language import Language
from src.exp_roadpp.baselines import transformer_baseline, ilp_baseline, gnn_baseline, lstm_baseline
from src.exp_roadpp.step03_visual import visualize_baseline_results

def baselines(input_data):
    print("\n--------- Step 03 Baselines ----------------------\n")
    if input_data["skip_baselines_03"] == 'True':
        return
    
    output_dir = input_data["output_dir"]
    test_output_dir = input_data["test_output_dir"]
    track_dir = input_data["dataset_path"] / "gt"
    dataset_path = input_data["dataset_path"]
    dataset_labels = input_data["dataset_labels"]
    language_model = Language(input_data["device"])
    device = input_data["device"]
    all_track_files =[os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.endswith("_gt.json")]
    if input_data['data_num'] != 'full':
        all_track_files = all_track_files[:int(input_data['data_num'])]
    train_indices, val_indices, test_indices = _split_example_indices(len(all_track_files))
    
    # train data
    train_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in train_indices]
    _tracks_to_atoms(track_dir, train_ids, language_model, output_dir)
    train_facts = _atoms_to_init_clauses(language_model, output_dir, 'train')
    val_facts = _atoms_to_init_clauses(language_model, output_dir, 'val')
    test_facts = _atoms_to_init_clauses(language_model, output_dir, 'test')
    # all_rules, all_rule_supports, all_head_supports = _facts_to_rules(train_facts, train_ids, language_model, output_dir)
    # val data
    val_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in val_indices]
    # test data
    test_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in test_indices]
    _tracks_to_atoms(track_dir, test_ids, language_model, output_dir)
    

    # train and test baselines
    result_summary = {}
    all_facts = {"train": train_facts, "val": val_facts, "test": test_facts}
    ilp_baseline.run(all_facts, dataset_labels, output_dir, result_summary, device)
    gnn_baseline.run(all_facts, dataset_labels, output_dir, result_summary, device)
    transformer_baseline.run(all_facts, dataset_labels, output_dir, result_summary, device)
    lstm_baseline.run(all_facts, dataset_labels, output_dir, result_summary, device)
    utils_data.save_json(result_summary, test_output_dir / "baselines_summary.json")
    visualize_baseline_results(result_summary, test_output_dir)
    
    print("\n--------- Step 03 Baseline Done! ---------------\n")



def main(input_data):
    baselines(input_data)
