


from pathlib import Path
import utils_data
from step03_language import Language

def _tracks_to_atoms(track_file, lang, output_dir):
    # Implement the logic to convert tracks to segments
    print(f"Converting tracks to segments for {track_file}")
    track_data = utils_data.load_json(track_file)
    vid = track_data["vid"]

    output_file = Path(output_dir) / f"{vid}_atoms.json"
    if output_file.exists():
        print(f"Atoms file already exists: {output_file}")
        return output_file

    agent_tubes = track_data["data"]['agent_tubes']
    action_tubes = track_data["data"]['action_tubes']
    segments_by_ego_actions = track_data["data"]['av_action_tubes']
    frames = track_data["data"]['frames']
    atoms = []
    atoms.extend(lang.segs2atoms('av', segments_by_ego_actions))
    atoms.extend(lang.segs2atoms('agents', agent_tubes, frames))
    utils_data.save_json(atoms, output_file)
    
    return output_file

def _atoms_to_facts(atom_file, lang, output_dir):
    # Implement the logic to convert atoms to rules
    print(f"Converting atoms to rules for {atom_file}")
    atom_data = utils_data.load_json(atom_file)
    vid = Path(atom_file).stem.replace("_atoms", "")

    output_file = Path(output_dir) / f"{vid}_facts.json"
    if output_file.exists():
        print(f"Facts file already exists: {output_file}")
        return output_file

    facts = lang.atoms2facts(atom_data)
    utils_data.save_json(facts, output_file)
    return output_file
def _facts_to_rules(fact_file, lang, output_dir):
    # Implement the logic to convert facts to rules
    print(f"Converting facts to rules for {fact_file}")
    fact_data = utils_data.load_json(fact_file)
    vid = Path(fact_file).stem.replace("_facts", "")

    output_file = Path(output_dir) / f"{vid}_rules.json"
    if output_file.exists():
        print(f"Rules file already exists: {output_file}")
        return output_file

    rules = lang.facts2rules(fact_data)
    utils_data.save_json(rules, output_file)
    return output_file


def main(input_data):
    print("\n------- Step 03 -------\n")
    output_dir = input_data["output_dir"]
    step02_output_dir = input_data["step02_output_dir"]
    step01_output_dir = input_data["step01_output_dir"]
    track_dir = step01_output_dir/ 'gt'
    # get the all the track files
    track_files = list(track_dir.glob("*_gt.json"))

    language_model = Language() 
    for track_file in track_files:
        print(f"Processing track file: {track_file}")
        atom_files = _tracks_to_atoms(track_file, language_model, output_dir)
        fact_files = _atoms_to_facts(atom_files, language_model, output_dir)
        rule_files = _facts_to_rules(fact_files, language_model, output_dir)


    print("\n--------- Step 03 Done ---------------\n")