

import os
import sys
import json
import time
from collections import Counter
import numpy as np
from pathlib import Path
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.PermutationBased.BOSS import boss
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.utils.GraphUtils import GraphUtils
from types import SimpleNamespace


from src.exp_roadpp import utils_data
from src.exp_roadpp import config, utils_pipe
from src.exp_roadpp.step03_pruner import Pruner


def train_video_ids(clauses):
    """Videos the clauses were mined on; keeps val/test out of the graphs."""
    vids = set()
    for entry in clauses.values():
        vids.update(entry["support_coverage"])
    return sorted(vids)


def load_clauses(clauses_file, prune_args=None):
    """{clause_id: entry}, optionally coarse-pruned exactly like step 03."""
    clauses = json.loads(Path(clauses_file).read_text())
    if prune_args is not None:
        clauses = Pruner(prune_args).coarse_prune(clauses)
    return clauses


def init_agent_counts(labels):
    """{(pred, agent_class, id): 0} for every agent action and location in label.json.

    Keys match step04_playground.atom_key, so they line up with the clause atoms.
    """
    agent_counts = {}
    for agent_class in range(len(labels["all_agent_labels"])):
        for action_id in range(len(labels["all_action_labels"])):
            agent_counts[("action", agent_class, action_id)] = 0
        for location_id in range(len(labels["all_loc_labels"])):
            agent_counts[("location", agent_class, location_id)] = 0
    return agent_counts


def init_av_counts(labels):
    """{("action", "av", av_action_id): 0} for every AV action in label.json."""
    return {("action", "av", i): 0 for i in range(len(labels["all_av_action_labels"]))}


def count_atom(agent_counts, av_counts, atom):
    """Increase the count of the atom's ungrounded type in the matching dict."""
    if "action_id" in atom:
        key = (atom["pred"], atom["agent_class"], atom["action_id"])
    else:
        key = (atom["pred"], atom["agent_class"], atom["location_name"])
    counts = av_counts if atom["agent_class"] == "av" else agent_counts
    if key not in counts:
        raise KeyError(f"atom type {key} is not in the label vocabulary")
    counts[key] += 1

def remove_consecutive_duplicates(full_dataset, row_video, row_slice):
    # remove consecutive rows within a video if they are duplicates
    remove_indices = []
    last_kept = 0
    for i in range(1, len(full_dataset)):
        if (row_video[i] == row_video[i - 1] and
            row_slice[i, 0] == row_slice[i - 1, 1] and
            np.array_equal(full_dataset[i], full_dataset[i - 1])):
            remove_indices.append(i)
            row_slice[last_kept, 1] = row_slice[i, 1]
        else:
            last_kept = i

    full_dataset = np.delete(full_dataset, remove_indices, axis=0)
    row_video = np.delete(row_video, remove_indices, axis=0)
    row_slice = np.delete(row_slice, remove_indices, axis=0)

    return full_dataset, row_video, row_slice

def to_tabular_dataset(atoms, labels, prune_args=None):
    """Convert atoms to a tabular dataset for causal discovery."""
    separator = {}
    agent_counts = init_agent_counts(labels)
    av_counts = init_av_counts(labels)
    table_column_nr = len(agent_counts) + 1  # av is only one var, not one hot
    columns = list(agent_counts.keys()) + [("action", "av", "action_id")]
    all_data = []
    row_video = []
    row_slice = []
    for id, info in atoms.items():
        all_seps = []
        for entry in info:
            count_atom(agent_counts, av_counts, entry)
            all_seps.append(int(entry["start_frame"]))
            all_seps.append(int(entry["end_frame"]) + 1)
        all_seps = sorted(set(all_seps))
        separator[id] = all_seps

        current_data = np.zeros((len(all_seps) - 1, table_column_nr), dtype=np.int8)
        current_data[:, -1] = -1
        discard_rows = []
        for i in range(len(all_seps) - 1):
            start = all_seps[i]
            end = all_seps[i + 1]
            for entry in info:
                if int(entry["start_frame"]) <= start and int(entry["end_frame"]) >= end - 1:
                    if entry["agent_class"] == "av":
                        if current_data[i, -1] != -1:
                            discard_rows.append(i)
                        current_data[i, -1] = entry["action_id"]
                    else:
                        key = (entry["pred"], entry["agent_class"], entry.get("action_id", entry.get("location_name")))
                        col_idx = list(agent_counts.keys()).index(key)
                        current_data[i, col_idx] = 1

        # Remove rows that have been marked for discarding
        current_data = np.delete(current_data, discard_rows, axis=0)
        discard = set(discard_rows)
        kept = [i for i in range(len(all_seps) - 1) if i not in discard]
        row_video.extend([id] * len(kept))
        row_slice.extend([(all_seps[i], all_seps[i + 1]) for i in kept])

        all_data.append(current_data)

    full_dataset = np.concatenate(all_data, axis=0)

    row_video = np.array(row_video)
    row_slice = np.array(row_slice)

    full_dataset, row_video, row_slice = remove_consecutive_duplicates(full_dataset, row_video, row_slice)

    # remove all columns that have less than min_support_features occurrences
    min_support_features = 1
    if prune_args is not None:
        min_support_features = prune_args.get("min_support_features", 1)
    if min_support_features >= 1:
        col_sums = np.sum(full_dataset[:, :-1], axis=0)
        cols_to_keep = col_sums >= min_support_features
        cols_to_keep = np.append(cols_to_keep, True)  # keep the last column (av action)
        columns = [col for col, keep in zip(columns, cols_to_keep) if keep]
        full_dataset = full_dataset[:, cols_to_keep]             

    full_dataset, row_video, row_slice = remove_consecutive_duplicates(full_dataset, row_video, row_slice)

    # merge identical columns
    groups = {}
    for j in range(full_dataset.shape[1] - 1):
        groups.setdefault(full_dataset[:, j].tobytes(), []).append(j)
    keep = sorted(g[0] for g in groups.values()) + [full_dataset.shape[1] - 1]
    aliases = {columns[g[0]]: [columns[j] for j in g[1:]] for g in groups.values() if len(g) > 1}
    columns = [columns[j] for j in keep]
    full_dataset = full_dataset[:, keep]

    # remap AV action ids to contiguous codes 0..k-1; av_values[code] is the original id
    av_values, av_codes = np.unique(full_dataset[:, -1], return_inverse=True)
    full_dataset[:, -1] = av_codes
    
    return full_dataset, row_video, row_slice, columns, aliases, av_values, separator


def node_names_from_columns(columns, labels):
    """Readable, unique node names: action(Car, Stop), location(Car, Jun), and AV for the AV column."""
    names = []
    for pred, agent_class, value_id in columns:
        if agent_class == "av":
            names.append("AV")
            continue
        values = labels["all_action_labels"] if pred == "action" else labels["all_loc_labels"]
        names.append(f"{pred}({labels['all_agent_labels'][agent_class]}, {values[value_id]})")
    return names


def apply_causal_discovery(full_dataset, node_names, args):
    """Run the causal discovery method chosen on the command line; returns a causal-learn GeneralGraph.

    pc / fci are constraint-based and take a CI test (fisherz, chisq, gsq) as args.cd_stat,
    ges / boss / grasp are score-based and take a score (bic, bdeu).
    args.cd_depth caps the conditioning-set size for pc / fci; None means unlimited.
    """
    method, stat = args.cd_method, args.cd_stat
    tests = {"fisherz", "chisq", "gsq"}
    scores = {"bic": "local_score_BIC_from_cov", "bdeu": "local_score_BDeu"}
    if method in ("pc", "fci") and stat not in tests:
        raise ValueError(f"{method} needs a CI test {sorted(tests)}, got {stat!r}")
    if method in ("ges", "boss", "grasp") and stat not in scores:
        raise ValueError(f"{method} needs a score {sorted(scores)}, got {stat!r}")

    X = full_dataset

    if method == "pc":
        return pc(X, args.cd_alpha, stat, show_progress=False, node_names=node_names,
                  max_k=args.cd_depth).G
    if method == "fci":
        graph, _ = fci(X, stat, args.cd_alpha, depth=-1 if args.cd_depth is None else args.cd_depth,
                       show_progress=False, node_names=node_names)
        return graph
    if method == "ges":
        return ges(X, score_func=scores[stat], node_names=node_names)["G"]
    if method == "boss":
        return boss(X, score_func=scores[stat], verbose=False, node_names=node_names)
    if method == "grasp":
        return grasp(X, score_func=scores[stat], verbose=False, node_names=node_names)
    raise ValueError(f"unknown causal discovery method {method!r}")


def save_graph(graph, columns, node_names, aliases, av_values, n_rows, args, runtime_s, out_dir):
    """Save a discovered graph as <tag>.npz (endpoint matrix) and <tag>.json (nodes, edges, settings).

    In the matrix, graph[i, j] is the endpoint mark at node i on the edge between i and j:
    -1 tail, 1 arrow, 2 circle, 0 no edge. So i --> j is graph[i, j] = -1, graph[j, i] = 1.
    Node i is columns[i] / node_names[i]. Returns the common path prefix of both files.
    """
    tag = f"{args.cd_method}-{args.cd_stat}"
    if args.cd_method in ("pc", "fci"):
        tag += f"-a{args.cd_alpha}" + ("" if args.cd_depth is None else f"-d{args.cd_depth}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / tag  # no with_suffix(): the alpha in the tag contains a dot

    np.savez_compressed(f"{prefix}.npz", graph=graph.graph.astype(np.int8))
    meta = {
        "method": args.cd_method,
        "stat": args.cd_stat,
        "alpha": args.cd_alpha,
        "depth": args.cd_depth,
        "exp": args.exp,
        "min_support_features": args.min_support_features,
        "n_rows": n_rows,
        "runtime_s": round(runtime_s, 1),
        "columns": [list(col) for col in columns],
        "node_names": node_names,
        "aliases": [[list(kept), [list(a) for a in rest]] for kept, rest in aliases.items()],
        "av_values": [int(v) for v in av_values],
        "edges": [str(edge) for edge in graph.get_graph_edges()],
    }
    Path(f"{prefix}.json").write_text(json.dumps(meta, indent=2))
    return prefix


def save_graph_pdf(graph, prefix):
    """Render the graph to <prefix>.pdf and AV with its direct neighbours to <prefix>_av.pdf.

    The full graph uses sfdp (force-directed; dot's layered layout is slow and unreadable at this
    density), the AV neighbourhood uses dot and includes the edges among the neighbours.
    """
    colors = {"AV": "lightcoral"}
    full = GraphUtils.to_pydot(graph, colors=colors, title=Path(prefix).name)
    full.set("overlap", "false")  # spread nodes apart instead of stacking them
    full.write(f"{prefix}.pdf", prog="sfdp", format="pdf")

    av = graph.get_node("AV")
    keep = set(graph.get_adjacent_nodes(av)) | {av}
    # subgraph() labels the cut matrix with the nodes in the order given, so keep the graph's own order
    neighbourhood = graph.subgraph([node for node in graph.get_nodes() if node in keep])
    GraphUtils.to_pydot(neighbourhood, colors=colors, title=f"{Path(prefix).name} AV neighbourhood").write(
        f"{prefix}_av.pdf", prog="dot", format="pdf")


def clause_node_mapper(columns, aliases):
    """Function mapping a clause atom to its graph node index, or None if its column was dropped.

    An atom maps to its own column, to the kept column if it is an alias, and any AV atom maps
    to the single AV node.
    """
    node_index = {col: i for i, col in enumerate(columns)}
    for kept_key, alias_keys in aliases.items():
        for alias_key in alias_keys:
            node_index[alias_key] = node_index[kept_key]
    av_index = next(i for i, col in enumerate(columns) if col[1] == "av")

    def node_of(atom):
        if atom["agent_class"] == "av":
            return av_index
        return node_index.get((atom["pred"], atom["agent_class"], atom.get("action_id", atom.get("location_name"))))

    return node_of


def apply_graph_strategy(clauses, graph, node_of, args):
    """Filter the clauses with the causal graph, using the strategy chosen on the command line.

    Every strategy sees each clause atom as the graph node given by node_of (see
    clause_node_mapper); an atom whose column was dropped has no node.

    adjacent: keep a clause only if every body atom is adjacent to the head in the graph,
              whatever the edge type or direction. Clauses with an atom that has no node, or
              whose body and head share a node (two AV atoms, an atom and its alias), cannot
              have such an edge and are removed as well.

    Returns the kept clauses, {clause_id: entry}, in the same format as the input, and a
    Counter of the reasons the other clauses were removed.
    """
    strategy = args.graph_strategy
    kept = {}
    removed = Counter()
    if strategy == "adjacent":
        for clause_id, entry in clauses.items():
            head = node_of(entry["clause"]["head"])
            bodies = [node_of(atom) for atom in entry["clause"]["body"]]
            if head is None or None in bodies:
                removed["atom not in graph"] += 1
            elif head in bodies:
                removed["body and head share a node"] += 1
            elif not all(graph.graph[body, head] != 0 for body in bodies):
                removed["no edge"] += 1
            else:
                kept[clause_id] = entry
    else:
        raise ValueError(f"unknown graph strategy {strategy!r}")

    print(f"graph strategy {strategy}: kept {len(kept)} of {len(clauses)} clauses, removed "
          + ", ".join(f"{n} ({reason})" for reason, n in removed.most_common()))
    return kept, removed


def clause_graph_relations(clauses, graph, node_of):
    """{clause_id: (relation, edge)} between each clause's first body atom and its head.

    relation uses step04_eval's vocabulary: b->h, h->b, bidirected, undirected (any other endpoint
    pair, e.g. FCI's o-> and o-o, as in the playground), none (no edge), and untested (an atom has
    no node, or body and head share a node). edge is the edge drawn from body to head, e.g. "o->",
    or None without an edge.
    """
    body_mark = {-1: "-", 1: "<", 2: "o"}
    head_mark = {-1: "-", 1: ">", 2: "o"}
    relations = {}
    for clause_id, entry in clauses.items():
        body, head = node_of(entry["clause"]["body"][0]), node_of(entry["clause"]["head"])
        if body is None or head is None or body == head:
            relations[clause_id] = ("untested", None)
            continue
        at_body, at_head = int(graph.graph[body, head]), int(graph.graph[head, body])
        if at_body == 0 and at_head == 0:
            relations[clause_id] = ("none", None)
            continue
        if (at_body, at_head) == (-1, 1):
            relation = "b->h"
        elif (at_body, at_head) == (1, -1):
            relation = "h->b"
        elif (at_body, at_head) == (1, 1):
            relation = "bidirected"
        else:
            relation = "undirected"
        relations[clause_id] = (relation, body_mark[at_body] + "-" + head_mark[at_head])
    return relations


def save_clause_results(kept, removed, clauses, relations, source, args, out_dir):
    """Write the filtered clauses in step04_eval's format, plus relations and a summary.

    out_dir/clauses_<source>_<strategy>.json  kept clauses, {cid: {**entry, "causal": {...}}};
                                              removed clauses are left out, as the eval expects
    out_dir/relations_<source>.json           {cid: relation} for every input clause
    out_dir/clause_relations.json             {cid: {source: relation}}, merged from all
                                              relations_*.json, for step04_eval --by-relation
    out_dir/summary_<source>_<strategy>.json  counts, removal reasons and run settings

    Evaluate every source and strategy in out_dir with step04_eval --playground-dir out_dir.
    Returns the path of the clause file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strategy = args.graph_strategy

    kept_entries = {}
    for clause_id, entry in kept.items():
        relation, edge = relations[clause_id]
        kept_entries[clause_id] = {**entry, "causal": {"source": source, "strategy": strategy, "decision": "keep",
                                                       "relation": relation, "edge": edge}}
    clause_file = out_dir / f"clauses_{source}_{strategy}.json"
    clause_file.write_text(json.dumps(kept_entries))

    relations_only = {clause_id: relation for clause_id, (relation, _) in relations.items()}
    (out_dir / f"relations_{source}.json").write_text(json.dumps(relations_only))
    merged = {}
    for relations_file in sorted(out_dir.glob("relations_*.json")):
        merged_source = relations_file.stem[len("relations_"):]
        for clause_id, relation in json.loads(relations_file.read_text()).items():
            merged.setdefault(clause_id, {})[merged_source] = relation
    (out_dir / "clause_relations.json").write_text(json.dumps(merged))

    settings = ("exp", "cd_method", "cd_stat", "cd_alpha", "cd_depth", "min_support_features",
                "prune_frequency_threshold", "prune_support_coverage_ratio_threshold", "graph_strategy")
    summary = {
        "source": source,
        "strategy": strategy,
        "n_input_clauses": len(clauses),
        "n_kept": len(kept),
        "n_removed": len(clauses) - len(kept),
        "removed_by_reason": dict(removed),
        "kept_by_relation": dict(Counter(relations[clause_id][0] for clause_id in kept)),
        "all_by_relation": dict(Counter(relation for relation, _ in relations.values())),
        "n_av_head_kept": sum(entry["clause"]["head"]["agent_class"] == "av" for entry in kept.values()),
        "settings": {key: getattr(args, key) for key in settings},
    }
    (out_dir / f"summary_{source}_{strategy}.json").write_text(json.dumps(summary, indent=2))
    return clause_file


def main(input_data=None):
    input_data, args = args_and_init(input_data)

    output_dir = input_data["output_dir"]
    step03_output_dir = Path(input_data["step03_output_dir"])
    device = input_data["device"]

    # Implement the causal reasoning logic here
    # For now, just print the input data
    print(f"Running step 04 with output_dir: {output_dir}, step03_output_dir: {step03_output_dir}, device: {device}")

    atoms_file = Path(step03_output_dir) / "atoms" / "all_atoms.json"
    clauses_file = Path(step03_output_dir) / "clauses" / "all_init_clauses_train.json"
    for f in (atoms_file, clauses_file):
        if not f.exists():
            raise FileNotFoundError(f"{f} is missing; run step 03 for this experiment first")

    atoms = utils_data.load_json(atoms_file)
    clauses = load_clauses(clauses_file, prune_args={
        "frequency_threshold": args.prune_frequency_threshold,
        "support_coverage_ratio_threshold": args.prune_support_coverage_ratio_threshold
    })
    print(f"{len(atoms)} videos of atoms, {len(clauses)} pruned clauses")

    # all clauses are based on the train set, only use train set atoms here as well
    train_ids = train_video_ids(clauses)
    atoms_by_video = {v: atoms[v] for v in train_ids if v in atoms}
    with open(args.dataset_path/"gt"/"label.json", 'r') as f:
        dataset_labels = json.load(f)
    full_dataset, row_video, row_slice, columns, aliases, av_values, separator = to_tabular_dataset(atoms_by_video, dataset_labels, prune_args={
        "min_support_features": args.min_support_features
    })

    node_names = node_names_from_columns(columns, dataset_labels)
    start = time.time()
    graph = apply_causal_discovery(full_dataset, node_names, args)
    runtime_s = time.time() - start
    prefix = save_graph(graph, columns, node_names, aliases, av_values, len(full_dataset), args,
                        runtime_s, Path(output_dir) / "graphs")
    save_graph_pdf(graph, prefix)
    print(f"{args.cd_method}/{args.cd_stat}: {len(graph.get_graph_edges())} edges in {runtime_s:.0f}s, "
          f"saved to {prefix}.npz / .json / .pdf / _av.pdf")

    node_of = clause_node_mapper(columns, aliases)
    kept_clauses, removed = apply_graph_strategy(clauses, graph, node_of, args)
    relations = clause_graph_relations(clauses, graph, node_of)
    clause_file = save_clause_results(kept_clauses, removed, clauses, relations, prefix.name, args,
                                      Path(output_dir) / "causal")
    print(f"kept clauses saved to {clause_file}")

    return {"graph": prefix, "clauses": clause_file}

def args_and_init(input_data):
    parser = utils_pipe.get_parser()
    parser.add_argument("--prune_frequency_threshold", type=int, default=10, help="Minimum frequency for a clause to be kept")
    parser.add_argument("--prune_support_coverage_ratio_threshold", type=float, default=0.5, help="Minimum support/coverage ratio for a clause to be kept")
    # --min_support_features, --cd_method, --cd_stat, --cd_alpha, --cd_depth, --graph_strategy
    # are now defined on the shared parser (utils_pipe.get_parser) since the causal-discovery
    # logic they configure also runs as a substep of step03.
    args = utils_pipe.parse_args(parser)

    if input_data is not None:
        # called from main.py: the pipeline already built our input, but this fresh args still
        # needs the machine-specific paths (dataset_path for label.json) that step_0_setup sets
        args = config.step_0_setup(args)
        return input_data, args

    # Standalone entry point for debugging step 04 without running steps 01-03.
    if "--exp" not in sys.argv:
        args.exp = os.environ.get("CAUVID_EXP", "medium")
        # parse_args loaded the config for the default --exp; reload it for the one we switched to
        for key, value in utils_data.load_json(config.exp_config_path / f"{args.exp}.json").items():
            setattr(args, key, value)
    args = config.step_0_setup(args)
    step04_input = config.get_step_04_input(args)

    step03_dir_override = os.environ.get("CAUVID_STEP03_DIR")
    if step03_dir_override:
        step04_input["step03_output_dir"] = Path(step03_dir_override)

    return step04_input, args

if __name__ == "__main__":
    main()

# Notes:
# - Prune args might differ from the previous steps
# - AV is constant within every video, this might cause problem down the line


def _causal_filter_clauses(pruned_clauses, atoms_train, dataset_labels, cfg):
    """Former step 04, now a substep of step 03: discover a causal graph over the
    atom vocabulary from the train-set atoms, then keep only the coarse-pruned
    clauses whose body/head atoms are adjacent in that graph (see
    `step04.apply_graph_strategy`). `pruned_clauses` is the {clause_id: entry}
    dict straight out of `Pruner.coarse_prune` (i.e. before ranking by support
    coverage), since that's the {id: {"clause": {"head", "body"}, ...}} format
    `step04`'s graph-filtering functions expect.
    """
    args = SimpleNamespace(
        cd_method=cfg["cd_method"], cd_stat=cfg["cd_stat"], cd_alpha=cfg["cd_alpha"],
        cd_depth=cfg["cd_depth"], graph_strategy=cfg["graph_strategy"],
        min_support_features=cfg["min_support_features"], exp=cfg["exp"],
    )

    full_dataset, row_video, row_slice, columns, aliases, av_values, separator = to_tabular_dataset(
        atoms_train, dataset_labels, prune_args={"min_support_features": args.min_support_features}
    )
    node_names = node_names_from_columns(columns, dataset_labels)
    start = time.time()
    graph = apply_causal_discovery(full_dataset, node_names, args)
    runtime_s = time.time() - start
    prefix = save_graph(
        graph, columns, node_names, aliases, av_values, len(full_dataset), args,
        runtime_s, cfg["output_dir"] / "graphs",
    )
    save_graph_pdf(graph, prefix)
    print(f"[Causal filter] {args.cd_method}/{args.cd_stat}: {len(graph.get_graph_edges())} edges "
          f"in {runtime_s:.0f}s, saved to {prefix}.npz / .json / .pdf / _av.pdf")

    node_of = clause_node_mapper(columns, aliases)
    kept_clauses, removed = apply_graph_strategy(pruned_clauses, graph, node_of, args)
    relations = clause_graph_relations(pruned_clauses, graph, node_of)
    save_clause_results(
        kept_clauses, removed, pruned_clauses, relations, prefix.name, args, cfg["output_dir"] / "causal"
    )
    return kept_clauses


def main_v2(input_data):
    cfg = {
        "cd_method": input_data["cd_method"],
        "cd_stat": input_data["cd_stat"],
        "cd_alpha": input_data["cd_alpha"],
        "cd_depth": input_data["cd_depth"],
        "graph_strategy": input_data["graph_strategy"],
        "min_support_features": input_data["min_support_features"],
        "exp": input_data["exp"],
        "output_dir": input_data["output_dir"],
    }
    causal_clause_file = input_data["output_dir"] / "causal_clauses.json"
    if causal_clause_file.exists():
        return utils_data.load_json(causal_clause_file)
    anchor_clauses = utils_data.load_json(input_data["anchor_clause_file"])
    atoms = utils_data.load_json(input_data["train_atom_file"])
    dataset_labels = input_data["dataset_labels"]
    causal_clauses = _causal_filter_clauses(anchor_clauses, atoms, dataset_labels, cfg)
    utils_data.save_json(causal_clauses, causal_clause_file)
    return causal_clauses 