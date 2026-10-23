from pathlib import Path
import torch
from tqdm import tqdm

from src.exp_roadpp import utils_data



def build_atom_vocab(train_unique_atoms, eval_unique_atoms):
    """Map train atom ids (used by the clauses) to eval atom ids.

    Both splits number their atoms independently (see `_compress_atom_file`),
    so ids are matched through the atom signatures. Train atoms that never
    occur in the eval split get new ids after the eval ones.
    Returns (train_to_eval, num_atoms).
    """
    vocab = {tuple(atom): i for i, atom in enumerate(eval_unique_atoms)}
    train_to_eval = [vocab.setdefault(tuple(atom), len(vocab)) for atom in train_unique_atoms]
    num_unseen = len(vocab) - len(eval_unique_atoms)
    print(f"[Atom Vocab] train={len(train_unique_atoms)}, eval={len(eval_unique_atoms)}, "
          f"train atoms unseen in eval={num_unseen}")
    return train_to_eval, len(vocab)


def build_clause_tensors(clauses, train_to_eval, num_atoms, device):
    """Encode the extended clauses as masks over eval atom ids.

    A clause's first body is its anchor (atom_index, "oid", body_slot); the
    other bodies are (atom_index, "oid", object_slot) as in step05's
    `build_features`: slot 0 is the head object, slot body_slot the body
    object and slot body_slot + 1 some other object in frame t.
      body[c]:     atoms the body object must hold at t (anchor included);
      head_obj[c]: atoms the head object must hold at t (body_slot != 0);
      other[c]:    atoms some third object must hold at t;
      head[c]:     the atom predicted for the head object at t+1;
      is_self[c]:  body_slot == 0, i.e. body and head object are the same.
    `body`/`head_obj` keep only the columns `object_atoms`, `other` only the
    columns `other_atoms`, so per-frame matmuls skip unused atoms.
    """
    n = len(clauses)
    body = torch.zeros((n, num_atoms), dtype=torch.bool)
    head_obj = torch.zeros((n, num_atoms), dtype=torch.bool)
    other = torch.zeros((n, num_atoms), dtype=torch.bool)
    head = torch.zeros((n, num_atoms), dtype=torch.bool)
    is_self = torch.zeros(n, dtype=torch.bool)
    num_legacy = 0
    for i, clause in enumerate(clauses):
        anchor_index, _tag, body_slot = clause["bodies"][0]
        is_self[i] = body_slot == 0
        body[i, train_to_eval[anchor_index]] = True
        for atom_index, _tag, slot in clause["bodies"][1:]:
            atom = train_to_eval[atom_index]
            if slot == body_slot:
                body[i, atom] = True
            elif slot == 0:
                head_obj[i, atom] = True
            else:
                # Older step05 runs gave each other entity its own slot
                # (body_slot + 1, + 2, ...); evaluate them as the shared
                # "some other object" slot of the current step05.
                other[i, atom] = True
        if any(slot > body_slot + 1 for _a, _t, slot in clause["bodies"][1:]):
            num_legacy += 1
        head[i, train_to_eval[clause["head"][0]]] = True
    if num_legacy:
        print(f"[Clauses] WARNING: {num_legacy}/{n} clauses use per-entity slots from an older "
              f"step05 run; their extra slots are treated as 'some other object'. "
              f"Regenerate step05 outputs for exact semantics.")

    object_atoms = torch.nonzero((body | head_obj).any(dim=0), as_tuple=True)[0]
    other_atoms = torch.nonzero(other.any(dim=0), as_tuple=True)[0]
    print(f"[Clauses] {n} clauses, {len(object_atoms)} object atoms, "
          f"{len(other_atoms)} other-object atoms, {int(is_self.sum())} self clauses")
    return {
        "body": body[:, object_atoms].float().to(device),
        "head_obj": head_obj[:, object_atoms].float().to(device),
        "other": other[:, other_atoms].float().to(device),
        "head": head.float().to(device),
        "is_self": is_self.to(device),
        "object_atoms": object_atoms.to(device),
        "other_atoms": other_atoms.to(device),
    }


def frame_to_matrix(frame_atoms, entities, num_atoms, device):
    """(E, A) bool matrix: entity e holds atom a in this frame.

    Compressed per-frame atoms are (atom_id, "tube_uid", tube_uid); the AV
    has no tube_uid (same entity key as step05's `_atom_entity_key`).
    """
    matrix = torch.zeros((len(entities), num_atoms), dtype=torch.bool)
    for atom_id, _tag, tube_uid in frame_atoms:
        matrix[entities[tube_uid if tube_uid is not None else "av"], atom_id] = True
    return matrix.to(device)


def predict_next_frame(current, present_next, ct):
    """Atoms the clauses predict for each entity at t+1.

    current: (E, A) bool atoms at t; present_next: (E,) entity has atoms at
    t+1 (the head object must exist there, as in step05). A clause fires for
    a (body entity b, head entity h) pair if b holds `body`, h holds
    `head_obj`, b == h for self clauses (b != h otherwise) and every `other`
    atom is held by some entity other than b and h.
    """
    num_entities = current.shape[0]
    missing = (~current[:, ct["object_atoms"]]).float()
    body_ok = (missing @ ct["body"].T) == 0          # (E, C)
    head_ok = (missing @ ct["head_obj"].T) == 0      # (E, C)

    same = torch.eye(num_entities, dtype=torch.bool, device=current.device)
    held = current[:, ct["other_atoms"]].int()        # (E, A_o)
    counts = held.sum(dim=0)
    # others[b, h, a]: number of entities other than b and h holding atom a.
    others = counts - held[:, None, :] - held[None, :, :] * (~same)[:, :, None].int()
    violated = ((others <= 0).float().reshape(num_entities * num_entities, -1)
                @ ct["other"].T).reshape(num_entities, num_entities, -1) > 0

    pair_ok = torch.where(ct["is_self"], same[:, :, None], ~same[:, :, None])
    fired = (body_ok[:, None, :] & head_ok[None, :, :] & present_next[None, :, None]
             & pair_ok & ~violated).any(dim=0)        # (E_head, C)
    return (fired.float() @ ct["head"]) > 0


def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "iou": iou, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def clauses_evaluation(clauses, train_unique_atoms, atoms, cfg):
    """Micro-averaged next-frame atom prediction on the eval frame pairs.

    For every (t, t+1) pair with atoms in both frames (same pairs step05
    trains on), the predicted (entity, atom) set at t+1 is compared with the
    ground truth atoms of the entities present at t+1.
    """
    summary_file = cfg["output_dir"] / "extended_clauses_evaluation.json"
    if summary_file.exists():
        return utils_data.load_json(summary_file)

    device = cfg["device"]
    train_to_eval, num_atoms = build_atom_vocab(train_unique_atoms, atoms["unique_atoms"])
    ct = build_clause_tensors(clauses, train_to_eval, num_atoms, device)

    videos = {}
    total = [0, 0, 0]
    for vid, video_atoms in tqdm(atoms.items(), desc="Evaluating videos"):
        if vid == "unique_atoms":
            continue
        tp = fp = fn = 0
        num_pairs = 0
        for current_frame, next_frame in video_atoms["per_frame_atoms"]:
            if len(current_frame) == 0 or len(next_frame) == 0:
                continue
            entities = {}
            for _atom_id, _tag, tube_uid in current_frame + next_frame:
                entities.setdefault(tube_uid if tube_uid is not None else "av", len(entities))
            current = frame_to_matrix(current_frame, entities, num_atoms, device)
            gt = frame_to_matrix(next_frame, entities, num_atoms, device)
            pred = predict_next_frame(current, gt.any(dim=1), ct)

            tp += int((pred & gt).sum())
            fp += int((pred & ~gt).sum())
            fn += int((~pred & gt).sum())
            num_pairs += 1

        videos[vid] = {**compute_metrics(tp, fp, fn), "frame_pairs": num_pairs}
        total = [total[0] + tp, total[1] + fp, total[2] + fn]
        m = videos[vid]
        print(f"Evaluation for video {vid}: recall={m['recall']:.4f}, precision={m['precision']:.4f}, "
              f"iou={m['iou']:.4f}, f1={m['f1']:.4f}")

    summary = {
        "num_clauses": len(clauses),
        "overall": compute_metrics(*total),
        "videos": videos,
    }
    utils_data.save_json(summary, summary_file)
    return summary



def main(input_data):
    print("\n--------- Step 06 Evaluation ----------------------\n")
    cfg = {
        "output_dir": Path(input_data["output_dir"]),
        "device": input_data.get("device", "cpu")
    }
    atoms = utils_data.load_json(input_data["val_atoms_file"])
    extended_clauses = utils_data.load_json(input_data["extended_clauses_file"])
    # Clause atom ids index the train split's unique atoms (saved by step03).
    train_unique_atoms = utils_data.load_json(input_data["anchor_clause_file"])["unique_atoms"]
    print(f"Extended Clauses Num: {len(extended_clauses)}")

    summary = clauses_evaluation(extended_clauses, train_unique_atoms, atoms, cfg)
    m = summary["overall"]
    print(f"[Overall] recall={m['recall']:.4f}, precision={m['precision']:.4f}, "
          f"iou={m['iou']:.4f}, f1={m['f1']:.4f}")

    print("\n--------- Step 06 Done ---------------\n")
    return summary
