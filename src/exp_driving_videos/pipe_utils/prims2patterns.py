import os
from collections import Counter
from itertools import combinations

from blinker import signal
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm

import config 
from src.exp_driving_videos.pipe_utils import exp_driving_utils as utils
from src.exp_driving_videos.pipe_utils.percept2matrix import percept2matrix, percept2ego_speed
from src.exp_driving_videos.pipe_utils.matrix2signal import matrix2signal
from src.exp_driving_videos.pipe_utils.signal2segs import signal2segs

from src.exp_driving_videos.pipe_utils import exp_driving_utils as utils
from src.exp_driving_videos.pipe_utils.predicate_vocab import extract_predicates


######################## Helper functions for pattern detection ########################

def _extract_objs_per_segs(seg_prims, objects):
    """
    For each segment primitive, collect the objects that are visible within that segment's
    frame range and return per-segment summaries compatible with extract_predicates.

    Parameters
    ----------
    seg_prims : list of dict
        Each dict has at least "start" and "end" (integer frame indices).
    objects : np.ndarray, shape (num_frames, num_objs), dtype=object
        Frame-centric object primitives returned by matrix2signal.  Each cell is
        either None or a dict with keys including 'dx_obj' and 'dz_obj'.

    Returns
    -------
    seg_objs : list of list of dict
        Parallel to seg_prims.  Each inner list contains one dict per object that
        has at least one valid observation inside the segment.  Each dict has:
            "position"  : (mean_dx, mean_dz)   – mean relative position over the segment
            "dz_series" : np.ndarray            – time-ordered dz_obj values in the segment
    """
    num_objs = objects.shape[1]
    seg_objs = []

    for prim in seg_prims:
        start, end = prim["start"], prim["end"]
        # slice rows for this segment; end is exclusive like Python slices
        segment_slice = objects[start:end]  # shape: (duration, num_objs)

        objs_in_seg = []
        for obj_id in range(num_objs):
            obj_cells = segment_slice[:, obj_id]  # one cell per frame in the segment

            # collect valid cells together with their absolute frame index
            valid_cells_with_idx = [
                (start + frame_offset, cell)
                for frame_offset, cell in enumerate(obj_cells)
                if cell is not None
                and cell.get("dx_obj") is not None
                and cell.get("dz_obj") is not None
            ]

            if not valid_cells_with_idx:
                continue

            frame_indices, valid_cells = zip(*valid_cells_with_idx)

            dx_vals = np.array([c["dx_obj"] for c in valid_cells], dtype=float)
            dz_vals = np.array([c["dz_obj"] for c in valid_cells], dtype=float)

            # most frequent label across visible frames
            labels = [c["label"] for c in valid_cells if c.get("label") is not None]
            label = Counter(labels).most_common(1)[0][0] if labels else None

            # per-frame bounding boxes: absolute_frame_index -> bbox
            frame_bboxes = {
                fi: c["bbox"]
                for fi, c in zip(frame_indices, valid_cells)
                if c.get("bbox") is not None
            }

            objs_in_seg.append({
                "position":    (float(np.mean(dx_vals)), float(np.mean(dz_vals))),
                "dz_series":   dz_vals,
                "label":       label,
                "frame_bboxes": frame_bboxes,
            })

        seg_objs.append(objs_in_seg)

    return seg_objs


def _flatten_predicates(pred_dict):
    """
    Convert a nested predicate dict (output of extract_predicates) into a flat
    boolean/categorical record suitable for rule mining.

    Existential quantification over the object list: a flag is True when *any*
    object in the segment satisfies the condition.  For the nearest-front object
    specifically, relative-motion is also captured as its own key.

    Parameters
    ----------
    pred_dict : dict
        As returned by extract_predicates:
        {
            "ego_speed_change": str,
            "ego_turn": str,
            "objects": [ {"front": bool?, "close": bool?, "relative_motion": str}, ... ]
        }

    Returns
    -------
    record : dict
        Flat key-value pairs, all hashable.
    """
    objects = pred_dict.get("objects", [])

    # --- existential flags over all visible objects ---
    has_front_obj       = any(o.get("front") for o in objects)
    has_close_obj       = any(o.get("close") for o in objects)
    has_approaching_obj = any("approaching" in o.get("relative_motion", "") for o in objects)
    has_moving_away_obj = any("moving_away" in o.get("relative_motion", "") for o in objects)
    has_front_close_obj = any(o.get("front") and o.get("close") for o in objects)
    front_obj_approaching = any(
        o.get("front") and "approaching" in o.get("relative_motion", "") for o in objects
    )
    close_obj_approaching = any(
        o.get("close") and "approaching" in o.get("relative_motion", "") for o in objects
    )

    # --- nearest front object (smallest dz among front objects) ---
    # objects list doesn't carry raw dz here, so "nearest front" is approximated
    # as the first front+close object found, falling back to first front object.
    nearest_front_rel_motion = None
    front_objs = [o for o in objects if o.get("front")]
    close_front_objs = [o for o in front_objs if o.get("close")]
    nearest = (close_front_objs or front_objs)
    if nearest:
        nearest_front_rel_motion = nearest[0].get("relative_motion")

    return {
        # consequent candidates
        "ego_speed_change":       pred_dict.get("ego_speed_change"),
        "ego_turn":               pred_dict.get("ego_turn"),
        # antecedent candidates
        "has_front_obj":          has_front_obj,
        "has_close_obj":          has_close_obj,
        "has_approaching_obj":    has_approaching_obj,
        "has_moving_away_obj":    has_moving_away_obj,
        "has_front_close_obj":    has_front_close_obj,
        "front_obj_approaching":  front_obj_approaching,
        "close_obj_approaching":  close_obj_approaching,
        "nearest_front_rel_motion": nearest_front_rel_motion,  # categorical
    }


def _gen_atomic_rules(all_predicates, min_support=2, min_confidence=0.0):
    """
    Mine atomic (|P|=1) rules of the form  P → Q  from a flat predicate table.

    Parameters
    ----------
    all_predicates : list of dict
        Nested predicate dicts as returned by extract_predicates, one per segment
        across *all* videos.
    min_support : int
        Minimum number of segments where both P and Q hold.
    min_confidence : float
        Minimum confidence (support / total_P) to keep a rule.

    Returns
    -------
    rules : list of dict
        Each rule dict:
        {
            "antecedent":  (feature_name, feature_value),
            "consequent":  (feature_name, feature_value),
            "support":     int,
            "total_P":     int,
            "total_Q":     int,
            "n":           int,
            "confidence":  float,
            "lift":        float,
        }
        Sorted by lift descending.
    """
    # ── 1. flatten all predicate dicts into a list of flat records ────────────
    flat_records = [_flatten_predicates(p) for p in all_predicates]
    n = len(flat_records)
    if n == 0:
        return []

    # ── 2. define antecedent and consequent feature sets ──────────────────────
    antecedent_keys = [
        "has_front_obj",
        "has_close_obj",
        "has_approaching_obj",
        "has_moving_away_obj",
        "has_front_close_obj",
        "front_obj_approaching",
        "close_obj_approaching",
        "nearest_front_rel_motion",
    ]
    consequent_keys = [
        "ego_speed_change",
        "ego_turn",
    ]

    # collect all unique values for each key
    def unique_vals(key):
        return {r[key] for r in flat_records if r[key] is not None}

    # ── 3. count occurrences ──────────────────────────────────────────────────
    rules = []
    for q_key in consequent_keys:
        for q_val in unique_vals(q_key):
            total_Q = sum(1 for r in flat_records if r[q_key] == q_val)

            for p_key in antecedent_keys:
                if p_key == q_key:
                    continue
                for p_val in unique_vals(p_key):
                    total_P = sum(1 for r in flat_records if r[p_key] == p_val)
                    support  = sum(
                        1 for r in flat_records
                        if r[p_key] == p_val and r[q_key] == q_val
                    )

                    if support < min_support or total_P == 0:
                        continue

                    confidence = support / total_P
                    if confidence < min_confidence:
                        continue

                    lift = confidence / (total_Q / n) if total_Q > 0 else 0.0

                    rules.append({
                        "antecedent":  (p_key, p_val),
                        "consequent":  (q_key, q_val),
                        "support":     support,
                        "total_P":     total_P,
                        "total_Q":     total_Q,
                        "n":           n,
                        "confidence":  confidence,
                        "lift":        lift,
                    })

    rules.sort(key=lambda r: r["lift"], reverse=True)
    return rules


def _ext_atomic_rules(all_predicates, atomic_rules_filtered, min_support=5, min_confidence=0.5, max_size=3):
    """
    Extend filtered atomic rules to conjunctive antecedents  P1 ∧ P2 ∧ … → Q
    with |P| up to *max_size*, using Apriori-style pruning.

    Parameters
    ----------
    all_predicates : list of dict
        Nested predicate dicts as returned by extract_predicates (one per segment).
    atomic_rules_filtered : list of dict
        Filtered atomic rules from _gen_atomic_rules.  The antecedent (key, val)
        pairs found here seed the candidate pool to keep the search space tractable.
    min_support : int
        Minimum number of segments where P ∧ Q both hold.
    min_confidence : float
        Minimum confidence (support / total_P) to keep a rule.
    max_size : int
        Maximum number of conjuncts in the antecedent (must be >= 2).

    Returns
    -------
    rules : list of dict
        Same schema as _gen_atomic_rules, but 'antecedent' is now a tuple of
        (feature_name, feature_value) pairs (one per conjunct), sorted by key.
        Sorted by lift descending.
    """
    flat_records = [_flatten_predicates(p) for p in all_predicates]
    n = len(flat_records)
    if n == 0 or max_size < 2:
        return []

    consequent_keys = {"ego_speed_change", "ego_turn"}

    # ── seed: unique antecedent (key, val) conditions from filtered atomic rules ──
    seed_conditions = [
        rule["antecedent"]
        for rule in atomic_rules_filtered
        if rule["antecedent"][0] not in consequent_keys
    ]
    seed_conditions = list(dict.fromkeys(seed_conditions))  # deduplicate, preserve order

    def unique_q_vals(key):
        return {r[key] for r in flat_records if r[key] is not None}

    def count_cond(conditions):
        """Count records satisfying all (key, val) conditions."""
        return sum(
            1 for r in flat_records
            if all(r.get(k) == v for k, v in conditions)
        )

    def count_cond_and_q(conditions, q_key, q_val):
        return sum(
            1 for r in flat_records
            if all(r.get(k) == v for k, v in conditions) and r[q_key] == q_val
        )

    # ── level-1 frequent itemsets from seed conditions ────────────────────────
    frequent_itemsets = {1: {}}
    for cond in seed_conditions:
        sup = count_cond([cond])
        if sup >= min_support:
            frequent_itemsets[1][frozenset([cond])] = sup

    rules = []

    for size in range(2, max_size + 1):
        prev = frequent_itemsets.get(size - 1, {})
        if not prev:
            break

        # ── candidate generation: union pairs of (size-1)-itemsets ───────────
        prev_list = list(prev.keys())
        candidates = set()
        for a, b in combinations(prev_list, 2):
            union = a | b
            if len(union) != size:
                continue
            # all feature keys must be distinct (no contradictory values)
            keys_in_union = [k for k, _v in union]
            if len(keys_in_union) != len(set(keys_in_union)):
                continue
            # Apriori pruning: every (size-1)-subset must be frequent
            if all(frozenset(sub) in prev for sub in combinations(union, size - 1)):
                candidates.add(union)

        # ── count support; mine rules ─────────────────────────────────────────
        frequent_itemsets[size] = {}
        for itemset in candidates:
            conditions = list(itemset)
            sup = count_cond(conditions)
            if sup < min_support:
                continue

            frequent_itemsets[size][itemset] = sup
            antecedent_tuple = tuple(sorted(conditions))  # deterministic, serialisable

            for q_key in consequent_keys:
                for q_val in unique_q_vals(q_key):
                    total_Q = sum(1 for r in flat_records if r[q_key] == q_val)
                    support = count_cond_and_q(conditions, q_key, q_val)

                    if support < min_support:
                        continue

                    confidence = support / sup
                    if confidence < min_confidence:
                        continue

                    lift = confidence / (total_Q / n) if total_Q > 0 else 0.0

                    rules.append({
                        "antecedent":  antecedent_tuple,
                        "consequent":  (q_key, q_val),
                        "support":     support,
                        "total_P":     sup,
                        "total_Q":     total_Q,
                        "n":           n,
                        "confidence":  confidence,
                        "lift":        lift,
                    })

    rules.sort(key=lambda r: r["lift"], reverse=True)
    return rules


######################## Visualization Functions #######################

def _vis_seg_predicates(vid, seg_prims, seg_objs, objects_mat, fps=10, output_dir=None):
    """Generate one short MP4 per segment showing the predicates for that segment
    overlaid on the original video frames.

    Output: *output_dir*/seg_predicate_vis/<vid>/seg_NNNN_framesS-E.mp4
    """
    import cv2
    import pathlib

    if output_dir is None:
        output_dir = config.get_output_path("pipeline_output") / "seg_predicate_vis"
    seg_dir = pathlib.Path(output_dir) / str(vid)
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Determine canonical frame size from first loadable frame
    canvas_w, canvas_h = 640, 360
    found = False
    for fi in range(objects_mat.shape[0]):
        if found:
            break
        for obj_id in range(objects_mat.shape[1]):
            cell = objects_mat[fi, obj_id]
            if cell is not None and cell.get("frame") is not None:
                try:
                    sample = utils.load_frame(cell["frame"])
                    canvas_h, canvas_w = sample.shape[:2]
                    found = True
                    break
                except Exception:
                    continue

    line_h = 28
    padding = 10

    for seg_idx, (prim, objs) in enumerate(zip(seg_prims, seg_objs)):
        start, end = prim["start"], prim["end"]
        predicates = extract_predicates(prim, objs)
        flat_rec = _flatten_predicates(predicates)
        pred_lines = [
            f"ego_speed_change : {flat_rec.get('ego_speed_change')}",
            f"ego_turn         : {flat_rec.get('ego_turn')}",
            f"has_front_obj    : {flat_rec.get('has_front_obj')}",
            f"has_close_obj    : {flat_rec.get('has_close_obj')}",
            f"has_approaching  : {flat_rec.get('has_approaching_obj')}",
            f"has_moving_away  : {flat_rec.get('has_moving_away_obj')}",
            f"front_approaching: {flat_rec.get('front_obj_approaching')}",
            f"close_approaching: {flat_rec.get('close_obj_approaching')}",
            f"nearest_front_rm : {flat_rec.get('nearest_front_rel_motion')}",
        ]
        txt_h = len(pred_lines) * line_h + 2 * padding
        total_h = canvas_h + txt_h

        out_path = seg_dir / f"seg_{seg_idx:04d}_frames{start}-{end}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (canvas_w, total_h))

        for fi in range(start, end):
            frame_img = None
            for obj_id in range(objects_mat.shape[1]):
                cell = objects_mat[fi, obj_id]
                if cell is not None and cell.get("frame") is not None:
                    try:
                        frame_img = utils.load_frame(cell["frame"])
                        break
                    except Exception:
                        continue
            if frame_img is None:
                frame_img = np.full((canvas_h, canvas_w, 3), 180, dtype=np.uint8)
            if frame_img.shape[0] != canvas_h or frame_img.shape[1] != canvas_w:
                frame_img = cv2.resize(frame_img, (canvas_w, canvas_h))

            header = (
                f"Seg {seg_idx}  [{start},{end}]  frame {fi}  "
                f"speed={flat_rec.get('ego_speed_change')}  turn={flat_rec.get('ego_turn')}"
            )
            cv2.putText(frame_img, header, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 50), 2, cv2.LINE_AA)

            # draw bounding boxes for all objects visible in this frame
            for obj in objs:
                bbox = obj.get("frame_bboxes", {}).get(fi)
                if bbox is not None:
                    try:
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        cv2.rectangle(frame_img, (x1, y1), (x2, y2), (0, 255, 80), 2)
                        lbl = obj.get("label") or ""
                        if lbl:
                            cv2.putText(frame_img, lbl, (x1, max(y1 - 4, 12)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1, cv2.LINE_AA)
                    except (TypeError, IndexError):
                        pass

            txt_panel = np.full((txt_h, canvas_w, 3), 30, dtype=np.uint8)
            for li, line in enumerate(pred_lines):
                y = padding + li * line_h + line_h // 2
                cv2.putText(txt_panel, line, (padding, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.50, (80, 200, 255), 1, cv2.LINE_AA)

            composite = np.vstack([frame_img, txt_panel])
            writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

        writer.release()

    print(f"Saved {len(seg_prims)} segment predicate videos to {seg_dir}")


def _vis_rules(vid, all_predicates, rules, top_k=10, fps=10, output_dir=None):
    """
    For a single video, produce an MP4 where every original video frame is shown
    with rule annotations:
      - top panel    : original frame with a yellow segment-info header
      - bottom panel : fixed-height dark panel listing the top-k rules whose
                       antecedent fires in the current segment

    Output: *output_dir*/<vid>_rules.mp4

    Parameters
    ----------
    vid : str
        Video ID to visualize.
    all_predicates : list of dict
        Per-segment predicate dicts (all videos). Predicates for *vid* are
        re-derived internally so that raw frame images are accessible.
    rules : list of dict
        Rules from _gen_atomic_rules / _ext_atomic_rules.  'antecedent' is
        either a single (key, val) tuple (atomic) or a tuple of (key, val)
        pairs (conjunctive).
    top_k : int
        Maximum number of rules checked per segment (highest lift first).
    fps : int
        Frames per second for the output MP4.
    output_dir : Path-like or None
        Destination directory; created if absent.
    """
    import cv2
    import pathlib

    if output_dir is None:
        output_dir = config.get_output_path("pipeline_output") / "rule_vis"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. re-derive segments / objects / predicates for this video ───────────
    ego_sig_seg = _pre_steps(vid)
    ego_segs    = _combine_sig_seg(ego_sig_seg)
    seg_prims   = _seg2prims(ego_segs)
    seg_objs    = _extract_objs_per_segs(seg_prims, ego_sig_seg["objects"])
    objects_mat = ego_sig_seg["objects"]          # shape (num_frames, num_objs)
    num_frames  = objects_mat.shape[0]

    vid_predicates = [
        extract_predicates(prim, seg_objs[p_i])
        for p_i, prim in enumerate(seg_prims)
    ]
    flat_records = [_flatten_predicates(p) for p in vid_predicates]

    # frame-index → (seg_idx, flat_record, prim) lookup
    frame_to_seg = {}
    for seg_idx, prim in enumerate(seg_prims):
        for fi in range(prim["start"], prim["end"]):
            frame_to_seg[fi] = (seg_idx, flat_records[seg_idx], prim)

    # ── 2. top-k rules by lift ────────────────────────────────────────────────
    top_rules = sorted(rules, key=lambda r: r["lift"], reverse=True)[:top_k]

    def rule_fires(flat_record, rule):
        ant = rule["antecedent"]
        if isinstance(ant[0], tuple):
            return all(flat_record.get(k) == v for k, v in ant)
        k, v = ant
        return flat_record.get(k) == v

    def rule_label(rule):
        ant = rule["antecedent"]
        if isinstance(ant[0], tuple):
            p_str = " & ".join(f"{k}={v}" for k, v in ant)
        else:
            k, v = ant
            p_str = f"{k}={v}"
        q_key, q_val = rule["consequent"]
        return (f"{p_str} -> {q_key}={q_val}  "
                f"[sup={rule['support']} conf={rule['confidence']:.2f} "
                f"lift={rule['lift']:.2f}]")

    # ── 3. canonical frame size from the first loadable frame ────────────────
    canvas_w, canvas_h = 640, 360  # fallback defaults
    for fi in range(num_frames):
        found = False
        for obj_id in range(objects_mat.shape[1]):
            cell = objects_mat[fi, obj_id]
            if cell is not None and cell.get("frame") is not None:
                try:
                    sample = utils.load_frame(cell["frame"])
                    canvas_h, canvas_w = sample.shape[:2]
                    found = True
                    break
                except Exception:
                    continue
        if found:
            break

    line_h    = 28
    padding   = 10
    max_txt_h = top_k * line_h + 2 * padding   # fixed height keeps video dims constant
    total_h   = canvas_h + max_txt_h

    # ── 4. open VideoWriter ───────────────────────────────────────────────────
    out_path = output_dir / f"{vid}_rules.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (canvas_w, total_h))

    # ── 5. write every frame ──────────────────────────────────────────────────
    for fi in range(num_frames):
        frame_img = None
        for obj_id in range(objects_mat.shape[1]):
            cell = objects_mat[fi, obj_id]
            if cell is not None and cell.get("frame") is not None:
                try:
                    frame_img = utils.load_frame(cell["frame"])
                    break
                except Exception:
                    continue

        if frame_img is None:
            frame_img = np.full((canvas_h, canvas_w, 3), 180, dtype=np.uint8)

        if frame_img.shape[0] != canvas_h or frame_img.shape[1] != canvas_w:
            frame_img = cv2.resize(frame_img, (canvas_w, canvas_h))

        seg_entry = frame_to_seg.get(fi)
        if seg_entry is not None:
            seg_idx, flat_rec, prim = seg_entry
            start, end = prim["start"], prim["end"]
            fired  = [r for r in top_rules if rule_fires(flat_rec, r)]
            header = (f"Seg {seg_idx}  [{start},{end}]  frame {fi}  "
                      f"speed={flat_rec.get('ego_speed_change')}  "
                      f"turn={flat_rec.get('ego_turn')}")
        else:
            fired  = []
            header = f"frame {fi}  (outside segmented range)"

        cv2.putText(frame_img, header, (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 50), 2, cv2.LINE_AA)

        txt_panel = np.full((max_txt_h, canvas_w, 3), 30, dtype=np.uint8)
        if not fired:
            cv2.putText(txt_panel, "No rules fired.",
                        (padding, padding + line_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            for li, rule in enumerate(fired):
                y = padding + li * line_h + line_h // 2
                cv2.putText(txt_panel, rule_label(rule), (padding, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 220, 100), 1, cv2.LINE_AA)

        composite = np.vstack([frame_img, txt_panel])          # (total_h, canvas_w, 3)
        writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved rule-vis video ({num_frames} frames) to {out_path}")


######################## Pipeline functions #########################
def _combine_sig_seg(sig_seg, min_duration=5):
    """
    The sig_seg is a dict including the signals and their segmentation for ego_w, ego_vs, and ego_vz.
    We will combine the segmentation results of the three signals 
    to get a more comprehensive segmentation for the ego motion.
    """
    ego_w_signal = sig_seg["ego_w_signal"]
    ego_vx_signal = sig_seg["ego_vx_signal"]
    ego_vz_signal = sig_seg["ego_vz_signal"]
    ego_w_segs = sig_seg["ego_w_segs"]
    ego_vx_segs = sig_seg["ego_vx_segs"]
    ego_vz_segs = sig_seg["ego_vz_segs"]
    
    # merge the segments from the three signals by splitting at all unique segment boundaries
    boundaries = set()
    for segs in [ego_w_segs, ego_vx_segs, ego_vz_segs]:
        for s, e in segs:
            boundaries.add(s)
            boundaries.add(e)
    boundaries = sorted(boundaries) 
    
    # merge the short segments that are shorter than a certain duration threshold (e.g., 5 frames) into their neighboring segments
    i = 0
    while i < len(boundaries) - 1:
        if boundaries[i + 1] - boundaries[i] < min_duration:
            if i > 0:
                boundaries.pop(i)
            else:
                boundaries.pop(i + 1)
        else:
            i += 1    
    combined_segs = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    
    # merge segments if their symbolic state is consistent across all three signals (e.g., all increasing or all decreasing)
    merged_prims = []
    for s, e in combined_segs:
        merged_prims.append({
            "start":s,
            "end":e,
            "signal_ego_w": ego_w_signal[s:e],
            "signal_ego_vx": ego_vx_signal[s:e],
            "signal_ego_vz": ego_vz_signal[s:e]
        })
    return merged_prims



def _seg2prims(sig_seg):
    # extract features for each segment, 
    # here we simply use the mean and variance of the signal within the segment as features
    prims = []
    for seg in sig_seg:
        s, e = seg["start"], seg["end"]
        seg_signal = seg["signal_ego_w"]  # assuming we're using ego_w signal for feature extraction
        
        # determine the trend of the signal within the segment: increasing, decreasing, or stable
        trend_type = 1 if seg_signal[-1] > seg_signal[0] else -1 if seg_signal[-1] < seg_signal[0] else 0
        
        prims.append({
            "start": s,
            "end": e,
            "duration": e - s,
            "mean": np.mean(seg_signal),
            "min": np.min(seg_signal),
            "max": np.max(seg_signal),
            "amplitude": np.max(seg_signal) - np.min(seg_signal),
            "var": np.var(seg_signal),
            "slope": (seg_signal[-1] - seg_signal[0]) / (e - s + 1e-6),  # to avoid division by zero
            "trend": trend_type,
            "mean_abs_diff": np.mean(np.abs(np.diff(seg_signal))),
            "max_abs_diff": np.max(np.abs(np.diff(seg_signal))),
            "energy": np.sum(seg_signal ** 2),
            "yaw_mean": np.mean(seg["signal_ego_w"])  # add mean yaw as a feature for predicate extraction
        })
        
    return prims


######################## Test functions #########################
def _pre_steps(vid):   
    # extract the time series position matrix for the video
    matrix = percept2matrix(vid, save_matrices_flag=True)
    flow = percept2ego_speed(vid, save_flow_flag=True)
    # convert the position matrix to signals
    objs, ego_motion = matrix2signal(matrix, vid, visualize_ego=True, save_primitives=True)
    signal_ego_w = ego_motion[:, 2]
    signal_ego_vz = ego_motion[:, 1]
    signal_ego_vx = ego_motion[:, 0]
    
    # segment the signals
    ego_w_segs = signal2segs(signal_ego_w)
    ego_vx_segs = signal2segs(signal_ego_vx)
    ego_vz_segs = signal2segs(signal_ego_vz)
    
    sig_seg = {
        "ego_w_signal": signal_ego_w,
        "ego_vx_signal": signal_ego_vx,
        "ego_vz_signal": signal_ego_vz,
        "ego_w_segs": ego_w_segs,
        "ego_vx_segs": ego_vx_segs, 
        "ego_vz_segs": ego_vz_segs,
        "objects": objs
    }
    return sig_seg     
    
def main_test():
    vids = config.get_mini_video_ids()

    # ── collect predicates across all videos and all segments ────────────────
    all_predicates = []
    for vid in vids:
        ego_sig_seg = _pre_steps(vid)
        ego_segs = _combine_sig_seg(ego_sig_seg)
        seg_prims = _seg2prims(ego_segs)
        seg_objs = _extract_objs_per_segs(seg_prims, ego_sig_seg["objects"])

        for p_i, prim in enumerate(seg_prims):
            objs = seg_objs[p_i]
            predicates = extract_predicates(prim, objs)
            all_predicates.append(predicates)

        # ── visualize each segment's predicates as a short MP4 ───────────────
        _vis_seg_predicates(vid, seg_prims, seg_objs, ego_sig_seg["objects"])


    # ── mine atomic (|P|=1) rules P → Q ──────────────────────────────────────
    # Q candidates: ego_speed_change, ego_turn
    # P candidates: object-presence and relative-motion flags (see _gen_atomic_rules)
    atomic_rules = _gen_atomic_rules(all_predicates, min_support=2, min_confidence=0.0)

    # score rules: based on support and confidence
    print(f"Found {len(atomic_rules)} atomic rules from {len(all_predicates)} segments.")
    for rule in atomic_rules[:20]:   # print top-20 by lift
        p_key, p_val = rule["antecedent"]
        q_key, q_val = rule["consequent"]
        print(
            f"  [{p_key}={p_val}] -> [{q_key}={q_val}] "
            f"sup={rule['support']} conf={rule['confidence']:.2f} lift={rule['lift']:.2f}"
        )
    
    # save rules to a CSV file
    rules_df = pd.DataFrame(atomic_rules)
    rule_file = config.get_output_path("pipeline_output") / "mined_atomic_rules.csv"
    rules_df.to_csv(rule_file, index=False)
    print(f"Saved mined rules to {rule_file}")
    # before the extensive rule mining, 
    # filter out rules with low support (e.g., less than 5 segments) or low confidence (e.g., less than 0.5) to focus on more promising patterns.
    atomic_rules_filtered = [r for r in atomic_rules if r["support"] >= 5 and r["confidence"] >= 0.5]
    print(f"Filtered down to {len(atomic_rules_filtered)} rules with support >= 5 and confidence >= 0.5.")
    
    
    # extend to more complex rules (|P|>1) by combining antecedents, 
    # e.g., P1 AND P2 → Q, and evaluate their support/confidence/lift similarly.  
    # This can be done by iterating over pairs of antecedent features and counting co-occurrences in the flat_records.
    rules_extended = _ext_atomic_rules(all_predicates, atomic_rules_filtered, min_support=5, min_confidence=0.5)
    
    # turn scored rules into a compact rule set that can be easily interpreted and visualized, 
    # e.g., by selecting top-k rules by lift or confidence, 
    # and by grouping rules with similar antecedents or consequents together.
    compact_rules_df = pd.DataFrame(rules_extended)
    compact_rule_file = config.get_output_path("pipeline_output") / "mined_extended_rules.csv"
    compact_rules_df.to_csv(compact_rule_file, index=False)
    print(f"Saved extended rules to {compact_rule_file}")
    
    
    # a visualization of the rules, the video at the top
    # below the video, each activated rule is displayed as a text description
    _vis_rules(vids[0], all_predicates, rules_extended)
    
    
    
    

if __name__ == "__main__":
    main_test()