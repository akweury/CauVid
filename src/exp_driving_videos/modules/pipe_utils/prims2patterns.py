from collections import defaultdict
from itertools import combinations
import json
import math 
import numpy as np
import pathlib

import config 
from exp_driving_videos.modules.pipe_utils.predicate_vocab import extract_predicates


def _get_pattern_cfg(cfg=None):
    from exp_driving_videos.modules.pipe_utils import exp_driving_utils as utils

    return utils.get_pattern_cfg(cfg)


######################## Helper functions for pattern detection ########################
def _extract_ego_per_segs(segments, prim_data):
    ego_in_segments = []
    for (start, end, seg_type) in segments:
        ego_speed_x = prim_data["ego_x_speed"][start:end]
        ego_speed_z = prim_data["ego_z_speed"][start:end]
        ego_stop_mask = prim_data["ego_stop_mask"][start:end]
        ego_turn_mask = prim_data["ego_turn_mask"][start:end]
    
        ego_in_segments.append({
            "start": start,
            "end": end,
            "type": seg_type,
            "speed_x": ego_speed_x,
            "speed_z": ego_speed_z,
            "stop_mask": ego_stop_mask,
            "turn_mask": ego_turn_mask,
        })
    return ego_in_segments
         
         
         

def _estimate_obj_z_motion(obj_speeds):
    """
    Return segment-level motion stats instead of frame labels.
    """
    vz = np.array([float(s[1]) for s in obj_speeds])
    
    valid_mask = vz != -1 
    vz = vz[valid_mask]
    
    if len(vz) == 0:
        obj_motion = -1
        return obj_motion
    
    # adaptive thresholds
    eps = 0.1 * np.std(vz) if np.std(vz) > 0 else 0.01
    
    # -- ratios --
    approach_ratio = (vz<-eps).mean()
    away_ratio = (vz>eps).mean()
    
    # -- mean / strength ---
    vz_mean = np.mean(vz)
    vz_std = np.std(vz)
    
    # --- change detection ---
    sign = np.sign(vz)
    change_ratio = (np.diff(sign) != 0).mean() if len(sign) > 1 else 0
    
    # final label
    if approach_ratio > 0.7:
        label = "approaching"
    elif away_ratio > 0.7:
        label = "moving_away"
    else:
        label = "transition"
        
    obj_motion = {
        "label": label,
        "approach_ratio": approach_ratio,
        "away_ratio": away_ratio,
        "vz_mean": vz_mean,
        "vz_std": vz_std,
        "change_ratio": change_ratio
    }
    return obj_motion 

def _estimate_obj_x_motion(obj_speeds):
    """
    Similar to estimate_obj_z_motion but for lateral (x) motion.
    """
    vx = np.array([float(s[0]) for s in obj_speeds])
    
    valid_mask = vx != -1 
    vx = vx[valid_mask]
    
    if len(vx) == 0:
        obj_motion = -1
        return obj_motion
    
    eps = 0.1 * np.std(vx) if np.std(vx) > 0 else 0.01
    
    left_ratio = (vx<-eps).mean()
    right_ratio = (vx>eps).mean()
    
    if left_ratio > 0.7:
        label = "left"
    elif right_ratio > 0.7:
        label = "right"
    else:
        label = "stable"
        
    obj_motion = {
        "label": label,
        "left_ratio": left_ratio,
        "right_ratio": right_ratio,
        "vx_mean": np.mean(vx),
        "vx_std": np.std(vx)
    }
    return obj_motion

def _extract_objs_per_segs(seg_prims, objects):
    """
    For each segment primitive, collect the objects that are visible within that segment's
    frame range and return per-segment summaries compatible with extract_predicates.
    """
    
    # seg_type: 0 normal, 1 stop, 2 turn
    obj_ids = list(objects['obj_speeds'].keys())
    objs_in_segments = []
    for (start, end, seg_type) in seg_prims:
        objs_in_seg = []
        # iterative over objects
        for obj_id in obj_ids:
            obj_seg_speed = objects['obj_speeds'][obj_id][start:end]
            obj_seg_rank = objects['obj_ranks'][obj_id][start:end]
            obj_seg_speeds_rel = objects['obj_speeds'][obj_id][start:end]
            obj_seg_vz = _estimate_obj_z_motion(objects['obj_speeds'][obj_id][start:end])
            obj_seg_vx = _estimate_obj_x_motion(objects['obj_speeds'][obj_id][start:end])

            if np.all(obj_seg_speed == -1):
                continue  # skip objects not visible in this segment
            objs_in_seg.append({
                "id":         obj_id,
                "speed":     obj_seg_speed,
                "rank":      obj_seg_rank,
                "speeds_rel": obj_seg_speeds_rel,
                "vz": obj_seg_vz,
                "vx": obj_seg_vx,
            })
        objs_in_segments.append(objs_in_seg)
    return objs_in_segments

def _gen_atoms(timeline_predicates):
    
    # todo: add obj id to the atoms
    
    timeline_atoms = []
    for t in range(len(timeline_predicates)):
        
        pred_t = timeline_predicates[t]
        atoms_t = []
        # ego
        for k,v in pred_t['ego'].items():
            atoms_t.append((k,v[0]))
        
        # focus objects
        for obj in pred_t['focus_objects']:
            atoms_t.append(("rel_motion",obj['rel_motion'][0]))
        
        # global
        for k, v in pred_t['global'].items():
            atoms_t.append((k,v))
        
        timeline_atoms.append(atoms_t)
    return timeline_atoms 

def _gen_atomic_rules(timeline_atoms, cfg=None):
    """
    
    output: rules[(P,Q,dt)] = count
    
    """
    def is_trivial_rule(p,q):
        return p[0]==q[0] and p[1]==q[1]
    
    def is_valid_rule(p,q):
        p_type = p[0]
        q_type = q[0]
        
        # only keep object -> ego
        if "ego" in q_type:
            return True
        return False
    
    def is_valid_Q(q):
        key, value = q
        if key == "ego_speed_change" and value != "constant_speed":
            return True
        if key == "ego_is_turning" and value != "not_turning":
            return True
        
        return False
    
    def is_informative_P(p):
        key, value = p
        
        if key == "rel_motion" and value == "moving_away":
            return False 
        
        return True
    
    cfg = _get_pattern_cfg(cfg)
    max_lag = cfg["rules"]["max_lag"]

    rule_counts = defaultdict(int)
    P_counts = defaultdict(int)
    Q_counts = defaultdict(int)
    
    T = len(timeline_atoms)
    for t in range(T):
        for q in timeline_atoms[t]:
            Q_counts[q] += 1
    
    for t in range(T):
        P_atoms = timeline_atoms[t]
        for p in P_atoms:
            P_counts[p] += 1
            
        for dt in range(1, max_lag + 1):
            if t+dt>=T:
                continue
            Q_atoms = timeline_atoms[t+dt]
            for p in P_atoms:
                for q in Q_atoms:
                    rule = (p,q,dt)
                    if is_trivial_rule(p,q):
                        continue
                    if not is_valid_rule(p,q):
                        continue
                    # if not is_valid_Q(q):
                    #     continue
                    if not is_informative_P(p):
                        continue
                    rule_counts[rule] +=1
    
    print(f"Atomic rules generated: {len(rule_counts)}")  
    print(f"Unique P conditions: {len(P_counts)}")
    print(f"Unique Q conditions: {len(Q_counts)}")      
    
    return rule_counts, P_counts, Q_counts

def _score_rules_global(global_rule_counts, global_P_counts, global_Q_counts, total_T, rule_video_support, num_videos, cfg=None):
    
    """
    Score rules based on confidence, lift, and stability across videos.
    """
    cfg = _get_pattern_cfg(cfg)
    eps = cfg["rules"]["eps"]
    if total_T == 0 or num_videos == 0:
        return []
    scored_rules = []
    for (p,q,dt), count in global_rule_counts.items():
        
        support = count
        
        if global_P_counts[p]==0:
            continue
        
        confidence = support / global_P_counts[p]
        
        # P(Q)
        p_q = global_Q_counts[q]/total_T
        
        # lift: confidence / P(Q)
        # lift > 1 means P increases the likelihood of Q compared to random chance; lift < 1 means P decreases the likelihood of Q.
        lift = confidence / (p_q + eps)  # add small constant to avoid division by zero
        
        # log-lift:
        log_lift = math.log(lift + eps)  # add small constant to avoid log(0)
        
        # stability: fraction of videos where the rule holds among those where P holds
        stability = len(rule_video_support[(p,q,dt)]) / num_videos
        
        # score
        score = confidence * log_lift * stability
        
        scored_rules.append({
            "rule": (p,q,dt),
            "support": support,
            "confidence": confidence,
            "lift": lift,
            "log_lift": log_lift,
            "stability": stability,
            "score": score
        })
    return scored_rules
    
def _filter_rules(scored_rules, cfg=None):
    cfg = _get_pattern_cfg(cfg)
    rule_cfg = cfg["rules"]
    min_support = rule_cfg["min_support"]
    min_conf = rule_cfg["min_conf"]
    min_lift = rule_cfg["min_lift"]
    min_stability = rule_cfg["min_stability"]

    filtered = []
    for r in scored_rules:
        if (r["support"] >= min_support and 
            r["confidence"] >= min_conf and 
            r["lift"] >= min_lift and
            r["stability"] >= min_stability):
            filtered.append(r)
    print(f"Rules after filtering: {len(filtered)}")
    return filtered

def _get_top_rules(filtered_rules, cfg=None):
    cfg = _get_pattern_cfg(cfg)
    top_k = cfg["rules"]["top_k"]
    top_rules = sorted(filtered_rules, key=lambda x: x["score"], reverse=True)[:top_k]
    return top_rules

def _format_rules(rules):
    formated_rules = []
    for r in rules:
        p, q, dt = r["rule"]
        formated_rules.append((
            f"{q[0]}={q[1]} (Δt={dt}) :- {p[0]}={p[1]}. | "
            f"conf={r['confidence']:.2f}, "
            f"lift={r['lift']:.2f}, "
            f"stab={r['stability']:.2f}, "
            f"score={r['score']:.3f}"
        ))
    return formated_rules

######################## Visualization Functions #######################

def _vis_seg_predicates(vid, frame_data, predicates, seg_objs, seg_ego, cfg=None, output_dir=None):
    """Generate one short MP4 showing segment frames with predicate annotations.

    Output: *output_dir*/seg_predicate_vis/<vid>/seg_framesS-E_predicates.mp4
    """
    import cv2
    from exp_driving_videos.modules.pipe_utils import exp_driving_utils as utils

    cfg = _get_pattern_cfg(cfg)
    fps = cfg["predicate"]["vis_fps"]

    fps = max(1, int(fps))
    start = int(seg_ego.get("start", 0))
    end = int(seg_ego.get("end", start + 1))
    seg_type = int(seg_ego.get("type", 0))
    duration = max(1, end - start)
    num_frames = len(frame_data)
    start = max(0, min(start, max(0, num_frames - 1)))
    end = max(start + 1, min(end, num_frames))
    duration = max(1, end - start)

    sample_img = None
    for sample_fi in range(start, end):
        try:
            sample_img = utils.load_frame(frame_data[sample_fi]["frame"])
            break
        except Exception:
            continue
    if sample_img is None:
        sample_img = np.full((360, 640, 3), 180, dtype=np.uint8)

    frame_h, canvas_w = sample_img.shape[:2]
    panel_h = max(280, min(360, int(frame_h * 0.45)))
    canvas_h = frame_h + panel_h
    pad = max(14, int(canvas_w * 0.025))
    line_h = max(20, int(panel_h * 0.075))
    font = cv2.FONT_HERSHEY_SIMPLEX
    body_scale = max(0.42, min(0.56, canvas_w / 1600))
    title_scale = max(0.50, min(0.66, canvas_w / 1350))
    frame_text_scale = max(0.68, min(0.95, canvas_w / 1050))
    bbox_text_scale = max(0.58, min(0.82, canvas_w / 1200))

    def _label_conf(value):
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return str(value[0]), float(value[1])
            except (TypeError, ValueError):
                return str(value[0]), None
        return str(value), None

    def _fmt_value(value):
        label, conf = _label_conf(value)
        if conf is None:
            return label
        return f"{label} ({conf:.2f})"

    def _obj_id_key(obj_id):
        if hasattr(obj_id, "item"):
            obj_id = obj_id.item()
        return str(obj_id)

    def _put_text(img, text, x, y, scale=0.55, color=(230, 230, 230), thickness=1):
        cv2.putText(img, str(text), (int(x), int(y)), font, scale, color, thickness, cv2.LINE_AA)

    def _draw_section(img, title, lines, x, y, w, h, accent):
        cv2.rectangle(img, (x, y), (x + w, y + h), (34, 34, 34), cv2.FILLED)
        cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)
        cv2.rectangle(img, (x, y), (x + 6, y + h), accent, cv2.FILLED)
        _put_text(img, title, x + 16, y + int(28 * title_scale / 0.62), title_scale, accent, 2)
        cy = y + int(54 * title_scale / 0.62)
        for line in lines:
            if cy > y + h - 12:
                _put_text(img, "...", x + 16, cy, body_scale, (160, 160, 160), 1)
                break
            _put_text(img, line, x + 16, cy, body_scale, (225, 225, 225), 1)
            cy += line_h

    def _load_frame_or_blank(fi):
        if fi < 0 or fi >= num_frames:
            return np.full((frame_h, canvas_w, 3), 180, dtype=np.uint8)
        try:
            frame_img = utils.load_frame(frame_data[fi]["frame"])
        except Exception:
            frame_img = np.full((frame_h, canvas_w, 3), 180, dtype=np.uint8)
        if frame_img.shape[0] != frame_h or frame_img.shape[1] != canvas_w:
            frame_img = cv2.resize(frame_img, (canvas_w, frame_h))
        return frame_img

    def _draw_frame_overlays(frame_img, fi):
        header = f"Video {vid} | frames {start}-{end} | frame {fi} | {seg_type_text}"
        overlay = frame_img.copy()
        header_h = max(48, int(54 * frame_text_scale / 0.68))
        cv2.rectangle(overlay, (0, 0), (canvas_w, header_h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.42, frame_img, 0.58, 0, frame_img)
        _put_text(frame_img, header, 10, int(header_h * 0.68), frame_text_scale, (255, 255, 90), 2)

        info = frame_data[fi] if 0 <= fi < num_frames else {}
        bboxes = info.get("bboxes", [])
        obj_ids = info.get("obj_ids", [])
        for idx, bbox in enumerate(bboxes):
            bb = bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)
            if len(bb) < 4:
                continue
            x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            obj_id = obj_ids[idx] if idx < len(obj_ids) else idx
            focus_obj = focus_obj_by_id.get(_obj_id_key(obj_id))
            if focus_obj is None:
                continue

            rel_text = _label_conf(focus_obj.get("rel_motion", "unknown"))[0]
            turning_text = _label_conf(focus_obj.get("is_turning", "unknown"))[0]
            bbox_label = f"{rel_text} | {turning_text}"
            bbox_color = (40, 40, 255)
            bbox_thickness = 3
            cv2.rectangle(frame_img, (x1, y1), (x2, y2), bbox_color, bbox_thickness)

            (text_w, text_h), baseline = cv2.getTextSize(bbox_label, font, bbox_text_scale, 2)
            label_y = max(text_h + 6, y1 - 8)
            label_x = min(max(0, x1), max(0, canvas_w - text_w - 8))
            bg_y1 = max(0, label_y - text_h - 4)
            bg_y2 = min(frame_h, label_y + baseline + 4)
            label_overlay = frame_img.copy()
            cv2.rectangle(label_overlay, (label_x - 4, bg_y1), (label_x + text_w + 4, bg_y2),
                          (0, 0, 0), cv2.FILLED)
            cv2.addWeighted(label_overlay, 0.42, frame_img, 0.58, 0, frame_img)
            _put_text(frame_img, bbox_label, label_x, label_y, bbox_text_scale, bbox_color, 2)

        ego_box_lines = [
            f"speed: {_fmt_value(ego_preds.get('ego_speed_change', 'unknown'))}",
            f"moving: {_fmt_value(ego_preds.get('ego_is_moving', 'unknown'))}",
            f"turning: {_fmt_value(ego_preds.get('ego_is_turning', 'unknown'))}",
        ]
        ego_scale = max(0.58, min(0.82, canvas_w / 1200))
        ego_thick = 2
        ego_pad = max(8, int(10 * ego_scale))
        text_sizes = [cv2.getTextSize(line, font, ego_scale, ego_thick) for line in ego_box_lines]
        box_w = max(size[0][0] for size in text_sizes) + 2 * ego_pad
        box_h = sum(size[0][1] + size[1] for size in text_sizes) + (len(ego_box_lines) + 1) * ego_pad
        box_x = pad
        box_y = max(header_h + pad, frame_h - box_h - pad)

        ego_overlay = frame_img.copy()
        cv2.rectangle(ego_overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(ego_overlay, 0.48, frame_img, 0.52, 0, frame_img)

        text_y = box_y + ego_pad
        for line, ((_, text_h), baseline) in zip(ego_box_lines, text_sizes):
            text_y += text_h
            _put_text(frame_img, line, box_x + ego_pad, text_y, ego_scale, (80, 220, 255), ego_thick)
            text_y += baseline + ego_pad

    def _draw_predicate_panel(progress, current_frame):
        panel = np.full((panel_h, canvas_w, 3), 22, dtype=np.uint8)
        _put_text(panel, "predicate summary", pad, 28, title_scale, (245, 245, 245), 2)
        _put_text(panel, f"segment frame {current_frame} / {end - 1}",
                  pad, 54, body_scale, (170, 170, 170), 1)

        bar_x, bar_y, bar_w, bar_h = pad, 70, canvas_w - 2 * pad, 8
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), cv2.FILLED)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h),
                      (80, 200, 255), cv2.FILLED)

        col_w = (canvas_w - 3 * pad) // 2
        row1_y = 96
        row_h = max(84, int((panel_h - row1_y - pad) * 0.38))
        row2_y = row1_y + row_h + pad
        row2_h = max(100, panel_h - row2_y - pad)
        _draw_section(panel, "Ego Predicates", ego_lines, pad, row1_y, col_w, row_h, (80, 200, 255))
        _draw_section(panel, "Global Predicates", global_lines,
                      2 * pad + col_w, row1_y, col_w, row_h, (120, 220, 120))
        _draw_section(panel, "Focus Objects", focus_lines, pad, row2_y, col_w, row2_h, (220, 190, 80))
        _draw_section(panel, "Object Predicates", object_lines,
                      2 * pad + col_w, row2_y, col_w, row2_h, (210, 140, 230))
        return panel

    ego_preds = predicates.get("ego", {})
    global_preds = predicates.get("global", {})
    object_preds = predicates.get("objects", [])
    focus_preds = predicates.get("focus_objects", [])
    focus_obj_by_id = {_obj_id_key(obj.get("id")): obj for obj in focus_preds}

    ego_lines = [
        f"speed_change : {_fmt_value(ego_preds.get('ego_speed_change', 'unknown'))}",
        f"is_moving    : {_fmt_value(ego_preds.get('ego_is_moving', 'unknown'))}",
        f"is_turning   : {_fmt_value(ego_preds.get('ego_is_turning', 'unknown'))}",
    ]
    global_lines = [
        f"any_approach    : {global_preds.get('any_approach', False)}",
        f"num_approaching : {global_preds.get('num_approaching', 0)}",
        f"visible_objects : {len(seg_objs)}",
        f"focus_objects   : {len(focus_preds)}",
    ]
    focus_lines = []
    for obj in focus_preds[:5]:
        focus_lines.append(
            f"id={obj.get('id')} score={float(obj.get('score', 0.0)):.2f} "
            f"rank={float(obj.get('rank', -1.0)):.1f} "
            f"rel={_label_conf(obj.get('rel_motion', 'unknown'))[0]} "
            f"lat={_label_conf(obj.get('lateral_motion', 'unknown'))[0]}"
        )
    if not focus_lines:
        focus_lines.append("none")

    object_lines = []
    for obj in object_preds[:10]:
        object_lines.append(
            f"id={obj.get('id')} rank={float(obj.get('rank', -1.0)):.1f} "
            f"rel={_label_conf(obj.get('rel_motion', 'unknown'))[0]} "
            f"lat={_label_conf(obj.get('lateral_motion', 'unknown'))[0]} "
            f"move={_label_conf(obj.get('is_moving', 'unknown'))[0]} "
            f"turn={_label_conf(obj.get('is_turning', 'unknown'))[0]}"
        )
    if len(object_preds) > 10:
        object_lines.append(f"... {len(object_preds) - 10} more objects")
    elif not object_lines:
        object_lines.append("none")

    out_path = output_dir / f"seg_frames{start}-{end}_predicates.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {out_path}")

    total_frames = max(fps * 2, min(fps * 4, duration))
    seg_type_text = {0: "normal", 1: "stopped", 2: "turning"}.get(seg_type, str(seg_type))
    for frame_i in range(total_frames):
        progress = 1.0 if total_frames <= 1 else frame_i / (total_frames - 1)
        current_frame = start + min(duration - 1, int(round(progress * (duration - 1))))
        frame_img = _load_frame_or_blank(current_frame)
        _draw_frame_overlays(frame_img, current_frame)
        pred_panel = _draw_predicate_panel(progress, current_frame)
        writer.write(cv2.cvtColor(np.vstack([frame_img, pred_panel]), cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved predicate visualization to {out_path}")
    return out_path

######################## Pipeline functions #########################

def load_primitive_data(path, cfg=None):
    from exp_driving_videos.modules.pipe_utils import exp_driving_utils as utils
    _get_pattern_cfg(cfg)

    path = pathlib.Path(path)
    pipeline_file = path / "pipeline_data.pkl"
    if pipeline_file.exists():
        import pickle

        def _read_stage(stage_name):
            entry = pipeline_data.get("stages", {}).get(stage_name)
            if isinstance(entry, dict) and entry.get("_stage_cache"):
                return entry.get("data")
            return entry

        with open(pipeline_file, "rb") as f:
            pipeline_data = pickle.load(f)

        ego_data = _read_stage("ego_data")
        video_data = _read_stage("video_data")
        obj_motion_data = _read_stage("obj_motion_data")
        if ego_data is None or video_data is None or obj_motion_data is None:
            missing = [
                name
                for name, value in {
                    "ego_data": ego_data,
                    "video_data": video_data,
                    "obj_motion_data": obj_motion_data,
                }.items()
                if value is None
            ]
            raise FileNotFoundError(
                f"{pipeline_file} is missing required stage(s): {', '.join(missing)}"
            )

        ego_x_speed = np.asarray(ego_data["ego_x_speeds"])
        ego_z_speed = np.asarray(ego_data["ego_z_speeds"])
        ego_stop_mask = np.asarray(ego_data["stopped_mask"])
        ego_turn_mask = np.asarray(ego_data["turning_mask"])
        frame_data = video_data["frames_data"]
    else:
        ego_x_speed = np.load(path / "ego_x_speeds.npy")
        ego_z_speed = np.load(path / "ego_z_speeds.npy")
        ego_stop_mask = np.load(path / "stop_mask.npy")
        ego_turn_mask = np.load(path / "turning_mask.npy")
        frame_data = utils.load_matrix(path / "video_data.pkl")['frames_data']
        # load npz file containing the object motion data, which is a dict with keys "obj_ids", "dz_series", "labels",
        # and "frame_bboxes"
        loaded = np.load(path / "obj_motion_data.npz", allow_pickle=True)
        obj_motion_data = {
            key: loaded[key].item()
            if loaded[key].shape == () and loaded[key].dtype == object
            else loaded[key]
            for key in loaded.files
        }
    primitive_data = {
        "ego_x_speed": ego_x_speed,
        "ego_z_speed": ego_z_speed, 
        "ego_stop_mask": ego_stop_mask,
        "ego_turn_mask": ego_turn_mask,
        "obj_motion_data": obj_motion_data,
        "frame_data": frame_data
    }
    return primitive_data


def segment_ego_signal(primitive_data, cfg=None):
    _get_pattern_cfg(cfg)
    ego_stop_mask = primitive_data["ego_stop_mask"]
    ego_turn_mask = primitive_data["ego_turn_mask"]
    # combine the two masks to get a more comprehensive segmentation of the ego signal
    combined_mask = np.zeros_like(ego_stop_mask, dtype=int)
    # stop mask has two values: 0 for non-stop, 1 for stop; 
    # turn mask has two values: 0 for non-turn, 1 for turn
    combined_mask[ego_stop_mask == 1] = 1   # stop segments
    combined_mask[ego_turn_mask == 1] = 2    # turn segments (overlapping segments will be labeled as turn for now)
    
    # add (s,e) tuples for each contiguous segment in the combined mask, along with the type of segment (stop or turn)
    segments = []
    current_type = 0
    start_idx = 0
    for i in range(1, len(combined_mask)):
        if combined_mask[i] != current_type:
            if current_type != 0:  # only consider non-zero segments
                segments.append((start_idx, i, current_type))
            start_idx = i
            current_type = combined_mask[i]
    # handle the last segment
    if current_type != 0:
        segments.append((start_idx, len(combined_mask), current_type))
    
    return segments
    
    
def extract_video_predicates(vid, out_path, cfg=None):
    cfg = _get_pattern_cfg(cfg)
    pred_cfg = cfg["predicate"]
    
    out_file = out_path / "predicates.json"
    # if out_file.exists() and not pred_cfg["force_recompute"]:
    #     print(f"Predicates for video {vid} already exist at {out_file}, loading...")
    #     with open(out_file, "r") as f:
    #         return json.load(f)
    
    # load ego data
    prim_data = load_primitive_data(out_path, cfg)
    
    # segment the ego data based on ego_stop_mask and ego_turn_mask
    segments = segment_ego_signal(prim_data, cfg)
    objs = _extract_objs_per_segs(segments, prim_data["obj_motion_data"])
    ego = _extract_ego_per_segs(segments, prim_data)
    timeline_predicates = []
    for seg_objs, seg_ego in zip(objs, ego):
        predicates_t = extract_predicates(seg_objs, seg_ego)
        timeline_predicates.append(predicates_t)
        # ── visualize each segment's predicates as a short MP4 ───────────────
        if pred_cfg["visualize"]:
            _vis_seg_predicates(vid, prim_data["frame_data"], predicates_t, seg_objs, seg_ego, cfg, output_dir=out_path)
    
    # save predicates to JSON file
    with open(out_file, "w") as f:
        json.dump(timeline_predicates, f, indent=2)
    print(f"Saved predicates for video {vid} to {out_file}")
    
    return timeline_predicates 


def main_test(cfg=None):
    """
    drawback: 
    - rules have to be filtered based on carefully designed functions such as is_valid_rule, is_trivial_rule
    
    
    """
    
    
    import pandas as pd

    cfg = _get_pattern_cfg(cfg)
    vids = cfg["video_ids"] or config.get_mini_video_ids()
    output_root = cfg["output_root"] or config.get_output_path("pipeline_output")
    
    global_rule_counts = defaultdict(float)
    global_P_counts = defaultdict(float)
    global_Q_counts = defaultdict(float)    
    rule_video_support = defaultdict(set)  # rule → set of vids that support it
    all_predicates = []
    
    total_T = 0
    
    for vid in vids:
        print(f"Processing video {vid}...")
        out_path = pathlib.Path(output_root) / f"{vid}"
        # preprocess the video to extract primitive data and save to out_path
        preprocess_cfg = cfg.get("preprocess", {})
        if preprocess_cfg.get("enabled", cfg.get("run_preprocess", False)):
            from exp_driving_videos.modules.perception_pipeline import run_single_video
            run_single_video(vid, cfg)
        
        # ---------------  Predicates Extraction -------------------------------
        video_predicates = extract_video_predicates(vid, out_path, cfg)
        all_predicates.extend(video_predicates)
        print(f"Extracted predicates video {vid}, total {len(video_predicates)} segments.")

        # --------------- Generate atomic rules (length-1, with time lag delta t) -------------------------------
        timeline_atoms = _gen_atoms(video_predicates)
        rule_counts, P_counts, Q_counts= _gen_atomic_rules(timeline_atoms, cfg)
        total_T += len(timeline_atoms)
        print(f"[{vid}] T={len(timeline_atoms)}, rules={len(rule_counts)}")
        
        # --------------- Aggregate rule counts across videos -------------------------------
        for r,c in rule_counts.items():
            global_rule_counts[r] += c 
            rule_video_support[r].add(vid)
            
        for p,c in P_counts.items():
            global_P_counts[p] += c
            
        for q,c in Q_counts.items():
            global_Q_counts[q] += c
        
        # debug
        print("Total segments:", total_T)
        print("Total unique rules:", len(global_rule_counts))

    # -------------- Scoring (support / confidence / causal strength) -------------------------------
    # score rules: based on support and confidence
    scored_rules = _score_rules_global(global_rule_counts, global_P_counts, global_Q_counts, total_T, rule_video_support, num_videos=len(vids), cfg=cfg)
    formated_rules = _format_rules(scored_rules)
    
    print("All scored rules:")
    for r in formated_rules:
        print(r)
    # -------------- Filtering ---------------------------------
    # before the extensive rule mining, filter the atomic rules to a smaller set based on minimum support and confidence thresholds,
    filtered = _filter_rules(scored_rules, cfg)
    top_rules = _get_top_rules(filtered, cfg)
    formated_rules = _format_rules(top_rules)
    for r in formated_rules:
        print(r)
    
    if cfg["extended_rules"]["enabled"]:
        print("Extended rule mining is enabled, but the current global atomic-rule schema does not provide antecedent/consequent fields required by _ext_atomic_rules.")

    compact_rules_df = pd.DataFrame(top_rules)
    compact_rule_file = pathlib.Path(output_root) / cfg["rules"]["output_file"]
    compact_rules_df.to_csv(compact_rule_file, index=False)
    print(f"Saved atomic rules to {compact_rule_file}")

    if cfg["rule_vis"]["enabled"]:
        print("Rule visualization is skipped because global atomic rules use the compact (p, q, dt) schema, while _vis_rules expects antecedent/consequent rule records.")
    
    
    
    

if __name__ == "__main__":
    main_test()
