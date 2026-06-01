"""Visualization helpers extracted from perception pipeline."""

import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

from exp_driving_videos.modules.pipe_utils import exp_driving_utils as utils
import config

def trajectory_with_frames_visual(matrices, smooth_matrices, input_data, output_path):
    os.makedirs(output_path, exist_ok=True)
    # for each trajectory, show the smoothed trajectory on the left, highlight the current position, 
    # and the image on the right,bounding box the the target object, saved as a gif file
    
    def create_trajectory_figure_with_current(position, current_idx, obj_id, label, target_height, title_suffix=""):
        """
        Create trajectory plot with current position highlighted.
        
        Args:
            position: Full trajectory positions (tensor)
            current_idx: Index of current frame to highlight
            obj_id: Object ID
            label: Object label
            target_height: Target height in pixels for the output image
        
        Returns:
            Trajectory image as numpy array (RGB)
        """
        position = torch.stack(position)  # Ensure position is a tensor of shape (num_frames, 3)
        # Create the trajectory plot
        fig, ax = plt.subplots(figsize=(5, 5))
        x = position[:, 0].numpy()
        z = position[:, 2].numpy()
        
        # Plot full trajectory
        ax.plot(x, z, marker='o', linestyle='-', linewidth=2, markersize=4, color='gray', alpha=0.5)
        
        # Highlight trajectory up to current frame
        if current_idx > 0:
            ax.plot(x[:current_idx+1], z[:current_idx+1], marker='o', linestyle='-', 
                   linewidth=2, markersize=4, color='blue', label='Trajectory')
        
        # Add start marker
        ax.scatter(x[0], z[0], color='green', s=150, marker='o', label='Start', zorder=5, edgecolors='black', linewidths=2)
        
        # Add current position marker (larger and distinct)
        ax.scatter(x[current_idx], z[current_idx], color='red', s=200, marker='*', 
                  label='Current', zorder=10, edgecolors='black', linewidths=2)
        
        # Add end marker
        ax.scatter(x[-1], z[-1], color='orange', s=150, marker='X', label='End', zorder=5, edgecolors='black', linewidths=2)
        
        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Z (meters)", fontsize=12)
        ax.set_title(f"Object {obj_id} ({label})\nFrame {current_idx+1}/{len(position)} {title_suffix}", 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        
        # Convert matplotlib figure to numpy array
        fig.canvas.draw()
        buf = np.array(fig.canvas.renderer.buffer_rgba())
        trajectory_img = buf[:, :, :3]  # Convert RGBA to RGB
        plt.close(fig)
        
        # Resize to match target height
        trajectory_height, trajectory_width = trajectory_img.shape[:2]
        new_width = int(trajectory_width * target_height / trajectory_height)
        trajectory_img_resized = cv2.resize(trajectory_img, (new_width, target_height))
        
        return trajectory_img_resized
        
        
    for obj_id, data in matrices.items():
        label = data["label"]
        position = data["position"]
        frames = data["frames"]
        bboxes = data["bboxes"]
        
        # each bbox is one frame, so we can visualize the trajectory and the frame with bbox side by side
        visual_frames = []
        for i, (bbox, frame) in enumerate(zip(bboxes, frames)):
            frame_img = utils.load_frame(frame, bbox, obj_id, label)  # Load the frame and draw the bounding box
            frame_height = frame_img.shape[0]
            
            # Create trajectory plot with current position highlighted
            trajectory_img_current = create_trajectory_figure_with_current(
                position, i, obj_id, label, frame_height
            )
            
            # Create smoothed trajectory plot with current position highlighted
            trajectory_img_current_smoothed = create_trajectory_figure_with_current(
                smooth_matrices[obj_id]["position"], i, obj_id, label, frame_height, title_suffix="(Smoothed)"
                
            )
            
            # Combine trajectory and frame side by side
            combined_img = np.hstack((trajectory_img_current, 
                                      trajectory_img_current_smoothed, 
                                      frame_img))
            visual_frames.append(combined_img)
        
        # save the visual frames as a gif, with 2 fps
        safe_label = label.replace(' ', '_').replace('/', '_')
        gif_path = output_path / f"obj_{obj_id}_{safe_label}_{len(bboxes)}frames.gif"
        imageio.mimsave(gif_path, visual_frames, fps=2)
        print(f"Saved trajectory gif for Object {obj_id} ({label}): {gif_path}")


def visualize_full_scene(frames_data, video_id, ego_data, obj_data, out_path, fps=5):
    """
    Produce a single MP4 showing every frame with ALL tracked objects annotated.

    Per frame:
      - Rank-1 (closest) object: bright-green bbox, thicker border, "Rank #1" label
      - All other present objects: cyan-yellow bbox, normal border, "Rank #N | <label>" label
      - Above every object bbox: object motion-mask state, e.g. "Rel: Approaching | Left"
      - Bottom-left: ego motion overlay (Moving/Stopped, Straight/Turning)

    Output: *out_path*/full_scene.mp4
    """
    out_file = out_path / f"full_scene.mp4"
    # if video is exist then return
    
    # if out_file.exists():
    #     print(f"Full-scene video already exists for video {video_id}, skipping generation: {out_file}")
    #     return
    
    
    

    obj_bboxes = obj_data.get("obj_bboxes", {})
    obj_ranks  = obj_data.get("obj_ranks",  {})
    obj_z_motion = obj_data.get("motion_vz")
    obj_x_motion = obj_data.get("motion_vx")

    _Z_LABEL = {
        -1: ("Absent",      (80,  80,  80)),
         0: ("Same Dist.",  (0,  200,   0)),
         1: ("Approaching", (0,   60, 220)),
         2: ("Moving Away", (220, 80,   0)),
    }
    _X_LABEL = {
        -1: ("Absent",  (80,  80,  80)),
         0: ("Stable",  (180, 180,   0)),
         1: ("Left",    (255, 100, 100)),
         2: ("Right",   (100, 100, 255)),
    }

    ego_vx_full      = list(ego_data["ego_x_speeds"])
    ego_vz_full      = list(ego_data["ego_z_speeds"])
    num_frames       = len(frames_data)
    ego_stopped_full = list(ego_data.get("stopped_mask", [False] * num_frames))
    ego_turn_full    = list(ego_data.get("turn_mask",    [False] * num_frames))
    for _sig in (ego_vx_full, ego_vz_full, ego_stopped_full, ego_turn_full):
        if len(_sig) < num_frames:
            _sig += [_sig[-1]] * (num_frames - len(_sig))
    ego_stopped_full = [bool(v) for v in ego_stopped_full[:num_frames]]
    ego_turn_full    = [bool(v) for v in ego_turn_full[:num_frames]]

    # precompute per-object label lookup {obj_id: {fi: label_str}}
    obj_label_map = {}
    for fi, frame_data in enumerate(frames_data):
        obj_ids_fi = frame_data.get("obj_ids", [])
        labels_fi  = frame_data.get("labels", [])
        for k, oid in enumerate(obj_ids_fi):
            if oid not in obj_label_map:
                obj_label_map[oid] = {}
            obj_label_map[oid][fi] = labels_fi[k] if k < len(labels_fi) else str(oid)



    first_img     = utils.load_frame(frames_data[0]["frame"])
    frame_h, frame_w = first_img.shape[:2]

    
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_file), fourcc, fps, (frame_w, frame_h))

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.7, frame_w / 1000)
    thickness  = max(2, int(font_scale * 2))
    pad        = int(10 * font_scale)

    # colour constants
    _COLOR_RANK1   = (0, 255, 0)        # bright green  – closest object
    _COLOR_OTHER   = (0, 220, 220)      # cyan-yellow   – all other objects
    _THICK_RANK1   = thickness * 2
    _THICK_OTHER   = thickness

    def _motion_text_and_color(oid, fi):
        """Return a concise object motion-mask row for the full-scene overlay."""
        prefix = "Rel"
        z_track = obj_z_motion.get(oid) if obj_z_motion is not None else None
        x_track = obj_x_motion.get(oid) if obj_x_motion is not None else None
        if z_track is None and x_track is None:
            return None, (200, 200, 200)

        z_cat = z_track[fi] if z_track is not None and fi < len(z_track) else -1
        x_cat = x_track[fi] if x_track is not None and fi < len(x_track) else -1
        z_text, z_color = _Z_LABEL.get(z_cat, ("Unknown", (128, 128, 128)))
        x_text, x_color = _X_LABEL.get(x_cat, ("Unknown", (128, 128, 128)))
        mask_color = x_color if x_cat in (1, 2) else z_color
        return f"{prefix}: {z_text} | {x_text}", mask_color

    def _draw_label_rows(img, x, y_top, rows):
        row_gap = max(2, int(3 * font_scale))
        line_sizes = [cv2.getTextSize(text, font, scale, thick) for text, scale, thick, _ in rows]
        total_h = sum(sz[0][1] + sz[1] for sz in line_sizes) + row_gap * (len(rows) - 1)
        max_w = max(sz[0][0] for sz in line_sizes)

        text_x = min(max(0, x), max(0, frame_w - max_w - 8))
        first_baseline = max(y_top - 4, total_h + 4)
        y_positions = []
        cursor = first_baseline
        for idx in range(len(rows) - 1, -1, -1):
            (_tw, th), baseline = line_sizes[idx]
            y_positions.insert(0, cursor)
            cursor -= th + baseline + row_gap

        bg_x1 = max(0, text_x - 4)
        bg_x2 = min(frame_w, text_x + max_w + 8)
        bg_y1 = max(0, y_positions[0] - line_sizes[0][0][1] - 3)
        bg_y2 = min(frame_h, y_positions[-1] + line_sizes[-1][1] + 3)
        label_ov = img.copy()
        cv2.rectangle(label_ov, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(label_ov, 0.4, img, 0.6, 0, img)

        for (text, scale, thick, color), y_pos in zip(rows, y_positions):
            cv2.putText(img, text, (text_x, y_pos), font, scale, color, thick, cv2.LINE_AA)

    for fi, frame_data in enumerate(frames_data):
        frame_img = utils.load_frame(frame_data["frame"])
        if frame_img.shape[0] != frame_h or frame_img.shape[1] != frame_w:
            frame_img = cv2.resize(frame_img, (frame_w, frame_h))
        frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)

        # ── draw all tracked objects ──────────────────────────────────────────
        for oid, fi_bbox_map in obj_bboxes.items():
            bb = fi_bbox_map.get(fi)
            if bb is None:
                continue
            bb = bb.tolist() if hasattr(bb, "tolist") else list(bb)
            ox1, oy1, ox2, oy2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])

            rank_track = obj_ranks.get(oid, [])
            rank_val   = rank_track[fi] if fi < len(rank_track) else -1

            is_closest = (rank_val == 1)
            color      = _COLOR_RANK1 if is_closest else _COLOR_OTHER
            bthick     = _THICK_RANK1 if is_closest else _THICK_OTHER

            cv2.rectangle(frame_bgr, (ox1, oy1), (ox2, oy2), color, bthick)

            rank_str  = f"Rank #{rank_val}" if rank_val >= 1 else "Rank: ?"
            obj_label = obj_label_map.get(oid, {}).get(fi, str(oid))
            title_str = f"{rank_str} | {obj_label}"
            motion_str, motion_color = _motion_text_and_color(oid, fi)

            # draw label rows just above the bbox
            label_rows = [(title_str, font_scale * 0.8, bthick, color)]
            if motion_str:
                label_rows.append((motion_str, font_scale * 0.7, _THICK_OTHER, motion_color))
            _draw_label_rows(frame_bgr, ox1, oy1, label_rows)

        # ── ego state label — bottom-left ─────────────────────────────────────
        ego_z_text  = "Stopped" if ego_stopped_full[fi] else "Moving"
        ego_z_color = (255, 165, 0)   if ego_stopped_full[fi] else (200, 200, 200)
        ego_x_text  = "Turning" if ego_turn_full[fi] else "Straight"
        ego_x_color = (180, 100, 220) if ego_turn_full[fi] else (200, 200, 200)

        e_lines = [("Ego", (255, 255, 255)), (ego_z_text, ego_z_color), (ego_x_text, ego_x_color)]
        e_sizes = [cv2.getTextSize(t, font, font_scale, thickness) for t, _ in e_lines]
        ebox_w  = max(sz[0][0] for sz in e_sizes) + 3 * pad
        ebox_h  = sum(sz[0][1] + sz[1] for sz in e_sizes) + (len(e_lines) + 1) * pad
        eby0    = frame_h - ebox_h - pad
        ego_ov  = frame_bgr.copy()
        cv2.rectangle(ego_ov, (pad, eby0), (pad + ebox_w, eby0 + ebox_h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(ego_ov, 0.45, frame_bgr, 0.55, 0, frame_bgr)
        ey = eby0
        for (etxt, ecol), ((_etw, _eth), _ebl) in zip(e_lines, e_sizes):
            ey += _eth + pad
            cv2.putText(frame_bgr, etxt, (2 * pad, ey), font, font_scale, ecol, thickness, cv2.LINE_AA)
            ey += _ebl

        writer.write(frame_bgr)

    writer.release()
    print(f"Saved full-scene visualization ({num_frames} frames) → {out_file}")


def visualize_obj_speed(frames_data, video_id, ego_data, obj_motion_data,
                        fps=5, chart_width=10, chart_height=3, output_dir=None):
    """
    Produce one MP4 per tracked object with the video frame on top and four
    signal subplots stacked below:
      1. Object vz (forward)  — Approaching (steelblue) / Moving Away (tomato)
      2. Object vx (lateral)  — Left (cornflowerblue)   / Right (salmon)
      3. Ego    vz (forward)  — plain speed signal
      4. Ego    vx (lateral)  — plain speed signal

    State labels for object z-motion and x-motion are overlaid top-left on the
    frame (z-motion on line 1, x-motion on line 2).

    Z-motion categories  (-1/0/1/2): absent / same-dist / approaching / moving-away
    X-motion categories  (-1/0/1/2): absent / stable    / left        / right

    Output: *output_dir*/<video_id>_obj_<obj_id>_speed.mp4
    """
    import pathlib

    # z-motion: category → (display text, BGR colour)
    _Z_LABEL = {
        -1: ("Absent",      (80,  80,  80)),
         0: ("Same Dist.",  (0,  200,   0)),
         1: ("Approaching", (0,   60, 220)),
         2: ("Moving Away", (220, 80,   0)),
    }
    # x-motion: category → (display text, BGR colour)
    _X_LABEL = {
        -1: ("Absent",  (80,  80,  80)),
         0: ("Stable",  (180, 180,   0)),
         1: ("Left",    (255, 100, 100)),
         2: ("Right",   (100, 100, 255)),
    }
    ego_motion = (ego_data["ego_x_speeds"], ego_data["ego_z_speeds"])
    obj_speeds = obj_motion_data.get("obj_speeds", {})
    obj_z_motion_abs_mask = obj_motion_data.get("mask_o_vz_abs", None)
    obj_x_motion_abs_mask = obj_motion_data.get("mask_o_vx_abs", None)
    obj_z_motion_rel_mask = obj_motion_data.get("mask_o_vz_rel", None)
    obj_x_motion_rel_mask = obj_motion_data.get("mask_o_vx_rel", None)
    obj_ranks  = obj_motion_data.get("obj_ranks",  {})
    obj_bboxes = obj_motion_data.get("obj_bboxes", {})
    
    if output_dir is None:
        output_dir = pathlib.Path(config.get_output_path("pipeline_output")) / "obj_speed_vis"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def is_absent(s):
        return float(s[0]) == -1.0 and float(s[1]) == -1.0

    num_frames = len(frames_data)

    # ── unpack and align ego signals + masks to num_frames ──────────────────
    ego_vx_full = list(ego_data["ego_x_speeds"])
    ego_vz_full = list(ego_data["ego_z_speeds"])
    ego_stopped_full = list(ego_data.get("stopped_mask", [False] * num_frames))
    ego_turn_full    = list(ego_data.get("turn_mask",    [False] * num_frames))
    for _sig in (ego_vx_full, ego_vz_full, ego_stopped_full, ego_turn_full):
        if len(_sig) < num_frames:
            _sig += [_sig[-1]] * (num_frames - len(_sig))
    ego_vx_full      = [float(v) for v in ego_vx_full[:num_frames]]
    ego_vz_full      = [float(v) for v in ego_vz_full[:num_frames]]
    ego_stopped_full = [bool(v)  for v in ego_stopped_full[:num_frames]]
    ego_turn_full    = [bool(v)  for v in ego_turn_full[:num_frames]]

    # ── precompute per-object bbox/label lookup ──────────────────────────────
    obj_frame_info = {}
    for fi, frame_data in enumerate(frames_data):
        obj_ids_fi = frame_data.get("obj_ids", [])
        bboxes_fi  = frame_data.get("bboxes", [])
        labels_fi  = frame_data.get("labels", [])
        for k, oid in enumerate(obj_ids_fi):
            if oid not in obj_frame_info:
                obj_frame_info[oid] = {}
            bbox  = bboxes_fi[k].tolist() if hasattr(bboxes_fi[k], "tolist") else list(bboxes_fi[k])
            label = labels_fi[k] if k < len(labels_fi) else str(oid)
            obj_frame_info[oid][fi] = (bbox, label)

    # ── canvas dimensions ────────────────────────────────────────────────────
    first_img  = utils.load_frame(frames_data[0]["frame"])
    frame_h, frame_w = first_img.shape[:2]

    # ── per object ───────────────────────────────────────────────────────────
    for obj_id, speed_track in obj_speeds.items():
        speed_track = np.array(speed_track, dtype=np.float32)
        if speed_track.ndim == 1:
            speed_track = speed_track.reshape(1, 2)
        if len(speed_track) < num_frames:
            pad = np.full((num_frames - len(speed_track), 2), -1.0, dtype=np.float32)
            speed_track = np.vstack([speed_track, pad])
        speed_track = speed_track[:num_frames]

        # ── trim trailing absent frames ──────────────────────────────────────
        last_present = -1
        for _fi in range(len(speed_track) - 1, -1, -1):
            if not (float(speed_track[_fi][0]) == -1.0 and float(speed_track[_fi][1]) == -1.0):
                last_present = _fi
                break
        if last_present == -1:
            continue
        active_frames = last_present + 1
        speed_track   = speed_track[:active_frames]

        # z-motion track
        z_track = None
        if obj_z_motion_rel_mask is not None:
            z_track = list(obj_z_motion_rel_mask.get(obj_id, [-1] * num_frames))
            if len(z_track) < num_frames:
                z_track += [-1] * (num_frames - len(z_track))
            z_track = z_track[:active_frames]

        # x-motion track (relative)
        x_track = None
        if obj_x_motion_rel_mask is not None:
            x_track = list(obj_x_motion_rel_mask.get(obj_id, [-1] * num_frames))
            if len(x_track) < num_frames:
                x_track += [-1] * (num_frames - len(x_track))
            x_track = x_track[:active_frames]

        # z-motion track (absolute)
        z_abs_track = None
        if obj_z_motion_abs_mask is not None:
            z_abs_track = list(obj_z_motion_abs_mask.get(obj_id, [-1] * num_frames))
            if len(z_abs_track) < num_frames:
                z_abs_track += [-1] * (num_frames - len(z_abs_track))
            z_abs_track = z_abs_track[:active_frames]

        # x-motion track (absolute)
        x_abs_track = None
        if obj_x_motion_abs_mask is not None:
            x_abs_track = list(obj_x_motion_abs_mask.get(obj_id, [-1] * num_frames))
            if len(x_abs_track) < num_frames:
                x_abs_track += [-1] * (num_frames - len(x_abs_track))
            x_abs_track = x_abs_track[:active_frames]

        # ── build segments for obj vz chart (Approaching / Moving Away) ──────
        vz_segments:   list = []
        vz_seg_colors: list = []
        vz_seg_labels: list = []
        if z_track is not None:
            _i = 0
            while _i < len(z_track):
                _cat = z_track[_i]
                if _cat in (1, 2):
                    _j = _i
                    while _j < len(z_track) and z_track[_j] == _cat:
                        _j += 1
                    vz_segments.append((_i, _j))
                    vz_seg_colors.append('steelblue' if _cat == 1 else 'tomato')
                    vz_seg_labels.append('Approaching' if _cat == 1 else 'Moving Away')
                    _i = _j
                else:
                    _i += 1

        # ── build segments for obj vx chart (Left / Right) ───────────────────
        vx_segments:   list = []
        vx_seg_colors: list = []
        vx_seg_labels: list = []
        if x_track is not None:
            _i = 0
            while _i < len(x_track):
                _cat = x_track[_i]
                if _cat in (1, 2):
                    _j = _i
                    while _j < len(x_track) and x_track[_j] == _cat:
                        _j += 1
                    vx_segments.append((_i, _j))
                    vx_seg_colors.append('cornflowerblue' if _cat == 1 else 'salmon')
                    vx_seg_labels.append('Left' if _cat == 1 else 'Right')
                    _i = _j
                else:
                    _i += 1

        # ── object speed signals (NaN where absent) ───────────────────────────
        vz_signal = [float('nan') if is_absent(s) else float(s[1]) for s in speed_track]
        vx_signal = [float('nan') if is_absent(s) else float(s[0]) for s in speed_track]

        # ── ego signals and masks trimmed to active_frames ──────────────────
        ego_vz_signal     = ego_vz_full[:active_frames]
        ego_vx_signal     = ego_vx_full[:active_frames]
        ego_stopped_signal = ego_stopped_full[:active_frames]
        ego_turn_signal    = ego_turn_full[:active_frames]

        # build Stopped segments for ego vz chart
        ego_vz_segs:   list = []
        ego_vz_colors: list = []
        ego_vz_slabels: list = []
        _i = 0
        while _i < len(ego_stopped_signal):
            if ego_stopped_signal[_i]:
                _j = _i
                while _j < len(ego_stopped_signal) and ego_stopped_signal[_j]:
                    _j += 1
                ego_vz_segs.append((_i, _j))
                ego_vz_colors.append('orange')
                ego_vz_slabels.append('Stopped')
                _i = _j
            else:
                _i += 1

        # build Turning segments for ego vx chart
        ego_vx_segs:   list = []
        ego_vx_colors: list = []
        ego_vx_slabels: list = []
        _i = 0
        while _i < len(ego_turn_signal):
            if ego_turn_signal[_i]:
                _j = _i
                while _j < len(ego_turn_signal) and ego_turn_signal[_j]:
                    _j += 1
                ego_vx_segs.append((_i, _j))
                ego_vx_colors.append('mediumpurple')
                ego_vx_slabels.append('Turning')
                _i = _j
            else:
                _i += 1

        info_map   = obj_frame_info.get(obj_id, {})
        obj_label  = next((v[1] for v in info_map.values()), str(obj_id))
        safe_label = obj_label.replace(' ', '_').replace('/', '_')

        # ── probe chart cell dimensions once ─────────────────────────────────
        # Each cell is half the frame width; use chart_width/2 so matplotlib
        # proportions stay reasonable.
        cell_w = frame_w // 2
        sample_cell = utils.create_timeline_line_chart_img(
            vz_signal, 0, vz_segments, vz_seg_colors,
            f"{obj_label} Obj Vz",
            width=chart_width / 2, height=chart_height,
            segment_labels=vz_seg_labels,
        )
        cell_h  = int(sample_cell.shape[0] * cell_w / sample_cell.shape[1])
        total_h = frame_h + 2 * cell_h

        out_path = output_dir / f"{video_id}_obj_{obj_id}_{safe_label}_speed.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, total_h))

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.7, frame_w / 1000)
        thickness  = max(2, int(font_scale * 2))
        pad        = int(10 * font_scale)

        def _draw_frame_axes(img):
            axis_len = max(36, int(min(frame_w, frame_h) * 0.08))
            origin_x = pad * 2
            origin_y = pad * 2
            label_scale = max(0.45, font_scale * 0.65)
            label_thickness = max(1, thickness - 1)
            axis_thickness = max(2, thickness)
            x_color = (255, 220, 80)
            z_color = (80, 220, 255)

            box_pad = max(6, pad // 2)
            box_x2 = origin_x + axis_len + int(24 * label_scale) + box_pad
            box_y2 = origin_y + axis_len + int(18 * label_scale) + box_pad
            axes_ov = img.copy()
            cv2.rectangle(
                axes_ov,
                (max(0, origin_x - box_pad), max(0, origin_y - box_pad)),
                (min(frame_w, box_x2), min(frame_h, box_y2)),
                (0, 0, 0),
                cv2.FILLED,
            )
            cv2.addWeighted(axes_ov, 0.35, img, 0.65, 0, img)

            cv2.arrowedLine(
                img,
                (origin_x, origin_y),
                (origin_x + axis_len, origin_y),
                x_color,
                axis_thickness,
                cv2.LINE_AA,
                tipLength=0.25,
            )
            cv2.arrowedLine(
                img,
                (origin_x, origin_y),
                (origin_x, origin_y + axis_len),
                z_color,
                axis_thickness,
                cv2.LINE_AA,
                tipLength=0.25,
            )
            cv2.putText(
                img,
                "x",
                (origin_x + axis_len + box_pad // 2, origin_y + int(5 * label_scale)),
                font,
                label_scale,
                x_color,
                label_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                "z",
                (origin_x - int(5 * label_scale), origin_y + axis_len + int(14 * label_scale)),
                font,
                label_scale,
                z_color,
                label_thickness,
                cv2.LINE_AA,
            )

        for fi, frame_data in enumerate(frames_data[:active_frames]):
            frame_img = utils.load_frame(frame_data["frame"])
            if frame_img.shape[0] != frame_h or frame_img.shape[1] != frame_w:
                frame_img = cv2.resize(frame_img, (frame_w, frame_h))

            frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
            _draw_frame_axes(frame_bgr)
            s = speed_track[fi]

            z_cat = z_track[fi] if z_track is not None else (0 if not is_absent(s) else -1)
            x_cat = x_track[fi] if x_track is not None else (0 if not is_absent(s) else -1)
            z_text, bbox_color = _Z_LABEL.get(z_cat, ("Unknown", (128, 128, 128)))
            x_text, x_color    = _X_LABEL.get(x_cat, ("Unknown", (128, 128, 128)))

            # absolute motion categories
            z_abs_cat = z_abs_track[fi] if z_abs_track is not None else (0 if not is_absent(s) else -1)
            x_abs_cat = x_abs_track[fi] if x_abs_track is not None else (0 if not is_absent(s) else -1)
            z_abs_text, _ = _Z_LABEL.get(z_abs_cat, ("Unknown", (128, 128, 128)))
            x_abs_text, _ = _X_LABEL.get(x_abs_cat, ("Unknown", (128, 128, 128)))

            obj_lbl_font_scale = font_scale * 1.4
            obj_lbl_thickness  = max(2, int(obj_lbl_font_scale * 2))

            # ── compute target bbox area for distance comparison ──────────────
            target_area = 0.0
            if fi in info_map:
                _tb = info_map[fi][0]
                _tx1, _ty1, _tx2, _ty2 = int(_tb[0]), int(_tb[1]), int(_tb[2]), int(_tb[3])
                target_area = max(0, _tx2 - _tx1) * max(0, _ty2 - _ty1)

            # ── draw all other objects with relative-distance color ───────────
            # Uses obj_bboxes {obj_id: {fi: bbox}} built by percept2obj_dist_rank_and_bboxes.
            # Larger bbox area → closer than target → red (255,0,0); smaller → farther → gray.
            for _oid, _fi_bbox_map in obj_bboxes.items():
                if _oid == obj_id:
                    continue   # target drawn separately below
                _bb = _fi_bbox_map.get(fi)
                if _bb is None:
                    continue
                _bb = _bb.tolist() if hasattr(_bb, "tolist") else list(_bb)
                _ox1, _oy1, _ox2, _oy2 = int(_bb[0]), int(_bb[1]), int(_bb[2]), int(_bb[3])
                _area = max(0, _ox2 - _ox1) * max(0, _oy2 - _oy1)
                # closer (larger area) → red; farther (smaller area) → yellow
                _other_color = (0, 0, 255) if _area > target_area else (0, 220, 220)
                _other_thickness = thickness * 2
                cv2.rectangle(frame_bgr, (_ox1, _oy1), (_ox2, _oy2), _other_color, _other_thickness)
                # rank label above bbox
                _o_rank_track = obj_ranks.get(_oid, [])
                _o_rank_val   = _o_rank_track[fi] if fi < len(_o_rank_track) else -1
                _o_rank_str   = f"Rank #{_o_rank_val}" if _o_rank_val >= 1 else ""
                # look up label from obj_frame_info
                _olbl = obj_frame_info.get(_oid, {}).get(fi, (None, ""))[1]
                # combine rank + label into two rows (rank on top, label below)
                _label_y = max(_oy1 - 4, 10)
                if _olbl:
                    cv2.putText(frame_bgr, _olbl, (_ox1, _label_y),
                                font, font_scale * 0.7, _other_color, _other_thickness, cv2.LINE_AA)
                    _label_y = max(_label_y - int(font_scale * 0.7 * 20) - 2, 10)
                if _o_rank_str:
                    cv2.putText(frame_bgr, _o_rank_str, (_ox1, _label_y),
                                font, font_scale * 0.7, _other_color, _other_thickness, cv2.LINE_AA)

            # ── draw target object with green bbox + abs/rel labels above ─────
            if fi in info_map:
                bbox, lbl = info_map[fi]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                _target_border = (0, 200, 0)   # green for target
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), _target_border, 2)

                # rank of this object among all tracked objects (1 = closest)
                _rank_track = obj_ranks.get(obj_id, [])
                _rank_val   = _rank_track[fi] if fi < len(_rank_track) else -1
                _rank_str   = f"Rank #{_rank_val}" if _rank_val >= 0 else "Rank: ?"

                # three label rows above bbox:
                #   row 1 (highest): rank
                #   row 2: abs z + abs x
                #   row 3 (closest): rel z + rel x
                rank_row = _rank_str
                abs_row  = f"Abs: {z_abs_text} | {x_abs_text}"
                rel_row  = f"Rel: {z_text} | {x_text}"
                (_nw, _nh), _nbl = cv2.getTextSize(rank_row, font, obj_lbl_font_scale, obj_lbl_thickness)
                (_aw, _ah), _abl = cv2.getTextSize(abs_row,  font, obj_lbl_font_scale, obj_lbl_thickness)
                (_rw, _rh), _rbl = cv2.getTextSize(rel_row,  font, obj_lbl_font_scale, obj_lbl_thickness)
                row_gap = 4
                total_label_h = (_nh + _nbl) + row_gap + (_ah + _abl) + row_gap + (_rh + _rbl)
                rel_y  = max(y1 - 4, total_label_h + 4)
                abs_y  = rel_y  - (_rh + _rbl + row_gap)
                rank_y = abs_y  - (_ah + _abl + row_gap)

                # semi-transparent background behind all three rows
                bg_x1 = max(0, x1)
                bg_x2 = min(frame_w, x1 + max(_nw, _aw, _rw) + 8)
                bg_y1 = max(0, rank_y - _nh - 2)
                bg_y2 = min(frame_h, rel_y + _rbl + 2)
                lbl_ov = frame_bgr.copy()
                cv2.rectangle(lbl_ov, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), cv2.FILLED)
                cv2.addWeighted(lbl_ov, 0.4, frame_bgr, 0.6, 0, frame_bgr)

                cv2.putText(frame_bgr, rank_row, (x1, rank_y),
                            font, obj_lbl_font_scale, (0, 200, 0), obj_lbl_thickness, cv2.LINE_AA)
                cv2.putText(frame_bgr, abs_row,  (x1, abs_y),
                            font, obj_lbl_font_scale, (200, 200, 200), obj_lbl_thickness, cv2.LINE_AA)
                cv2.putText(frame_bgr, rel_row,  (x1, rel_y),
                            font, obj_lbl_font_scale, bbox_color, obj_lbl_thickness, cv2.LINE_AA)
            else:
                overlay = frame_bgr.copy()
                cv2.rectangle(overlay, (0, 0), (frame_w, frame_h), (40, 40, 40), cv2.FILLED)
                cv2.addWeighted(overlay, 0.25, frame_bgr, 0.75, 0, frame_bgr)

            # ── ego state label — bottom-left ─────────────────────────────────
            ego_z_text  = "Stopped" if ego_stopped_signal[fi] else "Moving"
            ego_z_color = (255, 165, 0)   if ego_stopped_signal[fi] else (200, 200, 200)
            ego_x_text  = "Turning" if ego_turn_signal[fi] else "Straight"
            ego_x_color = (180, 100, 220) if ego_turn_signal[fi] else (200, 200, 200)

            e_lines = [("Ego", (255, 255, 255)), (ego_z_text, ego_z_color), (ego_x_text, ego_x_color)]
            e_sizes = [cv2.getTextSize(t, font, font_scale, thickness) for t, _ in e_lines]
            ebox_w  = max(sz[0][0] for sz in e_sizes) + 3 * pad
            ebox_h  = sum(sz[0][1] + sz[1] for sz in e_sizes) + (len(e_lines) + 1) * pad
            eby0    = frame_h - ebox_h - pad   # top of box anchored to bottom
            ego_ov  = frame_bgr.copy()
            cv2.rectangle(ego_ov, (pad, eby0), (pad + ebox_w, eby0 + ebox_h), (0, 0, 0), cv2.FILLED)
            cv2.addWeighted(ego_ov, 0.45, frame_bgr, 0.55, 0, frame_bgr)
            ey = eby0
            for (etxt, ecol), ((_etw, _eth), _ebl) in zip(e_lines, e_sizes):
                ey += _eth + pad
                cv2.putText(frame_bgr, etxt, (2 * pad, ey), font, font_scale, ecol, thickness, cv2.LINE_AA)
                ey += _ebl

            frame_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            fi_str = f"frame {fi}/{active_frames - 1}"
            if not is_absent(s):
                vz_speed_str = f"  vz={float(s[1]):.3f} m/s  rel:{z_text}  abs:{z_abs_text}"
                vx_speed_str = f"  vx={float(s[0]):.3f} m/s  rel:{x_text}  abs:{x_abs_text}"
            else:
                vz_speed_str = vx_speed_str = "  absent"

            # ── 2×2 chart grid ────────────────────────────────────────────────
            # top-left : obj vx   (Left/Right shading)
            obj_vx_cell = utils.create_timeline_line_chart_img(
                vx_signal, fi, vx_segments, vx_seg_colors,
                f"{obj_label} Obj vx{vx_speed_str}",
                width=chart_width / 2, height=chart_height,
                segment_labels=vx_seg_labels,
            )
            # bottom-left: ego vx  (Turning shading)
            ego_vx_cell = utils.create_timeline_line_chart_img(
                ego_vx_signal, fi, ego_vx_segs, ego_vx_colors,
                f"{fi_str}  ego vx={ego_vx_signal[fi]:.3f} m/s [Ego vx]",
                width=chart_width / 2, height=chart_height,
                segment_labels=ego_vx_slabels,
            )
            # top-right : obj vz   (Approaching/Moving Away shading)
            obj_vz_cell = utils.create_timeline_line_chart_img(
                vz_signal, fi, vz_segments, vz_seg_colors,
                f"{obj_label} Obj vz{vz_speed_str}",
                width=chart_width / 2, height=chart_height,
                segment_labels=vz_seg_labels,
            )
            # bottom-right: ego vz  (Stopped shading)
            ego_vz_cell = utils.create_timeline_line_chart_img(
                ego_vz_signal, fi, ego_vz_segs, ego_vz_colors,
                f"{fi_str}  ego vz={ego_vz_signal[fi]:.3f} m/s [Ego vz]",
                width=chart_width / 2, height=chart_height,
                segment_labels=ego_vz_slabels,
            )

            # resize all cells to (cell_w, cell_h) and tile into 2×2
            obj_vx_cell = cv2.resize(obj_vx_cell, (cell_w, cell_h))
            ego_vx_cell = cv2.resize(ego_vx_cell, (cell_w, cell_h))
            obj_vz_cell = cv2.resize(obj_vz_cell, (cell_w, cell_h))
            ego_vz_cell = cv2.resize(ego_vz_cell, (cell_w, cell_h))

            left_col  = np.vstack([obj_vx_cell, ego_vx_cell])   # obj vx top, ego vx bottom
            right_col = np.vstack([obj_vz_cell, ego_vz_cell])   # obj vz top, ego vz bottom
            chart_panel = np.hstack([left_col, right_col])      # 2×2 grid, full frame_w wide
            # guard against 1-pixel rounding difference (odd frame_w)
            if chart_panel.shape[1] != frame_w:
                chart_panel = cv2.resize(chart_panel, (frame_w, 2 * cell_h))

            composite = np.vstack([frame_img, chart_panel])
            writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"Saved object speed visualization ({active_frames} frames) → {out_path}")



def visualize_ego_speed(frames_data, ego_x_speeds, ego_z_speeds, video_id,
                        stopped_mask=None, turning_mask=None,
                        fps=5, stop_threshold=0.01,
                        chart_width=10, chart_height=3,
                        output_dir=None):
    """
    Produce an MP4 where every frame shows:
      - top panel    : original video frame
      - bottom panel : ego-speed line chart with a red dot at the current frame,
                       and orange-shaded regions where |speed| < stop_threshold.

    Output: *output_dir*/<video_id>_ego_speed.mp4

    Parameters
    ----------
    frames_data : list of dict
        As returned by raw2frame_data — one dict per frame with a "frame" key.
    ego_x_speeds : list of float
        Per-frame x-axis speed values aligned to frames_data (length == len(frames_data)).
    ego_z_speeds : list of float
        Per-frame z-axis speed values aligned to frames_data (length == len(frames_data)).
    video_id : str
    fps : int
        Output video frame rate.
    stop_threshold : float
        Frames whose |speed| is below this value are shaded as "stopped".
    chart_width : float
        Matplotlib figure width (inches) for the speed chart.
    chart_height : float
        Matplotlib figure height (inches) for the speed chart.
    output_dir : Path-like or None
        Destination directory; defaults to pipeline_output/ego_speed_vis/.
    """
    import pathlib

    num_frames = len(frames_data)
    
    ego_speeds = [np.sqrt(x**2 + z**2) for x, z in zip(ego_x_speeds, ego_z_speeds)]
    # guard against length mismatch
    ego_speeds = list(ego_speeds)
    if len(ego_speeds) < num_frames:
        ego_speeds += [ego_speeds[-1]] * (num_frames - len(ego_speeds))
    ego_speeds = ego_speeds[:num_frames]

    # ── determine canvas dimensions ───────────────────────────────────────────
    first_img = utils.load_frame(frames_data[0]["frame"])
    frame_h, frame_w = first_img.shape[:2]

    sample_chart = utils.create_timeline_line_chart_img(
        ego_speeds, 0, [], [], "Ego Speed (m/s)",
        width=chart_width, height=chart_height,
        stopped_mask=stopped_mask
    )
    chart_h = int(sample_chart.shape[0] * frame_w / sample_chart.shape[1])
    total_h = frame_h + chart_h

    # ── open VideoWriter ──────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = pathlib.Path(config.get_output_path("pipeline_output")) / "ego_speed_vis"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}_ego_speed.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, total_h))

    # ── write every frame ─────────────────────────────────────────────────────
    for fi, frame_data in enumerate(frames_data):
        frame_img = utils.load_frame(frame_data["frame"])
        if frame_img.shape[0] != frame_h or frame_img.shape[1] != frame_w:
            frame_img = cv2.resize(frame_img, (frame_w, frame_h))

        chart_img = utils.create_timeline_line_chart_img(
            ego_speeds, fi, [], [],
            f"Ego Speed (m/s) — frame {fi}/{num_frames - 1}  |  speed = {ego_speeds[fi]:.4f} m/s",
            width=chart_width, height=chart_height,
            stopped_mask=stopped_mask
        )
        chart_img = cv2.resize(chart_img, (frame_w, chart_h))

        composite = np.vstack([frame_img, chart_img])
        writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved ego speed visualization ({num_frames} frames) → {out_path}")
    return out_path

def visualize_mask(frames_data, ego_speeds, ego_rotations, stopped_mask, turning_mask, video_id,
                   fps=5, chart_width=10, chart_height=3, output_dir=None):
    """
    Produce an MP4 with three stacked panels per frame:
      - top    : original video frame
      - middle : ego-speed line chart; orange shading where mask == 1 (stopped)
      - bottom : ego yaw-rate line chart; orange shading where mask == 2 (turning)

    mask values: 0 = driving, 1 = stopped, 2 = turning  (as returned by merge_segs)

    Output: *output_dir*/<video_id>_mask_vis.mp4
    """
    import pathlib

    num_frames = len(frames_data)

    # ── align signal lengths to num_frames ────────────────────────────────────
    def _align(seq):
        seq = list(seq)
        if len(seq) < num_frames:
            seq += [seq[-1]] * (num_frames - len(seq))
        return seq[:num_frames]

    ego_speeds    = _align(ego_speeds)
    ego_rotations = _align(ego_rotations)
    stopped_mask  = _align(stopped_mask)
    turning_mask  = _align(turning_mask)

    # ── determine canvas dimensions ───────────────────────────────────────────
    first_img = utils.load_frame(frames_data[0]["frame"])
    frame_h, frame_w = first_img.shape[:2]

    sample_speed_chart = utils.create_timeline_line_chart_img(
        ego_speeds, 0, [], [], "Ego Speed (m/s)",
        width=chart_width, height=chart_height,
        stopped_mask=stopped_mask,
    )
    speed_chart_h = int(sample_speed_chart.shape[0] * frame_w / sample_speed_chart.shape[1])

    sample_rot_chart = utils.create_timeline_line_chart_img(
        ego_rotations, 0, [], [], "Ego Yaw-Rate (rad/s)",
        width=chart_width, height=chart_height,
        stopped_mask=turning_mask,
    )
    rot_chart_h = int(sample_rot_chart.shape[0] * frame_w / sample_rot_chart.shape[1])

    total_h = frame_h + speed_chart_h + rot_chart_h

    # ── open VideoWriter ──────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = pathlib.Path(config.get_output_path("pipeline_output")) / "mask_vis"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}_mask_vis.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, total_h))

    # state label config: mask value → (text, BGR colour)
    _STATE_LABEL = {
        0: ("Normal",  (100, 200, 100)),   # green
        1: ("Stopped", (0,   100, 255)),   # orange-red
        2: ("Turning", (255, 100,   0)),   # blue
    }

    # ── write every frame ─────────────────────────────────────────────────────
    for fi, frame_data in enumerate(frames_data):
        frame_img = utils.load_frame(frame_data["frame"])
        if frame_img.shape[0] != frame_h or frame_img.shape[1] != frame_w:
            frame_img = cv2.resize(frame_img, (frame_w, frame_h))

        # overlay state label top-left (work in BGR for cv2)
        frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
        if stopped_mask[fi]:
            mask_val = 1
        elif turning_mask[fi]:
            mask_val = 2
        else:            
            mask_val = 0
        state_text, label_color = _STATE_LABEL.get(mask_val, ("Unknown", (128, 128, 128)))
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.8, frame_w / 800)
        thickness  = max(2, int(font_scale * 2))
        pad        = int(10 * font_scale)
        (tw, th), baseline = cv2.getTextSize(state_text, font, font_scale, thickness)
        # semi-transparent dark background behind text
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay,
                      (pad, pad),
                      (pad + tw + pad, pad + th + baseline + pad),
                      (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, frame_bgr)
        cv2.putText(frame_bgr, state_text,
                    (pad * 2, pad + th),
                    font, font_scale, label_color, thickness, cv2.LINE_AA)
        frame_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        speed_chart = utils.create_timeline_line_chart_img(
            ego_speeds, fi, [], [],
            f"Ego Speed (m/s) — frame {fi}/{num_frames - 1}  |  speed = {ego_speeds[fi]:.4f} m/s",
            width=chart_width, height=chart_height,
            stopped_mask=stopped_mask,
        )
        speed_chart = cv2.resize(speed_chart, (frame_w, speed_chart_h))

        rot_chart = utils.create_timeline_line_chart_img(
            ego_rotations, fi, [], [],
            f"Ego Yaw-Rate (rad/s) — frame {fi}/{num_frames - 1}  |  yaw = {ego_rotations[fi]:.4f} rad/s",
            width=chart_width, height=chart_height,
            stopped_mask=turning_mask,
        )
        rot_chart = cv2.resize(rot_chart, (frame_w, rot_chart_h))

        composite = np.vstack([frame_img, speed_chart, rot_chart])
        writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved mask visualization ({num_frames} frames) → {out_path}")
    return out_path

def visualize_ego_rotation(frames_data, ego_rotations, video_id,
                           turn_mask=None,
                           fps=5, chart_width=10, chart_height=3,
                           output_dir=None):
    """
    Produce an MP4 where every frame shows:
      - top panel    : original video frame
      - bottom panel : ego yaw-rate line chart with a red dot at the current
                       frame and orange-shaded regions for turning frames.

    Output: *output_dir*/<video_id>_ego_rotation.mp4

    Parameters
    ----------
    frames_data : list of dict
        As returned by raw2frame_data — one dict per frame with a "frame" key.
    ego_rotations : list of float
        Per-frame smoothed yaw-rate values aligned to frames_data.
    video_id : str
    turn_mask : list of bool or None
        Per-frame turn flags as returned by percept2turn_mask.  Turning frames
        are shaded orange in the chart.  Pass None to disable.
    fps : int
        Output video frame rate.
    chart_width : float
        Matplotlib figure width (inches) for the rotation chart.
    chart_height : float
        Matplotlib figure height (inches) for the rotation chart.
    output_dir : Path-like or None
        Destination directory; defaults to pipeline_output/ego_rotation_vis/.
    """
    import pathlib

    num_frames = len(frames_data)

    # guard against length mismatch
    ego_rotations = list(ego_rotations)
    if len(ego_rotations) < num_frames:
        ego_rotations += [ego_rotations[-1]] * (num_frames - len(ego_rotations))
    ego_rotations = ego_rotations[:num_frames]

    # ── determine canvas dimensions ───────────────────────────────────────────
    first_img = utils.load_frame(frames_data[0]["frame"])
    frame_h, frame_w = first_img.shape[:2]

    sample_chart = utils.create_timeline_line_chart_img(
        ego_rotations, 0, [], [], "Ego Yaw-Rate (rad/s)",
        width=chart_width, height=chart_height,
        stopped_mask=turn_mask
    )
    chart_h = int(sample_chart.shape[0] * frame_w / sample_chart.shape[1])
    total_h = frame_h + chart_h

    # ── open VideoWriter ──────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = pathlib.Path(config.get_output_path("pipeline_output")) / "ego_rotation_vis"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}_ego_rotation.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, total_h))

    # ── write every frame ─────────────────────────────────────────────────────
    for fi, frame_data in enumerate(frames_data):
        frame_img = utils.load_frame(frame_data["frame"])
        if frame_img.shape[0] != frame_h or frame_img.shape[1] != frame_w:
            frame_img = cv2.resize(frame_img, (frame_w, frame_h))

        chart_img = utils.create_timeline_line_chart_img(
            ego_rotations, fi, [], [],
            f"Ego Yaw-Rate (rad/s) — frame {fi}/{num_frames - 1}  |  yaw = {ego_rotations[fi]:.4f} rad/s",
            width=chart_width, height=chart_height,
            stopped_mask=turn_mask
        )
        chart_img = cv2.resize(chart_img, (frame_w, chart_h))

        composite = np.vstack([frame_img, chart_img])
        writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved ego rotation visualization ({num_frames} frames) → {out_path}")
    return out_path

