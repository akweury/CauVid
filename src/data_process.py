
from typing import List, Dict

def window_smooth(scenes: List[Dict], window_size: int = 3) -> List[Dict]:
    """
    Apply a simple moving average smoothing over the scenes based on object positions.
    Args:
        scenes (List[Dict]): List of scene dictionaries.
        window_size (int): Size of the moving window.
    Returns:
        List[Dict]: Smoothed list of scene dictionaries.
    """
    smoothed_scenes = []
    half_window = window_size // 2
    num_scenes = len(scenes)

    for i in range(num_scenes):
        smoothed_scene = {}
        for obj_id in scenes[i]:
            x_sum, y_sum, count = 0, 0, 0
            for j in range(max(0, i - half_window), min(num_scenes, i + half_window + 1)):
                if obj_id in scenes[j]:
                    x_sum += scenes[j][obj_id]['position_x']
                    y_sum += scenes[j][obj_id]['position_y']
                    count += 1
            if count > 0:
                smoothed_scene[obj_id] = {
                    'position_x': x_sum / count,
                    'position_y': y_sum / count,
                    'color_r': scenes[i][obj_id].get('color_r'),
                    'color_g': scenes[i][obj_id].get('color_g'),        
                    'color_b': scenes[i][obj_id].get('color_b'),
                }
        smoothed_scenes.append(smoothed_scene)
    
    return smoothed_scenes