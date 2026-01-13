from typing import Dict, List


def scene_mid_x(frame_data: Dict) -> float:
    return frame_data['width']/2


def left(obj: Dict, scene: Dict, frame_data) -> bool:
    mid_x = scene_mid_x(frame_data)
    return obj["position_x"] < mid_x


def right(obj: Dict, scene: Dict, frame_data) -> bool:
    mid_x = scene_mid_x(frame_data)
    return obj["position_x"] > mid_x

def red(obj: Dict, scene: Dict, frame_data) -> bool:
    """Determine if object is red based on RGB color properties."""
    r = obj.get("color_r", 0)
    g = obj.get("color_g", 0)
    b = obj.get("color_b", 0)
    
    # Object is red if red component is dominant and above threshold
    return r > 100 and r > g and r > b

def blue(obj: Dict, scene: Dict, frame_data) -> bool:
    """Determine if object is blue based on RGB color properties."""
    r = obj.get("color_r", 0)
    g = obj.get("color_g", 0)
    b = obj.get("color_b", 0)
    
    # Object is blue if blue component is dominant and above threshold
    return b > 100 and b > r and b > g


def vy_positive(obj: Dict, eps: float = 1e-6) -> bool:
    return obj["vy"] > eps


def vy_negative(obj: Dict, eps: float = 1e-6) -> bool:
    return obj["vy"] < -eps


def moves_up(obj_now: Dict, obj_next: Dict, eps: float = 1e-6) -> bool:
    return (obj_now["position_y"] - obj_next["position_y"]) > eps


def moves_down(obj_now: Dict, obj_next: Dict, eps: float = 1e-6) -> bool:
    return (obj_next["position_y"] - obj_now["position_y"]) > eps
