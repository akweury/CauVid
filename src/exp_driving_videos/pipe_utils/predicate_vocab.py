



import numpy as np 

class Config:
    
    # ego speed threshold
    SPEED_ESP = 0.05

    # yaw thresholds based on integrated yaw or mean yaw-rate
    TURN_WEAK = 0.1
    TURN_STRONG = 3
    
    # distance thresholds
    CLOSE_DIST=10.0 # meters
    # relative distance change
    REL_EPS = 0.05
    
    
    
    
    
    
def ego_speed_change(segment, cfg=Config()):
    """
    segment: a dict
    """
    
    slope = segment["slope"]
    if slope > cfg.SPEED_ESP:
        return "speeding_up"
    elif slope < -cfg.SPEED_ESP:
        return "slowing_down"
    else:
        return "constant_speed"
    

def ego_turn(segment, cfg=Config()):
    """
    segment: a dict
    """
    
    yaw = abs(segment["yaw_mean"])
    
    if yaw < cfg.TURN_WEAK:
        return "straight"
    elif yaw < cfg.TURN_STRONG:
        return "weak_turn"
    else:
        return "strong_turn"
    
    
# -----------------------------
# Relative geometry predicates
# -----------------------------

def front(obj_pos):
    """
    obj_pos: (x,z)
    z-axis = forward, x-axis = left-right
    """
    
    res = (obj_pos[1]) > 0
    return res

def close(obj_pos, cfg=Config()):
    """
    obj_pos: (x,z)
    z-axis = forward, x-axis = left-right
    """
    dist = np.sqrt((obj_pos[0])**2 + (obj_pos[1])**2)
    res = dist < cfg.CLOSE_DIST
    return res


# ------------------------------
# Relative Motion Predicates
# ------------------------------

def relative_motion(dz_series, cfg=Config()):
    """
    dz_series: a time series of relative distance change in the z-axis (forward-backward)
    """
    delta = dz_series[-1] - dz_series[0]
    
    if delta > cfg.REL_EPS:
        return "moving_away (z-axis)"
    elif delta < -cfg.REL_EPS:
        return "approaching (z-axis)"
    else:
        return "stable (z-axis)"
    
    
# ------------------------------
# Full Predicate Extraction
# ------------------------------

def extract_predicates(segment, objects, cfg=Config()):
    """
    segment: a dict with features fo ego """
    predicates = {}
    
    
    # ego predicates 
    predicates["ego_speed_change"] = ego_speed_change(segment,cfg)
    predicates["ego_turn"] = ego_turn(segment,cfg)
    
    # object related predicates 
    obj_preds = []
    for obj in objects:
        p = {}
        if front(obj["position"]):
            p["front"] = True
        
        if close(obj["position"], cfg):
            p["close"] = True
        
        p["relative_motion"] = relative_motion(obj["dz_series"], cfg)
        
        obj_preds.append(p)
    predicates["objects"] = obj_preds
    
    return predicates