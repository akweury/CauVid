



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
    
def ego_speed_change(ego, eps=1e-2, ratio_th=0.7):
    vx = ego["speed_x"]
    vz = ego["speed_z"]

    speed = np.sqrt(vx**2 + vz**2)
    diff = np.diff(speed)
    
    valid = np.abs(diff) > eps
    if valid.sum() == 0:
        return "constant_speed", 0.0

    diff = diff[valid]

    
    pos_ratio = (diff > 0).mean()
    neg_ratio = (diff < 0).mean()

    
    if pos_ratio > ratio_th:
        label = "speeding_up"
    elif neg_ratio > ratio_th:
        label = "slowing_down"
    else:
        label = "constant_speed"
        
    confidence = abs(pos_ratio - neg_ratio)

    return label, confidence

def ego_is_moving(ego, eps=0.5, ratio_th = 0.7):
    stop_mask = ego['stop_mask'] 
    pos_ratio = (stop_mask==True).mean()
    neg_ratio = (stop_mask==False).mean()
    
    if pos_ratio > ratio_th:
        label = "stopped"
    elif neg_ratio > ratio_th:
        label = "moving"
    else:
        label = "uncertain"
    
    confidence = abs(pos_ratio - neg_ratio)
    return label, confidence

    
def ego_is_turning(ego, eps=0.1, ratio_th=0.7):
    turn_mask = ego['turn_mask']
    pos_ratio = (turn_mask==True).mean()
    neg_ratio = (turn_mask==False).mean()
    if pos_ratio > ratio_th:
        label = "turning"
    elif neg_ratio > ratio_th:
        label = "not_turning"
    else:
        label = "uncertain"
    confidence = abs(pos_ratio - neg_ratio)
    return label, confidence
    
def compute_focus_score(obj):
    """ 
    heuristic scoring function to select interesting objects to focus on for predicate extraction.
    """
    proximity = (1.0/(np.array(obj["rank"]) + 1e-6)).mean()
    
    vz_rel = np.mean(obj["speed"][:,1])  # relative speed in z-axis
    approaching = max(0, -vz_rel)  # only consider approaching speed
    
    vx_rel = np.mean(obj["speed"][:,0])  # relative speed in x-axis
    rel_speed = obj['vx']['label'] != "stable"
    
    lateral= abs(vx_rel)
    
    moving = obj["vz"]['label'] == "approaching" 
    
    weights = np.array([1.1, 1.0, 1.0])  # weights for each factor
    factors = np.array([proximity, lateral, moving])
    focus_score = np.dot(weights, factors)
    
    return focus_score
    
def select_focus_objects(objects, top_k, speed_th=0.5):
    
    """ 
    three kinds of approaches:
    1. heuristic closest object
    2. heuristic score
    3. learnable attention model (e.g., train a small MLP to predict which objects 
    are most relevant for predicate extraction, 
    using some proxy labels or self-supervised objectives)
    """
    scores = [(compute_focus_score(obj), obj) for obj in objects]
    scores.sort(reverse=True, key=lambda x: x[0])
    top_objects = [obj for _, obj in scores[:top_k]]
    top_scores = [score for score, _ in scores[:top_k]]
    return top_objects, top_scores
  
def obj_is_moving(obj, ratio_th=0.7):
    vx = obj['vx']
    vz = obj['vz']
    pos_ratio = np.mean(obj['mask_o_vz_rel'])
    neg_ratio = 1 - pos_ratio
    
    if pos_ratio > ratio_th:
        label = "moving"
    elif neg_ratio > ratio_th:
        label = "stopped"
    else:
        label = "uncertain"
    
    confidence = abs(pos_ratio - neg_ratio)
    return label, confidence  
   
def obj_is_turning(obj, ratio_th=0.7):
    turn_mask = obj['mask_o_vx_rel']
    pos_ratio = np.mean(turn_mask)
    neg_ratio = 1 - pos_ratio
    
    if pos_ratio > ratio_th:
        label = "turning"
    elif neg_ratio > ratio_th:
        label = "not_turning"
    else:
        label = "uncertain"
    
    confidence = abs(pos_ratio - neg_ratio)
    return label, confidence
    
    
def obj_lateral_motion(obj, ratio_th=0.7):
    vx = obj['vx']
    label = vx['label']
    if label == "left":
        confidence = vx['left_ratio']
    elif label == "right":
        confidence = vx['right_ratio']
    else:
        confidence = 1 - max(vx['left_ratio'], vx['right_ratio'])
    return label, confidence
    
def obj_relative_motion(obj, ratio_th=0.7):
    vz = obj['vz']
    label = vz['label']
    
    if label =="approaching":
        confidence = vz['approach_ratio']
    elif label == "moving_away":
        confidence = vz['away_ratio']
    else:
        confidence = 1 - max(vz['approach_ratio'],vz["away_ratio"])
        
    return label, confidence
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

def relative_motion(obj):
    motion, confidence = "unknown", 0.0

    vz_rels = obj["speeds_rel"][:,1].mean()
    vz_mean = np.mean(vz_rels)
    raise NotImplementedError("Need to consider lateral motion as well")
    return motion, confidence
    
    
# ------------------------------
# Full Predicate Extraction
# ------------------------------

def extract_predicates(objects, ego, cfg=Config()):
    """
    every predicate is related to a specific time segment
    """
    
    predicates = {}
    # ego predicates 
    predicates["ego_speed_change"] = ego_speed_change(ego)
    predicates['ego_is_moving'] = ego_is_moving(ego)
    predicates["ego_is_turning"] = ego_is_turning(ego)
    
    # obj predicates 
    obj_predicates =[]
    for obj in objects:
        obj_predicates.append({
            "id": obj["id"],
            "rel_motion": obj_relative_motion(obj),
            "lateral_motion": obj_lateral_motion(obj),
            # "is_moving": obj_is_moving(obj),
            # "is_turning": obj_is_turning(obj),
            "rank": np.mean(obj["rank"]),
        })
    
    # focus on interesting objects
    focus_objs, focus_obj_scores = select_focus_objects(objects, top_k=3)
    focus_preds = []
    for obj, score in zip(focus_objs, focus_obj_scores):
        focus_preds.append({
            "id": obj["id"],
            "rel_motion": obj_relative_motion(obj),
            "lateral_motion": obj_lateral_motion(obj),
            # "is_moving": obj_is_moving(obj),
            # "is_turning": obj_is_turning(obj),
            "rank": np.mean(obj["rank"]),
            "score": score,
        })
    
    
    # global predicates 
    global_preds = {
        "any_approach": any(
            obj_relative_motion(o)[0] == "approaching" for o in objects
        ),
        "num_approaching": sum(
            obj_relative_motion(o)[0] == "approaching" for o in objects
        )
    }
    
    all_preds= {
        "ego": predicates,
        "objects": obj_predicates,
        "focus_objects": focus_preds,
        "global": global_preds
    }
    
    return all_preds