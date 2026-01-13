from typing import Dict, List



def scene_mid_x(scene:Dict)->float:
    xs = [obj["position_x"] for obj in scene.values()]
    return sum(xs)/len(xs)

def left(obj:Dict, scene:Dict)->bool:
    mid_x = scene_mid_x(scene)
    return obj["position_x"] < mid_x

def right(obj:Dict, scene:Dict)->bool:
    mid_x = scene_mid_x(scene)
    return obj["position_x"] > mid_x

def vy_positive(obj:Dict, eps:float=1e-6)->bool:
    return obj["vy"]>eps 


def vy_negative(obj:Dict, eps:float=1e-6)->bool:
    return obj["vy"]<-eps

def moves_up(obj_now:Dict, obj_next:Dict, eps:float=1e-6)->bool:
    return (obj_next["position_y"] - obj_now["position_y"]) > eps

def moves_down(obj_now:Dict, obj_next:Dict, eps:float=1e-6)->bool:
    return (obj_now["position_y"] - obj_next["position_y"]) > eps

