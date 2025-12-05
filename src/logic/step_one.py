"""
Example usage of the rule discovery language.
"""
from concepts import *
from expressions import *
from rules import *
from evaluations import EvaluationContext

def create_basic_motion_rule() -> Rule:
    """Create basic physics motion rule: position_next = position_now + velocity_now"""
    
    # position(self).next() = position(self).now() + velocity(self).now()
    target = position_next()  # self is default
    expression = add(
        PropertyExpression(position_now()),
        PropertyExpression(velo_now())
    )
    action = assign(target, expression)
    
    return always(action, "basic_motion")

def create_collision_avoidance_rule() -> Rule:
    """Create collision avoidance rule"""
    
    # IF distance(self.position.now, nearest().position.now) < 30 THEN self.velocity.next = self.velocity.now * 0.5
    condition = less_than(
        distance(
            PropertyExpression(position_now()),
            PropertyExpression(position_now(Nearest()))
        ),
        constant(30.0)
    )
    
    action = assign(
        velo_next(),
        multiply(
            PropertyExpression(velo_now()),
            constant(0.5)
        )
    )
    
    return if_then(condition, action, "collision_avoidance")

def demo_rule_execution():
    """Demonstrate rule execution"""
    
    # Create context
    context = EvaluationContext(current_frame=5, current_object="obj_1")
    
    # Add some object states
    context.add_object_state(5, "obj_1", ObjectState("obj_1", 5, 100, 100, 5, 0))
    context.add_object_state(5, "obj_2", ObjectState("obj_2", 5, 120, 100, -3, 0))
    
    # Create and apply rules
    motion_rule = create_basic_motion_rule()
    avoidance_rule = create_collision_avoidance_rule()
    
    print("Basic motion rule:", motion_rule)
    print("Collision avoidance rule:", avoidance_rule)
    
    # Apply rules
    if motion_rule.can_apply(context):
        print("Applying motion rule...")
        motion_rule.apply(context)
    
    if avoidance_rule.can_apply(context):
        print("Applying avoidance rule...")
        avoidance_rule.apply(context)
    
    # Check predicted state
    if 6 in context.predicted_states and "obj_1" in context.predicted_states[6]:
        predicted = context.predicted_states[6]["obj_1"]
        print(f"Predicted position: ({predicted.position_x}, {predicted.position_y})")
        print(f"Predicted velocity: ({predicted.velocity_x}, {predicted.velocity_y})")




if __name__ == "__main__":
    demo_rule_execution()





