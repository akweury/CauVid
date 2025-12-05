"""
Rule system for the motion discovery language.
"""
from typing import Optional, List
from abc import ABC, abstractmethod
from src.logic.expressions import Expression, PropertyExpression
from src.logic.concepts import Property
from src.logic.evaluations import EvaluationContext

class Condition(ABC):
    """Base class for rule conditions"""
    
    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> bool:
        """Evaluate condition to true/false"""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass

class ComparisonCondition(Condition):
    """Comparison between two expressions"""
    
    def __init__(self, left: Expression, operator: str, right: Expression):
        self.left = left
        self.operator = operator
        self.right = right
    
    def evaluate(self, context: EvaluationContext) -> bool:
        left_val = self.left.evaluate(context)
        right_val = self.right.evaluate(context)
        
        if self.operator == '<':
            return left_val < right_val
        elif self.operator == '>':
            return left_val > right_val
        elif self.operator == '<=':
            return left_val <= right_val
        elif self.operator == '>=':
            return left_val >= right_val
        elif self.operator == '==':
            return abs(left_val - right_val) < 1e-6  # Float comparison
        else:
            raise ValueError(f"Unknown comparison operator: {self.operator}")
    
    def __str__(self) -> str:
        return f"{self.left} {self.operator} {self.right}"

class Action(ABC):
    """Base class for rule actions"""
    
    @abstractmethod
    def execute(self, context: EvaluationContext) -> None:
        """Execute this action"""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass

class Assignment(Action):
    """Assignment action: property = expression"""
    
    def __init__(self, target: Property, expression: Expression):
        self.target = target
        self.expression = expression
    
    def execute(self, context: EvaluationContext) -> None:
        """Execute assignment"""
        value = self.expression.evaluate(context)
        context.set_property_value(self.target, value)
    
    def __str__(self) -> str:
        return f"{self.target} = {self.expression}"

class Rule:
    """A complete rule with optional condition and action"""
    
    def __init__(self, action: Action, condition: Optional[Condition] = None, 
                 name: str = "unnamed_rule"):
        self.condition = condition
        self.action = action
        self.name = name
    
    def can_apply(self, context: EvaluationContext) -> bool:
        """Check if this rule can be applied"""
        if self.condition is None:
            return True
        return self.condition.evaluate(context)
    
    def apply(self, context: EvaluationContext) -> bool:
        """Apply this rule if condition is met"""
        if self.can_apply(context):
            self.action.execute(context)
            return True
        return False
    
    def __str__(self) -> str:
        if self.condition:
            return f"IF {self.condition} THEN {self.action}"
        else:
            return str(self.action)

# Helper functions for building rules
def if_then(condition: Condition, action: Action, name: str = "conditional_rule") -> Rule:
    """Create conditional rule"""
    return Rule(action, condition, name)

def always(action: Action, name: str = "unconditional_rule") -> Rule:
    """Create unconditional rule"""
    return Rule(action, None, name)

def less_than(left: Expression, right: Expression) -> ComparisonCondition:
    """Less than condition"""
    return ComparisonCondition(left, '<', right)

def greater_than(left: Expression, right: Expression) -> ComparisonCondition:
    """Greater than condition"""
    return ComparisonCondition(left, '>', right)

def assign(target: Property, expression: Expression) -> Assignment:
    """Assignment action"""
    return Assignment(target, expression)