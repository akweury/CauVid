"""
Expression system for the rule discovery language.
"""
import numpy as np
from typing import Union, List
from abc import ABC, abstractmethod
from src.logic.concepts import Property
from src.logic.evaluations import EvaluationContext

class Expression(ABC):
    """Base class for all expressions"""
    
    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> Union[float, np.ndarray]:
        """Evaluate this expression"""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """String representation"""
        pass

class PropertyExpression(Expression):
    """Expression wrapping a property"""
    
    def __init__(self, property: Property):
        self.property = property
    
    def evaluate(self, context: EvaluationContext) -> Union[float, np.ndarray]:
        return self.property.evaluate(context)
    
    def __str__(self) -> str:
        return str(self.property)

class ConstantExpression(Expression):
    """Constant value expression"""
    
    def __init__(self, value: Union[float, np.ndarray]):
        self.value = value
    
    def evaluate(self, context: EvaluationContext) -> Union[float, np.ndarray]:
        return self.value
    
    def __str__(self) -> str:
        if isinstance(self.value, np.ndarray):
            return f"[{', '.join(map(str, self.value))}]"
        return str(self.value)

class BinaryOperation(Expression):
    """Binary operation between two expressions"""
    
    def __init__(self, left: Expression, operator: str, right: Expression):
        self.left = left
        self.operator = operator
        self.right = right
    
    def evaluate(self, context: EvaluationContext) -> Union[float, np.ndarray]:
        left_val = self.left.evaluate(context)
        right_val = self.right.evaluate(context)
        
        if self.operator == '+':
            return left_val + right_val
        elif self.operator == '-':
            return left_val - right_val
        elif self.operator == '*':
            return left_val * right_val
        elif self.operator == '/':
            return left_val / right_val
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
    
    def __str__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"

class FunctionCall(Expression):
    """Function call expression"""
    
    def __init__(self, function_name: str, arguments: List[Expression]):
        self.function_name = function_name
        self.arguments = arguments
    
    def evaluate(self, context: EvaluationContext) -> Union[float, np.ndarray]:
        arg_values = [arg.evaluate(context) for arg in self.arguments]
        
        if self.function_name == 'distance':
            # Distance between two positions
            if len(arg_values) == 2:
                pos1, pos2 = arg_values
                return np.linalg.norm(pos1 - pos2)
        elif self.function_name == 'magnitude':
            # Magnitude of a vector
            if len(arg_values) == 1:
                return np.linalg.norm(arg_values[0])
        elif self.function_name == 'max':
            return np.maximum(*arg_values)
        elif self.function_name == 'min':
            return np.minimum(*arg_values)
        
        raise ValueError(f"Unknown function: {self.function_name}")
    
    def __str__(self) -> str:
        args_str = ', '.join(str(arg) for arg in self.arguments)
        return f"{self.function_name}({args_str})"

# Helper functions for building expressions
def add(left: Expression, right: Expression) -> BinaryOperation:
    """Addition expression"""
    return BinaryOperation(left, '+', right)

def multiply(left: Expression, right: Expression) -> BinaryOperation:
    """Multiplication expression"""
    return BinaryOperation(left, '*', right)

def distance(obj1_pos: Expression, obj2_pos: Expression) -> FunctionCall:
    """Distance function"""
    return FunctionCall('distance', [obj1_pos, obj2_pos])

def constant(value: Union[float, np.ndarray]) -> ConstantExpression:
    """Constant expression"""
    return ConstantExpression(value)