"""
Simple calculator tool - function-based example.

This demonstrates a simple, stateless tool that doesn't need
the complexity of a class.
"""


async def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    
    Supports basic arithmetic operations: +, -, *, /, **, (), and basic math functions.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
    
    Returns:
        Result of the calculation as a string
    """
    try:
        # Safety: only allow specific characters
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters. Only numbers and operators (+, -, *, /, **, ()) are allowed."
        
        # Evaluate the expression
        result = eval(expression)
        
        return f"Result: {result}"
    
    except ZeroDivisionError:
        return "Error: Division by zero"
    except SyntaxError:
        return "Error: Invalid expression syntax"
    except Exception as e:
        return f"Error: {str(e)}"

