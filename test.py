from enum import Enum, auto

class AutoStrEnum(str, Enum):
    """Enum where members are strings and can be used directly without .value"""
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()  # Or just `name` for case-sensitive

class Color(AutoStrEnum):
    RED = auto()  # Behaves like a string
    GREEN = auto()
    BLUE = auto()

    def __str__(self):
        return self._name_

# Usage
print(Color.RED)  # Directly prints "red" (no .value needed)
print(Color.RED.upper())  # Can use string methods: prints "RED"