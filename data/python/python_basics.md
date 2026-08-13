# Python Basics for Data Engineering

## 1. Python Overview

Python is a high-level programming language widely used for data engineering, automation, APIs, analytics, and machine learning.

Python emphasizes readability and provides a large standard library and ecosystem.

## 2. Variables and Data Types

Common built-in types include:

- `int`
- `float`
- `str`
- `bool`
- `list`
- `tuple`
- `set`
- `dict`
- `None`

Example:

```python
name = "customer"
count = 100
active = True
metadata = {"source": "kafka", "records": count}
```

## 3. Lists, Tuples, Sets, and Dictionaries

Lists are ordered and mutable.

```python
items = ["spark", "kafka", "python"]
items.append("sql")
```

Tuples are ordered and immutable.

```python
coordinates = (10, 20)
```

Sets store unique values.

```python
technologies = {"spark", "kafka", "spark"}
```

Dictionaries store key-value pairs.

```python
pipeline = {
    "name": "customer_daily_etl",
    "status": "FAILED"
}
```

## 4. Control Flow

```python
if status == "FAILED":
    print("Investigate pipeline")

for pipeline in pipelines:
    print(pipeline["name"])
```

## 5. Functions

Functions encapsulate reusable logic.

```python
def calculate_success_rate(successful, total):
    if total == 0:
        return 0
    return successful / total
```

Prefer small functions with clear inputs and outputs.

## 6. Exceptions

Use exceptions to handle expected failure conditions.

```python
try:
    value = int(raw_value)
except ValueError:
    value = 0
```

Do not catch every exception without logging or understanding the failure.

## 7. File Handling

```python
from pathlib import Path

path = Path("data/events.json")

text = path.read_text()
```

Use context managers when working with file handles:

```python
with open("data/events.txt") as file:
    content = file.read()
```

## 8. Modules and Packages

A module is a Python file that contains reusable code. Packages organize related modules.

A typical data engineering project may separate:

```text
loaders/
transformations/
validation/
utils/
```

## 9. List Comprehensions

```python
names = ["spark", "kafka", "python"]

upper_names = [name.upper() for name in names]
```

Use comprehensions when they improve readability; avoid overly complex expressions.

## 10. Classes

Classes combine data and behavior.

```python
class PipelineRun:
    def __init__(self, pipeline_id, status):
        self.pipeline_id = pipeline_id
        self.status = status

    def is_failed(self):
        return self.status == "FAILED"
```

## 11. Type Hints

Type hints improve readability and tooling.

```python
def get_pipeline_status(pipeline_id: str) -> str:
    return "SUCCESS"
```

## 12. JSON

JSON is common in APIs and metadata pipelines.

```python
import json

payload = {"pipeline": "customer_daily_etl"}

serialized = json.dumps(payload)
restored = json.loads(serialized)
```

## 13. Useful Data Engineering Libraries

Common libraries include:

- `pandas`
- `pyspark`
- `requests`
- `boto3`
- `pydantic`
- `fastapi`
- `pytest`

## 14. Practice Questions

1. What is the difference between a list and tuple?
2. When should a dictionary be used?
3. Why use a context manager for files?
4. What is the difference between `==` and `is`?
5. How do exceptions propagate?
6. Why use type hints?
7. What makes a Python function reusable?
