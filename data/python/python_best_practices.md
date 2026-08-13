# Python Best Practices for Data Engineering

## 1. Write Small, Focused Functions

Prefer:

```python
def load_config():
    ...

def validate_config(config):
    ...

def create_client(config):
    ...
```

over a single large function that performs every operation.

Small functions are easier to test and debug.

## 2. Use Meaningful Names

Prefer:

```python
pipeline_id = "customer_daily_etl"
retry_count = 3
```

over:

```python
x = "customer_daily_etl"
n = 3
```

Names should communicate intent.

## 3. Avoid Hard-Coded Configuration

Do not embed credentials, URLs, or environment-specific settings directly in source code.

Use environment variables or a configuration system.

```python
import os

api_url = os.environ["API_URL"]
```

Never commit secrets to Git.

## 4. Use Logging Instead of Print

Prefer structured logging:

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Starting pipeline", extra={"pipeline_id": pipeline_id})
```

Logging is essential for production troubleshooting.

## 5. Validate Inputs

Validate external data before processing.

For example:

```python
required_columns = {"customer_id", "amount"}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")
```

## 6. Handle Errors Intentionally

Catch exceptions where you can actually handle them.

Bad:

```python
try:
    process()
except Exception:
    pass
```

Better:

```python
try:
    process()
except ConnectionError as exc:
    logger.exception("Connection failed")
    raise
```

## 7. Make Pipelines Idempotent

An idempotent pipeline can be safely retried without creating incorrect duplicate results.

For example, writing a partition using a deterministic partition key can be safer than blindly appending the same data on every retry.

## 8. Use Configuration Objects

For larger applications, use typed configuration.

```python
from pydantic import BaseModel

class Settings(BaseModel):
    environment: str
    vector_db_url: str
    top_k: int = 5
```

## 9. Separate Business Logic from Infrastructure

A useful project structure is:

```text
src/
├── config/
├── ingestion/
├── transformations/
├── storage/
├── services/
├── api/
└── tests/
```

Business logic should not depend unnecessarily on one specific cloud provider or database.

## 10. Write Tests

Test important transformations and failure conditions.

```python
def test_success_rate():
    assert calculate_success_rate(8, 10) == 0.8
```

Use fixtures for reusable test data.

## 11. Use Dependency Pinning

Pin compatible dependency versions for reproducible environments.

Maintain:

```text
requirements.txt
```

or use a modern dependency manager such as Poetry or uv.

## 12. Follow Formatting and Linting

Useful tools include:

- Black or Ruff formatter
- Ruff linter
- mypy or another type checker

Automated formatting reduces style discussions during code review.

## 13. Avoid Premature Optimization

First measure.

Then optimize the actual bottleneck.

For data workloads, investigate:

- I/O
- network transfer
- serialization
- database queries
- Spark shuffles
- memory usage

before changing code based only on assumptions.

## 14. Be Careful With Large Data

Avoid loading huge datasets into Python memory unnecessarily.

Prefer streaming, batching, distributed processing, or database-side filtering where appropriate.

For example, filter before collecting data:

```python
df.filter(df.status == "FAILED").select("pipeline_id")
```

## 15. Secure Data and Credentials

Never commit:

```text
.env
API keys
cloud credentials
database passwords
private certificates
```

Use secret managers or environment-based configuration.

## 16. Design for Observability

Record useful information such as:

- request ID
- pipeline ID
- execution ID
- duration
- input/output counts
- error type
- retry count

Observability is especially important for AI and data pipelines.

## 17. Code Review Checklist

Before pushing code:

```text
[ ] No secrets committed
[ ] Functions have clear responsibilities
[ ] Errors are handled intentionally
[ ] Logging is present
[ ] Tests cover important paths
[ ] Inputs are validated
[ ] Configuration is externalized
[ ] Code is formatted
[ ] Dependencies are reproducible
[ ] README/documentation updated
```

## 18. Applying These Practices to RAG

For a RAG application, use the same principles.

Separate:

```text
Document Loader
      ↓
Cleaner
      ↓
Chunker
      ↓
Embedding Service
      ↓
Vector Store
      ↓
Retriever
      ↓
Prompt Builder
      ↓
LLM Service
```

This makes the RAG system easier to test and replace.

For example, you should be able to replace FAISS with Qdrant without rewriting the entire application.

## 19. Practical Exercise

Refactor a simple RAG script into:

```text
src/
├── config.py
├── loaders.py
├── chunker.py
├── embeddings.py
├── vector_store.py
├── retriever.py
├── generator.py
└── pipeline.py
```

Then add:

```text
tests/
├── test_chunker.py
├── test_retriever.py
└── test_pipeline.py
```

This exercise connects Python engineering practices directly to your RAG learning path.
