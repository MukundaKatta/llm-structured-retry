# llm-structured-retry

[![PyPI](https://img.shields.io/pypi/v/llm-structured-retry.svg)](https://pypi.org/project/llm-structured-retry/)
[![Python](https://img.shields.io/pypi/pyversions/llm-structured-retry.svg)](https://pypi.org/project/llm-structured-retry/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Retry LLM calls on validation failure by feeding the error back to the model.**

When an LLM returns output that fails validation (bad JSON, missing field,
wrong shape), this library injects the validation error as a follow-up user
message and asks the model to fix its own output — up to N attempts total.
Unlike local JSON-repair, the model sees what went wrong and corrects it.

Sync and async. Zero runtime dependencies. Provider-agnostic — you supply the
function that calls your LLM (OpenAI, Anthropic, anything that returns a string).

## Install

```bash
pip install llm-structured-retry
```

## Use

Your `llm_fn` takes the message list and returns the model's text reply. Your
`validate_fn` raises `ValidationError` when the reply is unacceptable.

```python
import json
from llm_structured_retry import StructuredRetry, ValidationError


def validate(content: str) -> None:
    try:
        json.loads(content)
    except Exception as exc:
        raise ValidationError(f"not valid JSON: {exc}") from exc


def call_llm(messages, **kwargs) -> str:
    # Call your provider here and return the assistant's text.
    return my_client.chat(messages, **kwargs)


retry = StructuredRetry(max_retries=3, validate_fn=validate)

result = retry.retry_call(
    call_llm,
    [{"role": "user", "content": "Return a JSON object with a 'name' field."}],
)

result.content     # the validated reply
result.succeeded   # True
result.attempts    # how many calls it took (1 = first try)
result.errors      # one entry per failed attempt
```

If every attempt fails, `MaxRetriesExceededError` is raised:

```python
from llm_structured_retry import MaxRetriesExceededError

try:
    retry.retry_call(call_llm, messages)
except MaxRetriesExceededError as e:
    print(e.attempts)     # total attempts made
    print(e.last_error)   # message from the final validation failure
```

## How the retry conversation is built

On each failed attempt the assistant's reply and a follow-up user message are
appended to a **copy** of your messages — your original list is never mutated:

```
[ ...your messages... ]
{ "role": "assistant", "content": "<the invalid reply>" }
{ "role": "user",      "content": "Your last response had an error: <msg>. Please try again." }
```

Customize the follow-up with `error_template` (must contain `{error}`):

```python
retry = StructuredRetry(
    max_retries=3,
    validate_fn=validate,
    error_template="The output was rejected: {error}. Return corrected JSON only.",
)
```

## Async

`retry_call_async` mirrors the sync API; pass an async `llm_fn`:

```python
async def call_llm(messages, **kwargs) -> str:
    return await my_async_client.chat(messages, **kwargs)

result = await retry.retry_call_async(call_llm, messages)
```

## Decorators

Wrap a call function so invoking it returns a `RetryResult` directly:

```python
retry = StructuredRetry(max_retries=3)

@retry.validated(validate)
def call_llm(messages, **kwargs) -> str:
    return my_client.chat(messages, **kwargs)

result = call_llm(messages)          # -> RetryResult

@retry.validated_async(validate)
async def call_llm_async(messages, **kwargs) -> str:
    return await my_async_client.chat(messages, **kwargs)

result = await call_llm_async(messages)
```

## API

- `StructuredRetry(max_retries=3, validate_fn=None, error_template=None)`
  - `max_retries` — total attempts allowed (first try + retries); must be `>= 1`.
  - `validate_fn` — `Callable[[str], None]` that raises `ValidationError` on bad
    output, or `None` to accept every response.
  - `error_template` — format string with an `{error}` placeholder.
  - `.retry_call(llm_fn, messages, **kwargs) -> RetryResult`
  - `.retry_call_async(llm_fn, messages, **kwargs) -> RetryResult`
  - `.validated(validate_fn)` / `.validated_async(validate_fn)` — decorators.
- `RetryResult` — `.content`, `.attempts`, `.succeeded`, `.errors`.
- `ValidationError` — raise from your `validate_fn`.
- `MaxRetriesExceededError` — `.attempts`, `.last_error`.

`**kwargs` passed to `retry_call` are forwarded unchanged to your `llm_fn` on
every attempt (e.g. `temperature=0.0`, `model="..."`).

## License

MIT
