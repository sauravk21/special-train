# Backend Serving for AI/ML Systems: A Complete Tutorial

This tutorial will take you from the fundamentals of backend serving to building a production-style API for AI/ML inference. We'll use Python, FastAPI, and practical examples throughout.

Since you're already working with computer vision and exploring generative AI, I'll use examples such as image classification, image-quality assessment, and model inference. The same principles apply to LLM APIs, RAG pipelines, and generative AI services.


---

## 1. The big picture: What is backend serving?

Suppose you have trained a computer vision model that takes an image and predicts its quality score. You now want a mobile application or another backend service to use it.

You need a way for the application to send an image to your model and receive a prediction.

### Request-to-prediction flow

```text
Client application (mobile app, web app, or another service)
    │ HTTP request / response
    ▼
FastAPI backend
    │ Validation · Authentication · Rate limiting · Timeouts
    ▼
Inference layer
    │ Preprocessing → Model → Postprocessing
    ▼
Prediction returned to client
      JSON response, result ID, or generated output URL
```

For a real production system, you might also need a database, object storage, a message queue, caching, logging, monitoring, and multiple server replicas.

### The six topics you'll learn

1. **FastAPI** — Create backend services, endpoints, and inference APIs.
2. **REST APIs** — HTTP methods, status codes, request/response formats, and API design.
3. **Asynchronous tasks** — Background work, job queues, polling, and long-running inference.
4. **Request validation** — Validate inputs with Pydantic, handle errors, and enforce limits.
5. **Timeouts and retries** — Handle slow dependencies, transient failures, and retry storms.
6. **Rate limiting** — Protect APIs from excessive traffic and manage inference costs.


---

## 2. FastAPI fundamentals

### 2.1 What is FastAPI?

FastAPI is a Python web framework for building APIs. It is particularly useful for ML services because it integrates naturally with Python libraries such as PyTorch, Transformers, NumPy, and Pillow.

FastAPI provides:

- HTTP routing: deciding which function handles a request.
- Input validation using Python type hints and Pydantic.
- JSON serialization and response generation.
- Automatic interactive API documentation.
- Asynchronous endpoint support using `async` and `await`.
- Dependency injection for shared resources such as authentication and database connections.

FastAPI does not automatically make a model fast, distribute GPU workloads, or provide a complete production job queue. Those concerns require additional design.

### 2.2 Set up your environment

On macOS, create a virtual environment:

```bash
mkdir ai-serving
cd ai-serving

python3 -m venv .venv
source .venv/bin/activate

pip install fastapi "uvicorn[standard]" httpx
```

- `fastapi`: web framework.
- `uvicorn`: ASGI server that runs your application.
- `httpx`: HTTP client used later for outbound calls and testing.

Create a file called `main.py`.

### 2.3 Your first API

```python

from fastapi import FastAPI

app = FastAPI(title="AI Serving API")


@app.get("/")
def home():
    return {"message": "AI serving is running"}


@app.get("/health")
def health():
    return {"status": "ok"}

```

Start the server:

```bash
uvicorn main:app --reload
```

Open these URLs in your browser:

- `http://127.0.0.1:8000/` — your first endpoint.
- `http://127.0.0.1:8000/health` — health check.
- `http://127.0.0.1:8000/docs` — interactive Swagger API documentation.

Understand the command `uvicorn main:app --reload`:

| Part       | Meaning                                           |
| ---------- | ------------------------------------------------- |
| `uvicorn`  | Starts the ASGI server                            |
| `main`     | Python file `main.py`                             |
| `app`      | FastAPI object inside that file                   |
| `--reload` | Restarts the development server when code changes |

Use `--reload` in development, not as a production deployment setting.

### 2.4 How a request is handled

When a client calls your endpoint, the typical flow is:

1. Uvicorn receives the HTTP request.
2. FastAPI matches the URL and HTTP method to a route.
3. FastAPI validates and parses the inputs.
4. Your Python function executes.
5. The returned value is serialized into a response.
6. The server sends the response to the client.

For an AI service, model inference usually occurs in step 4, although production systems often separate inference into another service.


---

## 3. REST APIs: How backend services communicate

REST is an architectural style for designing APIs around resources. In practice, a REST API commonly uses HTTP methods, URLs, JSON payloads, and status codes.

Imagine you're building an AI image-analysis service.

### 3.1 HTTP methods

| Method   | Purpose                          | Example                       |
| -------- | -------------------------------- | ----------------------------- |
| `GET`    | Retrieve information             | Get an inference job's status |
| `POST`   | Submit data or create a resource | Submit an image for analysis  |
| `PUT`    | Replace a resource               | Replace a configuration       |
| `PATCH`  | Partially update a resource      | Update a job's metadata       |
| `DELETE` | Delete a resource                | Delete a stored result        |

A useful API design might look like this:

| Endpoint            | Method | Purpose                          |
| ------------------- | ------ | -------------------------------- |
| `/health`           | GET    | Check basic application health   |
| `/v1/predictions`   | POST   | Run an inference request         |
| `/v1/jobs`          | POST   | Submit a long-running job        |
| `/v1/jobs/{job_id}` | GET    | Retrieve job status              |
| `/v1/jobs/{job_id}` | DELETE | Request cancellation or deletion |

The `/v1` prefix versions the API, allowing you to introduce breaking changes in `/v2` later.

### 3.2 Request and response

A client sends an HTTP request:

```http
POST /v1/predictions HTTP/1.1
Content-Type: application/json
```

For a text-based prediction API, the body might be:

```json
{
  "text": "This image is blurry",
  "model": "quality-classifier"
}
```

The server responds:

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "label": "low_quality",
  "confidence": 0.94
}
```

JSON is a common format for structured API data. Actual image uploads often use `multipart/form-data` or an authorized object-storage URL rather than encoding large image files as JSON.

### 3.3 HTTP status codes you should know

| HTTP status | Meaning |
|---|---|
| **200 OK** | The request succeeded. |
| **201 Created** | A new resource was created. |
| **202 Accepted** | A long-running job was accepted, but is not finished. |
| **400 Bad Request** | The request is malformed or otherwise invalid. |
| **401 Unauthorized** | Authentication is missing or invalid. |
| **403 Forbidden** | The caller lacks permission. |
| **404 Not Found** | The requested resource does not exist. |
| **413 Content Too Large** | The uploaded payload exceeds a size limit. |
| **422 Unprocessable Content** | The request data failed validation. |
| **429 Too Many Requests** | A rate limit was exceeded. |
| **500 Internal Server Error** | An unexpected server-side error occurred. |
| **503 Service Unavailable** | The service cannot currently handle the request. |
| **504 Gateway Timeout** | A gateway or proxy timed out waiting for an upstream service. |

A common interview question is: What is the difference between 200 and 202?

- `200 OK`: the requested operation has completed successfully.
- `202 Accepted`: the server has accepted the work for processing, but the result might not exist yet.

For example, an image-quality prediction that completes in 100 ms might return `200`. A video-analysis request that takes 30 seconds might return `202` with a job ID.

### 3.4 Test an API with curl

```bash
curl http://127.0.0.1:8000/health
```

For a JSON POST endpoint:

```bash
curl -X POST http://127.0.0.1:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"text":"hello","priority":1}'
```

`curl` is useful because you can test a backend independently of the mobile app or frontend.


---

## 4. Request validation with Pydantic

Request validation is one of the most important responsibilities of an API backend.

Never assume that the client sends valid data. A mobile application might have a bug, a client could be outdated, or a malicious caller could send unexpected inputs.

### 4.1 Validate using types and constraints

Suppose an endpoint accepts text and an inference priority.

```python

from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()


class PredictionRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=2000)]
    priority: int = Field(default=1, ge=1, le=5)


@app.post("/v1/score")
def score(request: PredictionRequest):
    return {
        "text_length": len(request.text),
        "priority": request.priority,
        "label": "example"
    }

```

What happens when the client sends:

```json
{
  "text": "",
  "priority": 10
}
```

Pydantic rejects the request because:

- `text` cannot be empty.
- `priority` must be between 1 and 5.

FastAPI normally returns `422 Unprocessable Content` with details describing the invalid fields. Your endpoint function does not run for this failed body validation.

### 4.2 Validation is not just about data types

For an AI inference endpoint, you should validate several layers.

- **Schema validation:** Are required fields present? Are values the right types and within the permitted range?
- **File validation:** Is the file an accepted image format? Is its size within the limit? Can it actually be decoded?
- **Business-rule validation:** Is this model enabled? Is the requested operation allowed for this user?
- **Security validation:** Is the caller authenticated and authorized? Are URLs and uploaded content handled safely?

Pydantic validates structured data. It does not replace authentication, authorization, file decoding checks, or resource limits.

### 4.3 File upload validation

For a real image endpoint, you might use `UploadFile`:

```python
from typing import Annotated
from fastapi import File, HTTPException, UploadFile

MAX_SIZE = 10 * 1024 * 1024  # 10 MB

@app.post("/v1/image-quality")
async def image_quality(
    file: Annotated[UploadFile, File()]
):
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(415, "Unsupported image type")

    content = await file.read(MAX_SIZE + 1)

    if len(content) > MAX_SIZE:
        raise HTTPException(413, "Image exceeds 10 MB")

    if not content:
        raise HTTPException(400, "Empty image")

    # Decode with a trusted image library and validate dimensions.
    # Run preprocessing and model inference here.
    return {"status": "accepted", "bytes": len(content)}
```

Important production details:

- `content_type` is provided by the client and can be spoofed.
- Reading a bounded amount of data helps limit application-level memory consumption, but multipart parsing may already have consumed resources. Enforce request-body limits at the proxy or server layer as well.
- Decode the image with a trusted library; reject malformed files and unreasonable dimensions.
- Consider decompression bombs and total pixel count, not just compressed file size.
- Close or otherwise properly release uploaded-file resources.

### 4.4 Validation versus exception handling

Validation answers: “Is this request acceptable?”

Exception handling answers: “What should happen if something fails while processing an acceptable request?”

For example:

```python
from fastapi import HTTPException

@app.get("/v1/models/{model_name}")
def get_model(model_name: str):
    if model_name not in {"quality-v1", "quality-v2"}:
        raise HTTPException(
            status_code=404,
            detail="Model not found"
        )

    return {"model": model_name}
```

Use explicit HTTP errors for expected failures. Unexpected internal exceptions should be logged with appropriate context and converted into safe client-facing responses without exposing stack traces, credentials, or internal implementation details.


---

## 5. Asynchronous tasks: `async`/`await` and background processing

This is a particularly important topic for AI-serving systems.

There are two different concepts that people often confuse:

1. Asynchronous I/O: a request handler can yield control while waiting for network or other asynchronous operations.
2. Background jobs: work is submitted for execution separately from the HTTP request, often through a durable queue and worker process.

They solve different problems.

### 5.1 Synchronous versus asynchronous execution

Consider a backend that calls an external LLM API.

**Synchronous version:**

```python
import time

def call_service():
    time.sleep(2)  # Simulates blocking I/O
    return {"result": "done"}
```

While a synchronous handler is sleeping, the thread executing it is blocked.

**Asynchronous version:**

```python
import asyncio

async def call_service():
    await asyncio.sleep(2)  # Simulates nonblocking I/O
    return {"result": "done"}
```

While the asynchronous function is waiting, its event loop can run other tasks.

The key is `await`: it allows the current coroutine to suspend while an awaitable operation is pending.

### 5.2 A concurrent FastAPI example

```python

import asyncio
from fastapi import FastAPI

app = FastAPI()


async def fake_external_call():
    await asyncio.sleep(2)
    return {"result": "done"}


@app.get("/v1/async-demo")
async def async_demo():
    result = await fake_external_call()
    return result

```

If several requests are waiting on independent network operations, the server can make progress on multiple requests without dedicating a blocked thread to each wait.

However, `async def` alone does not make CPU-heavy work faster.

### 5.3 Why `async` does not automatically speed up ML inference

Imagine this code:

```python
@app.post("/v1/predict")
async def predict():
    result = model.predict(large_input)
    return {"result": result}
```

If `model.predict()` performs substantial CPU work synchronously, it can block the event loop. The fact that the endpoint is declared `async` does not change that.

For AI systems, distinguish these workloads:

| Workload                            | Typical approach                                                        |
| ----------------------------------- | ----------------------------------------------------------------------- |
| Calling an external API             | Async HTTP client                                                       |
| Waiting for an async database query | Async database driver                                                   |
| Lightweight CPU processing          | Regular function or thread pool, depending on needs                     |
| Heavy CPU inference                 | Separate process or inference worker                                    |
| GPU inference                       | Dedicated model worker or inference server, with controlled concurrency |
| Long video generation or analysis   | Durable job queue and worker                                            |

Threads can help with blocking I/O and some native-library operations, but Python CPU-bound work may be constrained by the GIL. GPU workloads have their own memory and scheduling constraints. Benchmark your actual model and deployment configuration.

### 5.4 FastAPI `BackgroundTasks`

FastAPI provides `BackgroundTasks` for relatively small tasks that can run after the response is sent.

```python
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()


def record_event(event: str):
    # Example: perform a small, noncritical operation.
    print("Event:", event)


@app.post("/v1/events")
async def create_event(
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(
        record_event, "prediction_requested"
    )

    return {"status": "accepted"}
```

The HTTP response can be sent before `record_event` runs.

But there are limitations:

- The task is not a durable job stored in a queue.
- If the process crashes or restarts, work may be lost.
- Heavy tasks can consume server resources and interfere with request handling.
- There is no built-in distributed scheduling, robust retry system, or job-state database.

Use this feature for simple, noncritical work rather than a large video inference pipeline.

### 5.5 Production pattern: Job queue and workers

For long-running inference, use a job architecture.

```text
1. Client                         Uploads video or submits an inference request
       ↓
2. FastAPI                        Authenticates, validates, stores job metadata
       ↓
3. Durable queue                  Redis-backed queue, RabbitMQ, or another broker
       ↓
4. Worker                         Loads the model, processes input, records progress
       ↓
5. Result storage + status API    Stores output and makes it available to the client
```

A typical client flow is:

1. `POST /v1/jobs` submits the job.
2. The backend returns `202 Accepted` with a unique `job_id`.
3. The worker processes the input.
4. The client polls `GET /v1/jobs/{job_id}`.
5. The status changes from `queued` to `running`, then `succeeded` or `failed`.
6. The client retrieves the result when it is ready.

Example response:

```json
{
  "job_id": "job_82ab",
  "status": "queued",
  "status_url": "/v1/jobs/job_82ab"
}
```

You can implement this with Celery, Dramatiq, RQ, or another suitable worker framework. The choice depends on durability requirements, queue semantics, operational complexity, and workload type.

For GPU inference, avoid casually starting many workers that each load a full model onto the same GPU. A model-serving process often needs explicit batching and concurrency control.


---

## 6. Timeouts: Prevent requests from waiting forever

A timeout defines how long a caller is willing to wait for an operation.

Imagine your FastAPI backend calls an external embedding service, which becomes slow. Without appropriate timeouts, requests may remain pending, consuming connections and resources.

### 6.1 Types of timeouts

| Timeout              | What it limits                                             |
| -------------------- | ---------------------------------------------------------- |
| Connection timeout   | Time to establish a connection                             |
| Read timeout         | Time waiting for data from a connected service             |
| Write timeout        | Time allowed to send request data                          |
| Pool timeout         | Time waiting to obtain a connection from a connection pool |
| Application deadline | Total time allowed for an operation                        |
| Gateway timeout      | How long a proxy or gateway waits for an upstream response |

These are not interchangeable. A read timeout might limit an individual wait for incoming data rather than the entire operation.

### 6.2 Add a timeout to an outbound HTTP request

Install `httpx` if you haven't already:

```bash
pip install httpx
```

Then:

```python

import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/v1/external-data")
async def external_data():
    timeout = httpx.Timeout(
        5.0,
        connect=1.0,
        read=3.0,
        write=3.0,
        pool=1.0,
    )

    try:
        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:
            response = await client.get(
                "https://httpbin.org/json"
            )
            response.raise_for_status()
            return response.json()

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Upstream service timed out",
        )

    except httpx.HTTPStatusError:
        raise HTTPException(
            status_code=502,
            detail="Upstream service returned an error",
        )

```

This illustrates timeout configuration and error mapping. In a production application, prefer a reusable, appropriately configured HTTP client rather than constructing a new client for every request. Reusing clients helps connection pooling.

Also, `raise_for_status()` can produce errors for non-2xx upstream responses. You should decide deliberately how different upstream status codes map to your own API.

### 6.3 Timeout budgets across services

Suppose the end-to-end API deadline is 5 seconds.

**API total budget: 5 seconds**

| Component | Allocated time |
|---|---:|
| Validation and preprocessing | 0.6 seconds |
| Inference | 2.8 seconds |
| External dependency | 1.1 seconds |
| Serialization and overhead | 0.5 seconds |
| **Total** | **5.0 seconds** |

*Illustrative allocation, not a benchmark. Real budgets need measurement and safety margins.*

The important principle is that downstream operations must fit inside the overall deadline. Don't give every dependency a fresh 5-second timeout if the entire request must finish within 5 seconds.

Use a deadline or remaining-time budget, and reserve time for returning the response.

### 6.4 What happens when a timeout occurs?

A timeout means the caller stopped waiting. It does not necessarily mean that the underlying operation stopped.

For example:

- Your HTTP request to an inference service times out.
- The GPU worker continues processing the image.
- The client retries.
- Now two inference jobs are consuming GPU resources.

This is why timeouts, cancellation, retries, and idempotency must be designed together.

For long-running jobs, a better pattern is to acknowledge the job quickly and let the worker finish independently of the client's HTTP connection.


---

## 7. Retries: Recover from temporary failures

A retry means attempting an operation again after a failure.

Network failures are often transient. A service might temporarily return `503`, a connection might reset, or a request might time out.

However, retries are not always safe.

### 7.1 When should you retry?

| Failure                         | Typical strategy                                                   |
| ------------------------------- | ------------------------------------------------------------------ |
| Temporary connection failure    | Retry if safe                                                      |
| Timeout                         | Retry only if the operation is safe and within the deadline        |
| HTTP `429`                      | Respect `Retry-After`, if provided                                 |
| HTTP `502`, `503`, `504`        | Often retryable, depending on the operation                        |
| HTTP `400`, `401`, `403`, `404` | Usually do not retry unchanged                                     |
| Invalid image or schema         | Do not retry                                                       |
| GPU out of memory               | Usually requires a different recovery strategy, not blind retrying |

Retry decisions should consider the endpoint's semantics, error classification, remaining deadline, and whether the upstream server may already have performed the operation.

### 7.2 Exponential backoff

If a request fails repeatedly, don't immediately send another request. Doing so can make an overloaded service even less healthy.

Exponential backoff increases the delay between attempts:

\\[ d_n=\min(d\_{\max},d_0 2^n) \\]

Where:

- \\(d_0\\): initial delay.
- \\(n\\): retry number, starting from zero.
- \\(d\_{\max}\\): maximum delay.

Add jitter, a random variation in the delay, to avoid many clients retrying simultaneously.

Example with an initial delay of 0.5 seconds and a cap of 4 seconds:

| Retry number | Base delay before jitter |
| ------------ | ------------------------ |
| 1            | 0.5 s                    |
| 2            | 1 s                      |
| 3            | 2 s                      |
| 4            | 4 s                      |
| 5            | 4 s                      |

The cap prevents the delay from growing indefinitely.

### 7.3 Implement a bounded retry loop

Here is an educational example for retrying an outbound GET request:

```python

import asyncio
import random

import httpx


async def get_with_retries(
    client: httpx.AsyncClient,
    url: str,
    max_attempts: int = 3,
):
    for attempt in range(max_attempts):
        try:
            response = await client.get(url)

            if response.status_code not in {
                429, 502, 503, 504
            }:
                response.raise_for_status()
                return response

            # Do not retry when no attempts remain.
            if attempt == max_attempts - 1:
                response.raise_for_status()

            retry_after = response.headers.get(
                "Retry-After"
            )

            if retry_after is not None:
                # This simple example supports numeric seconds.
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = None
            else:
                delay = None

        except (
            httpx.ConnectError,
            httpx.TimeoutException,
        ):
            if attempt == max_attempts - 1:
                raise
            delay = None

        if delay is None:
            delay = min(4.0, 0.5 * (2 ** attempt))
            delay *= random.uniform(0.8, 1.2)

        await asyncio.sleep(delay)

    raise RuntimeError("Unreachable")

```

This is a teaching example, not a complete production retry library. A production implementation should also:

- Respect the overall deadline, not just the number of attempts.
- Bound and validate `Retry-After` values.
- Retry only specifically classified failures.
- Avoid retrying unsafe operations blindly.
- Use a shared connection pool.
- Track retry metrics.
- Account for cancellation and shutdown.

### 7.4 The idempotency problem

Consider this request:

```http
POST /v1/payments
```

The server charges a payment, but the response gets lost. The client sees a timeout and retries. The payment might be charged twice.

Inference APIs have an analogous problem: a retried request might duplicate expensive work or create duplicate jobs.

An idempotency key can help:

```http
POST /v1/jobs
Idempotency-Key: 7a4d9f-example-key
```

The server associates the key with a previously submitted operation. A repeat request with the same key and same logical payload returns the existing result rather than creating a second job.

Important implementation details:

1. Scope the key to the authenticated caller.
2. Persist the key and operation state atomically with job creation.
3. Reject conflicting payloads that reuse the same key.
4. Retain records for a defined period.
5. Coordinate worker execution so duplicate queue deliveries don't duplicate side effects.

A header alone does not make an API idempotent. The backend must implement the semantics.

### 7.5 Retry storms

Imagine 1,000 clients call an AI service at once. The service becomes overloaded and returns `503`. All 1,000 clients retry immediately.

The additional traffic makes the overload worse.

Avoid this with:

- Exponential backoff and jitter.
- Bounded attempts.
- Deadlines.
- Rate limiting.
- Circuit breakers.
- Queue-based load leveling for background jobs.

Retries should help a service recover, not force it to handle more work while it is already failing.


---

## 8. Rate limiting: Protect your backend from excessive traffic

Rate limiting controls how frequently a caller can make requests.

For AI services, this is particularly important because a single request might cost GPU time, external API credits, or substantial memory.

### 8.1 Common rate-limiting algorithms

| Algorithm | Description |
|---|---|
| **Token bucket** | Tokens refill over time. Each request consumes tokens. Allows controlled bursts while enforcing a long-term average rate. |
| **Sliding window** | Counts requests over a moving time interval. Gives smoother limits than a simple fixed-window counter. |
| **Fixed window** | Counts requests in fixed time intervals. Simple and efficient, but bursts can occur around window boundaries. |

### 8.2 Fixed-window example

Suppose a client is allowed 60 requests per minute.

- Requests 1–60: accepted.
- Request 61 within that window: rejected.
- Once the next window starts: the client can send requests again.

The problem is the boundary. A client might send 60 requests just before the window resets and another 60 just after it resets.

A token bucket or sliding-window approach can better control such bursts.

### 8.3 A simple FastAPI rate limiter

For learning, we can implement a per-client fixed-window limiter in memory.

```python

import time
from collections import defaultdict
from threading import Lock

from fastapi import FastAPI, HTTPException, Request

app = FastAPI()

LIMIT = 10
WINDOW_SECONDS = 60

# client key -> (window start, request count)
counters = defaultdict(lambda: [0.0, 0])
counter_lock = Lock()


def enforce_rate_limit(client_key: str):
    now = time.monotonic()

    with counter_lock:
        window = counters[client_key]

        if now - window[0] >= WINDOW_SECONDS:
            window[0] = now
            window[1] = 0

        if window[1] >= LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": "60"},
            )

        window[1] += 1


@app.get("/v1/limited")
def limited_endpoint(request: Request):
    # Suitable only when client IP information is trusted.
    client_key = (
        request.client.host
        if request.client
        else "unknown"
    )

    enforce_rate_limit(client_key)

    return {"message": "Request accepted"}

```

This example demonstrates the basic algorithm. It is not suitable as-is for production.

Limitations:

- Each process has its own counters.
- Restarting the process resets the counters.
- The dictionary can grow without bound unless entries are cleaned up.
- IP-based limiting may penalize users sharing an address.
- Proxy headers must not be trusted unless the proxy is configured as trusted.
- It does not provide per-user authentication, distributed atomic updates, or weighted limits.

### 8.4 Distributed rate limiting with Redis

Suppose your service runs on four API instances.

API 1

API 2

API 3

API 4

Shared Redis

Atomic counters or token-bucket state

All API instances consult shared rate-limit state, so a caller cannot bypass the quota simply by reaching another instance.

For a fixed-window counter, a common approach is an atomic Redis operation or Lua script that increments a counter and sets its expiry correctly. For token buckets and sliding windows, use an implementation that performs the relevant state update atomically.

Typical policies include:

- 60 requests/minute per user.
- 10 inference requests/minute for a free tier.
- 1,000 requests/minute per enterprise account.
- A maximum of 2 concurrent expensive inference jobs per user.

For AI workloads, rate limiting and concurrency limiting are different. A user might make only two requests per minute, but if each request occupies a GPU for 60 seconds, those two requests can still consume significant resources simultaneously.

### 8.5 Return useful rate-limit information

A rate-limited response might look like:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 20
Content-Type: application/json
```

```json
{
  "detail": "Rate limit exceeded"
}
```

Depending on your API design, you can also expose remaining quota and reset information in headers.

The `Retry-After` value should reflect the actual wait when possible. A hardcoded value is appropriate only for an illustrative example.


---

## 9. How all six topics work together

Let's put the concepts into a single scenario.

A mobile client submits an image to your AI image-quality API.

1. **REST API** — The client calls `POST /v1/predictions` with an image or authorized image reference.
2. **Rate limiting and validation** — Authenticate the caller, enforce quotas, check file size and format, and validate parameters.
3. **Timeout policy** — Apply an end-to-end deadline and appropriate timeouts to downstream services.
4. **Inference** — Run the model in a controlled inference process or worker. Use a job queue if processing is long-running.
5. **Retry and failure handling** — Retry only transient, safe operations within the deadline. Prevent duplicate jobs with idempotency.
6. **Response** — Return a prediction, or return `202` and a job ID if the result will be available later.

### 9.1 A few production decisions

| Question                                               | Practical answer                                                                                                    |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| Should I load the model inside every request?          | No. Usually load it once per dedicated inference process and manage its lifecycle.                                  |
| Should I use `async def` for every endpoint?           | No. Use it where async I/O is beneficial. Don't block the event loop with synchronous operations.                   |
| Should every operation be retried?                     | No. Retry only appropriate transient failures and consider duplicate side effects.                                  |
| Is a timeout enough to stop inference?                 | No. Cancellation must be supported by the execution layer, or the worker may continue.                              |
| Is an in-memory rate limiter enough?                   | Only for simple, single-process demonstrations. Multi-instance systems need shared state or a gateway-level policy. |
| Should video inference happen inside the HTTP request? | Usually not for long-running work. Use a durable queue, worker, and job-status endpoint.                            |


---

## 10. Production deployment: What else do you need?

The previous sections cover the six core topics. To deploy an actual backend, you also need to understand how it behaves under load and how you operate it.

### 10.1 Run multiple server workers

For example:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Multiple Uvicorn workers can help handle concurrent requests, but each worker is generally a separate process. If every worker loads a large ML model, your RAM or GPU memory consumption may multiply.

For GPU-heavy inference, a common design is to run a small API tier separately from the model-serving tier.

### 10.2 Monitor latency

Don't measure only average latency. Track percentiles.

| Metric             | Meaning                                      |
| ------------------ | -------------------------------------------- |
| P50 latency        | Half of requests finish within this time     |
| P95 latency        | 95% finish within this time                  |
| P99 latency        | 99% finish within this time                  |
| Request throughput | Requests handled per second                  |
| Error rate         | Percentage of requests that fail             |
| Queue depth        | Number of jobs waiting for workers           |
| GPU utilization    | How much of the GPU's capacity is being used |
| Retry rate         | How often operations need another attempt    |

For example, a P95 latency of 800 ms means approximately 95% of measured requests completed within 800 ms. It does not mean every request completes within that time.

A high P99 with a reasonable P50 often indicates a problem affecting a subset of requests, such as queueing, large inputs, cold starts, or slow dependencies.

### 10.3 Basic production checklist

- Validate request schemas and uploaded files.
- Add authentication and authorization.
- Enforce request-size and concurrency limits.
- Set explicit timeout and retry policies.
- Implement rate limiting.
- Use idempotency for operations that can be duplicated.
- Keep model loading separate from per-request initialization.
- Log request IDs, latency, error categories, and model versions.
- Track P95/P99, queue depth, and resource utilization.
- Store secrets in a secret manager or environment-based deployment configuration.
- Use a durable job queue for long-running inference.
- Test behavior under load and during dependency failures.
