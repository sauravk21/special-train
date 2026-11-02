# Data Engineering for AI Backend Systems

A concise, practical guide to moving raw data into reliable AI applications. The running example is an **image-ingestion pipeline**: upload an image, validate it, process it, and store a result.

**Prerequisites:** Basic Python and SQL. Examples use Python 3.10+, PostgreSQL, S3, Pydantic 2, and `confluent-kafka`. Infrastructure must be provisioned separately; snippets marked **pseudocode** describe application-specific steps.

**Topics covered:**

1. Pipeline architecture and processing models.
2. Python transformations and file formats.
3. SQL schemas, queries, and transactions.
4. Object storage and upload workflows.
5. Validation, data quality, and contracts.
6. Kafka concepts and delivery guarantees.
7. Kafka producers, consumers, and retries.
8. End-to-end ingestion and batch exports.
9. Reliability, monitoring, and reproducibility.
10. Revision questions and a practice project.

## 1. Pipeline Architecture and Processing Models

### 1.1 What data engineering does

Data engineering makes data usable, trustworthy, and available to applications and models. A pipeline typically performs these steps:

1. **Ingest:** Collect files, API requests, database changes, or events.
2. **Validate:** Check structure, values, ownership, and completeness.
3. **Transform:** Clean, normalize, enrich, or extract features.
4. **Store:** Persist raw inputs, structured metadata, and outputs.
5. **Serve:** Make results available for inference, analytics, or training.

For example, image preprocessing converts decoded pixels into model inputs; the pipeline also tracks which source image and model version produced each result.

### 1.2 Components and responsibilities

```text
Client → Object storage: upload image bytes
Client → API → PostgreSQL: confirm upload + write outbox event
Outbox relay → Kafka → Python worker
Python worker → Object storage: read verified image
Python worker → PostgreSQL: save processing result
PostgreSQL → Parquet export → Analytics or training
```

| Component | Responsibility | Example data |
| --- | --- | --- |
| Python | Validate, transform, and coordinate processing | Image dimensions and model inputs |
| PostgreSQL | Store transactional metadata and relationships | Owner, upload status, result |
| Object storage | Store large files and datasets | Images, Parquet files, model artifacts |
| Kafka | Retain and distribute events | `image.uploaded` notifications |
| Outbox relay | Publish events committed with database changes | Pending event rows |

Keep image bytes in object storage and pass object references through Kafka. The outbox is explained in section 7.4.

### 1.3 Batch, streaming, ETL, and ELT

| Approach | Meaning | Example |
| --- | --- | --- |
| Batch | Process a bounded dataset periodically | Export yesterday's results |
| Streaming | Process events as they arrive | Process each confirmed upload |
| Micro-batch | Process small groups at short intervals | Aggregate events every minute |
| ETL | Extract, transform, then load into the target | Clean records before loading a reporting table |
| ELT | Extract, load, then transform in the target system | Load raw data into a warehouse, then model it with SQL |

Choose based on freshness requirements, volume, cost, and recovery needs. A single system can use streaming for inference and batch processing for training datasets.

## 2. Python Transformations and File Formats

### 2.1 Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas pyarrow 'pydantic>=2,<3' 'psycopg[binary]' boto3 confluent-kafka
```

- **pandas:** Transform tabular data.
- **PyArrow:** Read and write Parquet.
- **Pydantic:** Validate event contracts.
- **psycopg:** Connect to PostgreSQL.
- **boto3:** Access S3.
- **confluent-kafka:** Produce and consume Kafka events.

Use environment variables for connection settings and workload identities for cloud credentials. Pin tested dependency versions in a real project.

### 2.2 Keep transformations independent of I/O

A small function is easier to reuse and test when it does not read files, call services, or modify its input.

```python
def add_megapixels(record: dict) -> dict:
    width, height = record["width"], record["height"]
    if width <= 0 or height <= 0:
        raise ValueError("Dimensions must be positive")
    return {**record, "megapixels": width * height / 1_000_000}


record = {"image_id": "img-1", "width": 2000, "height": 1000}
result = add_megapixels(record)
assert result["megapixels"] == 2.0
assert "megapixels" not in record
```

This function assumes numeric dimensions have already been parsed. Schema validation belongs at the input boundary.

### 2.3 Clean and transform tabular data

Separate rejected rows from accepted rows so that cleaning does not silently hide input problems.

```python
import pandas as pd


def clean_images(frame: pd.DataFrame):
    required = {"image_id", "width", "height"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame = frame.copy()
    for column in ("width", "height"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    valid = (
        frame["image_id"].notna()
        & frame["image_id"].astype("string").str.strip().ne("")
        & frame["width"].between(1, 50_000)
        & frame["height"].between(1, 50_000)
        & frame["width"].mod(1).eq(0)
        & frame["height"].mod(1).eq(0)
        & (frame["width"] * frame["height"] <= 100_000_000)
    ).fillna(False)

    clean = frame.loc[valid].copy()
    rejected = frame.loc[~valid].copy()
    clean["megapixels"] = clean["width"] * clean["height"] / 1_000_000
    return clean, rejected


frame = pd.read_csv("images.csv", dtype={"image_id": "string"})
clean, rejected = clean_images(frame)
clean.to_parquet("clean-images.parquet", index=False)
rejected.to_csv("rejected-images.csv", index=False)
```

- Define whether repeated IDs mean duplicates or legitimate updates before deduplicating.
- Preserve the original input for investigating rejected records.
- Use vectorized column operations for ordinary tabular calculations.
- Avoid filling missing values unless the replacement has a valid business meaning.

### 2.4 Process large inputs with bounded memory

A generator yields records as needed; chunking loads a limited number of rows at a time. Neither automatically makes a job distributed.

The following continues the previous example:

```python
from pathlib import Path

output = Path("export-run-001")
output.mkdir(exist_ok=False)  # Use a new directory for each run.

for number, chunk in enumerate(pd.read_csv(
    "large-images.csv", dtype={"image_id": "string"}, chunksize=10_000
)):
    clean, rejected = clean_images(chunk)
    clean.to_parquet(output / f"part-{number:05d}.parquet", index=False)
    rejected.to_csv(output / f"rejected-{number:05d}.csv", index=False)
```

Chunk-local deduplication cannot remove duplicates across the entire dataset. Use a database constraint, persistent state, or a separate global deduplication step.

### 2.5 Select a file format

| Format | Suitable for | Main tradeoff |
| --- | --- | --- |
| CSV | Simple exchange and manual inspection | Weak typing; values need parsing |
| JSON | API payloads and nested events | Verbose for large tabular datasets |
| JSONL | Incremental record processing | One JSON object per line; still verbose |
| Parquet | Analytics and training datasets | Typed, compressed columns; not plain text |

Read selected Parquet columns to avoid loading unused fields:

```python
import pandas as pd

features = pd.read_parquet(
    "clean-images.parquet", columns=["image_id", "megapixels"]
)
```

## 3. SQL Schemas, Queries, and Transactions

### 3.1 Model entities and enforce constraints

Use PostgreSQL for ownership, relationships, processing state, and queryable results. Store object keys instead of image bytes.

```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    email TEXT NOT NULL UNIQUE
);

CREATE TABLE images (
    image_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id),
    object_key TEXT NOT NULL UNIQUE,
    width INTEGER CHECK (width > 0),
    height INTEGER CHECK (height > 0),
    status TEXT NOT NULL DEFAULT 'pending_upload'
        CHECK (status IN (
            'pending_upload', 'uploaded', 'processing', 'completed', 'failed'
        )),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);
```

- **Primary key:** Identifies one row.
- **Foreign key:** Requires the referenced owner to exist.
- **Unique constraint:** Prevents duplicate values.
- **Check constraint:** Restricts allowed values.
- **Not-null constraint:** Requires a value.

Dimensions may be unknown before the image is decoded. A `CHECK (width > 0)` permits `NULL`; add `NOT NULL` when a field must always be present.

### 3.2 CRUD and parameterized queries

CRUD means create, read, update, and delete. Bind values through the database driver instead of inserting user input into SQL strings.

```python
import os
from uuid import uuid4

import psycopg

image_id = uuid4()
user_id = os.environ["DEMO_USER_ID"]  # Must already exist in users.

with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO images (image_id, user_id, object_key)
               VALUES (%s, %s, %s)""",
            (image_id, user_id, f"raw/{image_id}.jpg"),
        )
        cursor.execute(
            "SELECT image_id, status FROM images WHERE image_id = %s",
            (image_id,),
        )
        print(cursor.fetchone())
```

The connection context commits on successful exit and rolls back on an exception. Deleting a database row does not delete its object-storage file; coordinate both through a cleanup workflow.

### 3.3 Joins, aggregation, and window functions

Join related tables and aggregate records for reporting:

```sql
SELECT u.user_id, u.email, COUNT(i.image_id) AS upload_count
FROM users AS u
LEFT JOIN images AS i ON i.user_id = u.user_id
GROUP BY u.user_id, u.email
HAVING COUNT(i.image_id) >= 10;
```

- **INNER JOIN:** Return matching rows from both tables.
- **LEFT JOIN:** Keep every left-side row, including unmatched rows.
- **WHERE:** Filter rows before aggregation.
- **HAVING:** Filter groups after aggregation.

Use a common table expression (CTE) and a window function to find the latest upload per user without losing row details:

```sql
WITH ranked AS (
    SELECT image_id, user_id, created_at,
           ROW_NUMBER() OVER (
               PARTITION BY user_id ORDER BY created_at DESC, image_id
           ) AS position
    FROM images
)
SELECT image_id, user_id, created_at
FROM ranked
WHERE position = 1;
```

### 3.4 Transactions and concurrent updates

Transactions group related writes so they commit together or roll back together. Their ACID properties are atomicity, consistency, isolation, and durability; isolation guarantees depend on the chosen isolation level.

Use a conditional update to claim work:

```sql
UPDATE images
SET status = 'processing'
WHERE image_id = 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11'
  AND status = 'uploaded'
RETURNING image_id;
```

- A returned row means this update claimed the image.
- No row means the image is missing or no longer in the expected state.
- Recover abandoned claims with a lease or timeout; a worker can crash after claiming work.

### 3.5 Indexes and query performance

```sql
CREATE INDEX idx_images_user_created
ON images (user_id, created_at DESC);

CREATE INDEX idx_images_pending
ON images (created_at)
WHERE status = 'uploaded';
```

- Match indexes to frequent filters, joins, and sort orders.
- Use `EXPLAIN ANALYZE` to inspect actual query execution; it executes the statement.
- Account for index storage and extra write work.
- Keep large analytical scans away from latency-sensitive application queries when they compete for resources.

## 4. Object Storage and Upload Workflows

### 4.1 Buckets, keys, and data layout

An object consists of bytes, a key, and metadata inside a bucket. Prefixes resemble folders but are not ordinary filesystem directories.

```text
s3://ai-images/raw/2026/10/30/<image-id>.jpg
s3://ai-images/curated/image-results/date=2026-10-30/part-00000.parquet
s3://ai-images/models/image-quality/v3/model.pt
```

- **Raw:** Preserve source inputs for recovery and reprocessing.
- **Curated:** Store validated, transformed datasets.
- **Models:** Keep versioned inference artifacts.

Use generated IDs in keys and partition analytical datasets by useful query fields such as date. Avoid creating a separate partition for every unique image.

### 4.2 Upload files with Python

```python
import os
import boto3

s3 = boto3.client("s3")
bucket = os.environ["IMAGE_BUCKET"]

s3.upload_file(
    "local-image.jpg", bucket, "raw/image-456.jpg",
    ExtraArgs={"ContentType": "image/jpeg"},
)
```

Use the SDK's credential provider chain and workload roles. Do not embed access keys in examples or application code.

### 4.3 Direct uploads using presigned URLs

The backend can authorize a temporary direct upload, avoiding a large file transfer through the API server. This continues the S3 setup above:

```python
from uuid import uuid4

object_key = f"raw/{uuid4()}.jpg"
upload_url = s3.generate_presigned_url(
    "put_object",
    Params={
        "Bucket": bucket,
        "Key": object_key,
        "ContentType": "image/jpeg",
    },
    ExpiresIn=300,
)
```

1. Authenticate the user and create a `pending_upload` row.
2. Return a presigned URL for that row's object key.
3. Let the client upload with the required content type.
4. Verify the stored object and its ownership when upload completion is reported.
5. Mark it `uploaded` and create its outbox event in one database transaction.

A declared content type does not prove the bytes are a valid image. Decode and validate them before inference. For reproducible processing, bind the event to an immutable object or an exact storage version; a reusable upload URL can otherwise overwrite the key.

### 4.4 Integrity, versioning, and retention

Compute a checksum incrementally without loading the entire file:

```python
import hashlib


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
```

- Store checksums and object versions with metadata.
- Compare a checksum against a trusted expected value to verify integrity.
- Do not assume an S3 ETag is always an MD5 checksum.
- Apply lifecycle rules to temporary files and obsolete datasets.
- Keep files required by active models, replay windows, or retained dataset versions.

## 5. Validation, Data Quality, and Contracts

### 5.1 Validate at multiple boundaries

| Check | Example |
| --- | --- |
| Syntax | Payload is valid JSON |
| Schema | Required fields exist and types are valid |
| Range | Size and dimensions are within limits |
| Semantic | Completion time is valid for the processing state |
| Referential | Image belongs to the claimed user |
| Uniqueness | One logical result per image and pipeline version |
| Freshness | Data arrives within the expected interval |
| Distribution | Null rates or image-size distributions have not shifted unexpectedly |

Validate at ingestion, before processing, and before publishing datasets. Database constraints provide another integrity boundary.

### 5.2 Define a versioned event schema

Save this example as `schemas.py` for the producer and consumer examples in section 7. It validates the supported version and requires a timezone-aware timestamp.

```python
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, Field


class ImageUploadedEvent(BaseModel):
    event_id: UUID
    event_type: Literal["image.uploaded"]
    event_version: Literal[1]
    occurred_at: AwareDatetime
    image_id: UUID
    user_id: UUID
    object_key: str = Field(min_length=1, max_length=1024)
    object_version: str | None = None
    content_type: Literal["image/jpeg", "image/png", "image/webp"]
    size_bytes: int = Field(gt=0, le=20 * 1024 * 1024)
```

Parse incoming bytes with `ImageUploadedEvent.model_validate_json(payload)`. Pydantic raises `ValidationError` for invalid JSON or invalid field values. See the [Pydantic model documentation](https://docs.pydantic.dev/latest/concepts/models/).

This schema validates metadata, not the image itself. Verify ownership against stored records and measure actual dimensions after decoding. Enforce a pixel-count limit before expensive processing.

### 5.3 Handle invalid records deliberately

| Action | Use when |
| --- | --- |
| Reject | The caller can correct the input immediately |
| Quarantine | The original record must be preserved for investigation |
| Repair | A documented, deterministic correction is safe |
| Accept with warning | An optional field can be omitted without breaking the contract |

A quarantine record should contain:

- The source location or event ID.
- A stable error code and failure stage.
- The schema and pipeline versions.
- The failure timestamp and enough context for controlled replay.

Restrict access to quarantined payloads and avoid exposing them in ordinary logs.

### 5.4 Evolve contracts and measure quality

A contract specifies field types, units, meaning, ownership, and compatibility expectations.

- Add optional fields only when existing consumers tolerate them.
- Treat changes to required fields, units, or semantics as breaking changes.
- Migrate producers and consumers deliberately for a new schema version.
- Track acceptance rate, missing values, duplicates, freshness, and distribution shifts.
- Reconcile unique inputs against completed, quarantined, and pending records for the same time window.

## 6. Kafka Concepts and Delivery Guarantees

### 6.1 Core concepts

| Term | Meaning |
| --- | --- |
| Producer | Publishes records |
| Topic | Named event stream |
| Partition | Ordered log within a topic |
| Broker | Server that stores and serves records |
| Consumer | Reads records |
| Consumer group | Consumers sharing a processing workload |
| Offset | Position within one partition |
| Retention | Policy controlling how long or how much data is retained |

Consuming a record does not delete it. Other groups can read the same retained data independently.

### 6.2 Partitioning, ordering, and parallelism

```text
Topic: image.uploaded.v1
Partition 0 → Consumer A in inference group
Partition 1 → Consumer B in inference group
Partition 2 → Consumer C in inference group

A separate analytics group can read all three partitions independently.
```

- Kafka preserves log order within a partition, not across a topic.
- A stable image ID key keeps related records on the same partition while partition routing remains unchanged.
- Changing the partition count can change key routing; plan around ordering requirements.
- In the standard consumer-group model, each partition belongs to one consumer in a group at a time.
- Extra consumers beyond the partition count remain idle.
- Concurrent work inside a consumer must preserve any ordering the application requires.

### 6.3 Offsets and delivery semantics

A committed offset records the next position to consume after recovery.

| Semantics | Typical approach | Consequence |
| --- | --- | --- |
| At-most-once | Commit before processing | A crash can lose work |
| At-least-once | Persist the result, then commit | A crash can repeat work |
| Exactly-once within a defined boundary | Coordinate input progress and outputs transactionally | Guarantees depend on which systems participate |

For PostgreSQL side effects, use at-least-once delivery with idempotent writes. Kafka producer idempotence does not deduplicate arbitrary database operations or repeated business requests.

### 6.4 Replay, lag, and rebalancing

- **Replay:** Read retained events again to recover from bugs or rebuild outputs.
- **Lag:** The gap between the log end and a group's committed position; interpret it alongside event age and processing latency.
- **Rebalancing:** Reassign partitions when group membership or topic topology changes.

Keep processing bounded so consumers continue polling within their configured interval. Handle partition revocation before committing or completing work owned by another consumer. Use a task queue when scheduling, priorities, and delayed jobs are the main requirement.

## 7. Kafka Producers, Consumers, and Retries

### 7.1 Publish an event and check delivery

Prerequisites: Kafka is running at `localhost:9092`, and `image.uploaded.v1` exists. The IDs below are illustrative; a real producer uses the verified database row and persists a stable event ID in its outbox.

```python
from datetime import datetime, timezone
from uuid import uuid4

from confluent_kafka import Producer
from schemas import ImageUploadedEvent

producer = Producer({
    "bootstrap.servers": "localhost:9092",
    "enable.idempotence": True,
    "acks": "all",
})


def publish_confirmed(topic: str, key: str, payload: bytes) -> None:
    failures = []

    def delivered(error, message):
        if error is not None:
            failures.append(str(error))

    producer.produce(topic, key=key, value=payload, callback=delivered)
    remaining = producer.flush(timeout=10)
    if remaining or failures:
        raise RuntimeError(f"Delivery unconfirmed: {remaining=}, {failures=}")


event = ImageUploadedEvent(
    event_id=uuid4(), event_type="image.uploaded", event_version=1,
    occurred_at=datetime.now(timezone.utc),
    image_id=uuid4(), user_id=uuid4(),
    object_key="raw/image-456.jpg",
    content_type="image/jpeg", size_bytes=850_000,
)
publish_confirmed(
    "image.uploaded.v1", str(event.image_id),
    event.model_dump_json().encode("utf-8"),
)
```

`produce()` is asynchronous; the callback reports delivery and `flush()` waits for queued work. This single-record example checks both callback failures and messages left queued. For throughput, batch sends and track acknowledgements instead of flushing every record. See the [Confluent Python client guide](https://docs.confluent.io/kafka-clients/python/current/overview.html).

`acks=all` works with the topic's replication and minimum in-sync replica settings; it does not configure replication itself.

### 7.2 Consume, validate, persist, then commit

This is a **consumer template**. Implement `process_idempotently` and `quarantine_confirmed` before running it:

- `process_idempotently(event)` must return only after its database result is durable.
- `quarantine_confirmed(message, reason)` must return only after the rejected record is durably stored; raise on failure.

```python
from confluent_kafka import Consumer, KafkaException
from pydantic import ValidationError
from schemas import ImageUploadedEvent

consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "image-ingestion-v1",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
    "enable.auto.offset.store": False,
})
consumer.subscribe(["image.uploaded.v1"])

try:
    while True:
        message = consumer.poll(timeout=1.0)
        if message is None:
            continue
        if message.error():
            raise KafkaException(message.error())

        payload = message.value()
        if payload is None:
            quarantine_confirmed(message, "EMPTY_PAYLOAD")
        else:
            try:
                event = ImageUploadedEvent.model_validate_json(payload)
            except ValidationError:
                quarantine_confirmed(message, "INVALID_EVENT")
            else:
                process_idempotently(event)

        # Both success and confirmed quarantine allow progress.
        consumer.commit(message=message, asynchronous=False)
finally:
    consumer.close()
```

An uncaught processing or quarantine failure stops this template without committing the current record. Restarting resumes from the last committed position; production recovery also needs bounded retries and alerts. The message-based commit stores its offset plus one. See the [Confluent consumer API](https://docs.confluent.io/platform/7.7/clients/confluent-kafka-python/html/index.html).

### 7.3 Classify failures and bound retries

| Failure | Response |
| --- | --- |
| Temporary database or storage outage | Retry with exponential backoff and jitter |
| Malformed event or unsupported version | Quarantine with an error code |
| Corrupt image | Record a permanent processing failure |
| Retry limit exceeded | Send to a dead-letter queue (DLQ) and alert |

- Preserve the original event ID and record attempt count.
- Confirm retry or DLQ publication before committing the source record.
- Expect duplicates if publishing succeeds but the source commit fails.
- Make quarantine writes idempotent, for example by source topic, partition, and offset.
- Kafka does not automatically delay records in a retry topic; use a scheduler or explicit delay logic.
- Moving one event to a retry topic can let later events overtake it; account for entity ordering.

### 7.4 Prevent missing events with a transactional outbox

A database commit and Kafka publish are separate operations. If the database succeeds and publishing fails, work can remain unprocessed.

```sql
CREATE TABLE outbox_events (
    event_id UUID PRIMARY KEY,
    topic TEXT NOT NULL,
    event_key TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at TIMESTAMPTZ
);
```

1. Begin a database transaction.
2. Update the image from `pending_upload` to `uploaded`.
3. Insert the corresponding outbox event only if the transition succeeds.
4. Commit both changes together.
5. Have a relay publish the stored payload and mark it published after acknowledgement.

The relay can crash between publication and marking the row. Consumers must therefore tolerate duplicate events. Retrying upload confirmation must also avoid creating another logical event for the same transition.

## 8. End-to-End Ingestion and Batch Exports

### 8.1 Upload and processing lifecycle

```text
pending_upload → uploaded → processing → completed
                                ↓
                              failed
```

1. Create the image record and issue its upload URL.
2. Verify the uploaded object and bind its immutable identity.
3. Commit the `uploaded` state and outbox event together.
4. Publish the event through the relay.
5. Validate the event, retrieve the object, and decode the image.
6. Validate measured dimensions, preprocess, and run inference if required.
7. Commit the result and completion state together.
8. Commit the Kafka offset after the result is durable.

Use an explicit retry transition and lease recovery for abandoned `processing` states. A status field alone does not prevent duplicate work or recover crashed workers.

### 8.2 Store results idempotently

```sql
CREATE TABLE image_processing_results (
    image_id UUID NOT NULL REFERENCES images(image_id),
    pipeline_version TEXT NOT NULL,
    quality_label TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (image_id, pipeline_version)
);
```

Use `INSERT ... ON CONFLICT (image_id, pipeline_version) DO NOTHING` when the first completed result should win. Perform that insert and the image-state update in one transaction.

- The unique constraint remains safe when two workers race.
- A prior `result_exists()` check is only an optimization.
- A new pipeline version can produce a separate result.
- Repeated inference calls may still happen; result deduplication does not prevent external side effects or repeated compute costs.

### 8.3 Separate worker stages

**Pseudocode:** These helpers represent application-specific storage, image decoding, inference, and database operations.

```python
def process_idempotently(event):
    version = "quality-v3"
    if result_exists(event.image_id, version):
        return

    image_record = load_and_authorize_image(event)
    source = download_exact_object(image_record)
    verify_checksum(source, image_record)
    image = decode_with_size_limits(source)
    prediction = run_inference(preprocess(image))

    save_result_and_complete_transactionally(
        image_id=event.image_id,
        pipeline_version=version,
        prediction=prediction,
        width=image.width,
        height=image.height,
    )
```

The final database transaction enforces uniqueness even if the initial existence check races. Apply bounded resource use and temporary-file cleanup inside the relevant helpers.

### 8.4 Export analytics data incrementally

Use half-open intervals (`start <= timestamp < end`) so adjacent windows do not overlap. This example exports a fixed historical window with a server-side cursor and a new output directory:

```python
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg

start = datetime(2026, 10, 29, tzinfo=timezone.utc)
end = datetime(2026, 10, 30, tzinfo=timezone.utc)
output = Path("export-2026-10-29-run-001")
output.mkdir(exist_ok=False)

with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
    with connection.cursor(name="results_export") as cursor:
        cursor.execute(
            """SELECT image_id::text, pipeline_version, quality_label,
                      confidence, width, height, completed_at
               FROM image_processing_results
               WHERE completed_at >= %s AND completed_at < %s""",
            (start, end),
        )
        columns = [column.name for column in cursor.description]
        part = 0
        while rows := cursor.fetchmany(10_000):
            frame = pd.DataFrame(rows, columns=columns)
            frame.to_parquet(output / f"part-{part:05d}.parquet", index=False)
            part += 1
```

- Upload completed files under a date and run-specific prefix.
- Publish a manifest only after all parts are durable; readers consume published runs.
- Standardize the export schema across parts.
- Handle late commits with an overlap-and-deduplicate strategy, reconciliation, or change data capture (CDC). Timestamp windows alone do not guarantee complete extraction.
- Monitor long-running exports so database snapshots do not remain open unnecessarily.

## 9. Reliability, Monitoring, and Reproducibility

### 9.1 Recovery and idempotency

| Failure point | Recovery mechanism |
| --- | --- |
| Upload completes but confirmation never arrives | Reconciliation and orphan-object cleanup |
| Database commits but Kafka is unavailable | Retry pending outbox records |
| Worker crashes before writing a result | Replay uncommitted events |
| Worker writes a result but crashes before offset commit | Unique result key prevents duplicate stored output |
| Worker leaves a claimed row in `processing` | Lease expiry and controlled reassignment |
| Invalid data repeatedly fails | Durable quarantine and a repair/replay workflow |

Treat replay as a normal operation. Preserve source versions and stable identifiers so reruns are explainable.

### 9.2 Backpressure and capacity

Backpressure occurs when input arrives faster than downstream processing can finish.

- Bound in-memory queues, downloads, and concurrent inference requests.
- Batch database writes where appropriate.
- Scale consumers within useful partition parallelism.
- Rate-limit ingestion when downstream capacity is exhausted.
- Monitor database pools, storage throttling, worker memory, and event age.
- Investigate slow stages before increasing partition counts or concurrency.

### 9.3 Metrics, logs, traces, and alerts

| Signal | Capture |
| --- | --- |
| Metrics | Throughput, latency, validation failures, retries, DLQ size, consumer lag |
| Logs | Event ID, image ID, pipeline version, stage, and error code |
| Traces | Time spent across API, storage, messaging, worker, and database calls |
| Reconciliation | Unique inputs versus completed, quarantined, and pending records |

Alert on sustained processing delay, rising failure rates, stale outbox records, and missing expected output. Avoid logging payloads, credentials, or presigned URLs.

### 9.4 Access control and data lifecycle

- Authenticate users and verify image ownership server-side.
- Give each service only the database, topic, and bucket permissions it needs.
- Use TLS, private storage, and managed credentials.
- Coordinate deletion across raw objects, results, exports, and retained versions.
- Keep lifecycle rules consistent with replay and dataset-retention requirements.

### 9.5 Lineage and AI dataset reproducibility

Record which inputs, transformations, and model versions produced each output. A dataset manifest can identify exact artifacts:

```json
{
  "dataset": "image-quality-training",
  "dataset_version": "2026-10-30-v1",
  "pipeline_version": "quality-v3",
  "code_revision": "3a4f21c",
  "schema_version": 1,
  "model_version": "quality-model-v2",
  "input_manifest": "manifests/source-objects-v1.jsonl",
  "output_manifest": "manifests/parquet-parts-v1.jsonl",
  "accepted_records": 9842,
  "rejected_records": 37
}
```

The referenced manifests should list exact object versions and checksums, not only mutable prefixes.

For AI datasets:

- Split by user, source, or time when related examples could leak across train/test sets.
- Fit preprocessing statistics on the training split only.
- Keep labels and features valid for the prediction time.
- Version preprocessing with the model to reduce training-serving differences.

## 10. Revision and Practice Project

### 10.1 Explain the system in one minute

- Store image bytes in object storage and metadata in PostgreSQL.
- Verify uploads and commit an outbox event with the database update.
- Publish events to Kafka using a stable image ID key.
- Validate events and image contents in Python workers.
- Save versioned results idempotently before committing offsets.
- Quarantine bad inputs, bound retries, and monitor processing delay.
- Export curated Parquet datasets with reproducible manifests.

### 10.2 Common interview questions

| Question | Concise answer |
| --- | --- |
| Why use object storage with SQL? | Object storage holds large files; SQL manages relationships, constraints, and transactional state. |
| Batch or streaming? | Use batch for bounded scheduled work and streaming when results must follow incoming events quickly. |
| CSV or Parquet? | CSV is easy to exchange; Parquet preserves types and supports efficient analytical column reads. |
| Does Kafka guarantee global ordering? | No. Ordering is per partition, and application concurrency must preserve any required processing order. |
| What if a worker crashes after saving its result? | The event may repeat; a unique result key prevents duplicate stored output. |
| What does the outbox solve? | It commits the business change and its pending event together, allowing reliable publication retries. |
| Does producer idempotence cover database writes? | No. Database effects need their own idempotency and transaction design. |
| Why validate in Python and SQL? | Python checks incoming contracts; SQL constraints protect persisted relational data. |
| How do you handle late data? | Define a lateness policy, reprocess overlapping windows safely, and reconcile outputs. |
| How do you reproduce a dataset? | Preserve exact input versions, code, schemas, model versions, and published output manifests. |
