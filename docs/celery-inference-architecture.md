# Celery Batch Inference Architecture

## Goal

Use Celery workers to run the existing image inference pipeline over batches of
filesystem image URLs.

The core inference requirement is:

- Producers submit `file://` image URLs.
- URLs are accumulated in Redis.
- A Celery worker drains up to 50 URLs at a time.
- The worker reads the image files, calls `InferencePipeline.predict_many(...)`,
  and writes per-URL result state back to Redis.

This design keeps model inference batched while still letting producers submit
single image URLs independently.

## High-Level Shape

There are two separate queues:

```text
Redis URL buffer
  inference:url_queue
  contains file:// image URLs waiting for inference

Celery broker queue
  contains drain_url_batch task messages
```

Celery does not track the state of each URL inside a batch. Celery only knows
about the batch-draining task. Per-image state must be stored separately in
Redis using an application-level key.

```text
producer
  -> stores per-URL state as queued
  -> pushes URL to Redis URL buffer
  -> schedules drain_url_batch if no drainer is active

Celery worker
  -> runs drain_url_batch
  -> drains up to MAX_BATCH_SIZE URLs
  -> runs predict_many
  -> writes per-URL succeeded/failed state
  -> continues until queue is empty
```

## Redis Keys

Recommended keys:

```text
inference:url_queue
  Redis list containing file:// image URLs.

inference:url:<sha256(url)>
  JSON record for the latest known state of a specific URL.

inference:drainer_lock
  Redis lock that prevents many duplicate drainers from running at once.
```

Use a stable hash of the full URL for lookup:

```python
import hashlib


def url_key(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return f"inference:url:{digest}"
```

If the same URL may be submitted more than once and each submission must be
tracked independently, add job IDs later:

```text
inference:job:<job_id>
inference:url_latest:<sha256(url)> -> <job_id>
```

For the current design, a URL hash key is enough if querying "latest state for
this URL" is acceptable.

## URL Rules

Only explicit filesystem URLs should be accepted:

```text
file:///home/lasan/Dev/mark/mcq2.jpg
file://localhost/home/lasan/Dev/mark/mcq2.jpg
```

Reject:

```text
http://example.com/image.jpg
https://example.com/image.jpg
file://other-host/path/image.jpg
/plain/local/path.jpg
```

This keeps the worker behavior unambiguous: the producer and worker must share
the same filesystem path.

## State Lifecycle

Use a small set of per-URL statuses:

```text
queued
processing
succeeded
failed
```

Optional states for API/query layers:

```text
not_found
expired
```

### Queued Record

```json
{
  "url": "file:///home/lasan/Dev/mark/mcq2.jpg",
  "status": "queued",
  "submitted_at": "2026-07-02T10:15:30Z",
  "started_at": null,
  "finished_at": null,
  "batch_task_id": null,
  "result": null,
  "error": null
}
```

### Processing Record

```json
{
  "url": "file:///home/lasan/Dev/mark/mcq2.jpg",
  "status": "processing",
  "submitted_at": "2026-07-02T10:15:30Z",
  "started_at": "2026-07-02T10:15:32Z",
  "finished_at": null,
  "batch_task_id": "a9f4d6c2-...",
  "result": null,
  "error": null
}
```

### Succeeded Record

```json
{
  "url": "file:///home/lasan/Dev/mark/mcq2.jpg",
  "status": "succeeded",
  "submitted_at": "2026-07-02T10:15:30Z",
  "started_at": "2026-07-02T10:15:32Z",
  "finished_at": "2026-07-02T10:15:35Z",
  "batch_task_id": "a9f4d6c2-...",
  "result": [
    {
      "question": 1,
      "answer": "1",
      "crossed_options": ["1"],
      "scribble_options": ["4"],
      "segments": []
    }
  ],
  "error": null
}
```

### Failed Record

```json
{
  "url": "file:///home/lasan/Dev/mark/missing.jpg",
  "status": "failed",
  "submitted_at": "2026-07-02T10:15:30Z",
  "started_at": "2026-07-02T10:15:32Z",
  "finished_at": "2026-07-02T10:15:33Z",
  "batch_task_id": "a9f4d6c2-...",
  "result": null,
  "error": "File not found: /home/lasan/Dev/mark/missing.jpg"
}
```

## Producer Flow

The producer is responsible for pushing URLs into the Redis URL buffer and
triggering a drainer task.

Recommended producer behavior:

```text
1. Validate that the submitted URL is file:// and local.
2. Store inference:url:<sha256(url)> with status queued.
3. Push the URL into inference:url_queue.
4. Try to acquire inference:drainer_lock with SET NX EX.
5. If the lock was acquired, enqueue drain_url_batch.
6. If the lock already exists, do not enqueue another drainer.
```

Pseudocode:

```python
def submit_url(url: str) -> str:
    validate_file_url(url)
    key = url_key(url)

    redis.set(
        key,
        json.dumps(
            {
                "url": url,
                "status": "queued",
                "submitted_at": now_iso(),
                "started_at": None,
                "finished_at": None,
                "batch_task_id": None,
                "result": None,
                "error": None,
            }
        ),
        ex=RESULT_TTL_SECONDS,
    )
    redis.rpush("inference:url_queue", url)

    lock_acquired = redis.set(
        "inference:drainer_lock",
        new_lock_token(),
        nx=True,
        ex=DRAINER_LOCK_TTL_SECONDS,
    )

    if lock_acquired:
        drain_url_batch.delay()

    return key
```

## Worker Flow

The Celery worker is a long-running process, but the `drain_url_batch` task is
not an idle forever loop. The task runs only after it is scheduled, drains work,
and exits.

Recommended task behavior:

```text
1. Ensure this task owns or can acquire the drainer lock.
2. Pop up to MAX_BATCH_SIZE=50 URLs from inference:url_queue.
3. If no URLs are available, release the lock and exit.
4. Mark drained URLs as processing.
5. Read files from the file:// URLs.
6. Call InferencePipeline.predict_many(image_bytes, image_ids=image_urls).
7. Write succeeded records for each result.
8. Write failed records for files/inference inputs that failed.
9. Continue draining batches until the queue is empty.
10. Release the lock.
```

This means a single Celery task may process multiple internal batches before it
exits. Each internal batch should call `predict_many` once.

Pseudocode:

```python
MAX_BATCH_SIZE = 50


@celery_app.task(name="inference.drain_url_batch", bind=True)
def drain_url_batch(self):
    pipeline = get_cached_pipeline()

    try:
        while True:
            urls = pop_urls("inference:url_queue", limit=MAX_BATCH_SIZE)

            if not urls:
                return {"drained": 0, "queue_empty": True}

            mark_processing(urls, batch_task_id=self.request.id)

            image_bytes = []
            image_ids = []

            for url in urls:
                try:
                    path = file_url_to_path(url)
                    image_bytes.append(path.read_bytes())
                    image_ids.append(url)
                except Exception as exc:
                    mark_failed(url, exc, batch_task_id=self.request.id)

            if not image_bytes:
                continue

            try:
                results = pipeline.predict_many(image_bytes, image_ids=image_ids)
            except Exception as exc:
                for url in image_ids:
                    mark_failed(url, exc, batch_task_id=self.request.id)
                continue

            for url, result in zip(image_ids, results, strict=True):
                mark_succeeded(
                    url,
                    result=result.json_payload,
                    batch_task_id=self.request.id,
                )
    finally:
        release_drainer_lock()
```

## What Happens If More Than 50 URLs Are Waiting?

The drainer keeps draining while the Redis URL queue has work.

For 120 URLs:

```text
drain_url_batch starts
  batch 1: pop 50, run predict_many
  batch 2: pop 50, run predict_many
  batch 3: pop 20, run predict_many
  queue empty: release lock and exit
```

If using a simpler reschedule-after-each-batch approach, the flow would be:

```text
task 1 drains 50, sees 70 remain, schedules task 2
task 2 drains 50, sees 20 remain, schedules task 3
task 3 drains 20, sees 0 remain, exits
```

For this project, the lock-loop version is preferred because it avoids creating
many Celery tasks while still keeping inference batched.

## Why Use a Drainer Lock?

Without a lock, every producer could enqueue a drainer task. That works, but it
can create many tasks that wake up, find the URL queue empty, and exit.

The lock gives one active drainer ownership of the queue:

```text
producer A submits URL, acquires lock, schedules drainer
producer B submits URL, sees lock exists, does not schedule drainer
active drainer processes both URLs before releasing lock
```

If URLs arrive while the drainer is running, they remain in
`inference:url_queue`. The active drainer should pick them up in its next loop
iteration before releasing the lock.

If the worker crashes while holding the lock, the lock TTL expires. A later
producer can then acquire the lock and schedule a new drainer.

## Querying Result State

Query per-URL state from Redis using the same URL hash key.

```python
def get_url_state(url: str) -> dict | None:
    raw = redis.get(url_key(url))

    if raw is None:
        return None

    return json.loads(raw)
```

Example:

```python
state = get_url_state("file:///home/lasan/Dev/mark/mcq2.jpg")
```

Celery task IDs are not enough for this lookup because one Celery task may
process many URLs. Celery can answer "did this batch task finish?", but the app
needs Redis state records to answer "what happened to this specific URL?"

## Celery Configuration

Use Redis as both broker and result backend:

```python
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "mark_inference",
    broker=REDIS_URL,
    backend=REDIS_URL,
)
```

Recommended Celery settings:

```python
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
```

Use environment variables for runtime configuration:

```text
REDIS_URL=redis://localhost:6379/0
INFERENCE_USE_CPU=0
EMPTY_MODEL_PATH=mcq_empty_classifier.pt
MARK_MODEL_PATH=mcq_mark_type_classifier.pt
MAX_BATCH_SIZE=50
RESULT_TTL_SECONDS=86400
DRAINER_LOCK_TTL_SECONDS=300
```

`INFERENCE_USE_CPU=1` should force CPU inference. By default, keep the existing
pipeline behavior and use CUDA.

## Starting The Worker

Example worker command:

```bash
.venv/bin/celery -A celery_app worker --loglevel=info
```

For a single GPU, prefer one worker process unless memory and throughput testing
shows more processes are safe:

```bash
.venv/bin/celery -A celery_app worker --loglevel=info --concurrency=1
```

## Operational Notes

- The producer and worker must share the same filesystem paths.
- Store only prediction JSON in Redis results, not warped or annotated images.
- Set TTLs on per-URL result records so Redis does not grow forever.
- A bad image should become a per-URL `failed` record and should not stop later
  batches from running.
- A whole-batch inference exception should mark all URLs in that internal batch
  as `failed`.
- The Celery result backend is still useful for debugging batch tasks, but the
  source of truth for image status is `inference:url:<sha256(url)>`.

## Future Extensions

- Add `job_id` if the same URL needs multiple independent submissions.
- Add a lightweight HTTP endpoint for submit/query operations.
- Store large artifacts on disk or object storage and return paths/URLs in the
  Redis result record.
- Add metrics for queue length, batch size, inference latency, and failure rate.
