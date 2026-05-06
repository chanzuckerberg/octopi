# Remote Training: Known Issues & Notes

## `aiobotocore` / `aiohttp` Event Loop Warnings

### Symptom

During training on the HPC cluster with remote (S3-backed) tomograms, you may see repeated warnings like:

```
Task exception was never retrieved
RuntimeError: Task <Task pending ...> got Future <Task pending ...> attached to a different loop
```

These appear in stderr alongside normal training progress output (e.g., `Epoch 40/1000, avg_train_loss: ...`).

### Cause

**This is non-fatal.** Training continues normally.

The error originates from `SmartCacheDataset` in `octopi/datasets/generators.py`, which uses 8 background worker threads (`num_init_workers=8, num_replace_workers=8`) to load remote tomograms into cache. The chain is:

1. Each worker thread creates its own `asyncio` event loop
2. CoPick reads tomograms from S3 via `fsspec` → `s3fs` → `aiobotocore`
3. When a worker closes its S3 connection, `aiobotocore` tries to clean up async resources that were created in a *different* thread's event loop
4. Python raises `RuntimeError: Future attached to a different loop`

This is a known incompatibility between `aiohttp >= 3.9` and `aiobotocore` in multi-threaded contexts.

### Confirming It's Non-Fatal

- Epoch counter continues advancing normally
- Loss and validation metrics are logged correctly
- Model checkpoints save at the expected intervals
- The exceptions appear as "never retrieved" background task errors, meaning they do not propagate to the main training thread

### Fixes

**Option 1: Pin compatible package versions (cleanest)**

```bash
pip install "aiohttp<3.9" "aiobotocore>=2.5"
```

**Option 2: Reduce SmartCache worker threads**

In `octopi/datasets/generators.py`, reduce the worker counts to decrease the number of concurrent threads competing over event loops (at the cost of slower cache refresh):

```python
# CopickDataModule.create() ~line 141
self.train_ds = SmartCacheDataset(
    ...
    num_init_workers=2,
    num_replace_workers=2,
)
```

**Option 3: Do nothing**

If the HPC environment is not easily modified and training is proceeding correctly, the warnings are purely cosmetic. They do not affect model quality or convergence.

### Environment

- HPC cluster: `hpc/projects/group.czii`
- Python 3.11
- The error is triggered during `SmartCacheDataset.update_cache()`, called once per epoch in `ModelTrainer.train()`
- Only occurs when training against **remote/S3-backed CoPick projects** — local file-based projects do not trigger this
