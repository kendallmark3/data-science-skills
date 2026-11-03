# Examples

**Seeded dataset.** 11 planted defects; see `fixtures/SEEDS.md`.

```bash
dsk profile fixtures/datasets/customers.csv \
  --context fixtures/datasets/customers.context.yaml --out output/profiling-customers
```

Expected: `NOT_FIT_AS_IS`, driven by train/test contamination and `days_since_cancellation` leakage. Four visuals are produced.

**Clean control.** Nothing is planted, so the run must not invent defects.

```bash
dsk profile fixtures/datasets/customers_clean.csv \
  --context fixtures/datasets/customers_clean.context.yaml --out output/profiling-clean
```

Expected: `FIT_FOR_USE`, at most informational findings, and no charts.

**Stop condition.**

```bash
dsk profile fixtures/datasets/unreadable.csv --out output/profiling-stopped   # exit code 2
```

**Typical prompt to Claude:** "Profile `data/orders.parquet` before we train the late-delivery model. `is_late` is the label and `order_id` is the key."
