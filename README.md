# Half Space Trees

Minimal, practical Half‑Space Trees (HST) for streaming anomaly detection in Rust
----------------------------------------------------------------------------
This is a compact, dependency‑light sketch you can drop into your project.
It implements:
  * HalfSpaceTree with random axis‑aligned splits to fixed max_depth
  * Incremental updates via `insert(&x)` that accumulate decayed mass per node
  * A simple decay API you can call periodically to handle concept drift
  * A forest wrapper that averages scores across trees
  * A reasonable (but simplified) scoring function suitable to start tuning

## Design notes
HST literature (and river's implementation) maintains mass in subspaces over a
sliding window. Here we keep an exponentially decayed mass per node (call
`decay(alpha)` periodically with alpha in (0,1], e.g. 0.999 each tick or per N items).
The `score` returns higher values for sparser regions (lower mass, deeper leaves).

This is intentionally small so you can adapt:
  - Swap in time‑based decay
  - Swap the scoring for your preferred formulation
  - Change split strategy (uniform inside bounds, jitter, etc.)

## Example
```rust
use rand::SeedableRng;
use rand::rngs::StdRng;

// Feature space: 4 dims with observed ranges (min, max)
let bounds = vec![
    (0.0, 1.0),    // e.g., normalized latency
    (0.0, 1.0),    // normalized payload size
    (0.0, 1.0),    // method one‑hot projection
    (0.0, 1.0),    // status class
];
let mut rng = StdRng::seed_from_u64(42);
let mut forest = HalfSpaceForest::new(25, 12, &bounds, &mut rng);

// Stream some normal points
for i in 0..5000 {
    let x = vec![
        (i as f64 % 100.0) / 100.0,
        0.4,
        0.2,
        0.5,
    ];
    forest.insert(&x);
    if i % 200 == 0 { forest.decay(0.995); } // periodic decay
}

// Score a new request
let suspicious = vec![0.99, 0.99, 0.99, 0.01];
let score = forest.score(&suspicious);
println!("anomaly score = {score:.4}");
```
