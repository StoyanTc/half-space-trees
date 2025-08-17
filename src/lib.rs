//! Minimal, practical Half‑Space Trees (HST) for streaming anomaly detection in Rust
//! ----------------------------------------------------------------------------
//! This is a compact, dependency‑light sketch you can drop into your project.
//! It implements:
//!   * HalfSpaceTree with random axis‑aligned splits to fixed max_depth
//!   * Incremental updates via `insert(&x)` that accumulate decayed mass per node
//!   * A simple decay API you can call periodically to handle concept drift
//!   * A forest wrapper that averages scores across trees
//!   * A reasonable (but simplified) scoring function suitable to start tuning
//!
//! # Design notes
//! HST literature (and river's implementation) maintains mass in subspaces over a
//! sliding window. Here we keep an exponentially decayed mass per node (call
//! `decay(alpha)` periodically with alpha in (0,1], e.g. 0.999 each tick or per N items).
//! The `score` returns higher values for sparser regions (lower mass, deeper leaves).
//!
//! This is intentionally small so you can adapt:
//!   - Swap in time‑based decay
//!   - Swap the scoring for your preferred formulation
//!   - Change split strategy (uniform inside bounds, jitter, etc.)
//!
//! # Example
//! ```
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//!
//! // Feature space: 4 dims with observed ranges (min, max)
//! let bounds = vec![
//!     (0.0, 1.0),    // e.g., normalized latency
//!     (0.0, 1.0),    // normalized payload size
//!     (0.0, 1.0),    // method one‑hot projection
//!     (0.0, 1.0),    // status class
//! ];
//! let mut rng = StdRng::seed_from_u64(42);
//! let mut forest = HalfSpaceTrees::new(25, 12, &bounds, &mut rng);
//!
//! // Stream some normal points
//! for i in 0..5000 {
//!     let x = vec![
//!         (i as f64 % 100.0) / 100.0,
//!         0.4,
//!         0.2,
//!         0.5,
//!     ];
//!     forest.insert(&x);
//!     if i % 200 == 0 { forest.decay(0.995); } // periodic decay
//! }
//!
//! // Score a new request
//! let suspicious = vec![0.99, 0.99, 0.99, 0.01];
//! let score = forest.score(&suspicious);
//! println!("anomaly score = {score:.4}");
//! ```

use rand::Rng;
use rand::distr::{Distribution, Uniform};

pub type FeatureVector = [f64];

#[derive(Debug)]
pub struct HalfSpaceTrees {
    trees: Vec<HalfSpaceTree>,
}

impl HalfSpaceTrees {
    /// Create a forest of `n_trees` trees of depth `max_depth`.
    /// `bounds` is a per‑dimension (min,max) range used to generate random splits.
    pub fn new<R: Rng + ?Sized>(
        n_trees: usize,
        max_depth: u32,
        bounds: &[(f64, f64)],
        rng: &mut R,
    ) -> Self {
        let trees = (0..n_trees)
            .map(|_| HalfSpaceTree::new(max_depth, bounds, rng))
            .collect();
        Self { trees }
    }

    /// Insert a point with unit weight (after any global decay you apply externally).
    pub fn insert(&mut self, x: &FeatureVector) {
        for t in &mut self.trees {
            t.insert(x);
        }
    }

    /// Multiply all node masses by `alpha` (0,1]. Call periodically to handle drift.
    pub fn decay(&mut self, alpha: f64) {
        for t in &mut self.trees {
            t.decay(alpha);
        }
    }

    /// Average score across trees
    pub fn score(&self, x: &FeatureVector) -> f64 {
        let mut s = 0.0;
        for t in &self.trees {
            s += t.score(x);
        }
        s / (self.trees.len() as f64)
    }
}

#[derive(Debug)]
pub struct HalfSpaceTree {
    root: Node,
    max_depth: u32,
    n_dims: usize,
}

impl HalfSpaceTree {
    pub fn new<R: Rng + ?Sized>(max_depth: u32, bounds: &[(f64, f64)], rng: &mut R) -> Self {
        assert!(!bounds.is_empty(), "bounds must not be empty");
        let n_dims = bounds.len();
        let root = Node::randomized(0, max_depth, bounds, rng);
        Self {
            root,
            max_depth,
            n_dims,
        }
    }

    pub fn insert(&mut self, x: &FeatureVector) {
        assert_eq!(x.len(), self.n_dims);
        self.root.insert(x);
    }

    pub fn decay(&mut self, alpha: f64) {
        self.root.decay(alpha);
    }

    pub fn score(&self, x: &FeatureVector) -> f64 {
        assert_eq!(x.len(), self.n_dims);
        self.root.score(x, self.max_depth)
    }
}

#[derive(Debug)]
struct Node {
    // Split definition (valid for internal nodes)
    split_dim: usize,
    split_val: f64,
    // Tree structure
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    // Stats
    depth: u32,
    mass: f64, // exponentially decayed count
}

impl Node {
    fn randomized<R: Rng + ?Sized>(
        depth: u32,
        max_depth: u32,
        bounds: &[(f64, f64)],
        rng: &mut R,
    ) -> Self {
        // On construction we create a *full* binary tree to max_depth with random splits.
        let n_dims = bounds.len();
        let split_dim = rng.random_range(0..n_dims);
        let (lo, hi) = bounds[split_dim];
        let between = Uniform::try_from(lo..hi).unwrap();
        let split_val = between.sample(rng);

        if depth == max_depth {
            return Self {
                split_dim,
                split_val,
                left: None,
                right: None,
                depth,
                mass: 0.0,
            };
        }
        let left = Box::new(Node::randomized(depth + 1, max_depth, bounds, rng));
        let right = Box::new(Node::randomized(depth + 1, max_depth, bounds, rng));
        Self {
            split_dim,
            split_val,
            left: Some(left),
            right: Some(right),
            depth,
            mass: 0.0,
        }
    }

    fn insert(&mut self, x: &FeatureVector) {
        // Update local mass then descend
        self.mass += 1.0;
        match (&mut self.left, &mut self.right) {
            (Some(l), Some(r)) => {
                if x[self.split_dim] < self.split_val {
                    l.insert(x);
                } else {
                    r.insert(x);
                }
            }
            _ => {}
        }
    }

    fn decay(&mut self, alpha: f64) {
        self.mass *= alpha;
        if let Some(l) = &mut self.left {
            l.decay(alpha);
        }
        if let Some(r) = &mut self.right {
            r.decay(alpha);
        }
    }

    fn score(&self, x: &FeatureVector, max_depth: u32) -> f64 {
        // Traverse to a leaf (or max depth) and compute a rarity score from leaf mass and depth.
        let mut node = self;
        loop {
            match (&node.left, &node.right) {
                (Some(l), Some(r)) => {
                    node = if x[node.split_dim] < node.split_val {
                        l
                    } else {
                        r
                    };
                }
                _ => break,
            }
        }
        let depth_factor = 1.0 + (max_depth - node.depth) as f64 / (max_depth as f64 + 1.0);
        // Smooth rarity: small mass -> high score; clamp to avoid division blow‑ups.
        let rarity = 1.0 / (1.0 + node.mass.max(0.0));
        rarity * depth_factor
    }
}
