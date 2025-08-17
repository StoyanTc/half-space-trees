use half_space_trees::HalfSpaceTrees;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn basic_separation() {
    let bounds = vec![(0.0, 1.0); 3];
    let mut rng = StdRng::seed_from_u64(7);
    let mut forest = HalfSpaceTrees::new(50, 10, &bounds, &mut rng);

    // Train on clustered normal data around 0.2
    for i in 0..5000 {
        let x = vec![0.2 + 0.01 * ((i % 7) as f64), 0.22, 0.18];
        forest.insert(&x);
        if i % 250 == 0 {
            forest.decay(0.995);
        }
    }

    let normal = vec![0.21, 0.2, 0.19];
    let outlier = vec![0.95, 0.95, 0.95];
    let s_n = forest.score(&normal);
    let s_o = forest.score(&outlier);
    assert!(
        s_o > s_n,
        "outlier should have higher score: s_o={s_o}, s_n={s_n}"
    );
}
