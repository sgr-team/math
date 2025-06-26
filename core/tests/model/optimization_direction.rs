#[test]
fn compare() {
    assert_eq!(OptimizationDirection::Minimize.compare(&1.0, &2.0), Ordering::Less);
    assert_eq!(OptimizationDirection::Minimize.compare(&2.0, &1.0), Ordering::Greater);
    assert_eq!(OptimizationDirection::Minimize.compare(&2.0, &2.0), Ordering::Equal);

    assert_eq!(OptimizationDirection::Maximize.compare(&1.0, &2.0), Ordering::Greater);
    assert_eq!(OptimizationDirection::Maximize.compare(&2.0, &1.0), Ordering::Less);
    assert_eq!(OptimizationDirection::Maximize.compare(&2.0, &2.0), Ordering::Equal);
}

#[test]
fn sort() {
    assert_eq!(
        vec![1.0, 3.0, 2.0].sort_by(|a, b| OptimizationDirection::Minimize.compare(a, b)),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        vec![1.0, 3.0, 2.0].sort_by(|a, b| OptimizationDirection::Maximize.compare(a, b)),
        vec![3.0, 2.0, 1.0]
    );
}