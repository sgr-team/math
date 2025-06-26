/// Represents the size of a slice, either as a fixed count or a proportional value.
///
/// Proportional values represent parts of available space to distribute. For example,
/// if you have three slices with proportional values 1, 2, and 1, they will get 25%, 50%,
/// and 25% of the available space respectively.
///
/// # Examples
/// ```
/// use sgrmath_core::IterationSize;
///
/// // Fixed count of 100 elements
/// let fixed = IterationSize::Count(100);
///
/// // Proportional value of 2 (will get twice as much space as a value of 1)
/// let proportional = IterationSize::Proportional(2.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub enum IterationSize {
    /// Fixed number of elements
    Count(usize),
    /// Proportional value (must be positive)
    Proportional(f32),
}

impl Into<IterationSize> for usize {
    fn into(self) -> IterationSize {
        IterationSize::Count(self)
    }
}
