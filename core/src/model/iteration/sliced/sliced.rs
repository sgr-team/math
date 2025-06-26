use std::ops::Range;

/// A trait that provides range access for parameters.
///
/// This trait allows getting and setting the range of data to work with.
pub trait Sliced {
    /// Returns the current range.
    fn range(&self) -> Range<usize>;

    /// Updates the range to work with.
    ///
    /// # Arguments
    /// * `range` - The range to work with
    ///
    /// # Panics
    /// Panics if the range end exceeds the total length
    fn set_range(&mut self, range: Range<usize>);
}

impl<T> Sliced for Range<T> 
where
    T: Clone + Into<usize> + From<usize>
{
    fn range(&self) -> Range<usize> { 
        self.start.clone().into()..self.end.clone().into() 
    }

    fn set_range(&mut self, range: Range<usize>) { 
        *self = range.start.clone().into()..range.end.clone().into(); 
    }
}
