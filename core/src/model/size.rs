/// Represents the size of a 3D grid or dispatch region.
///
/// This struct is commonly used to specify the dimensions (width, height, depth)
/// for compute shader dispatches or buffer allocations.
#[derive(Clone, Debug)]
pub struct Size {
    /// The width (X dimension).
    pub width: usize,
    /// The height (Y dimension).
    pub height: usize,
    /// The depth (Z dimension).
    pub depth: usize,
}

impl Size {
    /// Creates a new `Size` from a value that can be converted to a `Size`.
    ///
    /// # Arguments
    /// * `size` - The value to convert to a `Size`
    ///
    /// # Returns
    /// A new `Size` instance.
    pub fn new<T>(size: T) -> Self 
    where
        T: Into<Self>
    {
        size.into()
    }

    /// Returns the total number of elements in the 3D region (width * height * depth).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.width * self.height * self.depth
    }

    /// Returns true if the size is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.width == 0 && self.height == 0 && self.depth == 0
    }
}

impl From<usize> for Size {
    /// Creates a `Size` from a single value, setting width to the value and height and depth to 1.
    fn from(size: usize) -> Self {
        Self { width: size, height: 1, depth: 1 }
    }
}

impl From<(usize, usize)> for Size {
    /// Creates a `Size` from a tuple (width, height), setting depth to 1.
    fn from(size: (usize, usize)) -> Self {
        Self { width: size.0, height: size.1, depth: 1 }
    }
}

impl From<(usize, usize, usize)> for Size {
    /// Creates a `Size` from a tuple (width, height, depth).
    fn from(size: (usize, usize, usize)) -> Self {
        Self { width: size.0, height: size.1, depth: size.2 }
    }
}
