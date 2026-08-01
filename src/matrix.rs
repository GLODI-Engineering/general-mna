use std::fmt;
use std::ops::{Index, IndexMut};

/// A compact row-major dense matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Clone> Matrix<T> {
    /// Creates a matrix filled with `value`.
    pub fn filled(rows: usize, cols: usize, value: T) -> Self {
        Self {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }
}

impl<T> Matrix<T> {
    /// Creates a matrix from row-major data.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, MatrixShapeError> {
        if data.len() != rows * cols {
            return Err(MatrixShapeError {
                rows,
                cols,
                actual: data.len(),
            });
        }
        Ok(Self { rows, cols, data })
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the row-major values.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns an iterator over the values.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Maps every value while preserving the shape.
    pub fn map<U, F>(&self, f: F) -> Matrix<U>
    where
        F: FnMut(&T) -> U,
    {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(f).collect(),
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(
            row < self.rows && col < self.cols,
            "matrix index out of bounds"
        );
        &self.data[row * self.cols + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(
            row < self.rows && col < self.cols,
            "matrix index out of bounds"
        );
        &mut self.data[row * self.cols + col]
    }
}

/// Error returned when row-major data has the wrong length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixShapeError {
    rows: usize,
    cols: usize,
    actual: usize,
}

impl fmt::Display for MatrixShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "a {}x{} matrix needs {} values, received {}",
            self.rows,
            self.cols,
            self.rows * self.cols,
            self.actual
        )
    }
}

impl std::error::Error for MatrixShapeError {}
