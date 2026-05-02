use serde::{Deserialize, Serialize};

pub mod async_mpi;
pub mod circle;
pub mod termination_detector;
pub mod walker;

pub use circle::Circle;

#[derive(Debug, Serialize, Deserialize)]
enum WorkResponse {
    Work(Vec<Vec<u8>>),
    Reject,
}

#[derive(Debug)]
pub enum CircleError {
    Mpi(ferrompi::Error),
    Serialization(postcard::Error),
}

impl std::fmt::Display for CircleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircleError::Mpi(err) => write!(f, "MPI error: {err}"),
            CircleError::Serialization(err) => write!(f, "Serialization error: {err}"),
        }
    }
}

impl std::error::Error for CircleError {}

impl From<ferrompi::Error> for CircleError {
    fn from(value: ferrompi::Error) -> Self {
        Self::Mpi(value)
    }
}

impl From<postcard::Error> for CircleError {
    fn from(value: postcard::Error) -> Self {
        Self::Serialization(value)
    }
}

impl From<CircleError> for std::io::Error {
    fn from(value: CircleError) -> Self {
        match value {
            CircleError::Mpi(err) => std::io::Error::new(std::io::ErrorKind::Other, err),
            CircleError::Serialization(err) => std::io::Error::new(std::io::ErrorKind::Other, err),
        }
    }
}
