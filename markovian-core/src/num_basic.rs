use rand::distributions::uniform::{SampleBorrow, SampleUniform};

pub trait Field:
    Clone
    + Copy
    + std::ops::SubAssign<Self>
    + std::ops::AddAssign<Self>
    + SampleUniform
    + SampleBorrow<Self>
    + std::cmp::PartialOrd
    + std::marker::Sized
{
    fn unit() -> Self;
    fn zero() -> Self;
    fn as_f64(self) -> f64;
}

impl Field for usize {
    fn unit() -> usize {
        1
    }
    fn zero() -> usize {
        0
    }
    fn as_f64(self) -> f64 {
        self as f64
    }
}

impl Field for f32 {
    fn unit() -> f32 {
        1.0
    }
    fn zero() -> f32 {
        0.0
    }
    fn as_f64(self) -> f64 {
        self as f64
    }
}
