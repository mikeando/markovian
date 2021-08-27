//! A wrapper for f32 that provides a reasonable implementation of
//! Hash and Ord.
//!
//! This implementation treats all NaNs as the same, and if careful
//! to ensure that positive and negative zero hash to the same value.
//!
//! Also we treat NaN as less than all other values.

use std::fmt::Formatter;
use std::hash::{Hash, Hasher};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct nf32(pub f32);

impl std::fmt::Debug for nf32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::fmt::Display for nf32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::ops::AddAssign<nf32> for nf32 {
    fn add_assign(&mut self, rhs: nf32) {
        self.0 += rhs.0
    }
}

impl PartialEq for nf32 {
    fn eq(&self, other: &Self) -> bool {
        (self.0.is_nan() && other.0.is_nan()) || (self.0 == other.0)
    }
}

impl Hash for nf32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // There's a few special cases
        // -0 and 0 have different bit representations,
        // but compare as equal and so must hash to zero
        // Also for this wrapper all NaNs compare equal.
        // which means we need a standard hash for them
        if self.0 == 0.0f32 {
            (0.0f32).to_bits().hash(state);
        } else if self.0.is_nan() {
            std::f32::NAN.to_bits().hash(state);
        } else {
            self.0.to_bits().hash(state);
        }
    }
}

impl Eq for nf32 {}

impl PartialOrd for nf32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for nf32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => self.0.partial_cmp(&other.0).unwrap(),
        }
    }
}
