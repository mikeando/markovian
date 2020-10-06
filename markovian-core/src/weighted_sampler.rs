use serde::{Deserialize, Serialize};

use crate::num_basic::Field;
use rand::Rng;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WeightedSampler<T, D>
where
    T: Ord,
{
    // Why does this need to be a BTreeMap, not just Vec<(T,usize)>?
    pub counts: BTreeMap<T, D>,
    pub total: D,
}

impl<T, D> WeightedSampler<T, D>
where
    T: Ord + Clone,
    D: Field,
{
    pub fn new() -> WeightedSampler<T, D> {
        WeightedSampler {
            counts: BTreeMap::new(),
            total: D::zero(),
        }
    }

    pub fn add_symbol(&mut self, s: T) {
        self.total += D::unit();
        *self.counts.entry(s).or_insert_with(D::zero) += D::unit();
    }

    pub fn add_symbol_with_weight(&mut self, s: T, w: D) {
        if w < D::zero() {
            return;
        }
        self.total += w;
        *self.counts.entry(s).or_insert_with(D::zero) += w;
    }

    pub fn sample_next_symbol<R: Rng>(&self, rng: &mut R) -> Option<T> {
        if self.total == D::zero() {
            return None;
        }
        let mut v = rng.gen_range(D::zero(), self.total);
        for (s, c) in self.counts.iter() {
            if v < *c {
                return Some(s.clone());
            }
            v -= *c;
        }
        None
    }
}

impl<T, D> WeightedSampler<T, D>
where
    T: Ord,
    D: Field,
{
    pub fn logp(&self, v: &T) -> Option<f32> {
        self.counts
            .get(v)
            .map(|v| (v.as_f64() / self.total.as_f64()).ln() as f32)
    }
}

impl<T, D> Default for WeightedSampler<T, D>
where
    T: Ord + Clone,
    D: Field,
{
    fn default() -> Self {
        Self::new()
    }
}

//TODO: Would be nice for this to be more general than D=f32
impl<T> WeightedSampler<T, f32>
where
    T: Ord + Clone,
{
    /// Preserves the totals, but maps the individual probabilities.
    /// (But reweights them back to the correct sum.)
    pub fn map_probabilities<F>(&self, f: F) -> WeightedSampler<T, f32>
    where
        F: Fn(f32) -> f32,
    {
        let original_sum_w = self.total;
        let mut result_sum_w = 0.0;
        let mut result = self.clone();
        for w in result.counts.values_mut() {
            let new_w = f(*w / original_sum_w);
            //TODO: Better handling on non-finite values?
            if new_w > 0.0 {
                result_sum_w += new_w;
                *w = new_w;
            } else {
                *w = 0.0;
            }
        }
        //TODO: What if original_sum_w is zero or result_sum_w is zero?
        for w in result.counts.values_mut() {
            *w *= original_sum_w / result_sum_w;
        }

        result
    }
}
