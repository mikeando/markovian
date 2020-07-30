use crate::num_basic::Field;
use rand::Rng;
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct WeightedSampler<T, D> {
    // Why does this need to be a BTreeMap, not just Vec<(T,usize)>?
    counts: BTreeMap<T, D>,
    total: D,
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

    pub fn sample_next_symbol<R: Rng>(&self, rng: &mut R) -> T {
        let mut v = rng.gen_range(D::zero(), self.total);
        for (s, c) in self.counts.iter() {
            if v < *c {
                return s.clone();
            }
            v -= *c;
        }
        unreachable!();
    }
}

impl<T, D> WeightedSampler<T, D>
where
    T: Ord,
    D: Field,
{
    pub fn logp(&self, v: &T) -> f32 {
        match self.counts.get(v) {
            None => -f32::INFINITY,
            Some(v) => (v.as_f64() / self.total.as_f64()).ln() as f32,
        }
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
