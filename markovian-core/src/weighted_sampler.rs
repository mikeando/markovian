use rand::Rng;
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct WeightedSampler<T> {
    // Why does this need to be a BTreeMap, not just Vec<(T,usize)>?
    counts: BTreeMap<T, usize>,
    total: usize,
}

impl<T> WeightedSampler<T>
where
    T: Ord + Clone,
{
    pub fn new() -> WeightedSampler<T> {
        WeightedSampler {
            counts: BTreeMap::new(),
            total: 0,
        }
    }

    pub fn add_symbol(&mut self, s: &T) {
        self.total += 1;
        *self.counts.entry(s.clone()).or_insert(0) += 1;
    }

    pub fn sample_next_symbol<R: Rng>(&self, rng: &mut R) -> T {
        let mut v = rng.gen_range(0, self.total);
        for (s, c) in self.counts.iter() {
            if v < *c {
                return s.clone();
            }
            v -= *c;
        }
        unreachable!();
    }
}
