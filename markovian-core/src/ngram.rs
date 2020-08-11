use crate::num_basic::Field;
use std::{collections::BTreeMap, iter::FromIterator};

pub struct BigramCount<S, D> {
    counts: BTreeMap<(S, S), D>,
}

impl<S, D> BigramCount<S, D>
where
    S: Ord + Clone,
    D: Field,
{
    pub fn new() -> Self {
        Self {
            counts: BTreeMap::new(),
        }
    }

    pub fn add_sequence(&mut self, values: &[S], weight: D) {
        let n = values.len();
        if n <= 1 {
            return;
        }
        for i in 0..(n - 1) {
            let t = (values[i].clone(), values[i + 1].clone());
            *self.counts.entry(t).or_insert_with(D::zero) += weight;
        }
    }

    pub fn add_entry(&mut self, value: (S, S), weight: D) {
        *self.counts.entry(value).or_insert_with(D::zero) += weight;
    }

    pub fn count(&self, k: &(S, S)) -> D {
        self.counts.get(k).cloned().unwrap_or_else(D::zero)
    }

    pub fn keys<'a>(&'a self) -> impl Iterator<Item = &(S, S)> + 'a {
        self.counts.keys()
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (&(S, S), &D)> + 'a {
        self.counts.iter()
    }
}

#[derive(Debug)]
pub struct TrigramCount<S, D> {
    counts: BTreeMap<(S, S, S), D>,
}

impl<S, D> TrigramCount<S, D>
where
    S: Ord + Clone,
    D: Field,
{
    pub fn new() -> Self {
        Self {
            counts: BTreeMap::new(),
        }
    }

    pub fn add_sequence(&mut self, values: &[S], weight: D) {
        let n = values.len();
        if n <= 2 {
            return;
        }
        for i in 0..(n - 2) {
            let t = (
                values[i].clone(),
                values[i + 1].clone(),
                values[i + 2].clone(),
            );
            self.add_entry(t, weight);
        }
    }

    pub fn add_entry(&mut self, value: (S, S, S), w: D) {
        *self.counts.entry(value).or_insert_with(D::zero) += w;
    }

    pub fn count(&self, k: &(S, S, S)) -> D {
        self.counts.get(k).cloned().unwrap_or_else(D::zero)
    }

    pub fn keys<'a>(&'a self) -> impl Iterator<Item = &(S, S, S)> + 'a {
        self.counts.keys()
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (&(S, S, S), &D)> + 'a {
        self.counts.iter()
    }

    pub fn into_iter(self) -> impl Iterator<Item = ((S, S, S), D)> {
        self.counts.into_iter()
    }
}

impl<S, D> FromIterator<(S, S)> for BigramCount<S, D>
where
    S: Ord + Clone,
    D: Field,
{
    fn from_iter<I: IntoIterator<Item = (S, S)>>(iter: I) -> Self {
        let mut bigrams = BigramCount::new();

        for i in iter {
            bigrams.add_entry(i, D::unit());
        }

        bigrams
    }
}

impl<S, D> FromIterator<((S, S), D)> for BigramCount<S, D>
where
    S: Ord + Clone,
    D: Field,
{
    fn from_iter<I: IntoIterator<Item = ((S, S), D)>>(iter: I) -> Self {
        let mut bigrams = BigramCount::new();

        for (i, w) in iter {
            bigrams.add_entry(i, w);
        }

        bigrams
    }
}

impl<'a, S, D> FromIterator<&'a [S]> for BigramCount<S, D>
where
    S: Ord + Clone + 'a,
    D: Field,
{
    fn from_iter<I: IntoIterator<Item = &'a [S]>>(iter: I) -> Self {
        let mut bigrams = BigramCount::new();

        for i in iter {
            bigrams.add_sequence(i, D::unit());
        }

        bigrams
    }
}

impl<'a, 'b, S, D> FromIterator<(&'a [S], D)> for BigramCount<S, D>
where
    S: Ord + Clone + 'b,
    D: Field,
    'b: 'a,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (&'a [S], D)>,
    {
        let mut bigrams = BigramCount::new();

        for (s, w) in iter {
            bigrams.add_sequence(s, w);
        }

        bigrams
    }
}

impl<S, D> FromIterator<(S, S, S)> for TrigramCount<S, D>
where
    S: Ord + Clone,
    D: Field,
{
    fn from_iter<I: IntoIterator<Item = (S, S, S)>>(iter: I) -> Self {
        let mut trigrams = TrigramCount::new();

        for i in iter {
            trigrams.add_entry(i, D::unit());
        }

        trigrams
    }
}

impl<S, D> FromIterator<((S, S, S), D)> for TrigramCount<S, D>
where
    S: Ord + Clone,
    D: Field,
{
    fn from_iter<I: IntoIterator<Item = ((S, S, S), D)>>(iter: I) -> Self {
        let mut trigrams = TrigramCount::new();

        for (i, w) in iter {
            trigrams.add_entry(i, w);
        }

        trigrams
    }
}

impl<'a, S, D> FromIterator<&'a [S]> for TrigramCount<S, D>
where
    S: Ord + Clone + 'a,
    D: Field,
{
    fn from_iter<I: IntoIterator<Item = &'a [S]>>(iter: I) -> Self {
        let mut trigrams = TrigramCount::new();

        for i in iter {
            trigrams.add_sequence(i, D::unit());
        }

        trigrams
    }
}

impl<'a, S, D> FromIterator<(&'a [S], D)> for TrigramCount<S, D>
where
    S: Ord + Clone + 'a,
    D: Field,
{
    fn from_iter<I: IntoIterator<Item = (&'a [S], D)>>(iter: I) -> Self {
        let mut trigrams = TrigramCount::new();

        for (s, w) in iter {
            trigrams.add_sequence(s, w);
        }

        trigrams
    }
}

impl<S, D> FromIterator<Vec<S>> for TrigramCount<S, D>
where
    S: Ord + Clone,
    D: Field,
{
    fn from_iter<I: IntoIterator<Item = Vec<S>>>(iter: I) -> Self {
        let mut trigrams = TrigramCount::new();

        for i in iter {
            trigrams.add_sequence(&i, D::unit());
        }

        trigrams
    }
}

#[cfg(test)]
mod test {
    use super::*;

    mod trigram {
        use super::*;

        #[test]
        pub fn test_add_and_count() {
            let mut trigrams = TrigramCount::<char, usize>::new();
            trigrams.add_sequence(&vec!['a', 'b', 'c', 'd'], 1);
            trigrams.add_sequence(&vec!['b', 'c', 'd'], 1);
            assert_eq!(trigrams.count(&('a', 'b', 'c')), 1);
            assert_eq!(trigrams.count(&('b', 'c', 'd')), 2);
            assert_eq!(trigrams.count(&('q', 'q', 'q')), 0);
        }

        #[test]
        pub fn test_keys() {
            let mut trigrams = TrigramCount::<char, usize>::new();
            trigrams.add_sequence(&vec!['a', 'b', 'c', 'd'], 1);
            assert_eq!(
                trigrams.keys().cloned().collect::<Vec<_>>(),
                vec![('a', 'b', 'c'), ('b', 'c', 'd')]
            );
        }

        #[test]
        pub fn test_iter() {
            let mut trigrams = TrigramCount::<char, usize>::new();
            trigrams.add_sequence(&vec!['a', 'b', 'c', 'd'], 1);
            trigrams.add_sequence(&vec!['b', 'c', 'd'], 1);
            assert_eq!(
                trigrams
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<Vec<_>>(),
                vec![(('a', 'b', 'c'), 1), (('b', 'c', 'd'), 2)]
            );
        }

        #[test]
        pub fn test_from_iter_tuple() {
            let v = vec![('a', 'b', 'c'), ('d', 'e', 'f')];
            let trigrams: TrigramCount<char, usize> = v.into_iter().collect();
            assert_eq!(trigrams.count(&('a', 'b', 'c')), 1);
            assert_eq!(trigrams.count(&('d', 'e', 'f')), 1);
        }

        #[test]
        pub fn test_from_iter_seq() {
            let v = vec!["abcd", "bcd"];
            let v: Vec<Vec<char>> = v.into_iter().map(|w| w.chars().collect()).collect();
            let trigrams: TrigramCount<char, usize> = v.iter().map(|v| &v[..]).collect();
            assert_eq!(trigrams.count(&('a', 'b', 'c')), 1);
            assert_eq!(trigrams.count(&('b', 'c', 'd')), 2);
        }

        #[test]
        pub fn test_from_iter_seq2() {
            let v = vec!["abcd", "bcd"];
            let trigrams: TrigramCount<char, usize> = v
                .iter()
                .map(|v| -> Vec<char> { v.chars().collect() })
                .collect();
            assert_eq!(trigrams.count(&('a', 'b', 'c')), 1);
            assert_eq!(trigrams.count(&('b', 'c', 'd')), 2);
        }

        #[test]
        pub fn trigram_to_bigram_map() {
            let mut trigrams = TrigramCount::<char, usize>::new();
            // We're using '^' and '$' as the start and end symbols respectively
            // When using trgrams we need a context of 2, so we append two
            // symbols to either end.
            // But when we convert to bigrams, we only want one.
            // So we need to filter out the ('^','^') case.
            // We can get the rest of the bigrams by taking just the start 2 symbols from each
            // trigram.
            trigrams.add_sequence(&vec!['^', '^', 'a', 'b', '$', '$'], 1);
            trigrams.add_sequence(&vec!['^', '^', 'b', '$', '$'], 1);
            let bigrams: BigramCount<char, usize> = trigrams
                .iter()
                .filter(|(v, _)| !matches!(**v, ('^', '^', _)))
                .map(|((a, b, _), count)| ((a.clone(), b.clone()), *count))
                .collect();
            assert_eq!(
                bigrams
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<Vec<_>>(),
                vec![
                    (('^', 'a'), 1),
                    (('^', 'b'), 1),
                    (('a', 'b'), 1),
                    (('b', '$'), 2)
                ]
            );
        }
    }

    mod bigram {
        use super::*;

        #[test]
        pub fn test_add_and_count() {
            let mut bigrams = BigramCount::<char, usize>::new();
            bigrams.add_sequence(&vec!['a', 'b', 'c', 'd'], 1);
            bigrams.add_sequence(&vec!['b', 'c', 'd'], 1);
            assert_eq!(bigrams.count(&('a', 'b')), 1);
            assert_eq!(bigrams.count(&('b', 'c')), 2);
            assert_eq!(bigrams.count(&('c', 'd')), 2);
            assert_eq!(bigrams.count(&('q', 'q')), 0);
        }

        #[test]
        pub fn test_keys() {
            let mut bigrams = BigramCount::<char, usize>::new();
            bigrams.add_sequence(&vec!['a', 'b', 'c', 'd'], 1);
            assert_eq!(
                bigrams.keys().cloned().collect::<Vec<_>>(),
                vec![('a', 'b'), ('b', 'c'), ('c', 'd')]
            );
        }

        #[test]
        pub fn test_iter() {
            let mut bigrams = BigramCount::<char, usize>::new();
            bigrams.add_sequence(&vec!['a', 'b', 'c'], 1);
            bigrams.add_sequence(&vec!['b', 'c'], 1);
            assert_eq!(
                bigrams
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<Vec<_>>(),
                vec![(('a', 'b'), 1), (('b', 'c'), 2)]
            );
        }

        #[test]
        pub fn test_from_slice_and_weight_iter() {
            let v: Vec<(&[i32], f32)> = vec![(&[1, 2, 3], 0.5), (&[2, 4], 0.5)];
            let bigrams: BigramCount<i32, f32> = v.into_iter().collect();
        }
    }
}
