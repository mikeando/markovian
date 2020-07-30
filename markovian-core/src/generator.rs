use crate::{
    ngram::TrigramCount,
    num_basic::Field,
    symbol::{SymbolTable, SymbolTableEntryId},
    weighted_sampler::WeightedSampler,
};
use rand::Rng;
use std::{
    collections::{BTreeMap, BTreeSet},
    iter,
};

pub struct TransitionTable<T, D> {
    weights_table: BTreeMap<(T, T), WeightedSampler<T, D>>,
}

impl<T, D> TransitionTable<T, D>
where
    T: Ord + Clone + std::fmt::Debug,
    D: Field,
{
    pub fn new(counts: TrigramCount<T, D>) -> TransitionTable<T, D> {
        let mut weights_table: BTreeMap<(T, T), WeightedSampler<T, D>> = BTreeMap::new();

        for ((a, b, c), w) in counts.into_iter() {
            weights_table
                .entry((a, b))
                .or_default()
                .add_symbol_with_weight(c, w);
        }

        TransitionTable { weights_table }
    }

    pub fn sample<R: Rng>(&self, key: &[T], rng: &mut R) -> T {
        //TODO: Make this error properly, or accept something such that it can't error like &[T;3]?
        assert!(key.len() == 2);

        let m = self.weights_table.get(&(key[0].clone(), key[1].clone()));
        //TODO: Dont unwrap - error sensibly!
        m.unwrap().sample_next_symbol(rng)
    }

    pub fn context_length(&self) -> usize {
        2
    }

    pub fn calculate_logp(&self, v: &[T]) -> f32 {
        //TODO: Handle the case where v.len() < 2
        if v.len() < 2 {
            unimplemented!("Can't handle v < 2");
        }
        let mut sum_log_p = 0.0;
        for i in 0..v.len() - 2 {
            //TODO: Handle unwrap fail!
            let log_p = self
                .weights_table
                .get(&(v[i].clone(), v[i + 1].clone()))
                .unwrap()
                .logp(&v[i + 2]);
            sum_log_p += log_p;
        }
        sum_log_p
    }
}

//TODO: Make this work with char too?
pub struct Generator<D> {
    symbol_table: SymbolTable<u8>,
    transition_table: TransitionTable<SymbolTableEntryId, D>,
    rev_transition_table: TransitionTable<SymbolTableEntryId, D>,
}

impl<D> Generator<D>
where
    D: Field,
{
    pub fn new(
        symbol_table: SymbolTable<u8>,
        transition_table: TransitionTable<SymbolTableEntryId, D>,
        rev_transition_table: TransitionTable<SymbolTableEntryId, D>,
    ) -> Generator<D> {
        Generator {
            symbol_table,
            transition_table,
            rev_transition_table,
        }
    }

    pub fn context_length(&self) -> usize {
        self.transition_table.context_length()
    }

    pub fn start_symbol_id(&self) -> SymbolTableEntryId {
        self.symbol_table.start_symbol_id()
    }

    pub fn end_symbol_id(&self) -> SymbolTableEntryId {
        self.symbol_table.end_symbol_id()
    }

    pub fn generate_initial_vector(&self) -> Vec<SymbolTableEntryId> {
        vec![self.start_symbol_id(); self.context_length()]
    }

    pub fn key<'a>(&self, v: &'a [SymbolTableEntryId]) -> &'a [SymbolTableEntryId] {
        &v[v.len() - self.context_length()..v.len()]
    }

    pub fn body<'a>(&self, v: &'a [SymbolTableEntryId]) -> &'a [SymbolTableEntryId] {
        &v[self.context_length()..(v.len() - self.context_length())]
    }

    pub fn augment_prefix(&self, prefix: &[SymbolTableEntryId]) -> Vec<SymbolTableEntryId> {
        iter::repeat(self.start_symbol_id())
            .take(self.context_length())
            .chain(prefix.iter().cloned())
            .collect()
    }

    pub fn augment_and_reverse_suffix(
        &self,
        suffix: &[SymbolTableEntryId],
    ) -> Vec<SymbolTableEntryId> {
        iter::repeat(self.end_symbol_id())
            .take(self.context_length())
            .chain(suffix.iter().rev().cloned())
            .collect()
    }

    pub fn build_prefix_sampler(
        &self,
        prefix_str: &str,
    ) -> WeightedSampler<Vec<SymbolTableEntryId>, f32> {
        let symbolified_prefixes = self
            .symbol_table
            .symbolifications_prefix(prefix_str.as_bytes());
        let prefixes_with_log_prob: Vec<_> = symbolified_prefixes
            .iter()
            .map(|prefix| {
                let w: Vec<SymbolTableEntryId> = self.augment_prefix(prefix);
                let logp: f32 = self.transition_table.calculate_logp(&w);

                (w, logp)
            })
            .collect();

        let mut min_log_p = 0.0;
        for (_, logp) in &prefixes_with_log_prob {
            if *logp < min_log_p {
                min_log_p = *logp;
            }
        }
        let mut sampler = WeightedSampler::<Vec<SymbolTableEntryId>, f32>::new();
        for (ss, logp) in prefixes_with_log_prob {
            let w = (logp - min_log_p).exp();
            sampler.add_symbol_with_weight(ss, w);
        }
        sampler
    }

    fn build_suffix_sampler(
        &self,
        suffix_str: &str,
    ) -> WeightedSampler<Vec<SymbolTableEntryId>, f32> {
        // Generate all possible symbolifications of the suffix
        // Calculate their probabilities and select one.
        // Generate using that symbolified prefix
        let symbolified_sufixes = self
            .symbol_table
            .symbolifications_suffix(suffix_str.as_bytes());
        let suffixes_with_log_prob: Vec<_> = symbolified_sufixes
            .iter()
            .map(|suffix| {
                let w: Vec<SymbolTableEntryId> = self.augment_and_reverse_suffix(suffix);
                let logp: f32 = self.rev_transition_table.calculate_logp(&w);
                (w, logp)
            })
            .collect();

        let mut min_log_p = 0.0;
        for (_, logp) in &suffixes_with_log_prob {
            if *logp < min_log_p {
                min_log_p = *logp;
            }
        }
        let mut sampler = WeightedSampler::<Vec<SymbolTableEntryId>, f32>::new();
        for (ss, logp) in suffixes_with_log_prob {
            let w = (logp - min_log_p).exp();
            sampler.add_symbol_with_weight(ss, w);
        }
        sampler
    }

    // TODO this probably belongs in TransitionTable
    // TODO as does self.key and self.context_length
    fn continue_prediction<R: Rng>(
        &self,
        transition_table: &TransitionTable<SymbolTableEntryId, D>,
        terminal: SymbolTableEntryId,
        mut v: Vec<SymbolTableEntryId>,
        rng: &mut R,
    ) -> Vec<SymbolTableEntryId> {
        loop {
            let next: SymbolTableEntryId = transition_table.sample(self.key(&v), rng);
            if next == terminal {
                v.extend(iter::repeat(terminal).take(self.context_length()));
                return v;
            }
            v.push(next);
        }
    }

    fn continue_fwd_prediction<R: Rng>(
        &self,
        v: Vec<SymbolTableEntryId>,
        rng: &mut R,
    ) -> Vec<SymbolTableEntryId> {
        let end_id = self.end_symbol_id();
        self.continue_prediction(&self.transition_table, end_id, v, rng)
    }

    fn continue_bwd_prediction<R: Rng>(
        &self,
        v: Vec<SymbolTableEntryId>,
        rng: &mut R,
    ) -> Vec<SymbolTableEntryId> {
        let start_id = self.start_symbol_id();
        self.continue_prediction(&self.rev_transition_table, start_id, v, rng)
    }

    pub fn generate<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<String> {
        // Generate an initial vector.
        let mut result = Vec::<String>::with_capacity(n);
        for _i in 0..n {
            let v = self.generate_initial_vector();
            let v = self.continue_fwd_prediction(v, rng);
            result.push(self.symbol_table.render(self.body(&v)))
        }
        result
    }

    pub fn generate_with_prefix<R: Rng>(
        &self,
        prefix_str: &str,
        n: usize,
        rng: &mut R,
    ) -> Vec<String> {
        let mut result = Vec::<String>::with_capacity(n);

        // Generate all possible symbolifications of the prefix
        // Calculate their probabilities
        let sampler = self.build_prefix_sampler(prefix_str);

        for _i in 0..n {
            // Choose one of the prefixes
            let chosen_prefix = sampler.sample_next_symbol(rng);

            // Generate using that symbolified prefix
            let v = self.continue_fwd_prediction(chosen_prefix, rng);
            result.push(self.symbol_table.render(self.body(&v)))
        }
        result
    }

    pub fn generate_with_suffix<R: Rng>(
        &self,
        suffix_str: &str,
        n: usize,
        rng: &mut R,
    ) -> Vec<String> {
        let mut result = Vec::<String>::with_capacity(n);

        // Generate all possible symbolifications of the suffix
        // Calculate their probabilities
        // NOTE: This sampler generates the suffix *reversed*
        let sampler = self.build_suffix_sampler(suffix_str);

        for _i in 0..n {
            // Choose one of the suffixes
            let chosen_suffix = sampler.sample_next_symbol(rng);

            // Generate using that symbolified prefix
            let v = self.continue_bwd_prediction(chosen_suffix, rng);

            // Need to reverse v before we render it.
            let mut v = self.body(&v).to_vec();
            v.reverse();
            result.push(self.symbol_table.render(&v))
        }

        result
    }

    pub fn generate_with_prefix_and_suffix<R: Rng>(
        &self,
        prefix_str: &str,
        suffix_str: &str,
        n: usize,
        rng: &mut R,
    ) -> Vec<String> {
        // TOOD: Should we add weights to any of the samplers to get a better result?
        let mut result = Vec::<String>::with_capacity(n);

        // TODO: Q. How big does N need to be? Currently it is a completely random guess.
        let N = (5 * n).max(10);
        let splice_length = self.context_length() + 1;

        // We generate N forward from prefix_str
        // Then store up all the "fwd-splice-points" after prefix
        let prefix_sampler = self.build_prefix_sampler(prefix_str);
        let mut fwd_completions = Vec::<(usize, Vec<SymbolTableEntryId>)>::with_capacity(N);

        for _i in 0..N {
            let chosen_prefix = prefix_sampler.sample_next_symbol(rng);
            let prefix_length = chosen_prefix.len();
            let completed_fwd = self.continue_fwd_prediction(chosen_prefix, rng);
            fwd_completions.push((prefix_length, completed_fwd));
        }

        let mut fwd_part_samplers: BTreeMap<
            &[SymbolTableEntryId],
            WeightedSampler<(usize, &[SymbolTableEntryId]), f32>,
        > = BTreeMap::new();
        for (k, v) in &fwd_completions {
            for (i, w) in v[*k..].windows(splice_length).enumerate() {
                fwd_part_samplers
                    .entry(w)
                    .or_default()
                    .add_symbol((k + i, v));
            }
        }

        //TODO: How do we handle duplication etc?

        // We generate N backward from suffix_str
        // Store up all the bwd-splice-points before suffix
        let suffix_sampler = self.build_suffix_sampler(suffix_str);
        let mut bwd_completions = Vec::<(usize, Vec<SymbolTableEntryId>)>::with_capacity(N);

        for _i in 0..N {
            let chosen_suffix = suffix_sampler.sample_next_symbol(rng);
            let suffix_length = chosen_suffix.len();
            let mut completed_bwd = self.continue_bwd_prediction(chosen_suffix, rng);
            completed_bwd.reverse();
            bwd_completions.push((suffix_length, completed_bwd));
        }

        let mut bwd_part_samplers: BTreeMap<
            &[SymbolTableEntryId],
            WeightedSampler<(usize, &[SymbolTableEntryId]), f32>,
        > = BTreeMap::new();
        for (k, v) in &bwd_completions {
            for (i, w) in v[..v.len() - *k].windows(splice_length).enumerate() {
                bwd_part_samplers.entry(w).or_default().add_symbol((i, v));
            }
        }

        // The we try to match up fwd and bwd splice points.
        let fwd_splice_point_keys: BTreeSet<&[SymbolTableEntryId]> =
            fwd_part_samplers.keys().cloned().collect();
        let bwd_splice_point_keys: BTreeSet<&[SymbolTableEntryId]> =
            bwd_part_samplers.keys().cloned().collect();

        // These might be one too short.
        // println!("fwd_splice_point_keys={:?}", fwd_splice_point_keys.iter().map( |v| self.symbol_table.render(&v) ).collect::<Vec<_>>());
        // println!("bwd_splice_point_keys={:?}", bwd_splice_point_keys.iter().map( |v| self.symbol_table.render(&v) ).collect::<Vec<_>>());

        let common_splice_point_keys: Vec<_> = fwd_splice_point_keys
            .intersection(&bwd_splice_point_keys)
            .collect();

        // println!("common_splice_point_keys={:?}", common_splice_point_keys.iter().map( |v| self.symbol_table.render(&v) ).collect::<Vec<_>>());

        for _i in 0..N {
            // Pick a splice point key

            let mut splice_point_sampler: WeightedSampler<&[SymbolTableEntryId], f32> =
                WeightedSampler::new();
            for sp in &common_splice_point_keys {
                splice_point_sampler.add_symbol(*sp)
            }

            let splice_point = splice_point_sampler.sample_next_symbol(rng);
            // println!("picked splice_point={:?}", self.symbol_table.render(splice_point) );

            // Pick a prefix for that key
            let fwd_part_sampler = fwd_part_samplers.get(splice_point).unwrap();
            let prefix = fwd_part_sampler.sample_next_symbol(rng);
            // println!("picked prefix u={}, v={}", prefix.0, self.symbol_table.render(prefix.1) );

            // Pick a suffix for that key
            let bwd_part_sampler = bwd_part_samplers.get(splice_point).unwrap();
            let suffix = bwd_part_sampler.sample_next_symbol(rng);
            // println!("picked suffix u={}, v={}", suffix.0, self.symbol_table.render(suffix.1) );

            // Join it all together
            // Finally an answer is PREFIX-FWD-SPLICE-BWD-SUFFIX

            let whole = [
                &prefix.1[..prefix.0],
                //splice_point,
                &suffix.1[suffix.0..],
            ]
            .concat();
            // Then we render the answer.
            // println!("whole={}", self.symbol_table.render(&whole) );

            let text = self.symbol_table.render(self.body(&whole));
            result.push(text);
        }

        result
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::ngram::TrigramCount;
    use std::iter;

    fn dumb_u8_symbol_table<T: AsRef<str>>(values: &[T]) -> SymbolTable<u8> {
        let mut symbol_table = SymbolTable::new();

        symbol_table.add(SymbolTableEntry::Start);
        symbol_table.add(SymbolTableEntry::End);
        for s in values {
            for c in s.as_ref().bytes() {
                symbol_table.add(SymbolTableEntry::Single(c));
            }
        }

        symbol_table
    }

    fn weight_for_symbolification(v: &[SymbolTableEntryId]) -> f32 {
        return 1.0 / ((v.len() * v.len()) as f32);
    }

    // TODO: Error if we can't get at least one symbolification
    // TODO: Move this into the Symbol table?
    // TODO: Provide weight_for_symbolification as argument.
    fn augment_and_symbolify(
        symbol_table: &SymbolTable<u8>,
        v: &[u8],
    ) -> Vec<(Vec<SymbolTableEntryId>, f32)> {
        let start_id = symbol_table.start_symbol_id();
        let end_id = symbol_table.end_symbol_id();
        let mut result = Vec::new();
        let order = 2;
        for x in symbol_table.symbolifications(v) {
            let w = weight_for_symbolification(&x);
            let ss: Vec<SymbolTableEntryId> = iter::repeat(start_id)
                .take(order)
                .chain(x.into_iter())
                .chain(iter::repeat(end_id).take(order))
                .collect();
            result.push((ss, w));
        }
        let sum_w: f32 = result.iter().map(|(_, w)| w).sum();
        result.into_iter().map(|(x, w)| (x, w / sum_w)).collect()
    }

    fn simple_generator() -> Generator<f32> {
        let values = vec!["hello"];
        let mut symbol_table = dumb_u8_symbol_table(&values);

        let start_id = symbol_table.start_symbol_id();
        let end_id = symbol_table.end_symbol_id();

        let symbolified_values: Vec<(Vec<SymbolTableEntryId>, f32)> = values
            .iter()
            .flat_map(|s| augment_and_symbolify(&symbol_table, s.as_bytes()))
            .collect();

        let trigrams: TrigramCount<SymbolTableEntryId, f32> = symbolified_values
            .iter()
            .map(|(v, w)| (&v[..], *w))
            .collect();
        let rev_trigrams: TrigramCount<SymbolTableEntryId, f32> = trigrams
            .iter()
            .map(|((a, b, c), w)| ((*c, *b, *a), *w))
            .collect();

        let transition_table = TransitionTable::new(trigrams);
        let rev_transition_table = TransitionTable::new(rev_trigrams);

        Generator::new(symbol_table, transition_table, rev_transition_table)
    }

    fn simple_generator_2() -> Generator<f32> {
        let values = vec!["word"];
        let symbol_table = dumb_u8_symbol_table(&values);

        let symbolified_values: Vec<(Vec<SymbolTableEntryId>, f32)> = values
            .iter()
            .flat_map(|s| augment_and_symbolify(&symbol_table, s.as_bytes()))
            .collect();

        let trigrams: TrigramCount<SymbolTableEntryId, f32> = symbolified_values
            .iter()
            .map(|(v, w)| (&v[..], *w))
            .collect();
        let rev_trigrams: TrigramCount<SymbolTableEntryId, f32> = trigrams
            .iter()
            .map(|((a, b, c), w)| ((*c, *b, *a), *w))
            .collect();

        let transition_table = TransitionTable::new(trigrams);
        let rev_transition_table = TransitionTable::new(rev_trigrams);

        Generator::new(symbol_table, transition_table, rev_transition_table)
    }

    fn larger_generator() -> Generator<f32> {
        let values: Vec<String> = std::fs::read_to_string(
            "/Users/michaelanderson/Code/markovian/resources/boys_names_2940_lc.txt",
        )
        .unwrap()
        .lines()
        .map(|n| n.trim().to_string())
        .filter(|s| s.len() >= 3)
        .collect();

        let symbol_table = dumb_u8_symbol_table(&values);

        let symbolified_values: Vec<(Vec<SymbolTableEntryId>, f32)> = values
            .iter()
            .flat_map(|s| augment_and_symbolify(&symbol_table, s.as_bytes()))
            .collect();

        let trigrams: TrigramCount<SymbolTableEntryId, f32> = symbolified_values
            .iter()
            .map(|(v, w)| (&v[..], *w))
            .collect();
        let rev_trigrams: TrigramCount<SymbolTableEntryId, f32> = trigrams
            .iter()
            .map(|((a, b, c), w)| ((*c, *b, *a), *w))
            .collect();

        let transition_table = TransitionTable::new(trigrams);
        let rev_transition_table = TransitionTable::new(rev_trigrams);

        Generator::new(symbol_table, transition_table, rev_transition_table)
    }

    #[test]
    pub fn test_augment_and_symbolify_hello() {
        let v = "hello";
        let symbol_table = dumb_u8_symbol_table(&[v]);
        let s = augment_and_symbolify(&symbol_table, v.as_bytes());
        assert_eq!(s.len(), 1);
    }

    #[test]
    pub fn test_symbolify_hello() {
        let v = "hello";
        let symbol_table = dumb_u8_symbol_table(&[v]);
        let ss = symbol_table.symbolifications(v.as_bytes());
        assert_eq!(ss.len(), 1);
    }

    #[test]
    pub fn generate_simple() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let m: String = g.generate(1, &mut rng)[0].clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_simple_2() {
        let mut rng = rand::thread_rng();
        let g = simple_generator_2();
        let m: String = g.generate(1, &mut rng)[0].clone();
        assert_eq!(m, "word");
    }

    #[test]
    pub fn generate_prefix() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let m: String = g.generate_with_prefix("hel", 1, &mut rng)[0].clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_prefix_empty() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let m: String = g.generate_with_prefix("", 1, &mut rng)[0].clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_suffix() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let m: String = g.generate_with_suffix("llo", 1, &mut rng)[0].clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_suffix_empty() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let m: String = g.generate_with_suffix("", 1, &mut rng)[0].clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_with_prefix_and_suffix() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let m: String = g.generate_with_prefix_and_suffix("h", "o", 1, &mut rng)[0].clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_with_prefix_and_suffix_big() {
        let mut rng = rand::thread_rng();
        let g = larger_generator();
        let prefix = "h";
        let suffix = "y";
        let m = g.generate_with_prefix_and_suffix(prefix, suffix, 10, &mut rng);

        for v in m {
            assert!(
                v.starts_with(prefix) && v.ends_with(suffix),
                "Expected {}..{} but got {}",
                prefix,
                suffix,
                v
            );
        }
    }
}
