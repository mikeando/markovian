use serde::{Deserialize, Serialize};

use crate::{
    num_basic::Field,
    renderer::Renderer,
    symbol::{SymbolTable, SymbolTableEntryId},
    weighted_sampler::WeightedSampler,
};
use rand::Rng;
use std::{
    collections::{BTreeMap, BTreeSet},
    iter,
};

#[derive(Debug)]
pub enum GenerationError {
    GenericError(String),
}

impl GenerationError {
    pub fn generic_error<T: Into<String>>(v: T) -> GenerationError {
        GenerationError::GenericError(v.into())
    }
}

pub fn create_trigrams<T, W, D>(words: &[(W, D)]) -> BTreeMap<(T, T, T), D>
where
    D: Field,
    W: AsRef<[T]>,
    T: Clone + Ord,
{
    let mut trigrams = BTreeMap::new();
    for (s, w) in words {
        let s = s.as_ref();
        let n = s.len();
        if n <= 2 {
            continue;
        }
        for i in 0..(n - 2) {
            let t = (s[i].clone(), s[i + 1].clone(), s[i + 2].clone());
            *trigrams.entry(t).or_insert_with(D::zero) += *w;
        }
    }
    trigrams
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionTable<T, D>
where
    T: Ord,
{
    weights_table: BTreeMap<(T, T), WeightedSampler<T, D>>,
}

impl<T, D> TransitionTable<T, D>
where
    T: Ord + Clone,
    D: Field,
{
    pub fn new(counts: BTreeMap<(T, T, T), D>) -> TransitionTable<T, D> {
        let mut weights_table: BTreeMap<(T, T), WeightedSampler<T, D>> = BTreeMap::new();

        for ((a, b, c), w) in counts.into_iter() {
            weights_table
                .entry((a, b))
                .or_default()
                .add_symbol_with_weight(c, w);
        }

        TransitionTable { weights_table }
    }

    pub fn sample<R: Rng>(&self, key: &[T], rng: &mut R) -> Option<T> {
        //TODO: Make this error properly, or accept something such that it can't error like &[T;3]?
        assert!(key.len() == 2);

        let m = self.weights_table.get(&(key[0].clone(), key[1].clone()))?;
        m.sample_next_symbol(rng)
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

// TODO: This serializes "badly" - there's a lot of redundency
// it should just be the symbol_table (which tends to be very small)
// then a list of the symbolid triples + weights. The TTs can be rebuilt from
// them quickly and easily.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Generator<T, D>
where
    T: Ord + Clone,
{
    pub symbol_table: SymbolTable<T>,
    transition_table: TransitionTable<SymbolTableEntryId, D>,
    rev_transition_table: TransitionTable<SymbolTableEntryId, D>,
}

impl<T, D> Generator<T, D>
where
    D: Field,
    T: Ord + Clone,
{
    pub fn from_trigrams(
        symbol_table: SymbolTable<T>,
        trigrams: BTreeMap<(SymbolTableEntryId, SymbolTableEntryId, SymbolTableEntryId), D>,
    ) -> Generator<T, D> {
        let rev_trigrams: BTreeMap<
            (SymbolTableEntryId, SymbolTableEntryId, SymbolTableEntryId),
            D,
        > = trigrams
            .iter()
            .map(|((a, b, c), w)| ((*c, *b, *a), *w))
            .collect();

        let transition_table = TransitionTable::new(trigrams);
        let rev_transition_table = TransitionTable::new(rev_trigrams);
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
        prefix: &[T],
    ) -> WeightedSampler<Vec<SymbolTableEntryId>, f32> {
        let symbolified_prefixes = self.symbol_table.symbolifications_prefix(prefix);
        let prefixes_with_log_prob: Vec<_> = symbolified_prefixes
            .iter()
            .map(|prefix| {
                let w: Vec<SymbolTableEntryId> = self.augment_prefix(prefix);
                let logp: f32 = self.transition_table.calculate_logp(&w);

                (w, logp)
            })
            .collect();

        let mut sampler = WeightedSampler::<Vec<SymbolTableEntryId>, f32>::new();
        if prefixes_with_log_prob
            .iter()
            .all(|(_k, logp)| *logp == -f32::INFINITY)
        {
            // They all have zero weight so we just assume all are equally likely
            for (ss, _logp) in prefixes_with_log_prob {
                sampler.add_symbol_with_weight(ss, 1.0);
            }
        } else {
            let mut min_log_p = 0.0;
            for (_, logp) in &prefixes_with_log_prob {
                if (*logp < min_log_p) && (*logp > -f32::INFINITY) {
                    min_log_p = *logp;
                }
            }
            for (ss, logp) in prefixes_with_log_prob {
                let w = (logp - min_log_p).exp();
                sampler.add_symbol_with_weight(ss, w);
            }
        }
        sampler
    }

    fn build_suffix_sampler(&self, suffix: &[T]) -> WeightedSampler<Vec<SymbolTableEntryId>, f32> {
        // Generate all possible symbolifications of the suffix
        // Calculate their probabilities and select one.
        // Generate using that symbolified prefix
        let symbolified_sufixes = self.symbol_table.symbolifications_suffix(suffix);
        let suffixes_with_log_prob: Vec<_> = symbolified_sufixes
            .iter()
            .map(|suffix| {
                let w: Vec<SymbolTableEntryId> = self.augment_and_reverse_suffix(suffix);
                let logp: f32 = self.rev_transition_table.calculate_logp(&w);
                (w, logp)
            })
            .collect();

        let mut sampler = WeightedSampler::<Vec<SymbolTableEntryId>, f32>::new();
        if suffixes_with_log_prob
            .iter()
            .all(|(_k, logp)| *logp == -f32::INFINITY)
        {
            // They all have zero weight so we just assume all are equally likely
            for (ss, _logp) in suffixes_with_log_prob {
                sampler.add_symbol_with_weight(ss, 1.0);
            }
        } else {
            let mut min_log_p = 0.0;
            for (_, logp) in &suffixes_with_log_prob {
                if (*logp < min_log_p) && (*logp > -f32::INFINITY) {
                    min_log_p = *logp;
                }
            }
            for (ss, logp) in suffixes_with_log_prob {
                let w = (logp - min_log_p).exp();
                sampler.add_symbol_with_weight(ss, w);
            }
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
    ) -> Result<Vec<SymbolTableEntryId>, GenerationError> {
        loop {
            let next: Option<SymbolTableEntryId> = transition_table.sample(self.key(&v), rng);
            let next = next.ok_or_else(|| {
                GenerationError::generic_error("Unable to find valid continuation")
            })?;
            if next == terminal {
                v.extend(iter::repeat(terminal).take(self.context_length()));
                return Ok(v);
            }
            v.push(next);
        }
    }

    fn continue_fwd_prediction<R: Rng>(
        &self,
        v: Vec<SymbolTableEntryId>,
        rng: &mut R,
    ) -> Result<Vec<SymbolTableEntryId>, GenerationError> {
        let end_id = self.end_symbol_id();
        self.continue_prediction(&self.transition_table, end_id, v, rng)
    }

    fn continue_bwd_prediction<R: Rng>(
        &self,
        v: Vec<SymbolTableEntryId>,
        rng: &mut R,
    ) -> Result<Vec<SymbolTableEntryId>, GenerationError> {
        let start_id = self.start_symbol_id();
        self.continue_prediction(&self.rev_transition_table, start_id, v, rng)
    }

    pub fn generate_multi<R: Rng>(
        &self,
        prefix: Option<&[T]>,
        suffix: Option<&[T]>,
        n: usize,
        rng: &mut R,
        renderer: &impl Renderer,
    ) -> Result<Vec<String>, GenerationError>
    where
        T: std::fmt::Debug, // TODO: Only used for error message - would be nice to remove
    {
        match (prefix, suffix) {
            (None, None) => self.generate(n, rng, renderer),
            (None, Some(suffix)) => self.generate_with_suffix(suffix, n, rng, renderer),
            (Some(prefix), None) => self.generate_with_prefix(prefix, n, rng, renderer),
            (Some(prefix), Some(suffix)) => {
                self.generate_with_prefix_and_suffix(prefix, suffix, n, rng, renderer)
            }
        }
    }

    pub fn generate<R: Rng>(
        &self,
        n: usize,
        rng: &mut R,
        renderer: &impl Renderer,
    ) -> Result<Vec<String>, GenerationError> {
        // Generate an initial vector.
        let mut result = Vec::<String>::with_capacity(n);
        for _i in 0..n {
            let v = self.generate_initial_vector();
            let v = self.continue_fwd_prediction(v, rng)?;
            result.push(renderer.render(self.body(&v)).unwrap())
        }
        Ok(result)
    }

    pub fn generate_with_prefix<R: Rng>(
        &self,
        prefix: &[T],
        n: usize,
        rng: &mut R,
        renderer: &impl Renderer,
    ) -> Result<Vec<String>, GenerationError> {
        let mut result = Vec::<String>::with_capacity(n);

        // Generate all possible symbolifications of the prefix
        // Calculate their probabilities
        let sampler = self.build_prefix_sampler(prefix);

        for _i in 0..n {
            // Choose one of the prefixes
            let chosen_prefix = sampler.sample_next_symbol(rng).unwrap();

            // Generate using that symbolified prefix
            let v = self.continue_fwd_prediction(chosen_prefix, rng)?;
            result.push(renderer.render(self.body(&v)).unwrap())
        }
        Ok(result)
    }

    pub fn generate_with_suffix<R: Rng>(
        &self,
        suffix: &[T],
        n: usize,
        rng: &mut R,
        renderer: &impl Renderer,
    ) -> Result<Vec<String>, GenerationError> {
        let mut result = Vec::<String>::with_capacity(n);

        // Generate all possible symbolifications of the suffix
        // Calculate their probabilities
        // NOTE: This sampler generates the suffix *reversed*
        let sampler = self.build_suffix_sampler(suffix);

        for _i in 0..n {
            // Choose one of the suffixes
            let chosen_suffix = sampler.sample_next_symbol(rng).unwrap();

            // Generate using that symbolified prefix
            let v = self.continue_bwd_prediction(chosen_suffix, rng)?;

            // Need to reverse v before we render it.
            let mut v = self.body(&v).to_vec();
            v.reverse();
            result.push(renderer.render(&v).unwrap())
        }

        Ok(result)
    }

    pub fn generate_with_prefix_and_suffix<R: Rng>(
        &self,
        prefix: &[T],
        suffix: &[T],
        n: usize,
        rng: &mut R,
        renderer: &impl Renderer,
    ) -> Result<Vec<String>, GenerationError>
    where
        T: std::fmt::Debug, // TODO: Only used for error message - would be nice to remove
    {
        // TOOD: Should we add weights to any of the samplers to get a better result?
        let mut result = Vec::<String>::with_capacity(n);

        // TODO: Q. How big does N need to be? Currently it is a completely random guess.
        let n_gen = (30 * n).max(10);
        let splice_length = self.context_length() + 1;

        // We generate N forward from prefix_str
        // Then store up all the "fwd-splice-points" after prefix
        let prefix_sampler = self.build_prefix_sampler(prefix);
        let mut fwd_completions = Vec::<(usize, Vec<SymbolTableEntryId>)>::with_capacity(n_gen);

        for _i in 0..n_gen {
            let chosen_prefix = prefix_sampler.sample_next_symbol(rng).unwrap();
            let prefix_length = chosen_prefix.len();
            let completed_fwd = self
                .continue_fwd_prediction(chosen_prefix, rng)
                .map_err(|e| {
                    GenerationError::generic_error(format!(
                        "Unable to generate continuation of prefix '{:?}' - {:?}",
                        prefix, e
                    ))
                })?;
            fwd_completions.push((prefix_length, completed_fwd));
        }

        type StemSampler<'a> = BTreeMap<
            &'a [SymbolTableEntryId],
            WeightedSampler<(usize, &'a [SymbolTableEntryId]), f32>,
        >;

        let mut fwd_part_samplers: StemSampler = BTreeMap::new();
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
        let suffix_sampler = self.build_suffix_sampler(suffix);
        let mut bwd_completions = Vec::<(usize, Vec<SymbolTableEntryId>)>::with_capacity(n_gen);

        for _i in 0..n_gen {
            let chosen_suffix = suffix_sampler.sample_next_symbol(rng).unwrap();
            let suffix_length = chosen_suffix.len();
            let mut completed_bwd =
                self.continue_bwd_prediction(chosen_suffix, rng)
                    .map_err(|e| {
                        GenerationError::generic_error(format!(
                            "Unable to generate backward continuation of suffix '{:?}' - {:?}",
                            suffix, e
                        ))
                    })?;
            completed_bwd.reverse();
            bwd_completions.push((suffix_length, completed_bwd));
        }

        let mut bwd_part_samplers: StemSampler = BTreeMap::new();
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

        for _i in 0..n {
            // Pick a splice point key

            let mut splice_point_sampler: WeightedSampler<&[SymbolTableEntryId], f32> =
                WeightedSampler::new();
            for sp in &common_splice_point_keys {
                splice_point_sampler.add_symbol(*sp)
            }

            let splice_point = splice_point_sampler.sample_next_symbol(rng).unwrap();
            // println!("picked splice_point={:?}", self.symbol_table.render(splice_point) );

            // Pick a prefix for that key
            let fwd_part_sampler = fwd_part_samplers.get(splice_point).unwrap();
            let prefix = fwd_part_sampler.sample_next_symbol(rng).unwrap();
            // println!("picked prefix u={}, v={}", prefix.0, self.symbol_table.render(prefix.1) );

            // Pick a suffix for that key
            let bwd_part_sampler = bwd_part_samplers.get(splice_point).unwrap();
            let suffix = bwd_part_sampler.sample_next_symbol(rng).unwrap();
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

            let text = renderer.render(self.body(&whole)).unwrap();
            result.push(text);
        }

        Ok(result)
    }
}

pub fn weight_for_symbolification(v: &[SymbolTableEntryId]) -> f32 {
    1.0 / ((v.len() * v.len()) as f32)
}

// TODO: Error if we can't get at least one symbolification
// TODO: Move this into the Symbol table?
// TODO: Provide weight_for_symbolification as argument.
// TOOD: Only use the shortest symbolifications?
pub fn augment_and_symbolify<T>(
    symbol_table: &SymbolTable<T>,
    v: &[T],
) -> Vec<(Vec<SymbolTableEntryId>, f32)>
where
    T: Ord + Clone,
{
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

#[cfg(test)]
mod test {

    use super::*;
    use crate::{renderer::RenderU8, symbol::SymbolTableEntry};

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

    fn simple_generator() -> Generator<u8, f32> {
        let values = vec!["hello"];
        let symbol_table = dumb_u8_symbol_table(&values);

        let symbolified_values: Vec<(Vec<SymbolTableEntryId>, f32)> = values
            .iter()
            .flat_map(|s| augment_and_symbolify(&symbol_table, s.as_bytes()))
            .collect();

        let trigrams = create_trigrams(&symbolified_values);
        Generator::from_trigrams(symbol_table, trigrams)
    }

    fn simple_generator_2() -> Generator<u8, f32> {
        let values = vec!["word"];
        let symbol_table = dumb_u8_symbol_table(&values);

        let symbolified_values: Vec<(Vec<SymbolTableEntryId>, f32)> = values
            .iter()
            .flat_map(|s| augment_and_symbolify(&symbol_table, s.as_bytes()))
            .collect();

        let trigrams = create_trigrams(&symbolified_values);
        Generator::from_trigrams(symbol_table, trigrams)
    }

    fn larger_generator() -> Generator<u8, f32> {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let values: Vec<String> =
            std::fs::read_to_string(format!("{}/../resources/Moby_Names_M_lc.txt", manifest_dir))
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

        let trigrams = create_trigrams(&symbolified_values);
        Generator::from_trigrams(symbol_table, trigrams)
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
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m: String = g.generate(1, &mut rng, &renderer).unwrap()[0].clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_simple_2() {
        let mut rng = rand::thread_rng();
        let g = simple_generator_2();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m: String = g.generate(1, &mut rng, &renderer).unwrap()[0].clone();
        assert_eq!(m, "word");
    }

    #[test]
    pub fn generate_prefix() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m: String = g
            .generate_with_prefix("hel".as_bytes(), 1, &mut rng, &renderer)
            .unwrap()[0]
            .clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_prefix_empty() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m: String = g
            .generate_with_prefix("".as_bytes(), 1, &mut rng, &renderer)
            .unwrap()[0]
            .clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_suffix() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m: String = g
            .generate_with_suffix("llo".as_bytes(), 1, &mut rng, &renderer)
            .unwrap()[0]
            .clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_suffix_empty() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m: String = g
            .generate_with_suffix("".as_bytes(), 1, &mut rng, &renderer)
            .unwrap()[0]
            .clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_with_prefix_and_suffix() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m: String = g
            .generate_with_prefix_and_suffix("h".as_bytes(), "o".as_bytes(), 1, &mut rng, &renderer)
            .unwrap()[0]
            .clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_with_prefix_and_suffix_big() {
        let mut rng = rand::thread_rng();
        let g = larger_generator();
        let prefix_str = "h";
        let suffix_str = "y";
        let prefix = prefix_str.as_bytes();
        let suffix = suffix_str.as_bytes();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m = g
            .generate_with_prefix_and_suffix(prefix, suffix, 10, &mut rng, &renderer)
            .unwrap();
        for v in m {
            assert!(
                v.starts_with(prefix_str) && v.ends_with(suffix_str),
                "Expected {}..{} but got {}",
                prefix_str,
                suffix_str,
                v,
            );
        }
    }
}
