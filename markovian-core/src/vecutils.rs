pub trait Sortable<T> {
    fn sorted(&self) -> Self;
}

pub trait Reversible {
    fn reversed(&self) -> Self;
}

impl<T> Sortable<T> for Vec<T>
where
    T: Ord + Clone,
{
    fn sorted(&self) -> Vec<T> {
        let mut r = self.clone();
        r.sort();
        r
    }
}

impl<T> Reversible for Vec<T>
where
    T: Clone,
{
    fn reversed(&self) -> Vec<T> {
        let mut r = self.clone();
        r.reverse();
        r
    }
}

pub fn select_by_lowest_value<T, F, V>(s: &[T], f: &F) -> Vec<T>
where
    V: Ord,
    F: Fn(&T) -> V,
    T: Clone,
{
    let min = s.iter().map(f).min();
    match min {
        None => vec![],
        Some(min) => s.iter().filter(|v| f(v) == min).cloned().collect(),
    }
}
