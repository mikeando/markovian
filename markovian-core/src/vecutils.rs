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
