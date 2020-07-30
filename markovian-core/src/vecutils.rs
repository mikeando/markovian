pub trait Sortable<T> {
    fn sorted(&self) -> Self;
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
