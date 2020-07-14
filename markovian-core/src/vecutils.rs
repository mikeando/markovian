pub fn replace_pair<T>(v: Vec<T>, key: (&T, &T), value: &T) -> Vec<T>
where
    T: Eq + Clone,
{
    let mut result: Vec<T> = vec![];
    let mut skip = false;
    for i in 0..v.len() - 1 {
        if skip {
            skip = false;
            continue;
        }
        if (&v[i] == key.0) && (&v[i + 1] == key.1) {
            result.push(value.clone());
            skip = true;
        } else {
            result.push(v[i].clone());
        }
    }
    if !skip {
        v.last()
            .iter()
            .cloned()
            .for_each(|s: &T| result.push(s.clone()));
    }
    result
}
