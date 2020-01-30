use std::collections::BTreeMap;

pub struct SequenceMapNode<K, T> {
    value: Option<T>,
    children: BTreeMap<K, SequenceMapNode<K, T>>,
}

impl<K, T> SequenceMapNode<K, T>
where
    K: Ord + Clone,
{
    pub fn transform_prefix<'a>(&self, seq: &'a [K]) -> Option<(&T, &'a [K])> {
        if !seq.is_empty() {
            if let Some(child) = self.children.get(&seq[0]) {
                if let Some(x) = child.transform_prefix(&seq[1..]) {
                    return Some(x);
                }
            }
        }
        self.value.as_ref().map(|v| (v, seq))
    }

    //TODO: Does this need a return value
    pub fn insert<'a>(&mut self, seq: &'a [K], value: Option<T>) {
        if seq.is_empty() {
            self.value = value;
            return;
        }
        self.children
            .entry(seq[0].clone())
            .or_insert_with(|| SequenceMapNode {
                value: None,
                children: BTreeMap::new(),
            })
            .insert(&seq[1..], value)
    }

    pub fn new() -> SequenceMapNode<K, T> {
        SequenceMapNode {
            value: None,
            children: BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_simple() {
        let mut seq_node: SequenceMapNode<i32, String> = SequenceMapNode {
            value: None,
            children: BTreeMap::new(),
        };
        seq_node.children.insert(
            1,
            SequenceMapNode {
                value: Some("hello".to_string()),
                children: BTreeMap::new(),
            },
        );
        let p = [1, 2, 3];
        let r = seq_node.transform_prefix(&p);
        assert_eq!(Some((&("hello".to_string()), &p[1..])), r);

        let p = [3, 2, 1];
        let r = seq_node.transform_prefix(&p);
        assert_eq!(None, r);
    }

    #[test]
    fn prefix_insert() {
        let mut seq_node: SequenceMapNode<i32, String> = SequenceMapNode {
            value: None,
            children: BTreeMap::new(),
        };

        seq_node.insert(&[1, 2], Some("Hi".to_string()));

        let p = [1, 2, 3];
        let r = seq_node.transform_prefix(&p);
        assert_eq!(Some((&("Hi".to_string()), &p[2..])), r);
    }

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
