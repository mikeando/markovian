// The idea behind the synced heap is to have a heap that can be kept in-sync with a secondary data structure.
// The heap will support fast addition, deletion and score adjustment.
//
// The data structure consists of two pieces, the heap itself and the client.
// Entries in the heap are identified to the client by a client specific key, 
// while on the heap side they are identified by their position in the heap (a usize)

use std::collections::HashMap;

mod heap_internals {
    
    pub trait SwapTracker {
        fn swap(&mut self, index_a:usize, index_b:usize);
    }

    pub struct SimpleHeap<'a, 'b, X, CMP, S> {
        pub heap:&'a mut [X],
        pub heap_len: usize,
        pub a_less_than_b:CMP,
        //TODO: I'm not sure this should be a reference.
        pub swap_tracker: &'b mut S,
    }

    impl <'a, 'b, X, CMP, S> SimpleHeap<'a, 'b, X, CMP, S>
        where CMP: Fn(&X, &X) -> bool,
        S: SwapTracker
    {
        pub fn bubble_up(&mut self, index:usize) 
            {
            let left = 2 * index + 1;
            let right = 2 * index + 2;
            let mut largest = index;
            
            if left < self.heap_len && (self.a_less_than_b)(&self.heap[largest], &self.heap[left]) {
                largest = left;
            }

            if right < self.heap_len && (self.a_less_than_b)(&self.heap[largest], &self.heap[right]) {
                largest = right;
            }
            if largest != index {
                self.heap.swap(index, largest);
                self.swap_tracker.swap(index,largest);
                self.bubble_up(largest);
            }
        }

        pub fn bubble_down(&mut self, index:usize) 
        {
            if index == 0 {
                return
            }
            let parent = (index - 1) / 2;
            if (self.a_less_than_b)(&self.heap[parent], &self.heap[index]) {
                self.heap.swap(index,parent);
                self.swap_tracker.swap(index,parent);
                self.bubble_down(parent);
            }
        }

        pub fn n_leaf_nodes(&self) -> usize {
            (self.heap_len+1) / 2
        }
    
        pub fn n_inner_nodes(&self) -> usize {
            self.heap_len/2
        }

        pub fn heapify(&mut self) 
        {
            let nn = self.n_inner_nodes();
            for i in (0..nn).rev() {
                self.bubble_up(i);
            }
        }

        pub fn is_heap(&self) -> bool {
            let nn = self.n_inner_nodes();
            for index in 0..nn {
                let left = 2 * index + 1;
                let right = 2 * index + 2;
                
                if left < self.heap_len && (self.a_less_than_b)(&self.heap[index], &self.heap[left]) {
                    return false;
                }
        
                if right < self.heap_len && (self.a_less_than_b)(&self.heap[index], &self.heap[right]) {
                    return false;
                }
            }
            true
        }

        // We can only pop a real value if we actually own the backing array.
        // But since we've only a reference to an array - we can only pop a reference
        pub fn pop(&mut self) -> Option<&X> 
        {
            if self.heap_len == 0 {
                return None;
            }
            self.heap.swap(0, self.heap_len-1);
            self.heap_len -= 1;
            self.bubble_up(0);
            Some(&self.heap[self.heap_len])
        }

        pub fn push(&mut self, v:X) {
            //TODO: Check we have enough space!
            self.heap[self.heap_len] = v;
            self.heap_len += 1;
            self.bubble_down(self.heap_len-1);
        }
    }


}

struct DoNothingTracker {

}

impl heap_internals::SwapTracker for DoNothingTracker {
    fn swap(&mut self, index_a:usize, index_b:usize) {}
}


struct SyncedHeap<K> {
    vs:Vec<(f32,K)>,
}

impl <K> SyncedHeap<K> {
    pub fn new<C>(_v:C) -> SyncedHeap<K> {
        SyncedHeap{ vs:vec![] }
    }

    pub fn heapify<C>(_v:C, mut vs:Vec<(f32,K)>) -> SyncedHeap<K> {
        let mut tracker = DoNothingTracker{};
        let n = vs.len();
        let mut h = heap_internals::SimpleHeap {
            heap: &mut vs[..],
            heap_len: n,
            a_less_than_b: |a:&(f32,K),b:&(f32,K)| a.0 < b.0,
            swap_tracker: &mut tracker,
            
        };
        h.heapify();
        SyncedHeap{vs}
    }

    pub fn add(&mut self, v:(f32, K)) {
        let mut tracker = DoNothingTracker{};
        self.vs.push(v);
        let n = self.vs.len();
        let mut h = heap_internals::SimpleHeap {
            heap: &mut self.vs[..],
            heap_len: n,
            a_less_than_b: |a:&(f32,K),b:&(f32,K)| a.0 < b.0,
            swap_tracker: &mut tracker,
        };
        h.bubble_down(n-1);
    }

    pub fn pop(&mut self) -> Option<(f32,K)> {
        let mut tracker = DoNothingTracker{};
        let n = self.vs.len();
        let mut h = heap_internals::SimpleHeap {
            heap: &mut self.vs[..],
            heap_len: n,
            a_less_than_b: |a:&(f32,K),b:&(f32,K)| a.0 < b.0,
            swap_tracker: &mut tracker,
        };
        h.pop();
        self.vs.pop()
    }
}




#[cfg(test)]
mod test {

    use super::*;

    struct EmptyClient {
    }

    impl EmptyClient {
        pub fn new() -> EmptyClient {
            EmptyClient{}
        }
    }

    struct TestSwapTracker {
        entries:Vec<(usize,usize)>
    }

    impl TestSwapTracker {
        pub fn new() -> TestSwapTracker {
            TestSwapTracker{ entries:vec![] }
        }
    }

    impl heap_internals::SwapTracker for TestSwapTracker {
        fn swap(&mut self, index_a:usize, index_b:usize) {
            if index_a < index_b {
                self.entries.push((index_a,index_b));
            } else {
                self.entries.push((index_b,index_a));
            }
        }
    }

    #[test]
    fn test_heapify() {
        let mut v:Vec<i32> = vec![4,7,3,2,8,1];
        let mut tracker = TestSwapTracker::new();
        let n = v.len();
        let mut h = heap_internals::SimpleHeap {
            heap: &mut v[..],
            heap_len: n,
            a_less_than_b: |a:&i32, b:&i32| a<b,
            swap_tracker: &mut tracker,
        };

        assert!(!h.is_heap());
        h.heapify();
        assert_eq!(&h.heap[0..h.heap_len], &vec![8,7,3,2,4,1][..]);
        assert!(h.is_heap());

        let p = h.pop();
        assert_eq!(p, Some(&8));
        assert_eq!(&h.heap[0..h.heap_len], &vec![7,4,3,2,1][..]);

        let p = h.pop();
        assert_eq!(p, Some(&7));

        let p = h.pop();
        assert_eq!(p, Some(&4));

        let p = h.pop();
        assert_eq!(p, Some(&3));

        let p = h.pop();
        assert_eq!(p, Some(&2));

        let p = h.pop();
        assert_eq!(p, Some(&1));

        let p = h.pop();
        assert_eq!(p, None); 
    }

    #[test]
    pub fn test_bubble_up() {
        // Start with a valid heap
        let mut v:Vec<i32> = vec![8, 7, 3, 2, 4, 1];
        let mut tracker = TestSwapTracker::new();
        let n = v.len();
        let mut h = heap_internals::SimpleHeap {
            heap: &mut v[..],
            heap_len: n,
            a_less_than_b: |a:&i32, b:&i32| a<b,
            swap_tracker: &mut tracker,
        };

        assert!(h.is_heap());
        // break an entry
        h.heap[1] = 0;
        assert!(!h.is_heap());

        h.bubble_up(1);

        // Check invariant restored
        assert!(h.is_heap());

        //For this simple case we know exactly what the result should be, and what the order of the swaps should be
        //  0  1  2  3  4  5
        //  8  0  3  2  4  1
        //     |     ^  ^
        //     +-----+--+
        //
        //  swap 1 and 4
        //
        // 8  4  3  2  0  1
        assert_eq!(v, vec![8,4,3,2,0,1]);
        assert_eq!(tracker.entries, vec![(1,4)]);
    }

    #[test]
    pub fn test_push() {
        // Start with a valid heap
        let mut vs:Vec<i32> = vec![8, 7, 3, 2, 4, 1, 999];
        let mut tracker = TestSwapTracker::new();
        let n = vs.len();
        let mut q:Vec<i32> = vec![];
        let mut h = heap_internals::SimpleHeap {
            heap: &mut vs[..],
            heap_len: n-1,
            a_less_than_b: |a:&i32, b:&i32| a<b,
            swap_tracker: &mut tracker,
        };

        assert!(h.is_heap());
        // break an entry
        h.push(9);
        // heap_internals::push(9, &mut vs, h.a_less_than_b, h.swap_tracker);
        assert!(h.is_heap(), "Heap is {:?}", vs);

        //For this simple case we know exactly what the result should be, and what the order of the swaps should be
        //  0  1  2  3  4  5  6
        //  8  7  3  2  4  1  9
        //        ^           |
        //        +-----------+
        //
        //  swap 2 and 6
        //
        //  8  7  9  2  4  1  3
        //  ^     |
        //  +-----+
        //
        // swap 1 and 2
        //  9  7  8  2  4  1  3
        assert_eq!(vs, vec![9,7,8,2,4,1,3]);
        assert_eq!(tracker.entries, vec![(2,6), (0,2)]);
    }


    #[test]
    fn simple_test() {
        let c = EmptyClient::new();
        let mut h = SyncedHeap::new(c);
        h.add((1.0,()));
        h.add((2.0,()));
        h.add((3.0,()));
        assert_eq!(h.pop(), Some((3.0,())));
        assert_eq!(h.pop(), Some((2.0,())));
        assert_eq!(h.pop(), Some((1.0,())));
        assert_eq!(h.pop(), None);
    }

    #[test]
    fn simple_test2() {
        let c = EmptyClient::new();
        let vs = vec![
            (1.0,()),
            (2.0,()),
            (3.0,())
        ];
        let mut h = SyncedHeap::heapify(c, vs);
        assert_eq!(h.pop(), Some((3.0,())));
        assert_eq!(h.pop(), Some((2.0,())));
        assert_eq!(h.pop(), Some((1.0,())));
        assert_eq!(h.pop(), None);
    }

    struct Donkey {
        speed:f32,
        heap_id:usize,
    }

    impl Donkey {
        pub fn speed(&self) -> f32 { self.speed }
    }

    struct MapProxy {}

    impl MapProxy {
        pub fn new() -> MapProxy {
            MapProxy{}
        }
    }

    #[test]
    fn simple_test3() {

        // We have a collection of donkeys
        let mut m: HashMap<String, Donkey> = HashMap::new();
        m.insert("John".to_string(), Donkey{ speed: 2.0, heap_id:0});
        m.insert("Fred".to_string(), Donkey{ speed: 5.0, heap_id:0});

        // We want to pair this with a record of each Donkeys speed.
        let vs:Vec<(f32, String)> =  m.iter_mut().enumerate().map( |(i, (k,v))| {
            v.heap_id = i;
            (v.speed, k.clone())
        }).collect();

        let h = SyncedHeap::heapify(MapProxy::new(/* m */), vs);

        //let (fastest_speed, fastest_id) = h.peek();
        //assert_eq!(fastest_speed, 5.0);
        //assert_eq!(fastest_id, "Fred");

        // Now we want to update one of the donkeys
        // How much auto-magic do we think we should get from this?

        // Now John should be the fastest.


    }

}