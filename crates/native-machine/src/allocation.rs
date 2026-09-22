//! Thread-local allocation counter behind the global allocator.
//!
//! Counting is gated by a thread-local flag, so the steady-state cost when no
//! measurement is active is one thread-local load per allocation. Tests and
//! benchmarks opt in through [`track`].

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

pub struct CountingAllocator;
pub struct TrackingGuard;

// SAFETY: each operation delegates to the platform allocator and only adds
// a thread-local counter update when the caller opts into tracking.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        TRACKING.with(|tracking| {
            if tracking.get() {
                ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
            }
        });
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        System.dealloc(pointer, layout);
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

pub fn track() -> TrackingGuard {
    ALLOCATIONS.with(|allocations| allocations.set(0));
    TRACKING.with(|tracking| tracking.set(true));
    TrackingGuard
}

impl TrackingGuard {
    pub fn count(&self) -> usize {
        ALLOCATIONS.with(Cell::get)
    }
}

impl Drop for TrackingGuard {
    fn drop(&mut self) {
        TRACKING.with(|tracking| tracking.set(false));
    }
}
