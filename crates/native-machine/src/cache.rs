//! Caller-owned, bounded cache for content-addressed encoded tiles.

use thiserror::Error;

const MAX_ADDRESS_BYTES: usize = 80;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InsertStatus {
    Hit,
    Inserted,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum TileCacheError {
    #[error("tile address is empty")]
    EmptyAddress,
    #[error("tile address is {actual} bytes, exceeding the maximum of {maximum}")]
    AddressTooLong { actual: usize, maximum: usize },
    #[error("tile bytes are empty")]
    EmptyTile,
    #[error("tile cache entry capacity is exhausted")]
    EntriesFull,
    #[error("tile cache needs {required} bytes but only {available} remain")]
    StorageFull { required: usize, available: usize },
    #[error("tile address already exists with different bytes")]
    AddressCollision,
}

#[derive(Clone, Copy)]
pub struct TileCacheEntry {
    address: [u8; MAX_ADDRESS_BYTES],
    address_len: usize,
    offset: usize,
    length: usize,
}

impl TileCacheEntry {
    pub const EMPTY: Self = Self {
        address: [0; MAX_ADDRESS_BYTES],
        address_len: 0,
        offset: 0,
        length: 0,
    };
}

pub struct TileCache<'a> {
    entries: &'a mut [TileCacheEntry],
    storage: &'a mut [u8],
    len: usize,
    used_bytes: usize,
}

impl<'a> TileCache<'a> {
    pub fn new(entries: &'a mut [TileCacheEntry], storage: &'a mut [u8]) -> Self {
        entries.fill(TileCacheEntry::EMPTY);
        Self {
            entries,
            storage,
            len: 0,
            used_bytes: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    pub fn get(&self, address: &str) -> Option<&[u8]> {
        let index = self.find(address)?;
        let entry = self.entries[index];
        Some(&self.storage[entry.offset..entry.offset + entry.length])
    }

    pub fn insert(&mut self, address: &str, bytes: &[u8]) -> Result<InsertStatus, TileCacheError> {
        if address.is_empty() {
            return Err(TileCacheError::EmptyAddress);
        }
        if address.len() > MAX_ADDRESS_BYTES {
            return Err(TileCacheError::AddressTooLong {
                actual: address.len(),
                maximum: MAX_ADDRESS_BYTES,
            });
        }
        if bytes.is_empty() {
            return Err(TileCacheError::EmptyTile);
        }
        if let Some(index) = self.find(address) {
            let entry = self.entries[index];
            let cached = &self.storage[entry.offset..entry.offset + entry.length];
            return if cached == bytes {
                Ok(InsertStatus::Hit)
            } else {
                Err(TileCacheError::AddressCollision)
            };
        }
        if self.len == self.entries.len() {
            return Err(TileCacheError::EntriesFull);
        }
        let available = self.storage.len() - self.used_bytes;
        if bytes.len() > available {
            return Err(TileCacheError::StorageFull {
                required: bytes.len(),
                available,
            });
        }

        let offset = self.used_bytes;
        self.storage[offset..offset + bytes.len()].copy_from_slice(bytes);
        let mut entry = TileCacheEntry::EMPTY;
        entry.address[..address.len()].copy_from_slice(address.as_bytes());
        entry.address_len = address.len();
        entry.offset = offset;
        entry.length = bytes.len();
        self.entries[self.len] = entry;
        self.len += 1;
        self.used_bytes += bytes.len();
        Ok(InsertStatus::Inserted)
    }

    fn find(&self, address: &str) -> Option<usize> {
        self.entries[..self.len]
            .iter()
            .position(|entry| entry.address[..entry.address_len] == *address.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inserts_and_deduplicates_by_address() {
        let mut entries = [TileCacheEntry::EMPTY; 2];
        let mut storage = [0_u8; 8];
        let mut cache = TileCache::new(&mut entries, &mut storage);

        assert_eq!(
            cache.insert("sha256:first", &[1, 2, 3, 4]),
            Ok(InsertStatus::Inserted)
        );
        assert_eq!(
            cache.insert("sha256:first", &[1, 2, 3, 4]),
            Ok(InsertStatus::Hit)
        );
        assert_eq!(cache.get("sha256:first"), Some(&[1, 2, 3, 4][..]));
        assert_eq!(cache.get("sha256:missing"), None);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.used_bytes(), 4);
    }

    #[test]
    fn refuses_capacity_overflow_and_address_collisions() {
        let mut entries = [TileCacheEntry::EMPTY; 1];
        let mut storage = [0_u8; 4];
        let mut cache = TileCache::new(&mut entries, &mut storage);
        cache
            .insert("sha256:first", &[1, 2, 3, 4])
            .expect("first tile inserts");

        assert_eq!(
            cache.insert("sha256:first", &[4, 3, 2, 1]),
            Err(TileCacheError::AddressCollision)
        );
        assert_eq!(
            cache.insert("sha256:second", &[5, 6, 7, 8]),
            Err(TileCacheError::EntriesFull)
        );

        let mut entries = [TileCacheEntry::EMPTY; 2];
        let mut storage = [0_u8; 3];
        let mut cache = TileCache::new(&mut entries, &mut storage);
        assert_eq!(
            cache.insert("sha256:first", &[1, 2, 3, 4]),
            Err(TileCacheError::StorageFull {
                required: 4,
                available: 3
            })
        );
    }

    #[test]
    fn cache_hit_does_not_allocate() {
        let mut entries = [TileCacheEntry::EMPTY; 1];
        let mut storage = [0_u8; 4];
        let mut cache = TileCache::new(&mut entries, &mut storage);
        let insert_allocations = {
            let guard = crate::allocation::track();
            assert_eq!(
                cache.insert("sha256:first", &[1, 2, 3, 4]),
                Ok(InsertStatus::Inserted)
            );
            guard.count()
        };

        let hit_allocations = {
            let guard = crate::allocation::track();
            assert_eq!(cache.get("sha256:first"), Some(&[1, 2, 3, 4][..]));
            guard.count()
        };
        assert_eq!(insert_allocations, 0);
        assert_eq!(hit_allocations, 0);
    }

    #[test]
    fn rejects_invalid_addresses_and_empty_tiles() {
        let mut entries = [TileCacheEntry::EMPTY; 1];
        let mut storage = [0_u8; 4];
        let mut cache = TileCache::new(&mut entries, &mut storage);

        assert_eq!(cache.insert("", &[1]), Err(TileCacheError::EmptyAddress));
        assert_eq!(
            cache.insert(&"x".repeat(MAX_ADDRESS_BYTES + 1), &[1]),
            Err(TileCacheError::AddressTooLong {
                actual: MAX_ADDRESS_BYTES + 1,
                maximum: MAX_ADDRESS_BYTES
            })
        );
        assert_eq!(
            cache.insert("sha256:first", &[]),
            Err(TileCacheError::EmptyTile)
        );
    }
}
