/// A dynamic descriptor pool/set allocator.
/// We allocate descriptor pools with ```MAX_SETS``` reserved.
/// Descriptor sets are cached and reused. If the descriptor set is no longer being used by the GPU,
/// and hasn't been used in awhile (```MAX_AGE_MS```), the descriptor set is freed.
/// If a descriptor pool has ```MAX_SETS```, it is immediately freed, since the implication is that
/// the application hasn't touched the pool in awhile...
use super::*;
use ash::{VkResult, vk};
use std::collections::HashMap;

const MAX_SETS: u32 = 32;
const MAX_AGE_MS: u128 = 1_000;

struct DescriptorSetAllocation {
    handle: vk::DescriptorSet,
    pool: vk::DescriptorPool,
    graphics_queue_time: u64,
    last_access_time: std::time::SystemTime,
}

struct DescriptorPoolStats {
    sets_remaining: u32,
}

pub struct DescriptorSetAllocator {
    pools: HashMap<vk::DescriptorPool, DescriptorPoolStats>,
    sizes: Vec<vk::DescriptorPoolSize>,
    sets: HashMap<Vec<vk::Buffer>, DescriptorSetAllocation>,
}

pub enum DescriptorSetCacheLookup {
    Hit(vk::DescriptorSet),
    Miss(vk::DescriptorSet),
}

impl DescriptorSetAllocator {
    pub fn make(bindings: &[vk::DescriptorSetLayoutBinding<'_>]) -> Self {
        let mut sizes = HashMap::new();
        for binding in bindings {
            *sizes.entry(binding.descriptor_type).or_insert(0 as u32) += binding.descriptor_count;
        }
        let sizes: Vec<_> = sizes
            .iter()
            .map(|(ty, descriptor_count)| vk::DescriptorPoolSize {
                ty: *ty,
                descriptor_count: *descriptor_count,
            })
            .collect();

        Self {
            pools: HashMap::new(),
            sizes: sizes,
            sets: HashMap::new(),
        }
    }

    pub fn destroy(&self, device: &Device) {
        for pool in self.pools.keys() {
            unsafe {
                device.destroy_descriptor_pool(*pool, None);
            }
        }
    }

    pub fn garbage_collection(&mut self, ctx: &DeviceContext, graphics_queue_time: u64) {
        let sets: Vec<DescriptorSetAllocation> = self
            .sets
            .extract_if(|_, set| {
                let age_ms = std::time::SystemTime::now()
                    .duration_since(set.last_access_time)
                    .unwrap()
                    .as_millis();
                graphics_queue_time >= set.graphics_queue_time && age_ms >= MAX_AGE_MS
            })
            .map(|(_, v)| v)
            .collect();

        for set in sets {
            unsafe {
                ctx.device
                    .free_descriptor_sets(set.pool, &[set.handle])
                    .expect("failed to free descriptor sets");
            }

            let pool = self.pools.get_mut(&set.pool).unwrap();
            pool.sets_remaining += 1;

            if pool.sets_remaining == MAX_SETS {
                self.pools.remove(&set.pool);
                unsafe {
                    ctx.device.destroy_descriptor_pool(set.pool, None);
                }
            }
        }
    }

    pub fn allocate_descriptor_set(
        &mut self,
        ctx: &DeviceContext,
        set_layout: vk::DescriptorSetLayout,
        buffers: Vec<vk::Buffer>,
    ) -> DescriptorSetCacheLookup {
        if self.pools.is_empty() {
            let _ = self.create_descriptor_pool(ctx);
        }

        // Cache hit
        if let Some(set) = self.sets.get_mut(&buffers).as_mut() {
            set.graphics_queue_time = ctx.graphics_timeline.value;
            set.last_access_time = std::time::SystemTime::now();
            return DescriptorSetCacheLookup::Hit(set.handle);
        }

        // Cache miss
        let mut pools = std::mem::take(&mut self.pools);
        let handle = (|| {
            for (pool, stats) in &mut pools {
                let result = self.allocate_descriptor_set_from_pool(
                    ctx,
                    *pool,
                    stats,
                    &buffers,
                    set_layout,
                );
                match result {
                    Ok(handle) => {
                        return Some(handle);
                    }
                    Err(err) => {
                        if err == vk::Result::ERROR_OUT_OF_POOL_MEMORY {
                            continue;
                        } else {
                            panic!("descriptor set allocation failed (vk::Result::{})", err);
                        }
                    }
                }
            }
            None
        })();
        self.pools = pools;

        if let Some(handle) = handle {
            return DescriptorSetCacheLookup::Miss(handle);
        } else {
            // All pools are full. Allocate a new one.
            let pool = self.create_descriptor_pool(ctx);
            let mut pools = std::mem::take(&mut self.pools);
            let stats = pools.get_mut(&pool).unwrap();
            let handle = self
                .allocate_descriptor_set_from_pool(ctx, pool, stats, &buffers, set_layout)
                .expect("failed to allocate descriptor set");
            self.pools = pools;
            DescriptorSetCacheLookup::Miss(handle)
        }
    }

    fn allocate_descriptor_set_from_pool(
        &mut self,
        ctx: &DeviceContext,
        pool: vk::DescriptorPool,
        stats: &mut DescriptorPoolStats,
        buffers: &Vec<vk::Buffer>,
        set_layout: vk::DescriptorSetLayout,
    ) -> VkResult<vk::DescriptorSet> {
        let set_layouts = [set_layout];
        let dsai = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&set_layouts);
        let result = unsafe { ctx.device.allocate_descriptor_sets(&dsai) };
        match result {
            Ok(sets) => {
                let handle = sets[0];
                self.sets.insert(
                    buffers.clone(),
                    DescriptorSetAllocation {
                        handle: handle,
                        pool: pool,
                        graphics_queue_time: ctx.graphics_timeline.value,
                        last_access_time: std::time::SystemTime::now(),
                    },
                );
                stats.sets_remaining -= 1;
                return Ok(handle);
            }
            Err(err) => {
                return Err(err);
            }
        }
    }

    fn create_descriptor_pool(&mut self, ctx: &DeviceContext) -> vk::DescriptorPool {
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(MAX_SETS)
            .pool_sizes(&self.sizes);
        let pool = unsafe {
            ctx.device
                .create_descriptor_pool(&create_info, None)
                .expect("failed to create descriptor pool")
        };
        self.pools.insert(
            pool,
            DescriptorPoolStats {
                sets_remaining: MAX_SETS,
            },
        );
        pool
    }
}
