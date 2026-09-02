use ash::{Device, Instance};
use std::ops::Deref;

/// Owns a `Device` and destroys it on drop.
pub struct DeviceOwner(Device);

impl DeviceOwner {
    pub fn new(device: Device) -> Self {
        Self(device)
    }
}

impl Deref for DeviceOwner {
    type Target = Device;

    fn deref(&self) -> &Device {
        &self.0
    }
}

impl Drop for DeviceOwner {
    fn drop(&mut self) {
        unsafe {
            self.0.destroy_device(None);
        }
    }
}

/// Owns an `Instance` and destroys it on drop.
pub struct InstanceOwner(Instance);

impl InstanceOwner {
    pub fn new(instance: Instance) -> Self {
        Self(instance)
    }
}

impl Deref for InstanceOwner {
    type Target = Instance;

    fn deref(&self) -> &Instance {
        &self.0
    }
}

impl Drop for InstanceOwner {
    fn drop(&mut self) {
        unsafe {
            self.0.destroy_instance(None);
        }
    }
}
