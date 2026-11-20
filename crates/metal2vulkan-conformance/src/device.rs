//! Which device a replay opens ([`crate::rt_side::RtDevice::open`]).

/// Which physical devices to open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Venues {
    /// A GPU: discrete, integrated or virtual.
    Gpu,
    /// A CPU implementation (lavapipe), which the no-Metal replay leg runs on.
    Cpu,
}
