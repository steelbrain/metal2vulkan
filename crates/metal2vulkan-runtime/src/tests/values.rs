use crate::error::{Error, Invalid, Problem};
use crate::values::{
    CommandBufferStatus, CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode,
};

/// Metal's raw values: `MTLResourceStorageModePrivate` is `2 << 4`,
/// `MTLResourceCPUCacheModeWriteCombined` 1, `MTLResourceHazardTrackingModeUntracked` `1 << 8`.
#[test]
fn resource_options_pack_as_metal_does() {
    assert_eq!(ResourceOptions::shared().raw(), 0);
    assert_eq!(ResourceOptions::private().raw(), 0x20);
    let options = ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::WriteCombined,
        HazardTrackingMode::Tracked,
    );
    assert_eq!(options.raw(), 0x10 | 0x1 | 0x200);
    let back = ResourceOptions::from_raw(options.raw()).unwrap();
    assert_eq!(back, options);
    assert_eq!(back.storage_mode(), Ok(StorageMode::Managed));
    assert_eq!(back.cpu_cache_mode(), Ok(CpuCacheMode::WriteCombined));
    assert_eq!(back.hazard_tracking_mode(), Ok(HazardTrackingMode::Tracked));
    for raw in [0x0, 0x20, 0x30, 0x101, 0x221] {
        assert_eq!(ResourceOptions::from_raw(raw).unwrap().raw(), raw);
    }
}

/// A resolved hazard tracking mode replaces the field and nothing else: the storage mode, the
/// cache mode and Metal's private bit 24 all survive it, which is what lets a resource report the
/// mode it resolved (`Texture::resource_options`) rather than its descriptor's `Default`.
#[test]
fn a_resolved_hazard_mode_replaces_its_field_alone() {
    let private = ResourceOptions::private();
    assert_eq!(
        private.hazard_tracking_mode(),
        Ok(HazardTrackingMode::Default)
    );
    for (mode, raw) in [
        (HazardTrackingMode::Tracked, 0x220),
        (HazardTrackingMode::Untracked, 0x120),
        (HazardTrackingMode::Default, 0x20),
    ] {
        let held = private.with_hazard_tracking_mode(mode);
        assert_eq!(held.raw(), raw, "{mode:?}");
        assert_eq!(held.hazard_tracking_mode(), Ok(mode));
        assert_eq!(held.storage_mode(), Ok(StorageMode::Private));
        assert_eq!(held.cpu_cache_mode(), Ok(CpuCacheMode::DefaultCache));
    }
    // A shared texture outside a heap is Metal's 0x200, and the private bit survives the field.
    assert_eq!(
        ResourceOptions::shared()
            .with_hazard_tracking_mode(HazardTrackingMode::Tracked)
            .raw(),
        0x200
    );
    let bit = ResourceOptions::from_raw(ResourceOptions::PRIVATE_BIT | 0x120).unwrap();
    assert_eq!(
        bit.with_hazard_tracking_mode(HazardTrackingMode::Tracked)
            .raw(),
        ResourceOptions::PRIVATE_BIT | 0x220
    );
}

/// Metal's private bit 24 is taken beside the modes, as plain Metal takes it, and only the device's
/// making of a resource drops it; bit 25 stays undefined.
#[test]
fn the_private_bit_is_taken_and_dropped_apart() {
    let raw = 0x100_0000 | 0x20;
    let options = ResourceOptions::from_raw(raw).unwrap();
    assert_eq!(options.raw(), raw);
    assert_eq!(options.storage_mode(), Ok(StorageMode::Private));
    assert_eq!(options.cpu_cache_mode(), Ok(CpuCacheMode::DefaultCache));
    assert_eq!(
        options.hazard_tracking_mode(),
        Ok(HazardTrackingMode::Default)
    );
    assert_eq!(
        options.without_private_bit(),
        (ResourceOptions::private(), true)
    );
    assert_eq!(
        ResourceOptions::private().without_private_bit(),
        (ResourceOptions::private(), false)
    );
    assert!(ResourceOptions::from_raw(0x200_0000).is_err());
}

#[test]
fn undefined_raw_values_are_refused() {
    let undefined = |ty, raw| {
        Err::<ResourceOptions, _>(Error::Invalid(Invalid::new(
            "from_raw",
            Problem::UndefinedValue { ty, raw },
        )))
    };
    assert_eq!(
        ResourceOptions::from_raw(0x1000),
        undefined("MTLResourceOptions", 0x1000)
    );
    assert_eq!(
        ResourceOptions::from_raw(0x40),
        undefined("MTLStorageMode", 4)
    );
    assert_eq!(
        ResourceOptions::from_raw(0x2),
        undefined("MTLCPUCacheMode", 2)
    );
    assert_eq!(
        ResourceOptions::from_raw(0x300),
        undefined("MTLHazardTrackingMode", 3)
    );
    assert!(StorageMode::from_raw(4).is_err());
    assert!(CpuCacheMode::from_raw(2).is_err());
    assert!(HazardTrackingMode::from_raw(3).is_err());
}

#[test]
fn every_mode_round_trips_its_raw_value() {
    for mode in [
        StorageMode::Shared,
        StorageMode::Managed,
        StorageMode::Private,
        StorageMode::Memoryless,
    ] {
        assert_eq!(StorageMode::from_raw(mode.raw()), Ok(mode));
    }
    for mode in [CpuCacheMode::DefaultCache, CpuCacheMode::WriteCombined] {
        assert_eq!(CpuCacheMode::from_raw(mode.raw()), Ok(mode));
    }
    for mode in [
        HazardTrackingMode::Default,
        HazardTrackingMode::Untracked,
        HazardTrackingMode::Tracked,
    ] {
        assert_eq!(HazardTrackingMode::from_raw(mode.raw()), Ok(mode));
    }
    let statuses = [
        CommandBufferStatus::NotEnqueued,
        CommandBufferStatus::Enqueued,
        CommandBufferStatus::Committed,
        CommandBufferStatus::Scheduled,
        CommandBufferStatus::Completed,
        CommandBufferStatus::Error,
    ];
    for (raw, status) in statuses.into_iter().enumerate() {
        assert_eq!(status.raw(), raw as u64);
    }
}
