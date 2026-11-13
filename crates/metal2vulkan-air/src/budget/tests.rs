use super::*;

fn past() -> Instant {
    // `Instant` cannot go below the platform epoch; now() is already "reached" for `>=`.
    Instant::now()
}

#[test]
fn deadline_is_transient_budget_time() {
    let budget = Budget {
        deadline: Some(past()),
        cancel: None,
    };
    let mut meter = Meter::new(&budget);
    let err = meter.tick(1).unwrap_err();
    assert_eq!(err, Error::Budget(BudgetExceeded::Time));
}

/// A meter whose deadline was ahead at its first poll and has passed since.
fn passed_after_the_first_poll() -> Meter {
    let budget = Budget::with_deadline(Instant::now().checked_add(Duration::from_millis(250)));
    let mut meter = Meter::new(&budget);
    meter.tick(1).unwrap();
    std::thread::sleep(Duration::from_millis(300));
    meter
}

#[test]
fn the_clock_is_read_once_in_clock_polls_polls() {
    let mut meter = passed_after_the_first_poll();
    for _ in 1..CLOCK_POLLS {
        meter.tick(1).unwrap();
    }
    assert_eq!(
        meter.tick(1).unwrap_err(),
        Error::Budget(BudgetExceeded::Time)
    );
}

#[test]
fn a_poll_of_clock_units_reads_the_clock() {
    let mut meter = passed_after_the_first_poll();
    let units = u32::try_from(CLOCK_UNITS).unwrap();
    assert_eq!(
        meter.tick(units).unwrap_err(),
        Error::Budget(BudgetExceeded::Time)
    );
}

#[test]
fn cancel_is_observed_at_the_next_tick() {
    let token = CancelToken::new();
    let budget = Budget {
        deadline: None,
        cancel: Some(token.clone()),
    };
    let mut meter = Meter::new(&budget);
    meter.tick(1).unwrap();
    token.cancel();
    let err = meter.tick(1).unwrap_err();
    assert_eq!(err, Error::Cancelled);
}

#[test]
fn work_cap_is_deterministic_and_permanent() {
    let mut meter = Meter::with_caps(&Budget::default(), 10, LEDGER_BYTES);
    meter.tick(10).unwrap();
    assert_eq!(meter.work(), 10);
    let err = meter.tick(1).unwrap_err();
    assert_eq!(err, Error::LimitExceeded(Limit::WorkUnits));
}

#[test]
fn work_cap_wins_over_cancel_and_deadline() {
    let token = CancelToken::new();
    token.cancel();
    let budget = Budget {
        deadline: Some(past()),
        cancel: Some(token),
    };
    let mut meter = Meter::with_caps(&budget, 0, LEDGER_BYTES);
    assert_eq!(
        meter.tick(1).unwrap_err(),
        Error::LimitExceeded(Limit::WorkUnits)
    );
}

#[test]
fn cancel_wins_over_deadline() {
    let token = CancelToken::new();
    token.cancel();
    let budget = Budget {
        deadline: Some(past()),
        cancel: Some(token),
    };
    assert_eq!(Meter::new(&budget).tick(0).unwrap_err(), Error::Cancelled);
}

#[test]
fn standard_budget_does_not_expire_immediately() {
    let mut meter = Meter::new(&Budget::standard());
    meter.tick(1).unwrap();
}

#[test]
fn ledger_refuses_past_450_mib() {
    let mut meter = Meter::new(&Budget::default());
    assert_eq!(LEDGER_BYTES, 450 * 1024 * 1024);
    meter.charge(LEDGER_BYTES as usize).unwrap();
    assert_eq!(
        meter.charge(1).unwrap_err(),
        Error::LimitExceeded(Limit::LedgerBytes)
    );
    assert_eq!(meter.ledger(), LEDGER_BYTES);
}

#[test]
fn reserve_charges_before_allocating() {
    let mut meter = Meter::new(&Budget::default());
    let mut v: Vec<u64> = Vec::new();
    let too_many = (LEDGER_BYTES / 8 + 1) as usize;
    assert_eq!(
        meter.reserve(&mut v, too_many).unwrap_err(),
        Error::LimitExceeded(Limit::LedgerBytes)
    );
    assert_eq!(v.capacity(), 0, "refusal happens before allocation");
    assert_eq!(meter.ledger(), 0);

    assert_eq!(
        meter.reserve(&mut v, usize::MAX).unwrap_err(),
        Error::LimitExceeded(Limit::LedgerBytes),
        "a byte count that overflows usize is refused, not wrapped"
    );

    meter.reserve(&mut v, 16).unwrap();
    assert!(v.capacity() >= 16);
    assert_eq!(meter.ledger(), 16 * 8);
}

/// A vector grown one push at a time holds up to twice its length; `grow` charges the capacity it
/// adds, so the ledger carries the whole capacity the allocator holds.
#[test]
fn grow_charges_the_capacity_each_growth_adds() {
    let mut meter = Meter::new(&Budget::default());
    let mut v: Vec<u64> = Vec::new();
    for i in 0..1000u64 {
        meter.grow(&mut v, 1).unwrap();
        v.push(i);
        assert_eq!(
            meter.ledger(),
            (v.capacity() * 8) as u64,
            "after {} pushes",
            i + 1
        );
    }
    // Doubling from the 4-element minimum: 4, 8, ..., 1024.
    assert_eq!(v.capacity(), 1024);
    // Within capacity nothing is charged.
    let before = meter.ledger();
    meter.grow(&mut v, 24).unwrap();
    assert_eq!(meter.ledger(), before);
    // A request past double is taken whole.
    meter.grow(&mut v, 5000).unwrap();
    assert_eq!(v.capacity(), 6000);
    assert_eq!(meter.ledger(), 6000 * 8);
}

/// A vector whose final length is known is reserved at exactly that length: the pushes that fill
/// it charge nothing and leave no doubling slack, and the ledger still carries its capacity.
#[test]
fn reserve_exact_charges_the_capacity_it_adds_and_leaves_no_slack() {
    let mut meter = Meter::with_caps(&Budget::default(), WORK_UNITS, 1000 * 8);
    let mut v: Vec<u64> = Vec::new();
    meter.reserve_exact(&mut v, 1000).unwrap();
    assert_eq!((v.capacity(), meter.ledger()), (1000, 1000 * 8));
    for i in 0..1000u64 {
        meter.grow(&mut v, 1).unwrap();
        v.push(i);
    }
    assert_eq!(v.capacity(), v.len(), "filled to its length with no growth");
    assert_eq!(meter.ledger(), 1000 * 8);
    // Within capacity nothing is charged; past the cap nothing is allocated.
    v.truncate(10);
    meter.reserve_exact(&mut v, 990).unwrap();
    assert_eq!(meter.ledger(), 1000 * 8);
    assert_eq!(
        meter.reserve_exact(&mut v, 991).unwrap_err(),
        Error::LimitExceeded(Limit::LedgerBytes)
    );
    assert_eq!(v.capacity(), 1000, "refusal happens before allocation");
    assert_eq!(
        meter.reserve_exact(&mut v, usize::MAX).unwrap_err(),
        Error::LimitExceeded(Limit::LedgerBytes),
        "a length that overflows usize is refused, not wrapped"
    );
}

#[test]
fn grow_charges_before_allocating() {
    let mut meter = Meter::with_caps(&Budget::default(), WORK_UNITS, 64);
    let mut v: Vec<u64> = Vec::new();
    meter.grow(&mut v, 8).unwrap();
    assert_eq!((v.capacity(), meter.ledger()), (8, 64));
    v.extend(0..8);
    assert_eq!(
        meter.grow(&mut v, 1).unwrap_err(),
        Error::LimitExceeded(Limit::LedgerBytes)
    );
    assert_eq!(v.capacity(), 8, "refusal happens before allocation");
    assert_eq!(
        meter.grow(&mut v, usize::MAX).unwrap_err(),
        Error::LimitExceeded(Limit::LedgerBytes),
        "a length that overflows usize is refused, not wrapped"
    );
}

/// A container charged at poll points: only growth is charged, a shrink keeps the high-water
/// mark, and a refusal leaves the charged amount unchanged.
#[test]
fn growth_charges_only_what_the_footprint_gained() {
    let mut meter = Meter::with_caps(&Budget::default(), WORK_UNITS, 100);
    let mut g = Growth::from_charged(10);
    g.charge_to(40, &mut meter).unwrap();
    assert_eq!((g.charged(), meter.ledger()), (40, 30));
    g.charge_to(25, &mut meter).unwrap();
    assert_eq!((g.charged(), meter.ledger()), (40, 30));
    assert_eq!(
        g.charge_to(200, &mut meter).unwrap_err(),
        Error::LimitExceeded(Limit::LedgerBytes)
    );
    assert_eq!((g.charged(), meter.ledger()), (40, 30));
}

#[test]
fn footprint_helpers_count_capacity() {
    let v: Vec<u32> = Vec::with_capacity(10);
    assert_eq!(vec_bytes(&v), 40);
    let m: std::collections::HashMap<u32, u64> = std::collections::HashMap::with_capacity(10);
    assert_eq!(map_bytes(&m), m.capacity() * (16 + 1));
    assert_eq!(tree_bytes(3, 8), 48);
}

#[test]
fn release_keeps_the_peak() {
    let mut meter = Meter::new(&Budget::default());
    meter.charge(100).unwrap();
    meter.release(60);
    meter.charge(10).unwrap();
    assert_eq!(meter.ledger(), 50);
    assert_eq!(meter.ledger_peak(), 100);
}

#[test]
fn fixed_caps_refuse_one_past_the_cap() {
    let meter = Meter::new(&Budget::default());
    meter
        .cap(Limit::InlinedInstructions, INLINED_INSTRUCTIONS)
        .unwrap();
    assert_eq!(
        meter
            .cap(Limit::InlinedInstructions, INLINED_INSTRUCTIONS + 1)
            .unwrap_err(),
        Error::LimitExceeded(Limit::InlinedInstructions)
    );
    assert_eq!(INLINED_INSTRUCTIONS, 1 << 21);
    assert_eq!(IRREDUCIBLE_DEPTH, 8);
}

#[test]
fn meter_caps_override_the_defaults() {
    let meter = Meter::with_caps(&Budget::default(), 5, 7);
    meter.cap(Limit::WorkUnits, 5).unwrap();
    meter.cap(Limit::LedgerBytes, 7).unwrap();
    assert!(meter.cap(Limit::WorkUnits, 6).is_err());
    assert!(meter.cap(Limit::LedgerBytes, 8).is_err());
}

#[test]
fn relative_limits_go_through_bound() {
    let meter = Meter::new(&Budget::default());
    let err = meter.cap(Limit::OutputBlocks, 1).unwrap_err();
    assert!(matches!(
        err,
        Error::Internal(InternalError {
            kind: InternalKind::Invariant,
            ..
        })
    ));
    let max = output_blocks_cap(10, 12);
    assert_eq!(max, 4 * 22 + 1024);
    meter.bound(Limit::OutputBlocks, max, max).unwrap();
    assert_eq!(
        meter.bound(Limit::OutputBlocks, max + 1, max).unwrap_err(),
        Error::LimitExceeded(Limit::OutputBlocks)
    );
    assert_eq!(output_blocks_cap(u64::MAX, 1), u64::MAX);
}

#[test]
fn every_limit_has_a_cap_policy() {
    // Exhaustive: a new `Limit` must decide between a fixed cap and an input-relative one.
    for limit in [
        Limit::InlinedInstructions,
        Limit::OutputBlocks,
        Limit::IrreducibleDepth,
        Limit::SpirvWords,
        Limit::WorkUnits,
        Limit::LedgerBytes,
        Limit::TypeCount,
        Limit::MetadataNodes,
        Limit::BitstreamNesting,
        Limit::InstructionCount,
        Limit::LayoutNesting,
        Limit::ConstructNesting,
        Limit::PrivateArenaBytes,
        Limit::CallStackBytes,
    ] {
        let relative = match limit {
            Limit::OutputBlocks => true,
            Limit::InlinedInstructions
            | Limit::IrreducibleDepth
            | Limit::SpirvWords
            | Limit::WorkUnits
            | Limit::LedgerBytes
            | Limit::TypeCount
            | Limit::MetadataNodes
            | Limit::BitstreamNesting
            | Limit::InstructionCount
            | Limit::LayoutNesting
            | Limit::ConstructNesting
            | Limit::PrivateArenaBytes
            | Limit::CallStackBytes => false,
        };
        assert_eq!(limit.fixed_cap().is_none(), relative, "{limit:?}");
    }
}

#[test]
fn budget_types_are_send_and_sync() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Meter>();
    send_sync::<Budget>();
    send_sync::<CancelToken>();
}

/// What a piece of work charged one meter, spent on another, leaves the second where the work
/// itself would have: the same work units, the same ledger, the same peak, and the same refusals
/// at the same caps. Measured from a ledger and a peak that are not zero, so neither is read as
/// the work's own.
#[test]
fn spent_work_charges_another_meter_as_the_work_did() {
    let work = |m: &mut Meter| -> Result<(), Error> {
        m.tick(70_000)?;
        m.charge(300)?;
        m.charge(200)?;
        m.release(400);
        m.tick(5)
    };
    let mut made = Meter::new(&Budget::default());
    made.tick(9).unwrap();
    made.charge(1_000).unwrap();
    made.release(900);
    let ((), spent) = made.measured(work).unwrap();
    assert_eq!(spent.held(), 100);
    assert_eq!(
        (made.work(), made.ledger(), made.ledger_peak()),
        (70_014, 200, 1_000)
    );

    let mut fresh = Meter::new(&Budget::default());
    work(&mut fresh).unwrap();
    let mut found = Meter::new(&Budget::default());
    found.spend(spent).unwrap();
    assert_eq!(
        (found.work(), found.ledger(), found.ledger_peak()),
        (fresh.work(), fresh.ledger(), fresh.ledger_peak())
    );
    assert_eq!(
        (found.work(), found.ledger(), found.ledger_peak()),
        (70_005, 100, 500)
    );

    // At the caps the work just fits, both fit; one under either, both are refused alike.
    let at = |work_cap, byte_cap| {
        let mut fresh = Meter::with_caps(&Budget::default(), work_cap, byte_cap);
        let mut found = Meter::with_caps(&Budget::default(), work_cap, byte_cap);
        (work(&mut fresh), found.spend(spent))
    };
    assert_eq!(at(70_005, 500), (Ok(()), Ok(())));
    let over = |limit| Err(Error::LimitExceeded(limit));
    assert_eq!(
        at(70_004, 500),
        (over(Limit::WorkUnits), over(Limit::WorkUnits))
    );
    assert_eq!(
        at(70_005, 499),
        (over(Limit::LedgerBytes), over(Limit::LedgerBytes))
    );
}
