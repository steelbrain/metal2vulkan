use super::*;
use crate::air::error::{
    AbiError, BitcodeError, BitstreamError, CanonError, ContainerError, ErrorClass, MissingInput,
    RequestError, VerifyError,
};

fn one_of_each() -> Vec<Error> {
    vec![
        Error::Unsupported(Unsupported::needs(
            Feature::SimdWidth,
            Capability::Feature(VkFeature::SubgroupSizeControl),
        )),
        Error::Unsupported(Unsupported::not_implemented(Feature::Tessellation)),
        Error::Unsupported(Unsupported::not_representable(Feature::HandleAsPointer)),
        Error::NeedsInput(RuntimeInput::VertexDescriptor { attribute: 3 }),
        Error::InvalidInput(InputError::new(
            InputErrorKind::Container(ContainerError::BadMagic),
            Location::container(0),
        )),
        Error::InvalidDescriptor(DescriptorError::FunctionConstantIndex { index: 9 }),
        Error::LimitExceeded(Limit::WorkUnits),
        Error::LimitExceeded(Limit::LedgerBytes),
        Error::Budget(BudgetExceeded::Time),
        Error::Cancelled,
        Error::Internal(InternalError::new(InternalKind::Panic, "boom")),
        Error::Internal(InternalError::new(InternalKind::Build, "ill-formed")),
        Error::Internal(InternalError::new(InternalKind::Verify, "rejected")),
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            "needs more",
        )),
        Error::Internal(InternalError::new(InternalKind::Invariant, "broken")),
        Error::Internal(InternalError::new(
            InternalKind::AllocationFailed,
            "reserving 64 ledger-admitted bytes",
        )),
    ]
}

/// Retry() for every outcome class.
#[test]
fn retry_table() {
    for err in one_of_each() {
        // Exhaustive over `Error`, so a new variant must be classified here.
        let retry = match &err {
            Error::Unsupported(u) => match u.needs {
                Needs::AnyOf(_) => Retry::Never,
                Needs::NotImplemented | Needs::NotRepresentable => Retry::Never,
            },
            Error::NeedsInput(_) => Retry::Never,
            Error::InvalidInput(_) => Retry::Never,
            Error::InvalidDescriptor(_) => Retry::Never,
            Error::LimitExceeded(_) => Retry::Never,
            Error::Budget(BudgetExceeded::Time) => Retry::Later,
            // `BudgetExceeded` and `InternalKind` are the front end's crate's, `#[non_exhaustive]`
            // there: each variant this pinned version has is listed, and a new one fails here until
            // it is.
            Error::Budget(other) => panic!("classify {other:?}"),
            Error::Cancelled => Retry::Later,
            // Every `InternalKind`: the host allocator failing a reservation the ledger admitted is
            // transient; every other kind is an invariant of the emitted, profile-dependent module.
            Error::Internal(internal) => match internal.kind {
                InternalKind::AllocationFailed => Retry::Later,
                InternalKind::Build
                | InternalKind::Verify
                | InternalKind::RequirementsExceedProfile
                | InternalKind::Panic
                | InternalKind::Invariant => Retry::Never,
                other => panic!("classify {other:?}"),
            },
        };
        assert_eq!(err.retry(), retry, "{err:?}");
        assert!(!err.to_string().is_empty());
    }
}

#[test]
fn only_transient_outcomes_are_retryable() {
    for err in one_of_each() {
        let transient = matches!(err, Error::Cancelled | Error::Budget(_))
            || matches!(&err, Error::Internal(i) if i.kind == InternalKind::AllocationFailed);
        assert_eq!(err.retry() == Retry::Later, transient, "{err:?}");
    }
}

fn map(kind: InputErrorKind) -> Error {
    Error::from(InputError::new(kind, Location::module()))
}

/// The input-error-kind mapping.
#[test]
fn input_error_mapping() {
    let malformed = [
        InputErrorKind::Container(ContainerError::Truncated { need: 8, have: 4 }),
        InputErrorKind::Bitstream(BitstreamError::BadVbr),
        InputErrorKind::Bitcode(BitcodeError::UnknownRecord {
            block: 12,
            code: 99,
        }),
        InputErrorKind::Verify(VerifyError::UseNotDominated),
        InputErrorKind::Abi(AbiError::TypeSizeMismatch),
        InputErrorKind::Intrinsic(IntrinsicError::SignatureMismatch),
        InputErrorKind::Intrinsic(IntrinsicError::ImmediateNotConstant),
        InputErrorKind::Canon(CanonError::UndefCondition),
        InputErrorKind::Canon(CanonError::CallTypeMismatch),
    ];
    for kind in malformed {
        let class = InputError::new(kind.clone(), Location::module()).class();
        assert_eq!(class, ErrorClass::Malformed, "{kind:?}");
        let err = map(kind.clone());
        assert_eq!(
            err,
            Error::InvalidInput(InputError::new(kind, Location::module()))
        );
    }

    let not_implemented = [
        (
            InputErrorKind::Unsupported(crate::air::error::Feature::AddressSpace(7)),
            Feature::AddressSpace(7),
            None,
        ),
        (
            InputErrorKind::Intrinsic(IntrinsicError::UnknownIntrinsic {
                name: DiagName::new("air.some_future_op"),
            }),
            Feature::UnknownIntrinsic,
            // The refusal names the intrinsic.
            Some(DiagName::new("air.some_future_op")),
        ),
        (
            InputErrorKind::Canon(CanonError::StaticInitNotEvaluable),
            Feature::StaticInitializer,
            None,
        ),
    ];
    for (kind, feature, symbol) in not_implemented {
        let class = InputError::new(kind.clone(), Location::module()).class();
        assert_eq!(class, ErrorClass::Unsupported, "{kind:?}");
        let mut expected = Unsupported::not_implemented(feature).at(Location::module());
        expected.symbol = symbol;
        assert_eq!(map(kind), Error::Unsupported(expected));
    }

    let request = RequestError::EntryNotFound {
        requested: DiagName::new("main0"),
        available: 2,
    };
    assert_eq!(
        InputError::new(InputErrorKind::Request(request.clone()), Location::module()).class(),
        ErrorClass::InvalidRequest
    );
    assert_eq!(
        map(InputErrorKind::Request(request)),
        Error::InvalidDescriptor(DescriptorError::EntryNotFound {
            requested: DiagName::new("main0"),
            available: 2,
        })
    );

    assert_eq!(
        InputError::new(
            InputErrorKind::Limit(Limit::BitstreamNesting),
            Location::module()
        )
        .class(),
        ErrorClass::LimitExceeded
    );
    assert_eq!(
        map(InputErrorKind::Limit(Limit::BitstreamNesting)),
        Error::LimitExceeded(Limit::BitstreamNesting)
    );
}

#[test]
fn single_capability_needs_is_one_alternative() {
    let u = Unsupported::needs(
        Feature::FramebufferFetch,
        Capability::Extension(VkExtension::ExtRasterizationOrderAttachmentAccess),
    );
    assert_eq!(
        u.needs,
        Needs::AnyOf(vec![vec![Capability::Extension(
            VkExtension::ExtRasterizationOrderAttachmentAccess
        )]])
    );
}

#[test]
fn locations_and_names_render() {
    let at = Location::bitstream(129, &[8, 12], Some(4));
    assert_eq!(
        at.to_string(),
        "bitstream bit 129 in blocks [8, 12], record 4"
    );
    let at = Location::function(DiagName::new("f"), Some(2), None);
    assert_eq!(at.to_string(), "function f, block 2");
    let at = Location::metadata(Some(DiagName::new("k")), Some(1), Some("air.buffer"));
    assert_eq!(
        at.to_string(),
        "metadata of entry k, argument 1, key air.buffer"
    );
}

/// A refusal renders the symbol it names before its reason and location.
#[test]
fn refusals_name_their_symbol() {
    let at = Location::function(DiagName::new("k"), Some(3), Some(17));
    let u = Unsupported::not_implemented(Feature::RayTracing)
        .at(at.clone())
        .naming(DiagName::new("air.intersect.triangle_data"));
    assert_eq!(u.needs, Needs::NotImplemented);
    assert_eq!(u.location, Some(at));
    assert_eq!(
        u.to_string(),
        "RayTracing (air.intersect.triangle_data); needs NotImplemented at function k, block 3, \
         instruction 17"
    );
    // Without a symbol the rendering is unchanged.
    assert_eq!(
        Unsupported::not_implemented(Feature::Tessellation).to_string(),
        "Tessellation; needs NotImplemented"
    );
}

#[test]
fn error_types_are_send_and_sync() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Error>();
    send_sync::<InputError>();
    send_sync::<BindError>();
    send_sync::<Unsupported>();
    send_sync::<Capability>();
    send_sync::<DescriptorError>();
    send_sync::<InternalError>();
}

/// The front end's outcome reaches the caller one variant to one variant: its refusals as
/// `NotImplemented`, its request errors as descriptor errors, and the shared caps, deadline,
/// cancellation and internal errors unchanged, so transient stays transient.
#[test]
fn front_end_errors_map_one_to_one() {
    use crate::air::error::{Error as Air, Feature as AirFeature, Unsupported as AirUnsupported};
    let at = Location::module();
    let cases = [
        (
            Air::Unsupported(
                AirUnsupported::not_implemented(AirFeature::Recursion)
                    .at(at.clone())
                    .naming(DiagName::new("f")),
            ),
            Error::Unsupported(
                Unsupported::not_implemented(Feature::Recursion)
                    .at(at.clone())
                    .naming(DiagName::new("f")),
            ),
        ),
        (
            Air::NeedsInput(MissingInput::LinkedFunction {
                symbol: DiagName::new("g"),
            }),
            Error::NeedsInput(RuntimeInput::LinkedFunction {
                symbol: DiagName::new("g"),
            }),
        ),
        (
            Air::InvalidRequest(RequestError::DuplicateFunctionConstant { index: 1 }),
            Error::InvalidDescriptor(DescriptorError::DuplicateFunctionConstant { index: 1 }),
        ),
        (
            Air::InvalidRequest(RequestError::LinkedFunction {
                name: DiagName::new("g"),
                problem: LinkProblem::Ambiguous,
            }),
            Error::InvalidDescriptor(DescriptorError::LinkedFunction {
                name: DiagName::new("g"),
                problem: LinkProblem::Ambiguous,
            }),
        ),
        (
            Air::LimitExceeded(Limit::WorkUnits),
            Error::LimitExceeded(Limit::WorkUnits),
        ),
        (
            Air::Budget(BudgetExceeded::Time),
            Error::Budget(BudgetExceeded::Time),
        ),
        (Air::Cancelled, Error::Cancelled),
        (
            Air::Internal(InternalError::new(InternalKind::AllocationFailed, "x")),
            Error::Internal(InternalError::new(InternalKind::AllocationFailed, "x")),
        ),
    ];
    for (air, product) in cases {
        let transient = matches!(
            air,
            Air::Budget(_)
                | Air::Cancelled
                | Air::Internal(InternalError {
                    kind: InternalKind::AllocationFailed,
                    ..
                })
        );
        let mapped = Error::from(air);
        assert_eq!(mapped, product);
        assert_eq!(mapped.retry() == Retry::Later, transient, "{mapped:?}");
    }
}
