//! Where a texture over a buffer's bytes lies (`texture_placement.rs`): each route the making
//! takes (`texture/route.rs`) answers its own placement, and every shadow cause its own reason.

use crate::in_place::{Access, PagesClass, Refusal, RowEnds, ShadowCause};
use crate::texture::{HostWay, Route};
use crate::{InPlaceAccess, InPlaceRefusal, InPlaceRowEnds, TexturePlacement, TextureShadowReason};

/// A route over host pages of `class` whose texels share a line with bytes beside them.
fn host(way: HostWay) -> Route {
    Route::Host {
        class: PagesClass::Pointer,
        rows: RowEnds::InLines,
        way,
    }
}

#[test]
fn each_route_answers_its_own_placement() {
    use TexturePlacement as P;
    use TextureShadowReason as R;
    let shadow = |cause| host(HostWay::Shadow(cause)).placement();
    assert_eq!(Route::Past.placement(), P::Shadow(R::PastLength));
    assert_eq!(Route::Device.placement(), P::NotHostPages);
    assert_eq!(
        host(HostWay::DeviceCopy).placement(),
        P::Shadow(R::DeviceCopy)
    );
    for storable in [false, true] {
        assert_eq!(host(HostWay::InPlace { storable }).placement(), P::InPlace);
    }
    assert_eq!(shadow(ShadowCause::Exact), P::Shadow(R::Policy));
    assert_eq!(
        shadow(ShadowCause::StoreWidth(2)),
        P::Shadow(R::StoreWidth { texel_bytes: 2 })
    );
    let accesses = [
        (Access::Reads, InPlaceAccess::Reads),
        (Access::Renders, InPlaceAccess::Renders),
        (Access::Writes, InPlaceAccess::Writes),
    ];
    let rows = [
        (RowEnds::InLines, InPlaceRowEnds::InLines),
        (RowEnds::OnLines, InPlaceRowEnds::OnLines),
    ];
    for class in PagesClass::ALL {
        for (access, public_access) in accesses {
            for (row_ends, public_rows) in rows {
                let pending = ShadowCause::CheckPending(class, access, row_ends);
                assert_eq!(shadow(pending), P::Pending);
                let failed = ShadowCause::CheckFailed(class, access, row_ends);
                let reason = R::CheckFailed {
                    access: public_access,
                    row_ends: public_rows,
                };
                assert_eq!(shadow(failed), P::Shadow(reason));
                let unchecked = ShadowCause::Unchecked(class, access, row_ends);
                let reason = R::Unchecked {
                    access: public_access,
                    row_ends: public_rows,
                };
                assert_eq!(shadow(unchecked), P::Shadow(reason));
            }
        }
    }
    let refusals = [
        (Refusal::Rows, InPlaceRefusal::Rows),
        (Refusal::Detached, InPlaceRefusal::Detached),
        (Refusal::Image, InPlaceRefusal::Image),
        (Refusal::Pitch, InPlaceRefusal::Pitch),
        (Refusal::Placement, InPlaceRefusal::Placement),
        (Refusal::Source, InPlaceRefusal::Source),
        (Refusal::DmaBuf, InPlaceRefusal::DmaBuf),
    ];
    for (refusal, public) in refusals {
        let refused = ShadowCause::Refused(refusal, String::new());
        assert_eq!(shadow(refused), P::Shadow(R::Refused(public)));
    }
}
