//! The in-place check's verdict a texture over host pages takes (`Device::in_place_check_report`):
//! its class's for its access, for texels that share a line of the device's memory
//! (`Device::in_place_check_line`, its `nonCoherentAtomSize`) with bytes beside them, or for
//! texels that share none (the report's `, rows on lines` lines). Worked out here from the
//! texture's layout, apart from the runtime's own reckoning (`in_place::RowEnds::of`).

use metal2vulkan_runtime::Device;

/// Whether a 2D texture's texels, `rows` rows of `row_bytes` bytes `pitch` apart from `first`
/// bytes past a page (a no-copy buffer's pages start one), share no line of the device's memory
/// with bytes beside them: their first byte and their end lie on a line, and so do each row's
/// start and end where bytes lie between rows.
pub(crate) fn on_lines(device: &Device, first: u64, row_bytes: u64, pitch: u64, rows: u64) -> bool {
    let line = device.in_place_check_line();
    let on = |at: u64| at.is_multiple_of(line);
    let end = first + (rows - 1) * pitch + row_bytes;
    let rows_apart = rows > 1 && pitch != row_bytes;
    on(first) && on(end) && (!rows_apart || (on(pitch) && on(first + row_bytes)))
}

/// Whether the device's check passed for `what` (`Pointer`, `Pointer renders`, `Fresh writes`,
/// as the report names a class and access), for texels that share no line with bytes beside
/// them where `on_lines`, else for those that share one.
pub(crate) fn passed(device: &Device, what: &str, on_lines: bool) -> bool {
    let line = match on_lines {
        true => format!("{what}, rows on lines: Passed"),
        false => format!("{what}: Passed"),
    };
    device.in_place_check_report().contains(&line)
}
