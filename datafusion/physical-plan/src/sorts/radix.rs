// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// TODO: replace with arrow_row::radix::radix_sort_to_indices once
// available in arrow-rs (see https://github.com/apache/arrow-rs/pull/9683)

//! MSD radix sort on row-encoded keys.

use arrow::array::UInt32Array;
use arrow::row::{RowConverter, Rows, SortField};
use arrow_ord::sort::SortColumn;
use std::sync::Arc;

/// Buckets smaller than this fall back to comparison sort.
const FALLBACK_THRESHOLD: usize = 32;

/// Maximum number of radix passes before falling back to comparison sort.
const MAX_DEPTH: usize = 8;

/// Sort row indices using MSD radix sort on row-encoded keys.
///
/// Returns a `UInt32Array` of row indices in sorted order.
pub(crate) fn radix_sort_to_indices(
    sort_columns: &[SortColumn],
) -> arrow::error::Result<UInt32Array> {
    let sort_fields: Vec<SortField> = sort_columns
        .iter()
        .map(|col| {
            SortField::new_with_options(
                col.values.data_type().clone(),
                col.options.unwrap_or_default(),
            )
        })
        .collect();

    let arrays: Vec<_> = sort_columns
        .iter()
        .map(|col| Arc::clone(&col.values))
        .collect();

    let converter = RowConverter::new(sort_fields)?;
    let rows = converter.convert_columns(&arrays)?;

    let n = rows.num_rows();
    let mut indices: Vec<u32> = (0..n as u32).collect();
    let mut temp = vec![0u32; n];
    let mut bytes = vec![0u8; n];
    msd_radix_sort(&mut indices, &mut temp, &mut bytes, &rows, 0, true);
    Ok(UInt32Array::from(indices))
}

/// The byte at `offset` in the row, or 0 if past the end.
///
/// Inline helper until `Row::byte_from` is available in released arrow-row.
#[inline(always)]
unsafe fn row_byte(rows: &Rows, idx: u32, byte_pos: usize) -> u8 {
    let row = unsafe { rows.row_unchecked(idx as usize) };
    let data = row.data();
    if byte_pos < data.len() {
        unsafe { *data.get_unchecked(byte_pos) }
    } else {
        0
    }
}

/// Row data starting at `offset`, or empty slice if past the end.
///
/// Inline helper until `Row::data_from` is available in released arrow-row.
#[inline(always)]
unsafe fn row_data_from(rows: &Rows, idx: u32, byte_pos: usize) -> &[u8] {
    let row = unsafe { rows.row_unchecked(idx as usize) };
    let data = row.data();
    if byte_pos <= data.len() {
        unsafe { data.get_unchecked(byte_pos..) }
    } else {
        &[]
    }
}

/// MSD radix sort using ping-pong buffers.
///
/// Each level scatters from `src` into `dst`, then recurses with the
/// roles swapped (dst becomes the next level's src). This avoids an
/// O(n) `copy_from_slice` at every recursion level.
///
/// `result_in_src` tracks where the caller expects the sorted output:
/// true means `src`, false means `dst`.
fn msd_radix_sort(
    src: &mut [u32],
    dst: &mut [u32],
    bytes: &mut [u8],
    rows: &Rows,
    byte_pos: usize,
    result_in_src: bool,
) {
    let n = src.len();

    if n <= FALLBACK_THRESHOLD || byte_pos >= MAX_DEPTH {
        // Compare only from byte_pos onward — earlier bytes are identical
        // within this bucket, having already been discriminated by prior
        // radix passes.
        if result_in_src {
            src.sort_unstable_by(|&a, &b| {
                let ra = unsafe { row_data_from(rows, a, byte_pos) };
                let rb = unsafe { row_data_from(rows, b, byte_pos) };
                ra.cmp(rb)
            });
        } else {
            dst.copy_from_slice(src);
            dst.sort_unstable_by(|&a, &b| {
                let ra = unsafe { row_data_from(rows, a, byte_pos) };
                let rb = unsafe { row_data_from(rows, b, byte_pos) };
                ra.cmp(rb)
            });
        }
        return;
    }

    // Extract bytes and build histogram in one pass. The bytes buffer
    // avoids chasing pointers through Rows a second time during scatter.
    let bytes = &mut bytes[..n];
    let mut counts = [0u32; 256];
    for (i, &idx) in src.iter().enumerate() {
        let b = unsafe { row_byte(rows, idx, byte_pos) };
        bytes[i] = b;
        counts[b as usize] += 1;
    }

    let mut offsets = [0u32; 257];
    let mut num_buckets = 0u32;
    for i in 0..256 {
        num_buckets += (counts[i] > 0) as u32;
        offsets[i + 1] = offsets[i] + counts[i];
    }

    // All rows share the same byte — no scatter needed, roles unchanged.
    if num_buckets == 1 {
        msd_radix_sort(src, dst, bytes, rows, byte_pos + 1, result_in_src);
        return;
    }

    // Scatter src → dst using the pre-extracted bytes
    let mut write_pos = offsets;
    for (i, &idx) in src.iter().enumerate() {
        let b = bytes[i] as usize;
        dst[write_pos[b] as usize] = idx;
        write_pos[b] += 1;
    }

    // Recurse with roles swapped: after scatter the data lives in dst,
    // so dst becomes the next level's src.
    for bucket in 0..256 {
        let start = offsets[bucket] as usize;
        let end = offsets[bucket + 1] as usize;
        let len = end - start;
        if len > 1 {
            msd_radix_sort(
                &mut dst[start..end],
                &mut src[start..end],
                &mut bytes[start..end],
                rows,
                byte_pos + 1,
                !result_in_src,
            );
        } else if len == 1 && result_in_src {
            // Single-element bucket: after scatter it's in dst, copy back
            // if the caller expects the result in src.
            src[start] = dst[start];
        }
    }
}
