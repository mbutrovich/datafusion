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

/// 256-bucket histogram + scatter costs more than comparison sort at small n.
const FALLBACK_THRESHOLD: usize = 64;

/// 8 bytes covers the discriminating prefix of most key layouts; deeper
/// recursion hits diminishing returns as buckets become sparse.
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
    msd_radix_sort(&mut indices, &mut temp, &rows, 0);
    Ok(UInt32Array::from(indices))
}

fn msd_radix_sort(indices: &mut [u32], temp: &mut [u32], rows: &Rows, byte_pos: usize) {
    let n = indices.len();

    if n <= FALLBACK_THRESHOLD || byte_pos >= MAX_DEPTH {
        indices.sort_unstable_by(|&a, &b| {
            let ra = unsafe { rows.row_unchecked(a as usize) };
            let rb = unsafe { rows.row_unchecked(b as usize) };
            ra.cmp(&rb)
        });
        return;
    }

    let mut counts = [0u32; 256];
    for &idx in &*indices {
        let row = unsafe { rows.row_unchecked(idx as usize) };
        let byte = row.data().get(byte_pos).copied().unwrap_or(0);
        counts[byte as usize] += 1;
    }

    // Skip scatter when all rows share the same byte
    if counts.iter().filter(|&&c| c > 0).count() == 1 {
        msd_radix_sort(indices, temp, rows, byte_pos + 1);
        return;
    }

    let mut offsets = [0u32; 257];
    for i in 0..256 {
        offsets[i + 1] = offsets[i] + counts[i];
    }

    let temp = &mut temp[..n];
    let mut write_pos = offsets;
    for &idx in &*indices {
        let row = unsafe { rows.row_unchecked(idx as usize) };
        let byte = row.data().get(byte_pos).copied().unwrap_or(0) as usize;
        temp[write_pos[byte] as usize] = idx;
        write_pos[byte] += 1;
    }
    indices.copy_from_slice(temp);

    for bucket in 0..256 {
        let start = offsets[bucket] as usize;
        let end = offsets[bucket + 1] as usize;
        if end - start > 1 {
            msd_radix_sort(
                &mut indices[start..end],
                &mut temp[start..end],
                rows,
                byte_pos + 1,
            );
        }
    }
}
