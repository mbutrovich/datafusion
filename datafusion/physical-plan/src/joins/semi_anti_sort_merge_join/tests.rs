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

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use super::SemiAntiSortMergeJoinExec;
use super::stream::SemiAntiSortMergeJoinStream;
use crate::ExecutionPlan;
use crate::RecordBatchStream;
use crate::common;
use crate::expressions::Column;
use crate::joins::SortMergeJoinExec;
use crate::joins::utils::{ColumnIndex, JoinFilter};
use crate::metrics::ExecutionPlanMetricsSet;
use crate::test::TestMemoryExec;

use arrow::array::{Int32Array, RecordBatch};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::JoinSide;
use datafusion_common::JoinType::*;
use datafusion_common::test_util::batches_to_sort_string;
use datafusion_common::{NullEquality, Result};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr_common::physical_expr::PhysicalExprRef;
use futures::Stream;

type JoinOn = Vec<(
    Arc<dyn datafusion_physical_expr_common::physical_expr::PhysicalExpr>,
    Arc<dyn datafusion_physical_expr_common::physical_expr::PhysicalExpr>,
)>;

fn build_table(
    a: (&str, &Vec<i32>),
    b: (&str, &Vec<i32>),
    c: (&str, &Vec<i32>),
) -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(a.0, DataType::Int32, false),
        Field::new(b.0, DataType::Int32, false),
        Field::new(c.0, DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(a.1.clone())),
            Arc::new(Int32Array::from(b.1.clone())),
            Arc::new(Int32Array::from(c.1.clone())),
        ],
    )
    .unwrap();
    TestMemoryExec::try_new_exec(&[vec![batch]], schema, None).unwrap()
}

fn build_table_nullable(
    a: (&str, &Vec<Option<i32>>),
    b: (&str, &Vec<Option<i32>>),
    c: (&str, &Vec<Option<i32>>),
) -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(a.0, DataType::Int32, true),
        Field::new(b.0, DataType::Int32, true),
        Field::new(c.0, DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(a.1.clone())),
            Arc::new(Int32Array::from(b.1.clone())),
            Arc::new(Int32Array::from(c.1.clone())),
        ],
    )
    .unwrap();
    TestMemoryExec::try_new_exec(&[vec![batch]], schema, None).unwrap()
}

fn build_table_from_batches(batches: Vec<RecordBatch>) -> Arc<dyn ExecutionPlan> {
    let schema = batches.first().unwrap().schema();
    TestMemoryExec::try_new_exec(&[batches], schema, None).unwrap()
}

/// Run the same query through both SortMergeJoinExec and
/// SemiAntiSortMergeJoinExec, assert identical sorted output.
async fn dual_join_collect(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    on: JoinOn,
    join_type: datafusion_common::JoinType,
    sort_options: Vec<SortOptions>,
    null_equality: NullEquality,
) -> Result<Vec<RecordBatch>> {
    let task_ctx = Arc::new(TaskContext::default());

    // Run through existing SMJ
    let smj = SortMergeJoinExec::try_new(
        Arc::clone(&left),
        Arc::clone(&right),
        on.clone(),
        None,
        join_type,
        sort_options.clone(),
        null_equality,
    )?;
    let smj_stream = smj.execute(0, Arc::clone(&task_ctx))?;
    let smj_batches = common::collect(smj_stream).await?;

    // Run through new operator
    let sa = SemiAntiSortMergeJoinExec::try_new(
        left,
        right,
        on,
        None,
        join_type,
        sort_options,
        null_equality,
    )?;
    let sa_stream = sa.execute(0, Arc::clone(&task_ctx))?;
    let sa_batches = common::collect(sa_stream).await?;

    // Compare sorted output (order may differ due to coalescing)
    let smj_total: usize = smj_batches.iter().map(|b| b.num_rows()).sum();
    let sa_total: usize = sa_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        smj_total, sa_total,
        "Row count mismatch for {join_type:?}: SMJ={smj_total}, SA={sa_total}"
    );
    if smj_total > 0 {
        let smj_str = batches_to_sort_string(&smj_batches);
        let sa_str = batches_to_sort_string(&sa_batches);
        assert_eq!(
            smj_str, sa_str,
            "SortMergeJoin and SemiAntiSortMergeJoin produced different results \
             for {join_type:?}"
        );
    }

    Ok(sa_batches)
}

/// Dual execution with filter.
async fn dual_join_collect_with_filter(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    on: JoinOn,
    filter: JoinFilter,
    join_type: datafusion_common::JoinType,
    sort_options: Vec<SortOptions>,
    null_equality: NullEquality,
) -> Result<Vec<RecordBatch>> {
    let task_ctx = Arc::new(TaskContext::default());

    let smj = SortMergeJoinExec::try_new(
        Arc::clone(&left),
        Arc::clone(&right),
        on.clone(),
        Some(filter.clone()),
        join_type,
        sort_options.clone(),
        null_equality,
    )?;
    let smj_stream = smj.execute(0, Arc::clone(&task_ctx))?;
    let smj_batches = common::collect(smj_stream).await?;

    let sa = SemiAntiSortMergeJoinExec::try_new(
        left,
        right,
        on,
        Some(filter),
        join_type,
        sort_options,
        null_equality,
    )?;
    let sa_stream = sa.execute(0, Arc::clone(&task_ctx))?;
    let sa_batches = common::collect(sa_stream).await?;

    let smj_total: usize = smj_batches.iter().map(|b| b.num_rows()).sum();
    let sa_total: usize = sa_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        smj_total, sa_total,
        "Row count mismatch for {join_type:?} with filter: SMJ={smj_total}, SA={sa_total}"
    );
    if smj_total > 0 {
        let smj_str = batches_to_sort_string(&smj_batches);
        let sa_str = batches_to_sort_string(&sa_batches);
        assert_eq!(
            smj_str, sa_str,
            "SortMergeJoin and SemiAntiSortMergeJoin produced different results \
             for {join_type:?} with filter"
        );
    }

    Ok(sa_batches)
}

// ==================== TESTS PORTED FROM SMJ ====================
// Each runs through both operators and compares results.

#[tokio::test]
async fn join_left_semi() -> Result<()> {
    let left = build_table(
        ("a1", &vec![1, 2, 2, 3]),
        ("b1", &vec![4, 5, 5, 7]),
        ("c1", &vec![7, 8, 8, 9]),
    );
    let right = build_table(
        ("a2", &vec![10, 20, 30]),
        ("b1", &vec![4, 5, 6]),
        ("c2", &vec![70, 80, 90]),
    );
    let on = vec![(
        Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
        Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
    )];
    let batches = dual_join_collect(
        left,
        right,
        on,
        LeftSemi,
        vec![SortOptions::default()],
        NullEquality::NullEqualsNothing,
    )
    .await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 3);
    Ok(())
}

#[tokio::test]
async fn join_left_anti() -> Result<()> {
    let left = build_table(
        ("a1", &vec![1, 2, 2, 3, 5]),
        ("b1", &vec![4, 5, 5, 7, 7]),
        ("c1", &vec![7, 8, 8, 9, 11]),
    );
    let right = build_table(
        ("a2", &vec![10, 20, 30]),
        ("b1", &vec![4, 5, 6]),
        ("c2", &vec![70, 80, 90]),
    );
    let on = vec![(
        Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
        Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
    )];
    let batches = dual_join_collect(
        left,
        right,
        on,
        LeftAnti,
        vec![SortOptions::default()],
        NullEquality::NullEqualsNothing,
    )
    .await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2);
    Ok(())
}

#[tokio::test]
async fn join_right_semi() -> Result<()> {
    let left = build_table(
        ("a1", &vec![10, 20, 30, 40]),
        ("b1", &vec![4, 5, 5, 6]),
        ("c1", &vec![70, 80, 90, 100]),
    );
    let right = build_table(
        ("a2", &vec![1, 2, 2, 3]),
        ("b1", &vec![4, 5, 5, 7]),
        ("c2", &vec![7, 8, 8, 9]),
    );
    let on = vec![(
        Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
        Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
    )];
    let batches = dual_join_collect(
        left,
        right,
        on,
        RightSemi,
        vec![SortOptions::default()],
        NullEquality::NullEqualsNothing,
    )
    .await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 3);
    Ok(())
}

#[tokio::test]
async fn join_right_anti() -> Result<()> {
    let left = build_table(
        ("a1", &vec![10, 20, 30]),
        ("b1", &vec![4, 5, 6]),
        ("c1", &vec![70, 80, 90]),
    );
    let right = build_table(
        ("a2", &vec![1, 2, 2, 3, 5]),
        ("b1", &vec![4, 5, 5, 7, 7]),
        ("c2", &vec![7, 8, 8, 9, 11]),
    );
    let on = vec![(
        Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
        Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
    )];
    let batches = dual_join_collect(
        left,
        right,
        on,
        RightAnti,
        vec![SortOptions::default()],
        NullEquality::NullEqualsNothing,
    )
    .await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2);
    Ok(())
}

// Multi-key from existing SMJ: join_right_semi_two
#[tokio::test]
async fn join_right_semi_multi_key() -> Result<()> {
    let left = build_table(
        ("a1", &vec![1, 2, 2, 3]),
        ("b1", &vec![4, 5, 5, 6]),
        ("c1", &vec![70, 80, 90, 100]),
    );
    let right = build_table(
        ("a1", &vec![1, 2, 2, 3]),
        ("b1", &vec![4, 5, 5, 7]),
        ("c2", &vec![7, 8, 8, 9]),
    );
    let on = vec![
        (
            Arc::new(Column::new_with_schema("a1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("a1", &right.schema())?) as _,
        ),
        (
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        ),
    ];
    let batches = dual_join_collect(
        left,
        right,
        on,
        RightSemi,
        vec![SortOptions::default(); 2],
        NullEquality::NullEqualsNothing,
    )
    .await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 3); // (1,4), (2,5), (2,5) match
    Ok(())
}

// From existing SMJ: join_right_anti_two_with_filter
#[tokio::test]
async fn join_right_anti_with_filter() -> Result<()> {
    let left = build_table(("a1", &vec![1]), ("b1", &vec![10]), ("c1", &vec![30]));
    let right = build_table(("a1", &vec![1]), ("b1", &vec![10]), ("c2", &vec![20]));
    let on = vec![
        (
            Arc::new(Column::new_with_schema("a1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("a1", &right.schema())?) as _,
        ),
        (
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        ),
    ];
    // Filter: c2 > c1 => 20 > 30 = false, so anti should emit the right row
    let filter = JoinFilter::new(
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("c2", 1)),
            Operator::Gt,
            Arc::new(Column::new("c1", 0)),
        )),
        vec![
            ColumnIndex {
                index: 2,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 2,
                side: JoinSide::Right,
            },
        ],
        Arc::new(Schema::new(vec![
            Field::new("c1", DataType::Int32, true),
            Field::new("c2", DataType::Int32, true),
        ])),
    );
    let batches = dual_join_collect_with_filter(
        left,
        right,
        on,
        filter,
        RightAnti,
        vec![SortOptions::default(); 2],
        NullEquality::NullEqualsNothing,
    )
    .await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1);
    Ok(())
}

// From existing SMJ: join_right_semi_two_with_filter
#[tokio::test]
async fn join_right_semi_with_filter() -> Result<()> {
    let left = build_table(("a1", &vec![1]), ("b1", &vec![10]), ("c1", &vec![30]));
    let right = build_table(("a1", &vec![1]), ("b1", &vec![10]), ("c2", &vec![20]));
    let on = vec![
        (
            Arc::new(Column::new_with_schema("a1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("a1", &right.schema())?) as _,
        ),
        (
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        ),
    ];
    // Filter: c1 > c2 => 30 > 20 = true, so semi should emit the right row
    let filter = JoinFilter::new(
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("c1", 0)),
            Operator::Gt,
            Arc::new(Column::new("c2", 1)),
        )),
        vec![
            ColumnIndex {
                index: 2,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 2,
                side: JoinSide::Right,
            },
        ],
        Arc::new(Schema::new(vec![
            Field::new("c1", DataType::Int32, true),
            Field::new("c2", DataType::Int32, true),
        ])),
    );
    let batches = dual_join_collect_with_filter(
        left,
        right,
        on,
        filter,
        RightSemi,
        vec![SortOptions::default(); 2],
        NullEquality::NullEqualsNothing,
    )
    .await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1);
    Ok(())
}

// From existing SMJ: join_right_semi_with_nulls
#[tokio::test]
async fn join_right_semi_with_nulls() -> Result<()> {
    let left = build_table_nullable(
        ("a1", &vec![Some(0), Some(1), Some(2), Some(2), Some(3)]),
        ("b1", &vec![Some(3), Some(4), Some(5), None, Some(6)]),
        ("c1", &vec![Some(60), None, Some(80), Some(85), Some(90)]),
    );
    let right = build_table_nullable(
        ("a1", &vec![Some(1), Some(2), Some(2), Some(3)]),
        ("b1", &vec![Some(4), Some(5), None, Some(6)]),
        ("c2", &vec![Some(7), Some(8), Some(8), None]),
    );
    let on = vec![
        (
            Arc::new(Column::new_with_schema("a1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("a1", &right.schema())?) as _,
        ),
        (
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        ),
    ];
    let batches = dual_join_collect(
        left,
        right,
        on,
        RightSemi,
        vec![SortOptions::default(); 2],
        NullEquality::NullEqualsNothing,
    )
    .await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 3); // (1,4), (2,5), (3,6) match; (2,NULL) doesn't
    Ok(())
}

// From existing SMJ: join_right_anti_with_nulls
#[tokio::test]
async fn join_right_anti_with_nulls() -> Result<()> {
    let left = build_table_nullable(
        ("a1", &vec![Some(0), Some(1), Some(2), Some(2), Some(3)]),
        ("b1", &vec![Some(3), Some(4), Some(5), None, Some(6)]),
        ("c2", &vec![Some(60), None, Some(80), Some(85), Some(90)]),
    );
    let right = build_table_nullable(
        ("a1", &vec![Some(1), Some(2), Some(2), Some(3)]),
        ("b1", &vec![Some(4), Some(5), None, Some(6)]),
        ("c2", &vec![Some(7), Some(8), Some(8), None]),
    );
    let on = vec![
        (
            Arc::new(Column::new_with_schema("a1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("a1", &right.schema())?) as _,
        ),
        (
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        ),
    ];
    let batches = dual_join_collect(
        left,
        right,
        on,
        RightAnti,
        vec![SortOptions::default(); 2],
        NullEquality::NullEqualsNothing,
    )
    .await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1); // (2,NULL) has no match
    Ok(())
}

// From existing SMJ: join_right_anti_with_nulls_with_options (NullEqualsNull + descending)
#[tokio::test]
async fn join_right_anti_nulls_equal_descending() -> Result<()> {
    let left = build_table_nullable(
        ("a1", &vec![Some(1), Some(2), Some(1), Some(0), Some(2)]),
        ("b1", &vec![Some(4), Some(5), Some(5), None, Some(5)]),
        ("c1", &vec![Some(7), Some(8), Some(8), Some(60), None]),
    );
    let right = build_table_nullable(
        ("a1", &vec![Some(3), Some(2), Some(2), Some(1)]),
        ("b1", &vec![None, Some(5), Some(5), Some(4)]),
        ("c2", &vec![Some(9), None, Some(8), Some(7)]),
    );
    let on = vec![
        (
            Arc::new(Column::new_with_schema("a1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("a1", &right.schema())?) as _,
        ),
        (
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        ),
    ];
    let sort_opts = vec![
        SortOptions {
            descending: true,
            nulls_first: false
        };
        2
    ];
    let batches = dual_join_collect(
        left,
        right,
        on,
        RightAnti,
        sort_opts,
        NullEquality::NullEqualsNull,
    )
    .await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    // With NullEqualsNull: (3,NULL) matches (0,NULL), (2,5)x2 match, (1,4) matches
    // So anti = rows that DON'T match = 3 rows
    assert_eq!(total, 3);
    Ok(())
}

// ==================== EMPTY INPUTS ====================

#[tokio::test]
async fn empty_left() -> Result<()> {
    let left = build_table(("a1", &vec![]), ("b1", &vec![]), ("c1", &vec![]));
    let right = build_table(
        ("a2", &vec![1, 2]),
        ("b1", &vec![4, 5]),
        ("c2", &vec![7, 8]),
    );
    let on = vec![(
        Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
        Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
    )];
    for join_type in [LeftSemi, LeftAnti, RightSemi, RightAnti] {
        dual_join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            join_type,
            vec![SortOptions::default()],
            NullEquality::NullEqualsNothing,
        )
        .await?;
    }
    Ok(())
}

#[tokio::test]
async fn empty_right() -> Result<()> {
    let left = build_table(
        ("a1", &vec![1, 2]),
        ("b1", &vec![4, 5]),
        ("c1", &vec![7, 8]),
    );
    let right = build_table(("a2", &vec![]), ("b1", &vec![]), ("c2", &vec![]));
    let on = vec![(
        Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
        Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
    )];
    for join_type in [LeftSemi, LeftAnti, RightSemi, RightAnti] {
        dual_join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            join_type,
            vec![SortOptions::default()],
            NullEquality::NullEqualsNothing,
        )
        .await?;
    }
    Ok(())
}

// ==================== ALL MATCH / ALL MISMATCH ====================

#[tokio::test]
async fn all_match_all_types() -> Result<()> {
    let left = build_table(
        ("a1", &vec![1, 2, 3]),
        ("b1", &vec![4, 5, 6]),
        ("c1", &vec![7, 8, 9]),
    );
    let right = build_table(
        ("a2", &vec![10, 20, 30]),
        ("b1", &vec![4, 5, 6]),
        ("c2", &vec![70, 80, 90]),
    );
    let on = vec![(
        Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
        Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
    )];
    for join_type in [LeftSemi, LeftAnti, RightSemi, RightAnti] {
        dual_join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            join_type,
            vec![SortOptions::default()],
            NullEquality::NullEqualsNothing,
        )
        .await?;
    }
    Ok(())
}

#[tokio::test]
async fn no_match_all_types() -> Result<()> {
    let left = build_table(
        ("a1", &vec![1, 2, 3]),
        ("b1", &vec![4, 5, 6]),
        ("c1", &vec![7, 8, 9]),
    );
    let right = build_table(
        ("a2", &vec![10, 20, 30]),
        ("b1", &vec![10, 11, 12]),
        ("c2", &vec![70, 80, 90]),
    );
    let on = vec![(
        Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
        Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
    )];
    for join_type in [LeftSemi, LeftAnti, RightSemi, RightAnti] {
        dual_join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            join_type,
            vec![SortOptions::default()],
            NullEquality::NullEqualsNothing,
        )
        .await?;
    }
    Ok(())
}

// ==================== DUPLICATES ====================

#[tokio::test]
async fn many_duplicates() -> Result<()> {
    let left = build_table(
        ("a1", &vec![1, 1, 1, 1, 2]),
        ("b1", &vec![5, 5, 5, 5, 5]),
        ("c1", &vec![1, 2, 3, 4, 5]),
    );
    let right = build_table(
        ("a2", &vec![10, 20, 30]),
        ("b1", &vec![5, 5, 5]),
        ("c2", &vec![70, 80, 90]),
    );
    let on = vec![(
        Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
        Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
    )];
    for join_type in [LeftSemi, LeftAnti, RightSemi, RightAnti] {
        dual_join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            join_type,
            vec![SortOptions::default()],
            NullEquality::NullEqualsNothing,
        )
        .await?;
    }
    Ok(())
}

// ==================== FILTER TESTS ====================

#[tokio::test]
async fn filter_always_true() -> Result<()> {
    let left = build_table(
        ("a1", &vec![1, 2, 3]),
        ("b1", &vec![4, 5, 6]),
        ("c1", &vec![7, 8, 9]),
    );
    let right = build_table(
        ("a2", &vec![10, 20, 30]),
        ("b1", &vec![4, 5, 6]),
        ("c2", &vec![70, 80, 90]),
    );
    let on = vec![(
        Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
        Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
    )];
    // c1 < c2 always true (7<70, 8<80, 9<90)
    let filter = JoinFilter::new(
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("c1", 0)),
            Operator::Lt,
            Arc::new(Column::new("c2", 1)),
        )),
        vec![
            ColumnIndex {
                index: 2,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 2,
                side: JoinSide::Right,
            },
        ],
        Arc::new(Schema::new(vec![
            Field::new("c1", DataType::Int32, false),
            Field::new("c2", DataType::Int32, false),
        ])),
    );
    for join_type in [LeftSemi, LeftAnti, RightSemi, RightAnti] {
        dual_join_collect_with_filter(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            filter.clone(),
            join_type,
            vec![SortOptions::default()],
            NullEquality::NullEqualsNothing,
        )
        .await?;
    }
    Ok(())
}

#[tokio::test]
async fn filter_always_false() -> Result<()> {
    let left = build_table(
        ("a1", &vec![1, 2, 3]),
        ("b1", &vec![4, 5, 6]),
        ("c1", &vec![700, 800, 900]),
    );
    let right = build_table(
        ("a2", &vec![10, 20, 30]),
        ("b1", &vec![4, 5, 6]),
        ("c2", &vec![70, 80, 90]),
    );
    let on = vec![(
        Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
        Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
    )];
    // c1 < c2 always false (700>70)
    let filter = JoinFilter::new(
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("c1", 0)),
            Operator::Lt,
            Arc::new(Column::new("c2", 1)),
        )),
        vec![
            ColumnIndex {
                index: 2,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 2,
                side: JoinSide::Right,
            },
        ],
        Arc::new(Schema::new(vec![
            Field::new("c1", DataType::Int32, false),
            Field::new("c2", DataType::Int32, false),
        ])),
    );
    for join_type in [LeftSemi, LeftAnti, RightSemi, RightAnti] {
        dual_join_collect_with_filter(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            filter.clone(),
            join_type,
            vec![SortOptions::default()],
            NullEquality::NullEqualsNothing,
        )
        .await?;
    }
    Ok(())
}

// ==================== INNER KEY GROUP SPANNING BATCHES ====================

#[tokio::test]
async fn inner_key_group_spans_batches() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a2", DataType::Int32, false),
        Field::new("b1", DataType::Int32, false),
        Field::new("c2", DataType::Int32, false),
    ]));
    let batch1 = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![10, 20])),
            Arc::new(Int32Array::from(vec![5, 5])),
            Arc::new(Int32Array::from(vec![70, 80])),
        ],
    )?;
    let batch2 = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![30])),
            Arc::new(Int32Array::from(vec![5])),
            Arc::new(Int32Array::from(vec![90])),
        ],
    )?;
    let right = build_table_from_batches(vec![batch1, batch2]);
    let left = build_table(
        ("a1", &vec![1, 2]),
        ("b1", &vec![5, 6]),
        ("c1", &vec![7, 8]),
    );
    let on = vec![(
        Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
        Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
    )];
    for join_type in [LeftSemi, LeftAnti] {
        dual_join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            join_type,
            vec![SortOptions::default()],
            NullEquality::NullEqualsNothing,
        )
        .await?;
    }
    Ok(())
}

// ==================== PENDING RE-ENTRY TESTS ====================
//
// These tests use a Pending-injecting stream wrapper to deterministically
// reproduce bugs that only manifest when async input streams return
// Poll::Pending at specific points.

/// A RecordBatch stream that yields Poll::Pending once before delivering
/// each batch at a specified index. This simulates the behavior of
/// repartitioned tokio::sync::mpsc channels where data isn't immediately
/// available.
struct PendingStream {
    batches: Vec<RecordBatch>,
    index: usize,
    /// If pending_before[i] is true, yield Pending once before delivering
    /// the batch at index i.
    pending_before: Vec<bool>,
    /// True if we've already yielded Pending for the current index.
    yielded_pending: bool,
    schema: SchemaRef,
}

impl PendingStream {
    fn new(batches: Vec<RecordBatch>, pending_before: Vec<bool>) -> Self {
        assert_eq!(batches.len(), pending_before.len());
        let schema = batches[0].schema();
        Self {
            batches,
            index: 0,
            pending_before,
            yielded_pending: false,
            schema,
        }
    }
}

impl Stream for PendingStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if self.index >= self.batches.len() {
            return Poll::Ready(None);
        }
        if self.pending_before[self.index] && !self.yielded_pending {
            self.yielded_pending = true;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        self.yielded_pending = false;
        let batch = self.batches[self.index].clone();
        self.index += 1;
        Poll::Ready(Some(Ok(batch)))
    }
}

impl RecordBatchStream for PendingStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// Helper: collect all output from a SemiAntiSortMergeJoinStream.
async fn collect_stream(stream: SemiAntiSortMergeJoinStream) -> Result<Vec<RecordBatch>> {
    common::collect(Box::pin(stream)).await
}

/// Reproduces the buffer_inner_key_group re-entry bug:
///
/// When buffer_inner_key_group buffers inner rows across batch boundaries
/// and poll_next_inner_batch returns Pending mid-way, the ready! macro
/// exits poll_join. On re-entry, the merge-scan reaches Equal again and
/// calls buffer_inner_key_group a second time — which starts with
/// clear(), destroying the partially collected inner rows. Previously
/// consumed batches are gone, so re-buffering misses them.
///
/// Setup:
/// - Inner: 3 single-row batches, all with key=1, filter values c2=[10, 20, 30]
/// - Outer: 1 row, key=1, filter value c1=10
/// - Filter: c1 == c2 (only first inner row c2=10 matches)
/// - Pending injected before 3rd inner batch
///
/// Without the bug: outer row emitted (match via c2=10)
/// With the bug: outer row missing (c2=10 batch lost on re-entry)
#[tokio::test]
async fn filter_buffer_pending_loses_inner_rows() -> Result<()> {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("a1", DataType::Int32, false),
        Field::new("b1", DataType::Int32, false),
        Field::new("c1", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("a2", DataType::Int32, false),
        Field::new("b1", DataType::Int32, false),
        Field::new("c2", DataType::Int32, false),
    ]));

    // Outer: 1 row, key=1, c1=10
    let outer_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![1])), // join key
            Arc::new(Int32Array::from(vec![10])), // filter value
        ],
    )?;

    // Inner: 3 single-row batches, key=1, c2=[10, 20, 30]
    let inner_batch1 = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(Int32Array::from(vec![100])),
            Arc::new(Int32Array::from(vec![1])), // join key
            Arc::new(Int32Array::from(vec![10])), // matches filter
        ],
    )?;
    let inner_batch2 = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(Int32Array::from(vec![200])),
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![20])), // doesn't match
        ],
    )?;
    let inner_batch3 = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(Int32Array::from(vec![300])),
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![30])), // doesn't match
        ],
    )?;

    let outer: SendableRecordBatchStream = Box::pin(PendingStream::new(
        vec![outer_batch],
        vec![false], // outer delivers immediately
    ));
    let inner: SendableRecordBatchStream = Box::pin(PendingStream::new(
        vec![inner_batch1, inner_batch2, inner_batch3],
        vec![false, false, true], // Pending before 3rd batch
    ));

    // Filter: c1 == c2
    let filter = JoinFilter::new(
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("c1", 0)),
            Operator::Eq,
            Arc::new(Column::new("c2", 1)),
        )),
        vec![
            ColumnIndex {
                index: 2,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 2,
                side: JoinSide::Right,
            },
        ],
        Arc::new(Schema::new(vec![
            Field::new("c1", DataType::Int32, false),
            Field::new("c2", DataType::Int32, false),
        ])),
    );

    let on_outer: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("b1", 1))];
    let on_inner: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("b1", 1))];

    let metrics = ExecutionPlanMetricsSet::new();
    let stream = SemiAntiSortMergeJoinStream::try_new(
        left_schema, // output schema = outer schema for semi
        vec![SortOptions::default()],
        NullEquality::NullEqualsNothing,
        outer,
        inner,
        on_outer,
        on_inner,
        Some(filter),
        LeftSemi,
        8192,
        0,
        &metrics,
    )?;

    let batches = collect_stream(stream).await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total, 1,
        "LeftSemi with filter: outer row should be emitted because \
         inner row c2=10 matches filter c1==c2. Got {total} rows."
    );
    Ok(())
}

/// Reproduces the no-filter boundary Pending re-entry bug:
///
/// When an outer key group spans a batch boundary, the no-filter path
/// emits the current batch, then polls for the next outer batch. If
/// poll returns Pending, poll_join exits. On re-entry, without the
/// BoundaryState fix, the new batch is processed fresh by the
/// merge-scan. Since inner already advanced past this key, the outer
/// rows with the matching key are skipped via Ordering::Less.
///
/// Setup:
/// - Outer: 2 single-row batches, both with key=1 (key group spans boundary)
/// - Inner: 1 row with key=1
/// - Pending injected on outer before 2nd batch
///
/// Without fix: only first outer row emitted (second lost on re-entry)
/// With fix: both outer rows emitted
#[tokio::test]
async fn no_filter_boundary_pending_loses_outer_rows() -> Result<()> {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("a1", DataType::Int32, false),
        Field::new("b1", DataType::Int32, false),
        Field::new("c1", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("a2", DataType::Int32, false),
        Field::new("b1", DataType::Int32, false),
        Field::new("c2", DataType::Int32, false),
    ]));

    // Outer: 2 single-row batches, both key=1
    let outer_batch1 = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![10])),
        ],
    )?;
    let outer_batch2 = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(Int32Array::from(vec![2])),
            Arc::new(Int32Array::from(vec![1])), // same key
            Arc::new(Int32Array::from(vec![20])),
        ],
    )?;

    // Inner: 1 row, key=1
    let inner_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(Int32Array::from(vec![100])),
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![50])),
        ],
    )?;

    let outer: SendableRecordBatchStream = Box::pin(PendingStream::new(
        vec![outer_batch1, outer_batch2],
        vec![false, true], // Pending before 2nd outer batch
    ));
    let inner: SendableRecordBatchStream =
        Box::pin(PendingStream::new(vec![inner_batch], vec![false]));

    let on_outer: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("b1", 1))];
    let on_inner: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("b1", 1))];

    let metrics = ExecutionPlanMetricsSet::new();
    let stream = SemiAntiSortMergeJoinStream::try_new(
        left_schema,
        vec![SortOptions::default()],
        NullEquality::NullEqualsNothing,
        outer,
        inner,
        on_outer,
        on_inner,
        None, // no filter
        LeftSemi,
        8192,
        0,
        &metrics,
    )?;

    let batches = collect_stream(stream).await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total, 2,
        "LeftSemi no filter: both outer rows (key=1) should be emitted \
         because inner has key=1. Got {total} rows."
    );
    Ok(())
}

/// Tests the filtered boundary Pending re-entry: outer key group spans
/// batches with a filter, and poll_next_outer_batch returns Pending.
///
/// Setup:
/// - Outer: 2 single-row batches, both key=1, c1=[10, 20]
/// - Inner: 1 row, key=1, c2=10
/// - Filter: c1 == c2 (first outer row matches, second doesn't)
/// - Pending before 2nd outer batch
///
/// Expected: 1 row (only the first outer row c1=10 passes the filter)
#[tokio::test]
async fn filtered_boundary_pending_outer_rows() -> Result<()> {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("a1", DataType::Int32, false),
        Field::new("b1", DataType::Int32, false),
        Field::new("c1", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("a2", DataType::Int32, false),
        Field::new("b1", DataType::Int32, false),
        Field::new("c2", DataType::Int32, false),
    ]));

    let outer_batch1 = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![10])), // matches filter
        ],
    )?;
    let outer_batch2 = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(Int32Array::from(vec![2])),
            Arc::new(Int32Array::from(vec![1])), // same key
            Arc::new(Int32Array::from(vec![20])), // doesn't match
        ],
    )?;

    let inner_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(Int32Array::from(vec![100])),
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![10])),
        ],
    )?;

    let outer: SendableRecordBatchStream = Box::pin(PendingStream::new(
        vec![outer_batch1, outer_batch2],
        vec![false, true], // Pending before 2nd outer batch
    ));
    let inner: SendableRecordBatchStream =
        Box::pin(PendingStream::new(vec![inner_batch], vec![false]));

    let filter = JoinFilter::new(
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("c1", 0)),
            Operator::Eq,
            Arc::new(Column::new("c2", 1)),
        )),
        vec![
            ColumnIndex {
                index: 2,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 2,
                side: JoinSide::Right,
            },
        ],
        Arc::new(Schema::new(vec![
            Field::new("c1", DataType::Int32, false),
            Field::new("c2", DataType::Int32, false),
        ])),
    );

    let on_outer: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("b1", 1))];
    let on_inner: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("b1", 1))];

    let metrics = ExecutionPlanMetricsSet::new();
    let stream = SemiAntiSortMergeJoinStream::try_new(
        left_schema,
        vec![SortOptions::default()],
        NullEquality::NullEqualsNothing,
        outer,
        inner,
        on_outer,
        on_inner,
        Some(filter),
        LeftSemi,
        8192,
        0,
        &metrics,
    )?;

    let batches = collect_stream(stream).await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total, 1,
        "LeftSemi filtered boundary: only first outer row (c1=10) matches \
         filter c1==c2. Got {total} rows."
    );
    Ok(())
}
