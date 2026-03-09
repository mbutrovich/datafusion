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

//! Execution plan for semi/anti sort-merge joins.

use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;

use super::stream::SemiAntiSortMergeJoinStream;
use crate::check_if_same_properties;
use crate::execution_plan::{EmissionType, boundedness_from_children};
use crate::expressions::PhysicalSortExpr;
use crate::joins::utils::{
    JoinFilter, JoinOn, JoinOnRef, build_join_schema, check_join_is_valid,
    estimate_join_statistics, symmetric_join_output_partitioning,
};
use crate::metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricsSet, SpillMetrics};
use crate::spill::spill_manager::SpillManager;
use crate::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    PlanProperties, SendableRecordBatchStream, Statistics,
};

use arrow::compute::SortOptions;
use arrow::datatypes::SchemaRef;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{
    JoinSide, JoinType, NullEquality, Result, assert_eq_or_internal_err, plan_err,
};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::MemoryConsumer;
use datafusion_physical_expr::equivalence::join_equivalence_properties;
use datafusion_physical_expr_common::physical_expr::fmt_sql;
use datafusion_physical_expr_common::sort_expr::{LexOrdering, OrderingRequirements};

/// Sort-merge join operator specialized for semi/anti joins.
///
/// # Motivation
///
/// The general-purpose [`SortMergeJoinExec`](crate::joins::sort_merge_join::SortMergeJoinExec) handles semi/anti joins by
/// materializing `(outer, inner)` row pairs, applying a filter, then using a
/// "corrected filter mask" to deduplicate. Semi/anti joins only need a boolean
/// per outer row (does a match exist?), not pairs. The pair-based approach
/// incurs unnecessary memory allocation and intermediate batches.
///
/// This operator instead tracks matches with a per-outer-batch bitset,
/// avoiding all pair materialization.
///
/// Supports: `LeftSemi`, `LeftAnti`, `RightSemi`, `RightAnti`.
///
/// # "Outer Side" vs "Inner Side"
///
/// For `Left*` join types, left is outer and right is inner.
/// For `Right*` join types, right is outer and left is inner.
/// The output schema always equals the outer side's schema.
///
/// # Algorithm
///
/// Both inputs must be sorted by the join keys. The stream performs a merge
/// scan across the two sorted inputs:
///
/// ```text
///   outer cursor ──►  [1, 2, 2, 3, 5, 5, 7]
///   inner cursor ──►  [2, 2, 4, 5, 6, 7, 7]
///                       ▲
///                   compare keys at cursors
/// ```
///
/// At each step, the keys at the outer and inner cursors are compared:
///
/// - **outer < inner**: Skip the outer key group (no match exists).
/// - **outer > inner**: Skip the inner key group.
/// - **outer == inner**: Process the match (see below).
///
/// Key groups are contiguous runs of equal keys within one side. The scan
/// advances past entire groups at each step.
///
/// ## Processing a key match
///
/// **Without filter**: All outer rows in the key group are marked as matched.
///
/// **With filter**: The inner key group is buffered (may span multiple inner
/// batches). For each buffered inner row, the filter is evaluated against the
/// outer key group as a batch. Results are OR'd into the matched bitset. A
/// short-circuit exits early when all outer rows in the group are matched.
///
/// ```text
///   matched bitset:  [0, 0, 1, 0, 0, ...]
///                     ▲── one bit per outer row ──▲
///
///   On emit:
///     Semi  → filter_record_batch(outer_batch, &matched)
///     Anti  → filter_record_batch(outer_batch, &NOT(matched))
/// ```
///
/// ## Batch boundaries
///
/// Key groups can span batch boundaries on either side. The stream handles
/// this by detecting when a group extends to the end of a batch, loading the
/// next batch, and continuing if the key matches. The `BoundaryState` enum
/// preserves loop context across async `Poll::Pending` re-entries.
///
/// # Memory
///
/// Memory usage is bounded and independent of total input size:
/// - One outer batch at a time (not tracked by reservation — single batch,
///   cannot be spilled since it's needed for filter evaluation)
/// - One inner batch at a time (streaming)
/// - `matched` bitset: one bit per outer row, re-allocated per batch
/// - Inner key group buffer: only for filtered joins, one key group at a time.
///   Tracked via `MemoryReservation`; spilled to disk when the memory pool
///   limit is exceeded.
/// - `BatchCoalescer`: output buffering to target batch size
///
/// # Degenerate cases
///
/// **Highly skewed key (filtered joins only):** When a filter is present,
/// the inner key group is buffered so each inner row can be evaluated
/// against the outer group. If one join key has N inner rows, all N rows
/// are held in memory simultaneously (or spilled to disk if the memory
/// pool limit is reached). With uniform key distribution this is small
/// (inner_rows / num_distinct_keys), but a single hot key can buffer
/// arbitrarily many rows. The no-filter path does not buffer inner
/// rows — it only advances the cursor — so it is unaffected.
///
/// **Scalar broadcast during filter evaluation:** Each inner row is
/// broadcast to match the outer group length for filter evaluation,
/// allocating one array per inner row × filter column. This is inherent
/// to the `PhysicalExpr::evaluate(RecordBatch)` API, which does not
/// support scalar inputs directly. The total work is
/// O(inner_group × outer_group) per key, but with much lower constant
/// factor than the pair-materialization approach.
#[derive(Debug, Clone)]
pub struct SemiAntiSortMergeJoinExec {
    pub left: Arc<dyn ExecutionPlan>,
    pub right: Arc<dyn ExecutionPlan>,
    pub on: JoinOn,
    pub filter: Option<JoinFilter>,
    pub join_type: JoinType,
    schema: SchemaRef,
    metrics: ExecutionPlanMetricsSet,
    left_sort_exprs: LexOrdering,
    right_sort_exprs: LexOrdering,
    pub sort_options: Vec<SortOptions>,
    pub null_equality: NullEquality,
    cache: Arc<PlanProperties>,
}

impl SemiAntiSortMergeJoinExec {
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        sort_options: Vec<SortOptions>,
        null_equality: NullEquality,
    ) -> Result<Self> {
        if !matches!(
            join_type,
            JoinType::LeftSemi
                | JoinType::LeftAnti
                | JoinType::RightSemi
                | JoinType::RightAnti
        ) {
            return plan_err!(
                "SemiAntiSortMergeJoinExec only supports semi/anti joins, got {:?}",
                join_type
            );
        }

        let left_schema = left.schema();
        let right_schema = right.schema();
        check_join_is_valid(&left_schema, &right_schema, &on)?;

        if sort_options.len() != on.len() {
            return plan_err!(
                "Expected number of sort options: {}, actual: {}",
                on.len(),
                sort_options.len()
            );
        }

        let (left_sort_exprs, right_sort_exprs): (Vec<_>, Vec<_>) = on
            .iter()
            .zip(sort_options.iter())
            .map(|((l, r), sort_op)| {
                let left = PhysicalSortExpr {
                    expr: Arc::clone(l),
                    options: *sort_op,
                };
                let right = PhysicalSortExpr {
                    expr: Arc::clone(r),
                    options: *sort_op,
                };
                (left, right)
            })
            .unzip();

        let Some(left_sort_exprs) = LexOrdering::new(left_sort_exprs) else {
            return plan_err!(
                "SemiAntiSortMergeJoinExec requires valid sort expressions for its left side"
            );
        };
        let Some(right_sort_exprs) = LexOrdering::new(right_sort_exprs) else {
            return plan_err!(
                "SemiAntiSortMergeJoinExec requires valid sort expressions for its right side"
            );
        };

        let schema =
            Arc::new(build_join_schema(&left_schema, &right_schema, &join_type).0);
        let cache =
            Self::compute_properties(&left, &right, Arc::clone(&schema), join_type, &on)?;

        Ok(Self {
            left,
            right,
            on,
            filter,
            join_type,
            schema,
            metrics: ExecutionPlanMetricsSet::new(),
            left_sort_exprs,
            right_sort_exprs,
            sort_options,
            null_equality,
            cache: Arc::new(cache),
        })
    }

    /// The outer (streamed) side: Left for LeftSemi/LeftAnti, Right for RightSemi/RightAnti.
    pub fn probe_side(join_type: &JoinType) -> JoinSide {
        match join_type {
            JoinType::RightSemi | JoinType::RightAnti => JoinSide::Right,
            _ => JoinSide::Left,
        }
    }

    fn maintains_input_order(join_type: JoinType) -> Vec<bool> {
        match join_type {
            JoinType::LeftSemi | JoinType::LeftAnti => vec![true, false],
            JoinType::RightSemi | JoinType::RightAnti => vec![false, true],
            _ => vec![false, false],
        }
    }

    fn compute_properties(
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
        join_type: JoinType,
        join_on: JoinOnRef,
    ) -> Result<PlanProperties> {
        let eq_properties = join_equivalence_properties(
            left.equivalence_properties().clone(),
            right.equivalence_properties().clone(),
            &join_type,
            schema,
            &Self::maintains_input_order(join_type),
            Some(Self::probe_side(&join_type)),
            join_on,
        )?;
        let output_partitioning =
            symmetric_join_output_partitioning(left, right, &join_type)?;
        Ok(PlanProperties::new(
            eq_properties,
            output_partitioning,
            EmissionType::Incremental,
            boundedness_from_children([left, right]),
        ))
    }

    fn with_new_children_and_same_properties(
        &self,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Self {
        let left = children.swap_remove(0);
        let right = children.swap_remove(0);
        Self {
            left,
            right,
            metrics: ExecutionPlanMetricsSet::new(),
            ..Self::clone(self)
        }
    }
}

impl DisplayAs for SemiAntiSortMergeJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let on = self
                    .on
                    .iter()
                    .map(|(c1, c2)| format!("({c1}, {c2})"))
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(
                    f,
                    "{}: join_type={:?}, on=[{}]{}",
                    Self::static_name(),
                    self.join_type,
                    on,
                    self.filter.as_ref().map_or_else(
                        || "".to_string(),
                        |filt| format!(", filter={}", filt.expression())
                    ),
                )
            }
            DisplayFormatType::TreeRender => {
                let on = self
                    .on
                    .iter()
                    .map(|(c1, c2)| {
                        format!("({} = {})", fmt_sql(c1.as_ref()), fmt_sql(c2.as_ref()))
                    })
                    .collect::<Vec<String>>()
                    .join(", ");

                writeln!(f, "join_type={:?}", self.join_type)?;
                writeln!(f, "on={on}")?;
                Ok(())
            }
        }
    }
}

impl ExecutionPlan for SemiAntiSortMergeJoinExec {
    fn name(&self) -> &'static str {
        "SemiAntiSortMergeJoinExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        let (left_expr, right_expr) = self
            .on
            .iter()
            .map(|(l, r)| (Arc::clone(l), Arc::clone(r)))
            .unzip();
        vec![
            Distribution::HashPartitioned(left_expr),
            Distribution::HashPartitioned(right_expr),
        ]
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        vec![
            Some(OrderingRequirements::from(self.left_sort_exprs.clone())),
            Some(OrderingRequirements::from(self.right_sort_exprs.clone())),
        ]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        Self::maintains_input_order(self.join_type)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&dyn crate::PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        let mut tnr = TreeNodeRecursion::Continue;
        for (left, right) in &self.on {
            tnr = tnr.visit_sibling(|| f(left.as_ref()))?;
            tnr = tnr.visit_sibling(|| f(right.as_ref()))?;
        }
        if let Some(filter) = &self.filter {
            tnr = tnr.visit_sibling(|| f(filter.expression().as_ref()))?;
        }
        Ok(tnr)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        check_if_same_properties!(self, children);
        match &children[..] {
            [left, right] => Ok(Arc::new(Self::try_new(
                Arc::clone(left),
                Arc::clone(right),
                self.on.clone(),
                self.filter.clone(),
                self.join_type,
                self.sort_options.clone(),
                self.null_equality,
            )?)),
            _ => datafusion_common::internal_err!(
                "SemiAntiSortMergeJoinExec wrong number of children"
            ),
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let left_partitions = self.left.output_partitioning().partition_count();
        let right_partitions = self.right.output_partitioning().partition_count();
        assert_eq_or_internal_err!(
            left_partitions,
            right_partitions,
            "Invalid SemiAntiSortMergeJoinExec, partition count mismatch \
             {left_partitions}!={right_partitions}"
        );

        let (on_left, on_right): (Vec<_>, Vec<_>) = self.on.iter().cloned().unzip();

        let (outer, inner, on_outer, on_inner) =
            if Self::probe_side(&self.join_type) == JoinSide::Left {
                (
                    Arc::clone(&self.left),
                    Arc::clone(&self.right),
                    on_left,
                    on_right,
                )
            } else {
                (
                    Arc::clone(&self.right),
                    Arc::clone(&self.left),
                    on_right,
                    on_left,
                )
            };

        let outer = outer.execute(partition, Arc::clone(&context))?;
        let inner = inner.execute(partition, Arc::clone(&context))?;
        let batch_size = context.session_config().batch_size();

        let reservation = MemoryConsumer::new(format!("SemiAntiSMJStream[{partition}]"))
            .register(context.memory_pool());
        let peak_mem_used =
            MetricBuilder::new(&self.metrics).gauge("peak_mem_used", partition);
        let spill_manager = SpillManager::new(
            context.runtime_env(),
            SpillMetrics::new(&self.metrics, partition),
            inner.schema(),
        )
        .with_compression_type(context.session_config().spill_compression());

        Ok(Box::pin(SemiAntiSortMergeJoinStream::try_new(
            Arc::clone(&self.schema),
            self.sort_options.clone(),
            self.null_equality,
            outer,
            inner,
            on_outer,
            on_inner,
            self.filter.clone(),
            self.join_type,
            batch_size,
            partition,
            &self.metrics,
            reservation,
            peak_mem_used,
            spill_manager,
            context.runtime_env(),
        )?))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Statistics> {
        estimate_join_statistics(
            self.left.partition_statistics(partition)?,
            self.right.partition_statistics(partition)?,
            &self.on,
            &self.join_type,
            &self.schema,
        )
    }
}
