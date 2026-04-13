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

//! Sort that deals with an arbitrary size of the input.
//! It will do in-memory sorting if it has enough memory budget
//! but spills to disk if needed.

use std::fmt;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::common::spawn_buffered;
use crate::execution_plan::{
    Boundedness, CardinalityEffect, EmissionType, has_same_children_properties,
};
use crate::expressions::PhysicalSortExpr;
use crate::filter_pushdown::{
    ChildFilterDescription, FilterDescription, FilterPushdownPhase,
};
use crate::limit::LimitStream;
use crate::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet, SpillMetrics,
};
use crate::projection::{ProjectionExec, make_with_child, update_ordering};
use crate::sorts::IncrementalSortIterator;
use crate::sorts::streaming_merge::{SortedSpillFile, StreamingMergeBuilder};
use crate::spill::get_record_batch_memory_size;
use crate::spill::spill_manager::{GetSlicedSize, SpillManager};
use crate::stream::RecordBatchStreamAdapter;
use crate::stream::ReservationStream;
use crate::topk::TopK;
use crate::topk::TopKDynamicFilters;
use crate::{
    DisplayAs, DisplayFormatType, Distribution, EmptyRecordBatchStream, ExecutionPlan,
    ExecutionPlanProperties, Partitioning, PlanProperties, SendableRecordBatchStream,
    Statistics,
};

use arrow::array::{Array, RecordBatch, RecordBatchOptions, StringViewArray};
use arrow::compute::{BatchCoalescer, lexsort_to_indices, take_arrays};
use arrow::datatypes::{DataType, SchemaRef};
use datafusion_common::config::SpillCompression;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{
    DataFusionError, Result, assert_or_internal_err, unwrap_or_internal_err,
};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{DynamicFilterPhysicalExpr, lit};

use futures::{StreamExt, TryStreamExt};
use log::trace;

struct ExternalSorterMetrics {
    /// metrics
    baseline: BaselineMetrics,

    spill_metrics: SpillMetrics,
}

impl ExternalSorterMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            baseline: BaselineMetrics::new(metrics, partition),
            spill_metrics: SpillMetrics::new(metrics, partition),
        }
    }
}

/// Sorts an arbitrary sized, unsorted, stream of [`RecordBatch`]es to
/// a total order. Depending on the input size and memory manager
/// configuration, writes intermediate results to disk ("spills")
/// using Arrow IPC format.
///
/// # Algorithm
///
/// Incoming batches are coalesced via [`BatchCoalescer`] to a target
/// row count before sorting. For radix-eligible schemas (primitives,
/// strings) the target is `sort_coalesce_target_rows` (default 32768);
/// for non-radix schemas (all-dictionary, nested types) it falls back
/// to `batch_size`. This gives sort kernels enough rows to amortize
/// overhead — radix sort is 2-3x faster than lexsort at 32K+ rows
/// but slower at small batch sizes.
///
/// Each coalesced batch is sorted immediately (radix or lexsort) and
/// stored as a pre-sorted run. Under memory pressure the coalescer
/// flushes early, producing smaller runs that fall back to lexsort.
///
/// 1. For each incoming batch:
///    - Reserve memory (2x batch size). If reservation fails, flush
///      the coalescer, spill all sorted runs to disk, then retry.
///    - Push batch into the coalescer.
///    - If the coalescer reached its target: sort the coalesced batch
///      and store as a new sorted run.
///
/// 2. When input is exhausted, merge all sorted runs (and any spill
///    files) to produce a total order.
///
/// # When data fits in available memory
///
/// Sorted runs are merged in memory using a loser-tree k-way merge
/// (via [`StreamingMergeBuilder`]).
///
/// ```text
///   ┌──────────┐     ┌────────────┐     ┌──────┐     ┌────────────┐
///   │ Incoming │────▶│  Batch     │────▶│ Sort │────▶│ Sorted Run │
///   │ Batches  │     │ Coalescer  │     │      │     │ (in memory)│
///   └──────────┘     └────────────┘     └──────┘     └─────┬──────┘
///                                                          │
///                                           ┌──────────────┘
///                                           ▼
///                                    k-way merge (loser tree)
///                                           │
///                                           ▼
///                                    total sorted output
/// ```
///
/// # When data does not fit in available memory
///
/// When memory is exhausted, sorted runs are spilled directly to disk
/// (one spill file per run — no merge needed since runs are already
/// sorted). [`MultiLevelMerge`] handles the final merge from disk
/// with dynamic fan-in.
///
/// ```text
///   ┌──────────┐     ┌────────────┐     ┌──────┐     ┌────────────┐
///   │ Incoming │────▶│  Batch     │────▶│ Sort │────▶│ Sorted Run │
///   │ Batches  │     │ Coalescer  │     │      │     │            │
///   └──────────┘     └────────────┘     └──────┘     └─────┬──────┘
///                                                          │
///                           memory pressure ◀──────────────┘
///                                  │
///                                  ▼
///                       .─────────────────.
///                      (   Spill to disk   )
///                      │  (one file/run)   │
///                       `─────────────────'
///                                  │
///              ┌───────────────────┘
///              ▼
///   MultiLevelMerge (dynamic fan-in)
///              │
///              ▼
///   total sorted output
/// ```
///
/// # Graceful degradation
///
/// The coalesce target (32K rows) is aspirational. Under memory
/// pressure, chunk sizes shrink and radix sort amortizes less — at
/// `batch_size` or below, the pipeline falls back to lexsort,
/// matching the old per-batch sort behavior.
/// ```
struct ExternalSorter {
    // ========================================================================
    // PROPERTIES:
    // Fields that define the sorter's configuration and remain constant
    // ========================================================================
    /// Schema of the output (and the input)
    schema: SchemaRef,
    /// Sort expressions
    expr: LexOrdering,
    /// The target number of rows for output batches
    batch_size: usize,
    /// Whether to use radix sort (decided once from expression types).
    use_radix: bool,

    // ========================================================================
    // STATE BUFFERS:
    // Fields that hold intermediate data during sorting
    // ========================================================================
    /// Accumulates incoming batches until `coalesce_target_rows` is reached,
    /// at which point the coalesced batch is sorted and stored as a run.
    /// Set to `None` after `sort()` consumes it.
    coalescer: Option<BatchCoalescer>,

    /// Pre-sorted runs of `batch_size`-chunked `RecordBatch`es. Each inner
    /// `Vec` is a single sorted run produced by sorting one coalesced batch.
    sorted_runs: Vec<Vec<RecordBatch>>,

    /// If data has previously been spilled, the locations of the spill files (in
    /// Arrow IPC format)
    /// Within the same spill file, the data might be chunked into multiple batches,
    /// and ordered by sort keys.
    finished_spill_files: Vec<SortedSpillFile>,

    // ========================================================================
    // EXECUTION RESOURCES:
    // Fields related to managing execution resources and monitoring performance.
    // ========================================================================
    /// Runtime metrics
    metrics: ExternalSorterMetrics,
    /// A handle to the runtime to get spill files
    runtime: Arc<RuntimeEnv>,
    /// Reservation for sorted_runs (and coalescer contents)
    reservation: MemoryReservation,
    spill_manager: SpillManager,

    /// Reservation for the merging of sorted runs. If the sort
    /// might spill, `sort_spill_reservation_bytes` will be
    /// pre-reserved to ensure there is some space for this sort/merge.
    merge_reservation: MemoryReservation,
    /// How much memory to reserve for performing in-memory sort/merges
    /// prior to spilling.
    sort_spill_reservation_bytes: usize,
}

impl ExternalSorter {
    // TODO: make a builder or some other nicer API to avoid the
    // clippy warning
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        partition_id: usize,
        schema: SchemaRef,
        expr: LexOrdering,
        batch_size: usize,
        sort_spill_reservation_bytes: usize,
        sort_coalesce_target_rows: usize,
        // Configured via `datafusion.execution.spill_compression`.
        spill_compression: SpillCompression,
        metrics: &ExecutionPlanMetricsSet,
        runtime: Arc<RuntimeEnv>,
    ) -> Result<Self> {
        let metrics = ExternalSorterMetrics::new(metrics, partition_id);
        let reservation = MemoryConsumer::new(format!("ExternalSorter[{partition_id}]"))
            .with_can_spill(true)
            .register(&runtime.memory_pool);

        let merge_reservation =
            MemoryConsumer::new(format!("ExternalSorterMerge[{partition_id}]"))
                .register(&runtime.memory_pool);

        let spill_manager = SpillManager::new(
            Arc::clone(&runtime),
            metrics.spill_metrics.clone(),
            Arc::clone(&schema),
        )
        .with_compression_type(spill_compression);

        let sort_data_types: Vec<DataType> = expr
            .iter()
            .map(|e| e.expr.data_type(&schema))
            .collect::<Result<_>>()?;
        let use_radix = use_radix_sort(&sort_data_types.iter().collect::<Vec<_>>());

        let coalesce_target_rows = if use_radix {
            sort_coalesce_target_rows
        } else {
            batch_size
        };

        let coalescer = BatchCoalescer::new(Arc::clone(&schema), coalesce_target_rows);

        Ok(Self {
            schema,
            coalescer: Some(coalescer),
            sorted_runs: vec![],
            finished_spill_files: vec![],
            expr,
            metrics,
            reservation,
            spill_manager,
            merge_reservation,
            runtime,
            batch_size,
            sort_spill_reservation_bytes,
            use_radix,
        })
    }

    /// Appends an unsorted [`RecordBatch`] to the coalescer.
    ///
    /// The coalescer accumulates batches until `coalesce_target_rows` is
    /// reached, then sorts the coalesced batch and stores it as a sorted run.
    /// Updates memory usage metrics, and possibly triggers spilling to disk.
    async fn insert_batch(&mut self, input: RecordBatch) -> Result<()> {
        if input.num_rows() == 0 {
            return Ok(());
        }

        self.reserve_memory_for_merge()?;
        self.reserve_memory_for_batch_and_maybe_spill(&input)
            .await?;

        let coalescer = self
            .coalescer
            .as_mut()
            .expect("coalescer must exist during insert phase");
        coalescer
            .push_batch(input)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

        self.drain_completed_batches()?;

        Ok(())
    }

    /// Drains completed (full) batches from the coalescer, sorts each,
    /// and appends the sorted chunks to `sorted_runs`.
    fn drain_completed_batches(&mut self) -> Result<()> {
        // Collect completed batches first to avoid borrow conflict
        let mut completed = vec![];
        if let Some(coalescer) = self.coalescer.as_mut() {
            while let Some(batch) = coalescer.next_completed_batch() {
                completed.push(batch);
            }
        }
        for batch in &completed {
            self.sort_and_store_run(batch)?;
        }
        Ok(())
    }

    /// Sorts a single coalesced batch and stores the result as a new run.
    ///
    /// Uses radix sort when the batch is large enough to amortize encoding
    /// overhead (more than `batch_size` rows). Otherwise falls back to lexsort.
    fn sort_and_store_run(&mut self, batch: &RecordBatch) -> Result<()> {
        let use_radix_for_this_batch =
            self.use_radix && batch.num_rows() > self.batch_size;

        let sorted_chunks = if use_radix_for_this_batch {
            sort_batch_chunked(batch, &self.expr, self.batch_size, true)?
        } else {
            vec![sort_batch(batch, &self.expr, None)?]
        };

        // After take(), StringView arrays may reference shared buffers from
        // multiple coalesced input batches, inflating reported memory size.
        // GC compacts them so reservation tracking stays accurate.
        let sorted_chunks = Self::gc_stringview_batches(sorted_chunks)?;
        self.sorted_runs.push(sorted_chunks);

        // The 2x reservation from input batches exceeds the 1x sorted output.
        // Shrink to release the excess back to the pool.
        let target = self.reservation.size().min(
            self.sorted_runs
                .iter()
                .flat_map(|r| r.iter())
                .map(get_record_batch_memory_size)
                .sum(),
        );
        self.reservation.shrink(self.reservation.size() - target);

        Ok(())
    }

    /// Compact StringView arrays in sorted batches to eliminate shared
    /// buffer references from `take()`. Skips work if no StringView columns.
    fn gc_stringview_batches(batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
        // Fast path: check schema for any StringView columns
        if let Some(first) = batches.first() {
            let has_stringview = first.schema().fields().iter().any(|f| {
                matches!(f.data_type(), DataType::Utf8View | DataType::BinaryView)
            });
            if !has_stringview {
                return Ok(batches);
            }
        }

        let mut result = Vec::with_capacity(batches.len());
        for batch in batches {
            let mut new_columns: Vec<Arc<dyn Array>> =
                Vec::with_capacity(batch.num_columns());
            let mut mutated = false;
            for array in batch.columns() {
                if let Some(sv) = array.as_any().downcast_ref::<StringViewArray>() {
                    new_columns.push(Arc::new(sv.gc()));
                    mutated = true;
                } else {
                    new_columns.push(Arc::clone(array));
                }
            }
            if mutated {
                result.push(RecordBatch::try_new(batch.schema(), new_columns)?);
            } else {
                result.push(batch);
            }
        }
        Ok(result)
    }

    /// Flushes any partially accumulated rows from the coalescer, sorts them,
    /// and stores as a run. Called before spilling and at sort() time.
    fn flush_coalescer(&mut self) -> Result<()> {
        if let Some(coalescer) = self.coalescer.as_mut() {
            coalescer
                .finish_buffered_batch()
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            self.drain_completed_batches()?;
        }
        Ok(())
    }

    fn spilled_before(&self) -> bool {
        !self.finished_spill_files.is_empty()
    }

    /// Returns true if there are sorted runs in memory.
    fn has_sorted_runs(&self) -> bool {
        !self.sorted_runs.is_empty()
    }

    /// Returns the final sorted output of all batches inserted via
    /// [`Self::insert_batch`] as a stream of [`RecordBatch`]es.
    ///
    /// This process could either be:
    ///
    /// 1. An in-memory merge of sorted runs (if the input fit in memory)
    ///
    /// 2. A combined streaming merge incorporating sorted runs
    ///    and data from spill files on disk.
    async fn sort(&mut self) -> Result<SendableRecordBatchStream> {
        self.flush_coalescer()?;
        self.coalescer = None;

        // Determine if we must take the spill path.
        //
        // We must spill if:
        // 1. We already spilled during the insert phase, OR
        // 2. We have multiple sorted runs but merge_reservation is 0.
        //
        // Case 2 matters because the in-memory merge needs to allocate
        // cursor infrastructure (RowCursorStream / FieldCursorStream)
        // at build time, before any run data is consumed. The cursor
        // allocation comes from merge_reservation. If that's 0, the
        // pool is fully occupied by sorted run data and the cursor
        // can't allocate. Spilling to disk frees pool memory, and
        // MultiLevelMerge handles the merge with dynamic fan-in —
        // reading from spill files that don't hold pool memory.
        let must_spill = self.spilled_before()
            || (self.sorted_runs.len() > 1 && self.merge_reservation.size() == 0);

        if must_spill {
            // Spill remaining sorted runs. Since runs are already sorted,
            // each is written directly as its own spill file (no merge needed).
            if self.has_sorted_runs() {
                self.spill_sorted_runs().await?;
            }

            StreamingMergeBuilder::new()
                .with_sorted_spill_files(std::mem::take(&mut self.finished_spill_files))
                .with_spill_manager(self.spill_manager.clone())
                .with_schema(Arc::clone(&self.schema))
                .with_expressions(&self.expr)
                .with_metrics(self.metrics.baseline.clone())
                .with_batch_size(self.batch_size)
                .with_fetch(None)
                .with_reservation(self.merge_reservation.take())
                .build()
        } else {
            // In-memory path: we have 0 runs, 1 run (no merge needed),
            // or multiple runs with merge_reservation > 0 providing
            // headroom for cursor allocation.
            //
            // Release merge_reservation back to the pool — in the
            // non-spill path, merge_sorted_runs allocates cursor memory
            // from the pool directly (freed merge_reservation bytes).
            self.merge_reservation.free();
            self.merge_sorted_runs(self.metrics.baseline.clone())
        }
    }

    /// How much memory is buffered in this `ExternalSorter`?
    fn used(&self) -> usize {
        self.reservation.size()
    }

    /// How much memory is reserved for the merge phase?
    #[cfg(test)]
    fn merge_reservation_size(&self) -> usize {
        self.merge_reservation.size()
    }

    /// How many bytes have been spilled to disk?
    fn spilled_bytes(&self) -> usize {
        self.metrics.spill_metrics.spilled_bytes.value()
    }

    /// How many rows have been spilled to disk?
    fn spilled_rows(&self) -> usize {
        self.metrics.spill_metrics.spilled_rows.value()
    }

    /// How many spill files have been created?
    fn spill_count(&self) -> usize {
        self.metrics.spill_metrics.spill_file_count.value()
    }

    /// Spills sorted runs to disk.
    ///
    /// Two strategies depending on available merge headroom:
    ///
    /// - **With headroom** (`merge_reservation > 0`): merge all runs into
    ///   a single globally-sorted stream, then write to one spill file.
    ///   Fewer spill files = lower fan-in for the final MultiLevelMerge.
    ///
    /// - **Without headroom** (`merge_reservation == 0`): spill each run
    ///   as its own file. Avoids allocating merge cursor infrastructure
    ///   when the pool has no room. MultiLevelMerge handles the higher
    ///   fan-in with dynamic memory management.
    async fn spill_sorted_runs(&mut self) -> Result<()> {
        assert_or_internal_err!(
            self.has_sorted_runs(),
            "sorted_runs must not be empty when attempting to spill"
        );

        if self.merge_reservation.size() > 0 && self.sorted_runs.len() > 1 {
            // Free merge_reservation to provide pool headroom for the
            // merge cursor allocation. Re-reserved at the end.
            self.merge_reservation.free();

            let mut sorted_stream =
                self.merge_sorted_runs(self.metrics.baseline.intermediate())?;
            assert_or_internal_err!(
                self.sorted_runs.is_empty(),
                "sorted_runs should be empty after constructing sorted stream"
            );

            let mut in_progress =
                self.spill_manager.create_in_progress_file("Sorting")?;
            let mut max_batch_memory = 0usize;

            while let Some(batch) = sorted_stream.next().await {
                let batch = batch?;
                max_batch_memory = max_batch_memory.max(batch.get_sliced_size()?);
                in_progress.append_batch(&batch)?;
            }

            drop(sorted_stream);
            self.reservation.free();

            let spill_file = in_progress.finish()?;
            if let Some(spill_file) = spill_file {
                self.finished_spill_files.push(SortedSpillFile {
                    file: spill_file,
                    max_record_batch_memory: max_batch_memory,
                });
            }
        } else {
            // No merge headroom or single run: spill each run directly.
            let all_runs = std::mem::take(&mut self.sorted_runs);
            for run in all_runs {
                let run_size: usize = run.iter().map(get_record_batch_memory_size).sum();

                let mut in_progress =
                    self.spill_manager.create_in_progress_file("Sorting")?;
                let mut max_batch_memory = 0usize;
                for batch in &run {
                    in_progress.append_batch(batch)?;
                    max_batch_memory = max_batch_memory.max(batch.get_sliced_size()?);
                }

                let spill_file = in_progress.finish()?;
                if let Some(spill_file) = spill_file {
                    self.finished_spill_files.push(SortedSpillFile {
                        file: spill_file,
                        max_record_batch_memory: max_batch_memory,
                    });
                }

                drop(run);
                self.reservation
                    .shrink(run_size.min(self.reservation.size()));
            }
        }

        self.reserve_memory_for_merge()?;

        Ok(())
    }

    /// Merges the pre-sorted runs stored in `sorted_runs` into a single
    /// sorted output stream. Each run is already sorted internally; this
    /// method k-way merges them using the loser tree.
    ///
    /// ```text
    ///   sorted_runs[0]                sorted_runs[1]
    ///   ┌─────┐ ┌─────┐              ┌─────┐ ┌─────┐
    ///   │ 1,2 │ │ 3,4 │              │ 1,3 │ │ 5,7 │
    ///   └──┬──┘ └──┬──┘              └──┬──┘ └──┬──┘
    ///      └───┬───┘                    └───┬───┘
    ///          ▼                            ▼
    ///     stream 0  ─ ─ ─ ─ ─ ─ ─▶  merge  ◀─ ─ ─  stream 1
    ///                                  │
    ///                                  ▼
    ///                          sorted output stream
    /// ```
    fn merge_sorted_runs(
        &mut self,
        metrics: BaselineMetrics,
    ) -> Result<SendableRecordBatchStream> {
        let all_runs = std::mem::take(&mut self.sorted_runs);

        if all_runs.is_empty() {
            return Ok(Box::pin(EmptyRecordBatchStream::new(Arc::clone(
                &self.schema,
            ))));
        }

        let elapsed_compute = metrics.elapsed_compute().clone();
        let _timer = elapsed_compute.timer();

        // Single run: stream the chunks directly, no merge needed
        if all_runs.len() == 1 {
            let run = all_runs.into_iter().next().unwrap();
            let reservation = self.reservation.take();
            let schema = Arc::clone(&self.schema);
            let output_rows = metrics.output_rows().clone();
            let stream =
                futures::stream::iter(run.into_iter().map(Ok)).map(move |batch| {
                    match batch {
                        Ok(batch) => {
                            output_rows.add(batch.num_rows());
                            Ok(batch)
                        }
                        Err(e) => Err(e),
                    }
                });
            return Ok(Box::pin(ReservationStream::new(
                Arc::clone(&schema),
                Box::pin(RecordBatchStreamAdapter::new(schema, stream)),
                reservation,
            )));
        }

        // Multiple runs: create one stream per run and merge.
        //
        // Memory model for the multi-run merge:
        // - self.reservation holds the sorted run data. It stays allocated
        //   for the lifetime of the ExternalSorter (freed on drop). This
        //   over-reserves as runs are consumed, but is conservative/safe.
        // - The merge cursor (RowCursorStream/FieldCursorStream) allocates
        //   from a new_empty() reservation, drawing from pool headroom
        //   freed by merge_reservation.free() in the caller.
        // - This works because sort() only enters this path when
        //   merge_reservation > 0, guaranteeing pool headroom for cursors.
        //   When merge_reservation == 0, sort() takes the spill path instead.
        let streams = all_runs
            .into_iter()
            .map(|run| {
                let schema = Arc::clone(&self.schema);
                let intermediate_metrics = self.metrics.baseline.intermediate();
                let output_rows = intermediate_metrics.output_rows().clone();
                let stream =
                    futures::stream::iter(run.into_iter().map(Ok)).map(move |batch| {
                        match batch {
                            Ok(batch) => {
                                output_rows.add(batch.num_rows());
                                Ok(batch)
                            }
                            Err(e) => Err(e),
                        }
                    });
                let boxed: SendableRecordBatchStream =
                    Box::pin(RecordBatchStreamAdapter::new(schema, stream));
                Ok(spawn_buffered(boxed, 1))
            })
            .collect::<Result<_>>()?;

        StreamingMergeBuilder::new()
            .with_streams(streams)
            .with_schema(Arc::clone(&self.schema))
            .with_expressions(&self.expr)
            .with_metrics(metrics)
            .with_batch_size(self.batch_size)
            .with_fetch(None)
            .with_reservation(self.reservation.new_empty())
            .build()
    }

    /// If this sort may spill, pre-allocates
    /// `sort_spill_reservation_bytes` of memory to guarantee memory
    /// left for the in memory sort/merge.
    fn reserve_memory_for_merge(&mut self) -> Result<()> {
        // Reserve headroom for next merge sort
        if self.runtime.disk_manager.tmp_files_enabled() {
            let size = self.sort_spill_reservation_bytes;
            if self.merge_reservation.size() != size {
                self.merge_reservation
                    .try_resize(size)
                    .map_err(Self::err_with_oom_context)?;
            }
        }

        Ok(())
    }

    /// Reserves memory to be able to accommodate the given batch.
    /// If memory is scarce, flushes the coalescer, spills sorted runs to disk,
    /// and retries.
    async fn reserve_memory_for_batch_and_maybe_spill(
        &mut self,
        input: &RecordBatch,
    ) -> Result<()> {
        let size = get_reserved_bytes_for_record_batch(input)?;

        match self.reservation.try_grow(size) {
            Ok(_) => Ok(()),
            Err(e) => {
                // Sort whatever the coalescer has accumulated, then spill
                // all sorted runs to disk to free pool memory.
                self.flush_coalescer()?;

                if !self.has_sorted_runs() {
                    return Err(Self::err_with_oom_context(e));
                }

                self.spill_sorted_runs().await?;
                self.reservation
                    .try_grow(size)
                    .map_err(Self::err_with_oom_context)
            }
        }
    }

    /// Wraps the error with a context message suggesting settings to tweak.
    /// This is meant to be used with DataFusionError::ResourcesExhausted only.
    fn err_with_oom_context(e: DataFusionError) -> DataFusionError {
        match e {
            DataFusionError::ResourcesExhausted(_) => e.context(
                "Not enough memory to continue external sort. \
                    Consider increasing the memory limit config: 'datafusion.runtime.memory_limit', \
                    or decreasing the config: 'datafusion.execution.sort_spill_reservation_bytes'."
            ),
            // This is not an OOM error, so just return it as is.
            _ => e,
        }
    }
}

/// Estimate how much memory is needed to sort a `RecordBatch`.
///
/// This is used to pre-reserve memory for the sort/merge. The sort/merge process involves
/// creating sorted copies of sorted columns in record batches for speeding up comparison
/// in sorting and merging. The sorted copies are in either row format or array format.
/// Please refer to cursor.rs and stream.rs for more details. No matter what format the
/// sorted copies are, they will use more memory than the original record batch.
///
/// This can basically be calculated as the sum of the actual space it takes in
/// memory (which would be larger for a sliced batch), and the size of the actual data.
pub(crate) fn get_reserved_bytes_for_record_batch_size(
    record_batch_size: usize,
    sliced_size: usize,
) -> usize {
    // Even 2x may not be enough for some cases, but it's a good enough estimation as a baseline.
    // If 2x is not enough, user can set a larger value for `sort_spill_reservation_bytes`
    // to compensate for the extra memory needed.
    record_batch_size + sliced_size
}

/// Estimate how much memory is needed to sort a `RecordBatch`.
/// This will just call `get_reserved_bytes_for_record_batch_size` with the
/// memory size of the record batch and its sliced size.
pub(crate) fn get_reserved_bytes_for_record_batch(batch: &RecordBatch) -> Result<usize> {
    batch.get_sliced_size().map(|sliced_size| {
        get_reserved_bytes_for_record_batch_size(
            get_record_batch_memory_size(batch),
            sliced_size,
        )
    })
}

impl Debug for ExternalSorter {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("ExternalSorter")
            .field("memory_used", &self.used())
            .field("spilled_bytes", &self.spilled_bytes())
            .field("spilled_rows", &self.spilled_rows())
            .field("spill_count", &self.spill_count())
            .finish()
    }
}

pub fn sort_batch(
    batch: &RecordBatch,
    expressions: &LexOrdering,
    fetch: Option<usize>,
) -> Result<RecordBatch> {
    let sort_columns = expressions
        .iter()
        .map(|expr| expr.evaluate_to_sort_column(batch))
        .collect::<Result<Vec<_>>>()?;

    if fetch.is_none()
        && use_radix_sort(
            &sort_columns
                .iter()
                .map(|c| c.values.data_type())
                .collect::<Vec<_>>(),
        )
    {
        let indices = super::radix::radix_sort_to_indices(&sort_columns)?;
        let columns = take_arrays(batch.columns(), &indices, None)?;
        let options = RecordBatchOptions::new().with_row_count(Some(indices.len()));
        return Ok(RecordBatch::try_new_with_options(
            batch.schema(),
            columns,
            &options,
        )?);
    }

    let indices = lexsort_to_indices(&sort_columns, fetch)?;
    let columns = take_arrays(batch.columns(), &indices, None)?;

    let options = RecordBatchOptions::new().with_row_count(Some(indices.len()));
    Ok(RecordBatch::try_new_with_options(
        batch.schema(),
        columns,
        &options,
    )?)
}

/// Returns true if radix sort should be used for the given sort column types.
///
/// Radix sort is faster for most multi-column sorts but falls back to
/// lexsort when:
/// - All sort columns are dictionary-typed (long shared row prefixes
///   waste radix passes before falling back to comparison sort)
/// - Any sort column is a nested type (encoding cost is high and lexsort
///   short-circuits comparison on leading columns)
pub(super) fn use_radix_sort(data_types: &[&DataType]) -> bool {
    if data_types.is_empty() {
        return false;
    }

    let mut all_dict = true;
    for dt in data_types {
        match dt {
            DataType::Dictionary(_, _) => {}
            DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(_, _)
            | DataType::Struct(_)
            | DataType::Map(_, _)
            | DataType::Union(_, _) => return false,
            _ => all_dict = false,
        }
    }

    !all_dict
}

/// Sort a batch and return the result as multiple batches of size `batch_size`.
/// This is useful when you want to avoid creating one large sorted batch in memory,
/// and instead want to process the sorted data in smaller chunks.
pub fn sort_batch_chunked(
    batch: &RecordBatch,
    expressions: &LexOrdering,
    batch_size: usize,
    use_radix: bool,
) -> Result<Vec<RecordBatch>> {
    IncrementalSortIterator::new(
        batch.clone(),
        expressions.clone(),
        batch_size,
        use_radix,
    )
    .collect()
}

/// Sort execution plan.
///
/// Support sorting datasets that are larger than the memory allotted
/// by the memory manager, by spilling to disk.
#[derive(Debug, Clone)]
pub struct SortExec {
    /// Input schema
    pub(crate) input: Arc<dyn ExecutionPlan>,
    /// Sort expressions
    expr: LexOrdering,
    /// Containing all metrics set created during sort
    metrics_set: ExecutionPlanMetricsSet,
    /// Preserve partitions of input plan. If false, the input partitions
    /// will be sorted and merged into a single output partition.
    preserve_partitioning: bool,
    /// Fetch highest/lowest n results
    fetch: Option<usize>,
    /// Normalized common sort prefix between the input and the sort expressions (only used with fetch)
    common_sort_prefix: Vec<PhysicalSortExpr>,
    /// Cache holding plan properties like equivalences, output partitioning etc.
    cache: Arc<PlanProperties>,
    /// Filter matching the state of the sort for dynamic filter pushdown.
    /// If `fetch` is `Some`, this will also be set and a TopK operator may be used.
    /// If `fetch` is `None`, this will be `None`.
    filter: Option<Arc<RwLock<TopKDynamicFilters>>>,
}

impl SortExec {
    /// Create a new sort execution plan that produces a single,
    /// sorted output partition.
    pub fn new(expr: LexOrdering, input: Arc<dyn ExecutionPlan>) -> Self {
        let preserve_partitioning = false;
        let (cache, sort_prefix) =
            Self::compute_properties(&input, expr.clone(), preserve_partitioning)
                .unwrap();
        Self {
            expr,
            input,
            metrics_set: ExecutionPlanMetricsSet::new(),
            preserve_partitioning,
            fetch: None,
            common_sort_prefix: sort_prefix,
            cache: Arc::new(cache),
            filter: None,
        }
    }

    /// Whether this `SortExec` preserves partitioning of the children
    pub fn preserve_partitioning(&self) -> bool {
        self.preserve_partitioning
    }

    /// Specify the partitioning behavior of this sort exec
    ///
    /// If `preserve_partitioning` is true, sorts each partition
    /// individually, producing one sorted stream for each input partition.
    ///
    /// If `preserve_partitioning` is false, sorts and merges all
    /// input partitions producing a single, sorted partition.
    pub fn with_preserve_partitioning(mut self, preserve_partitioning: bool) -> Self {
        self.preserve_partitioning = preserve_partitioning;
        Arc::make_mut(&mut self.cache).partitioning =
            Self::output_partitioning_helper(&self.input, self.preserve_partitioning);
        self
    }

    /// Add or reset `self.filter` to a new `TopKDynamicFilters`.
    fn create_filter(&self) -> Arc<RwLock<TopKDynamicFilters>> {
        let children = self
            .expr
            .iter()
            .map(|sort_expr| Arc::clone(&sort_expr.expr))
            .collect::<Vec<_>>();
        Arc::new(RwLock::new(TopKDynamicFilters::new(Arc::new(
            DynamicFilterPhysicalExpr::new(children, lit(true)),
        ))))
    }

    fn cloned(&self) -> Self {
        SortExec {
            input: Arc::clone(&self.input),
            expr: self.expr.clone(),
            metrics_set: self.metrics_set.clone(),
            preserve_partitioning: self.preserve_partitioning,
            common_sort_prefix: self.common_sort_prefix.clone(),
            fetch: self.fetch,
            cache: Arc::clone(&self.cache),
            filter: self.filter.clone(),
        }
    }

    /// Modify how many rows to include in the result
    ///
    /// If None, then all rows will be returned, in sorted order.
    /// If Some, then only the top `fetch` rows will be returned.
    /// This can reduce the memory pressure required by the sort
    /// operation since rows that are not going to be included
    /// can be dropped.
    pub fn with_fetch(&self, fetch: Option<usize>) -> Self {
        let mut cache = PlanProperties::clone(&self.cache);
        // If the SortExec can emit incrementally (that means the sort requirements
        // and properties of the input match), the SortExec can generate its result
        // without scanning the entire input when a fetch value exists.
        let is_pipeline_friendly = matches!(
            cache.emission_type,
            EmissionType::Incremental | EmissionType::Both
        );
        if fetch.is_some() && is_pipeline_friendly {
            cache = cache.with_boundedness(Boundedness::Bounded);
        }
        let filter = fetch.is_some().then(|| {
            // If we already have a filter, keep it. Otherwise, create a new one.
            self.filter.clone().unwrap_or_else(|| self.create_filter())
        });
        let mut new_sort = self.cloned();
        new_sort.fetch = fetch;
        new_sort.cache = cache.into();
        new_sort.filter = filter;
        new_sort
    }

    /// Input schema
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Sort expressions
    pub fn expr(&self) -> &LexOrdering {
        &self.expr
    }

    /// If `Some(fetch)`, limits output to only the first "fetch" items
    pub fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn output_partitioning_helper(
        input: &Arc<dyn ExecutionPlan>,
        preserve_partitioning: bool,
    ) -> Partitioning {
        // Get output partitioning:
        if preserve_partitioning {
            input.output_partitioning().clone()
        } else {
            Partitioning::UnknownPartitioning(1)
        }
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    /// It also returns the common sort prefix between the input and the sort expressions.
    fn compute_properties(
        input: &Arc<dyn ExecutionPlan>,
        sort_exprs: LexOrdering,
        preserve_partitioning: bool,
    ) -> Result<(PlanProperties, Vec<PhysicalSortExpr>)> {
        let (sort_prefix, sort_satisfied) = input
            .equivalence_properties()
            .extract_common_sort_prefix(sort_exprs.clone())?;

        // The emission type depends on whether the input is already sorted:
        // - If already fully sorted, we can emit results in the same way as the input
        // - If not sorted, we must wait until all data is processed to emit results (Final)
        let emission_type = if sort_satisfied {
            input.pipeline_behavior()
        } else {
            EmissionType::Final
        };

        // The boundedness depends on whether the input is already sorted:
        // - If already sorted, we have the same property as the input
        // - If not sorted and input is unbounded, we require infinite memory and generates
        //   unbounded data (not practical).
        // - If not sorted and input is bounded, then the SortExec is bounded, too.
        let boundedness = if sort_satisfied {
            input.boundedness()
        } else {
            match input.boundedness() {
                Boundedness::Unbounded { .. } => Boundedness::Unbounded {
                    requires_infinite_memory: true,
                },
                bounded => bounded,
            }
        };

        // Calculate equivalence properties; i.e. reset the ordering equivalence
        // class with the new ordering:
        let mut eq_properties = input.equivalence_properties().clone();
        eq_properties.reorder(sort_exprs)?;

        // Get output partitioning:
        let output_partitioning =
            Self::output_partitioning_helper(input, preserve_partitioning);

        Ok((
            PlanProperties::new(
                eq_properties,
                output_partitioning,
                emission_type,
                boundedness,
            ),
            sort_prefix,
        ))
    }
}

impl DisplayAs for SortExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let preserve_partitioning = self.preserve_partitioning;
                match self.fetch {
                    Some(fetch) => {
                        write!(
                            f,
                            "SortExec: TopK(fetch={fetch}), expr=[{}], preserve_partitioning=[{preserve_partitioning}]",
                            self.expr
                        )?;
                        if let Some(filter) = &self.filter
                            && let Ok(current) = filter.read().expr().current()
                            && !current.eq(&lit(true))
                        {
                            write!(f, ", filter=[{current}]")?;
                        }
                        if !self.common_sort_prefix.is_empty() {
                            write!(f, ", sort_prefix=[")?;
                            let mut first = true;
                            for sort_expr in &self.common_sort_prefix {
                                if first {
                                    first = false;
                                } else {
                                    write!(f, ", ")?;
                                }
                                write!(f, "{sort_expr}")?;
                            }
                            write!(f, "]")
                        } else {
                            Ok(())
                        }
                    }
                    None => write!(
                        f,
                        "SortExec: expr=[{}], preserve_partitioning=[{preserve_partitioning}]",
                        self.expr
                    ),
                }
            }
            DisplayFormatType::TreeRender => match self.fetch {
                Some(fetch) => {
                    writeln!(f, "{}", self.expr)?;
                    writeln!(f, "limit={fetch}")
                }
                None => {
                    writeln!(f, "{}", self.expr)
                }
            },
        }
    }
}

impl ExecutionPlan for SortExec {
    fn name(&self) -> &'static str {
        match self.fetch {
            Some(_) => "SortExec(TopK)",
            None => "SortExec",
        }
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        if self.preserve_partitioning {
            vec![Distribution::UnspecifiedDistribution]
        } else {
            // global sort
            // TODO support RangePartition and OrderedDistribution
            vec![Distribution::SinglePartition]
        }
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        // Apply to sort expressions
        let mut tnr = TreeNodeRecursion::Continue;
        for sort_expr in &self.expr {
            tnr = tnr.visit_sibling(|| f(sort_expr.expr.as_ref()))?;
        }

        // Apply to dynamic filter expression if present (when fetch is Some, TopK mode)
        if let Some(filter) = &self.filter {
            let filter_guard = filter.read();
            tnr = tnr.visit_sibling(|| f(filter_guard.expr().as_ref()))?;
        }

        Ok(tnr)
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut new_sort = self.cloned();
        assert_eq!(children.len(), 1, "SortExec should have exactly one child");
        new_sort.input = Arc::clone(&children[0]);

        if !has_same_children_properties(self.as_ref(), &children)? {
            // Recompute the properties based on the new input since they may have changed
            let (cache, sort_prefix) = Self::compute_properties(
                &new_sort.input,
                new_sort.expr.clone(),
                new_sort.preserve_partitioning,
            )?;
            new_sort.cache = Arc::new(cache);
            new_sort.common_sort_prefix = sort_prefix;
        }

        Ok(Arc::new(new_sort))
    }

    fn reset_state(self: Arc<Self>) -> Result<Arc<dyn ExecutionPlan>> {
        let children = self.children().into_iter().cloned().collect();
        let new_sort = self.with_new_children(children)?;
        let mut new_sort = new_sort
            .downcast_ref::<SortExec>()
            .expect("cloned 1 lines above this line, we know the type")
            .clone();
        // Our dynamic filter and execution metrics are the state we need to reset.
        new_sort.filter = Some(new_sort.create_filter());
        new_sort.metrics_set = ExecutionPlanMetricsSet::new();

        Ok(Arc::new(new_sort))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        trace!(
            "Start SortExec::execute for partition {} of context session_id {} and task_id {:?}",
            partition,
            context.session_id(),
            context.task_id()
        );

        let mut input = self.input.execute(partition, Arc::clone(&context))?;

        let execution_options = &context.session_config().options().execution;

        trace!("End SortExec's input.execute for partition: {partition}");

        let sort_satisfied = self
            .input
            .equivalence_properties()
            .ordering_satisfy(self.expr.clone())?;

        match (sort_satisfied, self.fetch.as_ref()) {
            (true, Some(fetch)) => Ok(Box::pin(LimitStream::new(
                input,
                0,
                Some(*fetch),
                BaselineMetrics::new(&self.metrics_set, partition),
            ))),
            (true, None) => Ok(input),
            (false, Some(fetch)) => {
                let filter = self.filter.clone();
                let mut topk = TopK::try_new(
                    partition,
                    input.schema(),
                    self.common_sort_prefix.clone(),
                    self.expr.clone(),
                    *fetch,
                    context.session_config().batch_size(),
                    context.runtime_env(),
                    &self.metrics_set,
                    Arc::clone(&unwrap_or_internal_err!(filter)),
                )?;
                Ok(Box::pin(RecordBatchStreamAdapter::new(
                    self.schema(),
                    futures::stream::once(async move {
                        while let Some(batch) = input.next().await {
                            let batch = batch?;
                            topk.insert_batch(batch)?;
                            if topk.finished {
                                break;
                            }
                        }
                        topk.emit()
                    })
                    .try_flatten(),
                )))
            }
            (false, None) => {
                let mut sorter = ExternalSorter::new(
                    partition,
                    input.schema(),
                    self.expr.clone(),
                    context.session_config().batch_size(),
                    execution_options.sort_spill_reservation_bytes,
                    execution_options.sort_coalesce_target_rows,
                    context.session_config().spill_compression(),
                    &self.metrics_set,
                    context.runtime_env(),
                )?;
                Ok(Box::pin(RecordBatchStreamAdapter::new(
                    self.schema(),
                    futures::stream::once(async move {
                        while let Some(batch) = input.next().await {
                            let batch = batch?;
                            sorter.insert_batch(batch).await?;
                        }
                        sorter.sort().await
                    })
                    .try_flatten(),
                )))
            }
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics_set.clone_inner())
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        let p = if !self.preserve_partitioning() {
            None
        } else {
            partition
        };
        let stats = Arc::unwrap_or_clone(self.input.partition_statistics(p)?);
        Ok(Arc::new(stats.with_fetch(self.fetch, 0, 1)?))
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        Some(Arc::new(SortExec::with_fetch(self, limit)))
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        if self.fetch.is_none() {
            CardinalityEffect::Equal
        } else {
            CardinalityEffect::LowerEqual
        }
    }

    /// Tries to swap the projection with its input [`SortExec`]. If it can be done,
    /// it returns the new swapped version having the [`SortExec`] as the top plan.
    /// Otherwise, it returns None.
    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        // If the projection does not narrow the schema, we should not try to push it down.
        if projection.expr().len() >= projection.input().schema().fields().len() {
            return Ok(None);
        }

        let Some(updated_exprs) = update_ordering(self.expr.clone(), projection.expr())?
        else {
            return Ok(None);
        };

        Ok(Some(Arc::new(
            SortExec::new(updated_exprs, make_with_child(projection, self.input())?)
                .with_fetch(self.fetch())
                .with_preserve_partitioning(self.preserve_partitioning()),
        )))
    }

    fn gather_filters_for_pushdown(
        &self,
        phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        config: &datafusion_common::config::ConfigOptions,
    ) -> Result<FilterDescription> {
        if phase != FilterPushdownPhase::Post {
            if self.fetch.is_some() {
                return Ok(FilterDescription::all_unsupported(
                    &parent_filters,
                    &self.children(),
                ));
            }
            return FilterDescription::from_children(parent_filters, &self.children());
        }

        // In Post phase: block parent filters when fetch is set,
        // but still push the TopK dynamic filter (self-filter).
        let mut child = if self.fetch.is_some() {
            ChildFilterDescription::all_unsupported(&parent_filters)
        } else {
            ChildFilterDescription::from_child(&parent_filters, self.input())?
        };

        if let Some(filter) = &self.filter
            && config.optimizer.enable_topk_dynamic_filter_pushdown
        {
            child = child.with_self_filter(filter.read().expr());
        }

        Ok(FilterDescription::new().with_child(child))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use super::*;
    use crate::coalesce_partitions::CoalescePartitionsExec;
    use crate::collect;
    use crate::empty::EmptyExec;
    use crate::execution_plan::Boundedness;
    use crate::expressions::col;
    use crate::filter_pushdown::{FilterPushdownPhase, PushedDown};
    use crate::test;
    use crate::test::TestMemoryExec;
    use crate::test::exec::{BlockingExec, assert_strong_count_converges_to_zero};
    use crate::test::{assert_is_pending, make_partition};

    use arrow::array::*;
    use arrow::compute::{SortOptions, concat_batches};
    use arrow::datatypes::*;
    use datafusion_common::ScalarValue;
    use datafusion_common::cast::as_primitive_array;
    use datafusion_common::config::ConfigOptions;
    use datafusion_common::test_util::batches_to_string;
    use datafusion_execution::RecordBatchStream;
    use datafusion_execution::config::SessionConfig;
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, MemoryPool,
    };
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_physical_expr::EquivalenceProperties;
    use datafusion_physical_expr::expressions::{Column, Literal};

    use futures::{FutureExt, Stream, TryStreamExt};
    use insta::assert_snapshot;

    #[derive(Debug, Clone)]
    pub struct SortedUnboundedExec {
        schema: Schema,
        batch_size: u64,
        cache: Arc<PlanProperties>,
    }

    impl DisplayAs for SortedUnboundedExec {
        fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
            match t {
                DisplayFormatType::Default
                | DisplayFormatType::Verbose
                | DisplayFormatType::TreeRender => write!(f, "UnboundableExec",).unwrap(),
            }
            Ok(())
        }
    }

    impl SortedUnboundedExec {
        fn compute_properties(schema: SchemaRef) -> PlanProperties {
            let mut eq_properties = EquivalenceProperties::new(schema);
            eq_properties.add_ordering([PhysicalSortExpr::new_default(Arc::new(
                Column::new("c1", 0),
            ))]);
            PlanProperties::new(
                eq_properties,
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Unbounded {
                    requires_infinite_memory: false,
                },
            )
        }
    }

    impl ExecutionPlan for SortedUnboundedExec {
        fn name(&self) -> &'static str {
            Self::static_name()
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            &self.cache
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            _: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn execute(
            &self,
            _partition: usize,
            _context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            Ok(Box::pin(SortedUnboundedStream {
                schema: Arc::new(self.schema.clone()),
                batch_size: self.batch_size,
                offset: 0,
            }))
        }
    }

    #[derive(Debug)]
    pub struct SortedUnboundedStream {
        schema: SchemaRef,
        batch_size: u64,
        offset: u64,
    }

    impl Stream for SortedUnboundedStream {
        type Item = Result<RecordBatch>;

        fn poll_next(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            let batch = SortedUnboundedStream::create_record_batch(
                Arc::clone(&self.schema),
                self.offset,
                self.batch_size,
            );
            self.offset += self.batch_size;
            Poll::Ready(Some(Ok(batch)))
        }
    }

    impl RecordBatchStream for SortedUnboundedStream {
        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
        }
    }

    impl SortedUnboundedStream {
        fn create_record_batch(
            schema: SchemaRef,
            offset: u64,
            batch_size: u64,
        ) -> RecordBatch {
            let values = (0..batch_size).map(|i| offset + i).collect::<Vec<_>>();
            let array = UInt64Array::from(values);
            let array_ref: ArrayRef = Arc::new(array);
            RecordBatch::try_new(schema, vec![array_ref]).unwrap()
        }
    }

    #[tokio::test]
    async fn test_in_mem_sort() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let partitions = 4;
        let csv = test::scan_partitioned(partitions);
        let schema = csv.schema();

        let sort_exec = Arc::new(SortExec::new(
            [PhysicalSortExpr {
                expr: col("i", &schema)?,
                options: SortOptions::default(),
            }]
            .into(),
            Arc::new(CoalescePartitionsExec::new(csv)),
        ));

        let result = collect(sort_exec, Arc::clone(&task_ctx)).await?;

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_rows(), 400);
        assert_eq!(
            task_ctx.runtime_env().memory_pool.reserved(),
            0,
            "The sort should have returned all memory used back to the memory manager"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_sort_spill() -> Result<()> {
        // trigger spill w/ 100 batches
        let session_config = SessionConfig::new();
        let sort_spill_reservation_bytes = session_config
            .options()
            .execution
            .sort_spill_reservation_bytes;
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(sort_spill_reservation_bytes + 12288, 1.0)
            .build_arc()?;
        let task_ctx = Arc::new(
            TaskContext::default()
                .with_session_config(session_config)
                .with_runtime(runtime),
        );

        // The input has 100 partitions, each partition has a batch containing 100 rows.
        // Each row has a single Int32 column with values 0..100. The total size of the
        // input is roughly 40000 bytes.
        let partitions = 100;
        let input = test::scan_partitioned(partitions);
        let schema = input.schema();

        let sort_exec = Arc::new(SortExec::new(
            [PhysicalSortExpr {
                expr: col("i", &schema)?,
                options: SortOptions::default(),
            }]
            .into(),
            Arc::new(CoalescePartitionsExec::new(input)),
        ));

        let result = collect(
            Arc::clone(&sort_exec) as Arc<dyn ExecutionPlan>,
            Arc::clone(&task_ctx),
        )
        .await?;

        assert_eq!(result.len(), 2);

        // Now, validate metrics
        let metrics = sort_exec.metrics().unwrap();

        assert_eq!(metrics.output_rows().unwrap(), 10000);
        assert!(metrics.elapsed_compute().unwrap() > 0);

        let spill_count = metrics.spill_count().unwrap();
        let spilled_rows = metrics.spilled_rows().unwrap();
        let spilled_bytes = metrics.spilled_bytes().unwrap();
        // Processing 40000 bytes of data using 12288 bytes of memory requires 3 spills
        // unless we do something really clever. It will spill roughly 9000+ rows and 36000
        // bytes. We leave a little wiggle room for the actual numbers.
        assert!((3..=10).contains(&spill_count));
        assert!((9000..=10000).contains(&spilled_rows));
        assert!((38000..=44000).contains(&spilled_bytes));

        let columns = result[0].columns();

        let i = as_primitive_array::<Int32Type>(&columns[0])?;
        assert_eq!(i.value(0), 0);
        assert_eq!(i.value(i.len() - 1), 81);
        assert_eq!(
            task_ctx.runtime_env().memory_pool.reserved(),
            0,
            "The sort should have returned all memory used back to the memory manager"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_reservation_error() -> Result<()> {
        // Pick a memory limit and sort_spill_reservation that make the first batch reservation fail.
        let merge_reservation: usize = 0; // Set to 0 for simplicity

        let session_config =
            SessionConfig::new().with_sort_spill_reservation_bytes(merge_reservation);

        let plan = test::scan_partitioned(1);

        // Read the first record batch to determine the actual memory requirement
        let expected_batch_reservation = {
            let temp_ctx = Arc::new(TaskContext::default());
            let mut stream = plan.execute(0, Arc::clone(&temp_ctx))?;
            let first_batch = stream.next().await.unwrap()?;
            get_reserved_bytes_for_record_batch(&first_batch)?
        };

        // Set memory limit just short of what we need
        let memory_limit: usize = expected_batch_reservation + merge_reservation - 1;

        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(memory_limit, 1.0)
            .build_arc()?;
        let task_ctx = Arc::new(
            TaskContext::default()
                .with_session_config(session_config)
                .with_runtime(runtime),
        );

        // Verify that our memory limit is insufficient
        {
            let mut stream = plan.execute(0, Arc::clone(&task_ctx))?;
            let first_batch = stream.next().await.unwrap()?;
            let batch_reservation = get_reserved_bytes_for_record_batch(&first_batch)?;

            assert_eq!(batch_reservation, expected_batch_reservation);
            assert!(memory_limit < (merge_reservation + batch_reservation));
        }

        let sort_exec = Arc::new(SortExec::new(
            [PhysicalSortExpr::new_default(col("i", &plan.schema())?)].into(),
            plan,
        ));

        let result = collect(Arc::clone(&sort_exec) as _, Arc::clone(&task_ctx)).await;

        let err = result.unwrap_err();
        assert!(
            matches!(err, DataFusionError::Context(..)),
            "Assertion failed: expected a Context error, but got: {err:?}"
        );

        // Assert that the context error is wrapping a resources exhausted error.
        assert!(
            matches!(err.find_root(), DataFusionError::ResourcesExhausted(_)),
            "Assertion failed: expected a ResourcesExhausted error, but got: {err:?}"
        );

        // Verify external sorter error message when resource is exhausted
        let config_vector = vec![
            "datafusion.runtime.memory_limit",
            "datafusion.execution.sort_spill_reservation_bytes",
        ];
        let error_message = err.message().to_string();
        for config in config_vector.into_iter() {
            assert!(
                error_message.as_str().contains(config),
                "Config: '{}' should be contained in error message: {}.",
                config,
                error_message.as_str()
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_sort_spill_utf8_strings() -> Result<()> {
        let session_config = SessionConfig::new()
            .with_batch_size(100)
            .with_sort_spill_reservation_bytes(100 * 1024);
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(500 * 1024, 1.0)
            .build_arc()?;
        let task_ctx = Arc::new(
            TaskContext::default()
                .with_session_config(session_config)
                .with_runtime(runtime),
        );

        // The input has 200 partitions, each partition has a batch containing 100 rows.
        // Each row has a single Utf8 column, the Utf8 string values are roughly 42 bytes.
        // The total size of the input is roughly 820 KB.
        let input = test::scan_partitioned_utf8(200);
        let schema = input.schema();

        let sort_exec = Arc::new(SortExec::new(
            [PhysicalSortExpr {
                expr: col("i", &schema)?,
                options: SortOptions::default(),
            }]
            .into(),
            Arc::new(CoalescePartitionsExec::new(input)),
        ));

        let result = collect(Arc::clone(&sort_exec) as _, Arc::clone(&task_ctx)).await?;

        let num_rows = result.iter().map(|batch| batch.num_rows()).sum::<usize>();
        assert_eq!(num_rows, 20000);

        // Now, validate metrics
        let metrics = sort_exec.metrics().unwrap();

        assert_eq!(metrics.output_rows().unwrap(), 20000);
        assert!(metrics.elapsed_compute().unwrap() > 0);

        let spill_count = metrics.spill_count().unwrap();
        let spilled_rows = metrics.spilled_rows().unwrap();
        let spilled_bytes = metrics.spilled_bytes().unwrap();

        // This test case is processing 840KB of data using 400KB of memory. Note
        // that buffered batches can't be dropped until all sorted batches are
        // generated, so we can only buffer `sort_spill_reservation_bytes` of sorted
        // batches.
        // The number of spills is roughly calculated as:
        //  `number_of_batches / (sort_spill_reservation_bytes / batch_size)`

        // If this assertion fail with large spill count, make sure the following
        // case does not happen:
        // During external sorting, one sorted run should be spilled to disk in a
        // single file, due to memory limit we might need to append to the file
        // multiple times to spill all the data. Make sure we're not writing each
        // appending as a separate file.
        assert!((4..=8).contains(&spill_count));
        assert!((15000..=20000).contains(&spilled_rows));
        assert!((900000..=1000000).contains(&spilled_bytes));

        // Verify that the result is sorted
        let concated_result = concat_batches(&schema, &result)?;
        let columns = concated_result.columns();
        let string_array = as_string_array(&columns[0]);
        for i in 0..string_array.len() - 1 {
            assert!(string_array.value(i) <= string_array.value(i + 1));
        }

        assert_eq!(
            task_ctx.runtime_env().memory_pool.reserved(),
            0,
            "The sort should have returned all memory used back to the memory manager"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_sort_fetch_memory_calculation() -> Result<()> {
        // This test mirrors down the size from the example above.
        let avg_batch_size = 400;
        let partitions = 4;

        // A tuple of (fetch, expect_spillage)
        let test_options = vec![
            // Since we don't have a limit (and the memory is less than the total size of
            // all the batches we are processing, we expect it to spill.
            (None, true),
            // When we have a limit however, the buffered size of batches should fit in memory
            // since it is much lower than the total size of the input batch.
            (Some(1), false),
        ];

        for (fetch, expect_spillage) in test_options {
            let session_config = SessionConfig::new();
            let sort_spill_reservation_bytes = session_config
                .options()
                .execution
                .sort_spill_reservation_bytes;

            let runtime = RuntimeEnvBuilder::new()
                .with_memory_limit(
                    sort_spill_reservation_bytes + avg_batch_size * (partitions - 1),
                    1.0,
                )
                .build_arc()?;
            let task_ctx = Arc::new(
                TaskContext::default()
                    .with_runtime(runtime)
                    .with_session_config(session_config),
            );

            let csv = test::scan_partitioned(partitions);
            let schema = csv.schema();

            let sort_exec = Arc::new(
                SortExec::new(
                    [PhysicalSortExpr {
                        expr: col("i", &schema)?,
                        options: SortOptions::default(),
                    }]
                    .into(),
                    Arc::new(CoalescePartitionsExec::new(csv)),
                )
                .with_fetch(fetch),
            );

            let result =
                collect(Arc::clone(&sort_exec) as _, Arc::clone(&task_ctx)).await?;
            assert_eq!(result.len(), 1);

            let metrics = sort_exec.metrics().unwrap();
            let did_it_spill = metrics.spill_count().unwrap_or(0) > 0;
            assert_eq!(did_it_spill, expect_spillage, "with fetch: {fetch:?}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_sort_memory_reduction_per_batch() -> Result<()> {
        // This test verifies that memory reservation is reduced for every batch emitted
        // during the sort process. This is important to ensure we don't hold onto
        // memory longer than necessary.

        // Create a large enough batch that will be split into multiple output batches
        let batch_size = 50; // Small batch size to force multiple output batches
        let num_rows = 1000; // Create enough data for multiple batches

        let task_ctx = Arc::new(TaskContext::default().with_session_config(
            SessionConfig::new().with_batch_size(batch_size), // Ensure we don't concat batches
        ));

        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        // Create unsorted data
        let mut values: Vec<i32> = (0..num_rows).collect();
        values.reverse();

        let input_batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(values))],
        )?;

        let batches = vec![input_batch];

        let sort_exec = Arc::new(SortExec::new(
            [PhysicalSortExpr {
                expr: Arc::new(Column::new("a", 0)),
                options: SortOptions::default(),
            }]
            .into(),
            TestMemoryExec::try_new_exec(
                std::slice::from_ref(&batches),
                Arc::clone(&schema),
                None,
            )?,
        ));

        let mut stream = sort_exec.execute(0, Arc::clone(&task_ctx))?;

        let mut previous_reserved = task_ctx.runtime_env().memory_pool.reserved();
        let mut batch_count = 0;

        // Collect batches and verify memory is reduced with each batch
        while let Some(result) = stream.next().await {
            let batch = result?;
            batch_count += 1;

            // Verify we got a non-empty batch
            assert!(batch.num_rows() > 0, "Batch should not be empty");

            let current_reserved = task_ctx.runtime_env().memory_pool.reserved();

            // After the first batch, memory should be reducing or staying the same
            // (it should not increase as we emit batches)
            if batch_count > 1 {
                assert!(
                    current_reserved <= previous_reserved,
                    "Memory reservation should decrease or stay same as batches are emitted. \
                     Batch {batch_count}: previous={previous_reserved}, current={current_reserved}"
                );
            }

            previous_reserved = current_reserved;
        }

        assert!(
            batch_count > 1,
            "Expected multiple batches to be emitted, got {batch_count}"
        );

        // Verify all memory is returned at the end
        assert_eq!(
            task_ctx.runtime_env().memory_pool.reserved(),
            0,
            "All memory should be returned after consuming all batches"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_sort_metadata() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let field_metadata: HashMap<String, String> =
            vec![("foo".to_string(), "bar".to_string())]
                .into_iter()
                .collect();
        let schema_metadata: HashMap<String, String> =
            vec![("baz".to_string(), "barf".to_string())]
                .into_iter()
                .collect();

        let mut field = Field::new("field_name", DataType::UInt64, true);
        field.set_metadata(field_metadata.clone());
        let schema = Schema::new_with_metadata(vec![field], schema_metadata.clone());
        let schema = Arc::new(schema);

        let data: ArrayRef =
            Arc::new(vec![3, 2, 1].into_iter().map(Some).collect::<UInt64Array>());

        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![data])?;
        let input =
            TestMemoryExec::try_new_exec(&[vec![batch]], Arc::clone(&schema), None)?;

        let sort_exec = Arc::new(SortExec::new(
            [PhysicalSortExpr {
                expr: col("field_name", &schema)?,
                options: SortOptions::default(),
            }]
            .into(),
            input,
        ));

        let result: Vec<RecordBatch> = collect(sort_exec, task_ctx).await?;

        let expected_data: ArrayRef =
            Arc::new(vec![1, 2, 3].into_iter().map(Some).collect::<UInt64Array>());
        let expected_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![expected_data])?;

        // Data is correct
        assert_eq!(&vec![expected_batch], &result);

        // explicitly ensure the metadata is present
        assert_eq!(result[0].schema().fields()[0].metadata(), &field_metadata);
        assert_eq!(result[0].schema().metadata(), &schema_metadata);

        Ok(())
    }

    #[tokio::test]
    async fn test_lex_sort_by_mixed_types() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new(
                "b",
                DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
                true,
            ),
        ]));

        // define data.
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![Some(2), None, Some(1), Some(2)])),
                Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                    Some(vec![Some(3)]),
                    Some(vec![Some(1)]),
                    Some(vec![Some(6), None]),
                    Some(vec![Some(5)]),
                ])),
            ],
        )?;

        let sort_exec = Arc::new(SortExec::new(
            [
                PhysicalSortExpr {
                    expr: col("a", &schema)?,
                    options: SortOptions {
                        descending: false,
                        nulls_first: true,
                    },
                },
                PhysicalSortExpr {
                    expr: col("b", &schema)?,
                    options: SortOptions {
                        descending: true,
                        nulls_first: false,
                    },
                },
            ]
            .into(),
            TestMemoryExec::try_new_exec(&[vec![batch]], Arc::clone(&schema), None)?,
        ));

        assert_eq!(DataType::Int32, *sort_exec.schema().field(0).data_type());
        assert_eq!(
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
            *sort_exec.schema().field(1).data_type()
        );

        let result: Vec<RecordBatch> =
            collect(Arc::clone(&sort_exec) as Arc<dyn ExecutionPlan>, task_ctx).await?;
        let metrics = sort_exec.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);
        assert_eq!(metrics.output_rows().unwrap(), 4);
        assert_eq!(result.len(), 1);

        let expected = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![None, Some(1), Some(2), Some(2)])),
                Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                    Some(vec![Some(1)]),
                    Some(vec![Some(6), None]),
                    Some(vec![Some(5)]),
                    Some(vec![Some(3)]),
                ])),
            ],
        )?;

        assert_eq!(expected, result[0]);

        Ok(())
    }

    #[tokio::test]
    async fn test_lex_sort_by_float() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float32, true),
            Field::new("b", DataType::Float64, true),
        ]));

        // define data.
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Float32Array::from(vec![
                    Some(f32::NAN),
                    None,
                    None,
                    Some(f32::NAN),
                    Some(1.0_f32),
                    Some(1.0_f32),
                    Some(2.0_f32),
                    Some(3.0_f32),
                ])),
                Arc::new(Float64Array::from(vec![
                    Some(200.0_f64),
                    Some(20.0_f64),
                    Some(10.0_f64),
                    Some(100.0_f64),
                    Some(f64::NAN),
                    None,
                    None,
                    Some(f64::NAN),
                ])),
            ],
        )?;

        let sort_exec = Arc::new(SortExec::new(
            [
                PhysicalSortExpr {
                    expr: col("a", &schema)?,
                    options: SortOptions {
                        descending: true,
                        nulls_first: true,
                    },
                },
                PhysicalSortExpr {
                    expr: col("b", &schema)?,
                    options: SortOptions {
                        descending: false,
                        nulls_first: false,
                    },
                },
            ]
            .into(),
            TestMemoryExec::try_new_exec(&[vec![batch]], schema, None)?,
        ));

        assert_eq!(DataType::Float32, *sort_exec.schema().field(0).data_type());
        assert_eq!(DataType::Float64, *sort_exec.schema().field(1).data_type());

        let result: Vec<RecordBatch> =
            collect(Arc::clone(&sort_exec) as Arc<dyn ExecutionPlan>, task_ctx).await?;
        let metrics = sort_exec.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);
        assert_eq!(metrics.output_rows().unwrap(), 8);
        assert_eq!(result.len(), 1);

        let columns = result[0].columns();

        assert_eq!(DataType::Float32, *columns[0].data_type());
        assert_eq!(DataType::Float64, *columns[1].data_type());

        let a = as_primitive_array::<Float32Type>(&columns[0])?;
        let b = as_primitive_array::<Float64Type>(&columns[1])?;

        // convert result to strings to allow comparing to expected result containing NaN
        let result: Vec<(Option<String>, Option<String>)> = (0..result[0].num_rows())
            .map(|i| {
                let aval = if a.is_valid(i) {
                    Some(a.value(i).to_string())
                } else {
                    None
                };
                let bval = if b.is_valid(i) {
                    Some(b.value(i).to_string())
                } else {
                    None
                };
                (aval, bval)
            })
            .collect();

        let expected: Vec<(Option<String>, Option<String>)> = vec![
            (None, Some("10".to_owned())),
            (None, Some("20".to_owned())),
            (Some("NaN".to_owned()), Some("100".to_owned())),
            (Some("NaN".to_owned()), Some("200".to_owned())),
            (Some("3".to_owned()), Some("NaN".to_owned())),
            (Some("2".to_owned()), None),
            (Some("1".to_owned()), Some("NaN".to_owned())),
            (Some("1".to_owned()), None),
        ];

        assert_eq!(expected, result);

        Ok(())
    }

    #[tokio::test]
    async fn test_drop_cancel() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, true)]));

        let blocking_exec = Arc::new(BlockingExec::new(Arc::clone(&schema), 1));
        let refs = blocking_exec.refs();
        let sort_exec = Arc::new(SortExec::new(
            [PhysicalSortExpr {
                expr: col("a", &schema)?,
                options: SortOptions::default(),
            }]
            .into(),
            blocking_exec,
        ));

        let fut = collect(sort_exec, Arc::clone(&task_ctx));
        let mut fut = fut.boxed();

        assert_is_pending(&mut fut);
        drop(fut);
        assert_strong_count_converges_to_zero(refs).await;

        assert_eq!(
            task_ctx.runtime_env().memory_pool.reserved(),
            0,
            "The sort should have returned all memory used back to the memory manager"
        );

        Ok(())
    }

    #[test]
    fn test_empty_sort_batch() {
        let schema = Arc::new(Schema::empty());
        let options = RecordBatchOptions::new().with_row_count(Some(1));
        let batch =
            RecordBatch::try_new_with_options(Arc::clone(&schema), vec![], &options)
                .unwrap();

        let expressions = [PhysicalSortExpr {
            expr: Arc::new(Literal::new(ScalarValue::Int64(Some(1)))),
            options: SortOptions::default(),
        }]
        .into();

        let result = sort_batch(&batch, &expressions, None).unwrap();
        assert_eq!(result.num_rows(), 1);
    }

    #[tokio::test]
    async fn topk_unbounded_source() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let schema = Schema::new(vec![Field::new("c1", DataType::UInt64, false)]);
        let source = SortedUnboundedExec {
            schema: schema.clone(),
            batch_size: 2,
            cache: Arc::new(SortedUnboundedExec::compute_properties(Arc::new(
                schema.clone(),
            ))),
        };
        let mut plan = SortExec::new(
            [PhysicalSortExpr::new_default(Arc::new(Column::new(
                "c1", 0,
            )))]
            .into(),
            Arc::new(source),
        );
        plan = plan.with_fetch(Some(9));

        let batches = collect(Arc::new(plan), task_ctx).await?;
        assert_snapshot!(batches_to_string(&batches), @r"
        +----+
        | c1 |
        +----+
        | 0  |
        | 1  |
        | 2  |
        | 3  |
        | 4  |
        | 5  |
        | 6  |
        | 7  |
        | 8  |
        +----+
        ");
        Ok(())
    }

    #[tokio::test]
    async fn should_return_stream_with_batches_in_the_requested_size() -> Result<()> {
        let batch_size = 100;

        let create_task_ctx = |_: &[RecordBatch]| {
            TaskContext::default()
                .with_session_config(SessionConfig::new().with_batch_size(batch_size))
        };

        // Smaller than batch size and require more than a single batch to get the requested batch size
        test_sort_output_batch_size(10, batch_size / 4, create_task_ctx).await?;

        // Not evenly divisible by batch size
        test_sort_output_batch_size(10, batch_size + 7, create_task_ctx).await?;

        // Evenly divisible by batch size and is larger than 2 output batches
        test_sort_output_batch_size(10, batch_size * 3, create_task_ctx).await?;

        Ok(())
    }

    #[tokio::test]
    async fn should_return_stream_with_batches_in_the_requested_size_when_sorting_in_place()
    -> Result<()> {
        let batch_size = 100;

        let create_task_ctx = |_: &[RecordBatch]| {
            TaskContext::default()
                .with_session_config(SessionConfig::new().with_batch_size(batch_size))
        };

        // Smaller than batch size and require more than a single batch to get the requested batch size
        {
            let metrics =
                test_sort_output_batch_size(10, batch_size / 4, create_task_ctx).await?;

            assert_eq!(
                metrics.spill_count(),
                Some(0),
                "Expected no spills when sorting in place"
            );
        }

        // Not evenly divisible by batch size
        {
            let metrics =
                test_sort_output_batch_size(10, batch_size + 7, create_task_ctx).await?;

            assert_eq!(
                metrics.spill_count(),
                Some(0),
                "Expected no spills when sorting in place"
            );
        }

        // Evenly divisible by batch size and is larger than 2 output batches
        {
            let metrics =
                test_sort_output_batch_size(10, batch_size * 3, create_task_ctx).await?;

            assert_eq!(
                metrics.spill_count(),
                Some(0),
                "Expected no spills when sorting in place"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn should_return_stream_with_batches_in_the_requested_size_when_having_a_single_batch()
    -> Result<()> {
        let batch_size = 100;

        let create_task_ctx = |_: &[RecordBatch]| {
            TaskContext::default()
                .with_session_config(SessionConfig::new().with_batch_size(batch_size))
        };

        // Smaller than batch size and require more than a single batch to get the requested batch size
        {
            let metrics = test_sort_output_batch_size(
                // Single batch
                1,
                batch_size / 4,
                create_task_ctx,
            )
            .await?;

            assert_eq!(
                metrics.spill_count(),
                Some(0),
                "Expected no spills when sorting in place"
            );
        }

        // Not evenly divisible by batch size
        {
            let metrics = test_sort_output_batch_size(
                // Single batch
                1,
                batch_size + 7,
                create_task_ctx,
            )
            .await?;

            assert_eq!(
                metrics.spill_count(),
                Some(0),
                "Expected no spills when sorting in place"
            );
        }

        // Evenly divisible by batch size and is larger than 2 output batches
        {
            let metrics = test_sort_output_batch_size(
                // Single batch
                1,
                batch_size * 3,
                create_task_ctx,
            )
            .await?;

            assert_eq!(
                metrics.spill_count(),
                Some(0),
                "Expected no spills when sorting in place"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn should_return_stream_with_batches_in_the_requested_size_when_having_to_spill()
    -> Result<()> {
        let batch_size = 100;

        let create_task_ctx = |generated_batches: &[RecordBatch]| {
            let batches_memory = generated_batches
                .iter()
                .map(|b| b.get_array_memory_size())
                .sum::<usize>();

            TaskContext::default()
                .with_session_config(
                    SessionConfig::new()
                        .with_batch_size(batch_size)
                        .with_sort_spill_reservation_bytes(1),
                )
                .with_runtime(
                    RuntimeEnvBuilder::default()
                        .with_memory_limit(batches_memory, 1.0)
                        .build_arc()
                        .unwrap(),
                )
        };

        // Smaller than batch size and require more than a single batch to get the requested batch size
        {
            let metrics =
                test_sort_output_batch_size(10, batch_size / 4, create_task_ctx).await?;

            assert_ne!(metrics.spill_count().unwrap(), 0, "expected to spill");
        }

        // Not evenly divisible by batch size
        {
            let metrics =
                test_sort_output_batch_size(10, batch_size + 7, create_task_ctx).await?;

            assert_ne!(metrics.spill_count().unwrap(), 0, "expected to spill");
        }

        // Evenly divisible by batch size and is larger than 2 batches
        {
            let metrics =
                test_sort_output_batch_size(10, batch_size * 3, create_task_ctx).await?;

            assert_ne!(metrics.spill_count().unwrap(), 0, "expected to spill");
        }

        Ok(())
    }

    async fn test_sort_output_batch_size(
        number_of_batches: usize,
        batch_size_to_generate: usize,
        create_task_ctx: impl Fn(&[RecordBatch]) -> TaskContext,
    ) -> Result<MetricsSet> {
        let batches = (0..number_of_batches)
            .map(|_| make_partition(batch_size_to_generate as i32))
            .collect::<Vec<_>>();
        let task_ctx = create_task_ctx(batches.as_slice());

        let expected_batch_size = task_ctx.session_config().batch_size();

        let (mut output_batches, metrics) =
            run_sort_on_input(task_ctx, "i", batches).await?;

        let last_batch = output_batches.pop().unwrap();

        for batch in output_batches {
            assert_eq!(batch.num_rows(), expected_batch_size);
        }

        let mut last_expected_batch_size =
            (batch_size_to_generate * number_of_batches) % expected_batch_size;
        if last_expected_batch_size == 0 {
            last_expected_batch_size = expected_batch_size;
        }
        assert_eq!(last_batch.num_rows(), last_expected_batch_size);

        Ok(metrics)
    }

    async fn run_sort_on_input(
        task_ctx: TaskContext,
        order_by_col: &str,
        batches: Vec<RecordBatch>,
    ) -> Result<(Vec<RecordBatch>, MetricsSet)> {
        let task_ctx = Arc::new(task_ctx);

        // let task_ctx = env.
        let schema = batches[0].schema();
        let ordering: LexOrdering = [PhysicalSortExpr {
            expr: col(order_by_col, &schema)?,
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        }]
        .into();
        let sort_exec: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(
            ordering.clone(),
            TestMemoryExec::try_new_exec(std::slice::from_ref(&batches), schema, None)?,
        ));

        let sorted_batches =
            collect(Arc::clone(&sort_exec), Arc::clone(&task_ctx)).await?;

        let metrics = sort_exec.metrics().expect("sort have metrics");

        // assert output
        {
            let input_batches_concat = concat_batches(batches[0].schema_ref(), &batches)?;
            let sorted_input_batch = sort_batch(&input_batches_concat, &ordering, None)?;

            let sorted_batches_concat =
                concat_batches(sorted_batches[0].schema_ref(), &sorted_batches)?;

            assert_eq!(sorted_input_batch, sorted_batches_concat);
        }

        Ok((sorted_batches, metrics))
    }

    #[tokio::test]
    async fn test_sort_batch_chunked_basic() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        // Create a batch with 1000 rows
        let mut values: Vec<i32> = (0..1000).collect();
        // Shuffle to make it unsorted
        values.reverse();

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(values))],
        )?;

        let expressions: LexOrdering =
            [PhysicalSortExpr::new_default(Arc::new(Column::new("a", 0)))].into();

        // Sort with batch_size = 250
        let result_batches = sort_batch_chunked(&batch, &expressions, 250, false)?;

        // Verify 4 batches are returned
        assert_eq!(result_batches.len(), 4);

        // Verify each batch has <= 250 rows
        let mut total_rows = 0;
        for (i, batch) in result_batches.iter().enumerate() {
            assert!(
                batch.num_rows() <= 250,
                "Batch {} has {} rows, expected <= 250",
                i,
                batch.num_rows()
            );
            total_rows += batch.num_rows();
        }

        // Verify total row count matches input
        assert_eq!(total_rows, 1000);

        // Verify data is correctly sorted across all chunks
        let concatenated = concat_batches(&schema, &result_batches)?;
        let array = as_primitive_array::<Int32Type>(concatenated.column(0))?;
        for i in 0..array.len() - 1 {
            assert!(
                array.value(i) <= array.value(i + 1),
                "Array not sorted at position {}: {} > {}",
                i,
                array.value(i),
                array.value(i + 1)
            );
        }
        assert_eq!(array.value(0), 0);
        assert_eq!(array.value(array.len() - 1), 999);

        Ok(())
    }

    #[tokio::test]
    async fn test_sort_batch_chunked_smaller_than_batch_size() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        // Create a batch with 50 rows
        let values: Vec<i32> = (0..50).rev().collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(values))],
        )?;

        let expressions: LexOrdering =
            [PhysicalSortExpr::new_default(Arc::new(Column::new("a", 0)))].into();

        // Sort with batch_size = 100
        let result_batches = sort_batch_chunked(&batch, &expressions, 100, false)?;

        // Should return exactly 1 batch
        assert_eq!(result_batches.len(), 1);
        assert_eq!(result_batches[0].num_rows(), 50);

        // Verify it's correctly sorted
        let array = as_primitive_array::<Int32Type>(result_batches[0].column(0))?;
        for i in 0..array.len() - 1 {
            assert!(array.value(i) <= array.value(i + 1));
        }
        assert_eq!(array.value(0), 0);
        assert_eq!(array.value(49), 49);

        Ok(())
    }

    #[tokio::test]
    async fn test_sort_batch_chunked_exact_multiple() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        // Create a batch with 1000 rows
        let values: Vec<i32> = (0..1000).rev().collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(values))],
        )?;

        let expressions: LexOrdering =
            [PhysicalSortExpr::new_default(Arc::new(Column::new("a", 0)))].into();

        // Sort with batch_size = 100
        let result_batches = sort_batch_chunked(&batch, &expressions, 100, false)?;

        // Should return exactly 10 batches of 100 rows each
        assert_eq!(result_batches.len(), 10);
        for batch in &result_batches {
            assert_eq!(batch.num_rows(), 100);
        }

        // Verify sorted correctly across all batches
        let concatenated = concat_batches(&schema, &result_batches)?;
        let array = as_primitive_array::<Int32Type>(concatenated.column(0))?;
        for i in 0..array.len() - 1 {
            assert!(array.value(i) <= array.value(i + 1));
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_sort_batch_chunked_empty_batch() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        let batch = RecordBatch::new_empty(Arc::clone(&schema));

        let expressions: LexOrdering =
            [PhysicalSortExpr::new_default(Arc::new(Column::new("a", 0)))].into();

        let result_batches = sort_batch_chunked(&batch, &expressions, 100, false)?;

        // Empty input produces no output batches (0 chunks)
        assert_eq!(result_batches.len(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_get_reserved_bytes_for_record_batch_with_sliced_batches() -> Result<()>
    {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        // Create a larger batch then slice it
        let large_array = Int32Array::from((0..1000).collect::<Vec<i32>>());
        let sliced_array = large_array.slice(100, 50); // Take 50 elements starting at 100

        let sliced_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(sliced_array)])?;
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(large_array)])?;

        let sliced_reserved = get_reserved_bytes_for_record_batch(&sliced_batch)?;
        let reserved = get_reserved_bytes_for_record_batch(&batch)?;

        // The reserved memory for the sliced batch should be less than that of the full batch
        assert!(reserved > sliced_reserved);

        Ok(())
    }

    /// Verifies that `ExternalSorter::sort()` transfers the pre-reserved
    /// merge bytes to the merge stream via `take()`, rather than leaving
    /// them in the sorter (via `new_empty()`).
    ///
    /// 1. Create a sorter with a tight memory pool and insert enough data
    ///    to force spilling
    /// 2. Verify `merge_reservation` holds the pre-reserved bytes before sort
    /// 3. Call `sort()` to get the merge stream
    /// 4. Verify `merge_reservation` is now 0 (bytes transferred to merge stream)
    /// 5. Simulate contention: a competing consumer grabs all available pool memory
    /// 6. Verify the merge stream still works (it uses its pre-reserved bytes
    ///    as initial budget, not requesting from pool starting at 0)
    ///
    /// With `new_empty()` (before fix), step 4 fails: `merge_reservation`
    /// still holds the bytes, the merge stream starts with 0 budget, and
    /// those bytes become unaccounted-for reserved memory that nobody uses.
    #[tokio::test]
    async fn test_sort_merge_reservation_transferred_not_freed() -> Result<()> {
        let sort_spill_reservation_bytes: usize = 10 * 1024; // 10 KB

        // Pool: merge reservation (10KB) + enough room for sort to work.
        // The room must accommodate batch data accumulation before spilling.
        let sort_working_memory: usize = 40 * 1024; // 40 KB for sort operations
        let pool_size = sort_spill_reservation_bytes + sort_working_memory;
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(pool_size));

        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::clone(&pool))
            .build_arc()?;

        let metrics_set = ExecutionPlanMetricsSet::new();
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));

        let mut sorter = ExternalSorter::new(
            0,
            Arc::clone(&schema),
            [PhysicalSortExpr::new_default(Arc::new(Column::new("x", 0)))].into(),
            128, // batch_size
            sort_spill_reservation_bytes,
            32768, // sort_coalesce_target_rows
            SpillCompression::Uncompressed,
            &metrics_set,
            Arc::clone(&runtime),
        )?;

        // Insert enough data to force spilling.
        let num_batches = 200;
        for i in 0..num_batches {
            let values: Vec<i32> = ((i * 100)..((i + 1) * 100)).rev().collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from(values))],
            )?;
            sorter.insert_batch(batch).await?;
        }

        assert!(
            sorter.spilled_before(),
            "Test requires spilling to exercise the merge path"
        );

        // Before sort(), merge_reservation holds sort_spill_reservation_bytes.
        assert!(
            sorter.merge_reservation_size() >= sort_spill_reservation_bytes,
            "merge_reservation should hold the pre-reserved bytes before sort()"
        );

        // Call sort() to get the merge stream. With the fix (take()),
        // the pre-reserved merge bytes are transferred to the merge
        // stream. Without the fix (free() + new_empty()), the bytes
        // are released back to the pool and the merge stream starts
        // with 0 bytes.
        let merge_stream = sorter.sort().await?;

        // THE KEY ASSERTION: after sort(), merge_reservation must be 0.
        // This proves take() transferred the bytes to the merge stream,
        // rather than them being freed back to the pool where other
        // partitions could steal them.
        assert_eq!(
            sorter.merge_reservation_size(),
            0,
            "After sort(), merge_reservation should be 0 (bytes transferred \
             to merge stream via take()). If non-zero, the bytes are still \
             held by the sorter and will be freed on drop, allowing other \
             partitions to steal them."
        );

        // Drop the sorter to free its reservations back to the pool.
        drop(sorter);

        // Simulate contention: another partition grabs ALL available
        // pool memory. If the merge stream didn't receive the
        // pre-reserved bytes via take(), it will fail when it tries
        // to allocate memory for reading spill files.
        let contender = MemoryConsumer::new("CompetingPartition").register(&pool);
        let available = pool_size.saturating_sub(pool.reserved());
        if available > 0 {
            contender.try_grow(available).unwrap();
        }

        // The merge stream must still produce correct results despite
        // the pool being fully consumed by the contender. This only
        // works if sort() transferred the pre-reserved bytes to the
        // merge stream (via take()) rather than freeing them.
        let batches: Vec<RecordBatch> = merge_stream.try_collect().await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            total_rows,
            (num_batches * 100) as usize,
            "Merge stream should produce all rows even under memory contention"
        );

        // Verify data is sorted
        let merged = concat_batches(&schema, &batches)?;
        let col = merged.column(0).as_primitive::<Int32Type>();
        for i in 1..col.len() {
            assert!(
                col.value(i - 1) <= col.value(i),
                "Output should be sorted, but found {} > {} at index {}",
                col.value(i - 1),
                col.value(i),
                i
            );
        }

        drop(contender);
        Ok(())
    }

    fn make_sort_exec_with_fetch(fetch: Option<usize>) -> SortExec {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let input = Arc::new(EmptyExec::new(schema));
        SortExec::new(
            [PhysicalSortExpr::new_default(Arc::new(Column::new("a", 0)))].into(),
            input,
        )
        .with_fetch(fetch)
    }

    #[test]
    fn test_sort_with_fetch_blocks_filter_pushdown() -> Result<()> {
        let sort = make_sort_exec_with_fetch(Some(10));
        let desc = sort.gather_filters_for_pushdown(
            FilterPushdownPhase::Pre,
            vec![Arc::new(Column::new("a", 0))],
            &ConfigOptions::new(),
        )?;
        // Sort with fetch (TopK) must not allow filters to be pushed below it.
        assert!(matches!(
            desc.parent_filters()[0][0].discriminant,
            PushedDown::No
        ));
        Ok(())
    }

    #[test]
    fn test_sort_without_fetch_allows_filter_pushdown() -> Result<()> {
        let sort = make_sort_exec_with_fetch(None);
        let desc = sort.gather_filters_for_pushdown(
            FilterPushdownPhase::Pre,
            vec![Arc::new(Column::new("a", 0))],
            &ConfigOptions::new(),
        )?;
        // Plain sort (no fetch) is filter-commutative.
        assert!(matches!(
            desc.parent_filters()[0][0].discriminant,
            PushedDown::Yes
        ));
        Ok(())
    }

    #[test]
    fn test_sort_with_fetch_allows_topk_self_filter_in_post_phase() -> Result<()> {
        let sort = make_sort_exec_with_fetch(Some(10));
        assert!(sort.filter.is_some(), "TopK filter should be created");

        let mut config = ConfigOptions::new();
        config.optimizer.enable_topk_dynamic_filter_pushdown = true;
        let desc = sort.gather_filters_for_pushdown(
            FilterPushdownPhase::Post,
            vec![Arc::new(Column::new("a", 0))],
            &config,
        )?;
        // Parent filters are still blocked in the Post phase.
        assert!(matches!(
            desc.parent_filters()[0][0].discriminant,
            PushedDown::No
        ));
        // But the TopK self-filter should be pushed down.
        assert_eq!(desc.self_filters()[0].len(), 1);
        Ok(())
    }

    #[test]
    fn test_sort_batch_radix_multi_column() {
        let a1: ArrayRef = Arc::new(Int32Array::from(vec![2, 1, 2, 1]));
        let a2: ArrayRef = Arc::new(Int32Array::from(vec![4, 3, 2, 1]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![a1, a2]).unwrap();

        let expressions = LexOrdering::new(vec![
            PhysicalSortExpr::new_default(Arc::new(Column::new("a", 0))),
            PhysicalSortExpr::new_default(Arc::new(Column::new("b", 1))),
        ])
        .unwrap();

        // No fetch -> should take the radix path
        let sorted = sort_batch(&batch, &expressions, None).unwrap();
        let col_a = sorted
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let col_b = sorted
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(col_a.values(), &[1, 1, 2, 2]);
        assert_eq!(col_b.values(), &[1, 3, 2, 4]);
    }

    #[test]
    fn test_sort_batch_lexsort_with_fetch() {
        let a: ArrayRef = Arc::new(Int32Array::from(vec![5, 3, 1, 4, 2]));
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![a]).unwrap();

        let expressions = LexOrdering::new(vec![PhysicalSortExpr::new_default(
            Arc::new(Column::new("a", 0)),
        )])
        .unwrap();

        // With fetch -> should use lexsort path
        let sorted = sort_batch(&batch, &expressions, Some(2)).unwrap();
        assert_eq!(sorted.num_rows(), 2);
        let col = sorted
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(col.values(), &[1, 2]);
    }

    #[test]
    fn test_use_radix_sort_heuristic() {
        // Primitive columns -> radix
        assert!(use_radix_sort(&[&DataType::Int32]));

        // All dictionary -> lexsort
        let dict_type =
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        assert!(!use_radix_sort(&[&dict_type]));

        // List column -> lexsort
        let list_type =
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true)));
        assert!(!use_radix_sort(&[&list_type]));

        // Mixed dict + primitive -> radix
        assert!(use_radix_sort(&[&dict_type, &DataType::Int32]));

        // Empty -> no radix
        assert!(!use_radix_sort(&[]));
    }

    #[test]
    fn test_sort_batch_radix_with_nulls_and_options() {
        let a: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(3),
            None,
            Some(1),
            None,
            Some(2),
        ]));
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(schema, vec![a]).unwrap();

        // Descending, nulls first
        let expressions = LexOrdering::new(vec![PhysicalSortExpr::new(
            Arc::new(Column::new("a", 0)),
            SortOptions {
                descending: true,
                nulls_first: true,
            },
        )])
        .unwrap();

        let sorted = sort_batch(&batch, &expressions, None).unwrap();
        let col = sorted
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        // nulls first, then descending: NULL, NULL, 3, 2, 1
        assert!(col.is_null(0));
        assert!(col.is_null(1));
        assert_eq!(col.value(2), 3);
        assert_eq!(col.value(3), 2);
        assert_eq!(col.value(4), 1);
    }

    #[test]
    fn test_sort_batch_radix_matches_lexsort() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(0xCAFE);

        for _ in 0..50 {
            let len = rng.random_range(10..500);
            let a1: ArrayRef = Arc::new(Int32Array::from(
                (0..len)
                    .map(|_| {
                        if rng.random_bool(0.1) {
                            None
                        } else {
                            Some(rng.random_range(-100..100))
                        }
                    })
                    .collect::<Vec<_>>(),
            ));
            let a2: ArrayRef = Arc::new(StringArray::from(
                (0..len)
                    .map(|_| {
                        if rng.random_bool(0.1) {
                            None
                        } else {
                            Some(
                                ["alpha", "beta", "gamma", "delta", "epsilon"]
                                    [rng.random_range(0..5)],
                            )
                        }
                    })
                    .collect::<Vec<_>>(),
            ));

            let schema = Arc::new(Schema::new(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Utf8, true),
            ]));
            let batch = RecordBatch::try_new(schema, vec![a1, a2]).unwrap();

            let desc = rng.random_bool(0.5);
            let nf = rng.random_bool(0.5);
            let opts = SortOptions {
                descending: desc,
                nulls_first: nf,
            };
            let expressions = LexOrdering::new(vec![
                PhysicalSortExpr::new(Arc::new(Column::new("a", 0)), opts),
                PhysicalSortExpr::new(Arc::new(Column::new("b", 1)), opts),
            ])
            .unwrap();

            // fetch=Some(len) forces the lexsort path while returning all rows
            let lexsort_result =
                sort_batch(&batch, &expressions, Some(len as usize)).unwrap();
            // fetch=None takes the radix path for these column types
            let radix_result = sort_batch(&batch, &expressions, None).unwrap();

            assert_eq!(
                radix_result.num_rows(),
                lexsort_result.num_rows(),
                "row count mismatch"
            );

            for col_idx in 0..batch.num_columns() {
                assert_eq!(
                    radix_result.column(col_idx).as_ref(),
                    lexsort_result.column(col_idx).as_ref(),
                    "column {col_idx} mismatch"
                );
            }
        }
    }

    /// Helper to create an ExternalSorter for testing
    fn test_sorter(
        schema: SchemaRef,
        expr: LexOrdering,
        batch_size: usize,
        sort_coalesce_target_rows: usize,
        pool: Arc<dyn MemoryPool>,
    ) -> Result<ExternalSorter> {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool)
            .build_arc()?;
        let metrics_set = ExecutionPlanMetricsSet::new();
        ExternalSorter::new(
            0,
            schema,
            expr,
            batch_size,
            10 * 1024 * 1024,
            sort_coalesce_target_rows,
            SpillCompression::Uncompressed,
            &metrics_set,
            runtime,
        )
    }

    /// Collect sorted output and verify ascending order on column 0.
    async fn collect_and_verify_sorted(
        sorter: &mut ExternalSorter,
    ) -> Result<Vec<RecordBatch>> {
        let schema = Arc::clone(&sorter.schema);
        let stream = sorter.sort().await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        let merged = concat_batches(&schema, &batches)?;
        if merged.num_rows() > 1 {
            let col = merged.column(0).as_primitive::<Int32Type>();
            for i in 1..col.len() {
                assert!(
                    col.value(i - 1) <= col.value(i),
                    "Not sorted at index {i}: {} > {}",
                    col.value(i - 1),
                    col.value(i)
                );
            }
        }
        Ok(batches)
    }

    /// Radix-eligible batches are coalesced to `sort_coalesce_target_rows`
    /// and chunked back to `batch_size` after sorting.
    #[tokio::test]
    async fn test_chunked_sort_radix_coalescing() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let expr: LexOrdering =
            [PhysicalSortExpr::new_default(Arc::new(Column::new("x", 0)))].into();

        let pool: Arc<dyn MemoryPool> =
            Arc::new(GreedyMemoryPool::new(100 * 1024 * 1024));
        let mut sorter = test_sorter(Arc::clone(&schema), expr, 8192, 32768, pool)?;

        // 8 batches × 8192 rows = 65536 rows → 2 coalesced chunks of 32K
        for i in 0..8 {
            let values: Vec<i32> = ((i * 8192)..((i + 1) * 8192)).rev().collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from(values))],
            )?;
            sorter.insert_batch(batch).await?;
        }

        assert_eq!(sorter.sorted_runs.len(), 2);
        // 32K rows / 8K batch_size = 4 chunks per run
        assert_eq!(sorter.sorted_runs[0].len(), 4);
        assert_eq!(sorter.sorted_runs[1].len(), 4);

        let batches = collect_and_verify_sorted(&mut sorter).await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 65536);

        Ok(())
    }

    /// When sort() is called before the coalesce target is reached,
    /// the partial coalescer contents are flushed and sorted.
    #[tokio::test]
    async fn test_chunked_sort_partial_flush() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let expr: LexOrdering =
            [PhysicalSortExpr::new_default(Arc::new(Column::new("x", 0)))].into();

        let pool: Arc<dyn MemoryPool> =
            Arc::new(GreedyMemoryPool::new(100 * 1024 * 1024));
        let mut sorter = test_sorter(Arc::clone(&schema), expr, 8192, 32768, pool)?;

        // 2 batches × 8192 = 16384 rows (below 32K target)
        for i in 0..2 {
            let values: Vec<i32> = ((i * 8192)..((i + 1) * 8192)).rev().collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from(values))],
            )?;
            sorter.insert_batch(batch).await?;
        }

        // Data is in the coalescer, not yet sorted into runs
        assert_eq!(sorter.sorted_runs.len(), 0);

        let batches = collect_and_verify_sorted(&mut sorter).await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 16384);

        Ok(())
    }

    /// Spilling writes one spill file per sorted run (no merge before spill).
    #[tokio::test]
    async fn test_spill_creates_one_file_per_run() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let expr: LexOrdering =
            [PhysicalSortExpr::new_default(Arc::new(Column::new("x", 0)))].into();

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(500 * 1024));
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool)
            .build_arc()?;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let mut sorter = ExternalSorter::new(
            0,
            Arc::clone(&schema),
            expr,
            8192,
            0,    // no merge headroom → per-run spill path
            8192, // coalesce to batch_size → 1 run per batch
            SpillCompression::Uncompressed,
            &metrics_set,
            runtime,
        )?;

        for i in 0..20 {
            let values: Vec<i32> = ((i * 8192)..((i + 1) * 8192)).rev().collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from(values))],
            )?;
            sorter.insert_batch(batch).await?;
        }

        assert!(sorter.spilled_before());
        // Each run spills as its own file (not merged into one)
        assert!(
            sorter.spill_count() > 1,
            "Expected multiple spill files, got {}",
            sorter.spill_count()
        );

        let batches = collect_and_verify_sorted(&mut sorter).await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 20 * 8192);

        Ok(())
    }

    /// With merge headroom (sort_spill_reservation_bytes > 0), runs are
    /// merged into a single sorted stream before spilling to one file.
    #[tokio::test]
    async fn test_spill_merges_runs_with_headroom() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let expr: LexOrdering =
            [PhysicalSortExpr::new_default(Arc::new(Column::new("x", 0)))].into();

        // Pool sized to trigger spilling after a few coalesced runs but
        // leave enough room for the merge-before-spill path to work.
        // merge_reservation must cover merge cursor infrastructure (~131KB
        // for i32 with spawn_buffered + SortPreservingMergeStream).
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(600 * 1024));
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool)
            .build_arc()?;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let mut sorter = ExternalSorter::new(
            0,
            Arc::clone(&schema),
            expr,
            8192,
            200 * 1024, // merge headroom: enough for merge cursor infrastructure
            32768,
            SpillCompression::Uncompressed,
            &metrics_set,
            runtime,
        )?;

        for i in 0..20 {
            let values: Vec<i32> = ((i * 8192)..((i + 1) * 8192)).rev().collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from(values))],
            )?;
            sorter.insert_batch(batch).await?;
        }

        assert!(sorter.spilled_before());
        // Runs merged before spilling → fewer spill files than runs
        let spill_count = sorter.spill_count();
        assert!(
            spill_count > 0 && spill_count < 20,
            "Expected merged spill files, got {spill_count}",
        );

        let batches = collect_and_verify_sorted(&mut sorter).await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 20 * 8192);

        Ok(())
    }
}
