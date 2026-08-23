# Stereo Quality Edge Reconstruction Task 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that the canonical Quality algorithms, bounded decoders, streaming control plane, and final-encoding coordinator can meet their exactness, memory, and runtime contracts before any production renderer behavior changes.

**Architecture:** Task 0 lives in an importable `scripts.quality_task0` prototype package and has independent scalar oracles in tests. Prototype code may later be promoted only after it passes exact-output and resource gates. Production modules remain unchanged until every blocking Task 0 report is complete and a gate decision records either `existing_dependencies` or `prebuilt_extension`.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, PyTorch, pytest, `tracemalloc`, process RSS probes, platform-native allocation adapters, FFmpeg/FFprobe 5/6/7 fixtures.

**Spec:** `docs/superpowers/specs/2026-08-19-stereo-quality-edge-reconstruction-canonical.md`

## Global Constraints

- Preserve Fast RGB and mask bytes; Task 0 must not dispatch or publish Quality output.
- Each production candidate starts with an independent scalar oracle and a test observed failing for the intended missing behavior.
- Fixed arenas, checked counters, exact tie order, fatal budget errors, and left-eye-then-right-eye shared budgets are semantic, not benchmark hints.
- Runtime JIT, an added runtime dependency, silent Fast fallback, approximate nearest-neighbour search, duplicate heap entries, and unbounded Python per-sample objects are forbidden.
- Run CPU and CUDA contexts where CUDA is available. A missing approved CUDA gate blocks default enablement, not CPU prototype work.
- The worktree test environment must prepend `E:\Code\depth-surge-3d\.venv\Scripts` to `PATH` and set `PYTHONPATH` to this worktree's `src`.

---

### Task 0.1: Reproducible Gate Harness

**Files:**
- Create: `scripts/quality_task0/__init__.py`
- Create: `scripts/quality_task0/reporting.py`
- Create: `scripts/run_quality_task0.py`
- Create: `tests/unit/test_quality_task0_reporting.py`

- [ ] Add RED tests for canonical JSON, exact run descriptors, process RSS sampling, checked uint64 counters, CPU/CUDA context labels, and fail-closed incomplete reports.
- [ ] Implement deterministic report construction and atomic report publication without importing production Quality code.
- [ ] Add CLI selection for individual gates and the complete suite; reject unknown or duplicate gate names.
- [ ] Record commit, dirty diff hash, Python/NumPy/Torch/OpenCV versions, platform, device context, wall time, peak RSS, exact arena bytes, visit totals/max/p95, and equality hashes.
- [ ] Verify the report reader rejects missing runs, noncanonical JSON, cap failures mislabeled as success, and reports produced from a different source diff.

### Task 0.2: Deterministic Per-region Implicit K-d Index

**Files:**
- Create: `scripts/quality_task0/implicit_kd.py`
- Create: `tests/unit/test_quality_task0_implicit_kd.py`

- [x] Add independent brute-force repair and geometry-nearest oracles covering row/column ties, unique row-major sample identities, noncanonical sparse-ID rejection, absent donor ranges, exact-radius candidates, and degenerate lines.
- [x] Add RED tests for the exact uint32 member/offset layouts, reverse row-major grouping, alternating axes, lower medians, deterministic median-of-medians partitioning, and fixed-stack overflow checks.
- [x] Implement one contiguous `member_index` array and one `region_offsets` array with implicit child ranges and no per-node index object.
- [x] Add RED tests proving repair queries debit the shared two-eye visit counter before using a node and fail without returning the current best when the next debit crosses the cap.
- [x] Implement exact integer repair lookup with radius pruning and `(distance2,row,column)` ties.
- [x] Add RED tests proving retained-zero geometry lookup uses the unclipped binary64 coordinate, separately rounded split-plane distance, nearer-child/lower-child tie order, and `(distance2,iy,ix)` ties.
- [x] Implement exact geometry lookup, query diagnostics, and checked totals; Task 0.6 owns the surrounding lifetime-release trace.
- [x] Compare both query modes against their independent scalar oracles over seeded random, adversarial, and degenerate fixtures.

### Task 0.3: Indexed RGB Geodesic Heap

**Files:**
- Create: `scripts/quality_task0/indexed_heap.py`
- Create: `scripts/quality_task0/region_solver.py`
- Create: `tests/unit/test_quality_task0_region_solver.py`

- [x] Add a scalar four-neighbour Dijkstra oracle with the exact BGR movement cost, seed ownership, and priority tuple.
- [x] Add RED tests for one live entry per unsettled pixel, indexed decrease-key, heap-position sentinels, neighbour order, equal-cost owner ties, and capacity/inconsistent-position failures.
- [x] Implement fixed `uint32` heap storage plus dense `int32` position state without stale duplicate pushes.
- [x] Add RED fixtures for a movable clean edge, competing highlight/fringe/true edge, disconnected validity, flat-colour geometry partition, and global components crossing synthetic band boundaries.
- [x] Implement low-resolution canonical regions, explicit half-pixel nearest labels, boundary dilation, erosion/skeleton seeds, and the global integer geodesic solve.
- [x] Assert exact equality with the scalar oracle and record live heap maximum, exact bytes, and queue operations.

### Task 0.4: Repair Records, Local Oracle, And Shared Budgets

**Files:**
- Create: `scripts/quality_task0/repair_records.py`
- Create: `scripts/quality_task0/local_strip.py`
- Create: `tests/unit/test_quality_task0_local_strip.py`

- [ ] Add RED tests for the frozen unaligned 16-byte dtype, run splitting, far-side identity, record order, contiguous/disjoint masks, and pre-fill reconstruction.
- [ ] Implement deterministic record construction with exact invalid-mask OR/popcount checks.
- [ ] Add the million-one-lane-run fixture and independent scalar local-strip oracle.
- [ ] Test complete whole-run slot/neighbourhood reservations, budget skip without reads/debits, full Chebyshev donor charges, direction-preserving donor order, and shared left-then-right counters.
- [ ] Implement local planning without dense donor helpers or pressure-dependent early termination.

### Task 0.5: Fallback And Exemplar Prototypes

**Files:**
- Create: `scripts/quality_task0/fallback.py`
- Create: `scripts/quality_task0/exemplar.py`
- Create: `tests/unit/test_quality_task0_fallback.py`
- Create: `tests/unit/test_quality_task0_exemplar.py`

- [ ] Reuse Task 0.2 only through its frozen array/query interface and verify the 48 MiB/16 MiB repair-arena partition.
- [ ] Add high-fallback 4K and degenerate donor fixtures, including fatal next-node overflow without a current-best result.
- [ ] Implement exact full-frame provisional fallback colours before exemplar work.
- [ ] Add RED tests for ROI/core ownership, pyramid masks, ties-to-even child means, frontier order, complete 7x7/9x9 supports, reflect-101 target gradients, deterministic candidate subsampling, and every budget boundary.
- [ ] Implement the bounded three-level exemplar oracle and prove unfinished targets retain provisional fallback.

### Task 0.6: Geometry Construction And Lifetime Trace

**Files:**
- Create: `scripts/quality_task0/quality_geometry.py`
- Create: `tests/unit/test_quality_task0_geometry.py`

- [ ] Add independent exact one-sided interpolation tests where a one-ULP primitive difference changes a projected fine lane.
- [ ] Implement the ten-step relative and metric prototype order with allocation/lifetime event tracing.
- [ ] Prove exact Fast bilinear equality outside the edge band and final-geodesic-region selection inside it.
- [ ] Prove region scratch ends before the retained-zero index begins and that derived relative/metric fields follow the frozen operation order.
- [ ] Compare metric clamp statistics and relative eye-offset arithmetic against existing Fast scalar oracles at `G/Q` ratios 0.5, 1, 2, and 4.

### Task 0.7: Native Content Reader And Owned NPZ Writer

**Files:**
- Create: `scripts/quality_task0/content_reader.py`
- Create: `scripts/quality_task0/metric_npz.py`
- Create: `tests/unit/test_quality_task0_content_reader.py`
- Create: `tests/unit/test_quality_task0_metric_npz.py`

- [ ] Freeze golden payloads for the owned writer, current `_atomic_save_npz()`, and supported Python/NumPy cells.
- [ ] Add strict ZIP/NPY parser tests for all three allowlisted layouts and malformed flags, descriptors, offsets, CRC, dtypes, shapes, compression, and trailing data.
- [ ] Implement one-frame `5*G_i` metric ownership, fixed buffers, streaming hashes, and release-before-next-frame audit.
- [ ] Implement the fixed writer with fault injection at compression, seek, patch, sync, replace, and disk-full boundaries.
- [ ] Prove no complete second array/member exists concurrently and all audit/writer RSS formulas hold.

### Task 0.8: Replayable Stereo Control Plane

**Files:**
- Create: `scripts/quality_task0/stereo_control.py`
- Create: `tests/unit/test_quality_task0_stereo_control.py`

- [ ] Add RED tests for replayable source iteration, packed action vectors, `P=0,R=N`, mixed actions, and `N=STEREO_CONTROL_FRAME_CAP` under four MiB.
- [ ] Implement capacity-one Quality lifecycle and bounded Fast slots without eager path/work-item lists.
- [ ] Add global stack-creation lock, parked-worker gate, restoration, reserve adapter, contention, teardown migration, and failure injection tests.
- [ ] Measure effective workers, live thread counts, queues, requested stacks, committed-memory deltas, and decoder/renderer/writer overlap for workers 1, 4, and 16.

### Task 0.9: Streaming Encoding Coordinator And Reservation Recovery

**Files:**
- Create: `scripts/quality_task0/encoding_provider.py`
- Create: `scripts/quality_task0/reservation_protocol.py`
- Create: `tests/unit/test_quality_task0_encoding_provider.py`
- Create: `tests/unit/test_quality_task0_reservation_protocol.py`

- [ ] Prototype `EncodingSequenceProvider` with one frozen generation for manifest streaming, scalar argv, and postpass.
- [ ] Exercise both sides of the FFmpeg launch boundary and signed-int image2 endpoint without an assembled frame list.
- [ ] Freeze legal FFprobe wrappers for FFmpeg 5/6/7 and Windows; reject duplicate authoritative top-level keys while allowing nested duplicates.
- [ ] Implement and fault-test portable output naming, fixed settings locator, write-ahead intents, authenticated physical extents, short-source recovery, job-control terminal reconciliation, and index-last retirement.
- [ ] Cover replace/video/input-manifest/final-manifest crash windows under the approved irreversible post-boundary publication policy.
- [ ] Measure every coordinator phase under its capped stdout/stderr/JSON/settings/ring and timeout formulas.

### Task 0.10: Scale Matrix And Gate Decision

**Files:**
- Create: `scripts/quality_task0/benchmarks.py`
- Create: `tests/unit/test_quality_task0_benchmarks.py`
- Create: `docs/superpowers/reports/2026-08-23-stereo-quality-task0.json`

- [ ] Run 1080p and 4K edge/query populations 25, 50, and 100 percent; run separate 25-percent fallback; run `G/Q` 0.5, 1, 2, and 4.
- [ ] Record scalar equality, heap occupancy, exact index bytes, visit total/max/p95, cap failures, wall time, RSS, allocation lifetime, and CPU/CUDA contexts.
- [ ] Run decoder/control/encoding scale matrices and every required fault injection.
- [ ] Publish one canonical report with all required runs or fail closed without a decision.
- [ ] If existing dependencies pass, record `existing_dependencies` and write the production implementation plan. If they fail runtime/RSS only, stop and prepare the separately reviewed prebuilt-extension plan. Do not edit the production renderer before this decision.

### Task 0.11: Verification And Review Boundary

- [ ] Run every `test_quality_task0_*.py` test with the worktree environment.
- [ ] Run Black, configured flake8/McCabe, mypy, and the complete pytest suite.
- [ ] Run `git diff --check`; inspect status and the full scoped diff.
- [ ] Confirm `src/depth_surge_3d/rendering/stereo_renderer.py` and other production behavior are unchanged in Task 0.
- [ ] Commit the complete Task 0 evidence as one reviewable gate result; do not claim Quality availability.
