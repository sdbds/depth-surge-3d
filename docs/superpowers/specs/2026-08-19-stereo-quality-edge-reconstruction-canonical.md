# Stereo Quality Edge Reconstruction Canonical Specification

## Status

This is the only implementation baseline for the stereo edge-reconstruction
change. It consolidates the chat-approved design and all completed PRO review
rounds. It is pending user review before implementation planning. No production
implementation is authorized by this document alone.

The earlier draft paths (the second now represented only in Git history)
`2026-08-19-stereo-quality-edge-reconstruction-design.md` and
`2026-08-19-stereo-quality-edge-reconstruction-revision-2.md` are superseded in
full. They are historical review records and must not be used to implement or
verify behavior. Existing Revision 5 behavior remains authoritative only where
this document explicitly retains it.

## Relationship to the Existing Renderer Contract

This specification extends and, where explicitly stated, supersedes:

- `docs/superpowers/specs/2026-08-14-stereo-edge-coverage-design.md`;
- `docs/superpowers/specs/2026-08-14-stereo-edge-coverage-revision-5.md`.

Revision 5 remains authoritative for the fixed 16-sample horizontal footprint,
packed strict z-buffer, deterministic source-index tie-break, balanced 16-lane
downsampling tree, complete-row banding, and the Fast pixel result.

The earlier prohibition on RGB-guided geometry reconstruction and a public
quality mode is superseded. It addressed an earlier serration defect. The
reported defect survives the 16-sample renderer and was isolated to cross-edge
low-resolution geometry interpolation plus unsafe disocclusion reconstruction.

## Relationship to Final Encoding Specifications

`docs/superpowers/specs/2026-08-14-direct-vr-ffmpeg-encoding-design.md`
remains authoritative only for the following product and layout decisions:

- `direct_vr_encode` is an opt-in strategy and defaults off;
- final eye sources are selected from stage 06 or 07 by the existing source-
  selection policy; its eager list-returning resolver interface is not retained;
- direct mode stacks the two eyes in the selected side-by-side or over-under
  layout and resizes both eyes when either one needs normalization;
- the existing encoder choice and quality flags remain unchanged; and
- direct mode retains its basic preview omission and encoding-progress behavior.

This canonical specification supersedes that document for input-sequence
validation, audio source/stream selection, output-level `-shortest`, FFmpeg
command normalization, final-video publication and its deterministic indexed
sibling temporary, retained encoding/final-video manifests, portable new-target
naming and contained legacy-target audit, the
closed schema-5 settings/job-control/reservation transactions, bounded encoder
and container-validation process lifecycles, resume, and cleanup. On any
conflict in one of those areas, this document is the sole authority. The older
document's claim that assembled encoding and its recovery path remain unchanged
is therefore no longer applicable to those transaction and validation concerns.

That authority is deliberately limited to job final-media encoding invoked by
`ProcessingOrchestrator` through `VideoEncoder.create_video` or
`VideoEncoder.create_video_from_stereo_sequences`. The independent Web
`/stitch_video` batch-stitching workflow is outside publication v4. Its
`*_stitched_*.mp4` products are not job final-media artifacts, never create or
consume `encoding_input_manifest.json` or `final_video_manifest.json`, never
enter `FinalMediaAuditDispositionV1` or `payload_pruned`, and must not be found
by a final-video glob. Bringing batch stitching under this transaction requires
a separate resource and migration design; this specification does not
authorize it.

## Reported Fixture and Root-cause Evidence

The local fixture root is:

```text
H:\3dtest\1787051840_f908a5f038277cf447b8a6a9b5072311_20260818_191720
```

The supplied SBS screenshot matches `frame_000089.png` with 95 SIFT/RANSAC
inliers. Its mapped source region is approximately `x=1291..3594,
y=51..563` in the SBS output. The screenshot is symptom evidence, not a
committed test asset.

The immutable diagnosis artifacts are:

| Artifact | SHA-256 |
|---|---|
| settings JSON | `13b66e657877c8227b100f10265114f1a88162faff4f6a950d8b2d7234d41fbe` |
| source frame 89 | `cca5b9dd367ab23d4931c73ec98b1d091c431c69385e5d1077608f4ec3fd060b` |
| raw depth frame 89 | `5096c6dd730eb05e6bd5f9777c4541e8a0fbf7c816844296e0434499e26be387` |
| canonical disparity frame 89 | `66d524dd394d82ad9b96c1fda89b725b1b9ff954a25b26fcd0ea0eed8138f77c` |
| Fast left eye frame 89 | `9f2b60cbc1aee589ed00afa6d854eaaa009ae708acc62534a47dddcb68473bfd` |
| Fast right eye frame 89 | `be77f8fc1f8552c4d8240350d1e1a5c6f1b7d89efca059b260eed5490426534a` |
| supplied screenshot | `364e7590cc3ce8c5b724ddf5b319fd1ac7caaee8b74bc17a2b1eef2c9100122` |

The fixture uses 1920x1080 source and per-eye output, 1080x608 relative stereo
geometry in width-height notation, `stereo_strength=1.5`, `convergence=0.5`,
and `occlusion_fill=background`.

Diagnosis established:

- current rerendering reproduces both saved stage-04 eyes byte-exactly;
- stage 99 is an exact concatenation of stage-04 eyes, so later transforms are
  not causal;
- bilinear geometry without fill has 12,171 left and 11,902 right full-hole
  pixels, with a seven-pixel maximum horizontal run;
- current fill changes 21,641 left and 22,214 right pixels, then reports no
  full holes;
- nearest geometry removes much of the soft fringe but creates steps and raises
  the true maximum hole width to 11 pixels;
- 8- and 10-pixel caps change no tested output under bilinear geometry, while a
  6-pixel cap exposes black cracks;
- the current fill repeats one horizontal boundary colour without a background
  barrier, and current frame writing discards coverage diagnostics.

The primary cause is therefore cross-edge bilinear geometry. Boundary-copy
fill is a secondary amplifier. Source antialiasing, motion blur, and raw model
misalignment remain irreducible inputs and must not be mislabeled as renderer
defects.

## Goals

- Preserve the current renderer as selectable, byte-compatible Fast mode.
- Add an offline-first Quality mode while retaining only two public modes.
- Prevent interpolation from constructing geometry between visibly separated
  foreground and background surfaces.
- Use RGB only to position a geometry-supported boundary; do not infer a new
  surface from line art or texture alone.
- Preserve smooth interpolation within one surface.
- Repair only actual uncovered fine lanes and never overwrite valid winners.
- Prevent every repair backend from copying across a source-depth region edge.
- Avoid repeated contaminated boundary colours through safe donors, local strip
  continuation, bounded exemplar synthesis, and explicit fallback accounting.
- Preserve coverage and repair diagnostics through generation, transactions,
  resume, and debugging.
- Reuse upstream source, depth, canonical disparity, and metric geometry when
  only stereo mode or Quality fill limit changes.
- Produce deterministic CPU/CUDA output independent of band height and I/O
  worker order.

## Non-goals

- Recovering a thin structure with no distinct evidence in input geometry.
- Treating arbitrary anime outlines as foreground geometry.
- Semantic segmentation, matting, learned view synthesis, neural inpainting,
  or a new third-party dependency.
- Optical-flow or sequential temporal stabilization in Quality v1.
- Changing model resolution, calibration, strength, convergence, camera
  equations, crop, distortion, upscaling, or VR assembly. Encoding changes are
  limited to the explicit N-frame/audio-duration and publication contracts.
- Hiding artifacts by blurring colour or lowering stereo strength.
- Making the fixed 16-lane sample count user-configurable.

## Review Disposition

| Finding | Disposition |
|---|---|
| Full-frame repair conflicts with banded fine-grid rendering | Replace the one-pass Quality path with banded analysis, global compact planning, and banded final rendering. |
| Metric derived fields could lose their algebraic relationship | Snap primitive fields only, copy the exact Fast baseline outside edge bands, then rerun existing derivation and clamp formulas. |
| Exemplar behavior and complexity were open-ended | Define one pure planner, exact pyramid/update/score rules, fixed memory and evaluation budgets, and deterministic fallback. |
| Dense watershed markers prevent useful motion | Remove watershed. Use eroded or skeleton seeds plus an exact integer geodesic label solver. |
| Fine-lane components and pixel proxies were ambiguous | Bind every unresolved lane segment to a source-region ID and admit only pure single-region proxy pixels as donors. |
| Diagnostics had no transaction or resume contract | Add a separately versioned diagnostics stage, per-frame commit manifests, ordered consolidation, and explicit legacy state. |
| Quality memory, disk, and I/O were unbudgeted | Add mode-specific GPU/host constants, sparse-plan limits, disk preflight, and renderer plus full-pipeline gates. |
| Older settings and resume overrides were ambiguous | Migrate every saved schema v1-v4 to Fast and resolve omitted resume overrides separately from new-job defaults. |
| Six reviewed frames were not hash-bound | Add one complete seven-frame manifest and canonical manifest hash. |
| Statistics were not reproducible | Fix bit assignments, denominators, run definitions, percentile method, count units, and canonical JSON rules. |
| An incorrect RGB edge inside the band was untested | Add a multi-edge fixture with an exact selected-boundary expectation. |
| Segment records lost far-side intent and could overlap | Encode `far_side`, define canonical run reconstruction, and assert mask disjointness, OR equality, and popcount equality. |
| Exemplar pyramid did not define unique state transitions | Define all five per-level arrays plus exact reduction, initialization, eligibility, termination, and target-only upsampling. |
| A one-lane pure proxy could contaminate repairs | Separate `pure_proxy` context from fully covered, barrier-cleared `safe_donor` copy sources. |
| Pixel-level union lost fine-lane connectivity | Make segment records component nodes and union only exact horizontal or vertical lane adjacency. |
| Run width and interpolation arithmetic were ambiguous | Define fine-coordinate spans, canonical component keys, a scalar float64 interpolation oracle, and exact nearest-marker arithmetic. |
| Fallback range and exemplar ROI conflicted | Execute fallback in the full-frame planner with explicit per-core, component, and eye budgets. |
| Compact diagnostics conflicted with the public mask API | Keep a compact pipeline result and materialize the four legacy boolean arrays only in the public wrapper, with memory accounted. |
| Diagnostics regeneration could leave stale downstream output | Every diagnostics-triggered stereo rerender invalidates tracked downstream stages before writing. |
| Persisted summaries cannot prove lane-level provenance | Treat lane statistics as producer-attested and independently cross-checked during generation; resume verifies committed hashes only. |
| A global Quality default was unsafe on CPU | Make the resolved new-job default device-aware: gated CUDA defaults to Quality, CPU defaults to Fast; explicit CPU Quality remains supported and warned. |
| Multiple overlapping design files were unsafe | Consolidate every active rule into this canonical specification and mark all earlier drafts superseded. |
| Local strip continuation and equal-depth behavior were non-unique | Split equal-depth pre-fill runs at the exact fine-lane midpoint and define a complete scalar strip oracle with bounded enumeration. |
| `MemoryError` could change RGB under one identity | Preallocate fixed scratch before plan mutation and make every allocation failure fatal; only semantic budgets may select fallback. |
| Metric Quality received geometry after bilinear damage | Carry native `MetricGeometryFrame` primitives into a Quality-only renderer entry point and compute projection statistics after one-sided resampling. |
| Stage and per-frame fill limits were conflated | Fingerprint configured/scaled limits plus policy versions only for background Quality; store its geometry-dependent limits only in frame diagnostics. |
| Fast component counts had no graph | Mark Fast component counts explicitly unavailable and validate Fast lane/pixel counters without a Quality plan. |
| Diagnostics lacked a strict state machine and schemas | Define exact metadata, frame manifest, stats, JSONL, summary, hashing, building/complete, and aggregate-rebuild contracts. |
| Metric clamp sidecars were outside the transaction | Put new-render clamp statistics in committed frame stats and derive the legacy-compatible clamp summary from diagnostics. |
| Disk reserve omitted RGB transaction overlap | Replace the mask-only estimate with allocation-rounded final payload, aggregate, and one-frame atomic-overlap bounds. |
| Exemplar core ownership was ambiguous | Partition the target bbox into nonoverlapping half-open 384-pixel interiors; halos never own targets. |
| Horizontal union could wrap between rows | Require equal rows and consecutive columns, with a dedicated row-boundary fixture. |
| Fallback lookup had no capacity or complexity contract | Use a capped deterministic per-region implicit k-d index and add a high-fallback 4K stress gate. |
| Tiled geodesic verification had no equivalent algorithm | Require one global host indexed heap and remove the unsupported tiled-equivalence claim. |
| Cleanup could leave retained manifests pointing at deleted stereo RGB | Add a terminal `payload_pruned` diagnostics state which preserves authenticated summaries and final-video identity without claiming reusable frame payloads. |
| Legacy partial repair and mask-policy changes had no state matrix | Make any damaged legacy stage and `false -> true` mask transition redraw all frames; make `true -> false` a manifest-only migration. |
| Quality capacity did not separate active scratch from queued slots | Force Quality v1 to one lifecycle slot and define phase-specific active-render byte formulas under the 512 MiB cap. |
| The global Dijkstra queue was unbounded | Use an indexed binary heap with one uint32 entry per band pixel, dense int32 positions, no stale entries, and typed overflow failure. |
| CUDA OOM retry did not define host rollback | Reset all partial Pass A state or all partial Pass B output at row zero, while preserving only the explicitly immutable state. |
| Fixed 64 KiB JSON estimates could undercount variable arrays | Derive deterministic schema bounds from `H`, `W`, `N`, known strings, integer/float widths, and maximum histogram cardinality before mutation. |
| Strict JSON keys did not fix numeric representation | Assign every numeric field an integer or binary64 JSON type and require source-order scalar aggregation with exact empty-value rules. |
| Quality `occlusion_fill=none` claimed a nonexistent repair graph | Add explicit no-repair availability and nullable budget counters without constructing sparse repair state. |
| Local strip mapping mirrored directional texture | Change to a direction-preserving translated strip and add oriented-gradient plus asymmetric-glyph fixtures. |
| Device type caused semantically identical Quality cache misses | Exclude device type from Quality RGB identity and keep hardware only in non-semantic execution provenance. |
| A final video could be published before its resolved encoder identity became recoverable | Persist and fsync a strict `final_video_manifest.json` from the actual executed command before any prune transition; never infer missing evidence after restart. |
| Reclaim estimates could double-count hard links or logical file sizes | Count unique physical allocations per volume only when every hard link is authorized for deletion; otherwise count zero and revalidate before mutation. |
| Renderer settings and direct-call defaults did not expose a unique mode dispatch | Append keyword-only mode/limit fields to `StereoRenderSettings`, keep `StereoSplatSettings` Fast-only, and make public `settings=None` permanently Fast. |
| Dense `repair_bits` ownership and offset temporaries contradicted the host formula | Keep dense repair bits outside the immutable plan, reuse one planar analysis allocation after planning, and build one eye offset into a preallocated int32 map with row scratch. |
| A legal 7x7 donor could read unsafe pixels through Sobel | Require an original same-region safe 9x9 support and score only its central 7x7 gradients without patch-edge reflection. |
| Quality none lacked its own OOM rollback | Reset its partial RGB, compact diagnostics, histograms, counters, and device state from row zero while preserving immutable geometry, offset, and the earlier eye. |
| Exact heap/k-d loops may miss the Python performance gates | Require a Task 0 prototype and authorize a project-owned prebuilt C++/Torch extension, but no runtime JIT, dependency, semantic relaxation, or silent Fast fallback. |
| Fast's historical 24-byte and interim 28-byte slots omitted native geometry and decoder temporaries | Replace the scalar coefficient with relative/metric lifecycle bounds over `Q`, per-frame `G_i`, compressed input bytes, JSON, and every overlapping slot. |
| Damaged retained aggregates in a pruned job had no audit disposition | Report historical diagnostics as irrecoverably damaged while preserving a separately valid final video; never demote the pruned state to `building`. |
| Final-video input identity was coupled to diagnostics manifests and the wrong stage | Add a retained content-only encoding-input sequence manifest over the exact resolved 06/07 eye files or 99 VR files; mask/stats identity never participates. |
| Metric Quality had no unique fill-control type or stage-plan variant | Add `QualityStereoControls` and a four-way discriminated stage-plan union; never borrow relative or Fast-only settings. |
| Strict uint64 JSON fields could overflow only during consolidation | Derive and checked-multiply every frame/root counter bound before mutation, then use checked addition while aggregating. |
| Prune authorization persisted mutable stage keys rather than paths | Commit strict output-root-relative `PruneEntry` objects and resume deletion only from those versioned paths. |
| Final encoding referred to nonexistent output reserve and unspecified validation | Explicitly decline a compressed-video size guarantee, always encode to a sibling temporary, reserve manifests, and require one exact ffprobe/full-decode contract. |
| Target Sobel could consume barrier or provisional values outside its 7x7 patch | Score a target gradient only when its reflected 3x3 support is entirely processed non-barrier context. |
| Quality-none identity included an unused fill limit | Make all repair-limit identity and diagnostics fields null for Quality none, so a limit-only change preserves RGB and downstream stages. |
| Public integer validation could accept booleans | Reject Python/NumPy booleans before the `Integral` check and normalize accepted values to Python `int`. |
| Relative Quality reassociated the legacy eye-offset arithmetic | Split relative and geometry offset builders; relative displacement, q-jump, and splat offsets all use the exact current Fast operation order. |
| A persisted prune path did not identify the audited directory object | Bind every prune entry to platform physical identity plus a durably committed random marker, and reject replacements or descendant mounts. |
| Zero-retained one-sided interpolation had an unbounded linear search | Add a deterministic per-region implicit k-d index, fixed memory and visit cap, fatal overflow, diagnostics, and Task 0 gates. |
| Quality guide and native geometry identity could reuse stat-only metadata | Add one retained source-ordered Quality content manifest over every guide and actual native geometry payload byte. |
| Configured I/O workers could exceed lifecycle capacity | Derive and create only effective workers, charge their bounded thread envelope, and attest configured/effective/actual counts. |
| Full-frame validation conflicted with output-level `-shortest` | Make the N-frame image sequence authoritative, remove audio `-shortest`, define audio selection/trimming, and fix normalized argument token grammar. |
| Encoding identities recorded but did not constrain PNG headers | Require uniform RGB8 IHDRs per consumed sequence and exact expected assembled dimensions before FFmpeg. |
| Pre/post hashing overstated protection against external ABA mutation | State the project-lock trust boundary explicitly and describe inputs as pre/post validated rather than absolutely observed bytes. |
| Full-resolution RGB labels were produced after one-sided interpolation | Fix one ten-step geometry construction order in which the final geodesic map selects every one-sided region, and forbid overlap between geodesic scratch and the retained-zero index. |
| The 512 MiB proof omitted the `O(N)` Python control graph | Replace frame/path/work-item lists with a replayable streaming source, a bounded two-bit action vector, lazy lifecycle items, streamed metadata, and an explicit 4 MiB control-plane cap. |
| Manifest-only migration had no thread or memory owner when `P=0` | Run every `R` item synchronously on the coordinator after render-pipeline teardown, with its own host peak, source-order progress, and crash-reclassification contract. |
| Marker-last cleanup had an unrecoverable empty-root crash window | Permit only handle-relative removal of an identity-matching, completely empty root when the committed marker is already absent; never extend that exception to descendant deletion. |
| Local and repair-fallback queries had no frame work cap | Add shared two-eye local-slot, local-pixel-charge, and fallback-node budgets, deterministic whole-run local bypass, fatal fallback overflow, diagnostics, and stress gates. |
| The approved Direct VR spec conflicted with final encoding rules | Add an explicit authority map which retains its opt-in/layout/product behavior and supersedes its validation, audio, publication, resume, and cleanup rules. |
| Final video did not bind square-pixel display geometry | Require explicit `setsar=1` in every encoder path and validate both SAR 1:1 and the reduced output DAR. |
| Quality frame-name and float-bit strings were not canonical | Make `frame_name` the minimal six-digit-padded stem, derive payload filenames by extension, and encode float32 value bits as numeric eight-digit lowercase hex independent of host byte order. |
| The 512 KiB native stack was only an aspiration | Specify the process-global stack-size lock, parked-thread creation, restoration and failure teardown, platform stack attestation, and concurrent-job gates. |
| Quality content validation was outside the 512 MiB host lifecycle | Add separate initial-audit and pre-consolidation phases, a one-frame 5G no-copy metric validator, and fixed streaming workspace to the stage maximum. |
| Provider changes after destructive mutation could restart against mixed generations | Freeze one audit generation, distinguish pre-mutation restart from post-mutation fatal abort, and cancel/join every parked or active worker on all initialization failures. |
| Quality RGB metadata had semantic fields but no strict artifact | Define immutable `StereoRgbMetadataV2`, its exact path/schema/bytes/fingerprint/reuse rules, and the sole filename-bearing Fast-v3 compatibility exception. |
| Final encoding still depended on eager source lists | Replace the interface with a replayable `EncodingSequenceProvider`, scalar image2 command construction, and an independent eight-MiB coordinator cap. |
| Pre-publication videos had no non-destructive audit state | Add an unauthenticated legacy-final-media disposition which preserves historical video, forbids manifest synthesis and pruning, and reencodes only explicitly from retained inputs. |
| One local `safe_donor` charge hid a neighbourhood scan | Charge a conservative full Chebyshev support for every donor predicate without allocating a dense mask, and advance Quality RGB identity to v8. |
| Canonical U64 frame indexes exceeded the unspecified image2 domain | Fix the supported image2 maximum to signed-int max, use checked last-index arithmetic, and require boundary integration fixtures. |
| Legacy final-media classification and states were not closed | Persist the pre-migration source settings schema, define `FinalMediaAuditDispositionV1` and its complete artifact-presence matrix, add `not_present`, and remove the redundant `final_video_valid` field. |
| Metric NPZ acceptance delegated syntax to NumPy/ZIP readers | Define `MetricNpzPayloadContractV1`, exact owned plus CPython-dependent historical archive forms, strict ZIP/NPY grammar, a project-owned schema-5 writer, and malformed plus supported-runtime fixture gates. |
| Parent-side FFprobe/decode capture was absent from the eight-MiB proof | Use exact `-show_entries`, byte-capped process drains, bounded incremental JSON state, fixed diagnostic rings, phase-specific peaks, and typed overflow failure. |
| A stereo provider could invent an unspecified directory fingerprint | Require one validated canonical upstream manifest for every schema-5 provider and confine manifest-less legacy Fast reuse to the existing read-only compatibility validator. |
| Publication scope could accidentally absorb batch stitching | Limit publication v4 to the orchestrated assembled/direct job encoders and explicitly exclude `/stitch_video` and `*_stitched_*.mp4`. |
| Quality identity requested provenance absent from its strict metadata | Make guide/native-geometry byte identity sufficient and remove the contradictory upstream-geometry-provenance requirement. |
| One historical NPZ form omitted newer CPython `zipfile` output | Retain the observed ordinary-size form, add the exact sentinel-size/version-45 form, classify by bytes rather than Python minor version, and require every supported lock-matrix cell to match one named grammar. |
| Source settings schema was not final-media producer evidence | Add a monotonic first-publication-v4-attempt marker before FFmpeg launch and require untouched legacy candidates to have its exact null state. |
| Source-audio probe/decode escaped the encoding coordinator proof | Specify exact audio commands, bounded parsers and drains, checked PCM accounting, cancellation/reap behavior, typed overflow, and separate audio phase peaks. |
| Final-media audit recomputed paths with a mutable naming helper | Persist one immutable expected relative path plus naming-algorithm version, migrate from retained output info, and make every encoder/manifest/token consume that path. |
| The owned NPZ writer lacked implementation authority and workspace limits | Authorize only metric-payload storage changes, require incremental CRC/DEFLATE plus seek-patch publication, and cap writer scratch without changing metric arrays or equations. |
| Legal FFprobe JSON section wrappers were rejected | Keep JSON, require authoritative top-level `streams`, accept bounded optional `programs`/`stream_groups` wrappers for audio, require them empty for generated MP4, and freeze FFmpeg 5/6/7 plus distributed-Windows goldens. |
| Settings rewrites had no single byte/resource/durability contract | Make every schema-5 artifact canonical, add `SettingsArtifactTransactionV1`, charge three isolated coordinator phases, and physically reserve every remaining settings extent through terminal completion. |
| Output naming bounded only source code-point count | Redefine the unimplemented v1 over the complete component with scalar/control validation, strict resolution tokens, 240-byte/UTF-16-unit limits, and deterministic hash truncation. |
| Final validation processes could live forever | Give work-unit-aware FFprobe a bounded wall deadline and final full decode a semantic-progress stall deadline through the shared termination/reap state machine. |
| Hash fallback rechecked only length, so dotted Windows device stems remained invalid | Select the longest prefix whose complete fallback passes every `PortableFinalComponentV1` predicate and freeze device-stem goldens. |
| The settings artifact name and closed schema were still implicit | Freeze a job-control locator, one schema-5 relative path, and the exact key/type/nullability/state contract for all five owned settings objects. |
| Final-encoding-only settings reserve could not record an earlier failure | Retain one authenticated terminal-settings extent for the entire nonterminal job and add generation-bound reservation descriptors for every finalization extent. |
| New portable limits rejected locatable historical Windows output | Split new-name portability from the containment-only validator used by read-only legacy audit. |
| The publishing FFmpeg process could stall forever | Add a semantic encode stall clock driven only by strictly advancing FFmpeg progress counters. |
| Final FFprobe timeout ignored counted AAC packets and file scan cost | Derive deterministic work units from video frames plus a supported AAC packet upper bound, then include final-file bytes. |
| Windows superscript device digits escaped the portable-name validator | Add `COM`/`LPT` plus U+00B9/U+00B2/U+00B3 to the exact reserved-base set and freeze true fallback plus non-device controls. |
| Completed migration derived seconds which its own multiply-and-floor invariant rejected | Make integer milliseconds authoritative and require the seconds field to equal one exact binary64 derivation. |
| Central reservation data did not reserve a target-directory entry or identify its future paths | Put every payload extent in its target parent, persist every source/temporary/target/parent/role binding, and precreate the final or replacement entry before downstream mutation. |
| Standalone settings extents had no durable owner across a crash | Publish one fixed write-ahead settings-transition index before creating any indexed extent. |
| Reservation and prune identities used incompatible Windows widths and POSIX mount evidence | Reuse exact Windows identities, persist a filesystem UUID plus opaque POSIX file handle, and keep `statx` mount IDs invocation-local. |
| Missing legacy output metadata was reinterpreted by a new naming algorithm | Treat the historical target as unknown and perform no recomputation or glob. |
| Encoder progress did not distinguish ignorable legal keys from malformed diagnostics | Freeze one bounded ASCII line grammar, semantic keys, control values, and discard behavior for all other legal keys. |
| Reservation plans could not replay the settings bytes they purported to protect | Persist the complete bootstrap/standalone target settings object, captured timestamps, content fingerprint, and raw hash before any dependent extent write. |
| `statx.stx_mnt_id` was incorrectly treated as restart-stable | Split persistent file identity from invocation mount identity and require a stable filesystem UUID plus opaque file handle for destructive POSIX recovery. |
| Finalization had no physical extent for the `payload_pruned` metadata commit | Add an indexed target-parent diagnostics-metadata extent between prune markers and cleanup-pending settings. |
| Indexed short extents and partial descriptor temporaries were classified as conflicts | Permit bounded zero-fill continuation before descriptor commit and publish every descriptor through one deterministic sibling temporary. |
| Canonical settings bytes had job identity but no whole-object corruption check | Add a self `content_fingerprint` while retaining the separate stable settings identity and transaction evidence. |
| Provisional placeholders could reach ordinary artifact audits | Make bootstrap and both active fixed-index recoveries globally precede settings, stage, final-media, and legacy audit. |
| A durable lifecycle-terminal prefix had no recovery owner when both fixed indexes were absent | Add one JobControl-owned extent reconciler before ordinary audit, with a closed initial/terminal state table and fixed crash actions. |
| Full, partial, and stale reservation prefixes had no byte-level boundary | Define one NUL-padded canonical payload frame, require a wholly zero tail, and zero the complete owned extent before replaying any non-framed write. |
| The FFmpeg sibling temporary was neither persisted nor uniquely derivable | Bind one short generation-derived `.tmp.mp4` component and its parent identity in the final index; active recovery alone owns it. |
| Persistent Linux UUID identity implicitly added an undeclared `libblkid` runtime | Remove `libblkid`; use one bounded `/dev/disk/by-uuid` capability adapter and fail before mutation when it cannot prove a unique device mapping. |
| The three write-ahead control artifacts had unnamed publication temporaries | Give each one exact sibling `.create-new.tmp` path and one common no-replace publication/recovery state machine. |
| Stage-04 retained a finalization marker-count name after physical ownership moved | Remove the stage variable entirely and introduce `final_prune_marker_file_count` only inside final-encoding preflight. |
| A full-length pre-descriptor extent could still contain only delayed, uncommitted allocation | Treat every indexed all-zero extent as an incomplete reservation until one whole-range rewrite, file sync, reopen, and allocation classification proves readiness or an unsupported physical-allocation capability. |
| Partial terminal replay had no durable attempt-local clock origin | Persist `attempt_started_at`, preserve it through nonterminal work, reset it only on explicit attempt restart, and derive every terminal duration from it with one integer rule. |
| Final-video rename had no pre-existing target directory entry | Persist the publication method in the final index, then materialize and identity-bind the exact final target through an index-owned target descriptor before FFmpeg. |
| Per-64-KiB payload sync made an O(N) manifest reservation perform thousands of syncs | Stream fixed-size zero writes, perform one file sync per whole extent-fill pass, and gate clean reservation byte, sync-count, and wall-time scaling at large N. |
| Stage-04 reserve still included future `payload_pruned` metadata | Use `stage_metadata_raw` only for stage-04 memory/disk formulas and charge `payload_pruned_metadata_raw` only in final-encoding preflight when pruning is requested. |
| Payload padding was reused as payload/descriptor directory-entry slack | Define one set of payload, descriptor, placeholder, and control-publication charge helpers; charge the source entry and both descriptor names independently, and reuse the helpers in every bootstrap, settings, and final-encoding formula. |
| Post-rewrite allocation states had no exact platform oracle | Add a closed allocation-evidence union, freeze Linux ext4/FIEMAP and Windows NTFS/allocated-range adapters, reject statically knowable missing capability before control publication, and keep an otherwise short synchronized allocation retryable. |
| Final-target descriptor and final index were said to retire together | Make final-index retirement an eight-step index-last protocol and admit the descriptor-absent, terminal-durable intermediate state explicitly in recovery. |
| Direct VR's target-preservation sentence forbade authenticated failure cleanup | Limit the prohibition to an active encode and defer untouched placeholder removal after durable terminal failure to the canonical cleanup contract. |
| Readiness allocation evidence was treated as a permanent descriptor invariant after truncation | Persist it as `allocation_evidence_at_readiness` and use separate full-source, short-source, and committed-target validation contracts. |
| Traditional ext4 block maps were discovered as unsupported only after index publication | Advance the Linux adapter to v2 and accept one uniform `FIEMAP_EXTENT_MERGED` mapping form under the same exact coverage and allocation checks. |
| One `A` per planned name was overstated as an exact directory-growth bound | Define the formulas as minimum admission forecasts; materialization is the proof, and namespace ENOSPC preserves the active authority without starting downstream mutation. |
| Deterministic terminal duration still looked like active compute time | Define both persisted duration fields as attempt wall-clock elapsed time, including pauses and downtime, and require that wording in API/UI presentation. |

## Public Settings Contract

The processing settings schema advances from 4 to 5. An explicit Quality
payload contains:

```json
{
  "stereo_render_mode": "quality",
  "occlusion_fill_max_px": 8
}
```

`stereo_render_mode` accepts exactly `fast` and `quality`.
`occlusion_fill_max_px` accepts non-boolean `numbers.Integral` values 1 through
32, normalizes them to Python `int`, and controls only local Quality
continuation. Python and NumPy booleans are rejected before the `Integral`
test. It is a 1080p-equivalent distance. For render height `H`:

Schema 5 also closes the existing `vr_resolution` string hole. It accepts only
exact `auto`, an exact member of the frozen v1 preset set below, or exact
`custom:<width>x<height>` where each decimal has no sign, whitespace, or leading
zero and normalizes to `1..10000`. Every CLI, Web, new-job, and explicit new-
target path runs this same validator before target-name persistence. A retained
legacy
`expected_output_filename` remains authoritative for read-only audit when its
complete copied component passes `ContainedLegacyFinalComponentV1`; its
historical resolution spelling does not get appended again and the new portable
limits are not retroactively imposed on it.

```text
JOB_OUTPUT_NAME_V1_RESOLUTION_PRESETS = {
    "square-480", "square-720", "square-1k", "square-2k",
    "square-3k", "square-4k", "square-5k",
    "16x9-480p", "16x9-720p", "16x9-1080p", "16x9-1440p",
    "16x9-4k", "16x9-5k", "16x9-8k",
    "ultrawide", "wide-2k", "wide-4k", "cinema-2k", "cinema-4k",
}
```

Every member is 1..32 ASCII bytes and matches
`[a-z0-9]+(?:-[a-z0-9]+)*`. Adding a future resolution preset without a new
output-name algorithm version is forbidden; already persisted targets are never
rechecked against a later preset table.

```text
safe_limit_px = max(1, floor(occlusion_fill_max_px * H / 1080 + 0.5))
```

The existing `occlusion_fill` setting remains exactly `none` or `background`:

- `none` skips reconstruction and retains Revision 5 black uncovered lanes;
- `background + fast` uses the current bounded boundary-copy implementation;
- `background + quality` uses the reconstruction contract below.

The removed `processing_mode` name remains rejected and is not an alias.

Saved schema-5 settings always contain both new fields. Omission is resolved
before persistence and is device-aware:

- a new job on a CUDA renderer resolves to `quality` only after all Quality
  release gates pass;
- a new job on CPU resolves to `fast`;
- an explicit `quality` choice on CPU is supported, emits a visible performance
  warning before rendering, and must pass the CPU correctness and resource
  gates below;
- every saved schema 1 through 4 migrates to `fast` and limit 8;
- a schema-5 resume with no override retains its persisted values.

Settings metadata also persists one non-semantic integer
`source_settings_schema_version`. A new schema-5 job writes `5`. Migration reads
the raw `metadata.settings_schema_version` before parsing or rewriting, retains
that value as `SavedSettingsResult.source_version`, and writes exactly that
integer, in `1..4`, to `source_settings_schema_version` while advancing
`settings_schema_version` to 5. A later migration or settings rewrite preserves
the source value byte-for-byte; it never replaces it with 5. A schema-5 file
with a missing, boolean, noninteger, out-of-range, or contradictory source value
has unknown legacy provenance and cannot qualify for the legacy-final-media
exception below. `metadata.project_version`, package metadata, and the current
runtime version never participate in that decision.

Schema-5 settings also persist the final-media target and one monotonic producer
marker with exactly these additional fields:

```text
metadata.final_media_producer_contract_version:  null | 4
metadata.final_media_publication_generation:     null | string
output_info.expected_final_relative_path:         string
output_info.output_name_algorithm_version:
    "job-output-name-v1" | "persisted-expected-output-filename-v1"
```

The two producer fields are either both null or exactly `4` plus a 128-bit OS-
CSPRNG value encoded as 32 lowercase hexadecimal characters. New schema-5 jobs
and untouched schema-1-through-4 migrations start with the null pair. Immediately
before the first publication-v4 FFmpeg launch attempt, atomically update and
fsync the settings artifact to `4` plus the fresh OS-CSPRNG generation already
authenticated by `FinalEncodingReservationV1`, through
`SettingsArtifactTransactionV1` below. Failure to commit, directory-sync, or
reopen-validate the producer marker is fatal and FFmpeg must not launch. Once
non-null, both values are immutable across failed or later encode attempts,
settings rewrites, resume, and cleanup. The marker records that current
publication code has irrevocably entered its pre-launch attempt gate; it does
not prove that FFmpeg launched, is not media-content identity, and is distinct
from every `EncodingSequenceProvider` generation. A later failed reencode does
not invalidate an older authenticated publication. Missing, crossed, malformed,
or downgraded marker fields in schema 5 are unknown current provenance, never
an implicit null pair.
For a still-raw schema-1-through-4 artifact only, exact absence of both keys is
the pre-contract null pair; presence of either key in such a schema is
contradictory rather than trusted.

`expected_final_relative_path` is a canonical nonempty output-root-relative
POSIX path containing exactly one filename segment, no `.`/`..`, slash,
backslash, drive, NUL, link, or reparse traversal, and ending in lowercase
`.mp4`. For `output_name_algorithm_version="job-output-name-v1"`, its complete
filename segment must satisfy `PortableFinalComponentV1`: every character is a
Unicode scalar value, no scalar has Unicode General Category `Cc`, no scalar is
in `<>:"/\|?*`, UTF-8 encoding is at most 240 bytes, and UTF-16 encoding is at
most 240 code units excluding a BOM. The substring before its first dot,
compared ASCII case-insensitively, is not `CON`, `PRN`, `AUX`, `NUL`,
`COM1`..`COM9`, `COM¹`, `COM²`, `COM³`, `LPT1`..`LPT9`, `LPT¹`,
`LPT²`, or `LPT³`. "ASCII case-insensitively" folds only `A`..`Z`; it
performs no Unicode normalization or case folding and leaves the three
superscript scalars unchanged. The emitted suffix ensures that the segment
ends in neither a space nor a dot. These deliberately platform-independent
limits sit below both the project-supported 255-byte POSIX component boundary
and the 255-UTF-16-code-unit Windows component boundary; an output volume unable
to create this tested component class is unsupported rather than permission to
change job identity.

`ContainedLegacyFinalComponentV1` is deliberately narrower policy and broader
syntax. It accepts one nonempty Unicode-scalar component other than `.` or `..`
which contains no NUL, `/`, or `\` and does not begin with the ASCII drive
prefix `[A-Za-z]:`; it imposes no 240-byte, 240-UTF-16-unit,
Windows-device-name, control-category, trailing-space/dot, or new portable-
character rule. The audit opens that exact child through the already opened
output-root directory handle without following a link or reparse point and
requires a locatable ordinary-file entry before it may report presence. It
never joins an absolute path, normalizes the component, interprets a drive, or
falls back to a glob. This validator exists only to locate and inspect an
already persisted schema-1-through-4 target. It never authorizes creation,
rename, replacement, manifest publication, or cleanup of that spelling.

The path is immutable after job creation or legacy migration. New jobs compute
it once with the frozen `job-output-name-v1` algorithm and persist it before any
frame work. That algorithm is defined over Unicode scalar values, never calls
`Path`, `os.path`, `sanitize_filename`, or `generate_output_filename`, performs
no Unicode normalization or case conversion, and has these exact steps:

1. `metadata.source_video_name` must be a string containing no `/`, surrogate,
   or General-Category-`Cc` character. An empty string selects base `output`.
   Otherwise, remove the final suffix only when the last `.` has at least one
   scalar before and after it; this frozen stem rule leaves `.profile`, `name.`,
   `.` and `..` unchanged.
2. In a nonempty source's stem, replace each scalar in `<>:"/\|?*` with `_`,
   collapse each maximal underscore run to one underscore, and strip leading and
   trailing underscores. A nonempty source which sanitizes to the empty string
   keeps that empty base; it does not become `output`. Do not truncate yet.
3. `processing_settings.vr_format` must be exactly `side_by_side` or
   `over_under`; its filename token is respectively `side-by-side` or
   `over-under`. The resolution must already pass the supported resolution
   setting validator. Exact `auto` has filename token `auto`. Any preset must be
   an exact `JOB_OUTPUT_NAME_V1_RESOLUTION_PRESETS` member and is used exactly.
   A custom token must match
   `custom:([1-9][0-9]{0,4})x([1-9][0-9]{0,4})`, both parsed integers must be at
   most 10,000, and its filename token is
   `custom-<canonical-width>x<canonical-height>` with no leading zero. Empty,
   unsupported, non-ASCII, control-bearing, or otherwise arbitrary resolution
   strings are metadata errors, not filename text.
4. Form the untruncated candidate exactly as
   `<sanitized-stem>_3D_<format-token>_<resolution-token>.mp4`. If it satisfies
   `PortableFinalComponentV1`, emit it unchanged. Otherwise compute
   `digest=SHA256(UTF8(untruncated candidate))`, take its first 32 lowercase hex
   digits, and form
   `<prefix>~<digest>_3D_<format-token>_<resolution-token>.mp4`, where `prefix`
   is the longest scalar prefix of `sanitized-stem` for which the **complete
   fallback filename passes every `PortableFinalComponentV1` predicate**, not
   merely both length limits. Enumerate prefix lengths from the complete scalar
   count down through zero and select the first passing result; there is no tie
   or implementation-dependent search. The fixed zero-prefix tail itself must
   pass the complete validator or metadata validation fails. The scan never
   splits a scalar; a non-BMP scalar counts as four UTF-8 bytes and two UTF-16
   code units. Thus `CON.foo` selects prefix `CON`, producing a first-dot prefix
   beginning `CON~`, rather than retaining the still-invalid `CON.foo`.

The hash suffix makes truncation deterministic and collision-resistant while
the persisted path, not a later recomputation, remains the identity authority.
Golden fixtures freeze hidden names, terminal and repeated dots, an input which
sanitizes to an empty base, invalid ASCII filename characters, repeated
underscores, preset and custom resolutions, invalid/overlong resolution tokens,
isolated surrogates, `Cc` controls, 60 non-BMP scalars, exact 239/240/241 UTF-8-
byte and UTF-16-unit boundaries, deterministic hash truncation, and exact
`CON.foo.mp4`, `con.foo.mp4`, `AUX.bar.mp4`, `NUL..mp4`, `COM1.part.mp4`, and
`LPT9.part.mp4` inputs. They also cover every superscript device base with
`COM¹.foo.mp4`, `COM².part.mp4`, `COM³.part.mp4`, `LPT¹.foo.mp4`,
`LPT².part.mp4`, and `LPT³.part.mp4`. For every fallback case the original
candidate fails the complete validator, fallback executes, and the emitted
result independently passes `PortableFinalComponentV1`. The exact side-by-side/
16x9-1080p goldens are:

```text
CON.foo.mp4   -> CON~afc0ef4e07a53c5302a14d2d1fd02201_3D_side-by-side_16x9-1080p.mp4
con.foo.mp4   -> con~3f9776dda8dc90294815dd61a8c3ace9_3D_side-by-side_16x9-1080p.mp4
AUX.bar.mp4   -> AUX~e8fac5846b90380bd5f72d652d5736c4_3D_side-by-side_16x9-1080p.mp4
NUL..mp4      -> NUL~9143c618d678d7b537d35615d8540440_3D_side-by-side_16x9-1080p.mp4
COM1.part.mp4 -> COM1~fe6252979b85daead85a257e026e2f3e_3D_side-by-side_16x9-1080p.mp4
LPT9.part.mp4 -> LPT9~7fc1ea62997d746b58e6250491ee1b78_3D_side-by-side_16x9-1080p.mp4
COM¹.foo.mp4  -> COM¹~f98034cae9fd53d32245fa532dab5a16_3D_side-by-side_16x9-1080p.mp4
COM².part.mp4 -> COM²~94e29ca23c82f11626cf8a52a80b5dda_3D_side-by-side_16x9-1080p.mp4
COM³.part.mp4 -> COM³~21b0d5fa00b2d2aa801dc222e836bd0d_3D_side-by-side_16x9-1080p.mp4
LPT¹.foo.mp4  -> LPT¹~95b3c94fd50b2c12b8af03aa3028b2a6_3D_side-by-side_16x9-1080p.mp4
LPT².part.mp4 -> LPT²~fee1cd545f4c01fc702da4e08e0a1b40_3D_side-by-side_16x9-1080p.mp4
LPT³.part.mp4 -> LPT³~7a788f8c065d09115bb8cddbe4b0b354_3D_side-by-side_16x9-1080p.mp4
```

`COM³.mp4` and `LPT³.mp4` are deliberate non-fallback controls. Their
complete candidates begin `COM³_3D_...` and `LPT³_3D_...`; the substring
before the first dot is therefore not a reserved device name. Both exact
candidates must pass the validator and the Windows create/open/rename/delete
integration gate. Rejecting either because the source stem alone was reserved
would over-apply the Windows rule to a different final component.

No production
implementation of the earlier 200-code-point draft existed; this paragraph is
the final definition of `job-output-name-v1`. A future helper change therefore
creates a new algorithm version rather than reinterpreting v1.

For schema-1-through-4 migration, require and validate
`output_info.expected_output_filename` with
`ContainedLegacyFinalComponentV1`. If present and valid, copy its exact scalar
sequence to `expected_final_relative_path` and record
`persisted-expected-output-filename-v1`; do not recompute it. An absent field is
`LegacyFinalTargetUnknownError`: migration writes neither locator nor schema-5
replacement, final-media audit performs no path lookup, and the raw legacy
artifact remains byte-exact. A present malformed/unlocatable old field is
`FinalMediaTargetMetadataError`. Neither case permits `job-output-name-v1`, the
historical mutable helper, a guessed `legacy-job-output-name`, or a glob. Valid
legacy writers in this repository persisted the field; its absence is unknown
provenance rather than evidence for any naming algorithm. A schema-5 file
missing either target field is likewise invalid rather than lazily repaired.

Read-only audit may therefore preserve and report a real historical component
which exceeds the new portable limits, including a 60-non-BMP-scalar Windows
name. An explicit reencode, replacement, or new manifest publication first
requires the persisted component to pass `PortableFinalComponentV1`; otherwise
it fails before temporary creation with `LegacyFinalTargetNotPortableError` and
leaves the historical file byte-exact. The user may start a new job, which gets
an independent `job-output-name-v1` target; this job never silently renames its
identity.

Both encoder entry points consume only this persisted path. The executed final
argument, `FinalVideoManifest.relative_path`, and normalized `@output:path64`
must decode to it exactly. Final-media audit reads it directly and never invokes
`generate_output_filename` or any successor.

### Processing Settings Artifact V5

Every newly written schema-5 settings artifact has one byte contract. Its bytes
are exactly the canonical ASCII JSON defined below: sorted keys, no indentation,
`separators=(",", ":")`, `ensure_ascii=True`, `allow_nan=False`, no BOM, and no
trailing LF. JSON strings must decode to Unicode scalar values; an isolated
surrogate is invalid even if written as an escape. The complete strict
`ProcessingSettingsArtifactV5` schema rejects duplicate, missing, and extra keys
at every owned object. Schema-1-through-4 input may retain the existing UTF-8
pretty-JSON spelling while it is read through the strict migration parser, but
its first schema-5 replacement uses these canonical bytes. A schema-5 artifact
whose raw bytes differ from its canonical re-encoding is invalid rather than
silently normalized during audit.

The following notation is normative. `scalar-string` contains only Unicode
scalar values. `u64` is a JSON integer, never a boolean, in `0..2^64-1`;
`positive-u64` excludes zero. `binary64` is a JSON number decoded as a Python
`float`, never an integer or boolean, and must be finite; a range following the
type is inclusive. `hex16`, `hex32`, and `hex64` are respectively 16, 32, and
64 lowercase ASCII hexadecimal characters. `utc-usec` is exactly 27 ASCII bytes in valid
Gregorian UTC form `YYYY-MM-DDTHH:MM:SS.ffffffZ`, with seconds `00..59`.
`T?` means JSON `null` or `T`. Every object named below is closed and every
listed key is required even when its value is null.

`ProcessingSettingsArtifactV5` has exactly these six keys and no others:

```text
content_fingerprint: hex64
metadata:            MetadataV5
video_properties:    VideoPropertiesV5
processing_settings: ProcessingSettingsV5
output_info:         OutputInfoV5
runtime_info:        RuntimeInfoV5
```

`content_fingerprint` is SHA-256 of the canonical ASCII JSON encoding of the
complete object with only the top-level `content_fingerprint` member omitted.
Strict parsing recomputes it before any state interpretation. Every creation or
transition recomputes it before the complete raw SHA-256 is calculated. This is
a deterministic corruption/stale-writer check, not a signature: a party able to
rewrite the artifact can also recompute it. Proof that a mutation belongs to the
protocol comes from the durable intent, revision equation, descriptor, and raw
hash contracts below, not from this self-fingerprint alone.

`MetadataV5` has exactly:

| Key | Type / value |
|---|---|
| `batch_name` | `scalar-string` |
| `job_id` | `hex32` |
| `settings_identity_fingerprint` | `hex64` |
| `settings_artifact_relative_path` | exact `processing-settings-v5.json` |
| `job_reservation_generation` | `hex32` |
| `job_terminal_reservation_id` | `hex32` |
| `settings_schema_version` | exact integer `5` |
| `source_settings_schema_version` | integer `1..5` |
| `settings_revision` | `u64` |
| `processing_attempt` | `positive-u64` |
| `source_video` | `scalar-string` |
| `source_video_name` | `scalar-string` |
| `source_video_fingerprint_algorithm` | exact `file-sample-blake2b-v1` |
| `source_video_fingerprint` | `hex32?` |
| `project_version` | nonempty `scalar-string` |
| `created_at` | `utc-usec` |
| `attempt_started_at` | `utc-usec` |
| `last_updated_at` | `utc-usec?` |
| `terminal_at` | `utc-usec?` |
| `processing_duration_ms` | `u64?` |
| `processing_status` | `"in_progress" | "completed" | "failed"` |
| `cleanup_status` | `"not_started" | "not_requested" | "pending" | "complete" | "incomplete_identity_mismatch" | "incomplete_error"` |
| `final_media_producer_contract_version` | `null | 4` |
| `final_media_publication_generation` | `hex32?` |

Define `utc_ordinal_usec_v1(t)` as the checked integer number of microseconds
from Gregorian `0001-01-01T00:00:00.000000Z` to the strictly parsed `utc-usec`
value, using civil-calendar integer arithmetic and no binary64 conversion.
Define:

```text
attempt_duration_ms_v1(terminal_at, attempt_started_at) =
    max(0, utc_ordinal_usec_v1(terminal_at)
           - utc_ordinal_usec_v1(attempt_started_at)) // 1000
```

This is **attempt wall-clock elapsed time**, not CPU, GPU, or active execution
time. An ordinary resume continues the same attempt and therefore includes every
pause, process outage, and machine-off interval between the two persisted UTC
instants. `processing_duration_ms` is the authoritative compatibility field and
`processing_time_seconds` is only its exact binary64 projection; neither may be
presented as active processing time. API documentation and UI labels use
"attempt elapsed time". Exact active execution accounting would require a
separate durably segmented clock protocol, including a policy for the
unobservable instant of an abrupt crash, and is outside schema 5.

A new job captures one `creation_now_usec` and sets both `created_at` and
`attempt_started_at` to it. Migration captures one `migration_now_usec` before
object construction. A valid legacy epoch value is a non-boolean finite JSON
number in
`0..253402300799.999999`; convert it by checked
`floor(binary64_value*1_000_000)` and format that integer as `utc-usec`.
`created_at` uses valid `created_timestamp`, otherwise `migration_now_usec`.
Legacy `processing_status` maps exact `completed` and `failed` unchanged, maps
absence, `in_progress`, or `paused` to `in_progress`, and rejects every other
value. An in-progress mapping sets `attempt_started_at=migration_now_usec` and
`last_updated_at=migration_now_usec`; the migration boundary is the only honest
start available for its schema-5 attempt. For a terminal mapping,
`attempt_started_at=created_at`; `terminal_at` uses valid `completed_timestamp`
only for completed, otherwise `migration_now_usec`;
`last_updated_at=terminal_at`; and `processing_duration_ms` equals
`attempt_duration_ms_v1(terminal_at,attempt_started_at)`. `terminal_at` and
duration are null for in-progress. It does not retain `created_timestamp`,
`last_updated`,
`last_updated_timestamp`, `completed_at`, `completed_timestamp`,
`processing_duration_seconds`, or `processing_duration_formatted` as extra
schema-5 keys. Any substituted legacy value emits one typed migration warning
outside this artifact; no locale or filesystem mtime participates.

`VideoPropertiesV5` has exactly:

```text
width:                           integer 1..10000
height:                          integer 1..10000
fps:                             binary64 > 0
frame_count:                     integer 1..STEREO_CONTROL_FRAME_CAP
duration:                        binary64 >= 0
codec:                           integer 0..4294967295
sample_aspect_ratio_numerator:   positive-u64
sample_aspect_ratio_denominator: positive-u64
sample_aspect_ratio:             scalar-string
```

The ratio integers are coprime and `sample_aspect_ratio` equals their canonical
unsigned-decimal `<numerator>:<denominator>` spelling. `duration` is the one
validated source value retained at creation/migration; later transitions
preserve it bit-exactly and do not recompute it from a different probe.

`ProcessingSettingsV5` has exactly the following 56 keys. A missing optional
legacy key is materialized to the null/default value stated here during the
single migration; schema 5 never represents omission.

| Type / closed value set | Exact keys |
|---|---|
| `binary64` under the existing named range | `stereo_strength`, `convergence`, `virtual_baseline_mm`, `max_disparity_percent`, `scene_cut_threshold`, `fisheye_fov`, `crop_factor`, `fisheye_crop_factor` |
| `binary64?` in `0..1000000` | `source_fps`, `preview_update_interval` |
| `"auto" | binary64 0.1..1000` | `metric_convergence_distance` |
| integer `1..1000000` | `min_scene_frames` |
| integer `1..16` | `stereo_io_workers` |
| integer `0..1000000` | `temporal_window_size`, `temporal_window_overlap` |
| integer `1..32` | `occlusion_fill_max_px` |
| integer `0..1000000` or null | `denoising_steps`, `seed` |
| integer `1..10000` or null | `per_eye_width`, `per_eye_height`, `vr_output_width`, `vr_output_height`, `source_width`, `source_height` |
| `"auto" | integer 1..10000` | `depth_resolution` |
| `null | "original" | integer 1..120` | `target_fps` |
| strict JSON boolean | `scene_detection`, `preserve_audio`, `keep_intermediates`, `direct_vr_encode`, `apply_distortion`, `experimental_frame_interpolation`, `use_metric_depth`, `enable_live_preview`, `verbose` |
| `"relative" | "metric_camera"` | `stereo_geometry_mode` |
| `"none" | "background"` | `occlusion_fill` |
| `"fast" | "quality"` | `stereo_render_mode` |
| `"auto" | "float16" | "float32"` | `raw_storage_dtype` |
| `"off" | "vdpp"` | `temporal_postprocessor` |
| `"archive" | "delete"` | `migrate_legacy` |
| `"side_by_side" | "over_under"` | `vr_format` |
| exact validated resolution grammar in this section | `vr_resolution` |
| `"equidistant" | "equisolid" | "orthogonal" | "stereographic"` | `fisheye_projection` |
| `"auto" | "none" | "1080p" | "4k"` | `super_sample` |
| `"none" | "x2" | "x4" | "x4-conservative"` | `upscale_model` |
| `"v2" | "v3" | "see_through" | "moge2"` | `depth_model_version` |
| `"auto" | "cuda" | "cpu"` | `device` |
| `"auto" | "libx264" | "nvenc"` | `video_encoder` |
| `scalar-string?` | `start_time`, `end_time`, `model_path`, `model_size`, `video_path`, `min_resolution` |
| `scalar-string` | `output_dir` |

The eight existing numeric ranges in the first row are exactly those frozen by
the schema-4 validator (`0..5`, `0..1`, `0..100`, `0..5`, `0..1`, `75..180`,
`0.5..1`, and `0.5..2` in the key order shown); migration normalizes integers
there to binary64. Missing schema-1-through-4 extension keys normalize exactly
as follows: nullable fields to null; `depth_model_version="v2"`,
`depth_resolution="auto"`, `device="auto"`, and `video_encoder="auto"`; and
`use_metric_depth`, `enable_live_preview`, and `verbose` to false.
All other keys use the defaults already defined in this Public Settings
Contract. Digit strings for `target_fps`/`depth_resolution` normalize to
integers. No environment-derived default remains unresolved in persisted
schema 5.

`OutputInfoV5` has exactly:

```text
output_directory:              scalar-string
expected_output_filename:      scalar-string
expected_final_relative_path:  scalar-string
output_name_algorithm_version:
    "job-output-name-v1" | "persisted-expected-output-filename-v1"
```

The two expected-path strings are byte-for-byte equal. With
`job-output-name-v1` they pass `PortableFinalComponentV1`; with the persisted
legacy version they pass `ContainedLegacyFinalComponentV1`. The second spelling
is retained only as a compatibility field and can never select a different
entry.

`MetricClampRuntimeSummaryV1`, when present, has exactly
`schema_version=1`, `affected_frame_count:u64`,
`mean_clamped_fraction:binary64 0..1`,
`max_clamped_fraction:binary64 0..1`, and
`compatibility_summary_sha256:hex64`. It is a bounded aggregate reference, not
the current O(N) `frame_names`/`clamped_fractions` object. `RuntimeInfoV5` has
exactly:

```text
final_output_relative_path:  null | scalar-string
frames_processed:            u64?
processing_time_seconds:     binary64? >= 0
metric_clamp_summary:        MetricClampRuntimeSummaryV1?
terminal_diagnostic_code:    null | 1..64 ASCII [A-Z][A-Z0-9_]*
terminal_diagnostic_message: null | scalar-string with <=1024 UTF-8 bytes
```

These state invariants are part of the schema, not post-parse conventions:

- Initial schema-5 creation, whether new or migrated, is revision zero and
  attempt one. A new job uses `in_progress` plus `not_started`; all nullable
  time/runtime/producer fields are null. A migrated in-progress job follows the
  adapter below and therefore has `attempt_started_at` plus only
  `last_updated_at` among nullable time fields non-null.
- Every successful settings replacement increments `settings_revision` by
  exactly one, preserves `created_at`, and sets `last_updated_at` to that
  transaction's one captured `utc-usec`. An explicit resume/reencode from
  `failed` or `completed` first
  reserves a new terminal extent, increments `processing_attempt`, resets the
  attempt-local terminal/runtime fields, sets `attempt_started_at` to that
  restart transition's captured time, and sets `in_progress`; it never clears a
  non-null producer pair. Every other nonterminal, producer, and cleanup-pending
  transition preserves `attempt_started_at` byte-for-byte.
- `in_progress` requires null `terminal_at`/duration/terminal diagnostics and
  cleanup `not_started` or `pending`. `pending` is allowed only after both final
  manifests are durable and only when `keep_intermediates=false`.
- `completed` requires non-null `last_updated_at=terminal_at`, duration,
  final-output path equal to the expected path, exact N frames, and processing
  time projection; this wording refers only to the compatibility field and not
  active execution. `processing_duration_ms` must equal
  `attempt_duration_ms_v1(terminal_at,attempt_started_at)` and terminal
  diagnostics are null. Except for the one untouched legacy
  migration case below, metric summary is non-null exactly for a metric job.
  Integer `processing_duration_ms` is authoritative. Define
  `duration_seconds_from_ms_v1(ms)` as the IEEE-754 binary64 result of
  `float64(float64(ms) / float64(1000.0))`, with integer conversion, division,
  and result rounding each using round-to-nearest, ties-to-even and with no
  extended-precision retention or fused operation. The stored
  `processing_time_seconds` must have the identical 64-bit binary64 pattern to
  that function's result. Multiplying seconds back by 1000, flooring, epsilon
  comparison, or accepting a caller-supplied display value is forbidden.
  Cleanup is `not_requested` exactly when intermediates are kept;
  otherwise it is `complete` or one of the two incomplete terminal values.
- `failed` requires non-null `last_updated_at=terminal_at`, duration equal to
  `attempt_duration_ms_v1(terminal_at,attempt_started_at)`, and diagnostic code,
  with all success runtime fields null. Its cleanup value is
  never `pending`; a failure before cleanup uses `not_started`.
- The producer version/generation fields are the null pair or exact `4` plus
  `hex32`. Every other nullable combination is invalid.

The one legacy-to-v5 runtime/state adapter is also exact. It starts from a
fully null `RuntimeInfoV5`. Migrated `in_progress` uses `not_started`. Migrated
`failed` uses `not_started`, code `LEGACY_FAILED_STATUS`, and null message.
Migrated `completed` sets final output to the persisted expected component,
frames to `resolved_legacy_N`, defined here as a valid non-boolean legacy
`runtime_info.frames_processed` in `1..frame_count` or otherwise `frame_count`,
and processing seconds to
`duration_seconds_from_ms_v1(processing_duration_ms)`. Its cleanup is
`not_requested` when
`keep_intermediates=true`; otherwise it is `incomplete_error`, because old
settings cannot authenticate prior deletion. A valid legacy metric clamp
summary is reduced to the bounded object above and its canonical raw SHA-256;
otherwise the migrated metric field is null. This null exception is allowed
only for a terminal, untouched migration with
`source_settings_schema_version<=4` and `processing_attempt=1`; any new completed
attempt requires the non-null metric summary. The producer pair stays null.
Legacy `video_properties` must supply valid width, height, FPS, and frame count;
duration normalizes to binary64 `frame_count/fps` when absent/invalid, codec to
integer zero when absent/invalid, and an absent legacy SAR normalizes exactly to
`1`, `1`, and `"1:1"`. No source reprobe or ambient runtime value participates.

No caller dictionary is merged into any of these objects. Transition APIs
accept a discriminated typed payload, construct the complete new object, and
reject any unknown field before calculating bytes.

The authoritative locator is the canonical, self-fingerprinted job-root file
`job-control-v1.json`, whose closed `JobControlV1` object has exactly:

```text
schema_version:                         1
algorithm_version:                      "job-control-v1"
job_id:                                 hex32
settings_artifact_relative_path:        "processing-settings-v5.json"
legacy_settings_artifact_relative_path: scalar-string?
settings_identity_fingerprint:          hex64
job_reservation_generation:             hex32
initial_settings_reservation_id:        hex32
job_terminal_reservation_id:            hex32
job_root_identity:                      DirectoryIdentity
bootstrap_extent_entries:               list[ReservationEntryV1]
bootstrap_settings_intent:               BootstrapSettingsIntentV1
fingerprint:                            hex64
```

Its nullable legacy component, when non-null, is one Unicode-scalar segment
ending exact `-settings.json`, other than `.`/`..`, with no NUL, slash,
backslash, or leading ASCII `[A-Za-z]:`; it is opened handle-relatively without
following a link. New jobs require null and migrations require the one selector
result described below.

Its fingerprint is SHA-256 of its canonical object with `fingerprint` omitted.
`BootstrapSettingsIntentV1` is exactly:

```text
algorithm_version:                    "settings-bootstrap-intent-v1"
settings_identity_fingerprint:        hex64
reservation_generation:              hex32
target_settings_revision:             0
target_settings_content_fingerprint:  hex64
target_settings_raw_sha256:           hex64
target_settings_payload:              ProcessingSettingsArtifactV5
```

The nested identity/generation equal the locator values. The payload is the
complete initial or migrated schema-5 object, including every captured
`utc-usec`, resolved user setting, runtime default, and its valid
`content_fingerprint`; its revision is zero. `target_settings_raw_sha256` hashes
the standalone canonical settings bytes, not the enclosing job-control bytes,
and the other target fingerprint equals the payload member. Bootstrap recovery
serializes only this durable typed payload and must reproduce the exact raw hash;
it never asks the caller for creation/migration arguments and never reads the
clock again.

`bootstrap_extent_entries` has exactly two items in order: one
`initial_settings` for a new job or `migration_settings` for migration, followed
by `job_terminal_settings`. Both use the locator generation, the matching two
locator IDs, null descriptor publication generation, target
`processing-settings-v5.json`, the exact target-local temporary/central
descriptor paths, the captured `job_root_identity`, settings payload maximum,
allocation unit, role, and null prune triple required by `ReservationEntryV1`
below. These complete entries are the physical bootstrap plan; the nested
settings intent is its semantic payload. Recovery never reconstructs that
payload, their parent identity, or their size from ambient state.
`settings_identity_fingerprint` is SHA-256 of this exact closed canonical object:

```text
algorithm_version:                    "settings-identity-v1"
job_id:                                hex32
settings_artifact_relative_path:       "processing-settings-v5.json"
source_video_fingerprint_algorithm:    "file-sample-blake2b-v1"
source_video_fingerprint:              hex32?
expected_final_relative_path:          scalar-string
output_name_algorithm_version:
    "job-output-name-v1" | "persisted-expected-output-filename-v1"
job_reservation_generation:            hex32
job_terminal_reservation_id:           hex32
```

The same settings identity and path values in `MetadataV5` must match the
locator. Reopening job control also requires the currently opened job-root
directory identity to equal `job_root_identity` before any child lookup.

`job_id`, the job reservation generation, the initial/terminal reservation IDs,
and every later reservation generation/ID are
independent 128-bit OS-CSPRNG values encoded as `hex32`. New jobs create this
locator once before settings and never change it. Migration
retains the exact contained legacy settings component in its nullable history
field, writes schema 5 only to the fixed authoritative path, and never replaces
the historical bytes. If no locator exists, a raw legacy operation uses an
explicitly supplied contained settings component; without one it enumerates
non-link ordinary `*-settings.json` children and requires exactly one. Zero is
not found and more than one is `AmbiguousLegacySettingsError`; mtime and name
sorting are forbidden. Once the locator exists, every transition, resume,
audit, cleanup, marker validation, and producer validation reads only its fixed
path, except that bootstrap alone recognizes the locator-declared absent or
zero-length placeholder state below. Extra `*-settings.json` entries are inert
history and are never candidates. A schema-5 settings file without a valid
matching locator, or a non-bootstrap locator whose fixed target is missing/
mismatched, is a control-artifact conflict rather than permission to glob or
adopt another file.

The bounded artifact constants are:

```text
SETTINGS_ARTIFACT_MAX_RAW_BYTES    = 512 KiB
SETTINGS_ARTIFACT_STREAM_BYTES     = 512 KiB
SETTINGS_ARTIFACT_JSON_STATE_BYTES = 512 KiB
SETTINGS_ARTIFACT_OBJECT_BYTES     = 1 MiB
JOB_CONTROL_MAX_RAW_BYTES          = 1 MiB
JOB_CONTROL_STREAM_BYTES           = 512 KiB
JOB_CONTROL_JSON_STATE_BYTES       = 512 KiB
SETTINGS_TRANSITION_INDEX_MAX_RAW_BYTES = 1 MiB
SETTINGS_TRANSITION_INDEX_STREAM_BYTES  = 512 KiB
SETTINGS_TRANSITION_INDEX_JSON_STATE_BYTES = 512 KiB
```

Job-control and settings-index validation is streaming and never retains their
complete raw bytes. In canonical enclosing JSON, the byte span of a nested
`target_settings_payload` object is exactly its standalone canonical settings
encoding. The parser hashes, strict-validates, and replays that span through the
same reusable stream/JSON-state buffers. After bootstrap, job-control validation
does not materialize its historical target object. Standalone transition
validation retains at most the current `SETTINGS_ARTIFACT_OBJECT_BYTES` object
while streaming the target span field-by-field; it never owns two complete
settings objects or raw copies.

Before a new-job write, migration, or rewrite, derive
`settings_artifact_raw_max` with `max_json_bytes` from the complete strict
schema, every retained concrete value, and the maximum spelling of every state,
timestamp, runtime, target, producer, and cleanup field that the remaining job
can write. Require both the current bounded input and that derived maximum to be
at most `SETTINGS_ARTIFACT_MAX_RAW_BYTES`. Parsing is incremental and owns at
most the stream, JSON-state, and typed-object bounds above; it may not use an
unbounded `json.load`, retain both legacy and schema-5 object trees, or hold a
complete old/new raw-byte pair. Canonical serialization streams from the one
typed object. The encoded-byte counter must not exceed either the derived bound
or the fixed raw cap.

Every schema-5 string/list/map field has a fixed schema cardinality or a
prevalidated concrete path/string whose escaped length is included in that
maximum. `runtime_info` is a closed object, not an arbitrary
`additional_info.update`: a terminal diagnostic uses the closed uppercase ASCII
grammar above at most 64 bytes and an optional message of at most 1,024 UTF-8
bytes before canonical escaping; traceback text, child output, and nested
exception objects
are forbidden. Oversize or unknown runtime data is reported outside the artifact
and cannot make a status transaction exceed its reserved extent.

### Create-new Control Artifact Publication V1

The three write-ahead control artifacts use one nonrecursive create-new
publication protocol. Their final and sole sibling temporary paths are exactly:

```text
job-control-v1.json
    -> .job-control-v1.json.create-new.tmp
.depth-surge-reservations-v1/settings-transition-reservation-v1.json
    -> .depth-surge-reservations-v1/.settings-transition-reservation-v1.json.create-new.tmp
.depth-surge-reservations-v1/final-encoding-reservation-v1.json
    -> .depth-surge-reservations-v1/.final-encoding-reservation-v1.json.create-new.tmp

CONTROL_ARTIFACT_WRITE_CHUNK_BYTES = 4 KiB
```

The writer opens the already validated parent, creates or reopens only that
ordinary non-link, link-count-one temporary, truncates it to zero, streams the
complete canonical object in fixed chunks, file-syncs once, and reopen-validates
its exact EOF, strict schema, self-fingerprint, and parent/job identity. It then
atomically renames-no-replace to the absent final name, syncs the parent, and
reopens the final object. POSIX and Windows use the same no-replace adapters as a
reservation descriptor. Direct final-name writes, random names, PID/time suffixes,
globs, and mtime selection are forbidden.
`control_artifact_create_new_charge(raw)` covers the temporary file data plus
its temporary and absent-final directory entries; none of these three control
artifacts is recursively reservation-backed.

Recovery opens only the exact final/temporary pair and applies this closed table:

| Final | Exact temporary | Action |
|---|---|---|
| absent | absent | no transaction is published |
| absent | complete valid canonical object | validate the current root/settings precondition, rename-no-replace, sync, reopen, then treat the final as authoritative |
| absent | partial or malformed ordinary file | because no downstream mutation was permitted before final publication, unlink and sync it; only a current typed initiating call may then rebuild from its complete in-memory object |
| complete valid canonical object | absent | use the final |
| complete valid canonical object | any ordinary non-link, link-count-one exact temporary | use the final, unlink only that temporary, and sync the parent |
| malformed or identity-mismatched final | any | `ControlArtifactConflictError`; preserve both |
| any | wrong type, link count, or parent | `ControlArtifactConflictError`; preserve it |

For a missing job-control final, passive recovery never invents creation or
migration arguments after removing a non-framed temporary; the job remains
absent. For either fixed index, settings and all existing artifacts remain at
the indexed preflight state and a later invocation may construct a new index.
Any reserved-name payload, placeholder, or descriptor which claims downstream
mutation while the owning final control artifact is absent is instead a conflict
and blocks both candidate publication and temporary removal. When inspecting
the two fixed indexes, classify
both final/temporary pairs first: two authoritative finals, two complete
temporary candidates, or one of each is a conflict before either candidate is
published. Crash injection is required after every chunk, file sync, no-replace
rename, parent sync, reopen validation, and temporary retirement.

### Reservation Extent V1

Disk reservation is persisted transaction state, not an anonymous zero file.
The non-link ordinary job-root directory `.depth-surge-reservations-v1` contains
only bounded canonical indexes, their exact control-publication temporaries,
descriptors, and their exact deterministic descriptor-publication temporaries.
It never contains payload data extents.
Every payload extent is instead a create-new ordinary file
inside the **already opened target parent directory**, so successful reservation
has already allocated the directory entry on the exact volume and mount where
publication will occur. It is filled non-sparsely with zero bytes, file-synced,
reopened without following a link, and only then paired with a central
descriptor.

```text
RESERVATION_ZERO_FILL_CHUNK_BYTES       = 64 KiB
RESERVATION_DESCRIPTOR_WRITE_CHUNK_BYTES = 4 KiB
```

Zero fill uses consecutive writes of exactly the first constant except for the
final remainder, then performs exactly one full file sync for that completed
whole-extent fill pass before reopen validation. A short-file continuation is
one pass over the missing suffix; a forced whole-range rewrite is a new pass and
has its own one final sync. There is no per-chunk sync. Descriptor serialization
uses the second constant except for its final remainder, then performs its one
full file sync before reopen validation. Neither chunk size is selected from
ambient memory or filesystem hints.

Each final-encoding reservation invocation emits three non-semantic scalar phase
metrics outside every canonical artifact: `reservation_write_bytes:u64` is the
sum of bytes returned by zero-fill writes, `reservation_fsync_count:u64` counts
only full-file sync calls for zero-filled payload extents, and
`reservation_wall_time_ms:u64` is checked monotonic elapsed time from immediately
before the first final-index temporary write through ready/error return. Index,
descriptor, placeholder, and directory syncs remain mandatory but are not folded
into the second metric.
On a clean preflight, write bytes equal the sum of new extent logical lengths and
the sync count equals the number of new payload extents; recovery suffix fills or
whole-range repair add their actual bytes and one sync per completed pass. These
counters are streamed/logged and never enter RGB, media, settings, or reservation
identity.

Every reservation payload uses `CanonicalReservationPayloadFrameV1`. At the
declared full `logical_byte_count`, its bytes have exactly one of these states:

```text
zero:
    bytes[0:logical_byte_count] are all 0x00

complete-full:
    0 < L <= payload_raw_max
    bytes[0:L] are the one complete kind-specific canonical artifact
    bytes[L:logical_byte_count] are all 0x00

partial:
    exact full logical length, but neither zero nor complete-full
```

`L` is the first raw NUL byte. Strict canonical JSON never contains a raw NUL;
U+0000 inside a string is escaped. For settings, manifests, and metadata,
`bytes[0:L]` ends at the canonical object's final `}` and has no LF. For a prune
marker, `L` includes its one mandatory LF after that `}`. The validator scans and
proves the entire zero tail; finding a valid shorter object does not permit it to
ignore later nonzero bytes. The extra allocation unit in `logical_byte_count`
guarantees a nonempty padding tail even for a maximum-size payload.

After the truncation step, the only complete short state has file length exactly
`L` and all bytes form that same complete canonical artifact. Any other short
nonzero length is a conflict; a full-length complete frame and its exact
truncated form carry the same semantic bytes. A zero-length target placeholder
is not a payload extent and is classified separately.

An authenticated owned `partial` frame carries no semantic intent. Before any
replay, the owner overwrites the **entire** full logical range with zeros in
`RESERVATION_ZERO_FILL_CHUNK_BYTES` chunks, performs one file sync after the
whole range, reopens, and proves the zero state plus original
identity/allocation. Only then may it
write a target again. Bootstrap and standalone settings replay their persisted
exact target; an internal producer, pending, terminal, manifest, marker, or
diagnostics transition may recapture only values the applicable contract marks
provisional. No implementation may overlay a shorter target on a nonzero prefix.
A `complete-full` or exact complete short payload must instead be published as
those exact bytes; if its typed object is not the legal next artifact/transition,
recovery reports a conflict rather than re-deriving it.

Filesystem identity has one shared persisted spelling. `PosixFileIdentityV1` is
exactly
`{filesystem_uuid:hex32,handle_type:integer -2147483648..2147483647,file_handle_hex:scalar-string}`.
`file_handle_hex` is lowercase hexadecimal of 1 through 128 opaque bytes, so its
length is even and in `2..256`. Linux obtains the UUID without `libblkid`, a
subprocess, or a new application dependency. Its sole v1 resolver is:

```text
LINUX_UUID_DIRECTORY         = "/dev/disk/by-uuid"
LINUX_UUID_ENTRY_CAP         = 4096
LINUX_UUID_SCAN_BUFFER_BYTES = 64 KiB
```

Capture the already opened object's `statx` device major/minor. Open `/dev`, then
`disk`, then `by-uuid` handle-relatively with no followed directory link. Stream
at most `LINUX_UUID_ENTRY_CAP` non-dot entries through the fixed buffer without
retaining a name list. A candidate name is exactly ASCII
`[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}`.
Its `readlinkat` payload must be at most 255 bytes and exactly `../../` plus one
nonempty `/dev` component other than `.` or `..` containing only ASCII
`[A-Za-z0-9._+-]`. Open that
component through the `/dev` handle without following a link, require a block
device, and compare its `major(st_rdev),minor(st_rdev)` to the captured filesystem
device pair. Re-read the same link, re-stat the same block node, and re-stat the
owned object before acceptance. Exactly one matching candidate is required; its
hyphens are removed and ASCII hex is lowercased to form `filesystem_uuid`.
Missing directories, permission denial, a cap excess, an unsafe link target,
zero or multiple matches, device drift, non-block/network/overlay storage, and a
UUID of another width are unsupported capabilities, never alternate identity.

The handle type/bytes are the result of exact
`name_to_handle_at(open_fd,"",...,AT_EMPTY_PATH)` for the already opened,
non-followed object, with no other flag. The size-discovery call and one bounded
retry follow the kernel-reported `handle_bytes`; zero, growth beyond 128, or a
second `EOVERFLOW` is unsupported. The returned mount ID is discarded from the
persisted object. The opaque handle pair is compared byte-for-byte; it is never
decoded as an inode. A filesystem without that unique stable 16-byte UUID,
without stable file handles, or returning a larger handle cannot use persisted
reservation or destructive cleanup v1. `WindowsFileIdentityV1` is exactly
`{volume_serial:hex16,file_id:hex32}`: the first member encodes the 64-bit
`FILE_ID_INFO.VolumeSerialNumber`, the second the 128-bit `FILE_ID_128`.
`ReservationFileIdentityV1` is exactly one of:

```text
POSIX:   {platform: "posix", file_identity: PosixFileIdentityV1,
          link_count: positive-u64}
Windows: {platform: "windows", file_identity: WindowsFileIdentityV1,
          link_count: positive-u64}
```

`DirectoryIdentity` is exactly `{kind:"posix",file_identity:
PosixFileIdentityV1}` or `{kind:"windows",file_identity:
WindowsFileIdentityV1}`. Reservation, prune, and target-parent persisted
comparisons reuse these exact nested types. POSIX reclaim accounting instead uses
the invocation-local allocation tuple defined in Disk and Performance Budget;
Windows reclaim accounting reuses `WindowsFileIdentityV1`. No zero-extension,
alternate field width, path-derived mount spelling, or platform-local alias is
allowed. A Windows port without `FileIdInfo` cannot use this reservation/cleanup
contract.

Linux mount identity is deliberately separate and never serialized.
`InvocationMountIdentityV1` is the in-memory pair
`{algorithm:"statx-mnt-id-unique"|"statx-mnt-id",value:u64}`. Prefer
`STATX_MNT_ID_UNIQUE`; otherwise use `STATX_MNT_ID`. One invocation captures the
job-root value and requires the same algorithm/value for every traversed child,
target parent, extent, and prune descendant. It detects a mount crossing only
during that invocation. A reboot/remount may change it without changing the
persisted filesystem UUID/file handle and therefore does not cause identity
conflict. Conversely, equality of a reusable mount ID is never accepted as
persisted evidence. New-job/migration preflight fails before mutation when the
platform cannot supply both the persisted and invocation-local identities;
inspection of an existing job remains read-only, but reservation adoption and
prune deletion are refused with a capability error. Linux support must pass a
create, unmount/remount, reopen, replacement-with-reused-inode, and submount
integration gate; no content/marker-only fallback may authorize destructive
rebinding. The v1 POSIX mutation adapter is therefore Linux-only; another POSIX
platform needs a separately frozen persistent-handle and mount-instance adapter
before it may claim this contract.

This is a capability matrix, not a distribution-name or util-linux-version
matrix. On Linux, failure of the fixed `/dev/disk/by-uuid`, `statx`, or file-
handle probes makes schema-5 creation, migration, reservation publication,
resume mutation, finalization, and destructive cleanup unsupported on that
storage; `keep_intermediates=true` does not bypass the terminal-settings reserve.
Read-only inspection may still report the capability failure without changing
bytes. Installation and troubleshooting documentation must state these exact
requirements and the actionable unsupported-storage error. The 64-KiB scan
buffer belongs to the isolated job-control validation phase and no system UUID
library allocator or process output exists outside that phase.

Physical-allocation readiness proof has one closed persisted spelling.
`ReservationAllocationEvidenceV1` is exactly one of:

```text
Linux: {
    platform:                     "linux",
    algorithm_version:            "linux-ext4-fiemap-sync-v2",
    filesystem_magic:             "0000ef53",
    logical_coverage_byte_count:  positive-u64,
    allocated_byte_count:         positive-u64
}
Windows: {
    platform:                     "windows",
    algorithm_version:            "windows-ntfs-allocated-ranges-v1",
    filesystem_name:              "NTFS",
    block_refcounting_supported:  false,
    logical_coverage_byte_count:  positive-u64,
    allocated_byte_count:         positive-u64
}
```

At readiness, the coverage value equals the descriptor's
`logical_byte_count`; allocation is at least that value. These are normalized
results, not caller-selected labels.
"Exclusive" in this contract means unshared, fully mapped allocation in the
named filesystem's accounting domain. Storage-controller compression, thin
provisioning, and deduplication below that filesystem are not observable through
either adapter and are outside the claim.
The persisted value is a readiness snapshot, not a permanent property of the
short final artifact. The implementation reruns the named full-range adapter
only while the descriptor's exact identity is still at its source path with
length `logical_byte_count` and a zero, partial, or `complete-full` frame, and
requires the same normalized evidence. A `complete-short` source or committed
target follows its separate phase contract below and never runs this full-range
adapter. Physical offsets, extent count, and Linux mapping form are not persisted
because lawful defragmentation or block-map conversion may change them without
changing the readiness allocation claim.

Before the first job-control or fixed-index temporary write, the read-only
capability gate opens every prospective target parent and checks every
statically knowable volume, parent, build, and syscall prerequisite of the
applicable adapter. It is not a prediction of future directory-tree growth or a
future regular file's mapping form. Reservation-backed mutation cannot use the
64-KiB allocation-unit
fallback: failure to query the exact unit is `ReservationCapabilityError` with
no job mutation. The v2 Linux adapter requires `fstatfs(parent_fd).f_type` equal
to `EXT4_SUPER_MAGIC` (`0x0000ef53`) and a successful
`ioctl(parent_fd, FS_IOC_GETFLAGS)` with neither `FS_COMPR_FL` nor
`FS_ENCRYPT_FL`; every other Linux filesystem needs a separately frozen adapter.
The v1 Windows adapter calls `GetVolumeInformationByHandleW(parent_handle)`,
requires exact filesystem name `NTFS`, and rejects
`FILE_VOLUME_IS_COMPRESSED`, `FILE_READ_ONLY_VOLUME`, and
`FILE_SUPPORTS_BLOCK_REFCOUNTING`. It also obtains
`FileAttributeTagInfo` through `GetFileInformationByHandleEx` and rejects a
target parent carrying `FILE_ATTRIBUTE_SPARSE_FILE`, `FILE_ATTRIBUTE_COMPRESSED`,
`FILE_ATTRIBUTE_ENCRYPTED`, `FILE_ATTRIBUTE_REPARSE_POINT`,
`FILE_ATTRIBUTE_OFFLINE`, or `FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS`.
Unsupported queries, remote storage, ReFS, and any other filesystem fail this
gate with `ReservationCapabilityError`; they do not publish an index and are not
reclassified as a content conflict.
A build which does not bind every named structure, constant, syscall/ioctl, and
handle query marks the corresponding adapter unavailable before job mutation;
there is no reduced probe.

After each whole-extent file sync, the Linux adapter reopens the owned file and
calls `statx(open_fd,"",AT_EMPTY_PATH|AT_STATX_FORCE_SYNC,
STATX_TYPE|STATX_NLINK|STATX_SIZE|STATX_BLOCKS,&stx)`, requiring every requested
result bit. The allocation count is checked `stx_blocks*512`. It then issues
`FS_IOC_FIEMAP` with only
`FIEMAP_FLAG_SYNC`, starting at logical byte zero, requesting the remaining
range, and using exactly 64 `fiemap_extent` slots per call. A full batch without
`FIEMAP_EXTENT_LAST` continues at the checked end of its last extent. Each
positive-length returned extent must advance in strictly increasing logical
order; after clipping to the requested file range, the extents must cover
`[0,logical_byte_count)` exactly once with no gap or overlap, and the final
covering extent must carry `FIEMAP_EXTENT_LAST`.

For Linux, exactly two uniform mapping forms are accepted. An extent-mapped file
has flag zero on every nonfinal covering extent and `FIEMAP_EXTENT_LAST` alone on
the final covering extent. A traditional block-mapped file has
`FIEMAP_EXTENT_MERGED` on every nonfinal covering extent and exactly
`FIEMAP_EXTENT_MERGED|FIEMAP_EXTENT_LAST` on the final covering extent. Mixed
forms are unsupported. The merged form is not treated as sparse or unallocated:
the kernel defines it as coalesced block-based mappings, and it must still pass
the identical no-gap/no-overlap coverage and `stx_blocks*512` checks.
`FIEMAP_EXTENT_UNKNOWN`, `FIEMAP_EXTENT_DELALLOC`, or
`FIEMAP_EXTENT_UNWRITTEN`, a coverage gap, or a short `stx_blocks*512` result is
an unready allocation. `FIEMAP_EXTENT_ENCODED`,
`FIEMAP_EXTENT_DATA_ENCRYPTED`, `FIEMAP_EXTENT_NOT_ALIGNED`,
`FIEMAP_EXTENT_DATA_INLINE`, `FIEMAP_EXTENT_DATA_TAIL`,
`FIEMAP_EXTENT_SHARED`, or any unknown extent flag is an
unsupported file state. `SEEK_DATA`/`SEEK_HOLE` and `st_blocks` alone are never
substitutes for the complete probe. This paging and flag behavior follows the
kernel's [FIEMAP userspace contract](https://docs.kernel.org/filesystems/fiemap.html).

After `FlushFileBuffers`, the Windows adapter requires
`GetFileInformationByHandleEx(FileStandardInfo)` to report exact
`EndOfFile=logical_byte_count`, link count one, a nondirectory/non-delete-pending
file, and `AllocationSize>=logical_byte_count`. Its file
`FileAttributeTagInfo` must have none of `FILE_ATTRIBUTE_SPARSE_FILE`,
`FILE_ATTRIBUTE_COMPRESSED`, `FILE_ATTRIBUTE_ENCRYPTED`,
`FILE_ATTRIBUTE_REPARSE_POINT`, `FILE_ATTRIBUTE_OFFLINE`, or
`FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS`. It then calls
`DeviceIoControl(FSCTL_QUERY_ALLOCATED_RANGES)` synchronously for exact input
range `[0,logical_byte_count)` with one `FILE_ALLOCATED_RANGE_BUFFER` output
slot. Success must return exactly one slot whose range is exactly the input;
zero/short coverage or short `AllocationSize` is unready. Extra ranges,
`ERROR_MORE_DATA`, a forbidden attribute, volume-flag drift, or an unsupported
control is an unsupported file/platform state. This uses the documented rule
that a non-sparse, non-compressed file returns the one requested range, rather
than treating allocation size alone as proof; see
[FSCTL_QUERY_ALLOCATED_RANGES](https://learn.microsoft.com/en-us/windows/win32/api/winioctl/ni-winioctl-fsctl_query_allocated_ranges),
[FILE_STANDARD_INFO](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_standard_info),
[FILE_ATTRIBUTE_TAG_INFO](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_attribute_tag_info),
and
[GetVolumeInformationByHandleW](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getvolumeinformationbyhandlew).

Before descriptor commit, an unready result enters the one whole-range repair
defined below; if still unready afterward it is always
`ReservationIncompleteError`. An unsupported file/platform result is
`ReservationCapabilityError`; an active index and all exact owned paths are
preserved, and lack of allocation evidence is never called
`ReservationConflictError`. `ENOTTY`, `EOPNOTSUPP`, `ENOSYS`, `EBADR`, or the
Windows `ERROR_INVALID_FUNCTION`/`ERROR_NOT_SUPPORTED` result from a named
adapter is capability failure; other syscall/I/O errors propagate as I/O failure
without changing the transaction classification. Once a descriptor is committed,
successful **full-source readiness-phase** adapter execution which proves its
normalized evidence changed is instead an evidence conflict; an adapter which
can no longer execute in that phase remains a capability failure. Short-source
and committed-target validation do not call the readiness adapter and therefore
cannot manufacture this conflict after the intentional truncate.

A central descriptor is the closed `ReservationExtentV1` object:

```text
schema_version:                  1
algorithm_version:               "reservation-extent-v1"
job_id:                          hex32
settings_artifact_relative_path: "processing-settings-v5.json"
settings_identity_fingerprint:   hex64
settings_revision_at_creation:   u64?
settings_raw_sha256_at_creation: hex64?
publication_generation:          hex32?
reservation_generation:          hex32
reservation_id:                  hex32
reservation_kind:
    "initial_settings" | "migration_settings" |
    "job_terminal_settings" | "nonterminal_settings" |
    "producer_settings" | "cleanup_pending_settings" |
    "encoding_input_manifest" | "final_video_manifest" |
    "prune_marker" | "diagnostics_payload_pruned"
transaction_index_relative_path: scalar-string
transaction_index_fingerprint:   hex64
source_extent_relative_path:     scalar-string
temporary_relative_path:         scalar-string
target_relative_path:            scalar-string
target_parent_identity:          DirectoryIdentity
publication_method:              "replace_placeholder" | "replace_existing"
reserved_placeholder_identity:   ReservationFileIdentityV1?
payload_role:                    scalar-string
payload_raw_max:                 positive-u64
logical_byte_count:              positive-u64
allocation_evidence_at_readiness: ReservationAllocationEvidenceV1
allocation_unit:                 positive-u64
filesystem_identity:             ReservationFileIdentityV1
prune_root_relative_path:        scalar-string?
marker_name:                     scalar-string?
marker_payload_sha256:           hex64?
fingerprint:                     hex64
```

The fingerprint is SHA-256 of canonical bytes with only `fingerprint` omitted.
The descriptor is a direct child named exactly
`d-p-<publication-token>-r-<reservation_generation>-<reservation_kind>-<reservation_id>.json`;
the publication token is literal `none` for null and otherwise the exact
`hex32`. Its entry's `descriptor_relative_path` is exactly
`.depth-surge-reservations-v1/<that-name>`. Its sole publication temporary is
the deterministic sibling `.depth-surge-reservations-v1/.<that-name>.create-new.tmp`.
After the owning index is durable, create or reopen that ordinary non-link,
link-count-one temporary, truncate it to zero, stream the complete canonical
descriptor, file-sync/reopen/validate it, then atomically rename-no-replace it to
the absent final name, sync the reservation directory, and reopen the final
descriptor. Linux uses `renameat2(RENAME_NOREPLACE)`; a non-Linux POSIX port
without a native atomic no-replace rename is unsupported. Windows uses
`MoveFileExW` with only
`MOVEFILE_WRITE_THROUGH`, never `MOVEFILE_REPLACE_EXISTING`. If the final
descriptor already exists, it must be the complete matching object; malformed
final bytes are a conflict and are never repaired. If the final name is absent, the exact indexed
partial temporary may be truncated and rewritten. Once a valid final descriptor
exists, an exact leftover temporary may be unlinked only after its ordinary-file,
parent, link-count, and index binding validate. No random descriptor temporary
or direct final-name write is allowed. `transaction_index_relative_path`
is exactly `job-control-v1.json` for bootstrap/lifecycle-terminal descriptors,
the fixed settings-transition index for standalone rewrite descriptors, or the
fixed final-encoding index for new invocation descriptors; its fingerprint must
equal the complete canonical file at that path. All paths are canonical job-root-
relative POSIX paths opened handle-relatively without links, `.` or `..`. In v1,
`source_extent_relative_path == temporary_relative_path`, whose last segment is
exactly
`.depth-surge-r-<reservation_generation>-<reservation_kind>-<reservation_id>.reserved`
inside the parent of `target_relative_path`. For `replace_placeholder`, preflight
requires the target absent, creates/fsyncs a zero-length ordinary file at that
final path, captures it as non-null `reserved_placeholder_identity`, then creates
the separate payload extent. For `replace_existing`, the target entry already
exists and the placeholder field is null. In both methods temporary and target
therefore exist before readiness, share the same already opened parent, and its
identity must equal `target_parent_identity`. There is no cross-directory extent
rename and no publication which first creates a target entry after readiness.

`payload_role` has the exact one-to-one mapping
`initial_settings -> settings_initial`,
`migration_settings -> settings_migration`,
`job_terminal_settings -> settings_terminal`,
`nonterminal_settings -> settings_nonterminal`,
`producer_settings -> settings_producer`,
`cleanup_pending_settings -> settings_cleanup_pending`; both manifest kinds,
`prune_marker`, and `diagnostics_payload_pruned` map to their identical kind
strings. The three prune fields are the
null triple except for `prune_marker`; for that kind they are all non-null,
`temporary_relative_path` and `target_relative_path` have parent
`prune_root_relative_path`, `marker_name` is the target's final segment, and the
hash is over the exact future marker payload. Crossed or redundant spellings are
invalid.

Every settings kind targets exact `processing-settings-v5.json`.
`encoding_input_manifest` targets exact `encoding_input_manifest.json` and
`final_video_manifest` exact `final_video_manifest.json`. A prune target is
exactly `<prune_root_relative_path>/<marker_name>`.
`diagnostics_payload_pruned` targets exact
`04_stereo_diagnostics/metadata.json`. No kind may select a caller-
chosen target or temporary mapping.

Initial/migration settings and prune markers require `replace_placeholder`.
Lifecycle terminal, nonterminal, producer, and cleanup-pending settings require
`replace_existing`; diagnostics payload-pruned metadata also requires
`replace_existing`. A manifest whose target entry is absent at index publication
uses `replace_placeholder`; a present target, whether valid or stale, uses
`replace_existing` and remains byte-exact until replacement. The selected method
is immutable in the descriptor/index. A live bootstrap or fixed write-ahead index
makes its zero-length placeholder provisional even before descriptor commit;
existence alone never makes settings, a manifest, or a marker committed.
`reserved_placeholder_identity` is non-null exactly for
`replace_placeholder`, has link count one, and must still identify the zero-
length target immediately before replacement; it is null for
`replace_existing`.

The settings revision/raw-hash fields are the null pair exactly for the two
bootstrap extents created before initial schema-5 publication. Every later
descriptor captures the current canonical settings revision and complete raw
SHA-256 as a non-null pair; crossed pairs are invalid. Initial, migration, and
lifecycle terminal descriptors have null `publication_generation` and the
immutable job reservation generation. A standalone rewrite descriptor uses the
current nullable producer generation and its settings-index generation. Every
new final-encoding descriptor uses the final index's non-null publication and
invocation reservation generations. Initial/terminal IDs come only from job
control; all other IDs/generations are independent 128-bit OS-CSPRNG `hex32`
values first persisted by their write-ahead index.

For maximum canonical payload `raw`, `payload_raw_max=raw` and
`logical_byte_count=payload_logical_bytes(raw)`. The final `A` in that logical
length is forced all-zero **file data padding**. It is physically allocated while
the payload is written; it is neither an unallocated publication reserve nor a
directory-entry charge.

These are the sole reservation-publication **minimum admission forecast**
helpers:

```text
payload_logical_bytes(raw) = alloc(raw) + A

payload_extent_create_charge(raw) =
      payload_logical_bytes(raw)
    + A  # minimum forecast for the source-extent directory entry

descriptor_create_new_charge(descriptor_raw) =
      alloc(descriptor_raw)
    + 2*A  # minimum forecast for temporary + absent final descriptor entries

placeholder_entry_charge(method) =
    A when method == "replace_placeholder" else 0  # minimum forecast

reservation_extent_charge(raw, descriptor_raw, method) =
      payload_extent_create_charge(raw)
    + descriptor_create_new_charge(descriptor_raw)
    + placeholder_entry_charge(method)

control_artifact_create_new_charge(raw) = alloc(raw) + 2*A

reservation_directory_bootstrap_forecast =
    2*A  # parent entry + at least one new directory data/index unit
```

The two descriptor names are forecast independently even though no-replace
rename is atomic. Thus the `replace_existing` minimum is
`alloc(raw)+alloc(descriptor_raw)+4*A`; `replace_placeholder` forecasts one more
`A`. No term can reuse payload padding as namespace slack. An `A` associated
with a future name is deliberately **not** a strict directory-growth upper
bound: an ext4 htree insertion can allocate a leaf and interior metadata, and a
new reservation directory also owns its initial directory data. The read-only
sum is only the threshold for entering indexed materialization. Successful
creation, sync, reopen, and allocation validation of every declared object is
the actual readiness proof. See the kernel's
[ext4 directory structure](https://docs.kernel.org/filesystems/ext4/directory.html).
Control-artifact
publication, bootstrap initial/terminal settings, standalone settings, final
manifests, prune markers, diagnostics `payload_pruned`, and the final-video
target descriptor all invoke these helpers rather than spelling private `+A`
variants.

The descriptor records the exact normalized
`allocation_evidence_at_readiness` observed after the non-sparse fill;
descriptor bytes remain a separate central file. Full-source reopen validation
requires matching job/settings/index, path/role/parent identity, ordinary-file
identity, link count one, exact logical size, allocation unit, and the applicable
`ReservationAllocationEvidenceV1`. A full-source file rejected by that adapter
does not qualify. Short-source and committed-target validation use the separate
contracts below rather than pretending that the final artifact still has its
former logical length.

Every non-bootstrap reservation is owned before extent creation by one fixed
write-ahead index. `ReservationEntryV1` has exactly:

```text
reservation_generation:             hex32
reservation_id:                     hex32
reservation_kind:                   closed descriptor kind
descriptor_publication_generation:  hex32?
descriptor_relative_path:           scalar-string
source_extent_relative_path:        scalar-string
temporary_relative_path:            scalar-string
target_relative_path:               scalar-string
target_parent_identity:              DirectoryIdentity
publication_method:                  "replace_placeholder" | "replace_existing"
payload_role:                        scalar-string
payload_raw_max:                     positive-u64
logical_byte_count:                  positive-u64
allocation_unit:                     positive-u64
prune_root_relative_path:            scalar-string?
marker_name:                         scalar-string?
marker_payload_sha256:               hex64?
```

Every value is concrete before the index is published and must equal the later
descriptor. Physical allocation size plus extent/placeholder identities intentionally live only
in that later self-fingerprinted descriptor: making them required in the
write-ahead entry would recreate the pre-index orphan window. Before the first
downstream mutation, an index is `ready` only as a derived condition when every
listed descriptor, placeholder where applicable, and target-local extent
validates. After mutation begins, each entry must be exactly one of unconsumed-
ready or consumed with its artifact/revision commit evidence; the two states
cannot overlap. No persisted Boolean may claim readiness or consumption.

A standalone settings transaction first publishes the fixed canonical
`.depth-surge-reservations-v1/settings-transition-reservation-v1.json` with the
closed `SettingsTransitionReservationV1` object:

```text
schema_version:                   1
algorithm_version:                "settings-transition-reservation-v1"
job_id:                           hex32
settings_artifact_relative_path:  "processing-settings-v5.json"
settings_identity_fingerprint:    hex64
settings_revision_at_preflight:   u64
settings_raw_sha256_at_preflight: hex64
target_settings_revision:         u64
target_settings_content_fingerprint: hex64
target_settings_raw_sha256:       hex64
target_settings_payload:          ProcessingSettingsArtifactV5
transition_kind:                  "nonterminal_rewrite" | "attempt_restart"
publication_generation:           hex32?
reservation_generation:           hex32
extent_entries:                   list[ReservationEntryV1]
fingerprint:                      hex64
```

The fingerprint omits only itself. `target_settings_revision` is checked
`settings_revision_at_preflight+1`; it and
`target_settings_content_fingerprint` equal the complete nested payload fields,
and `target_settings_raw_sha256` hashes that payload's standalone canonical
bytes. The payload includes the one already captured transition `utc-usec` and
every resolved caller value. A nonterminal rewrite has exactly one fresh
`nonterminal_settings` entry. Attempt restart has that entry followed by the
replacement `job_terminal_settings` entry carrying the immutable terminal ID
and job reservation generation from job control. The fresh rewrite descriptor
binds this settings index and uses its current nullable publication generation;
the replacement terminal descriptor uses null publication generation and binds
job control, so it remains authenticated after the settings index retires. The
complete target payload binds the one typed transition, so the index cannot be
reused for a different rewrite. Resume constructs no target value from current
time, environment, or repeated caller input; it publishes only the indexed
payload after validating the preflight artifact and legal typed delta.
The fixed index must be absent or byte-valid for the exact current transition;
mtime, random-name discovery, and newest-file selection are forbidden.
Before publishing it, derive
`settings_transition_index_raw=max_json_bytes(SettingsTransitionReservationV1,
the complete concrete target payload and entry list)`, require it not exceed
`SETTINGS_TRANSITION_INDEX_MAX_RAW_BYTES`, and require the checked minimum
admission forecast
`control_artifact_create_new_charge(settings_transition_index_raw)` plus one
`settings_transaction_extent(entry.publication_method)` for each entry. The
index publication forecast is bootstrap admission space; every declared object
is then materialized and synced before the typed settings mutation. A namespace
ENOSPC follows the indexed `ReservationIncompleteError` rule above.

A final-encoding invocation likewise first publishes canonical
`FinalEncodingReservationV1` at
`.depth-surge-reservations-v1/final-encoding-reservation-v1.json`:

```text
schema_version:                   1
algorithm_version:                "final-encoding-reservation-v1"
job_id:                           hex32
settings_artifact_relative_path:  "processing-settings-v5.json"
settings_identity_fingerprint:    hex64
settings_raw_sha256_at_preflight: hex64
publication_generation:           hex32
reservation_generation:           hex32
settings_revision_at_preflight:   u64
final_video_relative_path:        scalar-string
final_video_publication_method:   "replace_placeholder" | "replace_existing"
final_video_existing_target_identity_at_preflight: ReservationFileIdentityV1?
final_video_target_descriptor_relative_path: scalar-string
sibling_video_temporary_relative_path: scalar-string
sibling_video_parent_identity:    DirectoryIdentity
diagnostics_metadata_raw_sha256_at_preflight: hex64?
diagnostics_metadata_fingerprint_at_preflight: hex64?
extent_entries:                   list[ReservationEntryV1]
fingerprint:                      hex64
```

Entries are in fixed consumption order: producer if needed, encoding-input
manifest, final-video manifest, prune markers in canonical prune-entry order,
`diagnostics_payload_pruned` when intermediates are not kept, cleanup-pending if
needed, then the already existing job-terminal extent. The
last entry repeats the descriptor plan values authenticated by job control and
uses null descriptor publication generation; every new invocation entry uses
the index's non-null generation and fresh invocation reservation generation.
The index fingerprint omits only itself. Every descriptor created for a new
entry binds this exact index path/fingerprint; the imported terminal descriptor
continues to bind `job-control-v1.json` and its fingerprint and must validate
byte-for-byte against the repeated entry.

The final-video target is deliberately not a `ReservationEntryV1`: it reserves a
directory slot but has no bounded payload extent. Before index publication,
preflight opens the exact final parent and selects `replace_placeholder` only
when the final component is absent. In that case
`final_video_existing_target_identity_at_preflight` is null. An existing
ordinary non-link, link-count-one final selects `replace_existing` and its
captured identity is the non-null preflight field. Any other target type or
crossed nullability fails before publication. The index cannot contain the
future placeholder identity because the index is required to be durable before
that placeholder is created.

After index publication, the final target is identity-bound by a separate closed
`FinalVideoTargetReservationV1` descriptor:

```text
schema_version:                   1
algorithm_version:                "final-video-target-reservation-v1"
job_id:                           hex32
settings_artifact_relative_path:  "processing-settings-v5.json"
settings_identity_fingerprint:    hex64
settings_raw_sha256_at_preflight: hex64
settings_revision_at_preflight:   u64
publication_generation:           hex32
reservation_generation:           hex32
transaction_index_relative_path:
    ".depth-surge-reservations-v1/final-encoding-reservation-v1.json"
transaction_index_fingerprint:    hex64
final_video_relative_path:        scalar-string
sibling_video_temporary_relative_path: scalar-string
target_parent_identity:           DirectoryIdentity
publication_method:               "replace_placeholder" | "replace_existing"
reserved_target_identity:         ReservationFileIdentityV1
fingerprint:                      hex64
```

Its final name is exactly
`final-video-target-p-<publication_generation>-r-<reservation_generation>.json`
inside `.depth-surge-reservations-v1`; the index stores that complete relative
path. Its sole publication temporary is the deterministic sibling
`.<final-name>.create-new.tmp`. It uses the same canonical self-fingerprint,
4-KiB descriptor stream, one file sync, rename-no-replace, parent sync, and
reopen state machine as `ReservationExtentV1`, and its raw bytes must not exceed
`FINAL_RESERVATION_DESCRIPTOR_RAW_BYTES`.

For `replace_placeholder`, create-new the exact zero-length final component only
after the index is durable, file-sync it, sync the target parent, reopen it
without following links, require link count one, and capture
`reserved_target_identity`. For `replace_existing`, reopen the target and require
its identity to equal the non-null preflight identity before copying it into the
descriptor. Publish and reopen-validate the descriptor next. Thus the index is
the write-ahead authorization, while the later descriptor is the persistent
physical identity; no immutable pre-write index is rewritten to add a
post-create fact.

Recovery first tests the `FinalEncodingRetirementV1` state below. When its
terminal/cleanup preconditions are not satisfied, a missing final-target
descriptor means construction is still in progress: `replace_placeholder`
accepts only an absent target, which it creates as above, or an already present
exact zero-length ordinary non-link/link-count-one child, which it adopts and
identity-binds. `replace_existing` requires the exact persisted preflight
identity still at the target. A nonzero would-be placeholder, unsafe entry,
identity mismatch, or unindexed final-target descriptor is a conflict. A valid
final descriptor with its exact deterministic temporary uses the common both-
present retirement rule; a partial descriptor temporary may be rewritten only
while the final is absent and the indexed target state still validates.

`final_video_relative_path` is exactly the immutable settings
`expected_final_relative_path`. The sibling path is derived once before index
publication and is exactly the ASCII component
`.depth-surge-final-v4-<publication_generation>.tmp.mp4`, using the complete
32-digit generation. It is 62 UTF-8 bytes/code units regardless of the final
component's length, has the `.mp4` suffix required by FFmpeg muxer selection,
and shares the already opened final/output-root parent whose persisted identity
is `sibling_video_parent_identity` and equals JobControl's job-root identity.
Both fields are one-component contained relative paths and must differ. The
selected path must be absent at preflight; a collision fails before index
publication rather than choosing another spelling.

An active final index is ready for FFmpeg only when this final-target descriptor
and every required extent descriptor validate. Before the first video
replacement, the target must still have `reserved_target_identity`; after a
validated sibling atomically replaces it, the descriptor remains as index-owned
evidence that the directory slot was reserved, while the current target is the
derived consumed-uncommitted state until both manifests authenticate it. A crash
in that state still follows the existing reencode rule because the executed argv
was not durable, but it reencodes against the already existing target entry.
Once both manifests validate, the target is consumed-authenticated and recovery
continues prune/settings only. A missing target, a second location for the
reserved identity, or a wrong parent/type/link count is a conflict. The final-
target descriptor remains index-owned through the applicable terminal success/
failure commit and authenticated unused-placeholder handling. Final retirement
then removes and syncs that descriptor **before** removing the final index; the
index is always the last authority to disappear.

Once the index is durable, it exclusively owns that exact sibling path until
index retirement. Before any FFmpeg launch or relaunch, an existing entry there
must be an ordinary non-link, link-count-one child of the persisted parent; it is
unlinked and the parent synced, because an uncommitted video temporary is not
semantic intent and is never adopted. Wrong type/parent/link count is a conflict.
The runtime output argv always names this persisted path, while normalized argv
maps that whole argument to the indexed final path. Success atomically moves it
over the already existing final component; every prepublication failure removes
only this path.
Index retirement requires it absent. Without an active matching index, ordinary
audit neither discovers nor deletes lookalike `.depth-surge-final-v4-*` names.
The two diagnostics-preflight hashes are the null pair exactly when
`keep_intermediates=true`. Otherwise they are both non-null and authenticate the
same strict `complete` or `legacy_fast_unavailable`
`04_stereo_diagnostics/metadata.json` that the diagnostics entry will replace;
the first hashes its complete raw bytes and the second equals its self-
fingerprint. Replacement requires both still match. Crossed pairs, a different
status, or a diagnostics entry with the null pair are invalid.

Both fixed indexes are bounded by `FINAL_RESERVATION_ENTRY_CAP`. The final index
is bounded by `FINAL_RESERVATION_INDEX_RAW_BYTES` and its fixed reservation
stream/JSON-state caps. The settings index is instead bounded by the three
`SETTINGS_TRANSITION_INDEX_*` constants because it contains one complete target
settings payload. Each parser/serializer is incremental. The index is published
through Create-new Control Artifact Publication V1 and reopen-validated
**before** any final-video placeholder, final-target descriptor, new listed
target-local extent, or extent descriptor is created. The
settings index is both
the semantic payload intent and physical plan; the final index is the physical
ordered plan whose later internally derived settings payloads follow the durable-
prefix rule below. Neither index is itself backed by another reservation extent;
doing so would be circular. Its bounded publication is part of read-only free-
space preflight, and failure leaves settings, manifests, markers, and frame work
untouched.

After final-index commit, first create/adopt and sync the indexed final-video
target, then publish its target descriptor. For either fixed index, create/fsync
any declared artifact target placeholder, then create each listed target-local
payload extent, sync/reopen it, capture all identities/allocation, and publish/
sync the central descriptor through its fixed temporary. A crash after index
commit but before the final target, after that target but before its descriptor,
after an artifact placeholder, after only some extents, or between one extent
sync and descriptor commit has one result: resume reads the fixed index and exact
paths. Final-target recovery applies its method/identity rules above. Before an
extent descriptor commit, an absent extent is
created; an exact indexed extent path is adoptable when it is an ordinary
non-link file with link count one in the indexed parent, its length is in
`0..logical_byte_count`, and every present byte is zero. Resume continues
non-sparse zero writes from its current length in exact
`RESERVATION_ZERO_FILL_CHUNK_BYTES` chunks and performs one sync at the end of
that suffix-fill pass. At exact logical length it reopens and validates
allocation. A full-length all-zero file whose allocation is short or merely
returns the named adapter's unready state **before descriptor commit is still
incomplete, not conflicting**: the invocation overwrites the complete logical
range with zero chunks, performs one full file sync, reopens, rescans every byte,
and reruns that same platform adapter. It performs at most one such whole-range
repair pass per invocation.

If a suffix/whole write or its sync reports ENOSPC/error 112, or the completed
whole-range repair still has short allocation, a coverage gap, or the exact
retryable Linux flag set, raise `ReservationIncompleteError`; preserve the index
and exact extent for a later retry. Short or delayed allocation is never evidence
of tampering. If the closed adapter reports an unsupported file/platform state or
cannot run, raise `ReservationCapabilityError` and preserve the same state; do
not turn missing proof into a permanent conflict. Commit the descriptor only
after the reopened file meets the complete normalized evidence contract.
ENOSPC/error 112 while creating or extending the reservation directory, a
placeholder, source entry, descriptor temporary/final entry, final-video target
entry/descriptor, or any other declared namespace object during materialization
is likewise `ReservationIncompleteError`. If the owning control artifact is
already authoritative, preserve it and every exact owned path; if its final has
not committed, the Create-new Control Artifact table owns the sole partial
temporary. In both cases no frame, FFmpeg, settings, manifest, prune, or other
downstream mutation may begin. The minimum admission forecast is reported with
the current free count and failing path, but is never misreported as a guarantee
that this particular directory-tree state needed only one `A` per insertion.
`ReservationConflictError` is limited to nonzero bytes, excess length, wrong
parent/type/link count, an unindexed path, a malformed final descriptor, or a
successful full-source post-commit adapter result which contradicts the
descriptor's readiness evidence. Once a descriptor is committed, its recorded
identity remains strict in every phase; its readiness evidence remains strict
only while the full source exists and is not repaired by this pre-descriptor
exception.
Placeholder adoption still requires its exact path, zero length, parent, method,
and ordinary-file/link-count shape. No index means no new placeholder, final-
target descriptor, or extent is authorized. An unindexed reserved-name file,
extra descriptor, different path/
parent/identity, or two fixed indexes for incompatible work is
`ReservationConflictError`; it is preserved, never silently deleted. Thus there
is no pre-index random group, bounded directory scan, glob, or mtime recovery
rule.

The publication generation is generated before the final index. If the producer
pair is null after a crash, the one valid index plus ready descriptors supplies
the still-unpublished generation and the producer transaction commits exactly
it. If non-null, it must equal the index. For index revision `r`, current settings
revision equals checked
`r + producer_consumed + pending_consumed + terminal_consumed`; each term is
zero or one and becomes one only after that indexed transition validates.
Producer precedes all other settings transitions; pending precedes terminal on
cleanup success, while early failure may consume terminal with pending zero.
Manifest/marker/diagnostics publication does not change revision. At `r`, raw settings hash
must equal the index snapshot; later revisions require exact transition evidence.
No range comparison substitutes for this equation.

Consumption keeps the target-directory allocation and identity. First perform
the full-source phase validation and match
`allocation_evidence_at_readiness`. Write the bounded canonical payload over the
start of the still-full extent, file-sync, reopen the same identity, rerun the
readiness adapter, match that evidence again, and require the exact `complete-full`
`CanonicalReservationPayloadFrameV1` while length remains
`logical_byte_count`. Then truncate to that frame's exact `L`, file-sync, and
reopen; require the same identity, the exact complete-short canonical payload,
and a post-truncate allocated-byte counter no greater than
`allocation_evidence_at_readiness.allocated_byte_count-A`. That counter is
Linux `statx.stx_blocks*512` or Windows `FileStandardInfo.AllocationSize`; it is
only the release check and does not invoke FIEMAP or
`FSCTL_QUERY_ALLOCATED_RANGES` over the former logical range.

For `replace_placeholder`, reopen and require the exact persisted zero-length
placeholder identity. For `replace_existing`, reopen and authenticate the
current old target through the artifact-specific revision/hash contract; this is
necessary because a lifecycle settings extent may outlive earlier settings
replacements. Diagnostics replacement specifically requires both preflight
hashes in `FinalEncodingReservationV1`; its entry/index is therefore the durable
old-target binding. Only after the temporary is exact and truncated may the writer
atomically replace that already existing target within the same opened parent,
sync the directory, and reopen/validate the new target. Thus a crash after
truncate does not need free space to create a directory entry: recovery finds
the exact truncated temporary and retries between two pre-existing entries. A
placeholder stays provisional and an old target stays byte-exact until replace.

The descriptor remains until target commit is durable. The only valid recovery
states are: a full-length indexed extent at its source path in the exact zero,
partial, or complete-full framed state beside the declared placeholder/old
target; an exact complete-short payload at the source path beside that target;
or the exact committed extent identity at the target with the temporary absent.
Only zero/partial may enter the mandatory whole-extent zeroing/replay rule;
complete-full and complete-short publish their exact semantic bytes. The reserved
payload identity is temporary before rename and target afterward. An unindexed
location, two locations for that identity, neither location before a proven
commit, wrong/missing placeholder when required, different parent/identity, or
cross-directory rename is a conflict.

Descriptor validation is phase-discriminated in this order; no generic
"validate descriptor" entry point may run the readiness adapter before locating
and classifying the owned identity:

| Physical phase | Required validation |
|---|---|
| Full source (`zero`, `partial`, or `complete-full`) | Source path, parent, identity, link count, and length equal the descriptor; the full-range platform adapter reruns and exactly matches `allocation_evidence_at_readiness`; the target is the declared placeholder or authenticated old artifact. |
| Short source (`complete-short`) | The same reserved identity is at the exact source path with length `L`; canonical bytes, raw hash, typed schema, and artifact-specific transition are exact; the target remains the declared placeholder or authenticated old artifact; the post-truncate counter proves at least `A` released. The full-range readiness adapter is forbidden. |
| Committed target | Source is absent; the same reserved identity is at the exact target path and parent with length `L`; canonical bytes, raw hash, typed schema, and artifact-specific commit evidence are exact; the indexed placeholder/old-target replacement method is satisfied. The full-range readiness adapter is forbidden. |

In the committed-target row, a satisfied `replace_placeholder` means the
persisted placeholder identity is gone and the target has the reserved source
identity. A satisfied `replace_existing` means the owning index/transition
contains the required old-artifact binding, the current target has the reserved
source identity rather than that old artifact, and the source path is absent.
Recovery does not attempt to reopen a replaced identity after the atomic rename.

After the full-source phase, `allocation_evidence_at_readiness` is historical
proof that the source extent was physically ready before consumption, not a
continuing invariant of the truncated artifact. A short or target-phase mismatch
is an identity/content/transition conflict under its row, never an allocation-
evidence mismatch against the former full length.

For bounded artifact transactions, these pre-existing target and temporary
directory entries are the physical guarantee; a central regular-file allocation
is not accepted as a substitute. For final video, the target entry/descriptor is
ready before FFmpeg and the indexed sibling entry exists after FFmpeg has created
its output, so both names likewise exist before the final replace. Supported
POSIX and Windows filesystems must pass an integration gate which fills ambient
free space only after both same-parent entries exist and still replaces a
placeholder or old target from the other entry, directory/file flush, reopen,
and descriptor retirement. This explicitly covers POSIX `rename`/`renameat`
implementations for which an absent destination could otherwise require directory
growth and return ENOSPC, as permitted by the
[POSIX.1-2024 rename contract](https://pubs.opengroup.org/onlinepubs/9799919799/functions/rename.html).
A filesystem for which the pre-existing-target gate
returns ENOSPC/error 112 is unsupported for reserved publication; the lifecycle
terminal extent remains intact so the attempt can record failure.

New-job creation and valid legacy migration calculate the complete typed
schema-5 object, job control, identity fingerprint, paths, and maxima before any
mutation. Their read-only minimum admission forecast is:

```text
reservation_descriptor_raw_max = max_json_bytes(
    ReservationExtentV1,
    all concrete identity/path/kind/index/role values and the selected
    platform's maximum `ReservationAllocationEvidenceV1` spelling,
)
settings_transaction_extent(method) = reservation_extent_charge(
    settings_artifact_raw_max,
    reservation_descriptor_raw_max,
    method,
)
job_control_raw = max_json_bytes(
    JobControlV1,
    concrete legacy path or null,
    complete BootstrapSettingsIntentV1 target payload,
)
require job_control_raw <= JOB_CONTROL_MAX_RAW_BYTES
job_control_publication_extent =
    control_artifact_create_new_charge(job_control_raw)
initial_settings_reserve =
      job_control_publication_extent
    + reservation_directory_bootstrap_forecast
    + settings_transaction_extent("replace_placeholder")
    + settings_transaction_extent("replace_existing")
```

Bootstrap order is unique: perform that read-only admission check; publish,
sync, and reopen-validate `job-control-v1.json`; create/adopt and sync the
reservation directory; create/sync/reopen the initial-settings target
placeholder plus target-local initial-or-migration and terminal extents;
commit/sync both descriptors; publish schema-5 settings
through the initial/migration extent; retire the consumed initial descriptor;
then either revalidate the retained zero terminal reserve for `in_progress` or
remove the unused zero terminal reserve for a terminal migration. Only after
those steps may frame work begin. Job control's
two complete bootstrap entries are the physical write-ahead plan, its embedded
settings intent is the exact semantic write-ahead payload, and both descriptors
bind the job-control fingerprint.

A valid locator with no committed settings is exactly `locator_only`: its fixed
target is either still absent before placeholder creation or is the exact zero-
length placeholder declared by the first bootstrap entry; no other target bytes
or identity qualify. Recovery creates/adopts only the declared placeholder and
two extents, reconstructs the exact initial bytes only from
`bootstrap_settings_intent`, then resumes bootstrap. It is never an in-progress job, never starts
frame work, and needs no terminal failure write.
A crash before locator publication has no authoritative job artifact. The exact
control-artifact table either publishes a complete valid locator temporary or
removes a non-framed one and leaves the job absent; no payload extent may exist
yet. This ordering
deliberately does not claim that payload extents existed before locator
publication. If migration produces terminal completed/failed settings, reopen
validation permits removal of the unused terminal extent/descriptor; an explicit
new attempt later recreates the same terminal generation/ID through the fixed
settings-transition index. Legacy bytes remain unchanged.

For an in-progress job, `job_terminal_settings` remains a target-local physical
extent until one durable completed/failed transition consumes it. Depth,
canonicalization, stereo, FFmpeg growth, manifests, and cleanup cannot borrow
it. Any standalone nonterminal rewrite publishes its fixed settings index first
and reserves a separate extent while terminal remains. A terminal-to-new-attempt
transition indexes and readies both its rewrite extent and replacement terminal
extent before changing status. Once every indexed transition is committed, the
fixed settings index is removed and the reservation directory synced. Unknown
or mismatched artifacts are never cleaned up by free-space pressure.

`SettingsArtifactTransactionV1` is the only schema-5 creation or mutation API.
Every caller uses the eight-MiB final-control budget and six-MiB settings phase
peak. Under the job-writer lock it performs exactly:

1. Resolve/reopen the applicable bootstrap or lifecycle-terminal JobControl
   owner, standalone index, or final index; require every entry for this
   transition ready. Initial creation
   requires locator present and its exact reserved fixed-target placeholder.
   Migration opens only the locator's frozen legacy component and requires the
   same placeholder. Later transitions open only canonical
   `processing-settings-v5.json`.
2. Strict-parse the bounded current artifact. Bootstrap and standalone rewrites
   validate and use only their durable complete target payload. An internal
   lifecycle-terminal or final-index transition derives its closed target from
   durable settings and applicable failure/finalization evidence, captures its
   `utc-usec`, increments revision where required, preserves every other field,
   and recomputes `content_fingerprint`.
   Stream the selected canonical bytes into the authenticated target-local extent
   without changing its full logical length.
3. File-sync/reopen the same identity, strict-parse and byte/hash-validate the
   intended prefix, truncate/reopen to release `A`, then perform the declared
   placeholder or existing-target same-parent atomic replacement above.
4. Sync the parent and reopen the final path without following a link. Require
   the expected ordinary identity, byte count, raw SHA-256, canonical
   re-encoding, complete typed object, and transition-specific fields before
   retiring the descriptor/index entry.

Temporary write, flush, fsync, truncate, allocation-release check, replace,
directory sync, reopen, or validation failure is typed and never reported as a
commit. POSIX uses parent-directory `fsync`. Windows uses write-through
`ReplaceFileW`, or `MoveFileExW` with replace/write-through for creation, plus
`FlushFileBuffers` on the reopened target. Both adapters must pass the full-disk
and crash/reopen gates. Existing direct-overwrite settings helpers, pretty schema-
5 serialization, malformed minimal fallback, and an unindexed temporary are
forbidden.

For bootstrap and standalone settings, semantic intent exists before the first
payload-extent byte and a partial extent is first wholly zeroed, then rewritten
from that exact durable object. For producer, cleanup-pending, and terminal
transitions whose
outcome is not knowable when the final index is created, an in-memory timestamp
is provisional until the complete-full frame has been file-synced,
reopen-validated, and self/raw-hash checked. That exact complete-full frame is
then the durable semantic intent: recovery must publish those exact bytes. A
crash with a partial frame has committed no semantic intent; recovery first
restores the all-zero full extent and may then derive the still-applicable
internal transition again with a newly captured time. A replayed terminal
transition takes `attempt_started_at` only from the durable current settings and
computes both `processing_duration_ms` and `processing_time_seconds` through
`attempt_duration_ms_v1` and `duration_seconds_from_ms_v1`; process uptime,
`created_at`, `last_updated_at`, caller timing, and a prior invocation's clock
sample are forbidden substitutes. It never needs caller values. This distinction
prevents a partial write from inventing a committed timestamp, prevents old
longer tails from contaminating a shorter retry, and retains exact replay for
user-supplied changes.

Finalization schedule remains exact. Producer settings, when needed, commit
immediately before first FFmpeg launch. After both manifests are durable,
`keep_intermediates=false` publishes every indexed prune marker, consumes the
indexed diagnostics-metadata extent to commit `payload_pruned`, commits one
cleanup-pending transition, then performs
authorized cleanup. One final transaction commits terminal cleanup/status/time/
duration/runtime together.
Keeping intermediates writes `not_requested` in that final transaction and has
no pending write. Completion is not reported until terminal settings are synced
and reopen-validated. A failed terminal commit preserves authenticated media and
manifests and resumes only the missing declared transition.

### JobControl-owned Extent Reconciliation V1

`reconcile_job_control_owned_extents()` is the sole interpreter for JobControl's
two long-lived entries after bootstrap is committed and neither fixed index is
active. A present fixed index which imports `job_terminal_settings` has priority
and this reconciler does not open, reset, publish, or retire that entry in
parallel. The reconciler opens only JobControl's exact source, target, descriptor,
and descriptor-temporary paths; it never scans for substitutes. A missing final
descriptor may be completed through its deterministic temporary only when the
declared source extent is present, has the exact parent/type/link/length/
allocation shape, and its payload frame is valid for the state below. A missing
unconsumed source after committed bootstrap is not recreated from ambient free
space.
Every committed descriptor encountered here first uses the phase discriminator
above. In particular, the complete-short and target rows never run the full-
length readiness adapter merely because JobControl still owns the descriptor.

For the `initial_settings` or `migration_settings` entry, the closed states are:

| Canonical settings state | Residual entry state | Action |
|---|---|---|
| absent or the exact declared zero-length placeholder | any | not reachable here; only `locator_only` bootstrap may interpret it |
| exact `bootstrap_settings_intent` revision-0 target | source absent and descriptor absent, with at most its exact safe descriptor temporary | bootstrap entry is consumed; retire that temporary if present |
| exact `bootstrap_settings_intent` revision-0 target | descriptor binds that target's current reserved identity | target commit is durable; retire the descriptor and its exact safe temporary |
| legal matching schema-5 revision greater than zero | source absent, with or without an exact JobControl-bound residual descriptor/temporary | initial publication is historical proof for every later revision; retire only those exact residual control files |
| any committed settings | initial/migration source extent still present, two physical locations, wrong identity, or nonmatching bytes | `ReservationConflictError` |
| anything else | any | `SettingsArtifactConflictError` |

The historical-revision rule never deletes a payload path: a successful initial
rename made the source absent, and a later settings replacement may already have
unlinked its old physical identity. It removes only the still-authenticated
descriptor and descriptor temporary, allowing bootstrap-retirement crashes to
converge without demanding an identity which a later revision legitimately
replaced.

Let `T` be the `job_terminal_settings` entry and `r` the current settings
revision. Its closed states are:

| Current settings | Exact `T` state | Action |
|---|---|---|
| `in_progress` revision `r` | source is full-length zero | validate or finish its JobControl-bound descriptor and retain the reserve |
| `in_progress` revision `r` | source is full-length partial | restore the complete zero frame and retain it; no failed/completed intent exists |
| `in_progress` revision `r` | source is complete-full or complete-short and is the unique legal `completed` or `failed` revision `r+1` | publish those exact semantic bytes through `SettingsArtifactTransactionV1` |
| `in_progress` revision `r` | source is a framed object with another revision/status/delta, an invalid short file, or descriptor/identity mismatch | `ReservationConflictError` |
| `completed` or `failed` | target has the descriptor's reserved identity, source is absent | target commit is durable; retire the descriptor and its exact safe temporary before any attempt restart |
| `completed` or `failed` | source and descriptor are both absent, with at most its exact safe descriptor temporary | terminal entry was consumed; retire that temporary if present |
| exact terminal revision-0 migration target equal to `bootstrap_settings_intent` | unused source is full-length zero, with a matching descriptor or descriptor temporary | validate the never-consumed reserve, unlink only that source, sync its parent, then retire its control files |
| any other terminal combination | source present, two locations, nonzero unused source, or wrong identity | `ReservationConflictError` |

For the complete terminal cases, transition legality includes exact job/settings
identity, checked `r+1`, preserved immutable fields, closed status/runtime/
cleanup rules, valid self fingerprint, and raw canonical bytes. Reconciliation
does not turn an arbitrary valid JSON object into intent. Descriptor retirement
syncs the reservation directory; extent removal in the unused-migration row
syncs its target parent first. The same state table uniquely covers crashes
after complete-prefix sync, truncate, same-parent rename, target-directory sync,
target reopen, descriptor removal, and immediately before attempt restart.

### Recovery and Audit Entry Order

Every resume, inspection, finalization, and cleanup entry point uses this one
global order under the job-writer lock; a subsystem may not begin at its local
artifact audit:

1. Open the job root without following links and reconcile the exact job-control
   final/temporary pair through Create-new Control Artifact Publication V1.
   Strict-parse and fingerprint-check the authoritative final when present,
   establish its persisted object identity, and capture the current POSIX
   invocation mount or Windows volume identity. If neither authoritative final
   nor complete temporary remains, no schema-5 job is committed: resume/current
   audit reports it absent, while only an explicit new-job or raw schema-1-
   through-4 migration request may run its read-only selector and begin a new
   locator publication. No ordinary artifact audit is reachable first.
2. If the locator is `locator_only`, recover bootstrap from its two entries and
   complete `bootstrap_settings_intent`. Restart this sequence only after the
   initial settings target is durable. No other audit is reachable from this
   state.
3. Open and classify both fixed index final/temporary pairs directly. Apply the
   control-artifact table to the sole candidate; two candidates/finals, a
   malformed present final, or an index/control identity mismatch is a
   transaction conflict and stops here. Absence is not inferred by scanning.
4. Recover a present settings-transition index to commit or safely retain its
   exact declared work, retire it when complete, then restart at step 1.
5. Recover a present final-encoding index in its consumption order, including
   its final-target descriptor plus artifact-specific manifest/video checks, or
   the exact descriptor-absent `retirement-in-progress` row. Retain it if work is
   incomplete; after terminal commit use only the index-last retirement protocol,
   then restart at step 1.
6. Only when bootstrap is complete and neither fixed index is active, run
   `reconcile_job_control_owned_extents()`. If it publishes, resets, removes, or
   retires work, restart at step 1; if it retains a clean zero terminal reserve,
   continue.
7. Only after that reconciliation may the
   implementation audit canonical settings, stages/diagnostics, final media, and
   legacy/current disposition in that order.

An active index owns every exact placeholder, final-video target/target
descriptor, target-local extent, extent descriptor, descriptor temporary, and
final-index sibling video temporary it declares.
Those entries do not contribute a normal
artifact-presence bit, do not enter `V/I/P`, legacy-final classification, stage
completeness, or reusable-payload discovery, and cannot be deleted by ordinary
stage cleanup. Transaction recovery may validate a committed target as evidence
for consuming its entry, but ordinary audit is still deferred until the index is
retired. An unresolved/incomplete transaction returns its typed state and never
falls through to a later audit. Thus a zero-length manifest placeholder cannot
be mistaken for damaged publication evidence.

There is no persisted `auto` stereo-render mode. The resolver receives the
selected renderer device, writes the resolved two-mode value, and therefore
never reinterprets a saved job because hardware changed. The Web UI exposes a
two-value Fast/Quality segmented control and an advanced fill-limit control. CLI options are
`--stereo-render-mode {fast,quality}` and
`--occlusion-fill-max-px 1..32`, both with parser default `None` so omission is
distinguishable from override.

### Typed Renderer API

Append the two new fields to the existing public relative-render settings as
keyword-only fields, preserving its current positional constructor exactly:

```python
@dataclass(frozen=True)
class StereoRenderSettings:
    stereo_strength: float = 2.0
    convergence: float = 0.5
    occlusion_fill: Literal["none", "background"] = "background"
    stereo_render_mode: Literal["fast", "quality"] = field(
        default="fast", kw_only=True
    )
    occlusion_fill_max_px: int = field(default=8, kw_only=True)
```

Existing zero- through three-positional-argument construction is unchanged;
attempting to pass either new field positionally is a `TypeError`. Construction
validates the mode and applies this exact limit check even when Fast will ignore
the value:

```python
if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
    raise ValueError("occlusion_fill_max_px must be a non-boolean integer")
if not 1 <= int(value) <= 32:
    raise ValueError("occlusion_fill_max_px must be within 1..32")
```

`StereoSplatSettings` retains its existing two-field positional shape and is
explicitly Fast-only; it gains neither field.

The file pipeline uses one separate immutable control object:

```python
@dataclass(frozen=True)
class QualityStereoControls:
    occlusion_fill: Literal["none", "background"]
    occlusion_fill_max_px: int
```

It validates `occlusion_fill` against the two exact literals and applies the same
non-boolean integer normalization. Relative Quality derives it directly from
`StereoRenderSettings`; Metric Quality receives one explicitly from its stage
plan. It contains no relative strength/convergence or Fast shift field.

`StereoRenderer.render(frame, canonical, settings=None)` always constructs the
default settings above and therefore remains Fast on CPU and CUDA. Device-aware
default selection belongs only to CLI/Web job resolution, which passes an
explicit resolved settings object. `render()` dispatches on
`stereo_render_mode` before calling `build_relative_geometry`: Fast enters the
unchanged path, while Quality enters the new relative primitive/region compact
path and only the public wrapper materializes the four legacy masks.

`render_geometry(frame, StereoGeometryFrame, StereoSplatSettings)` remains a
public Fast-only API. It rejects any other settings type and never inspects a
mode field. Metric Quality is intentionally file-pipeline-internal through
`MetricStereoPrimitiveInput` and `render_metric_primitives_compact`; this version
adds no public metric-primitive renderer.

## Persistence and Identity Contract

The stereo RGB and diagnostics algorithm identities are:

```text
Fast RGB schema/algorithm: 1 / torch-horizontal-16x-zbuffer-v3
Quality RGB schema/algorithm: 2 / torch-horizontal-16x-rgb-geodesic-repair-v8
Diagnostics schema/algorithm: 1 / stereo-coverage-sidecar-v1
```

Fast output colour and legacy masks remain governed by Revision 5. Its
fingerprint retains the old render-setting shape and ignores the Quality-only
fill limit. Migrated legacy jobs may reuse an existing valid Fast v3 stage.
Resume replaces blanket legacy-schema invalidation with a semantic check for
that exact case.

Quality stage identity is constructed before rendering. It retains every
output-affecting current stage field, including geometry mode, ordered frame
names, render shape, `occlusion_fill`, encoding,
relative or metric projection settings, Quality RGB algorithm identity, and all
three hashes from the Quality content manifest below. Upstream metadata is never
a substitute for payload content identity. Byte-identical source guides and
native geometry with identical projection and repair settings are intentionally
reusable regardless of which upstream execution produced them; no separate
upstream-geometry-provenance field exists. Its repair-policy fields are
conditional but keep one strict key set:

```text
Quality background:
    configured_limit_1080p = resolved integer
    scaled_safe_limit_px = resolved integer
    predicted_gap_policy = "max-four-neighbour-eye-shift-v1"
    local_limit_formula = "min-scaled-safe-predicted-plus2-v1"

Quality none:
    configured_limit_1080p = null
    scaled_safe_limit_px = null
    predicted_gap_policy = null
    local_limit_formula = null
```

Quality none never computes a local limit. The saved user setting remains
available as non-semantic job provenance but does not enter RGB or diagnostics
identity.

### Quality RGB Metadata V2

Quality stores one immutable identity artifact at
`04_left_frames/metadata.json`; the right-eye directory never contains a second
metadata copy. `StereoRgbMetadataV2` has exactly:

```text
schema_version:                 2
algorithm_version:              "torch-horizontal-16x-rgb-geodesic-repair-v8"
geometry_mode:                  "relative" | "metric"
frame_names:                    list[string]
render_shape:                   [H, W]
occlusion_fill:                 "none" | "background"
encoding:                       "uint8_png"
source_guide_fingerprint:       string
native_geometry_fingerprint:    string
quality_input_manifest_sha256:  string
projection:                     RelativeRgbProjection | MetricRgbProjection
repair_policy:                  QualityRgbRepairPolicy
fingerprint:                    string
```

`frame_names` is the positive, unique, source-ordered list of canonical stems
defined below. `H` and `W` are positive integers. All fingerprint/hash strings
are exactly 64 lowercase hexadecimal characters. `RelativeRgbProjection` has
exactly:

```text
kind:             "relative"
stereo_strength:  binary64
convergence:      binary64
```

`MetricRgbProjection` has exactly:

```text
kind:                              "metric"
projection_algorithm_version:      "crop-aware-metric-pinhole-v2"
source_width:                       integer
retained_crop_width:                integer
center_crop_algorithm_version:      "integer-center-crop-v1"
sample_aspect_ratio:                "1:1"
virtual_baseline_mm:                binary64
requested_convergence_distance:     "auto" | binary64
effective_convergence_distance_m:   binary64
max_disparity_percent:              binary64
```

Both widths are positive, every numeric value is finite, and the existing
metric projection constraints still apply. `QualityRgbRepairPolicy` always has
the same four keys:

```text
configured_limit_1080p:  integer | null
scaled_safe_limit_px:    integer | null
predicted_gap_policy:    "max-four-neighbour-eye-shift-v1" | null
local_limit_formula:     "min-scaled-safe-predicted-plus2-v1" | null
```

Background Quality requires the two positive integers and the two exact strings;
Quality none requires all four values null. Missing/extra keys, booleans in an
integer field, non-finite binary64 values, or crossed projection variants are
invalid.

The self-fingerprint is
`SHA256(canonical_json(object without fingerprint))`. The file bytes are exactly
`canonical_json(the complete object)` using the sorted-key ASCII JSON contract
below, with no indentation, BOM, or trailing LF. Publish it atomically after the
Quality input content manifest and before diagnostics `building` metadata. It
has no `building`/`complete` field and never claims frame completeness by itself.

Quality reuse requires byte-canonical metadata, a valid self-fingerprint, exact
expected settings/projection/repair fields, exact content-manifest raw hash and
derived hashes, and valid current RGB plus diagnostics frame transactions for
every source item. Every Quality `FrameManifest.rgb_stage_fingerprint` and
`04_stereo_diagnostics/metadata.json.rgb_stage_fingerprint` must equal this one
metadata fingerprint; no independently reconstructed equivalent is accepted.
An orphan metadata artifact with no matching diagnostics state claims nothing.

Legacy Fast RGB metadata schema 1 at the same path is the sole exception to the
global stem rule: to preserve existing v3 metadata and cache bytes, its
`frame_names` entries remain complete `.png` filenames and its current
mode-dependent key set, serialization, fingerprint, device field, and reuse
validator remain unchanged. Fast never writes schema 2; Quality never writes
schema 1. Every new schema in this document uses canonical stems.

### Quality Input Content Manifest

Before a Quality reuse decision or stage mutation, stream the prospective object
and compare it with any retained
`04_stereo_diagnostics/quality_input_manifest.json`. It has exactly:

```text
schema_version:                       1
algorithm_version:                    "quality-input-content-v4"
geometry_mode:                        "relative" | "metric"
frame_names:                          list[string]
guides:                               list[QualityGuideIdentity]
geometry:                             list[RelativeGeometryIdentity] |
                                      list[MetricGeometryIdentity]
relative_encoding_scale_float32_bits: string | null
source_guide_fingerprint:             string
native_geometry_fingerprint:          string
fingerprint:                          string
```

All three lists have the same positive source-order length and exact frame-name
mapping. Throughout this specification, the canonical stem for numeric index
`k` in `0..U64_MAX` is `"frame_" + decimal(k).zfill(max(6,
len(decimal(k))))`. Thus 89 is `frame_000089` and 1000000 is
`frame_1000000`; redundant leading zeroes, an extension, or any other spelling
is noncanonical. Except for the explicitly frozen Fast RGB metadata schema 1,
every `frame_name` field and frame-name list contains this stem, never a filename.

`QualityGuideIdentity` has exactly `frame_name`, output-root-relative
`relative_path`, full-file `sha256`, `byte_count`, and `png_header`. The header
has exact integer `width`, `height`, `bit_depth`, and `color_type`; every guide
must be RGB PNG color type 2, bit depth 8, and render shape `[H,W]` before
OpenCV's fixed `IMREAD_COLOR` BGR8 decode contract is allowed. Its payload
filename is exactly `frame_name + ".png"`.

`RelativeGeometryIdentity` has exactly `frame_name`, `relative_path`, `sha256`,
`byte_count`, `payload_kind="canonical_png"`, and `png_header`. Every canonical
payload is a grayscale PNG color type 0, bit depth 16, at the one validated
native shape. `relative_encoding_scale_float32_bits` is its positive finite
IEEE-754 binary32 bit pattern as exactly eight lowercase hexadecimal digits;
the relative payload filename is exactly `frame_name + ".png"`.

`MetricGeometryIdentity` has exactly `frame_name`, `relative_path`, `sha256`,
`byte_count`, `payload_kind="metric_npz"`, `native_shape`,
`payload_contract="metric-npz-v1"`,
`archive_form="owned-zip32-v1" |
"numpy-force-zip64-local-u32-v1" |
"numpy-force-zip64-local-sentinel-v1"`,
`inverse_depth_dtype="float32"`, `valid_dtype="bool"`,
`focal_x_normalized_float32_bits`, and
`members=["inverse_depth.npy","valid.npy","focal_x_normalized.npy"]` in that
order. The metric payload filename is exactly `frame_name + ".npz"`. The parsed
arrays/scalar must satisfy the existing metric store contract; the focal bit
pattern is exactly eight lowercase hexadecimal digits. Relative
mode requires only relative entries and a non-null scale; metric mode requires
only metric entries and a null scale.

### Metric NPZ Payload Contract V1

`MetricNpzPayloadContractV1` is the sole parser and writer boundary for metric
Quality content audit. It does not delegate structural acceptance to
`zipfile`, `np.load`, or an installed NumPy version. It accepts exactly the three
`archive_form` values named above:

- schema-5 `_atomic_save_npz` uses the project-owned `owned-zip32-v1` writer;
- an existing payload written by the pre-schema-5
  `np.savez_compressed(...)` path uses exactly one of the two named historical
  forms according to its CPython `zipfile` behavior.

All three forms contain exactly three file records and three central-directory
records in this order, with these case-sensitive ASCII names:

```text
inverse_depth.npy
valid.npy
focal_x_normalized.npy
```

For every record, general-purpose flags are exactly zero and compression method
is exactly 8 (raw DEFLATE). Encryption, patched-data/data-descriptor bit 3,
UTF-8 bit 11, directories, aliases, duplicate names, prefixes, and path
separators are rejected. Local and central names, method, flags, DOS time/date,
CRC-32, and resolved compressed/uncompressed sizes must agree. DOS time/date are
exactly `00:00:00 1980-01-01`. The DEFLATE decoder must consume exactly the
declared compressed extent, reach end-of-stream once with no unused/trailing
compressed byte, produce exactly the declared and header-derived uncompressed
length, and match the central/local CRC-32. Checked offsets require the first
local header at byte zero, contiguous local header/data extents, the central
directory immediately after the third data extent, and EOCD immediately after
the third central record; prepended and trailing bytes are invalid.

`owned-zip32-v1` has version-needed 20, ordinary non-sentinel 32-bit sizes and
offsets, and zero-length local and central extra fields. Its version-made-by low
byte is 20; the platform byte is 0 or 3, internal attributes are zero, and
external attributes are exactly zero or `0x01800000`. The schema-5 writer emits
platform byte 0 and both attribute words zero. Every size, local offset, central
offset, and archive length must be at most `0xfffffffe`; overflow raises
`MetricNpzSizeError` before publication rather than selecting ZIP64.

The historical forms share these facts: every actual compressed/uncompressed
member size and local-header offset is at most `0xfffffffe`; the central
directory stores those ordinary uint32 values and has no extra field; DOS
metadata, flags, method, order, CRC, and comments follow the common rules above.
Version-made-by platform byte is 0 or 3, internal attributes are zero, and
external attributes are exactly `0x01800000`, matching the pre-schema-5 NumPy/
CPython writer. Each local header has exactly one extra field: header ID
`0x0001`, payload length 16, then little-endian uint64 actual uncompressed size
followed by little-endian uint64 actual compressed size. No other local extra
exists.

`numpy-force-zip64-local-u32-v1` is observed in CPython 3.10.11 and 3.11.0.
Local and central version-needed plus version-made-by low byte are 20. Each
local compressed/uncompressed size field contains its ordinary actual uint32
value; the two uint64 ZIP64-extra values must duplicate those fields exactly.

`numpy-force-zip64-local-sentinel-v1` is observed in CPython 3.11.9/3.11.14 and
3.12.1. Local and central version-needed plus version-made-by low byte are 45.
Each local compressed/uncompressed size field is exactly `0xffffffff`; the
local ZIP64 extra carries the actual values which must equal the central
ordinary uint32 values. The local CRC remains the actual CRC. It still has no
central ZIP64 extra or ZIP64 EOCD/locator.

These are two explicit historical layouts, not a general ZIP64 mode. Crossing
their version/size rules, a ZIP64 EOCD/locator, a central ZIP64 extra, sentinel
central sizes/offsets, reordered/partial ZIP64 values, multiple extras, or any
other extra-field ID is rejected. Archive classification never branches on
`sys.version_info`: CPython changed this behavior within the 3.11 patch line.
The release matrix must include the earliest and latest project-admitted patch
release in CPython 3.10, 3.11, and 3.12, every actual CI/distribution runtime,
and every compatible lowest/highest supported NumPy pin. A generated fixture
outside these two grammars blocks release of that exact dependency cell;
production metadata may not continue advertising it as supported until either
a separately named grammar is reviewed or the public Python/NumPy range is
narrowed.

All three forms use one single-disk ordinary EOCD with disk numbers zero, three
entries on disk and in total, exact checked central-directory size/offset, and
an empty archive comment. Per-member comments are empty. No digital signature,
archive-extra record, spanning marker, encryption header, or record other than
the six records and EOCD above is allowed.

Each inflated member is exactly one NPY v1.0 object: magic bytes
`93 4e 55 4d 50 59`, version bytes `01 00`, and a little-endian uint16 header
length no greater than 256. Preamble plus header length is a multiple of 64.
The header is ASCII, ends in exactly one LF preceded only by its required space
padding, and its unpadded bytes use this closed grammar with canonical positive
decimal `H` and `W`:

```text
inverse_depth.npy:
{'descr': '<f4', 'fortran_order': False, 'shape': (H, W), }

valid.npy:
{'descr': '|b1', 'fortran_order': False, 'shape': (H, W), }

focal_x_normalized.npy:
{'descr': '<f4', 'fortran_order': False, 'shape': (), }
```

There is exactly one ASCII space after each colon and comma as shown, no leading
zero in a dimension, and no other key, quote, token, escape, or whitespace.
The closing brace is followed by the minimum positive count of ASCII spaces
which makes preamble plus header a multiple of 64, then LF. This rejects NPY
v2/v3, native/big-endian float markers, Fortran order, object, structured,
subarray, and aligned dtypes without a general Python-literal parser. Member
data begins immediately after that header and ends at member EOF. Inverse and
focal bytes are little-endian IEEE-754 binary32; valid data is exactly one byte
per element and each byte is 0 or 1. Checked expected raw sizes are
`4*H*W`, `H*W`, and 4 respectively.

The owned writer emits those exact NPY header bytes, the fixed archive metadata,
and DEFLATE level 9. The DEFLATE bitstream itself is not a semantic identity:
any standards-conforming stream which passes the closed record grammar is
accepted, while the complete raw archive SHA-256 still binds the concrete
bytes. Before implementation, fixtures from the current `_atomic_save_npz()`
and every supported Python/NumPy lock-matrix cell must classify as exactly one
named historical form; an output outside both blocks that dependency
combination rather than widening the parser implicitly.

The owned writer has this additional-host-workspace contract:

```text
METRIC_NPZ_WRITE_INPUT_CHUNK_BYTES   = 512 KiB
METRIC_NPZ_WRITE_STREAM_BYTES        = 1 MiB
METRIC_NPZ_WRITE_DEFLATE_STATE_BYTES = 512 KiB
METRIC_NPZ_WRITE_RECORD_BYTES        = 4 KiB
metric_npz_write_additional_peak =
    METRIC_NPZ_WRITE_STREAM_BYTES
  + METRIC_NPZ_WRITE_DEFLATE_STATE_BYTES
  + METRIC_NPZ_WRITE_RECORD_BYTES
```

The input chunk is a non-owning contiguous memoryview into the already-owned
`MetricGeometryFrame`; it is not another 512-KiB allocation. One raw-DEFLATE
compressor consumes the exact NPY header then row-major array bytes in chunks no
larger than that value, incrementally updates CRC-32 and checked uint64 raw/
compressed counts, and writes every returned block immediately to one sibling
temporary. It performs no per-chunk sync/full flush and issues exactly one final
finish for each member. A returned block larger than
`METRIC_NPZ_WRITE_STREAM_BYTES`, native
deflater residency above its attested state allowance, count/offset overflow,
short write, or compression error is fatal. No complete raw or compressed member
copy, second metric array, `BytesIO`, or `np.savez_compressed` call is allowed.

Disk preflight uses checked uint64 arithmetic and this fixed conservative bound,
where `n` includes the exact NPY preamble/header and array data, `name_bytes` is
the ASCII member-name length, and `alloc`/`A` are the filesystem functions below:

```text
raw_deflate_bound(n) = n + (n >> 12) + (n >> 14) + (n >> 25) + 13
metric_npz_temp_logical_bound =
    sum(30 + name_bytes[j] + raw_deflate_bound(npy_raw_bytes[j])
        + 46 + name_bytes[j] for j in the three fixed members)
    + 22
metric_npz_temp_disk_reserve = alloc(metric_npz_temp_logical_bound) + A
```

Every `npy_raw_bytes[j]`, shift, addition, and archive-field conversion is
checked before mutation; a bound above the owned ZIP32 limits is
`MetricNpzSizeError`. The reserve is required on the target volume before
opening the temporary and coexists with every allocated extent of an old
destination. It covers the one temporary directory entry and may not claim the
old file as reclaimable.

For each member, write one owned-form local header with fixed-size placeholders,
stream its data, seek back to patch actual CRC and ordinary uint32 sizes, then
seek to the checked end before starting the next member. After all three members
are patched, write the bounded central directory and EOCD from scalar records,
flush/fsync the temporary, atomically replace the destination, and durably sync
its parent where supported. Any exception before replace closes and removes only
the temporary and leaves the old payload byte-exact; an error after replace is
reported through the metric-stage transaction and never fabricates complete
metadata. Task 0 measures the complete writer RSS, including native zlib state
and Python wrappers, against the additional peak rather than treating the
constants as payload sizes only.

Both float-bit strings use a numeric bit cast: interpret the IEEE-754 binary32
value's 32-bit pattern as an unsigned integer and format that integer with
exactly eight lowercase hexadecimal digits. Host memory byte order never enters
the string. For example, the value with bits `0x3f800001` is encoded as
`"3f800001"`; its little-endian memory bytes `01 00 80 3f` must not be emitted
as `"0100803f"`.

Every path is opened as a regular file beneath the acquired job root without
following links or reparse points. Hash the complete raw PNG or NPZ bytes, not a
sample, decoded-pixel subset, size/mtime tuple, upstream manifest, or parsed
array summary. Define:

```text
source_guide_fingerprint = SHA256(canonical_json(guides))
native_geometry_fingerprint = SHA256(canonical_json({
    "geometry_mode": geometry_mode,
    "relative_encoding_scale_float32_bits":
        relative_encoding_scale_float32_bits,
    "geometry": geometry,
}))
```

The object self-fingerprint then binds both derived hashes. The Quality RGB
identity includes `source_guide_fingerprint`, `native_geometry_fingerprint`, and
the raw SHA-256 of the complete canonical manifest bytes. Diagnostics metadata
stores those same three hashes. The embedded self-fingerprint is validated but
is not a fourth stage-identity field. Fast v3 neither writes nor consumes this
manifest and retains its existing identity unchanged.

The manifest is streamed without an `O(N)` Python object tree. Header parsing and
full hashing use the same non-following open handle for each file. Before the
content audit, a bounded header-only planning replay validates every PNG header
and every strict ZIP/NPY header, derives each `G_i`, and evaluates the host phase
bounds below. It retains no decoded frame and must finish before any stage
mutation.

Relative payload validation streams the complete guide and canonical-geometry
bytes through `QUALITY_CONTENT_STREAM_BYTES`; it never decodes a full image.
Metric validation uses one project-internal `OwnedMetricAuditFrame`, not
`MetricGeometryStore.validate_payloads`, `np.load`, the public
`MetricGeometryFrame` constructor, or its defensive-copy `__post_init__` path.
For one NPZ, the validator:

1. hashes all raw archive bytes, seeks the same contained regular-file handle
   back to zero, and accepts exactly one of the three complete
   `MetricNpzPayloadContractV1` archive forms;
2. parses the three closed-grammar NPY-v1 headers through the fixed stream
   buffer and requires the exact member order, shapes, dtypes, scalar focal
   shape, and C-order contract;
3. allocates exactly one C-contiguous float32 inverse array (`4*G_i`) and one
   C-contiguous bool validity array (`G_i`), then inflates each member directly
   into its final destination without an intermediate member array; and
4. validates row-major fixed-buffer chunks without constructing boolean index
   arrays or whole-frame predicates: every inverse is finite, invalid inverse
   bits are positive zero, valid inverse values are positive, and the float32
   focal scalar is finite and positive.

The decompressor, NPY parser, value-check scratch, and hashing buffer share the
single fixed stream allocation. The audit frame is released before opening the
next payload and is never transferred into render work. A short/overlong member,
CRC mismatch, malformed header/value, decompression error, or allocation failure
is fatal; it cannot select a different decoder, skip value checks, or trigger a
Fast fallback. Allocation failure reports the applicable
`QualityHostBudgetError` phase, while content errors report the offending
contained path and member.

A matching valid retained manifest is reused without rewrite. Otherwise, after
disk preflight, publish the replacement atomically before Quality RGB metadata
and `building` diagnostics metadata, both of which reference its raw hash.
Immediately before every Quality aggregate consolidation, after all P rendering
and R migration have ended and all worker/thread bytes are gone, release all
other phase-owned buffers, reopen every listed input, and reproduce the complete
manifest bytes and all three hashes with this same bounded audit. Any mismatch
fails the stage while it is still `building`; consolidation never starts. A
complete `P=0,R=0` reuse still performs the initial full content audit even
though it creates no renderer, worker, migration, or consolidation phase. A
crash leaving only an orphan manifest or RGB metadata claims no stage. Missing,
noncanonical, stale, or corrupt Quality content evidence forces `P=N` and normal
downstream invalidation; a mask-only migration is allowed only after this
validation and does not rewrite the manifest. The manifest remains with
historical diagnostics after `payload_pruned` and is a forbidden cleanup target.

`renderer_device_type` is not a Quality semantic identity field: CPU and CUDA
must produce the same bytes and may reuse one another's valid Quality stage.
Fast retains its existing device-bearing metadata solely for v3 cache
compatibility. Hardware, driver, and library versions belong in benchmark or
runtime execution provenance outside the RGB and diagnostics fingerprints.

For background Quality, `scaled_safe_limit_px` is stage-constant because render
shape is stage-constant. Frame-dependent `predicted_gap_px` and `local_limit_px`
are not stage fields; both eyes record them in frame stats and the frame manifest
repeats those values as a transaction-level assertion. They are already
determined by the upstream geometry fingerprint, guide, settings, and policy
versions. Quality none neither computes nor records any of these limits.

Changing mode, background-Quality limit, or Quality identity invalidates stage
04 and tracked downstream stages, but not source, depth, canonical disparity,
stabilization, or metric geometry. A limit-only change invalidates neither Fast
nor Quality none and does not rewrite their diagnostics or downstream stages.

Legacy reused Fast output has no reconstructable lane diagnostics and reports
`legacy_fast_unavailable`; masks or counts must not be fabricated from final
RGB. Every newly rendered Fast or Quality frame carries diagnostics.

## End-to-end Architecture

Quality geometry is constructed once on the host. Each eye is then processed
independently and sequentially through three phases. Left-eye compact planning
state is released before right-eye analysis begins; only the completed left
output and diagnostics remain resident.

```text
Quality primitive geometry + source-region labels
    -> Pass A: banded visibility analysis
    -> global compact repair planning on host
    -> Pass B: banded visibility replay and final 16-lane reduction
    -> eye RGB + compact diagnostics
```

`occlusion_fill=none` needs no repair plan and therefore uses one banded
visibility pass. It still emits compact coverage diagnostics. Fast retains the
Revision 5 single-pass renderer exactly.

### Eye-offset Builders

Quality never calls either current helper which returns both full-frame eye maps
and materializes full-frame float64 expressions. It allocates one contiguous
`int32[H,W]` output and one `float64[W]` row scratch. Relative and geometry paths
use different builders because their IEEE-754 operation order is not
interchangeable:

```python
def build_relative_eye_offsets_into(
    near_score: np.ndarray,
    *,
    stereo_strength: float,
    convergence: float,
    eye: Literal["left", "right"],
    output_int32: np.ndarray,
    row_float64_scratch: np.ndarray,
) -> None:
    ...

def build_geometry_eye_offsets_into(
    total_disparity_fraction: np.ndarray,
    *,
    eye: Literal["left", "right"],
    output_int32: np.ndarray,
    row_float64_scratch: np.ndarray,
) -> None:
    ...
```

Rows and columns are visited in ascending order. Throughout both formulas and
low-resolution region classification, `W` is always the final render width,
never the native primitive width. Relative first computes the row-constant scale
exactly as current Fast:

```text
scale = float64(float64(W) * float64(stereo_strength))
scale = float64(scale / float64(200.0))
shift = float64(float64(float32(near_score)) - float64(convergence))
shift = float64(shift * scale)
shift = float64(shift * float64(16.0))
left  = ceil(float64( shift - float64(0.5)))
right = ceil(float64(-shift - float64(0.5)))
```

Geometry paths preserve the current metric/full-geometry order for each source
binary64 value `f`:

```text
shift = float64(f * float64(W))
shift = float64(shift * float64(0.5))
shift = float64(shift * float64(16.0))
left  = ceil(float64( shift - float64(0.5)))
right = ceil(float64(-shift - float64(0.5)))
```

Relative Fast and Relative Quality use only the relative builder. Metric Fast
and Metric Quality use only the geometry builder. Relative source-region
displacement, four-neighbour q-jump, `predicted_gap_px`, and final splat offsets
all derive from the same relative `near_score`, strength, convergence, and
written operation order; `total_disparity_fraction` is not an alternate relative
offset authority. Low-resolution classification invokes the same scalar sequence
with final render `W`; it does not fake a native-size output map merely to call
the full-resolution builder.

Select only the requested eye, range-check before narrowing, and store int32. A
row-vectorized implementation is allowed only when it matches its scalar oracle
exactly. The map is immutable through the current eye's analysis/repair passes
and every CUDA OOM retry; it is never rebuilt after an OOM. Release it only after
that eye commits in memory, then reuse the same allocation for the other eye. The
row scratch is included in fixed runtime overhead. Host allocation or
construction failure is fatal rather than a band-height retry.

The exact relative fixture `W=1920`, `stereo_strength=2.0`,
`convergence=0.5`, and float32 `near_score=0.9052734375` must produce fine shift
`124.5` and left offset `124`. The reassociated total-fraction result `125` is a
test failure. One-ULP fixtures straddle both signs and every half-lane boundary.

### Native File-pipeline Input Boundary

The current file pipeline expands metric primitives inside decoder workers and
copies completed arrays again in `StereoGeometryFrame.__post_init__`. Fast and
Quality must both enter before that unbudgeted construction. Decoder workers
load only source RGB and one owned native primitive object:

```python
@dataclass(frozen=True)
class RelativeStereoPrimitiveInput:
    encoded_canonical: np.ndarray  # owned contiguous uint16 [Gh,Gw]
    encoding_scale: np.float32

@dataclass(frozen=True)
class MetricStereoPrimitiveInput:
    metric: MetricGeometryFrame
    virtual_baseline_mm: np.float64
    convergence_distance_m: np.float64
    max_disparity_percent: np.float64
    retained_crop_width: int
```

`metric` supplies native float32 `inverse_depth`, bool `valid`, and float32
`focal_x_normalized`. The store loader allocates those two arrays exactly once,
validates them in place, marks them read-only, and uses an internal owned factory
which performs no `MetricGeometryFrame` copy. The public constructor retains its
existing defensive-copy behavior. Relative decode likewise returns the uint16
PNG allocation directly rather than casting it to float32 in a worker. The
render shape comes from the uint8 source frame.

The internal decode/render union becomes:

```text
RelativeStereoPrimitiveInput | MetricStereoPrimitiveInput
```

- relative Fast and Quality receive `RelativeStereoPrimitiveInput` and convert
  it only on the serial render thread;
- metric Fast constructs the exact current bilinear geometry on that serial
  thread through a no-copy owned builder, then calls the public-compatible Fast
  `render_geometry` core;
- metric Quality sends `MetricStereoPrimitiveInput` with no precomputed stats to
  `render_metric_primitives_compact(frame, primitives, controls)`, which performs
  region solving, one-sided primitive resampling, projection, clamping, and
  final stats in that order;
- `render_geometry(frame, StereoGeometryFrame, StereoSplatSettings)` is
  structurally Fast-only and rejects a `StereoRenderSettings` object;
- metric Quality never silently accepts a render-size `StereoGeometryFrame`.

The metric Quality entry point is exactly:

```python
def render_metric_primitives_compact(
    frame: np.ndarray,
    primitives: MetricStereoPrimitiveInput,
    controls: QualityStereoControls,
) -> CompactStereoRenderResult:
    ...
```

The file planner is a discriminated union, not a `geometry_mode` plus runtime
settings-type cross product:

```text
FastRelativePlan    {kind="fast_relative",    primitives=relative,
                     settings=StereoRenderSettings(stereo_render_mode="fast")}
QualityRelativePlan {kind="quality_relative", primitives=relative,
                     settings=StereoRenderSettings(stereo_render_mode="quality")}
FastMetricPlan      {kind="fast_metric",      primitives=metric,
                     settings=StereoSplatSettings}
QualityMetricPlan   {kind="quality_metric",   primitives=metric, controls}

StereoStagePlan = FastRelativePlan | QualityRelativePlan |
                  FastMetricPlan | QualityMetricPlan
```

Each variant owns one matching decode and render function; an impossible
variant/input/settings combination is rejected when the plan is built, not
inferred in the frame loop. Relative Quality derives its controls exactly once
from its validated settings at dispatch; no second stored controls object can
disagree. Metric Quality stores the only explicit `QualityStereoControls`.

The compact Quality result carries its final `MetricProjectionStats` to the
writer. `_DecodeFrame`, `_DecodedMessage`, `_WriteItem`, tests, and mocks update
their unions accordingly. This keeps the old public Fast entry point compatible
while making it impossible to implement only relative Quality correctly.

### Source-region Identity and Fine-lane Components

Source geometry regions replace a retained global graph over fine-grid winners,
which would conflict with banded rendering. Fine-grid connectivity is retained
only for sparse unresolved segment records.

Source regions are constructed globally before either eye renders. Every valid
fine-lane winner can recover its full-frame source index from the existing
packed visibility key, then gather its uint32 source-region ID. A repair donor
must carry the same region ID as the far boundary selected for that unresolved
lane run. This prevents crossing a source depth edge without retaining a full
eye fine grid or merging an unbounded winner graph across render bands.

Target repair components are global and use one segment record as one graph
node. Two nodes may union only when they have the same source-region ID and are
fine-grid adjacent:

- horizontal adjacency requires equal rows, `left_column + 1 == right_column`,
  bit 15 in the left record, and bit 0 in the right record; a row's last pixel
  never connects to the next row's first pixel;
- vertical adjacency between rows `y` and `y+1` requires a nonzero bitwise AND
  of their lane masks;
- records inside one output pixel union only when their masks contain adjacent
  lane bits, exactly
  `(((mask_a << 1) & mask_b) | ((mask_b << 1) & mask_a)) & 0xffff != 0`;
  separated masks do not union merely because region IDs match.

Connectivity is evaluated after all Pass A records are sorted, so components
cross band boundaries and may span the full image height without retaining a
dense fine grid.

## Primitive and Derived Geometry Contract

### Unique Quality Geometry Construction Order

Every relative and metric Quality implementation executes these ten steps in
this order; no later section may reorder them:

1. Construct canonical low-resolution region IDs.
2. Construct the complete Fast bilinear primitive/derived baseline.
3. Nearest-label upsample the low-resolution IDs into the initial
   full-resolution region map.
4. Run the RGB-guided geodesic solver and make its result the immutable final
   full-resolution region map.
5. Release Dijkstra distance, indexed heap, heap-position, band, erosion,
   skeleton, and other morphology scratch.
6. Construct the exact per-region retained-zero k-d index over native samples.
7. Select each one-sided interpolation region from the final map.
8. Complete the relative primitive resample or both metric primitive resamples.
9. Release the retained-zero index.
10. Derive every dependent geometry field and projection/clamp statistic in its
    specified arithmetic order.

For every output pixel evaluated by one-sided interpolation, the invariant is
exactly:

```text
selected_region[y,x] == final_geodesic_region_map[y,x]
```

The solver's pre-geodesic nearest-label map is never an interpolation selector.
Step 6 cannot allocate any retained-zero index byte until step 5 has released
all geodesic/morphology scratch. Conversely, the final region map remains live
through step 8. The host contract therefore has two deliberately nonoverlapping
peaks, `quality_region_peak` and
`quality_geometry_build_with_nearest_index_peak`; summing their scratch sets or
allowing them to coexist is an implementation error, not a conservative option.

### Exact Fast Baseline First

After low-resolution region construction, Quality invokes the existing Torch
float32 bilinear helpers with
`align_corners=False` to build a complete Fast baseline. Every output outside a
classified edge band is copied directly from this baseline. Quality must not
reimplement ordinary bilinear coordinates in NumPy and then claim bit identity.

Only primitive resampling inside an edge band changes. Derived fields are
recomputed from the selected primitives in their existing evaluation order.

### Relative Dependency Order

The relative primitive is canonical float32 near score `r`. The compatibility
geometry field is recomputed after one-sided resampling in the current
`build_relative_geometry` order:

```text
total = float64(float64(r) - float64(convergence))
total = float64(total * float64(stereo_strength))
total = float64(total / float64(100.0))
total_disparity_fraction = total
```

Outside edge bands, both `near_score` and `total_disparity_fraction` are copied
from the exact Fast baseline rather than recomputed. This compatibility field is
never multiplied back into a Relative eye offset. Low-resolution region
displacement and full-resolution q-jumps instead stop the exact relative builder
after `shift * scale`, before its fine-lane multiplication, so region splitting,
limit prediction, and splatting cannot disagree by reassociation.

### Metric Dependency Order

Metric Quality preserves this unique order:

```text
primitive valid_weight          = float32(valid)
primitive weighted_inverse      = float32(inverse_depth * valid_weight)
resample both primitives
resized_valid                   = resized_weight >= float32(0.5)
resized_inverse                 = resized_weighted_inverse / resized_weight
                                  where resized_weight > 0
resized_inverse[~resized_valid] = float32(0)
raw_output_fraction             = existing float64 pinhole formula
clamped_output_fraction         = existing max-disparity clamp
near_score                      = resized_inverse
total_disparity_fraction        = existing retained-crop conversion
source_valid                    = all true, preserving the existing
                                  infinite-background policy
clamp statistics                = recomputed from final resized_valid and
                                  raw/clamped fractions
```

Within an edge band, both primitive fields use the same selected-region mask and
the same bilinear spatial weights. Weights are renormalized only across corners
whose region ID matches the selected output region. Derived values are never
interpolated independently. A low-resolution validity change always blocks a
region-graph edge, even when both samples happen to clamp to the same visible
shift.

The metric tests compare Quality output outside edge bands to the exact Fast
baseline, verify the inverse-depth/displacement equation inside bands, and
independently recount clamp statistics.

### Exact One-sided Interpolation Oracle

Inside an edge band, production must match this scalar oracle. Let source shape
be `(SH, SW)`, render shape `(DH, DW)`, and output coordinate `(y, x)`. Every
operation below is IEEE-754 binary64, evaluated in the written order without
contraction or reassociation, until the explicit final cast:

```text
sx = (((float64(x) + 0.5) * float64(SW)) / float64(DW)) - 0.5
sy = (((float64(y) + 0.5) * float64(SH)) / float64(DH)) - 0.5
cx = min(max(sx, 0.0), float64(SW - 1))
cy = min(max(sy, 0.0), float64(SH - 1))
x0 = floor(cx); x1 = min(x0 + 1, SW - 1); wx = cx - float64(x0)
y0 = floor(cy); y1 = min(y0 + 1, SH - 1); wy = cy - float64(y0)
ox = 1.0 - wx; oy = 1.0 - wy
w00 = oy * ox; w01 = oy * wx; w10 = wy * ox; w11 = wy * wx
```

For selected region `r`, let `mij` be binary64 1.0 only when that corner's
canonical region ID is `r`, otherwise 0.0. Compute:

```text
rw00 = w00 * m00; rw01 = w01 * m01
rw10 = w10 * m10; rw11 = w11 * m11
retained = ((rw00 + rw01) + rw10) + rw11
numerator = (((float64(v00) * rw00) + (float64(v01) * rw01))
             + (float64(v10) * rw10)) + (float64(v11) * rw11)
result = float32(numerator / retained)
```

Relative geometry uses this once for its primitive. Metric geometry computes
the weights once, then applies the same already-quantized `rw00..rw11` values
and the same ordered expression separately to `valid_weight` and
`weighted_inverse`. It does not call a reduction helper, `einsum`, or a fused
kernel whose association differs from the oracle. Synthetic tests use an
independent scalar implementation and include values where one-ULP changes
alter the projected lane.

If `retained == 0.0`, compare matching-region low-resolution sample `(iy, ix)`
against the unclipped `(sy, sx)` using binary64 in this order:

```text
dx = sx - float64(ix)
dy = sy - float64(iy)
distance2 = (dx * dx) + (dy * dy)
```

Choose the lowest `distance2`, then lowest `iy`, then lowest `ix`. Production
must disable fused multiply-add for this comparison or use the scalar oracle
path. It never linearly scans all region members per query.
Metric retained-zero handling performs one location query and reads both
primitive values from that selected native sample; it never counts or searches
the same output coordinate twice.

After low-resolution region IDs finalize, production releases union parent,
canonical-key, and rank scratch while retaining only final IDs. It does not yet
build the retained-zero index. First it constructs the Fast baseline, nearest
label map, and final geodesic map, then releases every geodesic and morphology
scratch allocation. Only after that release does it build a separate exact
per-region implicit two-dimensional k-d index. Let `R` be the positive region
count and `G=SH*SW`; require checked `G+1 <= UINT32_MAX` (which also proves
`R+1`) before allocation. The index owns exactly a uint32 `member_index[G]`, uint32
`region_offsets[R+1]`, and fixed 64-entry construction/query stacks inside
runtime overhead. It therefore adds at most `8*G+4` bytes and no pointer/object
per sample.

Build grouping is unique. Count region ID `r` into `region_offsets[r-1]`, prefix
entries `0..R-1` in place to region ends, and set the untouched sentinel
`region_offsets[R]=G`. Visit native samples in reverse row-major order,
pre-decrement `region_offsets[r-1]`, and write the uint32 sample index there. The
result is row-major members with starts in `0..R-1` and final sentinel `G`.
Within each range, reuse the repair index's in-place deterministic
median-of-medians partition: row axis first, alternate row/column by depth,
choose the lower median by `(axis coordinate, other coordinate, sample_index)`,
and use implicit child ranges.

Query retained-zero pixels in output row-major order. Visit the nearer child
first; an axis tie visits the lower child first. Visit the farther child exactly
when the separately rounded squared split-plane distance is less than or equal
to the current best distance, because an equal-distance point may win by
`(iy,ix)`. Candidate comparison is exactly `(distance2,iy,ix)` from the scalar
oracle. A selected region with no indexed member is an internal deterministic
error.

Because every partition uses its lower median, tree height is at most 32 for
`G<=UINT32_MAX`; the fixed 64-entry build and query stacks therefore cannot
overflow. Stack overflow is nevertheless checked and is an internal fatal error,
never permission to recurse or allocate.

Define:

```text
QUALITY_GEOMETRY_NEAREST_VISIT_CAP = 268_435_456
```

Each examined node increments one checked uint64 frame counter. Exceeding the
cap raises `QualityGeometryQueryBudgetError` and discards all partial geometry;
it never selects the current best, changes region, scans linearly, or falls back
to Fast. Allocation/index-build failure is likewise fatal before any frame
transaction commits. Release member/offset storage immediately after all
relative primitive or both metric primitive resamples complete and before any
derived geometry field is calculated. Every query uses the selected region from
the immutable final geodesic map, never the nearest-label initialization.

Frame diagnostics record `indexed_sample_count`, `query_count`,
`visited_nodes_total`, and `visited_nodes_max`; Task 0 additionally records p95
as non-semantic benchmark telemetry. Build work is `O(G log G)`, memory is
`O(G)`, expected query work is `O(log G)`, exact worst-case work is `O(G)`, and
the total examined-node cap is an explicit fatal resource boundary. Scalar
brute force remains the independent correctness oracle, not a production path.

## Deterministic RGB-guided Region Solver

Quality does not use marker-controlled watershed. It uses an integer geodesic
solver with sparse stable seeds.

### Low-resolution Region IDs

For every primitive sample, derive its final one-eye displacement in output
pixels through the selected relative or metric builder order above. Two
four-neighbour samples are disconnected when either condition holds:

- their metric validity differs;
- their absolute one-eye displacement difference is at least `1.0` output
  pixel.

Union every other neighbour pair. Visit low-resolution samples in row-major
order and examine neighbours in up then left order. The canonical component key
is the minimum row-major linear index among all members, independent of the
union-find root. Assign positive uint32 region IDs by ascending canonical key.
Zero is reserved for unknown and `0xffffffff` is reserved for no region.

### Mapping and Edge Band

Map low-resolution sample centre `(iy, ix)` to render coordinates with the
same half-pixel convention as `align_corners=False`. The formula uses binary64
and the written operation order:

```text
fy = (iy + 0.5) * render_height / geometry_height - 0.5
fx = (ix + 0.5) * render_width  / geometry_width  - 0.5
py = clamp(floor(fy + 0.5), 0, render_height - 1)
px = clamp(floor(fx + 0.5), 0, render_width  - 1)
```

Nearest-label upsampling supplies the initial full-resolution region map. A
render pixel `(y,x)` chooses the low-resolution label at
`clamp(floor(sy+0.5), 0, SH-1), clamp(floor(sx+0.5), 0, SW-1)`, where `(sy,sx)`
is the exact unclipped output-to-source coordinate from the interpolation
oracle. Thus no library-specific `nearest` versus `nearest-exact` behavior is
implicit. A mapped boundary is any four-neighbour pair with different
non-sentinel IDs. Its
four-connected dilation radius is:

```text
radius = ceil(max(render_width / geometry_width,
                  render_height / geometry_height)) + 1
```

Pixels outside the band are immutable seeds. Inside each connected band
component, erode each participating region by `radius` iterations with the
four-connected cross kernel and constant-zero border. If erosion removes a
connected region fragment completely, use the exact morphological skeleton of
that fragment as its seed. The skeleton is built by repeated cross-kernel
opening and erosion, accumulating `current AND NOT opened` until `current` is
empty. If numerical shape degeneracy still leaves no seed, retain the fragment's
lowest row-major pixel. Original low-resolution sample centres are not all
retained as markers.

### Integer Geodesic Assignment

Assign unknown band pixels by multi-source four-neighbour Dijkstra. For adjacent
uint8 BGR pixels `p` and `q`, traversal cost is the exact uint64 integer:

```text
256 + 8 * max(abs(Bp-Bq), abs(Gp-Gq), abs(Rp-Rq))
```

Production owns the complete host region, uint64 distance, owner/tie state, and
one global indexed binary heap for the render frame. It is not an
independent-tile algorithm and has no fixed-halo approximation. Implementations
may chunk array storage, but every relaxation still enters the same global heap
and follows the same priority order; no tile may finalize a pixel independently.

The queue contract is exact:

```text
heap_pixel:    uint32 [edge_band_pixel_count]
heap_position: int32  [H,W]
```

`heap_position=-1` means unseen/not queued and `-2` means settled; every other
value is that pixel's current heap slot. At most one live heap entry exists for
an unsettled band pixel, so maximum heap length is
`edge_band_pixel_count <= H*W`. Heap comparison reads distance, owner region
rank/ID, row, and column from the dense canonical arrays using the priority key
below. A lower-cost or equal-cost better-owner relaxation updates those arrays
and performs indexed insert or in-place decrease-key; stale duplicate pushes are
forbidden. Allocate the heap once after the band mask is known. Capacity
overflow or an inconsistent position is `QualityRegionQueueBudgetError`, which
fails the frame without changing mode or output policy. The phase-specific host
formula below includes the worst-case `H*W` heap plus its dense position array.

Neighbour visitation order is up, left, right, down. Priority order is total
cost, then descending region rank, then ascending region ID, row, and column.
Region rank is the maximum float32 `near_score` bit pattern in that source
region after positive-zero normalization. A lower cost always wins. This makes
a flat-colour band reduce to a deterministic geometry-distance partition while
a real RGB discontinuity raises the cost of crossing that discontinuity. No
Sobel confidence threshold or floating Lab conversion participates in label
selection.

For Dijkstra boundary ties, choose the greater region rank and then lower region
ID. Zero-weight interpolation follows the exact scalar nearest-marker oracle
above.

Quality requires a three-channel uint8 BGR guide. Other guide dtypes or shapes
raise an explicit error; Fast dtype behavior is unchanged.

### Worked Movement Contract

For the reported 1080-to-1920 horizontal scale, the scale factor is
`1.777...` and the band radius is 3. In a one-dimensional two-region fixture
with mapped geometry boundary `b`, fixed left and right markers begin outside
the inclusive corridor `[b-3, b+3]`. A clean full-channel BGR step placed at
`b+0`, `b+1`, `b+2`, or `b+3` must produce its region boundary at that exact
step. This proves that marker density does not collapse the permitted motion.

A second fixture places three parallel edges inside one band: a 32-level
foreground highlight at `b+1`, the 192-level foreground/background transition
at `b+2`, and a 16-level chromatic fringe at `b+3`. The selected region boundary
must be `b+2`. A scalar integer-geodesic oracle, independent of production queue
and union helpers, defines both fixtures.

## Two-pass Visibility Contract

### Temporary Pass A Winner Data

`forward_splat_band` keeps its packed strict z-buffer. Pass A additionally
decodes each valid packed key to a full-frame uint32 winner source index before
the packed buffer is released. Invalid lanes use `0xffffffff`. The band gathers
source-region IDs from the full-resolution Quality region map. Winner index and
region arrays exist only for the active GPU band.

Pass A reduces each band to output-resolution or sparse host records, copies
those records to their final host positions, and releases every fine-grid
tensor before advancing to the next band.

### Compact Analysis Representation

For one eye, Pass A retains exactly these dense arrays:

```text
coverage_count: uint8  [H, W]
pure_region_id: uint32 [H, W], 0xffffffff when absent or mixed
pure_bgr:       three independently owned uint8 [H,W] planes in B,G,R order,
                zero when pure_region_id is absent
```

`coverage_count` is the number of valid pre-fill lanes. A pixel is a pure proxy
for region `r` only when it has at least one valid lane and every valid lane's
winner belongs to `r`. Invalid lanes are ignored. Sum the valid lane colours in
ascending lane order using int32, divide each channel by `coverage_count`, and
round to nearest with ties to even. The planar layout is semantic storage, not a
temporary interleaved copy. A pixel containing winners from two regions is never
a pure proxy, even if one region owns 15 of 16 lanes.

`pure_proxy` is context only. A pixel is a `safe_donor` for region `r` only
when it is a pure proxy, `coverage_count == 16`, and every in-frame pixel in its
clipped Chebyshev neighbourhood of radius
`max(1, floor(H / 1080 + 0.5))` also has coverage 16 and pure region `r`. Thus a
one-lane or boundary-adjacent proxy can influence context scoring but can never
be copied by local, exemplar, or fallback repair. The fallback-index phase
evaluates this predicate while streaming source pixels and does not retain a
full mask. After that index is discarded, the exemplar phase may materialize a
temporary boolean safe-donor mask inside its reused 64 MiB repair arena. The
mask is never live with the fallback index and is not a fourth retained analysis
array. Local repair always scans this clipped neighbourhood directly from
`coverage_count` and `pure_region_id` in row-major order. It may short-circuit a
failed predicate physically, but it allocates no safe-donor mask, integral image,
distance map, or other dense accelerator.

Each maximal contiguous invalid sequence is a **pre-fill run** with inclusive
fine coordinates `[s,e]`. Inspect valid anchors `L=s-1` and `R=e+1` when they
exist. Unequal near scores assign the complete run to the farther anchor using
Revision 5 depth ordering. Equal near scores preserve the current per-lane
distance/tie rule by splitting into at most two **repair runs**:

```text
left repair run  = every j in [s,e] where (j - L) <= (R - j)
right repair run = every remaining j
```

Thus the left half has `ceil((e-s+1)/2)` lanes and the right half has the
remainder; an exact midpoint lane belongs left. Each nonempty repair run stores
the corresponding anchor's source-region ID and `far_side`. This is the only
equal-depth policy. If only `R` exists, the pre-fill run must touch the left
frame edge and becomes one right repair run; the symmetric rule applies to only
`L`. A row-wide run with no valid anchor is a typed no-donor error in background
mode.

Each repair run is split at output-pixel boundaries into records with this fixed
little-endian, unaligned 16-byte dtype:

```text
pixel_index: uint32
lane_mask:   uint16
region_id:   uint32
fill_bgr:    uint8[3]
backend:     uint8
far_side:    uint8
reserved:    uint8
```

Field byte offsets are exactly 0, 4, 6, 10, 13, 14, and 15 in the order shown;
the dtype asserts `itemsize == 16` at construction.

`lane_mask` bit `i` corresponds to fine lane `i`. Backend values are 0
unplanned, 1 local strip, 2 exemplar, and 3 safe fallback. `far_side` is 0 for
the left boundary and 1 for the right boundary. The reserved byte is zero and
participates in plan hashing. Records are ordered by pixel index, least set
lane, region ID, then far side. Multiple records may exist for one pixel when
its unresolved runs bind to different source regions.
An unplanned record initially has zero `fill_bgr` and backend 0. After the
deterministic fallback-precompute phase, a still-unplanned backend-0 record may
hold its provisional fallback colour in `fill_bgr`; Pass B never scatters a
backend-0 value. Exemplar success overwrites it and sets backend 2, while final
fallback retains it and sets backend 3.

Every record mask is nonzero and contiguous within its output pixel. Masks in
one pixel are pairwise disjoint. Convert a record to its inclusive full-row
fine-coordinate interval using output column and the least/greatest set bits.
Two sorted records reconstruct one repair run only when their intervals are
fine-grid consecutive in the same row and both `region_id` and `far_side`
match. Because construction splits only at output-pixel boundaries, two
consecutive pieces of one repair run occupy adjacent output pixels. The selected
boundary anchor is then uniquely derived as `run_start - 1` for left and
`run_end + 1` for right; it must be a valid lane. A one-sided candidate is legal
only when the opposite run end touches the frame edge.

Hole-run statistics continue to describe the original maximal pre-fill run,
before an equal-depth split. Repair planning, backend statistics, records, and
components operate on repair runs.

For each eye and pixel, the bitwise OR of all record masks must equal the Pass A
invalid mask, and the sum of record-mask popcounts must equal the invalid-lane
count. Pass B repeats both checks against replayed invalid masks. These two
checks, plus pairwise disjointness, prevent scatter order from deciding colour.

The complete Pass A record table is limited to 64 MiB per eye. Backend choice
does not change record count, so exceeding this cap raises
`QualityRepairBudgetError` immediately; fallback cannot pretend to make the
table smaller. Within a valid table, components are processed whole in
canonical component order. Exemplar-budget exhaustion routes remaining records
to deterministic backend 3 without changing their table representation. The
renderer never allocates an unbounded table or silently uses Fast.

The plan uses two independently releasable fixed arenas: 64 MiB for 16-byte
records and 64 MiB for graph work. With at most 4,194,304 records, the graph
arena is partitioned into at most 16 MiB uint32 union parents, 4 MiB uint8 union
ranks, 16 MiB uint32 member order, 16 MiB uint32 component keys/roots, and
12 MiB nonrecursive in-place sort/scan workspace. Records are heapsorted in
place before union, so there is no second record-order array. After path
compression, member order is heapsorted by canonical component key then record
order; component ranges are streamed rather than retained as a descriptor list.
Partitions may be released and reused only in that sequence. If any fixed
partition is insufficient, raise `QualityRepairBudgetError`; no heap allocation
or alternate graph is permitted. The host preflight below also guarantees
`16*H*W <= UINT32_MAX` before any uint32 fine index is constructed.

### Global Host Planning

Sort Pass A records and use each record as a sparse node. Union only the exact
fine-lane adjacencies defined above, including across render-band boundaries.
For a node, its canonical fine index is
`row * W * 16 + column * 16 + least_set_lane`. A component key is the minimum
canonical fine index among its member nodes, and component IDs are assigned by
ascending key, independent of union-find roots. This planning result is
independent of Pass A band height.

Components describe pre-fill connectivity and are not renumbered after local
repair. Local planning visits reconstructed runs within each component; exemplar
and fallback see only that component's still-unplanned records. A successful
local record may therefore bridge two residual subsets without changing their
component ID, bounding box, statistics identity, or budget order.

The planner has this pure interface:

```python
def plan_quality_repairs(
    analysis: QualityVisibilityAnalysis,
    *,
    render_shape: tuple[int, int],
    local_limit_px: int,
    budgets: QualityRepairBudgets,
) -> QualityRepairPlan:
    ...
```

It mutates no renderer tensor and performs no file I/O. The returned plan owns
the filled 16-byte segment table, deterministic scalar/histogram statistics, and
measured budget counters. It owns no dense `repair_bits` or output RGB array.

### Compact-map Materialization

Background planning creates no dense repair map. After every record has its
final backend and colour, no planner may read `pure_region_id` or `pure_bgr`
again. Release the region plane and the G/R colour planes, zero the independently
owned B plane, and reuse that exact contiguous `Q`-byte allocation as
`repair_bits`. Scan `coverage_count` and final records in canonical order to set
the fixed bits below; no additional dense allocation is permitted. The
`coverage_count` and derived `repair_bits` then become immutable compact
diagnostics. The B-plane ownership transfer, rather than allocator-dependent
free/reallocate behavior, is part of the host formula.

For Quality none there is no plan or `pure_bgr`: its one visibility pass writes
its own `coverage_count` and `repair_bits` output planes directly. A completed
eye therefore owns exactly RGB `3*Q` plus the two compact planes `2*Q` in either
fill mode.

### Pass B Replay

Pass B reruns the same banded packed visibility using the same immutable host
int32 eye-offset map. For each band it selects plan records by output-pixel
range. The union of their lane masks must equal Pass B invalid lanes exactly,
and every masked lane must still be invalid. Any mismatch is an internal
deterministic error.

Scatter each record's `fill_bgr` only into its masked invalid lanes. Keep every
valid winner colour unchanged. Execute the existing fixed balanced 16-lane
addition tree and ties-to-even output conversion in the original location, then
release the band. Pass B never uploads or materializes a full-frame fine grid.

This replay makes output independent of band height while retaining exact lane
coverage. One output pixel may use different background colours for different
unresolved lane runs without mixing their donor components.

### CUDA OOM Rollback

For a frame, let `h0` be the deterministic initial Quality band height and
`h1=max(1,floor(h0/2))`. The frame begins at `h0`; after the first CUDA OOM in
any eye/pass, every later CUDA pass for that frame uses `h1`. These are the only
attempted heights.

- Pass A OOM discards the current eye's complete partial dense analysis, every
  appended sparse record, Pass-A/run counters, and temporary device state. It
  preserves immutable geometry/source-region data, the current eye-offset map,
  and any already completed earlier eye. It restarts the current eye at row 0
  with `h1`.
- Pass B OOM preserves the immutable record plan, current eye-offset map,
  `coverage_count`, and derived `repair_bits`, but discards the current eye's
  complete partial RGB, Pass-B validation/scatter counters, and temporary device
  state. It restarts Pass B at row 0 with `h1`.
- Quality-none visibility OOM discards the current eye's complete partial RGB,
  `coverage_count`, `repair_bits`, hole-run histograms/counters, and temporary
  device state. It preserves immutable Quality geometry, source-region map,
  current eye-offset map, and any completed earlier eye, then restarts that eye
  at row 0 with `h1`. It is not interpreted as Pass A rollback and creates no
  records.
- An OOM while already using `h1` is fatal and reports `h0` and `h1`. It never
  resumes after the failed band, rebuilds a different repair plan, lowers an
  evaluation budget, changes fill quality, or switches to Fast.

Before any restart, synchronize and run the existing CUDA release routine.
Rollback tests inject OOM after every band position in Pass A, Pass B, and the
none visibility pass; duplicate records, retained counter increments, partially
retained RGB, or retained partial none diagnostics are errors.

## Bounded Repair Planner

The following semantic work budgets are shared by the complete frame, across
the left eye first and then the right eye in that fixed order:

```text
QUALITY_LOCAL_SLOT_CAP             = 16_777_216
QUALITY_LOCAL_NEIGHBOR_SAMPLE_CAP  = 268_435_456
QUALITY_REPAIR_FALLBACK_VISIT_CAP  = 268_435_456
```

All three counters are checked uint64 values initialized once before either eye
is planned. They are output semantics and therefore participate in Quality RGB
algorithm v8. A new eye does not reset them, and CUDA OOM rollback restores them
to the exact value at that eye/pass checkpoint rather than granting more work.

### Local Limit and Strip Fill

The safe local limit retains the original 1080p-equivalent setting formula. The
predicted gap is based on the largest four-neighbour full-resolution one-eye
shift jump plus the existing two-pixel footprint guard:

```text
safe_limit_px      = max(1, floor(setting * H / 1080 + 0.5))
predicted_gap_px   = ceil(max_neighbour_abs_q_jump_px) + 2
local_limit_px     = min(safe_limit_px, predicted_gap_px)
```

Each repair run tries local fill exactly when
`run_lane_count <= 16 * local_limit_px`. For a run beginning at full-row fine
column `start_fine` with positive length `L`, statistics derive:

```text
first_pixel = floor(start_fine / 16)
last_pixel  = floor((start_fine + L - 1) / 16)
touched_pixel_span = last_pixel - first_pixel + 1
physical_width_px  = float64(L) / 16.0
```

The cap uses repair-run lane count, not touched-pixel span. The public pre-fill
run statistics use the same formulas on the unsplit pre-fill run.

#### Scalar Direction-preserving Local-strip Oracle

For repair run `[s,e]` in row `y`, let `dir=-1` and `anchor_fine=s-1` for a left
far side, or `dir=+1` and `anchor_fine=e+1` for right. Let
`boundary_col=floor(anchor_fine/16)`. Enumerate unique touched target columns
from the boundary into the hole:

```text
left far side:  first_pixel, first_pixel+1, ..., last_pixel
right far side: last_pixel, last_pixel-1, ..., first_pixel
```

Call their count `P`. This order is the direction from known background into the
hole. Target index `j` always receives donor index `j`; no reversal is permitted
later.

The actual boundary context is ordered spatially toward the hole:

```text
context_col[i] = boundary_col + (2-i)*dir, i=0,1,2
```

For a left far side this is `boundary-2, boundary-1, boundary`; for a right far
side it is the symmetric decreasing sequence. All three coordinates must be in
frame and be pure proxies of the repair run's region. Do not skip an intervening
non-proxy and do not shorten context. Otherwise local repair is ineligible.

The bounded horizontal search is:

```text
search_limit_px = max(1, floor(64 * H / 1080 + 0.5))
row_deltas      = [0, -1, +1, -2, +2]
```

Before reading the boundary context or evaluating any candidate for this run,
compute its complete deterministic charges with checked arithmetic:

```text
safe_donor_radius                 = max(1, floor(H / 1080 + 0.5))
safe_donor_sample_reservation     = (2*safe_donor_radius + 1)**2
run_slot_charge                   = 5 * search_limit_px
run_neighborhood_sample_charge    =
    3 + run_slot_charge * (3 + P*safe_donor_sample_reservation)
```

The first term in `run_neighborhood_sample_charge` reserves the three actual-
boundary context samples; each slot then reserves three candidate-context
samples and the maximum full Chebyshev support for each of its `P` donor
predicates. A clipped in-frame neighbourhood contains no more than this support,
so the reservation is a deterministic upper bound on every real local pixel
sample. Charge the complete values even when coordinates are out of frame or an
early predicate makes physical loads unnecessary. If either charge would make
the shared two-eye frame counter exceed its cap, skip local for the whole run
without reading any local candidate, increment
`local_budget_skipped_run_count`, retain backend 0, and continue to the normal
exemplar/fallback path. A run is never partially evaluated and a skipped run
does not consume either charge. Otherwise debit both complete charges before
evaluation; later eligibility or score outcomes never refund them.

Enumerate candidate slots by `offset=1..search_limit_px` outermost and the five
`row_deltas` innermost. Slot `(offset,k)` has row `cy=y+row_deltas[k]`, first
donor column `cx=boundary_col+dir*offset`, and ordinal `(offset-1)*5+k`. Its
candidate context and direction-preserving donor sequence are:

```text
candidate_context_col[i] = cx + (3-i)*dir, i=0,1,2
donor_col[j]              = cx - j*dir,     j=0..P-1
```

Thus donor coordinates advance in the same spatial direction as target
coordinates. For a budget-admitted run, every slot consumes one of exactly
`5*search_limit_px` evaluation slots, including an immediately rejected
out-of-frame slot; after the whole-run precheck there is no mid-run or
pressure-dependent early termination.

A slot is eligible only when `cy`, all three candidate-context coordinates, and
all `P` donor coordinates are in frame; both contexts must be same-region pure
proxies and every donor must be a same-region `safe_donor`. Every donor must
also lie strictly on the far side:
`(donor_col[j] - boundary_col) * dir > 0`. This forbids crossing or repeating
the actual boundary. Define `C[i,c]` from
`pure_bgr[c][y,context_col[i]]` and `D[i,c]` from
`pure_bgr[c][cy,candidate_context_col[i]]`, channels in B,G,R order. Compute in
this exact integer order:

```text
bgr_l1 = int64(0)
for i = 0..2:
    for c = B,G,R:
        bgr_l1 += int64(abs(int32(C[i,c]) - int32(D[i,c])))

YC[i] = (29*int32(C[i,B]) + 150*int32(C[i,G])
         + 77*int32(C[i,R]) + 128) >> 8
YD[i] = (29*int32(D[i,B]) + 150*int32(D[i,G])
         + 77*int32(D[i,R]) + 128) >> 8

luma_first_difference_l1 = int64(0)
for i = 0..1:
    dc = int32(YC[i+1] - YC[i])
    dd = int32(YD[i+1] - YD[i])
    luma_first_difference_l1 += int64(abs(int32(dc - dd)))

score = int64(2) * bgr_l1 + luma_first_difference_l1
```

Choose the smallest `(score, ordinal)`. Fill target column `j` from the three
`pure_bgr[c][cy,donor_col[j]]` values; only that target column's repair-run lane
mask receives the colour. This is a translated, direction-preserving strip
rather than reflection, supplies distinct donor pixels, and never repeats the
boundary colour. If no slot is eligible, retain backend 0 and enter exemplar
repair. A standalone scalar implementation is the local-strip test oracle.

### Exemplar Input and ROI

The component repair function consumes a clipped pure proxy, safe-donor mask,
`pure_region_id`, target records, exact region ID, provisional target colours,
and `QualityRepairBudgets`. The full-frame planner computes the one provisional
fallback lookup before exemplar and later applies the stored result; the
component function never interprets absence from its
clipped ROI as absence from the frame. It returns a full-level colour only for
targets actually completed by exemplar iteration. Other targets remain backend
0 for the outer planner.

Let the component target bbox be half-open `[y0,y1) x [x0,x1)`. Its read/search
domain is that bbox expanded by 128 actual output pixels and clipped to frame.
Partition the unexpanded bbox into nonoverlapping half-open interiors with
origins `(y0+384*i, x0+384*j)` in row-major `(i,j)` order:

```text
interior_y = [origin_y, min(origin_y + 384, y1))
interior_x = [origin_x, min(origin_x + 384, x1))
```

The last row or column is simply partial. A core working ROI expands its unique
interior by 64 pixels on every side and clips to the component read/search
domain and frame, so it is at most 512x512. Target owner is the unique core
whose **interior** contains that target; halo overlap never affects ownership.

A nonowned target in a halo is read-only. If an earlier row-major owner has
completed it, it is processed context; otherwise it is a barrier for the
current core. The current core never copies to it or marks it processed. A
synthesized target, including earlier-core context, never becomes a donor.

### Exact Pyramid State

Use full, half, and quarter levels, omitting a coarser level when its shorter
working-ROI side would be below 32 pixels. Each level owns exactly:

```text
working_bgr:    uint8 [h,w,3]
donor_mask:     bool  [h,w]
target_mask:    bool  [h,w]
processed_mask: bool  [h,w]
barrier_mask:   bool  [h,w]
```

At full resolution:

- `working_bgr` is `pure_bgr` for same-region pure proxies, an already completed
  earlier-core colour for such component context, the planner-provided
  full-frame provisional safe-donor colour for current targets, and zero for
  barriers;
- `donor_mask` is true only for same-region `safe_donor` pixels;
- `target_mask` is true only for targets owned by the current core;
- `processed_mask` is true for all same-region pure proxies and completed
  earlier-core context, and false for current targets and barriers;
- `barrier_mask` is false only for same-region pure proxies, targets owned by
  the current core, and completed earlier-core context; later/noncompleted halo
  targets are barriers.

These invariants always hold: donor and target are disjoint, target and barrier
are disjoint, donor implies processed, and barrier implies not processed.

Build coarser levels bottom-up from each clipped set of existing 2x2 children.
For parent `p`:

```text
p.barrier = any(child.barrier)
p.target = (not p.barrier) and any(child.target)
p.donor = (not p.barrier) and all(child.donor)
p.processed = (not p.target) and (not p.barrier)
              and all(child.processed)
```

If `p.barrier`, set `p.working_bgr` to zero. Otherwise set it to the per-channel
integer mean of every existing child's `working_bgr`, using int32 accumulation
in child order top-left, top-right, bottom-left, bottom-right and round-to-
nearest ties-to-even. A cell containing both a target child and a barrier child
is therefore a barrier, not a coarse target; that target is repaired only at a
finer level. No synthetic padding is added.

Process levels coarsest to full. Before a finer level begins, overwrite only
its target working values whose parent is a non-barrier coarse target:

```text
fine.working_bgr[y,x] = coarse.working_bgr[floor(y/2), floor(x/2)]
```

Do not modify donor, proxy, completed-context, or barrier values, and do not
mark a finer target processed. If a coarse level ended with an unprocessed
target, its already-defined provisional working value is still replicated; the
finer target remains unprocessed and gets a fresh opportunity. No fallback is
committed merely because a coarse target was unfinished.

### Exact Patch Iteration

At each level, `processed_mask` is the known-context mask. Recompute the
four-neighbour frontier among `target_mask AND NOT processed_mask` after every
patch copy. A frontier target has at least one non-barrier, processed
four-neighbour. Its complete 7x7 target patch must lie inside both the frame and
working ROI and contain no barrier; patches are never clipped. Select the
eligible frontier centre with greatest processed sample count in that 7x7
patch; ties use row then column. Require at least eight processed samples.

Candidate donor centres are enumerated in row-major order. Their central 7x7
copy patch and its one-pixel Sobel border, a complete 9x9 support, must lie inside
the frame and working ROI. Every pixel in that 9x9 support must be an original
same-region `safe_donor`; provisional, completed target, proxy-only, or barrier
pixels make the candidate ineligible. For the target patch's known offsets,
calculate:

```text
Y = (29*B + 150*G + 77*R + 128) >> 8
Gx = [-1 0 1; -2 0 2; -1 0 1] applied in int32
Gy = transpose(Gx) applied in int32
score = 2 * sum(BGR_L1)
        + sum(abs(Gx_target-Gx_donor) + abs(Gy_target-Gy_donor))
```

Compute donor Sobel values from the 9x9 support and retain only gradients aligned
with its central 7x7 copy patch; donor patch boundaries never use reflection.
For each known central target offset, its BGR term always contributes. Its
gradient term contributes only when all coordinates in its target 3x3 Sobel
support are non-barrier and `processed_mask=true`. At a true working-ROI/frame
boundary, apply reflect-101 first and test the mapped coordinates; there is no
synthetic processed padding. Thus neither a barrier nor an unfinished target's
provisional colour outside the central 7x7 can affect the score. Use the aligned
donor gradient only for the same target-gradient-eligible offsets. Accumulate
the int64 score in row-major sample/channel order, BGR terms first and eligible
gradient terms second. Both contributing offset sets are target-defined and
therefore identical for every candidate in an iteration, so no division occurs.
Lowest score wins; a tie uses donor row then column.

Copy every currently unprocessed target pixel in the selected 7x7 patch from
the aligned donor patch and mark it processed. The update is immediately usable
as context but never changes `donor_mask`. A level ends when every target is
processed, no eligible frontier or candidate remains, 8,192 patch iterations
have run, or any applicable donor-evaluation budget is exhausted.

Budgets count every scored donor patch across all pyramid levels:

```text
per_core_evaluations      = 2,000,000
per_component_evaluations = 8,000,000
per_eye_evaluations       = 32,000,000
```

Components run by canonical component key, cores by row-major origin, levels
coarsest to full, and iterations in the order above. Before an iteration, `M`
is the minimum remaining core, component, and eye budget. If `N` row-major
candidates exceed `M`, score candidates at unique indexes `floor(k*N/M)` for
`k=0..M-1`. When `M == 0`, terminate before candidate selection. If a core
budget reaches zero, skip exemplar work for the rest of that core. If a
component budget reaches zero, skip its remaining cores. If the eye budget
reaches zero, skip all remaining components. Only a target processed at full
resolution receives backend 2; every skipped or unfinished full-level target
proceeds to outer fallback in canonical component/record order.

### Fallback and Memory Failure

After local planning and before exemplar work, the outer full-frame planner
indexes original safe donors by source-region ID for every still-backend-0
record. It computes and stores one provisional colour per such record. After
exemplar, every record still at backend 0 adopts that already stored colour as
backend 3. The nearest same-region safe donor must be within this
1080p-equivalent distance:

```text
fallback_limit_px = max(1, floor(256 * H / 1080 + 0.5))
```

Distance between output-pixel coordinates is exact integer squared Euclidean;
ties use donor row then column. The search sees the full analysis frame, not the
128-pixel exemplar expansion. Set backend 3. Pure-but-partial proxies and
synthesized pixels are ineligible. If no safe donor exists within the limit,
fail the frame explicitly rather than pulling foreground colour or leaving a
black hole.

The scalar fallback oracle enumerates every same-region safe donor inside the
clipped square search window in row-major order, rejects `distance2` above the
squared limit, and applies the distance/row/column key.

Production uses a deterministic per-region implicit two-dimensional k-d index:

- collect row-major uint32 pixel indexes only for safe donors whose region is
  referenced by at least one backend-0 record;
- partition each region's contiguous range in place, starting with row axis and
  alternating row/column by depth;
- choose the lower median by the total key `(axis coordinate, other coordinate,
  pixel_index)` using deterministic median-of-medians partitioning;
- child ranges are implicit and require no pointer per donor;
- the scalar query starts with no donor and `radius2=fallback_limit_px**2`. At
  each nonempty range it visits and counts the median node first, accepts it only
  when `distance2 <= radius2`, and keeps the smallest exact
  `(distance2, row, column)` key;
- let signed-int64 `delta = target_axis - node_axis`. The lower child is nearer
  when `delta <= 0`, otherwise the upper child is nearer. Query the nearer child,
  then query the farther child exactly when `delta*delta <= bound`, where `bound`
  is the smaller of `radius2` and the selected donor's distance, or `radius2`
  when none has been selected. Axis selection alternates exactly as in build.

The index must return the scalar-oracle donor exactly. Every examined node is
debited from the shared two-eye
`QUALITY_REPAIR_FALLBACK_VISIT_CAP` before its candidate or child ranges are
used. An attempted debit past the cap raises
`QualityRepairFallbackQueryBudgetError`, discards the complete frame, and never
returns the current best donor, changes the search radius, scans linearly, or
switches modes.

The index may occupy at most 48
MiB of the per-eye 64 MiB repair scratch arena, covering the uint32 donor array,
region descriptors, construction stack, and query stack. The remaining 16 MiB
stores one uint32 visited-node count for each queried backend-0 record; the
64 MiB record cap proves the query count cannot exceed 4,194,304. Compute total,
maximum, and scalar p95 in place before resetting this arena. Build complexity is
`O(N log N)` over indexed donors,
working memory is `O(N)`, expected query work is `O(log N)`, and exact worst case
remains `O(N)`; the only termination boundary is the fatal frame-level visit
cap, never a partial query result.
Diagnostics record indexed donor count, query count, total visited nodes,
maximum, and p95 visited nodes, plus local slot charges, local neighbourhood-
sample reservations, and whole-run budget skips. The two eye totals must add exactly to the
shared frame counters and the fallback total may not exceed its cap. Query p95
applies the exact scalar linear rule
below to the per-query visit counts and is `0.0` when there are no queries.
Wall-clock build time is benchmark telemetry, never hashed frame diagnostics.
The 4K fallback stress gate below constrains the practical case.

Before background-mode Pass A appends its first record, allocate the complete
64 MiB record, 64 MiB graph, and 64 MiB repair arenas. An OOM retry resets but
retains those arenas. Local fill uses no dynamic allocation. Fallback index
construction, its visited-count array, and all provisional queries use the first
phase of repair scratch. After every provisional colour and fallback diagnostic
scalar is stored, reset that same arena and use its complete 64 MiB for
safe-donor materialization and exemplar; the index is no longer queried. Index
and exemplar state are therefore never simultaneous allocations.

If either preallocation, index construction, or any later allocation raises
`MemoryError`, fail the frame and commit no RGB or diagnostics. Earlier planned
values are discarded. Runtime memory pressure never selects fallback and never
changes output under one RGB identity. Only the fixed core/component/eye
evaluation budgets may deterministically route records to backend 3.

If any residual record has no safe donor within the limit or required index
state exceeds its 48 MiB partition, propagate an actionable render error. There
is no unconstrained
OpenCV inpaint path and no silent Fast downgrade.

## Compact Coverage and Statistics

The file-production path uses an internal compact result and retains only:

```text
coverage_count: uint8 [H,W]
repair_bits:    uint8 [H,W]
```

The existing public `StereoRenderResult` API remains structurally unchanged: it
still owns concrete `left_valid_mask`, `right_valid_mask`, `left_hole_mask`, and
`right_hole_mask` NumPy boolean arrays. `StereoRenderer.render()` is a public
wrapper which materializes those four arrays. They are allocations, not
zero-copy views, and their four bytes per output pixel are included in the
public API memory gate. The frame generator calls an internal compact render
entry point and does not materialize them.

For Quality, public valid masks are `coverage_count > 0`. Under `none`, public
hole masks are `coverage_count == 0`; under successful `background`, they are
all false because any final unresolved lane is a render error. Fast retains its
existing exact mask computation. The fixed repair-bit mapping is:

| Bit | Hex | Meaning |
|---:|---:|---|
| 0 | `0x01` | pre-fill partial hole, coverage 1..15 |
| 1 | `0x02` | pre-fill full hole, coverage 0 |
| 2 | `0x04` | at least one lane locally filled |
| 3 | `0x08` | at least one lane remained after local fill |
| 4 | `0x10` | at least one lane exemplar filled |
| 5 | `0x20` | at least one lane fallback filled |
| 6 | `0x40` | at least one lane finally unresolved |
| 7 | `0x80` | reserved and always zero |

Bits 0 and 1 always describe pre-fill coverage. With `occlusion_fill=none`,
bits 2 through 5 remain zero and bit 6 is set exactly when coverage is below 16.
With background fill, bit 3 is set whenever a lane remains after the local
phase, including a run which bypassed local because it exceeded the cap. Newly
rendered Fast treats its existing boundary-copy fill as backend 1; bits 4 and 5
remain zero there.

All pixel counts and ratios are per eye and frame. A pixel ratio denominator is
exactly `H*W`. A horizontal pre-fill hole run is a maximal contiguous invalid
fine-lane sequence in one row. Report lane count, touched-pixel span from the
exact formula above, and physical width `lane_count / 16.0`. Maximum and p95 for
each measure operate over all nonempty runs, not one maximum per row. P95 uses
the exact scalar linear rule below; an empty run set reports integer maxima 0
and binary64 p95 values `0.0`.

Backend lane count is the number of lane bits assigned to that backend. Backend
pixel count is the number of pixels with at least one such bit. In Quality,
backend component count is the number of canonical sparse segment components
which used that backend at least once. Fast does not construct source regions,
segment records, or this graph, so all Fast backend component values are JSON
`null` with `availability="unavailable_fast_no_segment_graph"`; they are not
invented from coverage or final RGB.

`final_unresolved_lane_count` is the sum of unresolved lane bits after final
rendering. Repair bit 6 is set for a pixel exactly when its unresolved lane
count is nonzero. The public legacy `hole_mask` remains true only when all 16
lanes are finally unresolved.

Statistics contain only integers, finite nonnegative floats, strings, booleans,
lists, dictionaries, and JSON `null` at fields explicitly declared nullable by
the strict schema below. Normalize negative floating zero to positive zero.
Reject NaN and infinity. Canonical bytes are ASCII JSON with sorted keys,
`separators=(",", ":")`, `ensure_ascii=True`, and `allow_nan=False`; SHA-256 of
those bytes is the statistics fingerprint.

Lane-level counts are producer-attested, not independently reconstructable from
persisted PNG sidecars. For Quality background, one counter is accumulated from
final segment records and a separate Pass B counter from actual scatter masks;
lane totals, per-backend totals, mask OR, popcount, and final unresolved counts
must agree before commit. Quality none derives coverage/unresolved totals from
its one visibility pass and has no plan counter. Fast instead counts writes
inside the existing fill helper
and independently derives pre-fill, filled, and final lane totals from the
active band's pre/post validity arrays before releasing them. It never runs the
Quality planner for diagnostics. Resume verifies committed hashes and
PNG-derived pixel counts when masks exist, but does not claim to rederive lane
or component provenance. Documentation uses "committed diagnostics" rather than
"independently auditable lane plan."

## Diagnostics Stage and Transaction

Stereo RGB and diagnostics use the separate identities fixed in the Persistence
and Identity Contract. Fast RGB metadata and fingerprint remain byte-compatible
with existing v3. Every path below is relative to the job output root:

```text
04_stereo_diagnostics/
    metadata.json
    quality_input_manifest.json  # Quality only
    stereo_coverage_frames.jsonl
    stereo_coverage_summary.json
    frames/frame_000089/
        left_coverage.png
        left_repair.png
        right_coverage.png
        right_repair.png
        stats.json
        manifest.json
```

Coverage and repair PNGs are single-channel uint8 at render shape and exist only
when `keep_intermediates=true`. Per-frame `stats.json` and `manifest.json` exist
for every new Fast or Quality render through `building` and `complete`,
regardless of mask retention. A later successful `payload_pruned` transition is
the sole normal path allowed to remove them.

Diagnostics normalize the existing persisted `stereo_geometry_mode` value
`metric_camera` to the JSON token `metric`; `relative` remains `relative`. This
normalization is diagnostics-only and does not rename the existing RGB-stage
field or public setting.

### Canonical JSON and Hashing

All JSON objects reject missing or extra keys. Canonical JSON is the ASCII byte
encoding already defined for statistics. A 64-character lowercase hexadecimal
SHA-256 hashes the exact stated bytes. Every self-fingerprinted object computes
`fingerprint = SHA256(canonical_json(object without fingerprint))`.

For source-ordered frame names, define:

```text
ordered_frame_manifest_fingerprint = SHA256(canonical_json([
  {"frame_name": name, "manifest_sha256": SHA256(raw_manifest_bytes)},
  ...
]))
```

The JSONL bytes are each validated `stats.json` object re-encoded canonically,
followed by ASCII LF, in exact source order. Nonempty JSONL has a final LF; an
empty sequence is zero bytes. `frames_jsonl_sha256` hashes those exact bytes.

### Encoding Input Sequence Manifest

Final encoding never uses a diagnostics manifest or generic frame-stage
metadata as image-content identity. Before FFmpeg starts, resolve its actual
input files and stream the canonical representation for the future job-root
`encoding_input_manifest.json` with exactly:

```text
schema_version:    1
algorithm_version: "encoding-input-sequence-v2"
mode:              "direct_stereo" | "assembled_vr"
frame_names:       list[string]
left:              list[EncodingImageIdentity] | null
right:             list[EncodingImageIdentity] | null
frames:            list[EncodingImageIdentity] | null
fingerprint:       string
```

`EncodingImageIdentity` has exactly:

```text
relative_path: string
sha256:        string
byte_count:    integer
png_header:    {width: integer, height: integer, bit_depth: integer,
                 color_type: integer}
```

Final encoding obtains those identities only through this bounded interface:

```python
@dataclass(frozen=True)
class EncodingSequenceProvider:
    mode: Literal["direct_stereo", "assembled_vr"]
    frame_count: int
    first_index: int
    last_index: int
    left_parent: PurePosixPath | None
    right_parent: PurePosixPath | None
    frames_parent: PurePosixPath | None
    audit_generation_id: str
    generation_fingerprint: str

    def replay(self) -> Iterator[EncodingSequenceItem]: ...
```

`audit_generation_id` is a fresh 128-bit OS-CSPRNG token encoded as exactly 32
lowercase hexadecimal characters. Define exactly:

```text
generation_fingerprint = SHA256(canonical_json({
    "audit_generation_id": audit_generation_id,
    "mode": mode,
    "frame_count": frame_count,
    "first_index": first_index,
    "last_index": last_index,
    "left_parent": left_parent as canonical string or null,
    "right_parent": right_parent as canonical string or null,
    "frames_parent": frames_parent as canonical string or null,
}))
```

It is fixed before the first item is yielded; the separate encoding-input-
manifest self-fingerprint is produced by the complete content prepass.
`EncodingSequenceItem` has exactly the scalar source position, canonical numeric
frame index, canonical stem, and nullable left/right/assembled relative paths
for that one iteration. It carries the provider's two generation values. The
consumer starts with `expected_source_index=0`, requires each item to have that
exact source position and expected `first_index+source_position` frame index,
then increments with checked arithmetic. No item, path, or name is retained
after the next yield.

`VRFrameAssembler` continues to own only the policy which selects the 06 cropped
or 07 upscaled parents. Its current `tuple[list[Path], list[Path]]` return
interface is not used or retained by final encoding. The assembled path likewise
does not call an eager `get_frame_files()`. Provider construction retains only
the selected contained parent paths and scalar range. A streaming directory
pass parses every `frame_*.png` entry, requires a canonical index inside the
declared continuous range, and counts exactly `frame_count`; filename uniqueness
plus the range/count proof establishes completeness without a set or sort. Each
replay then constructs and opens the one expected filename directly.

Direct mode resolves the exact 06 cropped or 07 upscaled left/right sequences
selected by that policy. `left` and `right` are
non-null, have the same positive length and source-ordered frame names, and
`frames` is null. Assembled mode resolves the exact source-ordered
`99_vr_frames` sequence; `frames` is non-null and `left`/`right` are null. Every
path is output-root-relative POSIX form, resolves beneath the acquired job root,
and is opened as a regular file without following a symlink, junction, mount
point, or other reparse point.
The manifest contains only files FFmpeg reads as image inputs. Diagnostics masks,
stats, their manifests/fingerprints, mask-retention policy, mtimes, and semantic
stage fingerprints are forbidden.

The prepass validates every IHDR from the same open handle used for hashing.
Every consumed image is PNG color type 2, bit depth 8. In direct mode every left
header equals the first left header and every right header equals the first
right header. If both first headers equal the configured positive per-eye
dimensions, stack them without scaling; otherwise apply the existing
`scale=<per_eye_width>:<per_eye_height>:flags=bicubic+accurate_rnd` filter to
both inputs before stacking, even when only one input differed. In assembled
mode every header must already equal the exact configured final VR width and
height; assembled encoding never repairs a size mismatch in FFmpeg. Palette,
grayscale, RGBA, non-8-bit, and intra-sequence shape changes fail before FFmpeg.

Every encoder path normalizes square pixels before the encoder. In the notation
below, `STACK` is replaced by exactly `hstack` for side-by-side or `vstack` for
over-under; it is not a runtime token. Direct mode's two filter-graph forms are:

```text
no scale:
    [0:v]setsar=1[left];[1:v]setsar=1[right];
    [left][right]STACK=inputs=2:shortest=1[stacked];
    [stacked]setsar=1[vr]

scale:
    [0:v]scale=W:H:flags=bicubic+accurate_rnd,setsar=1[left];
    [1:v]scale=W:H:flags=bicubic+accurate_rnd,setsar=1[right];
    [left][right]STACK=inputs=2:shortest=1[stacked];
    [stacked]setsar=1[vr]
```

Assembled mode adds the exact argument pair `-vf`, `setsar=1` to its validated
image stream before encoder options and explicit video mapping. No branch may
rely on inherited PNG/container SAR metadata.
All of these filter tokens are present in the runtime and normalized argument
vectors, so changing or omitting one changes final-video identity.

`frame_names` are unique canonical stems in gap-free numeric order using the
minimal six-digit-padding rule defined by the Quality content manifest; each
identity path's filename is exactly its stem plus `.png`. Extra leading zeroes
or an extension in a `frame_names` element are rejected. For every selected
parent, the complete `frame_*.png` directory listing
must equal the manifest list, so an unmanifested trailing image cannot be read by
image2. Define the fixed project/runtime boundary:

```text
FFMPEG_IMAGE2_START_NUMBER_MAX = 2_147_483_647
MAX_SUPPORTED_IMAGE2_INDEX     = 2_147_483_646
last_index = checked_u64_add(first_index, N - 1)
require first_index <= MAX_SUPPORTED_IMAGE2_INDEX
require last_index  <= MAX_SUPPORTED_IMAGE2_INDEX
```

`start_number` is a signed-int option on every project-supported FFmpeg image2
build. The project maximum deliberately leaves one value of headroom because
the supported demuxer advances its internal next index once after delivering the
last requested frame; admitting 2,147,483,647 would wrap that read-ahead to
-2,147,483,648 and emit an image2 I/O error. The boundary is fixed rather than
dynamically widened from the canonical U64 naming domain. Overflow or an out-of-
range endpoint fails before manifest reservation or FFmpeg. Both commands pass
`first_index` as `-start_number` and `N` as `-frames:v`; assembled encoding gains
these arguments rather than depending on directory exhaustion. The supported-runtime gate must
decode a two-frame fixture at indexes 2,147,483,645 and 2,147,483,646 with no
error diagnostic, and must reject `(first_index=2,147,483,646,N=2)` and
`(first_index=2,147,483,647,N=1)` in project preflight without launching FFmpeg.

Hash every complete raw PNG, not decoded pixels or a sampled/stat identity. Any
image byte, byte count, header, order, or relative-path change changes the
self-fingerprint. A diagnostics-only or mask true-to-false migration leaves it
unchanged. Semantic stage identity may be recorded only in separate execution
provenance which is not an input to this object or final-video identity.

Audio has no entry in this image manifest. When `preserve_audio=false`, neither
encoder opens an audio source or emits an audio input, map, filter, or codec
option. When it is true, select `original_audio.flac` if and only if that path is
a non-link regular file; otherwise select the original-video path. There is no
fallback from a selected FLAC to the original video after selection.
The selected path itself must then open as a non-link regular file; otherwise
audio preflight is fatal.

Hash the selected source's complete raw bytes. Probe its first audio stream with
this exact command and no generic video-info helper:

```text
ffprobe -v error -select_streams a:0 \
  -show_entries stream=codec_type,sample_rate,channels \
  -of json <selected-audio-source>
```

`BoundedProcessCaptureV1` applies the audio-specific stdout/stderr caps below.
Its fixed-schema incremental parser permits exactly the root keys `streams`,
`programs`, and `stream_groups`, each at most once and in any order. `streams`
is required and is an array of zero or one stream object with exactly the three
requested stream keys. `programs` and `stream_groups` are optional arrays
because supported FFprobe JSON writers emit those section wrappers even when
their contents were not requested. For an arbitrary selected audio source they
may be empty or contain JSON objects; any other element type fails. A bounded
JSON value-skip state machine, not an object tree, consumes those objects with
maximum depth 8, at most 256 aggregate container starts, and at most 128 bytes
per ignored scalar token. Their duplicate stream
descriptions are non-authoritative: only the top-level `streams` array can
produce the zero/one audio result or contribute a count.

Invalid UTF-8, NUL, an unknown/duplicate root key, an extra/missing authoritative
stream key, JSON depth greater than eight, a requested-stream scalar token longer
than 32 bytes, an ignored-wrapper token/count overflow, a second top-level
stream, any stderr byte, nonzero exit, or cap/parser-state overflow is fatal.
Zero top-level streams is `audio_absent`. One stream requires
`codec_type="audio"`, `sample_rate` as a canonical nonzero decimal string which
normalizes to `S <= U32_MAX`, and `channels` as a non-boolean JSON integer in
`1..U32_MAX`; `N/A`, sign, whitespace, leading zero, float, or exponent forms
are invalid. Probe timeout raises `FinalAudioProbeTimeoutError`; byte/state
overflow raises `FinalAudioProbeBudgetError`. A present malformed stream is
never absence.

The metadata probe does not infer whether the selected clip contains samples.
For its one valid `a:0`, run exactly this full clipped decode, with the optional
resolved input arguments present in the shown order only when applicable:

```text
ffmpeg -v error -xerror \
  [ -ss <resolved-start> ] [ -t <resolved-duration> ] \
  -i <selected-audio-source> -map 0:a:0 -vn -sn -dn \
  -c:a pcm_s32le -f s32le pipe:1
```

Stdout is continuously drained by `readinto` on one reusable
`FINAL_ENCODING_AUDIO_STREAM_BYTES` bytearray and is never copied into a new
per-read bytes object, concatenated, or decoded as text. Each read increments
`decoded_pcm_bytes=checked_u64_add(decoded_pcm_bytes, read_count)`. Compute
`pcm_frame_bytes=checked_u64_mul(4, channels)` before launch. At clean EOF,
zero decoded bytes is `audio_absent`; otherwise the count must be divisible by
`pcm_frame_bytes`. Stderr is simultaneously drained into the fixed audio-decode
diagnostic ring. Any byte beyond its cap sets overflow, continues drain through
termination/reap, and raises `FinalAudioDecodeBudgetError`; nonzero exit or any
stderr byte is fatal. Counter multiplication/addition overflow raises
`FinalAudioPcmOverflowError` after termination/reap. Thus a usable stream has a
positive checked PCM count, while neither decoded PCM nor diagnostics scale in
resident memory.

Audio probe has a 30-second wall timeout. Full PCM decode has no duration-based
deadline because clip length is unbounded, but 120 seconds with no stdout,
stderr, exit, or cancellation-state progress is
`FinalAudioDecodeTimeoutError`. Timeout, user cancellation, input-generation
mismatch, pipe failure, parser failure, and counter failure all close stdin,
terminate the child, continue bounded drainage, wait five seconds, kill if
needed, and reap exactly once before returning. A cancelled or failed probe/
decode cannot fall back to the original video, declare absence, launch the final
encoder, or publish a manifest.

For usable audio, both encoder paths add exactly one audio input and explicitly
map that input's `a:0`; optional-map `?` syntax is forbidden after preprobe. Let
the reduced positive image rate be `fps_num/fps_den`, and compute with checked
integers:

```text
audio_numerator  = checked_u64_mul(checked_u64_mul(N, S), fps_den)
audio_end_sample = checked_u64_add(audio_numerator, fps_num - 1) // fps_num
```

Apply exactly `atrim=end_sample=<audio_end_sample>,asetpts=PTS-STARTPTS`, then
encode one AAC stream. A shorter stream ends naturally; no `apad` or synthetic
sample is allowed. A longer stream is trimmed at that sample boundary. Both
paths explicitly map their image-derived video stream and always pass
`-frames:v N`. Output-level `-shortest` is forbidden. Direct mode may retain
only the video-stack-local `hstack|vstack=inputs=2:shortest=1`, because both
validated image inputs have exactly N frames.

The normalized FFmpeg audio token contains the pre/post-validated source's full
raw SHA-256 and byte count; sampled or stat-only fingerprints are forbidden.
Clip selection, `audio_end_sample`, maps, filter, and codec options remain
authenticated by normalized executed arguments.

The encoder streams one image identity at a time, so it never materializes an
`O(N)` Python object tree. The pre-FFmpeg pass writes canonical bytes at the
beginning of the reserved temporary, records their length and raw SHA-256 plus
the self-fingerprint, verifies the already-frozen `EncodingSequenceProvider`
generation, then physically fills the rest of the schema bound. FFmpeg command
construction uses only that frozen generation's parents, first/last indexes, and
count to form its image2 patterns; it never calls the source resolver again.
After successful encoding, the postpass replays that same generation, reopens
and rehashes the same
contained image paths, streams the resulting canonical bytes over that same
extent, and requires the length, raw hash, self-fingerprint, generation token,
and every expected source index to match the pre-FFmpeg values. It also rehashes
the audio source against its captured full identity. A changed file, header,
order, parent selection, or identity aborts publication; a match truncates the
temporary to the canonical length and fsyncs it before publication.

After all manifest/audio/disk preflight and immediately before the first FFmpeg
launch attempt, validate the producer pair. Commit the immutable
`final_media_producer_contract_version=4` and the exact authenticated
reservation-preflight `final_media_publication_generation` if the pair is null;
that value was independently generated by the OS CSPRNG for this first attempt
and is never copied from the provider's `audit_generation_id`. Then replay the
frozen provider one last time.
This settings write is a monotonic provenance mutation, but launch remains the
image/audio publication mutation boundary: a mismatch after marker commit but
before launch may still discard temporaries and restart read-only audit, while
the marker is never cleared back to null.

Before FFmpeg process launch, a provider mismatch discards the read-only
encoding audit, authenticated invocation-only reservation/index set, and
preflight result, retains the lifecycle terminal extent, and may restart with a
new provider/reservation generation under the same immutable publication
generation. Launch is the final-encoding mutation boundary.
At or after launch, a mismatch raises `FinalEncodingInputChangedError`,
terminates and reaps an active FFmpeg process or rejects its completed output,
removes the sibling output, commits terminal failure from the retained extent,
then removes only its authenticated unused invocation reservations; it publishes
neither manifest nor video and never retries in the same call. A later
independent invocation starts a new read-only generation. This rule is subject
to the external-ABA limitation below.

The project-owned final-encoding coordinator has its own resident-memory
contract, independent of the stereo-stage 512 MiB phases:

```text
ENCODING_CONTROL_FRAME_CAP         = 4_194_304
FINAL_ENCODING_CONTROL_BUDGET      = 8 MiB
FINAL_ENCODING_CONTROL_OVERHEAD    = 4 MiB
FINAL_ENCODING_IMAGE_STREAM_BYTES  = 1 MiB
FINAL_ENCODING_JSON_STREAM_BYTES   = 1 MiB
FINAL_ENCODING_AUDIO_STREAM_BYTES  = 1 MiB
FINAL_ENCODE_DIAGNOSTIC_BYTES      = 256 KiB
FINAL_ENCODE_LINE_BYTES            = 8 KiB
FINAL_PROBE_STDOUT_BYTES           = 64 KiB
FINAL_PROBE_STDERR_BYTES           = 64 KiB
FINAL_PROBE_JSON_STATE_BYTES       = 512 KiB
FINAL_DECODE_DIAGNOSTIC_BYTES      = 256 KiB
FINAL_DECODE_PROGRESS_STATE_BYTES  = 64 KiB
FINAL_DECODE_LINE_BYTES            = 8 KiB
FINAL_AUDIO_PROBE_STDOUT_BYTES     = 32 KiB
FINAL_AUDIO_PROBE_STDERR_BYTES     = 32 KiB
FINAL_AUDIO_PROBE_JSON_STATE_BYTES = 256 KiB
FINAL_AUDIO_DECODE_DIAGNOSTIC_BYTES = 256 KiB
FINAL_RESERVATION_ENTRY_CAP         = 64
FINAL_RESERVATION_DESCRIPTOR_RAW_BYTES = 8 KiB
FINAL_RESERVATION_INDEX_STREAM_BYTES   = 256 KiB
FINAL_RESERVATION_INDEX_JSON_STATE_BYTES = 256 KiB
FINAL_RESERVATION_INDEX_RAW_BYTES      = 256 KiB
FINAL_AUDIO_PROBE_TIMEOUT_SECONDS   = 30
FINAL_AUDIO_DECODE_STALL_SECONDS    = 120
FINAL_ENCODE_STALL_SECONDS          = 120
FINAL_PROBE_TIMEOUT_BASE_SECONDS    = 30
FINAL_PROBE_TIMEOUT_PER_1000_WORK_UNITS_SECONDS = 2
FINAL_PROBE_TIMEOUT_MAX_SECONDS     = 21_600
PROJECT_AAC_SAMPLES_PER_PACKET_BOUND = 1024
PROJECT_AAC_PACKET_SLACK              = 2
FINAL_PROBE_BYTES_PER_WORK_UNIT       = 64 KiB
FINAL_DECODE_STALL_SECONDS          = 120
FINAL_PROCESS_TERMINATE_GRACE_SECONDS = 5

encoding_manifest_peak = 4 MiB + 1 MiB + 1 MiB         = 6 MiB
audio_probe_peak       = 4 MiB + 32 KiB + 32 KiB
                         + 256 KiB                      = 4.3125 MiB
audio_decode_peak      = 4 MiB + 1 MiB + 256 KiB       = 5.25 MiB
encoding_process_peak  = 4 MiB + 256 KiB               = 4.25 MiB
final_probe_peak       = 4 MiB + 64 KiB + 64 KiB
                         + 512 KiB                      = 4.625 MiB
final_decode_peak      = 4 MiB + 256 KiB + 64 KiB      = 4.3125 MiB
reservation_index_peak = 4 MiB + 256 KiB + 256 KiB     = 4.5 MiB
job_control_validation_peak = 4 MiB + 512 KiB + 512 KiB
                              + 64 KiB                    = 5.0625 MiB
settings_intent_commit_peak = 4 MiB + 512 KiB + 512 KiB + 1 MiB = 6 MiB
producer_settings_commit_peak = 4 MiB + 512 KiB + 512 KiB + 1 MiB = 6 MiB
pending_settings_commit_peak  = producer_settings_commit_peak     = 6 MiB
completion_settings_commit_peak = producer_settings_commit_peak   = 6 MiB
encoding_control_peak  = max(the twelve phase peaks)    = 6 MiB
require 1 <= N <= ENCODING_CONTROL_FRAME_CAP
require encoding_control_peak <= FINAL_ENCODING_CONTROL_BUDGET

audio_probe_units = (
    checked_u64_add(
        ceil_div(audio_end_sample, PROJECT_AAC_SAMPLES_PER_PACKET_BOUND),
        PROJECT_AAC_PACKET_SLACK,
    )
    when normalized argv contains one audio token else 0
)
stream_probe_units = checked_u64_add(N, audio_probe_units)
byte_probe_units = ceil_div(
    sibling_temporary_byte_count,
    FINAL_PROBE_BYTES_PER_WORK_UNIT,
)
probe_work_units = max(1, stream_probe_units, byte_probe_units)

FINAL_PROBE_TIMEOUT_SECONDS = min(
    FINAL_PROBE_TIMEOUT_MAX_SECONDS,
    checked_u64_add(
        FINAL_PROBE_TIMEOUT_BASE_SECONDS,
        checked_u64_mul(
            ceil_div(probe_work_units, 1000),
            FINAL_PROBE_TIMEOUT_PER_1000_WORK_UNITS_SECONDS,
        ),
    ),
)
```

`PROJECT_AAC_SAMPLES_PER_PACKET_BOUND` is a project encoder contract, not an
assumption about arbitrary AAC. The resolved native FFmpeg AAC-LC encoder must
report frame size 1024 in every supported FFmpeg 5/6/7 and distributed-Windows
gate. For project-generated trimmed audio, its counted packet total must be at
most `ceil_div(audio_end_sample,1024)+2`, where two covers priming/tail packets;
a runtime/encoder which cannot prove that bound is unsupported before launch.
The distributed-Windows 44,100-sample golden reports 44 counted AAC frames
against a bound of 46. This observed value is a gate fixture, not permission to
reduce the two-packet slack.
The stream term is a sum because `-count_frames` scans both video and audio, and
the byte term prevents a small packet count from assigning an unrealistically
small deadline to a large container. All additions, divisions, and the byte
count are checked after the encoder has closed/reopened/measured the sibling
temporary and before the final FFprobe child is launched.

This bound includes replay/container objects, directory enumeration, image hash
and IHDR parsing, manifest serialization, and the existing drained audio PCM
buffer. Image hashing plus manifest serialization may coexist only in
`encoding_manifest_peak`; both are released before the isolated audio probe/
decode phase. Audio rehash likewise runs after releasing image/JSON streams.
Every stream/parser/ring is released before final-container probe. Settings
transactions are isolated phases: their one stream, JSON state, and typed object
are released before FFmpeg/probe/decode or another settings transition. Encode,
probe, and full-decode phases use the separate fixed byte caps above and may not
retain a manifest/image/audio/settings stream buffer. No additional variable-
cardinality container is allowed. Reservation descriptor/index construction is
its own isolated streaming phase, released before producer settings or FFmpeg;
the persisted files, not a resident entry list, carry crash state afterward.

Encoding FFmpeg uses `-nostats -progress pipe:1` and one project-owned bounded
byte drain. A legal stdout progress line, including its terminator, is at most
`FINAL_ENCODE_LINE_BYTES` raw bytes and is exactly
`[a-z0-9_]{1,32}=<value>` followed by LF or CRLF, where `<value>` is zero or more
printable ASCII bytes `0x20..0x7e`. NUL, non-ASCII, a missing terminator at EOF,
an empty/overlong key, or any other syntax is malformed. Legal lines are parsed
and immediately discarded, so cumulative progress length is neither resident
nor diagnostic-ring input.

`frame`, `out_time_us`, legacy-compatible `out_time_ms`, and `total_size` require
canonical nonnegative decimal `0|[1-9][0-9]*` fitting `u64`; `total_size` alone
also accepts exact `N/A`, which has no semantic value. Track the maximum numeric
value for each. `progress` accepts only exact `continue` or `end`, controls child
state, and never resets the clock; successful encode requires one terminal
`progress=end` before clean process exit. Every other syntactically legal key,
including `fps`, `bitrate`, `speed`, `dup_frames`, `drop_frames`, and
`stream_0_0_q`, is silently discarded. It neither resets the clock nor enters
the diagnostic ring, and a future FFmpeg version may add another legal key
without changing failure behavior.
`progress=continue` may repeat; exact `progress=end` occurs exactly once and is
the final legal stdout record. A line or raw byte after it is a parser failure.

The semantic stall clock starts at successful child launch and resets only when
at least one of the four numeric maxima strictly increases. Repeated values,
regressions, `N/A`, either legal `progress` control, ignored legal keys, stderr
activity, and cancellation polling do not reset it. There is no total encode wall
deadline because valid video length is variable;
`FINAL_ENCODE_STALL_SECONDS` without semantic advance raises
`FinalEncodeTimeoutError`.

Only raw stderr and malformed/overlong stdout-line bytes enter the fixed tail
diagnostic ring. A malformed/overlong line or diagnostic bytes beyond the cap
marks the encode failed while the drain continues until termination/reap; legal
ignored progress keys can never cause diagnostic overflow. Encode timeout,
cancellation, parser/pipe failure, and output overflow all close stdin,
terminate, continue bounded drainage, wait
`FINAL_PROCESS_TERMINATE_GRACE_SECONDS`, kill if needed, and reap exactly once.
No timeout path may retain/publish the sibling output or either manifest. Audio
probe and final FFprobe use the
finite-output mode of `BoundedProcessCaptureV1`; PCM and final full-decode use
its streaming-drain mode. All five child phases share its one
cancel/terminate/grace/kill/
reap state machine and the separate caps/timeouts above.
`subprocess.run(..., capture_output=True)` and `communicate()` into an unbounded
bytes/string object, the current assembled-encoder capture path, and
`get_video_info_ffprobe` are forbidden in final publication. The implementation
must drain all configured pipes without deadlock on Windows and POSIX and must
count raw bytes before UTF-8 decoding. Any helper thread, event-loop object,
pipe wrapper, decoder, or native stack used for that drain is charged to and
measured inside `FINAL_ENCODING_CONTROL_OVERHEAD`; there is no unaccounted
process-I/O worker. FFmpeg/ffprobe subprocess address space
is not misrepresented as coordinator residency and remains governed by the
supported-runtime integration gates.
The committed manifest is retained as a final artifact, never intermediate
cleanup input.

### Transaction Trust Boundary

Content manifests, input revalidation, output validation, and the job writer
lock authenticate project-controlled transactions, not hostile filesystem
observation. The lock excludes every writer in this project. The operator must
prevent external processes from mutating or replacing job-root inputs, selected
source audio/video, encoder temporaries, and final output while a transaction is
running. An external process which changes bytes after the prepass and restores
them before the postpass is an unsupported ABA mutation and can evade this
model. Consequently this specification says **pre/post-validated inputs**, not
"the exact bytes FFmpeg consumed." A stronger adversarial model requires a
separately reviewed immutable snapshot or handle-only media pipeline and its
additional disk/resource contract. The same boundary applies to Quality guide
and native-geometry reads and to full-decode/hash validation of the final video.

### Final Video Publication Manifest

Every successful orchestrated job final encode through the assembled or direct
method named in the authority map, regardless of `keep_intermediates`, retains
`final_video_manifest.json` at the job root. Batch stitching is excluded. The
manifest has exactly:

```text
schema_version:                1
algorithm_version:             "final-video-publication-v4"
relative_path:                 string
sha256:                        string
byte_count:                    integer
input_stage_fingerprint:       string
resolved_encoding_arguments:   list[string]
encoding_settings_fingerprint: string
fingerprint:                   string
```

`resolved_encoding_arguments` is the normalized copy of the exact argument
vector passed to the successful FFmpeg process, including the actually selected
encoder after `auto` resolution, frame rate, clip bounds, audio policy/options,
filters, pixel format, VR format/resolution, and output options. Normalization
is whole-argument replacement only and has this closed ASCII grammar:

```text
argv[0] = "ffmpeg"
@image-sequence:sha256=<64-lowercase-hex>:view=left
@image-sequence:sha256=<64-lowercase-hex>:view=right
@image-sequence:sha256=<64-lowercase-hex>:view=frames
@audio:sha256=<64-lowercase-hex>:bytes=<canonical-decimal>:stream=a:0
@output:path64=<unpadded-base64url>
```

Every image token's hash is the encoding-input manifest self-fingerprint.
`canonical-decimal` is `0` or a nonzero digit followed by digits, although a
selected usable audio file is necessarily positive. `path64` is unpadded RFC
4648 base64url of UTF-8 bytes for the canonical nonempty output-root-relative
POSIX final path. It normalizes the sibling temporary output to that intended
final path; the runtime sibling is exactly the path persisted by
`FinalEncodingReservationV1`, and the normalized manifest never exposes its
temporary name. Its grammar is one or more
`[A-Za-z0-9_-]` characters, and decoding then canonical re-encoding must reproduce
the token exactly.

Assembled argv has image input 0 (`view=frames`) and optional audio input 1.
Direct argv has image inputs 0/1 (`left`/`right`) and optional audio input 2.
Only the complete path element immediately following each corresponding `-i`
and the final output element may be replaced. Any embedded path, unrecognized
path-bearing option, wrong input index, duplicate/missing token, absolute path,
process ID, temporary UUID, wall time, hardware probe output, or newly
re-resolved setting is fatal. Every other argv element and its order are copied
byte-for-byte after strict ASCII validation; paths never undergo substring
replacement or escaping. The fingerprint is:

```text
encoding_settings_fingerprint = SHA256(canonical_json({
    "input_stage_fingerprint": input_stage_fingerprint,
    "resolved_encoding_arguments": resolved_encoding_arguments,
}))
```

`relative_path` is exactly the persisted
`output_info.expected_final_relative_path` and follows its stricter one-segment
containment rules; it is never reconstructed from current settings.
`input_stage_fingerprint` is exactly the validated
`EncodingInputSequenceManifest.fingerprint`; no diagnostics/RGB-stage manifest,
generic mtime metadata, or independently selected hash may substitute. An audio
path token likewise contains the complete raw pre/post-validated audio/source
identity selected above. A settings fingerprint alone is never an input
identity.

After resolving `auto`, the encoder constructs the runtime argument vector and
its normalized copy together before execution, verifies their positional
one-to-one mapping, requires the indexed sibling path absent and the final-target
descriptor ready, and retains both unchanged with the running process. On
FFmpeg success and process exit, it revalidates every encoding input, then opens
that exact indexed sibling temporary output without following a link. It applies the exact
container validator below, hashes and measures the file, fsyncs that open handle,
and closes it. Only then does it atomically replace the already existing reserved
final path and durably sync the final-output directory where supported. It
reopens the final path
without following a link and verifies the same byte count and SHA-256. It next
atomically commits the revalidated canonical encoding-input manifest and then
the canonical self-fingerprinted final-video manifest from their separately
indexed/fsynced target-local extents, replacing each declared placeholder or old
target and syncing the job-root directory after each replacement.
If recovery finds that indexed sibling before final publication, it validates
its safe path shape, removes it, syncs the parent, and reruns FFmpeg; it never
guesses whether partial or complete media is reusable. A wrong-type sibling is a
transaction conflict, and an absent active index provides no cleanup authority.
Windows uses `FlushFileBuffers`; unsupported directory-sync operations are
recorded but do not permit skipping file flushes. The raw final-video manifest
SHA-256 plus its bound input-manifest fingerprint are the only evidence accepted
by a later prune.

Container validation is `final-video-container-validation-v4` and has one
implementation. Its probe command is exactly:

```text
ffprobe -v error -count_frames \
  -show_entries stream=codec_type,codec_name,width,height,pix_fmt,sample_aspect_ratio,display_aspect_ratio,avg_frame_rate,nb_read_frames \
  -of json <sibling-temporary>
```

It never requests `format`, tags, program/stream-group contents, chapters,
packets, or frames. Supported FFprobe JSON writers may nevertheless emit empty
section wrappers. `BoundedProcessCaptureV1` drains stdout and stderr concurrently
into the exact 64-KiB caps above. Exceeding either cap terminates/reaps the probe
and raises `FinalContainerProbeBudgetError`; invalid UTF-8, a NUL, malformed
JSON, a JSON nesting depth greater than four, a string longer than 128 bytes, or
more than two authoritative stream objects fails validation.

The root object permits exactly `streams`, `programs`, and `stream_groups`, each
at most once and in any order. `streams` is required. For a project-generated
final MP4, each optional wrapper must be an empty array; a first wrapper element
fails immediately. Any other root key, missing `streams`, or key outside the
requested top-level stream-field allowlist fails. A fixed-schema incremental
parser consumes stdout directly into at most two `FinalProbeStreamV1` scalar
records and remains within `FINAL_PROBE_JSON_STATE_BYTES`; a general
`json.loads` object tree is forbidden even after the byte cap. Nonzero exit or
any stderr byte also fails. The probe must exit within the derived
`FINAL_PROBE_TIMEOUT_SECONDS`; timeout raises
`FinalContainerProbeTimeoutError` after the common termination sequence.

Require exactly one video stream, no data/subtitle/attachment streams, the
resolved output width and height, `sample_aspect_ratio="1:1"`,
`display_aspect_ratio` equal to the reduced positive ratio
`output_width:output_height`, and `pix_fmt=yuv420p`. `N/A`, a missing field, an
unreduced or numerically different ratio, and zero/negative ratio components all
fail. The exact codec-name map
is `libx264 -> h264` and `libx265|hevc_nvenc -> hevc`; no other resolved encoder
token validates. Require reduced `avg_frame_rate` equal to the resolved output
rational and `nb_read_frames ==
EncodingInputSequenceManifest.frame_names.length`. If
`nb_read_frames` is unavailable or noninteger, validation fails rather than
substituting duration. Require zero audio streams when normalized argv contains
no audio token and exactly one `codec_name=aac` audio stream when it contains one;
`preserve_audio=true` with no usable selected stream therefore still requires
zero. The accepted audio stream also requires canonical integer
`nb_read_frames` in `1..audio_probe_units`; exceeding the preflight packet bound
is `FinalContainerAudioPacketBoundError` even if the probe finished before its
deadline. Finally run exactly:

```text
ffmpeg -v error -xerror -nostats -progress pipe:1 \
  -i <sibling-temporary> -map 0:v:0 -map 0:a? -f null -
```

Stdout is a streaming progress channel, not a cumulative capture. Each ASCII
`key=value` line is bounded by `FINAL_DECODE_LINE_BYTES`, parsed inside
`FINAL_DECODE_PROGRESS_STATE_BYTES`, and immediately discarded. Keys are 1..32
lowercase ASCII letters/digits/underscores and values are at most 256 bytes;
malformed/overlong lines fail while drainage continues. Track canonical
nonnegative decimal `frame`, `out_time_us`, and legacy-compatible `out_time_ms`
values plus `progress=end`. The `FINAL_DECODE_STALL_SECONDS` clock resets only
when `frame` or either time counter strictly increases, not for repeated or
unrelated lines.
There is no fixed total decode deadline because valid video duration is
variable. A stall raises
`FinalContainerDecodeTimeoutError`.

Stderr is simultaneously drained into the fixed
`FINAL_DECODE_DIAGNOSTIC_BYTES` tail ring. A byte beyond the cap sets overflow,
continues draining through process termination/reap, and fails validation with
`FinalContainerDecodeBudgetError`. Because all other stream types were rejected,
this completely decodes every accepted stream. Require `progress=end`, exit
status zero, and no stderr byte. Timeout, cancellation, progress/parser failure,
pipe failure, and budget overflow all close stdin, terminate, continue bounded
drainage, wait `FINAL_PROCESS_TERMINATE_GRACE_SECONDS`, kill if needed, and reap
exactly once before returning. No failure may publish or authenticate the file.
File existence or nonzero length alone is never validation.

A stale, missing, malformed, or mismatching encoding-input or final-video
manifest never authenticates an existing video. In particular, a crash after
video publication but before final-video-manifest commit requires a new encode to
a temporary and atomic replacement; resume must not inspect the container or
re-resolve `auto` to guess provenance.
Both manifests are retained final artifacts and are never listed for
intermediate cleanup. Final-encoding disk preflight physically reserves their
schema-derived target-local temporary allocations and any required final-path
placeholders, the exact final-video target entry and its identity descriptor,
plus a separate extent for each remaining
producer and pending-cleanup transaction before starting FFmpeg; it authenticates
and indexes the lifecycle terminal extent already retained since job creation
rather than allocating it again.

### Final Media Audit Disposition V1

This audit is reachable only from step 7 of Recovery and Audit Entry Order.
Calling it while bootstrap/either fixed index is active or before JobControl-
owned extent reconciliation is a protocol error;
transaction-owned placeholders are therefore never candidates for `V`, `I`, or
`P`.

`FinalMediaAuditDispositionV1` is a read-only, non-persisted result with exactly
three fields:

```text
final_video_present:                bool
final_video_authenticated:          bool
final_video_authentication_reason:  null |
    "not_present" |
    "legacy_no_publication_manifests" |
    "incomplete_publication_evidence" |
    "publication_validation_failed"
```

There is deliberately no `final_video_valid` field. Authentication already
implies that the current exact container validator passed; authentication
failure says nothing about whether an unauthenticated historical container is
readable. An explicitly requested best-effort legacy probe may report container
health in a separate inspection object, but it cannot change this disposition,
manufacture provenance, or authorize pruning.

Resolve exactly one expected job final-video path from the immutable
`output_info.expected_final_relative_path`. A raw schema-1-through-4 audit which
has not migrated requires the exact retained `expected_output_filename` in
memory; it does not persist during read-only inspection. The copied field uses
`ContainedLegacyFinalComponentV1`. Absence raises
`LegacyFinalTargetUnknownError` without a filesystem lookup; malformed or
unlocatable content raises `FinalMediaTargetMetadataError` before returning a
disposition. Never invoke v1, the current output-name helper, or a glob for MP4 files.
In particular, `*_stitched_*.mp4` and every `/stitch_video` product are invisible
to this audit. Define presence bits by checking directory entries at that exact
path and the two exact job-root manifest paths without following the final
component:

```text
V = an entry exists at the expected final-video path
I = an entry exists at encoding_input_manifest.json
P = an entry exists at final_video_manifest.json
```

`final_video_present` equals `V`. Authentication requires all three entries to
be non-link regular files, their final path/token to equal the persisted target,
the producer marker to be exactly `4` plus a valid generation, and the complete
binding/container contract above to validate. An unsafe entry kind or invalid
marker therefore never authenticates even though its presence bit is one.

The publication-contract boundary is settings schema 5, not a package or
project version. Read legacy provenance from raw settings metadata before any
migration:

```text
raw settings_schema_version in 1..4:
    audit_source_schema = raw settings_schema_version
raw settings_schema_version absent and the saved-schema-1 parser succeeds:
    audit_source_schema = 1
raw settings_schema_version == 5 and
source_settings_schema_version is a non-boolean integer in 1..5:
    audit_source_schema = source_settings_schema_version
otherwise:
    audit_source_schema = unknown
```

Migration uses `SavedSettingsResult.source_version` and persists it as specified
in the settings contract, so a migrated schema-5 file retains its pre-migration
value. `metadata.project_version`, `pyproject.toml`, runtime package version,
mtime, and filename dates are forbidden substitutes. Source schema describes
job ancestry only; it is never sufficient evidence of the current video's
producer. A migrated source-schema-4 job remains a legacy candidate only until
the first publication-v4 pre-launch attempt marker commits. If both manifests
disappear after that boundary, `V/I/P=1/0/0` is
`incomplete_publication_evidence`, regardless of the permanently retained
source value. Legacy classification is true if and only if all of these
predicates hold:

```text
V=1 and I=0 and P=0
audit_source_schema in 1..4
raw persisted metadata.processing_status == "completed"
final_media_producer_contract_version is exactly null
final_media_publication_generation is exactly null
the expected final-video entry is a non-link regular file
```

Missing, malformed, or contradictory settings metadata, a non-completed status,
an unknown/non-null producer marker, or an unsafe video entry makes that
predicate false. Apply this complete matrix
in row order; `valid` in the last two rows means the three artifacts validate as
one bound publication identity:

| V | I | P | Additional result | `present` | `authenticated` | `reason` |
|---:|---:|---:|---|---:|---:|---|
| 0 | 0 | 0 | none | false | false | `not_present` |
| 0 | 0 | 1 | any | false | false | `incomplete_publication_evidence` |
| 0 | 1 | 0 | any | false | false | `incomplete_publication_evidence` |
| 0 | 1 | 1 | any | false | false | `incomplete_publication_evidence` |
| 1 | 0 | 0 | legacy predicate true | true | false | `legacy_no_publication_manifests` |
| 1 | 0 | 0 | legacy predicate false | true | false | `incomplete_publication_evidence` |
| 1 | 0 | 1 | any | true | false | `incomplete_publication_evidence` |
| 1 | 1 | 0 | any | true | false | `incomplete_publication_evidence` |
| 1 | 1 | 1 | valid | true | true | null |
| 1 | 1 | 1 | invalid | true | false | `publication_validation_failed` |

Ordinary inspection of legacy final media is strictly read-only: preserve the
video, do not encode, delete, rewrite completion state, call it corrupt, or
synthesize either manifest. Its stereo-payload reuse and historical-diagnostics
facts are audited independently. The legacy video cannot authorize a new
`payload_pruned` commit. An explicit reencode request may create current
publication evidence only when the exact current image/audio inputs and their
stage evidence still validate and the target also passes
`PortableFinalComponentV1`; a contained-but-nonportable legacy target fails
before any write with `LegacyFinalTargetNotPortableError`. When cleanup from the
historical job removed
those inputs, preserve the video indefinitely and report it as unauthenticated;
container probing, full decode, or hashing can inspect bytes but can never infer
the missing executed command or manufacture publication identity.

### Metadata State Machine

`04_stereo_diagnostics/metadata.json` has exactly these keys:

```text
schema_version:                     1
algorithm_version:                  "stereo-coverage-sidecar-v1"
status:                             "building" | "complete" |
                                    "legacy_fast_unavailable" |
                                    "payload_pruned"
rgb_stage_fingerprint:              string
source_guide_fingerprint:           string
native_geometry_fingerprint:        string | null
quality_input_manifest_sha256:      string | null
frame_names:                        list[string]  # unique source-order stems
render_shape:                       [H, W]
render_mode:                        "fast" | "quality"
geometry_mode:                      "relative" | "metric"
occlusion_fill:                     "none" | "background"
mask_payloads_enabled:              bool
stats_schema_version:               1
frame_manifest_schema_version:      1
ordered_frame_manifest_fingerprint: string | null
frames_jsonl_sha256:                string | null
summary_sha256:                     string | null
metric_clamp_summary_sha256:        string | null
pruned_from:                        null | PrunedFrom
final_video_manifest_sha256:        string | null
prune_entries:                      null | list[PruneEntry]
fingerprint:                        string
```

`PrunedFrom` has exactly `previous_status`,
`previous_diagnostics_fingerprint`,
`ordered_frame_manifest_fingerprint`, `frames_jsonl_sha256`, `summary_sha256`,
and nullable `metric_clamp_summary_sha256`. `previous_status` is `complete` or
`legacy_fast_unavailable`. `PruneEntry` has exactly:

```text
stage_key:                 string
relative_path:             string
operation:                 "delete_tree"
cleanup_contract_version: 2
directory_identity:        DirectoryIdentity
marker:                    PruneMarker
```

`DirectoryIdentity` is the shared closed union defined by Reservation Extent V1.
Its POSIX `file_identity` values come from the already opened directory handle;
Linux obtains the stable filesystem UUID plus opaque file handle and separately
captures the invocation-only `statx` mount identity. Its
Windows `file_identity` comes from `GetFileInformationByHandleEx(FileIdInfo)`.
A platform/filesystem which cannot provide every shared field cannot prune and
reports an actionable cleanup error.

`PruneMarker` has exactly `name` and `sha256`. Final-encoding reservation has
already created and authenticated a zero-length final marker placeholder plus a
separate target-local payload extent inside each opened target root. Immediately
before the `payload_pruned` commit, consume that pair through write/truncate/
same-parent replace/fsync to publish a marker named
`.depth-surge-prune-v2-<32-lowercase-hex>.marker`, where the token is 128 bits
from the OS CSPRNG. Its bytes are canonical ASCII JSON plus LF with exactly the
keys `cleanup_contract_version=2`, `relative_path=<the entry path>`, and
`token=<the same 32-lowercase-hex suffix>`. Fsync the marker and containing
directory, capture `DirectoryIdentity` from the still-open root
handle, re-resolve the path without following links, and require the same
identity. `sha256` hashes the complete marker bytes.

`prune_entries` records the concrete paths approved by the read-only audit, in
pipeline order with no duplicates. Each path is canonical output-root-relative
POSIX form, contains no absolute prefix or `..`, and names a directory which was
inside the acquired job root at audit and revalidation. The diagnostics-frame
entry uses `stage_key="stereo_diagnostics_frames"` and
`relative_path="04_stereo_diagnostics/frames"`. Stage keys are display/audit
labels only; cleanup never resolves them through the current
`INTERMEDIATE_DIRS` mapping.
Each authorized POSIX root must also share the acquired job root's current
`InvocationMountIdentityV1`; each Windows root must share its
`file_identity.volume_serial`. A stage mounted underneath the job root is not a
prune target, and no persisted mount number is consulted after restart.

Resume supports the committed `cleanup_contract_version` or performs no deletion
and reports an actionable version error. In the normal recursive path it
validates containment, exact `DirectoryIdentity`, strict marker
name/content/hash, and cleanup version for every present entry, then deletes only
the persisted relative paths. A missing marker can enter only the terminal
empty-root rule below and can never enter recursive traversal. A different
ordinary directory at the same path is an identity mismatch, even when it is
inside the job root and contains a copied filename. Mapping changes in a future
release cannot redirect or omit a historical authorization. Final video, all
retained content/publication manifests, retained JSONL/root summary, and the
diagnostics metadata root are forbidden prune targets.
An already-absent persisted path is an idempotent successful no-op. A symlink,
junction, mount point, or other reparse-point replacement at that path is never
followed or recursively deleted; it fails cleanup with an actionable identity
error and leaves every other unprocessed entry untouched. Recursive traversal
also never follows a descendant link or reparse point; it may unlink that entry
itself but cannot visit or delete its target. Every POSIX descendant directory
must match the job root's current `InvocationMountIdentityV1`; every Windows
descendant must match committed
`file_identity.volume_serial` and must not be a reparse point. A mount/bind-
mount, identity-read failure, mismatching marker, missing
marker outside the terminal empty-root rule below, or volume crossing preserves
the residual tree and reports the offending path.
Enumeration, identity checks, child opens, and unlink/remove operations are all
handle-relative to the verified root; cleanup never switches back to recursive
path-string APIs after authorization.

A marker placeholder or fully written marker absent from committed
`prune_entries` is inert transaction state and never authorizes tree deletion.
Its reserved name is excluded from ordinary payload listings. Only its matching
live final-encoding index plus descriptor/identity, or terminal-failure evidence
which authenticates that unused indexed entry, may complete or unlink that one
placeholder/marker; content or name alone is insufficient. An unindexed entry is
preserved as a conflict. During authorized
cleanup, keep the committed marker until every other descendant has been
removed. Unlink the marker only after a handle-relative enumeration proves that
it is the sole remaining entry, then immediately remove the already-open root
handle-relative.

There is exactly one marker-absence recovery rule. If a restart opens the
persisted root without following a link and the committed marker is absent, it
may call handle-relative `rmdir` on that root only when the current
`DirectoryIdentity` exactly equals the committed identity, the root itself is
not a mount/reparse point, its current invocation mount or volume still equals the acquired
job root, and one complete handle-relative enumeration reports zero entries.
The exception authorizes no unlink, child open, traversal, or recursive delete;
the presence of even one entry makes it inapplicable. This closes only the crash
window between marker unlink and root removal and cannot authorize a copied,
repopulated, or replacement directory.

State-dependent nullability is exact:

| Status | Ordered manifest | JSONL/summary | Current clamp hash | Prune fields |
|---|---|---|---|---|
| `building` | null | both null | null | all null |
| `complete` | non-null | both non-null | metric non-null, relative null | all null |
| `legacy_fast_unavailable` | hash of `[]` | both non-null | legacy metric non-null, relative null | all null |
| `payload_pruned` | null | both non-null | null | all non-null |

`payload_pruned` always has `mask_payloads_enabled=false`; the previous policy
is authenticated indirectly by `pruned_from.previous_diagnostics_fingerprint`.
Quality requires `source_guide_fingerprint`, `native_geometry_fingerprint`, and
`quality_input_manifest_sha256` to match its retained content manifest in every
state. Fast retains its existing source provenance string and requires the latter
two fields null. `payload_pruned` preserves the Quality values because that
manifest remains a historical final artifact.

Before any Quality frame transaction, the content manifest and
`StereoRgbMetadataV2` are already durable and mutually validate.
Before any frame transaction in either mode, atomically write `building`
metadata with the four aggregate hashes null and all three prune fields null.
After every frame manifest validates, Quality first reproduces and revalidates
its complete input content manifest. Then write JSONL, summary, and metric compatibility summary to
same-directory temporaries, replace them, then atomically write `complete`
metadata last with their hashes and prune fields null. Relative mode requires
`metric_clamp_summary_sha256=null`; metric mode requires the hash of
`04_left_frames/clamp_summary.json`.
The diagnostics stage fingerprint is the metadata `fingerprint`; because it is
computed without itself but over all aggregate hashes, JSONL, summary, ordered
manifests, and metric compatibility output participate without a recursive hash.

A valid `building` resume validates and reuses committed frame manifests,
rerenders only missing/invalid frames, and rebuilds aggregates. A valid
`complete` stage with missing or corrupt JSONL, root summary, or metric clamp
summary is atomically demoted to `building` and rebuilds those derived files
without rerendering RGB when every frame transaction remains valid. A bad frame
manifest or its payload requires that frame's stereo rerender.

An absent diagnostics root may become `legacy_fast_unavailable` only for a saved
schema 1 through 4 job whose complete Fast v3 RGB stage and, for metric, every
old per-frame clamp sidecar plus summary all validate. It writes no frame
directories, an empty JSONL, and a strict root summary with
`availability="legacy_fast_unavailable"`; ordered manifest hash is the hash of
canonical `[]`. Its complete-like metadata hashes those two aggregate files,
has `mask_payloads_enabled=false`, and has all prune fields null. A legacy metric
stage additionally hashes its validated old clamp summary; legacy relative uses
null. It never fabricates counts.

Any missing or corrupt RGB, metadata, old clamp sidecar, or clamp summary in a
legacy candidate forbids partial repair: set `P=N`, redraw the complete stereo
stage, and finish as ordinary `available` diagnostics. A legacy stage also sets
`P=N` when masks are requested because they cannot be reconstructed from RGB.
The current per-frame legacy repair behavior is not used after schema 5.

After final video encoding and both durable final manifests validate,
`keep_intermediates=false` first creates and validates the prune markers and
directory identities, then consumes the indexed
`diagnostics_payload_pruned` target-parent extent to atomically transition
`complete` or `legacy_fast_unavailable` metadata to `payload_pruned`. The new
object copies
the prior hashes into `pruned_from`, retains the current JSONL and root summary
hashes, sets the current ordered-manifest and metric-compatibility hashes null,
records the raw `final_video_manifest.json` SHA-256 and ordered `prune_entries`,
and fingerprints that state. Only after this commit may cleanup delete every
listed intermediate stage, per-frame diagnostics directories/manifests, and the
metric compatibility summary. Extra old payloads inside a listed delete-tree
root may disappear when that exact entry resumes; an unlisted path is never
deleted. Their presence never makes them reusable.

Finalization order is exact: durably publish and validate the final video,
durably commit and validate `encoding_input_manifest.json`, durably commit and
validate `final_video_manifest.json`, consume/fsync every indexed target-local
prune-marker extent and recapture the matching open-handle directory identities,
construct the sole `payload_pruned` object from the authenticated prior metadata,
final-video-manifest raw hash, and index-ordered prune entries, consume/fsync/
truncate/replace/reopen its indexed diagnostics-metadata extent, then run
`SettingsArtifactTransactionV1` for `cleanup_status="pending"`, run authorized
cleanup from the persisted `prune_entries`, then commit one terminal settings
transaction containing the cleanup result and processing completion fields. A
crash after the prune commit resumes settings, cleanup, and final status only;
it never reopens a render transaction or consults current stage mappings. A crash
after both manifests commit but before prune validates their canonical bytes,
fingerprints, normalized executed arguments, and video payload, then retries only
the prune transition. A crash before final-video-manifest commit must reencode
even if a complete MP4 or encoding-input manifest is present.
With `keep_intermediates=true`, finalization skips only prune and cleanup: the
encoding-input and publication manifests still commit before the
settings/runtime completion mark.

The non-semantic settings/runtime completion record persists one
`cleanup_status` value: `not_requested`, `pending`, `complete`,
`incomplete_identity_mismatch`, or `incomplete_error`. It is not part of RGB,
diagnostics, encoding-input, or final-video identity. A restart which cannot
reproduce the persisted object/volume identity or the current invocation's
no-crossing proof leaves the final video and
both final manifests successful, sets
`cleanup_status="incomplete_identity_mismatch"`, performs no render or encode,
and leaves the residual path for an explicit later cleanup or manual action.
Other cleanup failures use `incomplete_error`. Only successful absence/removal
of every persisted entry sets `complete`; none of these incomplete values may
demote `payload_pruned` or reopen frame processing.
`keep_intermediates=true` commits `not_requested`; the `payload_pruned` commit
and any resumed cleanup first persist `pending`; each terminal result above is
then atomically persisted without changing final-video success.

A cleanup-only resume which does not enter final encoding must reopen the same
authenticated reservation index. If `pending` is already durable, only the
lifecycle `job_terminal_settings` extent remains; otherwise the indexed
`cleanup_pending_settings` extent plus that terminal extent must both remain.
It allocates neither again and consumes them in the same fixed order. Cleanup is
forbidden when either descriptor/index, the settings memory phase, or reopen
validation cannot be proved; an authenticated final video remains successful
but the job is not reported terminally completed.

A valid `payload_pruned` state authenticates only metadata, the applicable
retained Quality content manifest, retained JSONL/root summary, the raw
encoding-input/publication manifests, and the final video.
Historical statistics remain readable but no stereo or upstream intermediate
payload is reusable. Merely opening or auditing a completed job performs no
render. Any requested processing resume, mask generation, or invalid final-video
recovery sets `P=N` and starts a fresh `building` stage after preflight.

Audit never demotes `payload_pruned` to `building`. It reports
`FinalMediaAuditDispositionV1`, `historical_diagnostics_valid`, and the constant
`stereo_payload_reusable=false` independently. A true
`final_video_authenticated` requires the video, encoding-input manifest, and
publication manifest to validate as one bound identity. When those remain valid
but JSONL or root summary is
missing/corrupt, ordinary inspection performs no writes or render, preserves
final-video success, and
reports historical diagnostics as damaged and irrecoverable. A later explicit
request to process again, regenerate masks, or rebuild diagnostics follows the
matrix with `P=N`; deleted frame payload is never assumed available. Invalid
publication evidence similarly cannot be repaired from retained aggregates.

The mask-policy transition matrix is mandatory:

| Existing state | Requested policy | Action | `P` | Manifest-only `R` |
|---|---|---|---:|---:|
| no reusable current or legacy stage | either | create a new available diagnostics stage | `N` | 0 |
| `building`/`complete`, same policy | same | reuse valid transactions, redraw invalid | invalid frame count | 0 |
| `building`/`complete`, false | true | masks are unrecoverable; redraw every frame | `N` | 0 |
| `building`/`complete`, true | false | validate RGB/stats, rewrite all manifests without masks, rebuild aggregates, then delete masks | invalid frame count | valid frame count |
| intact legacy Fast | false | write `legacy_fast_unavailable` | 0 | 0 |
| legacy Fast | true, or any legacy damage | full upgrade to available diagnostics | `N` | 0 |
| `payload_pruned` | either | no payload reuse; full rebuild when processing is requested | `N` | 0 |

`P` always counts full frame transactions which encode RGB and stats; `R`
counts manifest-only mask-removal migrations and never includes a frame already
in `P`. A true-to-false migration does not invalidate downstream stages because
RGB and stats hashes are unchanged. Any `P>0` follows normal downstream
invalidation. These values feed progress, disk preflight, and execution before
any destructive mutation.

The threaded stereo executor creates exactly `P` lazy lifecycle items in source
order. Only those items enter decode/render/write capacity or set
`repaired_outputs`. After every pipeline thread has stopped and joined, the
coordinator calls `migrate_frame_manifests_sync` over the source-ordered `R`
action entries. `R` never enters a feeder, decoder, writer, lifecycle permit, or
queue, even when `P>0`. Consequently `P=0,R>0` creates no pipeline thread and is
a fully specified synchronous path.

Completion advances once for each committed `P` item and then once for each
committed `R` item; aggregate consolidation remains a separate final phase.
For each `R` frame, the coordinator validates the complete existing RGB, stats,
and mask payloads, atomically replaces its manifest with null mask payloads, and
only then advances progress. A per-source-index fault-injection point exists
before temporary write, after temporary fsync, after replace, and after
directory sync. No old mask is deleted until all `R` manifests commit. Then a
source-order sweep deletes only masks which no current manifest references,
before aggregate consolidation.

After a crash, the read-only audit derives actions again: a valid new
mask-disabled manifest is reuse, a valid old mask-enabled transaction is `R`,
and an invalid RGB/stats transaction is `P`, regardless of an orphan temporary
or extra unreferenced masks. This makes partial migration idempotent. A crash
may leave extra masks but can never leave a committed manifest which requires a
deleted mask.

### Frame Statistics Schema

Each `stats.json` has exactly:

```text
schema_version:     1
frame_name:         string
render_mode:        "fast" | "quality"
geometry_mode:      "relative" | "metric"
occlusion_fill:     "none" | "background"
render_shape:       [H, W]
eyes:               {"left": EyeStats, "right": EyeStats}
metric_projection:  null | MetricProjectionStats
geometry_nearest:   null | GeometryNearestStats
```

`MetricProjectionStats` has exactly `valid_pixel_count`,
`clamped_pixel_count`, and `clamped_fraction`. It is null in relative mode and
required in metric mode. `GeometryNearestStats` has exactly:

```text
indexed_sample_count
query_count
visited_nodes_total
visited_nodes_max
```

It is null for Fast and required for both Quality fill modes. Its indexed count
is exactly the frame's native primitive count `G_i`; the other three values
cover all retained-zero primitive queries in source output-row-major order. A
frame with no such query uses integer zeros for all three query/visit fields.
`EyeStats` has exactly:

```text
pixel_count
state_pixel_counts
state_pixel_ratios
hole_runs
backend_lane_counts
backend_pixel_counts
backend_component_counts
final_unresolved_lane_count
quality_limits
quality_budgets
```

`state_pixel_counts` and `state_pixel_ratios` each have exactly
`prefill_partial`, `prefill_full`, `local_filled`, `post_local_residual`,
`exemplar_filled`, `fallback_filled`, and `final_unresolved`. Ratios use
`pixel_count`.

`hole_runs` has exactly `count`, `lane_count_histogram`,
`touched_pixel_span_histogram`, `lane_count_max`, `lane_count_p95`,
`touched_pixel_span_max`, `touched_pixel_span_p95`,
`physical_width_px_max`, and `physical_width_px_p95`. A histogram is an
ascending list of unique `[positive_value, positive_count]` pairs. Maxima and
p95 are independently derived from histograms; physical width derives from the
lane histogram divided by 16. Empty histograms use the zero rules above.

`backend_lane_counts` and `backend_pixel_counts` each have exactly `local`,
`exemplar`, and `fallback`. `backend_component_counts` has exactly
`availability`, `local`, `exemplar`, and `fallback`. Quality background
availability is `quality_segment_graph` and all three counts are integers. Fast
availability is `unavailable_fast_no_segment_graph`; Quality with
`occlusion_fill=none` uses
`unavailable_quality_none_no_repair_graph`. Both unavailable cases require all
three values to be null. Quality none never constructs segment records or a
component graph merely for diagnostics.

`quality_limits` is null for Fast and Quality none. For Quality background it
has exactly `max_neighbour_abs_q_jump_px`, `predicted_gap_px`, and
`local_limit_px`.
`quality_budgets` is null for Fast. For Quality it has exactly:

```text
availability
segment_record_count
segment_table_bytes
local_slot_charge
local_neighborhood_sample_charge
local_budget_skipped_run_count
exemplar_evaluations
fallback_indexed_donor_count
fallback_query_count
fallback_visited_nodes_total
fallback_visited_nodes_max
fallback_visited_nodes_p95
```

Quality background uses `availability="quality_repair_plan"` and requires every
counter to have its numeric type below. The three local fields and
`fallback_visited_nodes_total` are the per-eye deltas of the shared frame
counters; checked left-plus-right sums must not exceed their corresponding
frame caps. Quality none uses
`availability="unavailable_quality_none_no_repair_plan"` and requires all eleven
counters to be null, not fabricated zeros. Its backend lane and pixel counts are
ordinary integer zeros because no repair write occurred; only graph-derived
component and budget values are unavailable.

### Frame Manifest and Transaction

Each `manifest.json` has exactly:

```text
schema_version:                1
algorithm_version:             "stereo-coverage-sidecar-v1"
frame_name:                    string
render_mode:                   "fast" | "quality"
geometry_mode:                 "relative" | "metric"
occlusion_fill:                "none" | "background"
render_shape:                  [H, W]
rgb_stage_fingerprint:         string
mask_payloads_enabled:         bool
quality_limits:                null | {"left": QualityLimits,
                                       "right": QualityLimits}
payloads:                      PayloadMap
fingerprint:                   string
```

`QualityLimits` has exactly `max_neighbour_abs_q_jump_px`, `predicted_gap_px`,
and `local_limit_px`. `quality_limits` is null for Fast and Quality none. For
Quality background, its two objects must equal the corresponding `stats.json`
eye values exactly; this repetition lets the manifest assert the
geometry-dependent limits without placing a frame list in stage identity.

`PayloadMap` has exactly `left_rgb`, `right_rgb`, `left_coverage`,
`left_repair`, `right_coverage`, `right_repair`, and `stats`. RGB and stats
entries are required objects; mask entries are objects exactly when enabled and
otherwise null. Each object has exactly `relative_path`, `sha256`, and
`byte_count`; paths are output-root-relative POSIX paths and cannot contain
`..` or an absolute prefix.

For one frame, encode every payload to a temporary in its destination
directory. Replace left RGB, right RGB, enabled masks, and stats in that order,
then write the canonical manifest last. Metric projection statistics are inside
stats, so there is no new-render per-frame `04_left_frames/clamp_stats` file.
Any failure removes the manifest and every final payload in that transaction,
including both RGB images. Resume verifies manifest self-fingerprint, every raw
payload hash/byte count, exact stats schema, path containment, and PNG header.

Any frame rerender triggered by diagnostics sets the existing
`repaired_outputs` condition and invalidates every tracked downstream frame
stage before replacement begins, even if regenerated RGB later hashes equally.

### Root Summary Schema

`04_stereo_diagnostics/stereo_coverage_summary.json` has exactly:

```text
schema_version:                       1
algorithm_version:                    "stereo-coverage-sidecar-v1"
availability:                         "available" |
                                      "legacy_fast_unavailable"
frame_names:                          list[string]
frame_count:                          int
render_shape:                         [H, W]
render_mode:                          "fast" | "quality"
geometry_mode:                        "relative" | "metric"
occlusion_fill:                       "none" | "background"
ordered_frame_manifest_fingerprint:   string
frames_jsonl_sha256:                  string
eyes:                                 {"left": EyeAggregate,
                                       "right": EyeAggregate} | null
metric_projection:                    MetricProjectionAggregate | null
geometry_nearest:                     GeometryNearestAggregate | null
```

For available diagnostics, `EyeAggregate` has exactly `pixel_count`, summed
`state_pixel_counts`, recomputed `state_pixel_ratios`, merged `hole_runs`, summed
`backend_lane_counts`, summed `backend_pixel_counts`, summed-or-null
`backend_component_counts`, summed `final_unresolved_lane_count`,
`quality_limit_ranges`, and `quality_budget_totals`. Merged hole histograms are
sorted and all max/p95 fields are recomputed from them. Quality limit ranges
contain min/max for each of the three per-frame limit fields. Both Quality fields
are null for Fast. `quality_limit_ranges` is also null for Quality none; only
Quality background supplies its min/max object. Quality none uses the explicitly
unavailable budget object defined below. Component availability and nullability
must be the same in every frame.

The nested `state_pixel_counts`, `state_pixel_ratios`, `hole_runs`,
`backend_lane_counts`, `backend_pixel_counts`, and `backend_component_counts`
use exactly the same key sets as `EyeStats`. `quality_limit_ranges` is either
null or has exactly:

```text
max_neighbour_abs_q_jump_px_min
max_neighbour_abs_q_jump_px_max
predicted_gap_px_min
predicted_gap_px_max
local_limit_px_min
local_limit_px_max
```

`quality_budget_totals` is either null or has exactly:

```text
availability
segment_record_count                 # sum
segment_table_bytes_max              # maximum eye/frame value
local_slot_charge                    # sum
local_neighborhood_sample_charge     # sum
local_budget_skipped_run_count       # sum
exemplar_evaluations                 # sum
fallback_indexed_donor_count         # sum
fallback_query_count                 # sum
fallback_visited_nodes_total         # sum
fallback_visited_nodes_max           # global maximum
fallback_visited_nodes_frame_p95_max # max of per-frame p95 values
```

It is null for Fast. Quality background uses
`availability="quality_repair_plan"` and numeric totals. Quality none uses
`availability="unavailable_quality_none_no_repair_plan"` with every total null.

`MetricProjectionAggregate` is required only for metric mode and has exactly
`valid_pixel_count`, `clamped_pixel_count`, `clamped_fraction`, ordered
`clamped_fractions`, `affected_frame_count`, `mean_clamped_fraction`, and
`max_clamped_fraction`. Global `clamped_fraction` divides summed clamped by
summed valid; mean is the arithmetic mean of ordered per-frame fractions.

`GeometryNearestAggregate` has exactly `indexed_sample_count`, `query_count`,
`visited_nodes_total`, and `visited_nodes_max`. The first three are checked
source-order sums and the last is the global maximum. It is required for
available Quality diagnostics, null for available Fast, and null when
`availability="legacy_fast_unavailable"`.

### Numeric JSON Types and Aggregation

JSON integer means a non-boolean integer in `0..2**64-1`. JSON binary64 means a
finite Python/IEEE-754 binary64 value serialized as a floating token, so zero is
`0.0`, never integer `0`; negative zero is normalized first. Strict parsing
rejects an integer token in a binary64 field and vice versa.

The exact field types are:

- JSON integers: dimensions, frame/pixel/run/count fields, every histogram
  value/count pair, `lane_count_max`, `touched_pixel_span_max`, non-null backend
  counts, `final_unresolved_lane_count`, `predicted_gap_px`, `local_limit_px`,
  all non-null budget counters except visited-node p95, every non-null geometry
  nearest counter, `byte_count`,
  `valid_pixel_count`, `clamped_pixel_count`, and `affected_frame_count`;
- JSON binary64: all state ratios, `lane_count_p95`,
  `touched_pixel_span_p95`, both physical-width fields,
  `max_neighbour_abs_q_jump_px`, `fallback_visited_nodes_p95`, every clamp
  fraction, mean, maximum, and every float min/max range;
- nullable integers: unavailable backend component counts and unavailable
  Quality budget counters whose corresponding ordinary type is integer;
- nullable binary64: unavailable `fallback_visited_nodes_p95` and any explicitly
  nullable binary64 aggregate; no other numeric null is legal.

The same rules apply to aggregate fields: q-jump range endpoints and
`fallback_visited_nodes_frame_p95_max` are binary64; predicted/local range
endpoints and all summed/max counters are integers. Schema versions are positive
JSON integers. Strings, booleans, arrays, and objects accept only their declared
types.

Before any mutation, diagnostics cardinality preflight proves that every possible
integer is representable. Define:

```text
U64_MAX = 2**64 - 1
R_cap   = min(16*Q, QUALITY_RECORD_ARENA_BYTES // 16)
E_cap   = 32_000_000
D_cap   = Q
```

`R_cap` is the maximum per-eye record/query count, `E_cap` is the fixed per-eye
exemplar evaluation cap, and `D_cap` is the conservative maximum indexed donor
and visited-node count for one query. Per-eye/per-frame upper bounds are:

```text
pixel and backend-pixel counts             <= Q
lane, unresolved, run, histogram counts    <= 16*Q
component and segment-record counts        <= R_cap
segment-table bytes                         <= 16*R_cap
local slot charges                          <= QUALITY_LOCAL_SLOT_CAP
local neighbourhood-sample charges          <= QUALITY_LOCAL_NEIGHBOR_SAMPLE_CAP
local budget-skipped runs                   <= R_cap
exemplar evaluations                        <= E_cap
fallback indexed donors                     <= D_cap
fallback queries                            <= R_cap
fallback visited-node total                 <= QUALITY_REPAIR_FALLBACK_VISIT_CAP
fallback visited-node maximum               <= D_cap
geometry nearest indexed samples            <= G_i
geometry nearest queries                    <= Q
geometry nearest visited-node total         <= QUALITY_GEOMETRY_NEAREST_VISIT_CAP
geometry nearest visited-node maximum       <= G_i
metric valid/clamped pixels                 <= Q
```

The local and repair-fallback caps are shared by both eyes, so each per-eye
value is individually bounded by the frame cap and the two eye values must have
a checked sum no greater than that cap. For every summed field in one root eye
aggregate, use `N` times its applicable per-frame bound. Geometry-nearest root sums use checked `sum(G_i)`, `N*Q`, and
`N*QUALITY_GEOMETRY_NEAREST_VISIT_CAP`; its maximum remains bounded by `G`.
Metric root counts use `N*Q`; frame and affected-frame counts use `N`.
Histogram values use `16*W` for lane length and `W` for touched span, while their
merged counts use `N*16*Q`. File/video/manifest `byte_count` values are checked
directly from the nonnegative platform size and must also fit uint64.

The preflight first validates that `N`, `H`, `W`, `Q=H*W`, every `G_i`, every
`F_i`, and every `J_frame_i` are nonnegative uint64 values, deriving each product
with checked arithmetic rather than an unchecked language multiplication. Any
future cross-eye aggregate must use its mathematical factor, including checked
`2*N` when a field actually combines both eyes; the current root schema keeps
the two eye aggregates separate and therefore applies `N` to each.

All products and sums use `checked_u64_mul` and `checked_u64_add`; evaluation is
ordered left to right and an operand or result outside `0..U64_MAX` raises
`DiagnosticsCardinalityError(field, operands)` during read-only preflight. JSON
size calculation runs only after these proofs, so its 20-byte integer token is a
derived fact. Frame generation and source-order consolidation repeat checked
addition defensively. Hitting exactly `U64_MAX` is legal; overflow never reaches
Python-bigint serialization or a partially replaced aggregate.

All derived binary64 values use source-order scalar operations. Define
`add64(a,b)=float64(float64(a)+float64(b))`; a scalar sum starts at positive
`float64(0.0)` and applies `add64` once per source-ordered value. Ratios are
`float64(float64(integer_numerator)/float64(integer_denominator))`. Global clamp
fraction first sums integer counts and then divides. Arithmetic mean uses the
source-order scalar binary64 sum divided by binary64 count; an empty allowed
mean is `0.0`. Min/max compare normalized binary64 values in source order.
For the only schema ratios whose integer denominator may be zero, metric clamp
fractions with no valid pixels, the result is exactly `0.0` without performing a
division. State-pixel denominators are `H*W` and must be positive.

Linear p95 is defined without a library reduction. For total sample count `n>0`:

```text
rank   = float64(float64(0.95) * float64(n - 1))
lo     = floor(rank)
hi     = ceil(rank)
alpha  = float64(rank - float64(lo))
result = float64(float64(value_at_rank(lo))
                 + float64(alpha * float64(value_at_rank(hi)
                                             - value_at_rank(lo))))
```

`value_at_rank` walks the ascending histogram counts without expanding them.
For an empty histogram, integer maxima/counts are `0` and binary64 p95/physical
values are `0.0`. Physical-width maximum and p95 are the corresponding
binary64 lane measures divided by `float64(16.0)`. Vectorized, tree-reordered,
parallel, extended-precision, or fused reductions are forbidden for persisted
statistics.

Legacy-unavailable summary uses the real frame names/count/shape/modes and
hashes, with `eyes=null` and `metric_projection=null`. Duplicate, missing, or
out-of-order names fail consolidation.

For new metric renders, atomically derive the existing-compatible
`04_left_frames/clamp_summary.json` from `MetricProjectionAggregate`; the
orchestrator/runtime summary reads the same aggregate. Missing or corrupt
derived clamp summary is aggregate damage and is rebuildable without RGB. A
legacy reused Fast metric stage continues validating and trusting its existing
per-frame clamp sidecars and summary. Starting any schema-5 metric rerender
deletes its old `clamp_stats` directory and derived summary before writing, so
old and new completion rules never coexist.

The derived compatibility file retains exactly the current six keys:
`schema_version`, `frame_names`, `clamped_fractions`, `affected_frame_count`,
`mean_clamped_fraction`, and `max_clamped_fraction`. Their values are copied or
derived from the corresponding aggregate fields; the diagnostics-only summed
valid/clamped counts are not added to this legacy shape.

Writer threads never append JSONL or update root aggregates. Stereo invalidation
deletes `04_stereo_diagnostics`, both diagnostic aggregate files, and new-render
derived clamp summary only after preflight authorizes replacement. During
`building` or `complete`, intermediates-disabled runs create no masks but retain
stats/manifests and aggregates; intermediates-enabled runs retain every manifest
payload. No normal cleanup removes a currently manifest-recorded file. The only
exception is the committed `payload_pruned` transition, after which the old
manifests are no longer part of current state and only the paths in committed
`prune_entries` may be deleted.

## Settings Migration and Override Resolution

Every saved processing schema from 1 through 4 migrates to:

```json
{
  "stereo_render_mode": "fast",
  "occlusion_fill_max_px": 8
}
```

This direct rule matches the current non-incremental migration implementation
and prevents any old job from being reinterpreted through the new Quality
default. Schema 5 requires both fields.

CLI and Web resume must distinguish omission from an explicit override. The
resolver contract is:

```python
def resolve_stereo_render_mode(
    *,
    persisted: object,
    override: object,
    is_resume: bool,
    renderer_device: Literal["cpu", "cuda"],
    quality_default_gates_passed: bool,
) -> Literal["fast", "quality"]:
    ...
```

- new CUDA job, omitted override: `quality` only after the release gates pass,
  otherwise `fast`;
- new CPU job, omitted override: `fast`;
- schema-v1-v4 resume, omitted override: migrated `fast`;
- schema-v5 resume, omitted override: persisted value;
- any resume with explicit override: validated override.

`--stereo-render-mode` and `--occlusion-fill-max-px` therefore use parser
default `None`, like the temporal postprocessor override. The fill-limit
resolver follows the same omission rules with new-job/migration default 8.
Web resume loads the persisted values before rendering controls and submits an
override only when the user changes them.

The implementation keeps every omitted new job on Fast until all Quality
performance, memory, disk, and visual gates pass on the clean CUDA candidate.
The final gate commit changes only the CUDA omitted-job resolver to Quality.
CPU omission stays Fast. Explicit Quality remains available on either device
after correctness gates pass, with the CPU warning required above.

## Resource and Capacity Contract

### GPU

Fast retains:

```text
GPU_TEMP_BUDGET = 256 MiB
FAST_SPLAT_BYTES_PER_PIXEL = 1280
```

Quality uses the same temporary budget independently in Pass A and Pass B:

```text
QUALITY_SPLAT_BYTES_PER_PIXEL = 1536
```

The clean CUDA gate must show peak live allocation plus 25 percent headroom no
greater than 1,536 bytes per source pixel for forced 1080p and 4K bands. No
full-frame device winner, region, repair, or fine-grid buffer is allowed.

### Host Lifecycle

Let `Q=H*W`, let `G_i` be frame `i`'s native primitive pixel count and
`G=max_i(G_i)`, let
`D_hist=272*(W+1)` be two eyes' dense uint64 lane/span histogram accumulators,
and let `J_frame` and `J_root` be the maximum raw frame-transaction and root-file
JSON bounds derived by disk preflight.

#### Streaming Control Plane

The file pipeline must not materialize `frame_files`, `depth_files`,
`list[_FileWorkItem]`, `P_set`, `R_set`, a metadata `frame_names` list, or any
other per-frame Python object graph. Define:

```text
STEREO_CONTROL_FRAME_CAP       = 4_194_304
STEREO_CONTROL_ACTION_BYTES    = 1 MiB
STEREO_CONTROL_OVERHEAD        = 4 MiB
```

Read-only audit builds one packed two-bit action vector indexed by source
position: `00=reuse`, `01=P render`, `10=R migrate`, and `11=invalid`. Its exact
allocation is `ceil(N/4)` bytes and `N` must be positive and no greater than
`STEREO_CONTROL_FRAME_CAP`, proving the one-MiB action bound. `P` and `R` are
checked scalar counts. Audit never emits `11`; encountering it on any replay is
`StereoControlStateError` before mutation. Names, source/native paths, and
destination paths are provided by a replayable source-order iterator over one
validated canonical upstream sequence manifest; reopening the provider
reproduces the same names and identity without retaining them. A schema-5
provider has no directory-enumeration identity fallback.

One read-only audit freezes a `StereoProviderGeneration` with exactly:

```text
audit_generation_id:         string  # 128-bit OS-CSPRNG, 32 lowercase hex
input_manifest_fingerprint:  string  # 64 lowercase hex
frame_count:                 integer
```

Every yielded item carries those first two values plus its zero-based
`source_index`. Every consumer initializes `expected_source_index=0`, requires
the exact generation and index before using any path or bytes, and increments it
with checked arithmetic; exhaustion must occur exactly at `frame_count`. The
input fingerprint is exactly the SHA-256 of the complete canonical raw bytes of
that validated upstream manifest. If it has an embedded self-fingerprint, that
field is independently validated first but is not substituted for the raw-byte
hash. Missing, noncanonical, stale, corrupt, or unsupported upstream evidence
makes the upstream stage nonreusable; it must be rebuilt and durably publish a
current canonical manifest before a stereo provider can be constructed. The
provider must not derive a replacement fingerprint from directory paths,
entries, stems, sizes, headers, stats, or payload samples.

The sole manifest-less exception is the already-defined saved-schema-1-through-
4 intact Fast-v3 compatibility decision. That read-only validator may admit the
whole completed Fast RGB stage as `legacy_fast_unavailable`; it does not create
a `StereoProviderGeneration`, cannot render or migrate an individual frame, and
cannot authorize Quality, masks, `P>0`, or `R>0`. Any such requested work first
rebuilds the current upstream manifest and then uses the ordinary schema-5
provider. The random audit ID prevents an iterator from an older otherwise-
equal transaction from being admitted into this one.

Before the first durable stage mutation, any generation, fingerprint, source-
index, name, count, path, header, or content mismatch between two prospective-
input replays of that generation discards the action vector,
host/disk preflight, and staged identity bytes. If workers have already parked,
the coordinator closes their start/work gate in cancel mode and joins every
started thread. Only then may the current call restart a completely read-only
audit with a fresh generation.
Damage or staleness found only in retained output/diagnostics is instead the
ordinary input to the action vector and does not cause this restart loop.

Set `mutation_started=true` immediately before the first downstream reset,
unlink, or atomic stage replacement. From that point, the same mismatch raises
`StereoStageInputChangedError`: close every queue and gate in cancel mode,
notify and join all parked or active workers, perform no same-call audit restart,
render, R migration, content revalidation, aggregate consolidation, or
`complete` write, and return the fatal transaction error. A committed
diagnostics `building` state is preserved for a later independent resume; if
failure preceded that commit, orphan Quality identity artifacts still claim no
stage. `P=0` follows the same fatal boundary without creating threads.

The remaining three MiB covers fixed streaming JSON/directory buffers,
progress/audit scalars, replay-iterator/container control, and
attested Python allocator residency. Frame manifests, root metadata,
`frame_names`, JSONL, and compatibility summaries are parsed and written with
streaming state machines; no general JSON decoder may build their
variable-cardinality arrays. The implementation enforces and Task 0 measures
the complete persistent control-plane RSS against the four-MiB limit, including
interpreter/container overhead, rather than treating the constants as payload
sizes only. Exceeding the frame or resident-byte cap raises
`StereoControlBudgetError` during read-only preflight.

For `P`, the feeder acquires a lifecycle permit before constructing a work item,
creates paths from the current streamed stem, and transfers that same object
through decoder, renderer, and writer; it is never copied into a retained list.
Each handoff rechecks the item's generation and `source_index` before touching
payload or stage state.
Thus at most `capacity` work items exist. Fast charges each item's paths/strings
to `FAST_SLOT_OVERHEAD`; Quality has capacity one and charges its single item to
`QUALITY_RUNTIME_OVERHEAD`. For `R`, the coordinator replays the provider
after all pipeline permits and threads are gone and consults the same action
vector. Consolidation replays it again. The mathematical names `P_set` and
`R_set` remain useful in formulas, but they denote action-vector positions, not
resident sets.

Fast keeps its RGB bytes but replaces every scalar bytes-per-output-pixel slot
estimate. Before mutation, read and validate every source PNG and native
canonical/metric header. For frame `i`, let `G_i` be its native primitive pixel
count, `F_i` the sum of the exact compressed source-image and native-geometry
payload byte counts which a bounded decoder may retain, and `J_frame_i` its raw
transaction JSON bound. `G_i > Q` is legal for every backend and is charged by
the formulas; no benchmark-only `G<=Q` assumption exists.

Decoder workers hold a lifecycle permit, load the source plus native primitive
exactly once, and never construct full-resolution geometry. The serial render
thread uses explicit owned outputs and fixed reuse:

- relative decode owns `3*Q + 2*G_i`; construction may own source 3Q, encoded
  uint16 `2*G_i`, float32 native canonical `4*G_i`, and resized near-score 4Q together,
  giving `7*Q + 6*G_i`. It then releases native storage and constructs final
  13Q geometry without `StereoGeometryFrame.__post_init__` copies;
- metric decode owns `3*Q + 5*G_i`. Because invalid native inverse is canonical
  positive zero, the builder resizes it directly as weighted inverse, uses one
  `4*G_i` validity-weight plane, and retains at most two 4Q resize outputs. Its peak is
  `11*Q + 9*G_i`. It derives resized validity/inverse in place, releases native
  state, computes projection statistics before in-place clamp, and uses one 8Q
  float64 fraction output; final geometry is again 13Q with no constructor copy;
- first-eye render/second-eye offset construction remains at most 25Q and second
  render at most 22Q under the one-eye-map schedule;
- writer ownership is at most `23*Q + J_frame_i`: two RGB/compact eyes plus
  bounded encoded RGB/mask payloads and canonical transaction bytes.

Both builders must match the current Fast scalar/bilinear oracles byte for byte.
They construct internal owned `StereoGeometryFrame` values through a validating
no-copy factory; the public constructor keeps its copy contract. A decoder uses
a bounded streaming reader which retains no more than `F_i` compressed bytes;
any backend with additional full-array temporaries violates this contract.

Define exact per-frame lifecycle bounds:

```text
STEREO_HOST_BUDGET        = 512 MiB
FAST_SLOT_OVERHEAD        = 1 MiB
FAST_JSON_STREAM_BYTES    = 1 MiB
STEREO_IO_THREAD_BYTES    = 2 MiB

fast_relative_item[i] = max(
    3*Q + 2*G_i + F_i,       # decoder
    7*Q + 6*G_i,             # serial geometry construction
    25*Q,                    # render
    23*Q + J_frame_i,        # writer
)
fast_metric_item[i] = max(
    3*Q + 5*G_i + F_i,
    11*Q + 9*G_i,
    25*Q,
    23*Q + J_frame_i,
)
fast_slot_bytes = max(item[i] for pending i in the selected geometry mode)
                + FAST_SLOT_OVERHEAD
fast_active_scratch = D_hist + FAST_JSON_STREAM_BYTES
                    + STEREO_CONTROL_OVERHEAD
candidate_capacity_max = min(2*configured_io_workers, P)
effective_workers(c) = min(configured_io_workers, c)
io_thread_bytes(c) = (1 + 2*effective_workers(c))*STEREO_IO_THREAD_BYTES
eligible = {c in 1..candidate_capacity_max:
            fast_active_scratch + io_thread_bytes(c)
            + c*fast_slot_bytes <= 512 MiB}
fast_capacity = max(eligible)
effective_io_workers = effective_workers(fast_capacity)
fast_io_thread_bytes = io_thread_bytes(fast_capacity)
require fast_active_scratch + fast_io_thread_bytes
        + fast_capacity*fast_slot_bytes <= 512 MiB
fast_consolidation_peak = D_hist + J_frame + J_root + 1 MiB + 16 MiB
                        + STEREO_CONTROL_OVERHEAD
require fast_consolidation_peak <= 512 MiB
```

Here `P` is the pending render-item count from the lifecycle matrix and
`configured_io_workers` is its validated 1..16 setting. When `P=0`, capacity,
effective workers, and thread bytes are all zero and no pipeline thread is
created; the pending-item maximum is not evaluated. When `P>0`, an empty
`eligible` set raises the typed host-budget error
before mutation. Selecting the largest eligible integer makes the apparent
worker/capacity fixed point unique.

The capacity charge deliberately repeats the single active construction peak in
every slot rather than relying on a smaller empirical overlap estimate. Before
enqueue, the main thread serializes dense histogram results into that slot's
`J_frame_i` reservation and releases the dense arrays; writer items never retain
both forms. Capacity below one raises the typed host-budget error before any
stage mutation.

The file pipeline creates exactly one feeder, `effective_io_workers` decoders,
and the same number of writers. All four queues have `maxsize=fast_capacity`;
queue nodes belonging to a lifecycle item are charged to that slot and fixed
queue control is charged to the thread envelope.

Python's `threading.stack_size()` setting is process-global for subsequently
started threads. All project-owned thread creation therefore uses one module-
global `_STEREO_THREAD_CREATION_LOCK`. For a nonzero stereo thread count, the
coordinator completes read-only/disk preflight, acquires that lock, saves the
previous stack setting, sets exactly 512 KiB, constructs and starts the feeder,
decoder, and writer threads in their fixed order, and waits for every thread to
report its platform stack facts while parked on an unreleased start gate. A
`finally` block restores the exact previous setting before releasing the global
lock. Concurrent jobs serialize only this creation/attestation interval;
existing threads are unchanged and any other project-owned thread creator waits
for the same lock.

If setting, construction, start, attestation, or restoration fails, close the
start gate in cancel mode, join every thread which did start, restore the prior
setting if restoration has not already succeeded, and raise
`StereoThreadBudgetError`. No parked worker has received a work item or touched
stage state, so this failure occurs before the first mutation. Only after every
thread passes and the process-global setting is restored may the coordinator
perform the first stage mutation and release the work gate. `P=0` neither calls
`threading.stack_size` nor creates a thread.

After successful parking but before the work gate is released, any failure in
downstream invalidation, incompatible payload/clamp deletion, Quality content or
RGB-metadata publication, or diagnostics `building` publication uses the same
cancel/close/join teardown. No parked worker may remain live or observe partial
initialization. If such a failure occurs after `mutation_started`, it is never
converted into a current-call read-only restart.

Each parked thread obtains its actual reserved stack bounds through a supported
platform adapter: POSIX uses `pthread_getattr_np` plus
`pthread_attr_getstack` (or the documented equivalent on that supported OS),
and Windows uses `GetCurrentThreadStackLimits` plus `VirtualQuery`. The accepted
reserve is at most 512 KiB rounded up once to the platform page/allocation
granularity; an unavailable API, an unbounded/default reserve, or a larger
result is fatal. Task 0 also drives the production maximum call depth and
measures committed stack/TLS/queue/Python state. The complete attested envelope
remains 2 MiB per thread: 512 KiB requested native stack and at most 1.5 MiB for
committed stack growth, Python thread state, bounded queue control, and
OpenCV/Torch TLS. Frame arrays, codec buffers, and decoder workspaces remain
charged to the lifecycle slot.

Decode and write tasks may use existing library calls but may not create nested
worker pools. Torch/OpenCV global pools are initialized and measured before the
stage as process-baseline runtime state and cannot lazily create a thread while
the temporary global stack setting is active. The OS thread-count delta while
the pipeline is live must be exactly `1 + 2*effective_io_workers`. Any lazy
nested or uncoordinated project thread creation fails the supported-runtime
gate; truly external in-process code which ignores the project lock is outside
the supported runtime contract, not silently assumed safe.

`StereoPipelineStats` records configured and effective workers, actual feeder,
decoder, and writer thread counts, each queue capacity, calculated thread bytes,
the requested/previous/restored stack settings, platform granularity, every
reported reserve, and the existing permit/timing counters. Pipeline teardown joins every thread
before aggregate consolidation, so thread bytes are absent from the separate
consolidation bound.

Manifest-only migration has its own common host phase. With the schema-derived
raw bounds defined below, require:

```text
R_MIGRATION_STREAM_BYTES = 1 MiB
r_migration_peak = 0 when R=0 else
    STEREO_CONTROL_OVERHEAD
    + max(stats_raw[i] + 2*manifest_raw[i]
          + R_MIGRATION_STREAM_BYTES for i in R_set)
require r_migration_peak <= STEREO_HOST_BUDGET
```

The two manifest terms are the validated old canonical bytes and the complete
new canonical bytes. The one-MiB term contains the strict stats parser, hash,
atomic-writer buffers, and the one current migration item's stem/path objects.
The sibling atomic temporary's allocated file extent
is separately charged by `manifest_atomic[i]` in the disk contract. Migration
retains no decoded image array; payload validation hashes RGB/masks through the
same bounded stream. This phase begins only after pipeline teardown, so neither
thread bytes nor any render lifecycle slot overlaps it.

Fast builds only one eye's int32 offset at a time with one float64 row scratch.
Relative Fast follows its existing near-score/settings operation order; metric
Fast follows its existing total-fraction order. Retain the fraction plane until
the second eye map is complete, release it before second-eye rendering, and never
overlap two eye maps.

Quality v1 deliberately forces `quality_capacity=1` and
`quality_effective_io_workers=1` when `P>0`; both are zero when `P=0`. Its one
feeder, one decoder, and one writer therefore add exactly 6 MiB under the same
thread contract. The lifecycle permit is held from decode through transaction
write, so no second decoded/writing frame overlaps the active renderer. This is
an offline-quality choice, not an estimate derived from the Fast selection
formula. Parallel Quality frame lifecycle slots require a later measured design
revision.

Quality fixed allocations are:

```text
QUALITY_RECORD_ARENA_BYTES   = 64 MiB
QUALITY_GRAPH_ARENA_BYTES    = 64 MiB
QUALITY_REPAIR_ARENA_BYTES   = 64 MiB  # fallback phase, then exemplar phase
QUALITY_CONTENT_STREAM_BYTES = 1 MiB
QUALITY_JSON_STREAM_BYTES    = 1 MiB
QUALITY_RUNTIME_OVERHEAD     = 16 MiB
```

The persistent render coefficient is exactly 20 bytes per output pixel: source
BGR 3, `near_score` 4, `total_disparity_fraction` 8, `source_valid` 1, and final
source-region ID 4. The indexed region solver adds uint64 distance 8, int32 heap
position 4, worst-case uint32 heap pixel 4, and four one-byte band/morphology
work masks per output pixel, plus one uint32 rank per possible native region.
Solver scratch is released before Pass A.

The 13 geometry bytes hold the exact Fast baseline while the region solver runs
and the final one-sided geometry afterward; both versions are never live
simultaneously.

Every Quality-owned host phase must satisfy its corresponding byte bound:

```text
quality_relative_content_audit_peak = QUALITY_CONTENT_STREAM_BYTES
                                      + QUALITY_RUNTIME_OVERHEAD
                                      + STEREO_CONTROL_OVERHEAD
quality_metric_content_audit_peak   = max_i(5*G_i)
                                      + QUALITY_CONTENT_STREAM_BYTES
                                      + QUALITY_RUNTIME_OVERHEAD
                                      + STEREO_CONTROL_OVERHEAD
quality_input_content_audit_peak =
    applicable relative or metric content-audit peak
quality_input_content_revalidation_peak =
    quality_input_content_audit_peak when Quality consolidation is planned
    else 0

quality_relative_decode_peak = max_i(3*Q + 2*G_i + F_i) + 16 MiB
quality_metric_decode_peak   = max_i(3*Q + 5*G_i + F_i) + 16 MiB
quality_lowres_region_peak  = 3*Q + 22*G + 16 MiB
quality_geometry_build_with_nearest_index_peak = 32*Q + 25*G + 16 MiB
quality_region_peak         = 40*Q + 13*G + 16 MiB
quality_planning_peak       = 37*Q + D_hist + 64 MiB + 64 MiB + 64 MiB + 16 MiB
quality_pass_b_peak         = 42*Q + D_hist + 64 MiB + 16 MiB
quality_none_visibility_peak = 34*Q + D_hist + 16 MiB
quality_writer_peak         = 23*Q + D_hist + J_frame + 16 MiB
quality_consolidation_peak  = D_hist + J_frame + J_root + 1 MiB + 16 MiB
quality_public_peak         = quality_pass_b_peak + 4*Q
quality_none_public_peak    = quality_none_visibility_peak + 4*Q

quality_io_thread_bytes = 6 MiB when P>0 else 0
quality_lifecycle_peak = max(all applicable decode-through-writer peaks)
quality_file_pipeline_peak = 0 when P=0 else
                             quality_lifecycle_peak
                             + quality_io_thread_bytes
                             + STEREO_CONTROL_OVERHEAD
quality_consolidation_stage_peak = 0 when no Quality consolidation is planned
                                   else quality_consolidation_peak
                                        + STEREO_CONTROL_OVERHEAD
quality_stage_peak = max(quality_input_content_audit_peak,
                         quality_input_content_revalidation_peak,
                         quality_file_pipeline_peak,
                         quality_consolidation_stage_peak,
                         r_migration_peak)
quality_public_call_peak = max(
    applicable quality_lowres_region_peak,
    quality_geometry_build_with_nearest_index_peak,
    quality_region_peak,
    quality_planning_peak when background,
    quality_pass_b_peak when background,
    quality_none_visibility_peak when none,
    quality_public_peak when background,
    quality_none_public_peak when none,
)
require quality_stage_peak <= 512 MiB
require quality_public_call_peak <= 512 MiB when public Quality is invoked
```

`quality_region_peak` covers steps 3 through 5 of the unique construction
order and contains no retained-zero member/offset storage.
`quality_geometry_build_with_nearest_index_peak` covers steps 6 through 9 and
contains no Dijkstra distance, heap, heap-position, band, or morphology scratch.
Allocation tracing must prove the end of the first lifetime precedes the first
byte of the second. The immutable final region map is the only full-resolution
region allocation carried across that boundary.

Only the applicable relative or metric decode row participates. The native
primitive load follows the same one-allocation/`F_i` decoder contract as Fast;
Quality capacity one prevents overlap with another lifecycle item.

The initial and revalidation content-audit phases use the exact validator and
one-frame ownership contract above. Their `5*G_i` is one final inverse plus one
final validity array, not a public constructor input plus defensive copies. The
one-MiB term includes decompression and value-validation scratch; no compressed
member, boolean selection, or second decoded array may coexist. Header-only
planning, `P=0,R=0` reuse audit, malformed/decompression paths, and an attempted
allocation are all measured against these rows. Initial audit, revalidation,
R migration, file pipeline, and consolidation are mutually exclusive host
phases; the metric audit frame is released before the next file or phase.

The low-resolution region phase's `22*G` covers owned primitives, float64
displacement, uint32 union parent/canonical-key storage, uint8 rank, and final
region IDs with the documented in-place reuse. Union work is released before
the full-resolution solver.

The geometry-build-with-nearest-index bound's `25*G` includes the previous
`17*G` of owned native
inverse/valid, float32 valid-weight and weighted-inverse, and low-resolution
region IDs, plus the exact per-region nearest index's worst-case `8*G+4` bytes;
the four-byte constant is in runtime overhead. Its
`32*Q` output
slots are exactly source BGR 3, region map 4, three float32 primitive/resample
planes 12, one float64 fraction plane 8, final float32 near-score 4, and final
bool validity 1. Buffers are reused in that sequence; a fourth primitive plane
or second fraction plane is forbidden. Relative geometry is bounded by the same
formula. Planning's
`37*Q` is persistent 20, one int32 eye-offset map 4, current dense analysis 8,
and the already completed eye RGB/compact diagnostics 5. Pass B adds the current
eye RGB/diagnostics 5, releases the 64 MiB graph and 64 MiB repair arenas, and
retains only the 64 MiB record arena. Writer's `23*Q` covers both
eye RGB/compact diagnostics plus the worst 1.25-times encoded image payloads;
`J_frame` covers variable JSON buffers. Quality none's `34*Q` is persistent 20,
one eye-offset map 4, and both completed eye RGB/compact results 10; it has no
dense repair analysis or record arena.

The planning coefficient does not omit a dense repair map: none exists while
donor planning reads the 8-byte analysis set. After planning, the B plane's
existing `Q`-byte allocation becomes `repair_bits` while the G/R and region
allocations are released, exactly as specified above. The one-row float64 offset
scratch is within `QUALITY_RUNTIME_OVERHEAD`; no second eye map or full-frame
float64 offset temporary is legal.

`D_hist` is exact: per eye, uint64 lane bins need `16*(W+1)*8` bytes and span
bins need `(W+1)*8` bytes. Serialization streams nonzero bins in ascending order
instead of constructing unbounded Python dictionaries.

Quality none allocates none of the three fixed repair-plan arenas, builds no
plan/analysis table, and uses only its one-pass visibility/public formulas.
Background Quality uses planning/Pass-B/public formulas as applicable.

Metric native inverse/valid plus low-resolution region IDs (`9*G`) remain live
through region solving and are released immediately after one-sided geometry
construction; relative input is smaller and uses the same conservative region
formula.

Canonical JSONL and aggregate generation reads one strict frame transaction at
a time and streams JSONL directly to its temporary and SHA-256. Histogram pairs
are emitted from the dense arrays without constructing Python pair dictionaries;
frame-name and metric-fraction lists are streamed in source order. No full JSONL
or variable-cardinality Python object tree may be materialized. At most one
`J_frame` input, one `J_root` canonical output, and the fixed 1 MiB serializer
buffer coexist during consolidation; that phase is included above.

At 4K with `G<=Q`, the file pipeline peaks at about 507.7 MiB in planning after
its three thread envelopes under the old array-only accounting. The persistent
control plane raises that proven bound to about 511.7 MiB, leaving only about
0.3 MiB and making the Task 0 allocator/RSS attestation a hard gate. Its indexed
geometry-build phase is about 476.9 MiB including control.
The threadless public call peaks at about 501.7 MiB in planning; its final
Pass-B-plus-four-masks phase is about 444.9 MiB and its indexed geometry phase is
about 466.9 MiB. Variable `J_frame`/`J_root` phases are evaluated separately and
can reject a pathologically large job rather than invalidate this statement. The
renderer constructs only one eye-offset map,
reuses all three fixed arenas between eyes, and releases left-eye planning state
before right-eye Pass A while retaining the completed left result.

Before allocating or mutating stage state, calculate every applicable phase and
raise `QualityHostBudgetError(phase, required_bytes, 512 MiB)` if any bound is
too large. Reject `16*Q > UINT32_MAX` through the same preflight before
evaluating record or heap indexes, and reject any checked `G_i+1 > UINT32_MAX`
before a per-region offset array can be requested. The clean memory gate measures all
stereo-owned NumPy, Python, OpenCV, and allocator-resident memory and must remain
within both the formula
and 512 MiB. Arena or later allocation failure is fatal, never an alternate
output path.

### Disk and I/O

Preflight estimates the complete pending stage-04 transaction, not diagnostics
alone. Reuse the metric-stage allocation-unit query with its 64 KiB filesystem
unit fallback; 64 KiB is not a JSON-size assumption. Define
`A=max(4096, queried_or_fallback_unit)` and
`alloc(n)=ceil(n/A)*A`.
The fallback is legal only for a non-reservation stage estimate. Any bootstrap,
standalone settings, or final-encoding transaction must obtain the exact queried
unit for every charged target volume or fail the read-only capability gate before
publishing its control artifact.

JSON bounds are schema-derived. `max_json_bytes(schema, cardinalities,
known_strings)` walks the exact sorted-key canonical schema and counts ASCII
punctuation plus token maxima: each integer is at most 20 bytes, each binary64
token at most 32 bytes, each hash string is exactly 66 bytes including quotes,
and every known string/path uses its actual canonical escaped byte length.
Nullable fields use the larger of their numeric/object token and `null`. The
checked uint64 cardinality contract runs first; a 20-byte integer is therefore a
proved schema maximum rather than an assumption.

A sparse histogram pair needs at most 44 bytes including its following comma:
two unsigned 20-byte integers plus brackets/comma. For one frame, the maximum
number of distinct positive lane lengths is:

```text
K_lane_frame = min(16*W, floor((isqrt(1 + 128*H*W) - 1) / 2))
```

because `K` distinct lengths require at least `K*(K+1)/2` invalid lanes and the
frame has at most `16*H*W`. Per-frame touched-span cardinality is at most `W`.
Merged root cardinalities are at most `16*W` and `W`. Metadata/root frame-name
lists use exact cardinality `N`; metric fraction lists use `N`. These bounds,
not observed data, are inputs to `max_json_bytes`:

```text
stats_raw[i]    = max_json_bytes(FrameStats, K_lane_frame, W, frame_name[i])
manifest_raw[i] = max_json_bytes(FrameManifest, known payload paths for i)
stage_metadata_raw = max_json_bytes(
    Metadata,
    every building/complete/legacy-fast-unavailable state,
    N,
    all frame names,
)
summary_raw     = max_json_bytes(RootSummary, 16*W, W, N, all frame names)
clamp_raw       = max_json_bytes(ClampSummary, N, all frame names)  # metric only
quality_input_manifest_raw = max_json_bytes(
    QualityInputContentManifest,
    N,
    every guide/native-geometry relative path and parsed header,
)  # Quality only; zero for Fast
quality_rgb_metadata_raw = max_json_bytes(
    StereoRgbMetadataV2,
    N,
    every frame name and concrete projection/repair-policy string,
)  # Quality only; zero for Fast
jsonl_raw       = sum(stats_raw[i] + 1 for i in source order)
J_frame         = max({0}, {stats_raw[i] + manifest_raw[i] for all i})
J_root          = max(stage_metadata_raw, summary_raw,
                      clamp_raw when metric else 0)
encoding_input_manifest_raw = max_json_bytes(
    EncodingInputSequenceManifest,
    N,
    every resolved image relative path and PNG header,
)
final_video_manifest_raw = max_json_bytes(
    FinalVideoManifest,
    len(resolved_encoding_arguments),
    every normalized argument and final relative path,
)
settings_artifact_raw_max = max_json_bytes(
    ProcessingSettingsArtifactV5,
    every retained metadata/video/output/settings value,
    the maximum remaining producer/status/cleanup/timestamp/runtime values,
)
require settings_artifact_raw_max <= SETTINGS_ARTIFACT_MAX_RAW_BYTES
prune_marker_raw[j] = max_json_bytes(
    PruneMarkerPayload,
    prune_entry[j].relative_path,
    prune_entry[j].marker token,
) + 1  # mandatory LF
```

`ProcessingSettingsArtifactV5` means the complete persisted settings JSON shape,
including all preexisting schema-5 values and the additional metadata/output,
producer, runtime, status, and cleanup fields in the Public Settings Contract;
it is not a marker-only projection. This formula uses the schema-5 canonical
settings serializer, not the legacy pretty serializer. Its maximum is shared by
every remaining settings transition, so later timestamps or terminal status
cannot exceed the preflight bound.

The preflight helper constructs and tests a maximum-token schema skeleton for
every formula; an encoded object exceeding its bound is an internal error before
replacement. Let mathematical `P_set` and `R_set` be the disjoint source
positions encoded by the control action vector, `P` and `R` their checked scalar
counts, and `M` be 1 only when target manifests retain masks. No implementation
set/list is implied. Then:

```text
rgb_bound  = alloc(ceil(1.25 * H * W * 3))
mask_bound = alloc(ceil(1.25 * H * W))

pending_rgb   = P * 2 * rgb_bound
pending_masks = P * M * 4 * mask_bound
pending_json  = sum(alloc(stats_raw[i]) for i in P_set)
              + sum(alloc(manifest_raw[i]) for i in (P_set union R_set))

root_final = alloc(stage_metadata_raw) + alloc(jsonl_raw) + alloc(summary_raw)
           + (alloc(clamp_raw) when metric else 0)
           + (alloc(quality_input_manifest_raw) when Quality else 0)
           + (alloc(quality_rgb_metadata_raw) when Quality else 0)
root_atomic_reserve = 2 * root_final

frame_atomic[i] = 2*rgb_bound + M*4*mask_bound
                + alloc(stats_raw[i]) + alloc(manifest_raw[i])  # i in P_set
manifest_atomic[i] = alloc(manifest_raw[i])                     # i in R_set
one_transaction_overlap = max({0}, all frame_atomic, all manifest_atomic)

pending_file_count = P*(4 + 4*M) + R
root_file_count = 3 + (1 when metric else 0) + (2 when Quality else 0)
atomic_file_count = max((4 + 4*M) if P>0 else 0, 1 if R>0 else 0)
filesystem_slack = A * (pending_file_count + root_file_count
                        + atomic_file_count)

required_bytes = pending_rgb + pending_masks + pending_json
               + root_atomic_reserve + one_transaction_overlap
               + filesystem_slack
```

The stage-04 calculation neither derives nor charges `payload_pruned` metadata,
maximum-width marker tokens, or prune identities. It owns no marker or
`payload_pruned` replacement space and retains no ambient-free-space promise
across later FFmpeg growth. Marker tokens and concrete directory identities are
captured by final-encoding preflight, whose authenticated extents below are the
sole physical owner. No prune-marker reserve/count variable or finalization-only
metadata maximum exists in the stage-04 byte, file, or `J_root` formula.

The later final-encoding preflight is a separate transaction and deliberately
makes no claim that free space can hold the complete CRF-compressed video. There
is no deterministic "existing video reserve". Its read-only calculation admits
manifest, settings-transaction, descriptor/index, and directory-entry
materialization; only successful indexed materialization guarantees those
objects before FFmpeg.
It first reopen-validates the already allocated `job_terminal_settings` extent;
that lifecycle reserve is indexed into this invocation and is not allocated or
counted again. Before allocation derive only the additional nonterminal settings
transactions introduced by finalization:

```text
additional_settings_transaction_count =
      (1 when the producer pair is null else 0)     # producer marker
    + (1 when keep_intermediates is false else 0)   # cleanup pending
diagnostics_payload_pruned_transaction_count =
      (1 when keep_intermediates is false else 0)
payload_pruned_metadata_raw = (
    max_json_bytes(
        Metadata,
        status="payload_pruned",
        every prior-state hash,
        all final manifest hashes,
        every concrete persisted prune path, DirectoryIdentity, marker name/hash,
    )
    when keep_intermediates is false else 0
)
additional_settings_transaction_reserve =
    additional_settings_transaction_count
    * settings_transaction_extent("replace_existing")
```

The addition, allocation rounding, and multiplication above use checked uint64
arithmetic; the additional count is exactly `0..2`. The validated persistent
terminal descriptor contributes one index entry but zero newly required bytes.
The base three entries are the two manifests plus that terminal descriptor;
pruning adds the separate diagnostics entry and its marker entries. The singleton
final-video target descriptor is a required top-level index field/charge, not an
extent entry and not part of this count.

Apply the sole physical charge helpers from `Reservation Extent V1`; no final-
encoding term defines a private `+A` variant:

```text
final_video_publication_method = (
    "replace_placeholder" when final target is absent
    else "replace_existing"
)
final_video_target_descriptor_raw = max_json_bytes(
    FinalVideoTargetReservationV1,
    every concrete path/method/generation/index value and the selected
    platform's maximum `ReservationFileIdentityV1` spelling,
)
require final_video_target_descriptor_raw
        <= FINAL_RESERVATION_DESCRIPTOR_RAW_BYTES
final_video_target_reservation_charge =
      descriptor_create_new_charge(final_video_target_descriptor_raw)
    + placeholder_entry_charge(final_video_publication_method)

final_prune_marker_file_count = (number of planned prune entries
                                 when keep_intermediates is false else 0)
final_encoding_entry_count =
    3 + additional_settings_transaction_count
      + diagnostics_payload_pruned_transaction_count
      + final_prune_marker_file_count
require final_encoding_entry_count <= FINAL_RESERVATION_ENTRY_CAP

final_encoding_index_raw = max_json_bytes(
    FinalEncodingReservationV1,
    final_encoding_entry_count,
    every concrete source/temporary/target path, parent identity, role,
    marker binding, old diagnostics hashes, size, generation, descriptor path,
    final/sibling-video paths, sibling parent identity, final-video publication
    method, nullable preflight target identity, and target-descriptor path,
)

final_encoding_new_reserve =
      control_artifact_create_new_charge(final_encoding_index_raw)
    + final_video_target_reservation_charge
    + additional_settings_transaction_reserve
    + reservation_extent_charge(
          encoding_input_manifest_raw,
          descriptor_raw["encoding_input_manifest"],
          indexed_method["encoding_input_manifest"])
    + reservation_extent_charge(
          final_video_manifest_raw,
          descriptor_raw["final_video_manifest"],
          indexed_method["final_video_manifest"])
    + sum(reservation_extent_charge(
              prune_marker_raw[j], descriptor_raw["prune_marker", j],
              "replace_placeholder")
          for every planned prune marker j)
    + (reservation_extent_charge(
           payload_pruned_metadata_raw,
           descriptor_raw["diagnostics_payload_pruned"],
           "replace_existing")
       when keep_intermediates is false else 0)
```

The checked preflight first atomically publishes/fsyncs the fixed write-ahead
index, then creates/adopts, syncs, and identity-binds the exact final-video target
through `FinalVideoTargetReservationV1`, then creates/fsyncs every declared
artifact target placeholder, target-parent-local extent, and central descriptor.
Read-only admission requires at least `final_encoding_new_reserve`; that value is
not a strict directory-metadata upper bound. FFmpeg requires the stronger derived
`ready` state after every listed object has actually been materialized and
validated. A materialization ENOSPC preserves the final index and returns
`ReservationIncompleteError` without launching FFmpeg.
The index is a bounded bootstrap publication rather than a recursively reserved
payload extent. Existing settings, the terminal extent, and any old final video/
manifests already consume blocks and are never counted as free. The final-target
descriptor and a zero-length final-video placeholder when needed are materialized,
not left as a future directory-growth assumption. Every other new
term is materialized as an authenticated non-sparse target-local extent plus
descriptor and any declared placeholder, not left as an unprotected free-space
estimate.

Immediately before each declared settings transition, consume exactly one
indexed `settings_transaction_extent` through
`SettingsArtifactTransactionV1`. A consumed slice is never assumed to be
replenished from the replaced old settings file. Additional unused extents plus
the preexisting terminal extent remain physically reserved across FFmpeg growth,
video publication, both manifest commits, and cleanup until the one terminal
settings transaction is durable. Thus a producer write cannot spend the space
needed for `cleanup_status="pending"`, and neither that pending write nor a large
video can spend the lifecycle terminal extent. The producer slice commits before
launch; pending uses its additional slice; completion/failure consumes only the
original job-terminal extent. If processing fails before pending, the terminal
failure write still has that extent and every unused additional slice is released
only after the terminal transition is reopen-validated.

Before FFmpeg, the final-video target entry/descriptor and each manifest/marker/
diagnostics allocation already exist; the latter use a validated zero-filled
target-local extent plus descriptor and, when needed, final-path placeholder.
The artifact transaction writes
and validates canonical bytes at full reserved length, then truncates and proves
at least `A` released. It atomically replaces the declared placeholder or old
target according to the indexed method and syncs the parent. It never
unlinks an extent and asks the allocator for a new temporary or target-directory
entry. Only durable target reopen validation may
retire the descriptor and advance to the next indexed extent.
Prune markers and then the diagnostics payload-pruned metadata follow the same
one-at-a-time rule after both manifests are durable. A crash before terminal
commit revalidates the remaining entries rather than estimating space again.
Thus these artifacts need no new payload blocks from ambient free space. Actual
normalized FFmpeg arguments, all input paths, and every settings maximum are
resolved before this check.

#### Final-Encoding Index-Last Retirement V1

Retirement begins only after the exact terminal settings transition is durable
and every invocation entry is either committed with its artifact-specific
evidence or authenticated as unused under the terminal failure path. Two files
are never described as removed "together." Under the job writer lock, the only
retirement order is:

1. Reopen and validate the terminal settings bytes, revision equation, attempt,
   job identity, and terminal success/failure evidence.
2. Reconcile the exact sibling-video temporary: validate its parent/type/link
   shape if present, unlink only that indexed child, sync its parent, and require
   it absent. Apply the create-new control-artifact table to the final index's
   exact publication temporary and require that temporary absent as well.
3. Reconcile every unused entry in index order and then the final target. For an
   unused artifact entry, validate descriptor/index identity, unlink its source
   extent first, unlink its `replace_placeholder` target second when present,
   then sync that target parent; `replace_existing` is never unlinked. An
   untouched `replace_placeholder` final target may then be unlinked only after
   exact zero-length identity validation, followed by its target-parent sync. A
   published target is retained according to its manifest/failure evidence.
4. In index order, for every entry now proved consumed or safely unused, require
   its source extent absent, unlink its exact safe descriptor temporary first and
   its final `ReservationExtentV1` descriptor second if present, and do not
   create replacement work.
5. Unlink the final-target descriptor's exact safe temporary first and its final
   descriptor second if present. The final index still owns both names throughout
   this step.
6. Sync `.depth-surge-reservations-v1`, then reopen and prove every entry
   descriptor/temporary and the final-target descriptor/temporary absent.
7. Unlink the final `final-encoding-reservation-v1.json`. It is the last
   transaction authority to disappear.
8. Sync `.depth-surge-reservations-v1` and reopen the exact index path to prove
   it absent before ordinary settings/stage/final-media audit may begin.

Recovery classifies the physical suffix of that sequence before applying the
ordinary missing-descriptor construction rule:

| Final index | Durable terminal evidence | Entry/target cleanup | Final-target descriptor pair | Action |
|---|---|---|---|---|
| present | absent | any legal prefix | any indexed legal state | continue ordinary final-index recovery; descriptor absence is not retirement |
| present | present | incomplete | any indexed legal state | resume steps 2 through 6; retain the index |
| present | present | complete | valid final descriptor and/or its exact safe temporary present | resume steps 5 through 8 |
| present | present | complete | final and temporary both absent | `retirement-in-progress`; execute steps 7 and 8 without recreating the descriptor |
| absent | any | any | final or temporary present | unindexed reservation artifact; `ReservationConflictError`, preserve it |

Here `complete` means the sibling is absent, every extent entry is committed/
retired or authenticated-unused/removed, the final target is in its unique
success or failure state, and every required target-parent sync is durable. It
is proved from the index plus terminal/artifact evidence, never from an in-memory
retirement Boolean. Consequently a crash after descriptor unlink/sync but before
index unlink converges by the fourth row, while an index-first orphan can never
be produced.
Within a terminal-durable `incomplete` row, each exact absence produced by the
ordered unlinks in steps 2 through 5 is a monotonic cleanup prefix and is never
recreated. The same absence before terminal evidence is durable remains subject
to ordinary construction/consumption recovery and cannot be inferred as cleanup.

The prelaunch provider-mismatch discard is not terminal retirement. While the
final index is active and the video-publication mutation boundary has not been
crossed, a freshly revalidated mismatch may clean only that exact invocation.
Because it persists no discard Boolean, every prefix must remain ordinary-
construction recoverable: after sibling cleanup, validate each unconsumed
descriptor/source pair, unlink and sync its descriptor pair **before** unlinking
its source/placeholder and syncing the target parent; do the same for the final-
target descriptor before its untouched placeholder. The producer settings entry
may already be consumed by the monotonic v4 marker and is validated as such.
Finally sync the reservation directory, unlink the final index last, and sync it
again. A crash while the index remains present returns to the first recovery row,
may recreate exact reservation work from an absent descriptor/source/placeholder,
and must revalidate the provider before deciding to discard again. It never uses
descriptor absence alone as abort intent.

Both assembled and direct encoding write only to the final index's exact
generation-derived sibling temporary on the final-output volume. FFmpeg
ENOSPC/Windows error 112, interruption, timeout,
validation failure, or any error before publication removes that temporary and
any unauthenticated artifact temporaries, retains the indexed terminal extent
until the terminal failure transition is durable, then removes only matching
unused reservation entries under the authenticated crash policy. An unconsumed
`replace_placeholder` final target may then be unlinked only when its identity is
still the descriptor's exact zero-length reserved identity; sync the parent and
then enter `Final-Encoding Index-Last Retirement V1`. `replace_existing` is
never removed, and a target already changed by a prior unmanifested publication
is preserved as incomplete publication evidence rather than mistaken for the old
placeholder.
The path leaves the old final video plus both old retained manifests unchanged
whenever an old final existed. Only a successfully
closed and validated temporary may enter the publication sequence. Output-file
size is reported from the failed temporary, but it is never presented as a
preflight guarantee. When job root and output share a volume, manifest extents
are still tested independently from the unknown growing video temporary;
reclaim credit is never used to promise video completion.

The 1.25 factor matches metric geometry. Allocation rounding and explicit
filesystem slack are both intentionally retained. Root reserve is doubled
because existing aggregates and replacement temporaries coexist. Existing valid
files consume disk and are not treated as free. Bytes from an entirely
incompatible stage may count as `reclaimable_bytes` only after a read-only audit
has enumerated exact files authorized for deletion; valid per-frame payloads
being atomically replaced are never counted as reclaimable.

`reclaimable_bytes` means uniquely releasable physical allocation, never the sum
of logical path sizes. Compute a separate ledger for every target volume and
never transfer reclaim credit between volumes. The audit runs under the acquired
job writer lock and uses `lstat` or non-following handle opens:

- on Linux/POSIX, allocation identity is the invocation-local tuple
  `(InvocationMountIdentityV1,stx_dev_major,stx_dev_minor,stx_ino)` and allocated
  bytes are `st_blocks * 512`; it is never serialized or substituted for the
  persisted UUID/file-handle identity;
- on Windows, allocation identity is the exact shared
  `WindowsFileIdentityV1` `(16-hex volume_serial,32-hex file_id)` and bytes are
  the filesystem allocation size from the opened handle, not `st_size`;
- each allocation identity contributes at most once, and only when its reported
  hard-link count equals the number of directory entries with that identity in
  the authorized deletion set after a complete job-tree audit;
- a link count larger than the authorized count, an identity reachable from a
  retained path, an external/unknown link, or failure to obtain stable identity,
  link-count, or allocation-size data contributes zero;
- symlinks and Windows reparse points are never followed. Only their own link
  object allocation may count, under the same reliable-identity rule;
- filesystems with clone/reflink, deduplication, compression, or other shared
  extents contribute zero unless the platform exposes a reliable unique-release
  allocation value for that identity.

Immediately before the first stage mutation, re-open and revalidate every
credited identity, allocation size, link count, and authorized path. Any change
invalidates the entire audit and reruns preflight; it never merely subtracts a
delta. Directories themselves receive zero reclaim credit. These rules make the
existing crop-stage `os.link` optimization conservative rather than double
counting its stage paths.

Define the no-reclaim first-mutation requirement:

```text
first_mutation_reserve = alloc(stage_metadata_raw) + A
                       + (alloc(quality_input_manifest_raw) + A
                          when Quality manifest publication is planned else 0)
                       + (alloc(quality_rgb_metadata_raw) + A
                          when Quality RGB metadata publication is planned else 0)
```

Preflight requires both `current_free >= first_mutation_reserve` for the Quality
content-manifest, RGB-metadata, and building-metadata transactions (or just the
latter for Fast) and
`current_free + reclaimable_bytes >= required_bytes` for the complete plan.
Both inequalities are evaluated independently on every volume containing a
planned temporary or final payload.

The mutation order is mandatory:

```text
read-only audit existing state and prospective provider scalar identity
read every native geometry/image header and compressed payload byte count
checked-u64 diagnostics cardinality preflight
compute the applicable Quality content-audit host bound
perform the full initial Quality content audit when applicable
freeze one provider generation for the selected Fast or Quality inputs
derive lifecycle transition, packed action vector, P, R, and exact reclaimable paths
compute JSON/image bounds and every remaining Fast/Quality host phase bound
disk preflight using current_free + reclaimable_bytes
revalidate every credited physical allocation; restart audit on any change
create and attest parked workers when P > 0
replay and validate the complete frozen provider generation one final time
then and only then:
    set mutation_started=true
    invalidate downstream when P > 0
    delete audited incompatible payload/clamp sidecars
    publish Quality input content manifest when planned
    publish Quality StereoRgbMetadataV2 when planned
    write building metadata
    release the work gate and render P items
    teardown and join the pipeline, then synchronously migrate R manifests
    revalidate Quality input content in its isolated host phase
    consolidate aggregates
```

`_prepare_stereo_stage`, downstream reset, clamp deletion, and any payload
replacement must move after preflight. An incompatible old diagnostics root is
removed before the new Quality manifest is published, never after it.
Every line after `mutation_started=true` obeys the fatal post-mutation generation
and worker-teardown rules above. A failure never falls through to a later line.
Insufficient space leaves the old stage untouched and reports required, current
free, and audited reclaimable bytes.
Re-evaluate if the packed action vector or its `P`/`R` counts, target mask
policy, shape, names, geometry mode, or allocation unit changes. A later
ENOSPC/Windows error 112 reports the
persisted estimate, current free bytes, and failing path. With masks disabled,
stats/manifests remain through `complete`; final successful cleanup follows the
`payload_pruned` state instead of leaving dangling manifests.

Benchmarks include PNG encode/write time and actual diagnostic bytes. Fast
renderer-only p95 may regress at most 5 percent. Fast full-pipeline p95 with
intermediates disabled may regress at most 5 percent; with diagnostic masks
enabled it may regress at most 25 percent. Quality is gated below.

## Hash-bound Seven-frame Fixture

The canonical settings-payload hash is
`861ba59c027f57c62b94460b23906d6ddcb7c0fde50c96e002f72ce9544180da`.
The source-stage fingerprint is
`29ccafdddc83b6c408708b29c41e5d2fabc392028d7d4149a11bdb9b9fbc60c3`,
the source-frame fingerprint is
`af573cc60b900ab279466c11e53c9e64daca173b1818e30509d562e8485d78e4`,
and the canonical-stage fingerprint is
`bfc9dca3f7a4ee61816df04e97b1fc56b89d54a536cb5b434862883d0a7ec7fa`.

Every source is uint8 `[1080,1920,3]`; every canonical PNG is uint16
`[608,1080]`.

| Frame | Source SHA-256 | Canonical SHA-256 |
|---:|---|---|
| 89 | `cca5b9dd367ab23d4931c73ec98b1d091c431c69385e5d1077608f4ec3fd060b` | `66d524dd394d82ad9b96c1fda89b725b1b9ff954a25b26fcd0ea0eed8138f77c` |
| 111 | `6e5213f560e651495c8df6f98663b8d01190403ca55fa51d811415ce8131a11c` | `cf63208d7f948274ecdbd14445b19b557efe13cf329bfb35879a01bfc22d5281` |
| 171 | `6a3f46b3d952ab44bb6dcff85364e97b40e320168ed9c52272fd087f32b310cb` | `deb8512a101beee29fe08b53db2ac779ea864b47a1e2d21da355fafd22aa6fff` |
| 176 | `154969bd5ff228c94bfd5f1b0b0f7c66531d55758abcdbd03de53d363192d4bb` | `48e12f34b40345b81dcca7104e216cf5ae8a71614d8faed6a6a1f8061d55c1eb` |
| 231 | `be5f4a523a88386d0a9f781332a02cd6d9b716dc49e0504a052f91949f99a73b` | `1fcf5555f1679e564dd73aaa324ee0c873468e2f0ba6165ccbf5a7dfb3bb8d35` |
| 301 | `0d9182b0aba89afdfafd5adce9a7d2ca0db5742e8bc5b2db04d07f4f1884c932` | `296511484ffddee5b6dd74ad8bc7b51e7da2757a7dc4444ac08fee9ae9f2745b` |
| 401 | `fdf65df83cfe7d647f66b651eeb9cabe0a53c83578a66aad8f7b40662293ba3a` | `4cecaffe00821a84ff81f1422ebc2ef6b50269a0541eb4f23593a1fc80d20347` |

The canonical ordered manifest SHA-256 is
`ee9b8b6126667baa995e6c8055340e8494b38e4d6f9d65c64554d4389127c979`.
The verifier embeds the complete manifest and refuses shape, dtype, hash,
fingerprint, order, or settings mismatch.

Nearest-neighbour geometry is a verifier-only dependency-injected diagnostic.
It is not a setting, saved mode, resume identity, or production branch.

## Verification and Release Gates

### Settings, API, and Resume

1. Schema 1 through 4 migration yields Fast and limit 8; schema 5 requires both
   fields. Invalid modes, booleans, noninteger limits, out-of-range limits, and
   `processing_mode` are rejected.
2. Omitted new-job mode resolves by device and gate state exactly as specified;
   saved settings contain no `auto`. Resume omission preserves the migrated or
   persisted value, while an explicit override wins.
3. A matching migrated legacy Fast v3 stage remains reusable. Mode or
   background-Quality limit changes invalidate stereo and downstream only. A
   limit-only change in Fast or Quality none changes no RGB, diagnostics, or
   downstream identity and performs no rewrite.
4. Public `StereoRenderResult.__dataclass_fields__` remains compatible and its
   four masks have the existing dtype, shape, and semantics. The internal frame
   path proves those arrays are not allocated.
5. A diagnostics-triggered rerender always invalidates tracked downstream
   stages before writing, including when regenerated RGB hashes happen to match.
6. Background-Quality metadata fingerprints configured/scaled limits and both
   policy versions but not frame-dependent limits; each background frame
   manifest records both eyes' recomputed values. Quality none fixes all four
   repair-policy identity fields and every frame/root limit object to null.
7. Quality CPU and CUDA executions produce the same RGB/diagnostics identity and
   reuse each other's valid stage. Fast alone retains its old device-bearing
   metadata as a compatibility field; execution provenance never enters either
   Quality fingerprint.
8. The old zero- through three-positional `StereoRenderSettings` calls remain
   valid; the two new fields reject positional use. Direct `settings=None` is
   Fast on CPU/CUDA, while CLI/Web pass an explicit resolved mode. Relative
   Quality dispatch occurs before `build_relative_geometry`,
   `StereoSplatSettings`/`render_geometry` remain Fast-only, and metric Quality
   is not exposed as a new public primitive API.
9. Python/NumPy booleans and non-`Integral` limits are rejected by both public
   settings and `QualityStereoControls`; accepted NumPy integers normalize to
   Python `int`. The four discriminated plan variants reject every crossed
   primitive/settings/control combination before decoding.
10. Completed-state fixtures migrate, canonically serialize, strict-reopen, and
    bit-compare `processing_time_seconds` for these exact millisecond/binary64-
    hex pairs: `1/3f50624dd2f1a9fc`, `7/3f7cac083126e979`,
    `1001/3ff004189374bc6a`, `1003/3ff00c49ba5e353f`,
    `1007/3ff01cac083126e9`, `123456789/40fe240c9fbe76c9`,
    `18446744073709547520/4350624dd2f1a9fb`,
    `18446744073709549568/4350624dd2f1a9fb`, and
    `18446744073709551615/4350624dd2f1a9fc`. The old multiply/floor check fails
    at 1001/1003/1007 but the authoritative derivation passes; epsilon is never
    accepted. An in-progress crash/offline/resume fixture preserves one
    `attempt_started_at`, advances `terminal_at` across the offline interval, and
    proves both persisted fields include that wall-clock gap. API/UI fixtures
    require the label "attempt elapsed time" and reject "active processing time"
    or equivalent active-compute claims.
11. Raw schema-1-through-4 fixtures with an absent
    `output_info.expected_output_filename` raise
    `LegacyFinalTargetUnknownError`, write no locator/schema 5, perform no final-
    path stat/open/glob, and leave every legacy byte unchanged. A present valid
    component alone may migrate.
12. Portable-name fixtures cover all six superscript device bases. The six true
    dotted-device candidates above fail then hash-fallback; their results pass
    complete validation and Windows create/open/rename/delete. Exact
    `COM³.mp4` and `LPT³.mp4` remain non-fallback controls whose suffixed
    candidates pass the same integration operations.

### Geometry and Numeric Oracles

1. Smooth ramps and every pixel outside an edge band equal the exact current
   Torch bilinear baseline. Strength zero remains byte-identical for both eyes.
2. Vertical, diagonal, T-junction, and geometry-supported one-sample thin-line
   fixtures contain no cross-region intermediate geometry. Texture edges
   outside the band cannot create or move a region.
3. Relative and metric fixtures prove primitive dependency order, validity,
   pinhole equations, clamping, convergence, and independently recounted clamp
   statistics.
   Metric Quality receives native `MetricGeometryFrame` arrays, never calls the
   old bilinear `build_metric_geometry`, and rejects a full-resolution
   `StereoGeometryFrame`; metric Fast moves construction to the serial render
   thread but preserves the old numeric oracle exactly with no defensive-copy
   overlap.
4. An independent scalar binary64 oracle covers boundary clipping, all four
   corner masks, zero retained weight, one-ULP projected-lane changes, and exact
   metric weight reuse. The retained-zero production index matches exhaustive
   `(distance2,iy,ix)` search across region ties, degenerate rows/columns, empty
   query sets, and `G_i/Q` below, equal to, and above one. Its allocation is at
   most `8*G_i+4`, its grouping/order is reproducible, and injected allocation,
   stack, missing-region, and visit-cap failures are fatal with no partial frame.
5. Adversarial union orders produce identical source-region IDs because IDs use
   minimum member index rather than the implementation's union root.
6. The four boundary-movement fixtures and multi-edge fixture match an
   independent scalar integer-geodesic oracle exactly. Production's one global
   indexed heap matches the scalar global queue; no stale duplicate exists,
   equal-cost owner decrease-key is covered, maximum live entries equal the band
   population, and injected capacity corruption fails typed. No independent
   tiled oracle or gate is claimed.
7. The separate relative and geometry one-eye offset builders each match their
   current full-map scalar arithmetic for left/right, int32 limits, signed
   half-lane boundaries, and one-ULP inputs. The exact relative fixture
   `(W,strength,convergence,near_score)=(1920,2.0,0.5,float32(0.9052734375))`
   produces left offset 124; a reassociated total-fraction result of 125 fails.
   Relative displacement, q-jump, gap prediction, and splat fixtures all use
   that same builder order. Peak live state is exactly one int32 map plus one
   float64 row; CUDA OOM retry preserves its allocation and bytes instead of
   rebuilding it.
8. Relative/metric native decode and owned geometry-builder oracles prove every
   array in the `G_i` formulas, no-copy construction, direct invalid-zero metric
   weighting, and release order. Fixtures cover `G_i<Q`, `G_i=Q`, and `G_i>Q`;
   capacity zero fails before mutation rather than clamping depth resolution.
9. Allocation and data-flow tracing proves the ten-step Quality construction
   order. One-sided interpolation reads the post-geodesic map on a fixture where
   that map differs from nearest-label initialization; substituting the initial
   map changes the expected output and fails. The geodesic scratch lifetime ends
   before retained-zero index allocation, while the final map remains immutable
   through both primitive resamples and derived-field construction.

### Visibility, Components, and Repair

1. Pass A records and Pass B invalid masks match by OR and popcount at
   full-frame, one-row, planned, and forced OOM-retry band heights. Record masks
   are nonzero, contiguous, disjoint, and preserve `far_side` under mirrored
   left/right fixtures.
   Equal-depth runs of odd and even length split exactly by per-lane distance,
   with the midpoint assigned left.
2. Pass B RGB and compact diagnostics are byte-identical across band heights,
   CPU/CUDA, and I/O worker counts. Every changed lane was invalid before fill.
3. A vertically full-height component proves cross-band planning without a
   dense fine grid. Horizontal lane-0/lane-15 and vertical disjoint-mask cases
   prove nonadjacent segments do not union; overlapping vertical masks do. A
   last-column/next-row fixture proves horizontal union cannot wrap rows, while
   adjacent same-pixel masks do union.
4. Different union-find strategies produce component IDs ordered by minimum
   canonical fine index. Separated same-pixel masks remain separate, while
   immediately adjacent same-region masks union exactly once.
5. A length-16 run starting at lane 8 reports lane count 16, touched span 2,
   and physical width 1.0; cap classification uses lane count only.
6. Mixed-region pixels are not pure proxies. A same-region pixel with one valid,
   deliberately foreground-contaminated lane may be context but is rejected by
   local, exemplar, and fallback donor selection. Clearance-radius boundaries
   are rejected as donors.
7. Local fills match the independent scalar oracle for both sides, odd/even
   equal-depth splits, target traversal, all five row positions, search limit,
   insufficient context, ineligible slots, exact BGR/first-difference scores,
   integer order, ordinal ties, and evaluation-slot exhaustion. An oriented
   gradient and asymmetric glyph prove donor order preserves direction and does
   not reflect content. Over-cap or failed-local runs reach exemplar or fallback
   instead of black output. A fixture with millions of one-lane runs proves
   whole-run precharge including the conservative full Chebyshev support for
   every donor, no partial candidate reads, exact skipped-run counters, shared
   left-then-right frame caps, and deterministic exemplar/fallback after local
   budget exhaustion. Allocation tracing proves local creates no donor mask,
   integral image, distance map, or other dense helper.
8. Exact pyramid tests cover odd ROI sizes, nonoverlapping half-open core
   interiors, partial last cores, clipped halos, unique ownership, read-only
   later-core targets, target-plus-barrier parents,
   bottom-up working colour, target-only nearest replication for all children,
   unfinished coarse targets, no patch clipping, update visibility, earlier-core
   context, and full-level backend assignment. Donor Sobel fixtures require the
   complete original-safe 9x9 support, reject a nominally legal 7x7 donor with a
   one-pixel adjacent foreground/barrier, and prove no donor patch-edge reflect.
   Target fixtures put barriers and unfinished provisional targets immediately
   outside a legal central 7x7 and prove their adjacent offsets contribute BGR
   but no gradient; processed non-barrier 3x3 support restores the exact gradient
   and expected donor selection.
9. Core, component, and eye budget exhaustion each take the unique specified
   fallback order. Candidate subsampling, 8,192-iteration termination, no-donor
   failure, and segment/scratch/index limits have exact unit expectations.
   The fixed 64 MiB record, 64 MiB graph, and 64 MiB shared repair arenas are
   allocated before mutation; fallback provisional queries precede repair-arena
   reset and exemplar.
   Injected preallocation, index-build, arena-reset, mid-component, and
   post-component `MemoryError` always fail with no committed frame; they never
   change backend selection or retain earlier exemplar results. Planning owns no
   dense repair map; the planar B allocation is demonstrably transferred to
   immutable `repair_bits` without increasing the measured planning peak.
10. A legal donor 129 through 256 1080p-equivalent pixels outside the exemplar
     ROI proves full-frame fallback lookup. A donor one pixel beyond the limit is
     rejected. Brute-force and implicit k-d queries agree across ties, degenerate
     coordinates, regions, and radius boundaries. A degenerate same-coordinate
     index crosses `QUALITY_REPAIR_FALLBACK_VISIT_CAP` on the next node and fails
     the complete frame without returning its current best. Arena overflow and
     index-build failure are fatal. No synthesized value ever becomes a donor.
11. `background + quality` returns zero final unresolved lanes or an explicit
    error. `none` retains Revision 5 black-lane behavior and mask semantics.
12. OOM injection after every Pass A, Pass B, and Quality-none visibility band
    proves exact row-zero rollback. Pass A drops analysis/records/counters;
    Pass B preserves records, offset, and immutable compact maps but drops RGB
    and replay counters; none drops partial RGB/maps/histograms. Completed
    earlier-eye state survives, and an OOM at `h1` reports both heights without a
    third attempt or Quality downgrade.

### Diagnostics and Transactions

1. Quality background plan and Pass B counters independently agree on OR masks,
   popcounts, backend lanes, and final unresolved lanes. Quality none constructs
   no plan and reports the exact no-repair graph/budget availability with null
   component/budget values. Fast fill-helper and pre/post-band counters agree
   without constructing Quality regions or records; every Fast component count
   is null with the exact unavailable reason.
2. Coverage/repair masks, run measures/histograms, ratios, backend
   pixel/component counts,
   geometry-nearest and local/fallback frame/root counters, canonical JSON,
   positive zero, reserved
   bit 7, finite numbers, and hashes all match independent synthetic
   expectations. Fast/legacy nearest fields are null; both Quality fill modes
   report exact integers including the all-zero-query case.
3. Strict-key tests reject every missing, extra, integer-versus-binary64,
   nullable, noncanonical, bad path, settings content/identity fingerprint,
   other self-fingerprint, payload-hash, and
   state-transition mutation in `ProcessingSettingsArtifactV5`, diagnostics
   metadata, Quality RGB metadata,
   Quality-input/frame/encoding-input/final-video manifests, prune
   identities/markers, JSONL, and summary. Scalar aggregation fixtures
   cover source-order last-bit differences, positive zero, empty p95, and
   histogram rank lookup. Every counter-bound multiplication/addition tests
   `U64_MAX-1`, `U64_MAX`, and overflow; consolidation repeats checked addition.
   Naming fixtures accept stems `frame_000089` and `frame_1000000`, reject
   extensions/redundant zeroes, and require the corresponding `.png`/`.npz`
   payload names. The frozen Fast-v3 metadata fixture alone accepts full `.png`
   names and remains byte-identical. Float-bit fixtures require `3f800001` from
   both endian hosts and reject the little-endian byte spelling `0100803f`.
4. Fault injection after final-index publication, final-video target placeholder
   create/file sync/parent sync, target-descriptor temporary/final publication,
   final-video close, file fsync, reserved-target atomic publish,
   encoding-input/publication-manifest fsync/replace, reserved prune-marker
   placeholder/write/truncate/replace/fsync,
   directory-identity capture, prune metadata, and every cleanup step proves the
   exact recovery sequence. Missing/stale input or
   publication evidence always reencodes; two valid manifests after a crash
   permit prune without re-resolving `auto`; no state falsely claims reusable
   payload. Cleanup injection between marker unlink and root removal exercises
   only the identity-matching empty-root `rmdir` exception; a single descendant,
   mount/reparse point, or identity mismatch permits no delete and persists the
   specified incomplete cleanup status without rerendering.
5. Parallel frame completion produces source-ordered, duplicate-free JSONL and
   the same ordered-manifest, JSONL, summary, metadata, and clamp-summary hashes
   as serial completion.
6. Relative stats require null metric projection. New Fast/Quality metric stats
   replace per-frame clamp sidecars, rebuild the exact compatibility summary,
   and preserve legacy Fast validation. Fault injection covers projection stats
   and the derived clamp summary.
7. Disk preflight covers every lifecycle row and exact packed `P`/`R` actions, maximum
   per-frame/root histogram cardinalities, largest canonical uint64 frame stems
   and long contained relative paths, long metric
   fraction lists, masks on/off, metric/relative aggregates, non-reservation
   allocation-unit fallback, reservation capability rejection when the exact
   unit is unavailable, all retained-manifest, prune-marker, and payload-pruned-
   metadata reservations, and
   simultaneous old RGB plus all new temporaries. It explicitly makes no
   video-size promise. Stage-04 goldens contain no prune-marker count,
   `payload_pruned_metadata_raw`, or finalization-only physical charge; final-
   encoding goldens introduce `final_prune_marker_file_count` and
   `payload_pruned_metadata_raw` only when intermediates are removed and
   materialize every marker extent/descriptor/placeholder exactly once.
   Formula goldens independently assert `payload_logical_bytes`,
   `payload_extent_create_charge`, `descriptor_create_new_charge`, both
   `reservation_extent_charge` methods, every `settings_transaction_extent`,
   `control_artifact_create_new_charge`, and
   `final_video_target_reservation_charge`, plus
   `reservation_directory_bootstrap_forecast`. They prove the payload padding,
   source entry, descriptor temporary entry, descriptor final entry, and optional
   target placeholder are five distinct minimum forecast terms and that no
   caller spells a private `+A` substitute or calls the sum a directory-growth
   upper bound.
   Direct and assembled FFmpeg ENOSPC tests leave an old final video and both old manifests
   byte-exact
   while removing sibling/owned reservation temporaries. Reservation fixtures
   cover every exact target-local temporary and central descriptor filename,
   all three final/`.create-new.tmp` control-artifact pairs and their complete/
   partial/both-present matrix, kind/role/path/prune binding, both fixed write-
   ahead index schemas, complete
   bootstrap/standalone target settings objects with frozen timestamps/content/
   raw hashes, job/
   settings/publication-generation binding, logical versus allocated bytes,
   closed `ReservationAllocationEvidenceV1`, exact
   `allocation_evidence_at_readiness`, and non-sparse enforcement.
   Maximum and one-byte-over job-control/settings-index
   fixtures prove the streaming six-MiB phase and raw caps without retaining two
   settings objects. They inject crashes after every control-artifact write
   chunk, fsync, no-replace rename, parent sync, and retirement, plus before/after
   index publication,
   after every target-parent zero-fill write returns but before the pass's single
   file sync, including the final write which first reaches full logical length,
   after that one sync, after every deterministic descriptor-temporary write
   chunk and its rename-no-replace commit, full-length payload
   validation, truncate/released-`A` check, same-parent rename, target reopen,
   and each of the eight index-last retirement steps. Faults specifically occur
   after final-target descriptor unlink but before reservation-directory sync,
   after that sync but before final-index unlink, and after index unlink but
   before its sync; terminal-durable/index-present/descriptor-absent recovery
   removes and syncs the index without recreating the descriptor. Matching
   prelaunch-discard faults instead return to ordinary indexed construction,
   revalidate the provider mismatch, and discard descriptor-first/index-last
   again. Matching partial work is adopted without reallocation. A full-length all-zero pre-
   descriptor extent with short/delayed allocation performs one complete
   rewrite/sync/reopen; ENOSPC or any still-unready evidence remains retryable
   incomplete. Unsupported adapters/flags/filesystems are capability errors and
   preserve an active index, never content conflicts; only a successful full-
   source probe contradicting committed readiness evidence conflicts. Two
   independent fixtures stop (a) after truncate plus file sync and before rename
   and (b) after rename plus parent sync and before descriptor retirement. Both
   recover from descriptor state without invoking the full-length allocation
   adapter: the first validates the exact short source and released `A`; the
   second validates the exact committed target identity/content and then retires
   the descriptor. Every
   payload kind proves all-zero, complete-full, partial, and
   complete-short classification; a 1,200-byte old partial followed by a
   900-byte retry is wholly zeroed before replay and cannot publish the old tail.
   Complete frames with a nonzero tail or illegal typed transition never pass as
   semantic intent. Unindexed, stale, wrong-parent, wrong-mount, or mismatched files
   are preserved as conflicts. Bootstrap fixtures prove the sole order is free-
   space proof, locator, reservation directory, initial target placeholder, both
   settings extents/descriptors, initial settings, then frame work;
   `locator_only` has no other
   interpretation. Admission-boundary fixtures leave one unit below, exactly,
   and one unit above the calculated minimum. One unit below fails before the
   control artifact final is published; exact and above may enter
   materialization but are not asserted sufficient. Separate ext4 htree leaf-
   split/interior-growth and new-reservation-directory fixtures inject ENOSPC
   after the authority is durable and require `ReservationIncompleteError`, the
   active locator/index plus exact owned paths preserved, and zero downstream
   mutation. A control-artifact namespace ENOSPC before its final publication
   leaves only its table-owned temporary and likewise permits no downstream
   mutation. Successful materialization fixtures, rather than forecast
   arithmetic, prove the transition to `ready`. A
   full-disk integration gate removes ambient free space only after readiness and
   proves manifest/settings/marker/diagnostics publication uses the pre-existing
   placeholder/old-target and payload entries; truncate/rename never creates a
   new directory entry. A separate full-disk final-video gate
   creates both the indexed sibling and index-owned final target first, then
   proves same-parent replacement does not extend the directory or return
   ENOSPC/error 112. Final-target fixtures exhaust absent/existing methods, null
   preflight identity, post-index placeholder adoption, target-descriptor
   publication plus raw-cap/one-byte-over rejection, consumed-uncommitted
   reencode, index-last terminal retirement, and wrong identity/type/link/parent
   conflicts. No payload
   extent is accepted in the central reservation directory. Persisted POSIX
   identities require one uniquely device-matching `/dev/disk/by-uuid` entry,
   32 lowercase filesystem-UUID digits, and the bounded opaque handle type/bytes;
   mixed-case UUID spelling normalizes, reboot/remount preserves the result, and
   current `statx` mount IDs are captured only for each invocation. Missing/
   unsafe/cap-exceeded directories, zero/two device matches, non-block storage,
   changed symlink/device/object probes, missing file-handle capability, and any
   `libblkid`/subprocess call fail before mutation. Replacement with a reused
   inode is rejected. Windows fixtures require exactly 16 lowercase hex
   volume digits plus 32 file-ID digits in reservation, prune, and reclaim.
   Allocation-adapter fixtures page Linux FIEMAP through 0, 1, 64, and 65
   mappings; accept only contiguous ext4 coverage with final `LAST`, in either
   the uniform extent form or uniform `MERGED` block-map form; reject mixed
   forms; classify
   `UNKNOWN`/`DELALLOC`/`UNWRITTEN`, gaps, and short `stx_blocks*512` as unready;
   and classify every other known or unknown extent flag and unsupported ioctl as
   capability failure. Windows fixtures require NTFS, exact FileStandardInfo,
   clear forbidden FileAttributeTagInfo bits, block-refcounting false, and one
   exact allocated range. They reject ReFS, a compressed/encrypted/reparse/
   offline parent or file, extra ranges, `ERROR_MORE_DATA`, volume-flag drift,
   and unsupported controls with `ReservationCapabilityError`. Every statically
   knowable preflight capability failure leaves the job-control/index temporary
   and all payload paths absent; a file-specific failure discovered under an
   authoritative index preserves that index and starts no downstream mutation.
   Active-index fixtures prove bootstrap, both fixed indexes, and then
   JobControl-owned extent reconciliation globally precede ordinary settings/
   stage/final-media/legacy audit;
   every provisional placeholder is absent from presence bits and ordinary
   cleanup. Early terminal fixtures crash after full-frame sync, truncate,
   rename, target-directory sync/reopen, and descriptor retirement; exact bytes
   publish once, partial frames reset once, and a committed terminal target frees
   the deterministic descriptor before attempt restart. The final index includes
   the diagnostics payload-pruned extent after
   marker entries, and full-disk recovery consumes it before cleanup-pending
   settings. Hard-link fixtures cover the
   existing no-op crop links, retained links, external-link counts,
   symlinks/reparse points, duplicate identities,
   allocation-size lookup failure, per-volume ledgers, and revalidation races.
   One-byte-below/equal/above tests run before a sentinel old stage and prove no
   mutation on failure. The verifier rejects mutation, wrong order, dtype,
   shape, fingerprint, or hash in any of the seven bound source/canonical
   fixtures before rendering.
8. The lifecycle matrix asserts exact `P`/`R` and invalidation behavior for
   same-policy damage, legacy one-frame damage, legacy metric-sidecar damage,
   mask false-to-true, mask true-to-false, and pruned resume. Final-video or
   either retained-manifest damage cannot reinterpret `payload_pruned` as
   reusable stereo. Cleanup tests change `INTERMEDIATE_DIRS` after prune commit
   and prove resume uses only contained version-2 paths whose directory identity
   and durable marker both match. Replacing a target with a new ordinary
   directory, copying a marker into it, removing/changing the marker, changing a
   file ID, inserting a POSIX mount/bind-mount, or crossing a Windows volume
   preserves the residual and reports the exact path. Unknown cleanup versions
   and platforms without reliable identity delete nothing. `P=0,R=N` and mixed
   `P/R` fixtures prove that migration is coordinator-only after pipeline
   teardown, uses the migration peak, advances in source order, reclassifies
   partial commits after every injected crash, and deletes old masks only after
   all new manifests are durable.
9. A pruned job with authenticated final video but damaged JSONL/summary reports
   `final_video_authenticated=true`, `historical_diagnostics_valid=false`, and
   `stereo_payload_reusable=false`. Inspection neither writes nor renders, the final video remains
   successful, and an explicit diagnostics/mask rebuild uses `P=N` rather than
   aggregate demotion.
10. Encoding-input fixtures prove direct mode authenticates the actually resolved
    06 or 07 left/right bytes and assembled mode the actual 99 sequence. RGB
    byte/order/header/path changes alter its fingerprint; mask/stats changes and
    true-to-false diagnostics migration do not. Direct per-eye header drift,
    non-RGB8 input, and assembled dimension drift fail before FFmpeg; the direct
    two-scale/no-scale branches match their declared header conditions.
    Quality-input fixtures mutate one guide or native geometry byte while
    preserving size and mtime and prove all applicable content/stage hashes
    change while Fast v3 identity does not.
11. Audio fixtures cover `preserve_audio=false`, no `a:0`, zero decoded samples,
    short audio, long audio, malformed audio, and a valid non-first container
    stream. Both encoder paths map exactly `a:0`, emit exactly N video frames,
    omit or produce one AAC stream as specified, never pad short audio, and trim
    long audio at the checked `audio_end_sample`. Runtime argv and normalized
    argv have a positional bijection; every valid image/audio/output token and
    every malformed grammar, path64, index, duplicate, and embedded-path case is
    covered. Probe/decode fixtures assert the exact commands, zero/one/two stream
    shapes, strict sample-rate/channel JSON types, `4*channels` and PCM-counter
    overflow, partial PCM frames, invalid UTF-8/NUL/extra keys, stdout/stderr/
    parser/ring boundaries, 30-second probe timeout, decode stall, cancellation,
    terminate/kill/reap, and no generic helper call. Every audio phase remains
    within its independent peak. Root-wrapper fixtures accept absent, empty, and
    bounded nonempty `programs`/`stream_groups`, reject unknown/duplicate roots
    and every skip-state bound crossing, and prove duplicated wrapper streams do
    not change the authoritative top-level zero/one result. Raw golden outputs
    from FFmpeg 5.x, 6.x, 7.1.5, and the project-resolved Windows distribution
    (currently n8.0.1-52-geef9672b02-20260203) all parse. Output-level
    `-shortest` is absent.
12. ffprobe fixtures reject extra/missing streams, dimensions, pixel format,
    codec, rational rate, frame count, non-1:1 SAR, missing or wrong reduced DAR,
    audio-token mismatch, and any full-decode error; nonempty garbage never
    publishes. Probe fixtures assert the exact `-show_entries` command and reject
    format/tags, invalid UTF-8/JSON, over-depth/overlong tokens, a third stream,
    stdout/stderr one byte over cap, and parser-state overflow with the exact
    typed error. FFmpeg 5.x/6.x/7.1.5 and the same Windows-distribution goldens
    accept absent or empty `programs`/`stream_groups`; a nonempty final wrapper
    and every unknown/duplicate root fail. Probe tests exercise the derived wall
    timeout at its boundary using no-audio, ordinary 30/60-fps audio, long
    low-rational-fps audio, AAC priming/tail, maximum accepted audio packet
    count, one packet over bound, and byte-dominant container fixtures. They
    assert checked `N + audio_probe_units`, byte work units, max selection, and
    timeout cap. Encode-progress fixtures accept LF/CRLF bounded printable-ASCII
    lines for standard and future `[a-z0-9_]{1,32}` keys; `fps`, `bitrate`,
    `speed`, `dup_frames`, `drop_frames`, `stream_0_0_q`, and an unknown legal
    key are discarded without ring growth. Only strict increases of canonical
    `frame`/`out_time_us`/`out_time_ms`/`total_size` reset the 120-second clock;
    repeats, regressions, `N/A`, either legal progress control, ignored keys, and
    a silent child do not. Invalid `progress` values, noncanonical/overflow
    counters, missing terminator, NUL/non-ASCII, bad/overlong keys or lines, and
    any post-`end` byte or cap-plus-one stderr take the parser/budget failure path
    and common terminate/drain/grace/kill/reap sequence. A clean success requires
    exactly one final `progress=end`.
    Full-decode progress fixtures prove monotonic
    `frame`/`out_time_us`/`out_time_ms` resets the stall clock while repeated or
    unrelated lines do not, require `progress=end`, and exercise the 120-second
    timeout.
    Full-decode and encoding diagnostic rings test cap-minus-one, cap, and cap-
    plus-one while proving every child follows the one terminate/grace/kill/reap
    path. The publication path never calls either generic capture helper. Direct
    scale/no-scale and assembled fixtures all contain their exact `setsar=1`
    filter and produce square-pixel output. Internal concurrent mutation is
    blocked by the project writer lock and persistent pre/post changes fail
    validation. Tests and documentation make no claim to detect the explicitly
    unsupported external ABA restoration.
13. Quality content-audit fixtures cover relative and metric initial audit,
    complete `P=0,R=0` reuse, `P>0` pre-consolidation revalidation, and `R>0`
    migration. Metric traces own exactly `5*G_i` plus the fixed stream/runtime
    envelope one frame at a time and never enter `np.load`, the public metric
    constructor, or a second full-array predicate. Golden fixtures cover
    `owned-zip32-v1`, the exact current `_atomic_save_npz()` output, and every
    supported Python/NumPy lock-matrix cell as exactly
    `numpy-force-zip64-local-u32-v1` or
    `numpy-force-zip64-local-sentinel-v1`. Golden CPython 3.10.11/3.11.0 outputs
    require version-20 ordinary local sizes; 3.11.9/3.11.14/3.12.1 outputs
    require version-45 sentinel local sizes, and crossed forms fail. No test or
    parser selects a form from the Python minor version. Member/order/flag/
    method/header disagreement, wrong historical external attributes, descriptor,
    encryption, ordinary/ZIP64 sentinel and EOCD variants, every forbidden comment/extra/
    record, NPY v2/v3, endian/Fortran/structured dtype,
    duplicate/extra NPZ members, CRC/deflate failure, short/overlong output,
    invalid values, and injected allocation failure are typed and never start
    mutation or consolidation as applicable. Both named historical local-ZIP64-
    extra fixtures pass; every other ZIP64 shape fails. Owned-writer fixtures
    assert fixed scratch, incremental CRC/DEFLATE, seek-patched headers, no full
    compressed member, atomic old-file preservation, and every failure boundary.
14. Stereo-provider fixtures change generation token, upstream fingerprint,
    source index, name, count, path, and bytes at every replay boundary. Before
    mutation they cancel/join parked workers, discard actions/preflight, and
    restart from a new read-only generation. After downstream reset, deletion,
    either identity publication, `building`, first/last P item, R migration, and
    content revalidation, they raise `StereoStageInputChangedError`, join every
    worker, retain any committed `building` state, and perform no same-call
    restart, consolidation, or complete write. A missing/noncanonical upstream
    manifest forces upstream rebuild before provider construction and no
    directory-derived identity is called. Only an intact migrated Fast-v3
    whole-stage reuse reaches `legacy_fast_unavailable` without a provider; any
    requested per-frame work exits that exception.
15. `StereoRgbMetadataV2` fixtures assert the exact path, key and nested-union
    shapes, binary64/integer types, canonical bytes without LF, conditional
    repair fields, self-fingerprint, publication order, and one-to-one equality
    with all frame/diagnostics `rgb_stage_fingerprint` values. An orphan object
    proves no completeness. Fast schema 1 remains byte-compatible and cannot be
    parsed or rewritten as schema 2.
16. Synthetic direct and assembled encoding sources at N=1, 10,000, 100,000,
    1,000,000,
    and `ENCODING_CONTROL_FRAME_CAP` prove O(1) provider state, exact streamed
    directory completeness, same-generation prepass/argv/postpass, and at most
    six MiB coordinator residency across the distinct manifest, audio-probe,
    audio-decode, reservation-index, encode, final-probe, final-decode,
    job-control, replayable-settings-intent, producer-settings, pending-settings,
    and completion-settings phase peaks,
    including bounded parent-
    process output and JSON/settings state. No source resolver or assembled
    frame-list interface is called. Pre-launch changes restart only after
    exact indexed temporary cleanup; post-launch changes terminate/reap FFmpeg,
    remove that same sibling, and publish nothing. Clean N=100,000, N=1,000,000,
    and cap reservations also assert exact zero-fill write bytes, one sync per new
    payload extent, no per-chunk sync, and all three wall-time gates. Direct and
    assembled goldens
    persist and execute
    `.depth-surge-final-v4-<complete-generation>.tmp.mp4`, bind its parent
    identity/final target, keep the component at 62 units beside a maximum
    240-unit final name, reject a pre-index collision and wrong path type, and
    never remove a lookalike or any such name without the active index. Crashes
    with unfinished and closed-unpublished media both delete/reencode rather than
    adopting bytes. The supported FFmpeg boundary fixture decodes indexes
    2,147,483,645..2,147,483,646 without error; checked U64 overflow, a two-frame
    range beginning at 2,147,483,646, and any endpoint at 2,147,483,647 fail
    before launch.
17. A pre-contract completed job with only its final video reports
    `(present=true,authenticated=false,
    reason=legacy_no_publication_manifests)`. Inspection is byte-preserving and
    performs no encode/delete/manifest synthesis or prune commit. Explicit
    reencode succeeds only with valid retained inputs; the no-input case remains
    preserved and unauthenticated. Current/incomplete one-manifest transactions
    are never mislabeled legacy. Fixtures exhaust all ten rows of the disposition
    matrix, unsafe path kinds, completed/non-completed status, absent schema-1,
    raw schemas 1..4, new schema 5, migrated schema 5 with preserved source
    schema, missing/corrupt source provenance, and mismatched `project_version`;
    `not_present` is unique to `V/I/P=0/0/0`. A migrated schema-4 legacy video
    starts with the null producer pair; immediately before its first attempted
    v4 launch, the coordinator commits the marker.
    Deleting both manifests afterward yields incomplete evidence, never legacy,
    including after a failed later encode. Target-path fixtures prove new jobs,
    and retained `expected_output_filename` each persist one immutable contained
    component; only new names are required portable. A missing retained field is
    target-unknown and invokes neither new nor historical naming logic. Helper
    changes cannot redirect audit, argv, manifest, or `path64`.
    Naming goldens cover invalid scalar/control input,
    preset/custom resolution grammar, multi-byte and non-BMP stems, both 240-unit
    limits and their adjacent values, exact deterministic hash truncation,
    `CON.foo.mp4`, `con.foo.mp4`, `AUX.bar.mp4`, `NUL..mp4`,
    `COM1.part.mp4`, `LPT9.part.mp4`, all six dotted superscript-device inputs,
    and the two valid unsplit superscript controls. They assert candidate/fallback
    decisions, full-validator results, and rejection before any filesystem create
    when the fixed tail cannot fit. A locatable historical 60-emoji component audits read-only
    despite exceeding 240 UTF-8 bytes; explicit reencode returns
    `LegacyFinalTargetNotPortableError` without touching it. Absolute,
    separator-bearing, dot-segment, NUL, link/reparse, and unlocatable legacy
    targets fail containment.
18. Batch-stitching fixtures create `_stitched_*.mp4` beside missing, legacy,
    current, and pruned job artifacts. The job audit and cleanup result are
    unchanged, neither publication manifest is created or consumed, and only
    the two orchestrated `VideoEncoder` entry points can enter publication v4.
19. Settings fixtures accept bounded legacy pretty JSON only through migration
    and require exact canonical schema-5 bytes thereafter. They inject raw,
    parser, typed-object, serializer, temporary-create/write/fsync, replace,
    directory-sync, and both reopen-validation failures at producer, pending,
    terminal-success, and terminal-failure transitions. Producer failure never
    launches FFmpeg. Exhaustive schema fixtures remove and add every top-level,
    metadata, video, normalized-settings, output, runtime, and metric-summary
    key; cross every integer/binary64/null/timestamp/state condition; and prove
    two independent serializers emit identical canonical bytes and raw maxima.
    Locator fixtures prove the fixed schema-5 path wins, extra legacy files and
    mtime changes are inert, zero legacy candidates are absent, multiple are
    ambiguous, and migration retains one exact history path without replacing
    it. Disk-full fixtures consume the unknown video space immediately after
    initial settings and after every early pipeline stage while proving the
    target-local lifecycle terminal extent can still commit `failed`. With both
    fixed indexes absent, fault injection leaves that extent zero, partial,
    complete-full, complete-short, renamed-before-directory-sync, committed-
    before-descriptor-retirement, and terminal-migration-unused. The global
    reconciler has one expected action for every state, publishes only an exact
    legal `r+1` frame, and clears the old descriptor before attempt restart.
    They also prove the fixed standalone settings index is durable before any rewrite/
    replacement extent, partial index groups recover uniquely, and no random
    nonterminal extent can exist without a persistent owner. Independent
    producer/pending extents remain with intermediates off, the
    same original terminal extent is indexed rather than double-counted, and a
    terminal-to-new-attempt transition preallocates both rewrite/replacement
    extents, captures a fresh `attempt_started_at`, and then preserves it through
    producer/pending rewrites. Cleanup-only resume reuses its one or two persisted
    indexed extents and allocates neither again. Partial success/failure terminal
    frames rederive a new terminal time and produce the exact attempt-start-based
    wall-clock integer duration/binary64 seconds pair, including any persisted
    attempt outage and never a created/updated/process-uptime value. Completed
    legacy fixtures also run every frozen
    millisecond/binary64 pair through migration, canonical serialization, raw-
    maximum construction, and terminal reopen validation. No
    direct pretty-JSON overwrite runs, every phase remains at six MiB or less,
    and no job reports completion before the terminal canonical settings bytes
    are directory-synced and reopen-validated.

### Fast Compatibility

Run the complete Revision 5 independent discrete oracle, procedural fixtures,
CPU/CUDA comparison, banding, OOM retry, zero-strength, benchmark, and public
mask tests. Fast eye arrays and masks must be exactly equal, not merely visually
similar. Its one-eye relative/metric offset builders must equal the current two
full-map helpers before those helpers are retired from production Fast. Frame 89
must reproduce both pinned Fast eye hashes above.

### Real-fixture and Temporal Review

Render frames 89, 111, 171, 176, 231, 301, and 401 in Fast, Quality,
Quality-without-fill, and verifier-only nearest modes. Produce full-frame views,
400-percent crops around hair, feather, gun, and ribbon structures, both compact
diagnostic images, and a JSON report bound to candidate commit and input hashes.

The report must prove zero cross-region intermediate geometry in edge bands,
zero writes to valid lanes, zero donor-region violations, zero final unresolved
lanes, and only expected deterministic downstream transforms. Human review
rejects any named contour with a background stretch, soft halo wider than one
output pixel, nearest-style stair step, missing thin structure, or copied
foreground streak. Whole-frame PSNR/SSIM cannot override crop review.

A license-free sequence with a one-pixel foreground line moving by quarter-pixel
increments must move monotonically, preserve the line whenever geometry has a
region, and match serial/parallel output. Contiguous real-frame windows around
the seven fixtures receive flicker review. Failure returns to a separately
reviewed temporal design and does not authorize hidden optical flow.

### Performance and Capacity

The clean CUDA reference report must record GPU and driver, CPU model, physical
and logical core counts, installed RAM, OS, Python/Torch/CUDA/NumPy/OpenCV
versions, Torch intra/inter-op thread counts, OpenCV thread count, frame-worker
count, and every benchmark command. The current intended host reference is a
13th Gen Intel Core i9-13900K (24 cores/32 logical processors, 63.58 GiB RAM)
with RTX 4090; a changed host requires a newly recorded baseline rather than
being described only as an RTX 4090 run.

With five warmups and 30 measured frames on that clean CUDA environment:

- Quality renderer p95 is at most 5.0 seconds at 1920x1080 and 20.0 seconds at
  3840x2160, and at most 12 times matching Fast p95;
- a 4K synthetic component covering 25 percent of the image finishes within
  30.0 seconds, within budgets or with explicitly counted safe fallback;
- a 4K synthetic frame routing at least 25 percent of pixels to fallback builds
  its at-most-48 MiB index and completes exact queries within 30.0 seconds;
  query p95 visits at most 4,096 nodes and diagnostics report all visits;
- the separate geometry retained-zero index is exercised at `G/Q` 0.5, 1, 2,
  and 4 with 25, 50, and 100 percent queried edge-band pixels. It matches brute
  force, owns no more than `8*G+4` bytes, reports every visit, and either finishes
  within 30.0 seconds at supported full-pipeline shapes or raises the specified
  pre-mutation host/query budget error; an injected cap crossing never returns a
  nearest sample;
- full-pipeline Quality p95 including diagnostics is at most 6.0 seconds at
  1080p and 24.0 seconds at 4K;
- GPU allocation plus 25 percent headroom fits 1,536 bytes per source pixel;
- Quality lifecycle capacity is exactly one; geometry, region solver, planning,
  Pass B, writer, consolidation, and public-wrapper measurements each stay below
  their calculated phase bound and 512 MiB. The indexed heap never exceeds one
  live entry per band pixel; geodesic scratch is absent from
  `quality_geometry_build_with_nearest_index_peak`, its index is absent from
  `quality_region_peak`, and region scratch is absent during repair planning.
  File-pipeline RSS includes the complete four-MiB control plane and stays within
  the approximately 511.7 MiB 4K bound rather than relying on its former margin,
  and configured/effective/actual worker counts are respectively recorded and
  equal to `configured/1/(1 feeder,1 decoder,1 writer)` when `P>0`; `P=0`
  creates none;
- relative and metric Quality content audits, including a complete reused
  `P=0,R=0` stage and a post-render revalidation, stay within their explicit
  phase formulas and 512 MiB. Metric traces show one `5*G_i` owned audit frame,
  one fixed stream buffer, no public-constructor copy, and release before the
  next frame or phase;
- Fast 1080p/4K runs at worker counts 1, 4, and 16, masks disabled/enabled, and
  native geometry ratios `G/Q` below, equal to, and above one. They verify each
  frame's compressed-byte/header-derived slot, calculated maximum capacity, and
  forced decoder plus active render plus writer overlap. Actual Python, NumPy,
  OpenCV, Torch, and allocator-resident memory remains within
  `fast_active_scratch + fast_io_thread_bytes +
  capacity*fast_slot_bytes` and 512 MiB. Reports bind configured/effective/
  actual thread counts, all four queue capacities, requested stack size, and the
  supported-platform 2 MiB per-thread envelope;
- synthetic replayable sources at N=1, 10,000, 1,000,000, and
  `STEREO_CONTROL_FRAME_CAP` prove the packed action vector and all persistent
  Python/control residency stay within `STEREO_CONTROL_OVERHEAD`; N above the
  cap fails before mutation. Heap snapshots and allocation traces prove there
  is no frame/path/work-item list, only capacity-bounded lazy items, while
  metadata and frame-name arrays stream byte-exactly;
- direct and assembled final-encoding providers at the same cardinalities up to
  `ENCODING_CONTROL_FRAME_CAP` remain at or below the independent six-MiB
  calculated peak and eight-MiB cap through directory audit, manifest prepass,
  source-audio probe/decode, reservation descriptor/index streaming, argv
  construction, encode progress, final validation, postpass, and isolated
  producer/pending/terminal settings commits, with no eager path/name collection
  or uncapped settings object;
- on the recorded reference local storage, clean direct and assembled final-
  reservation runs at N=100,000, N=1,000,000, and
  `ENCODING_CONTROL_FRAME_CAP` record filesystem/storage identity plus
  `reservation_write_bytes`, `reservation_fsync_count`, and
  `reservation_wall_time_ms`. For each run, write bytes equal the calculated new
  payload logical lengths, sync count equals the calculated new payload-extent
  count rather than chunk count, and no counter or trace contains a per-64-KiB
  sync. Each of three measured runs finishes within respectively 30, 180, and
  600 seconds; direct and assembled are gated independently. A production call
  has no wall-time fallback or alternate semantics, but a release candidate
  exceeding this gate must optimize the identical sequential protocol rather
  than reduce N, skip zero/allocation validation, or reintroduce per-chunk sync;
- simultaneous stereo jobs repeatedly create and tear down threads while a
  third project thread creator contends for the global creation lock. Every run
  restores the previous process stack setting on success and each injected
  failure, existing threads retain their stacks, platform-reported reserves fit
  the rounded 512 KiB request, and maximum-depth committed deltas fit 2 MiB per
  thread. Unsupported stack introspection fails before mutation;
- total stage-04 disk use and transaction peak do not exceed preflight reserve;
- Fast renderer and no-mask pipeline p95 regress at most 5 percent, and Fast
  pipeline p95 with masks enabled regresses at most 25 percent.

On the same host with renderer device CPU, five warmups and ten measured frames
must give Quality renderer p95 at most 120 seconds at 1080p and 480 seconds at
4K, with the same 512 MiB host limit and byte-identical CUDA output. CPU never
becomes the omitted-job Quality default, even when this reference gate passes.
The same completed Quality cache is reusable across CPU/CUDA because device
provenance is non-semantic.

The CUDA omitted-job default changes to Quality only after every numeric gate
and seven-frame human crop review passes. A failure cannot be hidden by raising
budgets, disabling diagnostics, lowering strength, or weakening RGB/depth
barriers without another reviewed canonical revision.

### Repository Gate

Black, configured flake8 including McCabe complexity 10, mypy, the full unit
suite, and at least 85 percent unit coverage must pass. No full-frame device
fine grid, silent Fast downgrade, unbounded repair allocation, or new neural
dependency is allowed. The candidate is not merged until the crop/report review
receives explicit user approval.

## Implementation Boundary

The implementation plan must begin with Task 0, before production renderer
changes. Prototype the exact indexed geodesic heap, the repair-donor implicit
k-d lookup, and the separate per-region geometry-nearest index at 1080p and 4K.
Use edge-band/query populations 25, 50, and 100 percent, a separate
25-percent-repair-fallback case, and native geometry ratios `G/Q` 0.5, 1, 2,
and 4. Every run compares the applicable independent scalar brute-force oracle
and records output equality, maximum live heap entries, exact index bytes,
visited-node totals/max/p95, cap failures, wall time, and process RSS under the
fixed arenas. Run both CPU and CUDA pipeline contexts; the host algorithms
remain semantically identical.

Task 0 also includes the million-one-lane-run local fixture and degenerate
repair k-d fixture. It verifies exact whole-run local charges/skips, shared
left-then-right frame budgets, fatal fallback-node overflow without a current-
best result, full-neighbourhood sample reservation with no dense local helper,
and algorithm-v8 diagnostics. It separately traces the ten-step
geometry order and proves `quality_region_peak` ends before
`quality_geometry_build_with_nearest_index_peak` begins.

The same Task 0 also prototypes the no-copy relative/metric native decoders and
serial Fast geometry builders at `G/Q` below, equal to, and above one. It records
every live array, compressed input retention, allocator-resident peak, scalar
output equality, and capacity for configured workers 1, 4, and 16. It also
records calculated effective workers, actual thread counts, every queue
capacity, requested stack size, and committed-memory deltas with forced decoder,
renderer, and writer overlap. A hidden constructor copy, uncharged thread/TLS
state, or library workspace outside the fixed formula blocks production work
just as a heap/k-d gate failure does.

Task 0 separately implements the Quality content reader rather than reusing the
current double-copy metric store validator. At `G/Q` below, equal to, and above
one, initial and revalidation passes prove exact raw hashes/value errors, one
`5*G_i` owned metric allocation, the fixed decompression/check buffer, per-frame
release, and both audit-phase RSS formulas. Malformed archives, decompression
failure, and every allocation failure point are injected before any production
renderer work is authorized. It first freezes golden
`MetricNpzPayloadContractV1` fixtures for the owned writer, the current real
`_atomic_save_npz()`, and every supported Python/NumPy lock-matrix cell. The
schema-5 project writer and fixed parser must agree on every ZIP/NPY field; a
runtime form outside the three-name allowlist blocks implementation rather than
falling through to NumPy.
It streams each member through the fixed writer buffers, injects compressor,
seek, patch, fsync, replace, and disk-full failures, and proves no complete
compressed member or second array exists at any point. Both measured additional
RSS and native deflater state must fit the writer formula.

The same gate replaces the current eager `_FileWorkItem`/path lists with the
replayable source and packed action vector before renderer work begins. Synthetic
N up to `STEREO_CONTROL_FRAME_CAP` measures the entire persistent Python/control
graph under four MiB and confirms only capacity-bounded items exist. It runs
`P=0,R=N` and mixed actions through synchronous post-teardown migration and
measures `r_migration_peak`. It also implements the global stack-creation lock,
parked-worker start gate, previous-setting restoration, POSIX/Windows reserve
adapter, concurrent-job contention, and every failure injection described by
the host contract. Any unavailable adapter, restoration race, over-cap control
RSS, or per-thread envelope failure blocks production implementation.

That gate also replaces the current final-encoding resolver return interface and
assembled frame list with `EncodingSequenceProvider`. It measures the complete
coordinator at N up to `ENCODING_CONTROL_FRAME_CAP`, exercises both sides of the
FFmpeg-launch mutation boundary and the signed-int image2 endpoint, and proves
the same generation supplies streamed manifest bytes, scalar argv construction,
and postpass. It separately measures manifest, source-audio probe, clipped PCM
decode, encode, exact final ffprobe, final full-decode, and all three settings-
commit phases with capped stdout/stderr/JSON/settings/ring state and injected
overflow, timeout, cancellation, sync, reopen, and reap paths on Windows and
POSIX. It freezes legal FFprobe wrapper output from FFmpeg 5/6/7 and the resolved
Windows distribution, proves top-level stream authority, and gives full decode
observable monotonic progress, low-rational-fps AAC work-unit deadlines, and
publishing-encoder stall termination. It also proves portable deterministic
target-name generation, containment-only legacy audit, immutable target-path
use, the complete closed settings schema and fixed job-control locator, every
authenticated physical extent/index from initial settings through terminal
status, all three control-artifact temporary state tables, canonical payload
framing, JobControl-owned terminal reconciliation, the bounded Linux UUID
capability adapter, the persisted sibling-video path, crash reuse without double
allocation, and the first-v4-attempt marker
across failed/repeated encodes. Any eager resolver call, generic capture helper,
direct pretty-settings overwrite, mtime-selected settings path, over-cap RSS,
mixed generation, leaked process/thread, unreserved status write, or publication
after mismatch blocks production implementation.

Task 0 first evaluates an implementation using only existing NumPy, Torch, and
OpenCV facilities. If it misses any performance or RSS gate, this specification
authorizes one project-owned C++/Torch host extension for indexed-heap and k-d
inner loops. It must ship as prebuilt artifacts for every supported
OS/Python/architecture combination, use Torch's already present C++/pybind11
toolchain, expose the same fixed dtypes/arenas/tie order/typed failures, and match
the scalar oracle byte for byte. Runtime JIT compilation, a new third-party
runtime dependency, an alternate RGB identity, relaxed exactness, unbounded
allocation, and silent Fast fallback remain forbidden. If neither existing-deps
nor prebuilt-extension path passes, explicit Quality is unavailable with an
actionable error and the implementation plan returns for architecture approval.

Implementation may change settings, CLI, Web controls, resume behavior, stereo
geometry, renderer internals, frame writer, diagnostics, final encoding, tests,
benchmark and verifier scripts, and stereo documentation. It may change
`processing/frames/metric_geometry.py` payload storage only as required to
implement `MetricNpzPayloadContractV1` and its bounded owned writer. Metric array
semantics, convergence, projection, stage equations, and non-payload metadata
remain unchanged. Prefer focused Quality geometry,
coverage, and reconstruction modules over growing `stereo_renderer.py` into a
second monolith. Update at least `docs/ARCHITECTURE.md`, `docs/PARAMETERS.md`,
`docs/TROUBLESHOOTING.md`, and affected resume/performance documentation.

Do not modify depth inference, canonicalization, scene analysis, temporal
postprocessing, crop, distortion, upscaling, or VR layout. Encoded media
semantics change only where this revision explicitly corrects the contradictory
duration contract: the N image frames are authoritative, output-level
`-shortest` is removed, and selected `a:0` audio follows the fixed no-padding and
sample-trim policy. Every path also normalizes `setsar=1` and validates the
resulting SAR/DAR. Content-only encoding-input identity, normalized executed FFmpeg
arguments, sibling-temporary encoding for both paths, exact container validation,
durable final-video publication, and identity-plus-marker prune entries are also
authorized transaction changes for the two orchestrated job encoder entry
points only. Schema-5 settings creation/migration/status writes may replace the
pretty serializer only with `SettingsArtifactTransactionV1`; raw schema-1-
through-4 files remain byte-untouched until an authorized migration. Output-name
generation may change only to the frozen portable v1 contract above.
`/stitch_video` remains unchanged and outside job publication. No
compressed-video size preflight guarantee is authorized or claimed.
NumPy, Torch, and OpenCV are already production dependencies; no new external
runtime dependency is authorized. The Linux identity adapter uses only standard-
library/native OS calls plus the fixed system-managed `/dev/disk/by-uuid`
capability; `libblkid`, the `blkid` executable, a Python binding, and bundled UUID
filesystem-superblock parsers are explicitly outside implementation authority.

## Rejected Alternatives

- **Nearest geometry:** confirms the cause but replaces halos with steps and
  increases true holes.
- **Full-image guided/bilateral filtering:** can transfer texture or line art
  into geometry where no source boundary exists.
- **Fill-cap-only changes:** did not affect 8/10-pixel experiments and exposed
  black cracks at 6 pixels.
- **Current boundary copy plus masks:** improves observability but repeats a
  potentially contaminated boundary colour.
- **Generic OpenCV inpaint:** cannot enforce source-region donor barriers.
- **Neural repair or matting:** adds downloads, nondeterminism, and a much wider
  validation surface before the proven causes are addressed.
- **Optical flow in Quality v1:** serializes frame work and can warp outlines;
  temporal state requires a separate reviewed design.

## Approval Criteria

Approval of this canonical specification accepts:

1. Exactly two modes: byte-compatible Fast and offline-first Quality.
2. Quality geometry modifies primitive interpolation only inside a
   geometry-supported band and preserves the exact Fast baseline elsewhere.
   Its final RGB-geodesic map, never the nearest-label initialization, selects
   one-sided regions in the fixed nonoverlapping ten-step lifetime order.
3. Quality repair uses banded analysis, a capped global segment plan, and banded
   replay rather than a dense full-frame fine grid.
4. Segment records preserve far-side intent and exact fine-lane connectivity;
   every fill writes only replay-confirmed invalid lanes.
5. Only fully covered, barrier-cleared same-region pixels may be copied; bounded
   local, exemplar, and full-frame fallback follow the unique deterministic
   contracts and shared frame work caps. Local donor checks reserve the complete
   neighbourhood-sample upper bound without a dense helper; fallback query
   overflow is fatal.
6. Compact production diagnostics coexist with an unchanged allocating public
   mask API and producer-attested lane statistics.
7. Schema 1 through 4 jobs migrate to Fast; schema 5 preserves resolved intent.
   Their raw source schema records ancestry, while a separate monotonic first-
   v4-attempt marker determines whether final media can still be legacy.
   Neither depends on package version. A persisted attempt start, not creation,
   last-update, or process uptime, is the sole terminal-duration origin; its
   result is attempt wall-clock elapsed time, including pause and downtime, and
   is never labeled active processing time. Gated
   CUDA jobs may default Quality, while CPU jobs default Fast.
8. No neural dependency, generic inpainting, or temporal state enters v1.
9. Quality v1 serializes one decoded/rendered/written frame lifecycle. Fast uses
   geometry-aware relative/metric slot bounds over `Q`, every `G_i`, compressed
   input bytes, JSON, queue overlap, and only capacity-derived I/O threads under
   their attested envelope. A four-MiB streaming control plane replaces every
   `O(N)` Python work/path/name list, manifest-only `R` runs synchronously after
   pipeline teardown, and thread creation uses the restored global stack-lock
   contract. Quality initial/revalidation content audits are separate bounded
   phases and metric validation owns exactly one `5*G_i` frame. Either mode
   rejects a phase over 512 MiB rather than assuming `G<=Q` or silently reducing
   quality. One frozen provider generation may restart only before mutation;
   after mutation a change is fatal and every worker is joined.
10. Retained content-only manifests authenticate pre/post-validated Quality
    guide/native-geometry inputs and the resolved 06/07 or 99 image inputs,
    independent of stat-only provenance, diagnostics, and mask policy. A
    durably fsynced manifest of the actually executed FFmpeg arguments binds that
    fingerprint after final-video publication and before `payload_pruned`;
    missing evidence forces reencode, never inference. The project writer lock
    excludes internal mutation; external ABA mutation is explicitly outside the
    transaction trust model. Frame identities use only canonical minimal-padded
    stems and numeric-endian-independent float32 bit strings, except for the
    explicitly frozen filename-bearing Fast RGB schema 1. Metric payloads pass
    the closed three-form `MetricNpzPayloadContractV1`; schema-5 writing no longer
    delegates archive syntax to NumPy and uses only bounded incremental writer
    workspace. Quality uses the one strict immutable
    `StereoRgbMetadataV2` fingerprint throughout diagnostics, with byte/content
    identity rather than an absent upstream-provenance field.
11. With intermediates disabled, valid historical aggregates may remain readable
    after pruning, but deleted stereo payload is never claimed reusable and later
    aggregate damage is reported as irrecoverable without invalidating a valid
    final video. `FinalMediaAuditDispositionV1` exhausts every video/manifest
    presence combination without a redundant validity boolean. A pre-contract
    final video with no publication manifests is preserved as unauthenticated
    legacy media, cannot authorize pruning, and is never upgraded by inferred
    manifests. Any v4 pre-launch attempt marker makes later missing evidence
    incomplete, even when the job originally came from schema 1 through 4.
12. Direct public renderer omission remains Fast on every device; device-aware
    Quality defaulting belongs only to resolved CLI/Web jobs. Metric Quality uses
    explicit `QualityStereoControls` and a discriminated plan, while Quality none
    excludes the unused fill limit from semantic and diagnostics identity.
13. Prune recovery deletes only a committed version-2 relative path whose
    physical directory identity and durable random marker both match, without
    crossing descendant mounts. Strict uint64 cardinality is proved before
    mutation; after marker unlink it may only remove an identity-matching empty
    root, and an identity mismatch preserves successful final output with an
    incomplete cleanup status. Final encoding promises manifest space plus
    sibling-temporary recovery rather than an invented video-size reserve.
    Publication requires all N image frames, explicit square-pixel filters, and
    the exact SAR/DAR-aware ffprobe/full-decode validator under this document's
    authority over the older Direct VR transaction rules. Direct and assembled
    sources use one O(1) replayable encoding provider under an eight-MiB
    coordinator cap which includes bounded parent-process output and parser
    state for source-audio probe/decode, final validation, and every settings
    commit. Legal FFprobe section wrappers never override authoritative top-level
    streams; final probe has a video/audio/byte-work-derived deadline and full
    decode has a semantic-progress stall deadline, while the publishing encoder
    has its own semantic-progress stall deadline. One closed settings schema,
    fixed job-control locator, three deterministic control-artifact temporaries,
    NUL-padded payload frame, and canonical transaction own producer/pending/
    terminal durability. A JobControl reconciler owns the lifecycle terminal
    extent whenever neither fixed index does; exact durable bytes publish and a
    partial frame is wholly zeroed. Generation-bound descriptors/indexes make
    every finalization reserve crash-reusable, persist the one short sibling-
    video path, and materialize an exact final-video target entry plus its post-
    index identity descriptor before launch. One charge-helper family forecasts
    payload padding, every source/descriptor directory name, the reservation
    directory bootstrap, and an optional target placeholder separately without
    claiming a strict namespace-growth bound. Indexed materialization, not the
    read-only forecast, proves readiness; namespace ENOSPC remains incomplete and
    starts no downstream mutation. Payload zero fill uses fixed chunks but only
    one sync per whole pass; a full-length pre-descriptor allocation shortfall
    remains retryable after one synchronized whole rewrite, while an unsupported
    named adapter is a capability failure rather than a content conflict. Linux
    FIEMAP accepts both uniform extent and uniform `MERGED` block-map mappings.
    Readiness evidence is rerun only for a full source; exact short-source and
    committed-target phases use identity/content/release contracts. Final
    retirement removes/syncs all descriptors before the final index. Stage-04
    never pays for future payload-pruned metadata. Linux destructive mutation
    additionally requires the bounded no-dependency UUID/file-handle capability
    adapter and the exact ext4/FIEMAP allocation adapter; Windows reservation
    requires the exact NTFS/allocated-range adapter. A new persisted immutable
    target
    meets both 240-byte and 240-UTF-16-unit limits; a contained locatable legacy
    target remains read-only instead of being retroactively renamed/rejected.
    Checked last-index arithmetic and the fixed signed-int image2 endpoint
    prevent historical reinterpretation. Batch stitching remains outside this
    publication and audit scope.
14. Task 0 must prove the exact geodesic, geometry-nearest, and repair-donor
    heap/k-d performance paths, local/fallback semantic caps, ten-step allocation
    order, no-copy Quality content audit, bounded replayable stereo and encoding
    control planes, mutation-generation boundaries, synchronous migration, and
    the configured/effective/actual/restored-stack thread envelope. Schema-5
    stereo providers require canonical upstream manifests and never synthesize
    a directory fingerprint. It also proves both CPython historical ZIP forms,
    the bounded owned metric writer, audio/final process lifecycles, FFprobe
    version goldens, portable new and contained-legacy immutable final targets,
    the canonical settings transaction/reservation schedule, and the monotonic
    publication marker. A
    project-owned prebuilt extension is permitted only under the fixed oracle and packaging
    constraints; it cannot weaken output or add a silent fallback.
15. All deterministic, resource, transaction, seven-frame visual, temporal, and
    Fast compatibility gates pass before implementation is considered releasable.
