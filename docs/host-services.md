# Host Services v2

Host Services v2 gives a live-compiled piece durable GPU resources, explicit scheduling,
reusable CUDA graphs, and a seekable authored timeline without hiding CUDA behind a scene
framework. The original staged lifecycle remains intact, and all existing examples compile
against the extended macros without source changes.

## Manifest contract

A package can opt into graph replay and timeline controls and declare up to eight resources:

```json
{
  "work_items": 4096,
  "graph": true,
  "timeline_seconds": 32.0,
  "bpm": 112.0,
  "resources": [
    {"name": "score", "kind": "buffer", "bytes": 65536},
    {"name": "memory-a", "kind": "surface2d", "width": 1024, "height": 1024}
  ]
}
```

Resource order is the stable device slot order; names make the manifest and host telemetry
legible. Buffers use the stream-ordered allocator. A `surface2d` allocation supplies a CUDA
array plus paired texture and surface objects, with normalized linear texture sampling and
wrap addressing.

## Device contract

Every staged macro receives two additional parameters:

```cpp
const CudalabResource* resources, int resource_count
```

The table resides on the GPU. Buffer slots can be accessed safely with:

```cpp
float4* score = cudalab_buffer<float4>(resources, resource_count, 0);
```

Surface slots expose `texture`, `surface`, `width`, and `height`. Kernels remain responsible
for bounds, element formats, synchronization, and avoiding read/write hazards. Resource
names are deliberately not copied into device memory; kernels use manifest order while the
studio uses names for authors and diagnostics.

All resources are cleared when a package is loaded, reconfigured, looped, or explicitly
sought. Recompiling source preserves allocations but requests the package's `reset` stage,
keeping iteration quick and predictable.

## Scheduler

The runtime owns three non-blocking streams:

```text
simulation stream: reset → simulate → record simulation-done
                                      │
render stream:          map PBO → wait event → graph/render → unmap PBO
                                      │
audio stream:                       wait event → synthesize → host queue
```

The simulation event is the explicit state-visibility boundary. Rendering and audio never
observe partially updated named buffers. Audio has its own device buffer and transfer path,
so its work no longer occupies the render stream.

## Graph replay

When `graph` is true, the runtime lazily creates a CUDA driver graph after the OpenGL pixel
buffer is mapped. The graph contains the `render` node followed by `composite` when present.
It is instantiated once per compiled module. Each frame updates kernel parameters and grid
dimensions before replay, allowing resize, pointer input, time, and the mapped PBO address
to change without rebuilding graph topology.

Pointer input includes normalized position, per-frame `mouse_dx`/`mouse_dy`, and
`mouse_down`. Gesture-driven pieces can therefore distinguish hovering, sweeping, and a
pressed brush without private host code.

`params.beaufort` carries the studio's number-key wind setting from 0 (calm) through 9
(strong gale). Wind-aware pieces remain responsible for mapping pointer position to a
direction and for showing that direction in their composition.

Reset and simulation stay outside the graph on the simulation stream. This keeps one-time
reset behavior explicit and makes the cross-stream dependency visible instead of relying on
legacy default-stream ordering.

## Deterministic timeline

`timeline_seconds` adds a play/pause control and seek slider to Preview. Authored timeline
pieces receive `params.time` and a deterministic 60 Hz `params.frame` derived from the
playhead. Seeking or looping clears state and named resources, resets audio sample position,
and runs the normal reset stage before the next frame.

A piece that supports arbitrary seeking should derive its primary state from absolute
`params.time`; temporal surfaces can then accumulate from the reset point. **Choreograph**
uses this model for four eight-second movements and two ping-pong memory surfaces.

## Headless snapshots

The hidden application path can render two seconds of lead-in and export an exact authored
frame without opening the studio:

```bash
./build/dev/cudalab --snapshot choreograph 20 frame.ppm
./build/dev/cudalab --snapshot ocean-procession 7 gale.ppm 0.5 0.5 9
```

The optional final three arguments set normalized pointer X/Y and Beaufort force, making
interaction and weather states reproducible. This is intended for visual regression
fixtures, documentation, and composition review.

For a gallery-wide behavioral audit, run:

```bash
./build/dev/cudalab --interaction-probe captures/probe
```

The probe launches the same hidden SDL/OpenGL/CUDA application used for snapshots, then
renders every package through five smooth 45-frame input intervals: centered, northwest,
northeast with the primary button held, southeast with the button held, and southwest at
calm wind. The intervals also sweep Beaufort force through 4, 2, 6, 9, and 0. Each endpoint
is saved as a PPM, while `report.csv` records the injected state, final GPU time, and mean
normalized RGB change from the preceding endpoint. The images are the authoritative visual
check; the change metric is a useful warning for unexpectedly inert or overreactive pieces.

The regular `--smoke-test` still compiles and renders every gallery entry, including graph,
resource, timeline, and audio paths.
