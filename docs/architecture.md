# Architecture

## Product shape

Cudalab keeps Artlab's most important interaction loop while replacing the browser
runtime with native GPU machinery:

| Artlab idea | Cudalab translation |
|---|---|
| Example gallery | Manifest-scanned CUDA package catalog |
| Monaco source pane | Native, dockable CUDA source pane |
| Ctrl+Enter module rebuild | NVRTC compile + atomic module swap |
| Canvas preview | CUDA/OpenGL interoperable texture |
| Errors/log/compile panel | NVRTC diagnostics and timing |
| FPS counter | UI FPS plus CUDA event kernel time |
| `setup/update/teardown` | Stable render ABI now; lifecycle ABI next |
| Browser portability | Native Windows/Linux builds |

## Frame path

```text
CUDA kernel
    │ writes uchar4 (device memory)
    ▼
OpenGL pixel buffer object registered with CUDA
    │ glTexSubImage2D from bound PBO (no host copy)
    ▼
OpenGL texture
    │
    ▼
Dear ImGui preview panel
```

On resize, the runtime recreates the preview texture and PBO. Every frame it maps the
PBO into CUDA, launches `render`, records CUDA events, unmaps, and updates the texture.
The CPU never reads the pixels.

## Live compilation

NVRTC targets the current device's virtual architecture (`compute_XY`) and emits PTX.
The driver JIT loads that PTX and resolves the required `render` symbol. Compilation is
transactional: Cudalab loads a candidate module completely before unloading the current
one, so broken edits keep the last successful visual running.

The current render ABI is intentionally tiny. It is the first rung, not the ceiling.
The next ABI revision adds optional lifecycle entry points and an opaque host service
table for persistent buffers, multiple streams, CUDA graphs, texture/surface objects,
and library handles.

## Capability ladder

Cudalab should make modern features available without making every demo architecture-
specific:

1. **Portable:** grids, shared memory, warp intrinsics, cooperative groups, CUB/Thrust.
2. **Pipeline:** async memory pools, streams/events, CUDA graphs, graph conditionals.
3. **Interop:** OpenGL today; Vulkan external memory/semaphores next; D3D12 on Windows.
4. **Architecture family:** family-specific targets such as `compute_120f` when a demo opts in.
5. **Architecture specific:** tensor memory accelerator, tensor cores, cluster launch,
   distributed shared memory, and other features guarded by a manifest capability.

The manifest will declare requirements; the gallery will explain unsupported pieces
instead of attempting a bad launch.

## Dependency policy

- CUDA Toolkit: required, 13.0 minimum; current toolkit recommended.
- SDL 3: native windows, input, DPI, and OpenGL context.
- Dear ImGui docking branch: studio shell, pinned in CMake.
- CUDA libraries are added per demo family, not linked speculatively.
- Assets in reference demos must be generated, original, or clearly redistributable.
