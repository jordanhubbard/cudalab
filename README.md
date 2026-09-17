# Cudalab

A native CUDA creative-coding studio for Linux and Windows. Edit a CUDA C++ kernel,
press **Ctrl+Enter**, and see it running immediately—without moving the rendered frame
through system memory.

Cudalab borrows the essential workflow that makes [Artlab](https://github.com/jordanhubbard/artlab)
fun: a curated gallery, visible source, a live preview, one-keystroke iteration, honest
diagnostics, tiny package manifests, and examples worth opening just to watch. Here the
medium is native CUDA C++ rather than JavaScript.

> **Status:** early, working studio. The live compiler, staged kernel lifecycle,
> persistent device state, GPU audio, CUDA/OpenGL interop, package catalog, editor,
> diagnostics, GPU timing, and first thirteen pieces are present. The roadmap deliberately
> starts narrow and deep rather than shipping fifty mediocre demos.

## The first gallery

| Demo | Technique |
|---|---|
| **Hello, Spectrum** | Minimal one-thread-per-pixel kernel and pointer interaction |
| **CUDA Cosmos** | Curved-ray black-hole renderer, turbulent accretion disk, Doppler color |
| **Mandelbulb Cathedral** | Distance-estimated 3D fractal, orbit traps, shadows, atmosphere |
| **Glass Menagerie** | Stochastic light transport through glass, pearl, and metal |
| **Reaction Garden** | Persistent Gray–Scott chemistry and pointer-seeded growth |
| **Fluid Calligraphy** | Semi-Lagrangian ink, smoke, and gold-leaf advection |
| **Firefly Constellation** | 262,144 persistent agents, warp exchange, and atomic trails |
| **Tensor Tapestry** | Thousands of live WMMA tensor-core matrix products |
| **Warp Loom** | Warp shuffle and ballot operations made into iridescent textile |
| **Shared Memory Rose** | Thread-block collaboration as a stained-glass rose window |
| **Memory Corruption** | CUB block radix sort used as a live image-making operation |
| **Feedback Cathedral** | Persistent recursive framebuffer and temporal image warping |
| **Spectral Orchard** | CUDA-synthesized stereo score and synchronized nocturnal world |

Every example is procedural, self-contained, live-editable, and runs through the same
small staged ABI. A failed compile leaves the last good composition running.

## Quick start

Requirements:

- NVIDIA GPU and current driver
- CUDA Toolkit 13.0 or newer (13.3+ recommended)
- CMake 3.28+, Ninja, a C++20 compiler
- SDL 3 and OpenGL development packages
- `clang-format` for the optional in-studio Format command

Linux:

```bash
git clone https://github.com/jordanhubbard/cudalab.git
cd cudalab
cmake --preset dev
cmake --build --preset dev
./build/dev/cudalab
```

Windows (Developer PowerShell with CUDA and Visual Studio installed):

```powershell
git clone https://github.com/jordanhubbard/cudalab.git
cd cudalab
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH=C:\path\to\SDL3
cmake --build build
.\build\cudalab.exe
```

Dear ImGui and the MIT-licensed ImGuiColorTextEdit component are fetched and pinned by
CMake. SDL remains a system dependency so the app uses each platform's supported
windowing package.

## Authoring model

A package is a directory under `examples/` with a `cudalab.json` manifest and one `.cu`
entry. The entry exports `render` and may add `reset`, `simulate`, `composite`, and `audio`:

```cpp
#include <cudalab.cuh>

CUDALAB_KERNEL {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height) return;
  // Write pixels[y * params.width + x].
}
```

Stateful pieces declare `state_bytes` and `work_items` in their manifest. Cudalab allocates
persistent stream-ordered GPU memory and runs the stages as:

```text
reset (once) → simulate → render → composite → audio
```

The host passes resolution, time, frame delta, normalized pointer position, frame number,
and quality. NVRTC compiles for the **actual installed GPU**. CUDA writes into an OpenGL
pixel buffer registered with CUDA graphics interop; OpenGL then displays it directly.
Optional stereo audio is synthesized on the GPU and queued to SDL's native audio stream.

## Studio controls

- **Ctrl+Enter** or **F5** — compile and run
- **Ctrl+S** — save
- **Ctrl+Alt+F** — format CUDA source with the project style
- **Space** — pause/resume while the editor is not focused
- Move over the preview — update `params.mouse_x/y`

The native editor provides CUDA C++ syntax highlighting, line numbers, bracket matching,
a source minimap, and inline NVRTC error/warning markers. Formatting uses `clang-format`
and the repository's `.clang-format` file.

## Design principles

1. **The demo is the documentation.** Every abstraction earns its place in a readable piece.
2. **Keep the last good frame alive.** Compilation errors belong beside the work, not in a crash dialog.
3. **GPU-resident by default.** Interop, persistent allocations, graphs, streams, and libraries are first-class.
4. **Show the numbers.** Kernel time, frame rate, target architecture, driver/runtime, and workload stay visible.
5. **Curate hard.** A demo must be visually or technically exceptional—and preferably both.
6. **Scale up without hiding CUDA.** Helpers remove ceremony; they do not disguise execution, memory, or synchronization.

See [Architecture](docs/architecture.md), [CUDA editor and formatting](docs/editor.md),
[Demo standard](docs/demo-standard.md), [CUDA feature atlas](docs/feature-atlas.md),
and [Roadmap](docs/roadmap.md).

## Build and test

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

The project targets portable C++20 today. CUDA 13.4 adds an NVCC C++23 mode; the
runtime will adopt it after the supported-toolkit floor and host compiler matrix can do
so without making the studio fragile.

## License

MIT. See [LICENSE](LICENSE).
