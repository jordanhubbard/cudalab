# CUDA feature atlas: every mechanism becomes a medium

This is not a checklist of SDK calls. It is the production map for turning CUDA's major
capabilities into authored experiences. A row is complete only when the work is visually,
sonically, or interactively compelling *and* its source makes the CUDA idea legible.

| CUDA territory | Artwork | Experience | Status |
|---|---|---|---|
| SIMT and fast math | **Hello, Spectrum** | Touch color and watch a wave field answer | Shipped |
| Adaptive ray marching | **CUDA Cosmos** | Orbit a lensed black hole and turbulent disk | Shipped |
| Distance estimation | **Mandelbulb Cathedral** | Fly around an illuminated fractal monument | Shipped |
| Monte Carlo light transport | **Glass Menagerie** | Inspect glass, pearl, and molten metal | Shipped |
| Persistent simulation | **Reaction Garden** | Seed living chemical botany | Shipped |
| Semi-Lagrangian fields | **Fluid Calligraphy** | Paint with ink, smoke, and gold leaf | Shipped |
| Atomics and persistent particles | **Firefly Constellation** | Bend a quarter-million-agent organism | Shipped |
| Tensor cores / WMMA | **Tensor Tapestry** | Retune a textile woven by matrix products | Shipped |
| Warp collectives | **Warp Loom** | See 32-thread choirs exchange notes | Shipped |
| Shared memory and barriers | **Shared Memory Rose** | Thread blocks assemble stained glass | Shipped |
| CUB block algorithms | **Memory Corruption** | Sort synthetic memory into prismatic instability | Shipped |
| Temporal frame state | **Feedback Cathedral** | Pull recursive video light through space | Shipped |
| GPU audio synthesis | **Spectral Orchard** | Hear and see one CUDA-authored score | Shipped |
| CUDA graphs | **Choreograph** | A seekable, deterministic multi-part GPU performance | Shipped |
| Iterative heightfields, flocking, and buoyancy | **Ocean Procession** | Conduct wind, waves, and a persistent collision-avoiding flotilla | Shipped |
| Deep volumetric integration | **Volumetric Tempest** | Enter a turbulent cathedral wrapped around living lightning | Shipped |
| Streams, events, async allocation | **Confluence** | Independent visual and musical rivers meet without stalling | Next host-services wave |
| `cuda::pipeline`, async copy, TMA | **Memory Weather** | Data arrival itself becomes wind and precipitation | Planned |
| Cooperative groups and clusters | **City of Choirs** | Neighborhoods synchronize locally, then answer across a city | Planned |
| cuFFT | **Fourier Garden** | Live sound grows a navigable spectral ecosystem | Planned |
| NPP and camera/video ingest | **Chromatic Witness** | A live camera image becomes material, delay, and architecture | Planned |
| OptiX | **House of Impossible Light** | A cinematic, editable world of caustics and volumetric geometry | SDK integration required |
| CUTLASS / cuBLASLt / FP8-BF16 | **Low-Precision Dreams** | Precision changes become visible changes in memory and texture | Planned |
| Thrust | **Taxonomy of Dust** | Millions of motes repeatedly sort, partition, and classify themselves | Planned |
| cuSPARSE | **Negative Space** | Sparse matrices become shifting skeletal cities | Planned |
| Dynamic parallelism | **Recursive Bloom** | Forms decide on the GPU when and where to spawn descendants | nvJitLink/device-runtime work |
| Unified and virtual memory | **Archive of Light** | An apparently boundless zoomable image reveals residency and migration | Planned |
| Multi-GPU, peer access, NCCL | **Two Suns** | Multiple GPUs exchange and merge independently evolved worlds | Requires 2+ GPUs |
| Profiling and occupancy | **Pulse Room** | Register pressure, occupancy, and bandwidth become a live spatial score | Planned |
| Vulkan / D3D12 interop | **Borderless Material** | CUDA output crosses graphics APIs without a host copy | Planned |

## Curation rule

No piece is accepted because a primitive is fast. It must use that primitive to create
an effect that would be qualitatively poorer without it. Performance instrumentation is
part of the explanation, never the subject of the artwork.

Some rows require hardware or separately licensed SDKs not present on the development
machine. Cudalab will report those requirements explicitly and keep the rest of the
gallery functional; it will not ship a counterfeit fallback under the same technical claim.
