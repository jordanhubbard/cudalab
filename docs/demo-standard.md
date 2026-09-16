# The Cudalab demo standard

The gallery is not a sample dump. An example belongs only if it passes all four tests.

## 1. Immediate payoff

The first valid frame must communicate the idea. No ten-second warmup, empty default
camera, invisible scale, or required README archaeology. If interaction matters, state
it in `controls` and make pointer motion useful immediately.

## 2. A CUDA lesson worth learning

Each demo names its central technique: memory coalescing, warp cooperation, shared-memory
tiling, persistent kernels, graph replay, tensor cores, library composition, interop,
or another concrete mechanism. Source should be readable enough to modify in the studio.

## 3. Measured behavior

Publish the GPU, resolution, toolkit, architecture target, kernel-time distribution,
memory use, and visual workload. “Fast” is not a measurement. A fallback may reduce
density, never silently change the demonstration's core claim.

## 4. A finished composition

Technical novelty is necessary but not sufficient. Color, camera, pacing, interaction,
and silhouette should look authored. Prefer a single legible idea over every effect at once.

## Planned families

- Ray/path tracing and signed-distance rendering
- Particles, flocking, cloth, fluids, and rigid/deformable simulation
- Fractals, cellular automata, reaction diffusion, and generative systems
- Image/video pipelines with NPP and codec interop
- Audio synthesis, FFT, and spectrogram worlds with cuFFT
- Linear algebra and tensor-core visualizations with cuBLASLt/CUTLASS
- Scientific fields, volumes, marching cubes, and astronomy
- Multi-GPU and topology-aware demonstrations
- CUDA graphs, stream-ordered allocation, cooperative launch, and cluster features
- Neural graphics and differentiable experiments where dependencies remain responsible
