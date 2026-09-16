# Contributing

Cudalab is intentionally selective. Start with [the demo standard](docs/demo-standard.md),
then keep changes small enough to measure and understand.

## Development loop

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
./build/dev/cudalab
```

Before a pull request: format touched C++/CUDA, build with warnings enabled, run tests,
compile every example through NVRTC, and record performance claims with hardware and
resolution. Do not add binary assets without provenance and license notes.

## Package manifest

```json
{
  "name": "short-directory-name",
  "title": "Human Title",
  "description": "One concrete sentence.",
  "category": "RAY MARCHING",
  "entry": "main.cu",
  "controls": "Move pointer to orbit"
}
```

`name`, `title`, `description`, `category`, and `entry` are required. Keep the initial
frame finished, the kernel ABI exact, and the hot-reload path safe.
