# CUDA editor and formatting

Cudalab's source pane is a native CUDA C++ editor built on the maintained
[ImGuiColorTextEdit](https://github.com/pthom/ImGuiColorTextEdit) component. CMake
fetches a pinned revision and links it into the application; there is no browser,
language server, or separate editor process in the live-authoring path.

## Editing loop

The editor owns the current UTF-8 source document. Selecting a gallery entry loads
its manifest's `.cu` file, resets editor history and diagnostics, and asks NVRTC to
compile it for the installed GPU. The normal loop is:

1. Edit CUDA C++ in the **Kernel** pane.
2. Press **Ctrl+Enter** or **F5** to compile and atomically install the candidate
   module.
3. Read compiler diagnostics inline and in the **Output** pane.
4. Press **Ctrl+S** to save, or **Ctrl+Alt+F** to save and format.

A failed compile never unloads the last good CUDA module, so the preview and GPU
audio continue while the source is repaired.

## CUDA-aware highlighting

The base C++ tokenizer is extended with CUDA language elements rather than treating
a `.cu` file as generic C++. Highlighted vocabulary includes:

- execution-space and launch qualifiers such as `__global__`, `__device__`,
  `__shared__`, and `__launch_bounds__`;
- built-in coordinates and warp operations such as `threadIdx`, `blockIdx`,
  `__shfl_sync`, and `__ballot_sync`;
- CUDA vector, stream, event, texture, surface, half, and bfloat types;
- Cudalab's staged ABI macros: `CUDALAB_RESET`, `CUDALAB_SIMULATE`,
  `CUDALAB_RENDER`, `CUDALAB_COMPOSITE`, and `CUDALAB_AUDIO`;
- library namespaces used by the gallery, including CUB, Thrust, cooperative
  groups, and WMMA.

The pane also provides line numbers, automatic indentation, paired delimiters,
matching-bracket feedback, multi-cursor editing, find/replace, and a scrollbar
minimap.

## NVRTC diagnostics

Every compile clears stale markers and parses NVRTC's file and line locations.
Warnings receive amber gutter and line markers; errors receive red markers. The
full diagnostic is available as a hover tooltip and remains in the Output pane.
On failure, the editor moves the primary cursor to the first error and centers it
in view.

This presentation is deliberately downstream of NVRTC. It does not attempt to
duplicate CUDA parsing or invent diagnostics that differ from the compiler used to
build the running module.

## Formatting

Repository style lives in [`.clang-format`](../.clang-format). The Format command:

1. writes the current editor document to its package entry file;
2. launches `clang-format -i --style=file` without passing source through a shell;
3. reloads the formatted file into the editor; and
4. clears markers whose line positions are no longer valid.

The launcher uses `_spawnlp` on Windows and `posix_spawnp` on Linux. Consequently,
`clang-format` must be present on `PATH` when the command is used. Editing,
compilation, and saving continue to work when it is absent, and a failed formatter
launch is reported in the Output pane.

The checked-in CUDA sources are formatted with the same file:

```bash
clang-format -i $(rg --files examples include -g '*.cu' -g '*.cuh')
```

After a formatting or editor change, use the full native verification path:

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
./build/dev/cudalab --smoke-test
```

The smoke test compiles and renders every gallery entry, which catches formatting
changes that remain valid host C++ but accidentally alter NVRTC source behavior.
