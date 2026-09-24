# Isolated FNN research workspaces

Use `tools/research/fnn_workspace.py` to map one explicit local revision and optional patch to a
remote checkout, build directory, and private result root. The helper is pinned to the audited
`fnn` identity and the protected release/service checkout paths. Re-audit those identities before
changing the pins.

The audited remote has Git worktree support, but it does not contain the selected local base commit
and its selected NInfer checkout is dirty. The helper therefore exports the exact requested commit
to a self-contained bundle and creates a separate repository under the worker root. It does not
attach worktree metadata to the selected checkout's shared Git directory.

The preexisting `/home/aramirezfamily/.local/share/ninfer` path did not exist. Worker roots are
therefore direct children of the already-private `/home/aramirezfamily/.local/share`, so setup
does not create shared parent directories outside the unique worker root.

## Plan and create

The plan command is read-only. It checks the remote host, user, machine ID fingerprint, GPU UUID,
artifact size, source checkout state, toolchain metadata, and candidate path without creating files.

```sh
python3 tools/research/fnn_workspace.py plan \
  --worker E0002-LW04 \
  --revision 70ebb1290dc7abe246c20696a24d21f77faee8d4
```

Create the exact planned workspace with `create`. It exports only the requested commit and its
ancestors into a temporary local Git bundle, transfers that bundle into the unique remote root,
and checks out the requested commit detached. A dirty local worktree is never inferred as a patch;
pass `--patch /absolute/path/to/change.patch` to apply and record an explicit patch. The optional
`--branch name` creates a branch only inside this worker's isolated clone.

```sh
python3 tools/research/fnn_workspace.py create \
  --worker E0002-LW04 \
  --revision 70ebb1290dc7abe246c20696a24d21f77faee8d4
```

The checkout, empty build directory, and result root live under one mode-700 path:

```text
/home/aramirezfamily/.local/share/ninfer-e0002-lw04-70ebb1290dc7-clean/
  source/
  build/
  input/source.bundle
  results/provenance.json
```

The worker root and its checkout/build/input/result directories are mode 700, and the provenance
file is mode 600. The manifest records the local revision, patch and bundle hashes, branch state,
checkout/build/result paths, source checkout state before and after setup, Git/CMake/CUDA/C++/Python
versions, GPU UUID/driver/memory, and the explicit model artifact path/hash/size. The artifact
hash is carried forward from the
E0001-LW04 preflight and its size is checked again; this setup does not reread the 18 GB model.
The build directory is recorded as `not-configured`; this workflow does not build or run anything.
After an authorized build, record its binary and CMake configuration without rerunning it:

```sh
python3 tools/research/fnn_workspace.py record-build \
  --identity e0002-lw04-70ebb1290dc7-clean \
  --binary /home/aramirezfamily/.local/share/ninfer-e0002-lw04-70ebb1290dc7-clean/build/apps/ninfer-serve
```

This metadata-only command accepts binaries only beneath the assigned build directory and records
the binary hash, CMake cache hash, selected build options, and current toolchain/hardware identity.

## GPU serialization

`lock-plan` shows the worker paths and the shared lock target. `gpu-lock` runs the supplied command
from that worker's isolated source tree and exports `NINFER_RESEARCH_ROOT`,
`NINFER_RESEARCH_CHECKOUT`, `NINFER_RESEARCH_BUILD_DIR`, and `NINFER_RESEARCH_RESULT_ROOT`.
It uses a blocking exclusive `flock` on the preexisting owner-only
`/home/aramirezfamily/.local/share` directory. All workers use the same inode, so a waiter sleeps
inside the kernel lock instead of polling; no lock file or shared directory is created. Pass
explicit build and report paths under the exported worker paths.

```sh
python3 tools/research/fnn_workspace.py lock-plan --identity e0002-lw04-70ebb1290dc7-clean
python3 tools/research/fnn_workspace.py gpu-lock \
  --identity e0002-lw04-70ebb1290dc7-clean -- \
  cmake --build /home/aramirezfamily/.local/share/ninfer-e0002-lw04-70ebb1290dc7-clean/build -j
```

Independent workers receive separate Git repositories and paths. A worker can use a local branch
inside its own clone without reserving or changing a branch in the selected production checkout.
GPU work remains serialized across those independent repositories.

## Cleanup

Cleanup defaults to a dry run and validates the generated identity marker, owner, canonical path,
and production-path guards. `--apply` removes only the exact generated research root; it does not
remove the shared checkout, model artifact, or GPU lock target.

```sh
python3 tools/research/fnn_workspace.py cleanup \
  --identity e0002-lw04-70ebb1290dc7-clean
python3 tools/research/fnn_workspace.py cleanup \
  --identity e0002-lw04-70ebb1290dc7-clean --apply
```

Do not use the helper to build from `/home/aramirezfamily/josh/ninfer-4090` or to place outputs
under `/home/aramirezfamily/josh/cinference-4090`; both trees are rejected as protected paths.
