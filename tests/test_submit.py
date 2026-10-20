from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from tools.dreamrsi.contracts import canonical_hash, validate_submission
from tools.dreamrsi.submit import SubmissionError, freeze_submission


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.name", "Bench Test")
    _git(path, "config", "user.email", "bench@example.invalid")
    (path / "src" / "models").mkdir(parents=True)
    (path / "src" / "models" / "kept.cpp").write_text("int kept;\n", encoding="utf-8")
    (path / "models").mkdir()
    (path / "models" / "weights.bin").write_bytes(b"private weights")
    (path / "README.md").write_text("research source\n", encoding="utf-8")
    (path / "linked.h").symlink_to("src/models/kept.cpp")
    _git(path, "add", "src", "README.md", "linked.h")
    _git(path, "commit", "-qm", "base")
    return path


def _freeze(repo: Path, package: Path, **kwargs):
    return freeze_submission(
        repo,
        package,
        epoch_id="epoch-1",
        worker_id="worker-1",
        hypothesis_id="idea-1",
        contract_id="a" * 64,
        policy_id="b" * 64,
        **kwargs,
    )


def test_freeze_captures_exact_manifest_and_preserves_src_models(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "candidate")
    submission = _freeze(repo, tmp_path / "packages")
    assert validate_submission(submission) == submission
    assert submission["candidate_commit"] == _git(repo, "rev-parse", "HEAD")
    entries = {entry["path"]: entry for entry in submission["source"]["manifest"]}
    assert "src/models/kept.cpp" in entries
    assert entries["linked.h"]["type"] == "symlink"
    assert "models/weights.bin" not in entries
    archive = Path(submission["source"]["archive_path"])
    assert archive.is_file()
    with tarfile.open(archive) as package:
        assert set(package.getnames()) == set(entries)
        assert package.extractfile("src/models/kept.cpp").read() == b"int kept;\n"
    source_manifest = json.loads((archive.parent / "source-manifest.json").read_text())
    assert canonical_hash(source_manifest["entries"]) == submission["source"]["snapshot_sha256"]


def test_repeated_freeze_has_stable_source_and_submission_identity(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "candidate")
    first = _freeze(repo, tmp_path / "packages-a")
    second = _freeze(repo, tmp_path / "packages-b")
    assert first["submission_id"] == second["submission_id"]
    assert first["source"]["snapshot_sha256"] == second["source"]["snapshot_sha256"]
    assert first["source"]["archive_sha256"] == second["source"]["archive_sha256"]


def test_explicit_base_ref_handles_multi_commit_candidate_and_rejects_unrelated_base(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "candidate")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "first.cpp").write_text("int first;\n", encoding="utf-8")
    _git(repo, "add", "first.cpp")
    _git(repo, "commit", "-qm", "candidate change one")
    (repo / "second.cpp").write_text("int second;\n", encoding="utf-8")
    _git(repo, "add", "second.cpp")
    _git(repo, "commit", "-qm", "candidate change two")
    candidate = _git(repo, "rev-parse", "HEAD")

    submission = _freeze(repo, tmp_path / "based", ref=candidate, base_ref=base)
    assert submission["base_commit"] == base
    assert submission["candidate_commit"] == candidate

    _git(repo, "checkout", "--orphan", "unrelated")
    _git(repo, "rm", "-rf", ".")
    (repo / "unrelated.cpp").write_text("int unrelated;\n", encoding="utf-8")
    _git(repo, "add", "unrelated.cpp")
    _git(repo, "commit", "-qm", "unrelated root")
    unrelated = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(SubmissionError, match="ancestor"):
        _freeze(repo, tmp_path / "bad-base", ref=candidate, base_ref=unrelated)


def test_dirty_source_requires_explicit_permission_and_tracks_untracked_content(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "candidate")
    (repo / "src" / "models" / "kept.cpp").write_text("int changed;\n", encoding="utf-8")
    (repo / "new_source.cpp").write_text("int new_source;\n", encoding="utf-8")
    with pytest.raises(SubmissionError, match="dirty"):
        _freeze(repo, tmp_path / "denied")
    submission = _freeze(repo, tmp_path / "accepted", include_dirty=True)
    assert submission["source"]["dirty"] is True
    assert submission["candidate_commit"] is None
    paths = {entry["path"] for entry in submission["source"]["manifest"]}
    assert "new_source.cpp" in paths
    assert "src/models/kept.cpp" in paths
    assert all("weights.bin" not in path for path in paths)


def test_freeze_rejects_external_symlink_and_lfs_pointer(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "symlink")
    (repo / "escape").symlink_to("../../outside")
    _git(repo, "add", "escape")
    _git(repo, "commit", "-qm", "unsafe symlink")
    with pytest.raises(SubmissionError, match="symlink"):
        _freeze(repo, tmp_path / "unsafe-package")

    lfs_repo = _repo(tmp_path / "lfs")
    (lfs_repo / "large.dat").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 10\n"
    )
    _git(lfs_repo, "add", "large.dat")
    _git(lfs_repo, "commit", "-qm", "lfs pointer")
    with pytest.raises(SubmissionError, match="LFS"):
        _freeze(lfs_repo, tmp_path / "lfs-package")


def test_freeze_rejects_submodules_instead_of_silently_omitting_them(tmp_path: Path) -> None:
    child = _repo(tmp_path / "child")
    parent = _repo(tmp_path / "parent")
    subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "-C", str(parent),
         "submodule", "add", str(child), "vendor/child"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _git(parent, "commit", "-qam", "add submodule")
    with pytest.raises(SubmissionError, match="submodule.*unsupported"):
        _freeze(parent, tmp_path / "submodule-package")


def test_dirty_worktree_private_and_ignored_files_are_not_packaged(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "candidate")
    (repo / ".env").write_text("SECRET=private\n", encoding="utf-8")
    (repo / "untracked.cpp").write_text("int selected;\n", encoding="utf-8")
    (repo / "ignored.dat").write_text("ignored\n", encoding="utf-8")
    with (repo / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("ignored.dat\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "ignore generated data")
    submission = _freeze(repo, tmp_path / "package", include_dirty=True)
    paths = {entry["path"] for entry in submission["source"]["manifest"]}
    assert ".env" not in paths
    assert "ignored.dat" not in paths
    assert "untracked.cpp" in paths


def test_research_memory_roots_stay_excluded_even_with_dirty_inclusion(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "candidate")
    (repo / "selected.cpp").write_text("int selected;\n", encoding="utf-8")
    (repo / "Dream-RSI").mkdir()
    (repo / "Dream-RSI" / "notes.md").write_text("private research\n", encoding="utf-8")
    (repo / "Dream-RSI.zip").write_bytes(b"research archive")
    (repo / ".obsidian").mkdir()
    (repo / ".obsidian" / "vault.md").write_text("private memory\n", encoding="utf-8")
    (repo / ".omo").mkdir()
    (repo / ".omo" / "state.json").write_text("{}\n", encoding="utf-8")
    submission = _freeze(repo, tmp_path / "package", include_dirty=True)
    paths = {entry["path"] for entry in submission["source"]["manifest"]}
    assert "selected.cpp" in paths
    assert not any(path.startswith(("Dream-RSI/", ".obsidian/", ".omo/")) for path in paths)
    assert "Dream-RSI.zip" not in paths
