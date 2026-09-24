import errno
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.research.fnn_workspace import (
    build_identity,
    paths_overlap,
    validate_child_path,
    validate_identity,
    validate_research_path,
    worker_slug,
)
from tools.research.fnn_workspace_remote import (
    checked_file,
    guard_path,
    require_identity_marker,
    verify_directory_flock,
)


class FnnWorkspaceGuards(unittest.TestCase):
    def test_identity_is_deterministic_and_distinguishes_worker_and_patch(self):
        revision = "70ebb1290dc7abe246c20696a24d21f77faee8d4"
        patch = "0123456789abcdef" * 4
        identity = build_identity("E0002-LW04", revision, patch)
        self.assertEqual(identity, "e0002-lw04-70ebb1290dc7-0123456789ab")
        self.assertEqual(identity, build_identity("E0002-LW04", revision, patch))
        self.assertNotEqual(identity, build_identity("E0002-LW05", revision, patch))
        self.assertNotEqual(identity, build_identity("E0002-LW04", revision, None))
        self.assertEqual(validate_identity(identity), identity)

    def test_path_guard_accepts_only_unique_research_children(self):
        root = "/home/aramirezfamily/.local/share"
        allowed = validate_research_path(root + "/ninfer-e0002-lw04-70ebb1290dc7-clean", root)
        self.assertEqual(allowed, root + "/ninfer-e0002-lw04-70ebb1290dc7-clean")
        with self.assertRaisesRegex(ValueError, "child of the research root"):
            validate_research_path(root, root)
        with self.assertRaisesRegex(ValueError, "child of the research root"):
            validate_research_path("/tmp/ninfer-research", root)

    def test_path_guard_rejects_production_roots_and_their_ancestors(self):
        root = "/home/aramirezfamily"
        with self.assertRaisesRegex(ValueError, "protected production path"):
            validate_research_path("/home/aramirezfamily/josh/ninfer-4090/build-sm89", root)
        with self.assertRaisesRegex(ValueError, "protected production path"):
            validate_research_path("/home/aramirezfamily/josh", root)
        self.assertTrue(paths_overlap("/home/aramirezfamily/josh", "/home/aramirezfamily/josh/cinference-4090"))

    def test_worker_identifier_rejects_path_components(self):
        for unsafe in ("../x", "worker/name", "worker..name", "trailing-"):
            with self.subTest(worker=unsafe), self.assertRaises(ValueError):
                worker_slug(unsafe)

    def test_binary_path_must_stay_inside_the_assigned_build_directory(self):
        build = "/home/aramirezfamily/.local/share/ninfer-worker/build"
        self.assertEqual(validate_child_path(build + "/apps/ninfer-serve", build), build + "/apps/ninfer-serve")
        with self.assertRaisesRegex(ValueError, "inside its assigned directory"):
            validate_child_path("/home/aramirezfamily/josh/cinference-4090/releases/server", build)
        with self.assertRaisesRegex(ValueError, "inside its assigned directory"):
            validate_child_path(build, build)

    def test_remote_path_guard_rejects_symlink_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "worker"
            outside = Path(temporary) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "input-link").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(RuntimeError, "symlink component"):
                guard_path(str(root / "input-link" / "source.bundle"), str(root), [])

    def test_remote_input_guard_rejects_outside_paths_before_chmod(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary).resolve()
            input_directory = temporary_root / "input"
            input_directory.mkdir()
            outside = temporary_root / "source.bundle"
            outside.write_bytes(b"owned file outside the assigned input directory")
            os.chmod(outside, 0o644)
            digest = hashlib.sha256(outside.read_bytes()).hexdigest()

            with self.assertRaisesRegex(RuntimeError, "outside the unique research root"):
                checked_file(str(outside), "source.bundle", digest, os.getuid(), str(input_directory), [])

            self.assertEqual(os.stat(outside).st_mode & 0o777, 0o644)

    def test_remote_input_guard_accepts_only_exact_named_file_and_records_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary).resolve()
            input_directory = temporary_root / "input"
            input_directory.mkdir()
            bundle = input_directory / "source.bundle"
            bundle.write_bytes(b"verified bundle")
            os.chmod(bundle, 0o644)
            digest = hashlib.sha256(bundle.read_bytes()).hexdigest()

            with self.assertRaisesRegex(RuntimeError, "unexpected input file path"):
                checked_file(
                    str(input_directory / "other.file"),
                    "source.bundle",
                    digest,
                    os.getuid(),
                    str(input_directory),
                    [],
                )

            with self.assertRaisesRegex(RuntimeError, "input checksum mismatch"):
                checked_file(str(bundle), "source.bundle", "0" * 64, os.getuid(), str(input_directory), [])
            self.assertEqual(os.stat(bundle).st_mode & 0o777, 0o644)

            result = checked_file(str(bundle), "source.bundle", digest, os.getuid(), str(input_directory), [])
            self.assertEqual(result["sha256"], digest)
            self.assertEqual(result["size_bytes"], len(b"verified bundle"))
            self.assertEqual(os.stat(bundle).st_mode & 0o777, 0o600)

    def test_identity_marker_must_be_a_private_regular_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "worker"
            root.mkdir(mode=0o700)
            marker = root / ".ninfer-research-identity"
            marker.write_text("e0002-lw05-70ebb1290dc7-clean\n", encoding="utf-8")
            os.chmod(marker, 0o600)
            self.assertEqual(
                require_identity_marker(str(root), "e0002-lw05-70ebb1290dc7-clean"),
                str(marker),
            )

            marker.unlink()
            target = Path(temporary) / "external-marker"
            target.write_text("e0002-lw05-70ebb1290dc7-clean\n", encoding="utf-8")
            marker.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "cannot be opened safely"):
                require_identity_marker(str(root), "e0002-lw05-70ebb1290dc7-clean")

    def test_lock_capability_probe_accepts_contention_and_rejects_unsupported_filesystem(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("tools.research.fnn_workspace_remote.fcntl.flock") as lock:
                verify_directory_flock(temporary)
                self.assertEqual(lock.call_count, 2)

                lock.reset_mock(side_effect=True)
                lock.side_effect = BlockingIOError(errno.EAGAIN, "lock held by another worker")
                verify_directory_flock(temporary)
                self.assertEqual(lock.call_count, 1)

                lock.reset_mock(side_effect=True)
                lock.side_effect = OSError(errno.ENOTSUP, "filesystem does not support flock")
                with self.assertRaisesRegex(RuntimeError, "shared directory flock is unavailable"):
                    verify_directory_flock(temporary)


if __name__ == "__main__":
    unittest.main()
