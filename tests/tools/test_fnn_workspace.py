import unittest

from tools.research.fnn_workspace import (
    build_identity,
    paths_overlap,
    validate_child_path,
    validate_identity,
    validate_research_path,
    worker_slug,
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


if __name__ == "__main__":
    unittest.main()
