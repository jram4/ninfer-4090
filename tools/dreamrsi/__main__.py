"""Run the research bench without installing a package."""
import importlib
import sys

COMMANDS = {
    # Control plane
    "submit": "submit",
    "queue": "queue",
    "promote": "promotion",
    "policy": "policy",
    "coordinator": "coordinator",
    "transport": "transport",
    "epoch": "epochs",
    "worker": "workers",
    # Evidence plane
    "evaluate": "evaluator",
    "capture": "capture",
    "replay": "replay",
    "corpus": "corpus",
    "compare": "quick_eval",
    "stage": "stage",
}

DESCRIPTIONS = {
    "submit": "freeze an immutable source snapshot for evaluation",
    "queue": "inspect or operate the local submission queue",
    "promote": "plan or perform a compare-and-swap promotion",
    "policy": "build or inspect a trusted evaluation policy",
    "coordinator": "claim, evaluate, and finish queued submissions",
    "transport": "plan or run trusted-harness installation on the build host",
    "epoch": "manage durable manual research epochs",
    "worker": "register bounded workers and dispatch assignments",
    "evaluate": "evaluate one frozen submission under a trusted policy",
    "capture": "capture bounded private model traffic",
    "replay": "replay one captured request",
    "corpus": "build a session-stratified screen/holdout plan",
    "compare": "reduce a paired baseline/candidate comparison",
    "stage": "run a trusted stage program",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("NInfer research bench")
        print("Usage: python3.11 -m tools.dreamrsi <command> [options]\n")
        for name in COMMANDS:
            print(f"  {name:<12} {DESCRIPTIONS[name]}")
        return 0
    command = COMMANDS.get(sys.argv[1])
    if command is None:
        raise SystemExit(f"Unknown command: {sys.argv[1]}")
    return importlib.import_module(f"tools.dreamrsi.{command}").main(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
