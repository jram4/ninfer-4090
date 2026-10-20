"""Run the research bench without installing a package."""
import importlib
import sys

COMMANDS = {
    "submit": "submit", "queue": "queue", "evaluate": "evaluator",
    "promote": "promotion", "capture": "capture", "replay": "replay",
    "corpus": "corpus", "compare": "quick_eval", "epoch": "epochs",
    "worker": "workers",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("NInfer research bench\nCommands: " + ", ".join(COMMANDS))
        return 0
    command = COMMANDS.get(sys.argv[1])
    if command is None:
        raise SystemExit(f"Unknown command: {sys.argv[1]}")
    return importlib.import_module(f"tools.dreamrsi.{command}").main(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
