"""Dependency-light test runner for environments without pytest."""
import importlib
import sys
from pathlib import Path
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    module = importlib.import_module("tests.test_c1_c2_task06")
    tests = [getattr(module, name) for name in sorted(dir(module)) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print(f"RESULT {len(tests)-failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
