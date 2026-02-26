# eval/test/test.py – replace this file when you switch tasks
import sys
from pathlib import Path
workspace = Path(__file__).resolve().parent.parent.parent / "workspace"
sys.path.insert(0, str(workspace))

import add
assert add.add(1, 2) == 3
assert add.add(-1, 1) == 0
assert add.add(0, 0) == 0
print("PASSED 3 3")