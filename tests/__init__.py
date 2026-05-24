import sys
from os.path import abspath, dirname, join as path_join


# inserts the root of the repo folder into the python path so we can just import the codebase
# without relative import issues or needing to install the package.
repo_root = abspath(dirname(dirname(abspath(__file__))))

if repo_root not in sys.path:
    sys.path.insert(0, path_join(repo_root, "src"))
    print(f"\nAdding repo root to sys.path: {repo_root} for running tests\n")
