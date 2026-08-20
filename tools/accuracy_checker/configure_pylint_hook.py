"""Configure the Accuracy Checker Pylint pre-commit hook."""

import subprocess
import sys


subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit", "pylint==2.10.2"], check=True)
print("pre-commit version:")
subprocess.run(["pre-commit", "--version"], check=True)
print("python version:")
subprocess.run([sys.executable, "--version"], check=True)

subprocess.run(["pre-commit", "install"], check=True)
