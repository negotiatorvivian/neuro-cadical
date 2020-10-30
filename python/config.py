import os
import sys

PYTHON_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(PYTHON_DIR)
CGEN_PATH = os.path.join(PROJECT_DIR, "cgen", "cgen")

CADICAL_PATH = os.path.join(PROJECT_DIR, "cadical", "build", "cadical")
SATENV_PATH = os.path.join(PROJECT_DIR, 'satenv', 'scripts')
TORCHSCRIPT_PATH = os.path.join(PROJECT_DIR, "torchscript")
TORCHSCRIPT_MODEL_PATH = os.path.join(TORCHSCRIPT_PATH, "deploy")
HOME = os.path.expandvars("$HOME")

