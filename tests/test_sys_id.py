import sys
import os
import casadi as ca
import pytest

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.symbolic_var import SymbolicVar

