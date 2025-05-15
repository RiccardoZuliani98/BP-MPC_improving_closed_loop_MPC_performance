import sys
import os
import casadi as ca
from numpy.random import randint, rand
from test_dynamics import sample_dynamics

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.scenario import scenario

