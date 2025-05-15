import sys
import os
import casadi as ca

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sample_elements import sample_dynamics, sample_ingredients
from src.dynamics import Dynamics

def test_init():

    # generate sample affine dynamics
    dynamics_dict, dynamics_matrices = sample_dynamics(use_d=True,use_w=True,use_theta=True,nonlinear=False)
    
    # create dynamics
    dynamics = Dynamics(dynamics_dict)

    # create sample QP
    ingredients, *_ = sample_ingredients(dynamics,p=False,horizon=1)

    # check that dynamics are correct
    print('me')

    # construct dynamics from matrices
    A,B,c = dynamics_matrices['A_nom'],dynamics_matrices['B'],dynamics_matrices['c']

    # get equality constraints (sparse) from ingredients
    F = ingredients.sparse['F']
    f = ingredients.sparse['f']

    # create symbolic variable for x and u
    x = ca.SX.sym('x',A.shape[0],1)
    u = ca.SX.sym('x',B.shape[1],1)

    # get next state
    x_next = A@x+B@u+c

    # apply to equality constraints
    



# test to ensure that a quadratic penalty is imposed on slack if present