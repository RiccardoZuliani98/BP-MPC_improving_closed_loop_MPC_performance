import sys
import os
import casadi as ca
from numpy.random import randint, rand

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.ingredients import Ingredients
from test_dynamics import sample_dynamics
from src.dynamics import Dynamics

def sample_QP(slack=False,nonlinear=False):

    # generate sample affine dynamics
    dynamics_dict, dynamics_matrices = sample_dynamics(use_d=True,use_w=True,
                                                       use_theta=True,nonlinear=nonlinear)
    
    # create dynamics object
    dynamics = Dynamics(dynamics_dict)

    # get dimensions for simplicity
    n = dynamics.dim

    # horizon is 1
    horizon = 1

    # create sample state cost
    Q_half = ca.DM(rand(n['x'],n['x']))
    Q = Q_half@Q_half.T + 0.01*ca.DM.eye(n['x'])

    # create sample input cost
    R_half = ca.DM(rand(n['u'],n['u']))
    R = R_half@R_half.T + 0.01*ca.DM.eye(n['u'])

    # create sample references
    x_ref = ca.DM(rand(n['x']))
    u_ref = ca.DM(rand(n['u']))

    # create sample constraints
    Hx = ca.DM(rand(randint(1,3),n['x']))
    hx = ca.DM(rand(Hx.shape[0]))
    Hu = ca.DM(rand(randint(1,3),n['u']))
    hu = ca.DM(rand(Hu.shape[0]))

    # create output dictionaries
    constraints = {'Hx':Hx, 'hx':hx, 'Hu':Hu, 'hu':hu}
    cost = {'Qx':Q, 'Ru':R, 'x_ref':x_ref, 'u_ref':u_ref}

    # check if slack is required
    if slack:

        # create quadratic penalty
        s_quad = ca.DM(rand()**2)

        # create linear penalty
        s_lin = ca.DM(rand()**2)

        # create slack matrix
        Hx_e = ca.DM(rand(n['x'],randint(1,Hx.shape[0])))

        # add to constraint dictionary
        constraints['Hx_e'] = Hx_e

        # add to cost dictionary
        cost['s_quad'] = s_quad
        cost['s_lin'] = s_lin

    return dynamics, constraints, cost, dynamics_matrices

def test_init():

    # create sample QP
    dynamics, constraints, cost, dynamics_matrices = sample_QP()

    # create ingredients
    ingredients = Ingredients(1,dynamics,cost,constraints)

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