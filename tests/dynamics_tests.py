import sys
import os
import casadi as ca
from numpy.random import randint, rand

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.dynamics import Dynamics

def sample_dynamics(use_d=False,use_w=False,use_theta=False,nonlinear=False):

    # generate random state and input dimension
    n_x = randint(1,4)
    n_u = randint(1,4)

    # create state and input variables
    x = ca.SX.sym('x',n_x,1)
    u = ca.SX.sym('u',n_u,1)

    # add to output dictionary
    out = {'x':x,'u':u}

    # create random A and B matrices
    A = ca.SX(rand(n_x,n_x))
    B = ca.SX(rand(n_x,n_u))

    # nominal A matrix
    A_nom = A

    # if model uncertainty is present, add it!
    if use_d:

        # generate random dimension for d
        n_d = randint(1,4)

        # create symbolic variable
        d = ca.SX('d',n_d,1)

        # add to model 
        A = A + ca.SX(rand(n_x,n_d))@d

        # add d to output dictionary
        out = out | {'d':d}

    # if theta is present, add it!
    if use_theta:

        # generate random dimension for theta
        n_theta = randint(1,4)

        # create symbolic variable
        theta = ca.SX('theta',n_theta,1)

        # add to nominal model
        A_nom = A_nom + ca.SX(rand(n_x,n_theta))@theta

        # add to output dictionary
        out = out | {'w':w}

    # create random affine part
    c = ca.SX(rand(n_x,1))

    # create next state
    x_next = A@x + B@u + c
    x_next_nom = A_nom@x + B@u + c

    # if noise is present, add it!
    if use_w:

        # generate random dimension for w
        n_w = randint(1,4)

        # create symbolic variable
        w = ca.SX('w',n_w,1)

        # add to model 
        x_next = x_next + ca.SX(rand(n_x,n_w))@w

        # add to output dictionary
        out = out | {'w':w}

    # if model is nonlinear, add quadratic term
    if nonlinear:
        x_next = x_next + x**2
        x_next_nom = x_next_nom + x**2

    return out | {'x_next':x_next, 'x_next_nom':x_next_nom}


def test_affine():

    