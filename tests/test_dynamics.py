import sys
import os
import casadi as ca
from numpy.random import randint, rand

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.dynamics import Dynamics

def sample_dynamics(use_d:bool=False,use_w:bool=False,use_theta:bool=False,nonlinear:bool=False) -> dict:
    """
    This function generates a dictionary that can be used to setup a Dynamics object.
    The true dynamics are given by

        x_next = (A_1 + A_2@d) @ x + x**2 + B@u + c + B_d@w,

    whereas the nominal dynamics are given by

        x_next_nom = (A_1 + A_3@theta) @ x + x**2 + B@u + c.

    The terms A_1,A_2,A_3,B,B_d,c are randmly generated with entries between 0 and 1.

    Args:
        use_d (bool, optional): if true the dynamics contain model uncertainty d
        use_w (bool, optional): if true the dynamics contain noise w
        use_theta (bool, optional): if true the dynamics contain nominal model theta
        nonlinear (bool, optional): if true the dynamics contain quadratic term x**2
        
    Returns:
        dict: dictionary that can be used to setup a Dynamics object
        dict: dictionary containing A=A_1+A_2@d, A_nom=A_1+A_3@theta, B, c
    """

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
        d = ca.SX.sym('d',n_d,1)

        # add to model 
        A = A + ca.SX(rand(n_x,n_d))@d

        # add d to output dictionary
        out = out | {'d':d}

    # if theta is present, add it!
    if use_theta:

        # generate random dimension for theta
        n_theta = randint(1,4)

        # create symbolic variable
        theta = ca.SX.sym('theta',n_theta,1)

        # add to nominal model
        A_nom = A_nom + ca.SX(rand(n_x,n_theta))@theta

        # add to output dictionary
        out = out | {'theta':theta}

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
        w = ca.SX.sym('w',n_w,1)

        # add to model 
        x_next = x_next + ca.SX(rand(n_x,n_w))@w

        # add to output dictionary
        out = out | {'w':w}

    # if model is nonlinear, add quadratic term
    if nonlinear:
        x_next = x_next + x**2
        x_next_nom = x_next_nom + x**2

    return out | {'x_next':x_next, 'x_next_nom':x_next_nom}, {'A':A,'A_nom':A_nom,'B':B,'c':c}


def test_affine():
    """
    Test if the dynamics correctly identify that the model is affine.
    """

    # generate affine dynamics
    dynamics_1 = Dynamics(sample_dynamics(use_d=True,use_w=True,use_theta=True)[0])
    dynamics_2 = Dynamics(sample_dynamics()[0])

    assert dynamics_1._is_affine, 'Model is not recognized to be affine.'
    assert dynamics_2._is_affine, 'Model is not recognized to be affine.'

    # generate nonlinear dynamics
    dynamics_nonlinear_1 = Dynamics(sample_dynamics(nonlinear=True)[0])
    dynamics_nonlinear_2 = Dynamics(sample_dynamics(use_d=True,use_w=True,use_theta=True,nonlinear=True)[0])

    assert not dynamics_nonlinear_1._is_affine, 'Model is not recognized to be nonlinear.'
    assert not dynamics_nonlinear_2._is_affine, 'Model is not recognized to be nonlinear.'

def test_nominal_and_derivatives():
    """
    Test correctness of nominal dynamics generation and their derivatives.
    """

    # generate affine dynamics
    dynamics_1_dict,dynamics_1_matrices = sample_dynamics(use_d=True,use_w=True,use_theta=True)
    dynamics_1 = Dynamics(dynamics_1_dict)
    dynamics_2_dict,dynamics_2_matrices = sample_dynamics()
    dynamics_2 = Dynamics(dynamics_2_dict)

    # initialization should contain None values in x and u
    assert len(dynamics_1.init) == 0, 'Init is not empty.'
    assert len(dynamics_2.init) == 0, 'Init is not empty.'

    # get dimensions
    n1 = dynamics_1.dim
    n2 = dynamics_2.dim

    # create initializations
    x1 = ca.DM(rand(n1['x'],1))
    u1 = ca.DM(rand(n1['u'],1))
    d1 = ca.DM(rand(n1['d'],1))
    w1 = ca.DM(rand(n1['w'],1))
    x2 = ca.DM(rand(n2['x'],1))
    u2 = ca.DM(rand(n2['u'],1))

    # create initialization dictionaries
    init1 = {'x':x1,'u':u1,'d':d1}
    init2 = {'x':x2,'u':u2}

    # pass initializations
    dynamics_1._set_init(init1)
    dynamics_2._set_init(init2)

    # helper function to check if two dictionaries containing DM variables coincide
    def compare_dicts(dict1,dict2):

        # check that keys are equal
        dict_equal = dict1.keys() == dict2.keys()

        # check that numerical values are equal
        dict_equal = dict_equal and all([ca.mmax(dict1[key] - dict2[key]) == 0 for key in dict1.keys()])

        return dict_equal

    # check that initializations are correct
    assert compare_dicts(dynamics_1.init,init1), 'Initialization is not handled correctly.'
    assert compare_dicts(dynamics_2.init,init2), 'Initialization is not handled correctly.'

    # add one more initialization
    dynamics_1._set_init({'w':w1})
    assert compare_dicts(dynamics_1.init,init1 | {'w':w1}), 'Passing initialization in two steps failed.'

    # try adding a wrong type
    test_passed = False
    try:
        dynamics_1._set_init({'ciao':1})
    except:
        test_passed = True

    assert test_passed, 'Wrong initialization was not rejected.'

    # check that f and fc coincide with non-noisy model
    assert ca.sum1(ca.fabs(ca.DM(ca.cse(dynamics_2.f_nom(dynamics_2.param_nom['x'],dynamics_2.param_nom['u']) - dynamics_2.x_next)))) == 0, 'Nominal and true dynamics should coincide'
    assert ca.sum1(ca.fabs(ca.DM(ca.cse(dynamics_2.f(dynamics_2.param_nom['x'],dynamics_2.param_nom['u']) - dynamics_2.x_next_nom)))) == 0, 'Nominal and true dynamics should coincide'
    
    # check derivatives
    assert ca.mmin(dynamics_2.A(x2,u2) == ca.DM(dynamics_2_matrices['A'])), 'x derivative does not match.'
    assert ca.mmin(dynamics_2.B(x2,u2) == ca.DM(dynamics_2_matrices['B'])), 'u derivative does not match.'
    assert ca.mmin(dynamics_2.B_nom(x2,u2) == ca.DM(dynamics_2_matrices['B'])), 'u derivative does not match.'
    assert ca.mmin(dynamics_2.A_nom(x2,u2) == ca.DM(dynamics_2_matrices['A'])), 'u derivative does not match.'

    # create model where nonlinearity occurs only if d is not zero
    # x_next = A*x + B*u + c + d*x**2
    # dyn = {'x':x, 'u':u, 'd':d, 'x_next':x_next, 'x_dot':x_dot, 'd0':0}
    # mod = scenario()
    # mod.makeDynamics(dyn)

    # # check that model is recognized as nonlinear
    # if not mod.dyn.type == 'nonlinear':
    #     raise Exception('Model was not recognized as nonlinear.')