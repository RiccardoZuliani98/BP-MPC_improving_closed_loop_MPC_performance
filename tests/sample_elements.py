import sys
import os
import casadi as ca
from numpy.random import randint, rand
from typing import Optional, Union

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingredients import Ingredients
from src.dynamics import Dynamics
from src.qp import QP
from src.upper_level import UpperLevel
from src.utils import quadCostAndBounds, gradient_descent

def sample_dynamics(
        use_d:bool=False,
        use_w:bool=False,
        use_theta:bool=False,
        nonlinear:bool=False
    ) -> dict:
    """
    This function generates a dictionary that can be used to setup a Dynamics object.
    The true dynamics are given by

        x_next = (A_1 + A_2@d) @ x + x**2 + B@u + c + B_d@w,

    whereas the nominal dynamics are given by

        x_next_nom = (A_1 + A_3@theta) @ x + x**2 + B@u + c.

    The terms A_1,A_2,A_3,B,B_d,c are randomly generated with entries between 0 and 1.

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


def sample_ingredients(
        dynamics:Dynamics,
        p:Optional[bool]=True,
        pf:Optional[bool]=False,
        slack:Optional[bool]=False,
        horizon:Optional[int]=1
    ) -> Union[Ingredients, Optional[ca.SX], Optional[ca.SX]]:
    """
    Generate sample cost and constraint ingredients for a given system dynamics, with optional parameters and slack variables.
    Args:
        dynamics: An object representing the system dynamics, expected to have a 'dim' attribute with keys 'x' (state dimension) and 'u' (input dimension).
        p (bool, optional): If True, include a symbolic parameter in the state cost matrix. Defaults to True.
        pf (bool, optional): If True, include a symbolic parameter in the input cost matrix. Defaults to False.
        slack (bool, optional): If True, include slack variables and penalties in the constraints and cost. Defaults to False.
        horizon (int, optional): Prediction horizon for the ingredients. Defaults to 1.
    Returns:
        tuple: (ingredients, p, pf)
            - ingredients: An Ingredients object containing the generated cost and constraint dictionaries.
            - p: The symbolic parameter for the state cost if enabled, otherwise None.
            - pf: The symbolic parameter for the input cost if enabled, otherwise None.
    """

    # get dimensions for simplicity
    n = dynamics.dim

    # create sample state cost
    Q_half = ca.DM(rand(n['x'],n['x']))
    Q = Q_half@Q_half.T + 0.01*ca.DM.eye(n['x'])

    if p:
        p = ca.SX.sym('p',2,1)#ca.SX.sym('p',randint(1,4),1)
        Q = Q + ca.SX.eye(Q.shape[0])*ca.sum1(p)
    else:
        p = None
    
    if pf:
        pf = ca.SX.sym('p',randint(1,4),1)
        R = R + ca.SX.eye(R.shape[0])*pf
    else:
        pf = None

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

    # create ingredients
    ingredients = Ingredients(horizon,dynamics,cost,constraints)

    return ingredients, p, pf

def sample_upper_level(p:ca.SX,mpc:QP,pf:ca.SX=None,horizon:int=2):

    # initialize upper level
    if pf is not None:
        # create upper level with parameter
        upper_level = UpperLevel(p=p,pf=pf,horizon=horizon,mpc=mpc)
    else:
        # create upper level without parameter
        upper_level = UpperLevel(p=p,horizon=horizon,mpc=mpc)

    # extract closed-loop variables for upper level
    x_cl = ca.vec(upper_level.param['x_cl'])
    u_cl = ca.vec(upper_level.param['u_cl'])

    # get dimensions
    n_x, n_u = upper_level.param['x_cl'].shape[0], upper_level.param['u_cl'].shape[0]

    # create random cost
    Q_temp, R_temp = ca.DM(rand(n_x,n_x)), ca.DM(rand(n_u,n_u))
    Q = Q_temp@Q_temp.T + 0.01*ca.DM.eye(n_x)
    R = R_temp@R_temp.T + 0.01*ca.DM.eye(n_u)

    # create random bounds
    x_max = ca.DM(rand(n_x))
    x_min = -x_max
    u_max = ca.DM(rand(n_u))
    u_min = -u_max

    # create tracking cost and constraint violation
    track_cost, cst_viol_l1, _ =  quadCostAndBounds(Q,R,x_cl,u_cl,x_max,x_min)

    # set cost
    upper_level.set_cost(track_cost+cst_viol_l1,track_cost,cst_viol_l1)

    # create update function
    upper_level.set_alg(*gradient_descent(rho=0.0001,eta=0.51,log=True))

    return upper_level