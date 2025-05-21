import sys
import os
import casadi as ca
from numpy.random import randint, rand
from typing import Optional, Tuple
from copy import copy

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qp import QP
from src.dynamics import Dynamics
from src.ingredients import Ingredients
from src.upper_level import UpperLevel
from utils.parameter_update import gradient_descent
from utils.cost_utils import quad_cost_and_bounds

def sample_dynamics(
        use_d:bool=False,
        use_w:bool=False,
        use_theta:bool=False,
        nonlinear:bool=False
    ) -> Tuple[dict,dict]:
    """
    This function generates a dictionary that can be used to set up a Dynamics object.
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
        dict: dictionary that can be used to set up a Dynamics object
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
        dim:dict,
        p:Optional[bool]=True,
        pf:Optional[bool]=False,
        slack:Optional[bool]=False,
        horizon:Optional[int]=1
    ) -> Tuple[Optional[ca.SX],Optional[ca.SX],Optional[dict],Optional[dict]]:
    """
    Generate sample ingredients for testing optimal control problems, including cost matrices, references, and constraints.
    Args:
        dim (dict): Dictionary specifying dimensions with keys 'x' (state dimension) and 'u' (input dimension).
        p (Optional[bool], optional): If True, include symbolic parameter 'p' in state cost matrices. Defaults to True.
        pf (Optional[bool], optional): If True, include symbolic parameter 'pf' in input cost matrices. Defaults to False.
        slack (Optional[bool], optional): If True, include slack variables and penalties in constraints and cost. Defaults to False.
        horizon (Optional[int], optional): Prediction horizon length. Defaults to 1.
    Returns:
        Tuple[
            Optional[ca.SX],         # Symbolic parameter 'p' if enabled, else None
            Optional[ca.SX],         # Symbolic parameter 'pf' if enabled, else None
            dict,                    # Cost dictionary with keys: 'Qx', 'Ru', 'x_ref', 'u_ref', and optionally 's_quad', 's_lin'
            dict                     # Constraints dictionary with keys: 'Hx', 'hx', 'Hu', 'hu', and optionally 'Hx_e'
        ]
    """

    # create sample state cost
    Q_half = [ca.DM(rand(dim['x'],dim['x'])) for _ in range(horizon)]
    Q = [elem@elem.T + 0.01*ca.DM.eye(dim['x']) for elem in Q_half]

    # create sample input cost
    R_half = [ca.DM(rand(dim['u'],dim['u'])) for _ in range(horizon)]
    R = [elem@elem.T + 0.01*ca.DM.eye(dim['u']) for elem in R_half]

    if p:
        p = ca.SX.sym('p',randint(1,4),1)
        Q = [elem + ca.SX.eye(dim['x'])*ca.sum1(p) for elem in Q]
    else:
        p = None
    
    if pf:
        pf = ca.SX.sym('p',randint(1,4),1)
        R = [elem + ca.SX.eye(dim['u'])*ca.sum1(pf) for elem in R]
    else:
        pf = None

    # create sample references
    x_ref = [ca.DM(rand(dim['x'])) for _ in range(horizon)]
    u_ref = [ca.DM(rand(dim['u'])) for _ in range(horizon)]

    # create sample constraints
    n_hx,n_hu = randint(1,3),randint(1,3)
    Hx = [ca.DM(rand(n_hx,dim['x'])) for _ in range(horizon)]
    hx = [ca.DM(rand(n_hx)) for _ in range(horizon)]
    Hu = [ca.DM(rand(n_hu,dim['u'])) for _ in range(horizon)]
    hu = [ca.DM(rand(n_hu)) for _ in range(horizon)]

    # create output dictionaries
    constraints = {'Hx':Hx, 'hx':hx, 'Hu':Hu, 'hu':hu}
    cost = {'Qx':Q, 'Ru':R, 'x_ref':x_ref, 'u_ref':u_ref}

    # check if slack variables are required
    if slack:

        # create quadratic penalty
        s_quad = [ca.DM(rand()**2) for _ in range(horizon)]

        # create linear penalty
        s_lin = [ca.DM(rand()**2) for _ in range(horizon)]

        # create slack matrix
        n_hx_e = randint(1,n_hx) if n_hx > 1 else n_hx
        Hx_e = [ca.DM(rand(n_hx,n_hx_e)) for _ in range(horizon)]

        # add to constraint dictionary
        constraints['Hx_e'] = Hx_e

        # add to cost dictionary
        cost['s_quad'] = s_quad
        cost['s_lin'] = s_lin

    return p, pf, cost, constraints

def sample_mpc(
        horizon:int=5,
        use_d:bool=False,
        use_w:bool=False,
        use_theta:bool=False,
        nonlinear:bool=False,
        use_p:bool=True,
        use_pf:bool=False,
        use_slack:bool=False
    ) -> Tuple[Dynamics, Ingredients, dict, dict]:
    """
    Generate dummy dynamics and ingredients.

    Args:
        horizon (int, optional): Prediction horizon for the MPC. Defaults to 5.
        use_d (bool, optional): Whether to include disturbance in the dynamics. Defaults to False.
        use_w (bool, optional): Whether to include process noise in the dynamics. Defaults to False.
        use_theta (bool, optional): Whether to include parameter uncertainty in the dynamics. Defaults to False.
        nonlinear (bool, optional): Whether to use nonlinear dynamics. Defaults to False.
        use_p (bool, optional): Whether to add a parameter p in the ingredients. Defaults to True.
        use_pf (bool, optional): Whether to add a parameter pf in the ingredients. Defaults to True.
        use_slack (bool, optional): Whether to include slack variables. Defaults to False.

    Returns:
        Tuple[Dynamics, Ingredients, dict, dict]: 
            - Dynamics: Generated dynamics object.
            - Ingredients: Parsed ingredients object for the MPC.
            - dict: Dictionary containing cost, constraints, and model information.
            - dict: Dictionary containing the symbolic variables p and pf (None if not initialized).
    """

    # create dummy dynamics
    dynamics_dict, _ = sample_dynamics(use_d=use_d,use_w=use_w,use_theta=use_theta,nonlinear=nonlinear)
    dynamics = Dynamics(dynamics_dict)

    # get model
    _ = dynamics.linearize(horizon=horizon)[0]

    # create dictionary that can be passed to ingredients
    p,pf,cost,constraints = sample_ingredients(dynamics.dim,p=use_p,pf=use_pf,slack=use_slack,horizon=horizon)
    # ing_dict = cost | constraints | model

    # create ingredients
    ingredients = Ingredients(horizon,dynamics,cost,constraints)

    # create dictionary with symbolic variables
    out_dict = {'p':p,'pf':pf,'theta':dynamics_dict['theta']} if use_theta else {'p':p,'pf':pf}

    return dynamics, ingredients, out_dict

def sample_upper_level(p:ca.SX,mpc:QP,pf:ca.SX=None,horizon:int=2):
    """
    Initializes and configures an upper-level optimization problem for closed-loop MPC performance improvement.

    This function creates an instance of the `UpperLevel` class, optionally with an additional parameter `pf`, 
    and sets up a random quadratic tracking cost and state constraints for the closed-loop system. 
    It also configures a gradient descent algorithm for updating the upper-level variables.

    Args:
        p (ca.SX): Symbolic parameter for the upper-level problem.
        mpc (QP): An instance of the lower-level QP-based MPC controller.
        pf (ca.SX, optional): Additional symbolic parameter for the upper-level problem. Defaults to None.
        horizon (int, optional): Prediction horizon for the upper-level problem. Defaults to 2.

    Returns:
        UpperLevel: Configured upper-level optimization problem instance.
    """

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
    q_temp, r_temp = ca.DM(rand(n_x,n_x)), ca.DM(rand(n_u,n_u))
    q = q_temp@q_temp.T + 0.01*ca.DM.eye(n_x)
    r = r_temp@r_temp.T + 0.01*ca.DM.eye(n_u)

    # create random bounds
    x_max = ca.DM(rand(n_x))
    x_min = -x_max

    # create tracking cost and constraint violation
    track_cost, cst_viol_l1, _ = quad_cost_and_bounds(q,r,x_cl,u_cl,x_max,x_min)

    # set cost
    upper_level.set_cost(track_cost+cst_viol_l1,track_cost,cst_viol_l1)

    # create update function
    upper_level.set_alg(*gradient_descent(rho=0.0001,eta=0.51,log=True))

    return upper_level