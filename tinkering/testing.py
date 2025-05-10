import sys
import os

# add root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scenario import Scenario
from src.dynamics import Dynamics
from src.qp import QP
from src.Ingredients import Ingredients
import src.utils as utils
import examples.dynamics.cart_pend_theta as cart_pend
import casadi as ca
from src.upper_level import UpperLevel
from src.simVar import simVar
import jax.numpy as jnp
import numpy as np
from jaxadi import convert

def get_scenario():

    # decide what to compile
    COMPILE_DYNAMICS = False
    COMPILE_QP_SPARSE = False
    COMPILE_QP_DENSE = False
    COMPILE_JAC = False


    ### CREATE DYNAMICS ------------------------------------------------------------------------

    # create dictionary with parameters of cart pendulum
    dyn_dict = cart_pend.dynamics(dt=0.015)

    # model uncertainty parameter
    theta = dyn_dict['theta']

    # create dynamics object
    dyn = Dynamics(dyn_dict)

    # get state and input dimensions
    n_x, n_u, n_w, n_d = dyn.dim['x'], dyn.dim['u'], dyn.dim['w'], dyn.dim['d']

    # set initial conditions
    x0 = ca.vertcat(0,0,-ca.pi,0)
    u0 = 0.1
    w0 = ca.DM(n_w,1)
    d0 = ca.DM(n_d,1)


    ### CREATE MPC -----------------------------------------------------------------------------

    # upper level cost
    Q_true = ca.diag(ca.vertcat(100,1,100,1))
    R_true = 1e-6

    # mpc horizon
    N = 11

    # constraints are simple bounds on state and input
    x_max = ca.vertcat(5,5,ca.inf,ca.inf)
    x_min = -x_max
    u_max = 4
    u_min = -u_max

    # parameter = terminal state cost and input cost
    c_q = ca.SX.sym('c_q',int(n_x*(n_x+1)/2),1)
    c_r = ca.SX.sym('c_r',1,1)

    # stage cost (state)
    Qx = [Q_true] * (N-1)

    # stage cost (input)
    Ru = c_r**2 + 1e-6

    # create parameter
    p = ca.vcat([c_q,c_r])
    pf = theta

    # MPC terminal cost
    Qn = utils.param2terminalCost(c_q) + 0.01*ca.SX.eye(n_x)

    # append to Qx
    Qx.append(Qn)

    # slack penalties
    c_lin = 15
    c_quad = 5

    # add to mpc dictionary
    cost = {'Qx': Qx, 'Ru':Ru}

    # turn bounds into polyhedral constraints
    Hx,hx,Hu,hu = utils.bound2poly(x_max,x_min,u_max,u_min)

    # add to mpc dictionary
    cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu}

    # create QP ingredients
    ing = Ingredients(N=N,dynamics=dyn,cost=cost,constraints=cst)

    # create MPC
    MPC = QP(ingredients=ing,p=p,pf=pf)


    ### UPPER LEVEL -----------------------------------------------------------

    # upper-level horizon
    T = 170

    # create upper level
    UL = UpperLevel(p=p,pf=pf,horizon=T,mpc=MPC)

    # extract linearized dynamics at the origin
    A = dyn.A_nom(ca.DM(n_x,1),ca.DM(n_u,1),ca.DM(n_d,1))
    B = dyn.B_nom(ca.DM(n_x,1),ca.DM(n_u,1),ca.DM(n_d,1))

    # compute terminal cost initialization
    p_init = ca.vertcat(utils.dare2param(A,B,Q_true,R_true),1e-3)

    # extract closed-loop variables for upper level
    x_cl = ca.vec(UL.param['x_cl'])
    u_cl = ca.vec(UL.param['u_cl'])

    track_cost, cst_viol_l1, cst_viol_l2 = utils.quadCostAndBounds(Q_true,R_true,x_cl,u_cl,x_max,x_min)

    # put together
    cost = track_cost

    # create upper-level constraints
    Hx,hx,_,_ = utils.bound2poly(x_max,x_min,u_max,u_min,T+1)
    _,_,Hu,hu = utils.bound2poly(x_max,x_min,u_max,u_min,T)
    cst_viol = ca.vcat([Hx@ca.vec(x_cl)-hx,Hu@ca.vec(u_cl)-hu])

    # store in upper-level
    UL.set_cost(cost,track_cost,cst_viol)

    # create algorithm
    p = UL.param['p']
    J_p = UL.param['J_p']
    k = UL.param['k']

    # hyperparameters
    rho = 0.0001
    eta = 0.51

    # create GD update rule
    p_next = p -(rho*ca.log(k+2)/(k+2)**eta)*J_p

    # create update function
    UL.set_alg(p_next)

    scenario = Scenario(dyn,MPC,UL)

    # initialize
    scenario.set_init({'theta':d0,'p':p_init,'pf':ca.DM(n_d,1),'x': x0,'u': u0, 'w': w0, 'd': d0})

    return scenario

scenario = get_scenario()

p,pf,w,d,theta,y,x = scenario._get_init_parameters()

qp = scenario.qp

n = scenario.dim

n_x = n['x']

# get number of models that need to be tested
n_models = 10

# choose whether or not to compile
compilation_options = {}

# extract dynamics and linearization
# A = scenario.dyn.A_nom.map(n_models,[True,True,False],[False],compilation_options)
# B = scenario.dyn.B_nom.map(n_models,[True,True,False],[False],compilation_options)
A = convert(scenario.dyn.A_nom,compile=True)
B = convert(scenario.dyn.B_nom,compile=True)
f = scenario.dyn.f

# vectorize A and B

# create simVar for current simulation
S = simVar(n)

# set initial condition
S.setState(0,x)

# extract parameter indexing
idx_qp = scenario.upper_level.idx['qp']
idx_jac = convert(scenario.upper_level.idx['jac'],compile=True)

# get qp solver
solver = qp.solve

# initialize Jacobians
j_x_p = jnp.zeros((n['x'],n['p'],n_models))
j_y_p = jnp.zeros((n['y'],n['p'],n_models))
# S.setJx(0,j_x_p)

y_all = None
lam = None
mu = None

# get list of inputs to dynamics and to nominal dynamics
var_in_fixed = {'d':d} if d is not None else {}
var_in_nom_fixed = {'theta':theta} if theta is not None else {}

# simulation loop
for t in range(n['T']):
    
    # replace first entry of state with current state
    y_lin = y

    # parameter to pass to the QP
    p_t = idx_qp(x,y_lin,p,pf,t)

    # solve QP and get solution
    lam,mu,y_all = solver(p_t,y_all,lam,mu)

    # get optimization variable without slacks
    y = y_all[qp.idx['out']['y']]

    # get first input entry
    u = y_all[qp.idx['out']['u0']]

    # store optimization variables
    # S.setOptVar(t,lam,mu,y,p_t)

    # store input
    # S.setInput(t,u)

    # get current state and input
    current_var = {'x':x,'u':u}
    
    # update variables
    var_in = var_in_fixed | current_var
    var_in_nom = var_in_nom_fixed | current_var

    # check if noise is present
    if w is not None:
        var_in['w'] = w[t]

    # get conservative jacobian of optimal solution of QP with respect to parameter
    # vector p.
    j_qp_p = qp.J_y_p(np.array(lam),np.array(mu),np.array(p_t))@idx_jac(j_x_p.reshape((n['x'],n['p']*n_models)),j_y_p.reshape((n['y'],n['p']*n_models)),t,multiplier=n_models)

    # select entries associated to y
    j_y_p = j_qp_p[qp.idx['out']['y'],:]

    # select rows corresponding to first input u0
    j_u0_p = j_qp_p[qp.idx['out']['u0'],:]

    # propagate jacobian of closed loop state x
    j_x_p = np.einsum('mnr,ndr->mdr',
                      np.array(A.call(var_in_nom)['A']).reshape((n['x'],n['x'],n_models)),
                      j_x_p) \
            + np.einsum('ijk,ljk->ilk',
                        np.array(B.call(var_in_nom)['B']).reshape((n['x'],n['u'],n_models)),
                        np.array(j_u0_p).reshape((n['u'],n['p'],n_models)))
    # j_x_p = np.einsum('mnr,ndr->mdr',
    #                   np.array(A.call(var_in_nom)['A']).reshape((n['x'],n['x'],n_models)),
    #                   j_x_p) \
    #         + np.einsum('ijk,ljk->ilk',
    #                     np.array(B.call(var_in_nom)['B']).reshape((n['x'],n['u'],n_models)),
    #                     np.array(j_u0_p).reshape((n['u'],n['p'],n_models)))

    # store conservative jacobians of state and input
    # S.setJx(t+1,j_x_p)
    # S.setJu(t,j_u0_p)
    # S.setJy(t,j_y_p)

    # get next state
    x = f.call(var_in)['x_next']

    # store next state
    # S.setState(t+1,x)

print('Done')