import sys
import os

# add root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scenario import Scenario
from src.dynamics import Dynamics
from src.qp import QP
from src.ingredients import Ingredients
from utils.cleanup import cleanup
from utils.cost_utils import quad_cost_and_bounds,bound2poly,param2terminal_cost,dare2param
# import tests.tests as tests
import examples.dynamics.random_linear as random_linear
import casadi as ca
from src.plotter import Plotter
from src.upper_level import UpperLevel
import numpy as np
from utils.parameter_update import average_gradient_descent, robust_gradient_descent, gradient_descent
from utils.sys_id import rls

# cleanup jit files
cleanup()

# decide what to compile
COMPILE_DYNAMICS = False
COMPILE_QP_SPARSE = False
COMPILE_QP_DENSE = False
COMPILE_JAC = False

# horizons
UPPER_HORIZON = 20
MPC_HORIZON = 10
ITERATIONS = 100

# penalties on constraint violation (closed-loop)
L2_PENALTY = 10
L1_PENALTY = 15

# uncertainty on theta
THETA_UNCERTAINTY_RANGE = 1

# how spread out the initial condition is
X0_MAG = 2

# decide whether to include noise or not
NOISE = True
NOISE_MAG = 0.5


### CREATE DYNAMICS ------------------------------------------------------------------------

# create dictionary with parameters of cart pendulum
dyn_dict,true_theta = random_linear.dynamics(Ts=0.3,n_x=4,use_w=NOISE,pole_mag=[-2,1])
print(true_theta)

# model uncertainty parameter
theta = dyn_dict['theta']

# create dynamics object
dyn = Dynamics(dyn_dict,jit=COMPILE_DYNAMICS)

# get state and input dimensions
n_x, n_u = dyn.dim['x'], dyn.dim['u']

# set initial conditions
x0 = ca.DM.ones(n_x,1)#ca.DM( X0_MAG * (np.ones((n_x,1)) + 2*np.random.rand(n_x,1)) )
theta_uncertainty = THETA_UNCERTAINTY_RANGE*(np.ones(theta.shape)+2*np.random.rand(*theta.shape))
theta0 = ca.DM( np.multiply(theta_uncertainty,np.array(true_theta)) )
print(f'Initial condition: {x0}')
print(f'Initial parameter estimate: {theta0}')


# sample noise if requested
if NOISE:
    w0 = ca.horzsplit(NOISE_MAG*(2*np.random.rand(dyn.dim['w'],UPPER_HORIZON)-np.ones((dyn.dim['w'],UPPER_HORIZON))))

### CREATE MPC -----------------------------------------------------------------------------

# upper level cost
Q_true = 10*ca.DM.eye(n_x)
R_true = 1

# constraints are simple bounds on state and input
x_max = 5*ca.DM.ones(n_x,1)
x_min = -x_max
u_max = 1.5
u_min = -u_max

# parameter = terminal state cost and input cost
c_q = ca.SX.sym('c_q',int(n_x*(n_x+1)/2),1)
c_r = ca.SX.sym('c_r',1,1)

# stage cost (state)
Qx = [Q_true] * (MPC_HORIZON-1)

# stage cost (input)
Ru = c_r**2 + 1e-6

# create parameter
p = ca.vcat([c_q,c_r])
pf = theta

# MPC terminal cost
Qn = param2terminal_cost(c_q) + 0.01*ca.SX.eye(n_x)

# append to Qx
Qx.append(Qn)

# add to mpc dictionary
cost = {'Qx': Qx, 'Ru':Ru, 's_quad':5, 's_lin':5}
# cost = {'Qx': Qx, 'Ru':Ru}

# turn bounds into polyhedral constraints
Hx,hx,Hu,hu = bound2poly(x_max,x_min,u_max,u_min)

# add to mpc dictionary
cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu, 'Hx_e':ca.SX.eye(hx.shape[0])}
# cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu}

# create QP ingredients
ing = Ingredients(horizon=MPC_HORIZON,dynamics=dyn,cost=cost,constraints=cst)

# create options
qp_options = {'compile_qp_sparse':COMPILE_QP_SPARSE,
              'compile_qp_dense':COMPILE_QP_DENSE,
              'compile_jac':COMPILE_JAC,
              'solver':'daqp'}

# create MPC
mpc = QP(ingredients=ing,p=p,pf=pf,options=qp_options)


### UPPER LEVEL -----------------------------------------------------------

# create upper level
upper_level = UpperLevel(p=p,pf=pf,horizon=UPPER_HORIZON,mpc=mpc)

# extract linearized dynamics at the origin
A = dyn.A_nom(ca.DM(n_x,1),ca.DM(n_u,1),theta0)
B = dyn.B_nom(ca.DM(n_x,1),ca.DM(n_u,1),theta0)

# compute terminal cost initialization
p_init = ca.vertcat(ca.DM.ones(p.shape[0]-1,1)*1e-3,1)#ca.vertcat(dare2param(A,B,Q_true,R_true),1e-1)

# extract closed-loop variables for upper level
x_cl = ca.vec(upper_level.param['x_cl'])
u_cl = ca.vec(upper_level.param['u_cl'])

track_cost, cst_viol_l1, cst_viol_l2 = quad_cost_and_bounds(Q_true,R_true,x_cl,u_cl,x_max,x_min)

# put together
cost = track_cost + L2_PENALTY*cst_viol_l2 + L1_PENALTY*cst_viol_l1

# create upper-level constraints
Hx,hx,_,_ = bound2poly(x_max,x_min,u_max,u_min,UPPER_HORIZON+1)
_,_,Hu,hu = bound2poly(x_max,x_min,u_max,u_min,UPPER_HORIZON)
cst_viol = ca.vcat([Hx@ca.vec(x_cl)-hx,Hu@ca.vec(u_cl)-hu])

# store in upper-level
upper_level.set_cost(cost,track_cost,cst_viol)

# create algorithm
p = upper_level.param['p']
j_p = upper_level.param['J_p']
k = upper_level.param['k']

# create update function
parameter_update, parameter_init = gradient_descent(rho=0.0001,eta=0.8,log=True)

# create system identification
sys_id_update, sys_id_init, _ = rls(
    dynamics=dyn,
    horizon=UPPER_HORIZON,
    lam=0.1,
    theta0=theta0,
    jit=False,
    idx_pf=range(theta0.shape[0]))

# update upper-level algorithm
upper_level.set_alg(
    parameter_update=parameter_update,
    parameter_init=parameter_init,
    sys_id_update=sys_id_update,
    sys_id_init=sys_id_init)


# upper_level.set_alg(*average_gradient_descent(rho=0.0001,eta=0.51,log=True))
# upper_level.set_alg(*robust_gradient_descent(rho=0.0001,eta=0.51,n_models=len(theta0),n_p=p.shape[0],log=True,verbose=False))

# test derivatives
# # out = tests.derivatives(mod)


### CREATE SCENARIO -----------------------------------------------------------

scenario = Scenario(dyn,mpc,upper_level)

# initialize
init_dict = {'p':p_init,'pf':theta0,'x': x0,'theta':theta0}
if NOISE:
    init_dict['w'] = w0
scenario.set_init(init_dict)

# test closed loop
sim_list,_,p_best = scenario.closed_loop(options={'use_true_model':False,'max_k':ITERATIONS,'true_theta':np.array(true_theta)})

# retrieve thetas
estimation_error = [ca.norm_2(ca.fabs(elem.psi['theta']-true_theta)) for elem in sim_list]

print(estimation_error)

# get last value of p
# p_final = SIM[-1].p