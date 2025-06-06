import sys
import os

# add root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scenario import Scenario
from src.dynamics import Dynamics
from src.qp import QP
from src.Ingredients import Ingredients
import src.utils as utils
# import tests.tests as tests
import dynamics.cart_pend_theta as cart_pend
import casadi as ca
from src.plotter import Plotter
from src.upper_level import UpperLevel
import numpy as np


# cleanup jit files
utils.cleanup()

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

# test derivatives
# # out = tests.derivatives(mod)


### CREATE SCENARIO -----------------------------------------------------------

scenario = Scenario(dyn,MPC,UL)

# initialize
scenario.set_init({'p':p_init,'pf':ca.DM(n_d,1),'x': x0,'u': u0, 'w': w0, 'd': d0})

# simulate with initial parameter
S,qp_data_sparse,_ = scenario.simulate()

# create plot but do not show
Plotter.plotTrajectory(S,options={'x':[0,1,2,3],'x_legend':['Position untrained','Velocity untrained','Angle untrained','Angular velocity untrained'],'u':[0],'u_legend':['Force untrained'],'color':'blue'},show=False)

# test closed loop
SIM,_,p_best = scenario.closed_loop(options={'max_k':5})

# get last value of p
p_final = SIM[-1].p

# create plots
Plotter.plotTrajectory(SIM[-1],options={'x':[0,1,2,3],'x_legend':['Position tuned','Velocity tuned','Angle tuned','Angular velocity tuned'],'u':[0],'u_legend':['Force tuned'],'color':'red'},show=False)

# create nonlinear solver
NLP = scenario.make_trajectory_opt(theta=ca.DM(n_d,1))

# create warm start trajectories
x_warm = SIM[-1].x_mat
u_warm = SIM[-1].u_mat

# solve
nlp_out,nlp_solved = NLP(x0,x_warm,u_warm)
print('NLP solved correctly') if nlp_solved else print('NLP failed')

# plot best solution
Plotter.plotTrajectory(nlp_out,options={'x':[0,1,2,3],'x_legend':['Position best','Velocity best','Angle best','Angular velocity best'],'u':[0],'u_legend':['Force best'],'color':'orange'},show=True)

# after training, test speed of dense and sparse qp formulations
# start with sparse
SIM_sparse,time_sparse,p_final = scenario.closed_loop(init={'p':p_final},options={'max_k':3,'mode':'simulate'})

# printout
max_time = np.max(time_sparse['qp'])
mean_time = np.mean(time_sparse['qp'])
print(f'Max qp time (sparse): {max_time}, mean qp time (sparse): {mean_time}')

# create sparse QP
MPC.make_dense_qp(p=p_final)
scenario.update(qp=MPC)
SIM_dense,time_dense,p_final = scenario.closed_loop(init={'p':p_final},options={'max_k':3,'mode':'dense'})

# printout
max_time = np.max(time_dense['qp'])
mean_time = np.mean(ca.DM(time_dense['qp']))
print(f'Max qp time (sparse): {max_time}, mean qp time (sparse): {mean_time}')