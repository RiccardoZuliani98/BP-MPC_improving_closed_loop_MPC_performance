from BPMPC.scenario import scenario
import BPMPC.utils as utils
import BPMPC.tests as tests
import Examples.cart_pend as cart_pend
from casadi import *
from BPMPC.plotter import plotter

# create model
mod = scenario()

### CREATE DYNAMICS ------------------------------------------------------------------------

# create LTI dynamics

A=DM(
[[1, 0.05, -0.000710385, -1.15764e-05], 
 [00, 1, -0.0290474, -0.000710385], 
 [00, 00, 1.13959, 0.0522748], 
 [00, 00, 5.70783, 1.13959]])
B=DM([0.00433578, 0.173643, -0.0466539, -1.90766])

# create symbols
x = SX.sym('x',4)
u = SX.sym('u',1)

# create dynamics
x_dot = A@x + B@u
x_next = x_dot

# store in model
mod.makeDynamics({'x':x,'u':u,'x_dot':x_dot,'x_next':x_next})

# extract dimensions for simplicity
n = mod.dim

# set initial conditions
x0 = vertcat(-1,0,0,0)
u0 = 0.1
mod.setInit({'x': x0,'u':u0})


### CREATE MPC -----------------------------------------------------------------------------

# upper level cost
Q_true = diag(vertcat(1,0.01,1,0.01))
R_true = 0.1

# mpc horizon
N = 6

# constraints are simple bounds on state and input
x_max = vertcat(inf,inf,0.1,0.2)
x_min = -x_max
u_max = 0.75
u_min = -0.75

# parameter = terminal state cost
p = SX.sym('p',int(n['x']*(n['x']+1)/2),1)

# stage cost (state)
Qx = kron(SX.eye(N-1),Q_true)

# stage cost (input)
Ru = SX.eye(N)*R_true

# MPC terminal cost
Qn = utils.param2terminalCost(p) + SX.eye(n['x'])

# slack penalty
s_lin = 5
s_quad = 5

# add to mpc dictionary
mpc_cost = {'Qx':Qx, 'Qn':Qn, 'Ru':Ru, 's_lin':s_lin, 's_quad':s_quad}
# mpc_cost = {'Qx':Qx, 'Qn':Qn, 'Ru':Ru}

# turn bounds into polyhedral constraints
Hx,hx,Hu,hu = utils.bound2poly(x_max,x_min,u_max,u_min,N)

# add to mpc dictionary
mpc_cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu}

# add to model
# mod.makeMPC(N=N,cost=mpc_cost,cst=mpc_cst,model=model,p=p,options={'jac_tol':7,'solver':'qpoases'})
mod.makeMPC(N=N,cost=mpc_cost,cst=mpc_cst,p=p,options={'jac_tol':7,'solver':'qpoases'})
# mod.makeMPC(N=N,cost=mpc_cost,cst=mpc_cst,p=p,options={'jac_tol':7,'solver':'daqp'})


### UPPER LEVEL -----------------------------------------------------------

# upper-level horizon
T = 100

# create upper level
mod.makeUpperLevel(T=T)

# initial terminal cost
p_init = utils.dare2param(A,B,Q_true,R_true)

# full parameter initialization
mod.setInit({'p':p_init})

# extract closed-loop variables for upper level
params = mod.param
x_cl = vec(params['x_cl'])
u_cl = vec(params['u_cl'])

track_cost, cst_viol_l1, cst_viol_l2 = utils.quadCostAndBounds(Q_true,R_true,x_cl,u_cl,x_max,x_min)

# put together
cost = track_cost

# create upper-level cost
mod.setUpperLevelCost(cost,track_cost,cst_viol_l1)

# create algorithm
p = params['p']
Jp = params['Jp']
k = params['k']

# hyperparameters
rho = 0.0015
eta = 0.8

# create GD update rule
# p_next = p -(rho*log(k+1)/(k+1)**eta)*if_else(norm_2(Jp)>500,Jp*500/norm_2(Jp),Jp)
p_next = p -(rho*log(k+1)/(k+1)**eta)*Jp

# create update function
mod.setUpperLevelAlg(p_next)

# test derivatives
# out = tests.derivatives(mod)

# simulate with initial parameter
# S,qp_data_sparse,_ = mod.simulate()

# create plot but do not show
# plotter.plotTrajectory(S,options={'x':[0,1,2,3],'x_legend':['Position 1','Velocity 1','Angle 1','Angular velocity 1'],'u':[0],'u_legend':['Force 1'],'color':'blue'},show=False)

# test closed loop
SIM,time_sparse,p_final = mod.closedLoop(options={'max_k':30})

# printout
max_time = mmax(DM(time_sparse['qp']))
print(f'Max qp time (sparse): {max_time}')

# create plots
plotter.plotTrajectory(SIM[-1],options={'x':[0,1],'x_legend':['Position','Velocity'],'u':[0],'u_legend':['Force sparse'],'color':'red'},show=True)

# print parameter
print(p_final)

# compute best performance
