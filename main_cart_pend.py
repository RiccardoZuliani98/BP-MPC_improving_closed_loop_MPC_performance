from BPMPC.scenario import scenario
import BPMPC.utils as utils
import BPMPC.tests as tests
import Examples.cart_pend as cart_pend
from casadi import *
from BPMPC.plotter import plotter

# create model
mod = scenario()

### CREATE DYNAMICS ------------------------------------------------------------------------

# create dictionary with all variables
dyn = cart_pend.dynamics(dt=0.05)

# store in model
mod.makeDynamics(dyn)

# extract dimensions for simplicity
n = mod.dim

# set initial conditions
x0 = vertcat(-3,0,0,0)
u0 = 0.1
mod.setInit({'x': x0,'u':u0})


### CREATE MPC -----------------------------------------------------------------------------

# upper level cost
Q_true = diag(vertcat(1,0.01,1,0.01))
R_true = 0.1

# mpc horizon
N = 6

# constraints are simple bounds on state and input
x_max = vertcat(inf,0.6,0.1,0.6)
x_min = -x_max
u_max = 0.9
u_min = -u_max

# parameter = terminal state cost
p = SX.sym('c_q',int(n['x']*(n['x']+1)/2),1)

# stage cost (state)
Qx = Q_true

# stage cost (input)
Ru = R_true

# MPC terminal cost
Qn = utils.param2terminalCost(p) + SX.eye(n['x'])

# slack cost
s_lin = 100
s_quad = 100

# add to mpc dictionary
mpc_cost = {'Qx':Qx, 'Qn':Qn, 'Ru':Ru, 's_lin':s_lin, 's_quad':s_quad}

# turn bounds into polyhedral constraints
Hx,hx,Hu,hu = utils.bound2poly(x_max,x_min,u_max,u_min)

# add to mpc dictionary
mpc_cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu}

# MPC uses linearized model at the origin
A = mod.dyn.A_nom(DM(n['x'],1),DM(n['u'],1))
B = mod.dyn.B_nom(DM(n['x'],1),DM(n['u'],1))
model = {'A':A,'B':B,'x0':mod.param['x']}

# add to model
mod.makeMPC(N=N,cost=mpc_cost,cst=mpc_cst,model=model,p=p,options={'jac_tol':7,'solver':'daqp'})


### UPPER LEVEL -----------------------------------------------------------

# upper-level horizon
T = 120

# create upper level
mod.makeUpperLevel(T=T)

# compute terminal cost initialization
p_init = 0.1*DM.ones(mod.dim['p'],1)#utils.dare2param(A,B,Q_true,R_true)
mod.setInit({'p':p_init})

# extract closed-loop variables for upper level
params = mod.param
x_cl = vec(params['x_cl'])
u_cl = vec(params['u_cl'])

track_cost, cst_viol_l1, cst_viol_l2 = utils.quadCostAndBounds(Q_true,R_true,x_cl,u_cl,x_max,x_min)

# put together
cost = track_cost + 100*(cst_viol_l1 + cst_viol_l2)

# create upper-level cost
mod.setUpperLevelCost(cost,track_cost+100*cst_viol_l1,cst_viol_l1)

# create algorithm
p = params['p']
Jp = params['Jp']
k = params['k']

# hyperparameters
rho = 0.02
eta = 0.51

# create GD update rule
p_next = p -(rho*log(k+1)/(k+1)**eta)*if_else(norm_2(Jp)>100,Jp*100/norm_2(Jp),Jp)
# p_next = p -(rho*log(k+1)/(k+1)**eta)*Jp

# create update function
mod.setUpperLevelAlg(p_next)

# test derivatives
# out = tests.derivatives(mod)

# simulate with initial parameter
# S,qp_data_sparse,_ = mod.simulate(*mod.getInitParameters())

# create plot but do not show
# plotter.plotTrajectory(S,options={'x':[0,1,2,3],'x_legend':['Position 1','Velocity 1','Angle 1','Angular velocity 1'],'u':[0],'u_legend':['Force 1'],'color':'blue'},show=False)

# test closed loop
SIM,time_sparse,p_final = mod.closedLoop(options={'max_k':500})

# printout
max_time = mmax(DM(time_sparse['qp']))
print(f'Max qp time (sparse): {max_time}')

# create plots
# plotter.plotTrajectory(SIM[-1],options={'x':[0,1,2,3],'x_legend':['Position sparse','Velocity sparse','Angle sparse','Angular velocity sparse'],'u':[0],'u_legend':['Force sparse'],'color':'red'},show=False)

mod.makeDenseQP(p_final,solver='qpoases')
SIM,time_dense,p_final = mod.closedLoop(options={'max_k':25,'mode':'simulate'})

# printout
max_time = mmax(DM(time_dense['qp']))
print(f'Max qp time (sparse): {max_time}')

# plotter.plotTrajectory(SIM[-1],options={'x':[0,1,2,3],'x_legend':['Position dense','Velocity dense','Angle dense','Angular velocity dense'],'u':[0],'u_legend':['Force dense'],'color':'blue'},show=True)