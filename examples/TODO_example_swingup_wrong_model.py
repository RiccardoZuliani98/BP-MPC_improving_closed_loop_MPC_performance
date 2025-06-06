from BPMPC.scenario import scenario
import BPMPC.utils as utils
import BPMPC.tests as tests
import Examples.cart_pend as cart_pend
from casadi import *
from BPMPC.plotter import plotter

# cleanup jit files
utils.cleanup()

# decide what to compile
compile_dynamics = True
compile_qp_sparse = True
compile_qp_dense = True
compile_jac = False

# create model
mod = scenario()

### CREATE DYNAMICS ------------------------------------------------------------------------

# create dictionary with all variables
dyn = cart_pend.dynamics(dt=0.015)

# store in model
mod.makeDynamics(dyn,compile=compile_dynamics)

# extract dimensions for simplicity
n = mod.dim

# set initial conditions
x0 = vertcat(0,0,-pi,0)
u0 = 0.1
mod.setInit({'x': x0,'u':u0})


### CREATE MPC -----------------------------------------------------------------------------

# upper level cost
Q_true = diag(vertcat(100,1,100,1))
R_true = 1e-6

# mpc horizon
N = 11

# constraints are simple bounds on state and input
x_max = vertcat(5,5,inf,inf)
x_min = -x_max
u_max = 4
u_min = -u_max

# parameter = terminal state cost and input cost
c_q = SX.sym('c_q',int(n['x']*(n['x']+1)/2),1)
c_r = SX.sym('c_r',1,1)

# stage cost (state)
Qx = kron(SX.eye(N-1),Q_true)

# stage cost (input)
Ru = kron(SX.eye(N),c_r**2)#+1e-6)

# create parameter
p = vcat([c_q,c_r])

# MPC terminal cost
Qn = utils.param2terminalCost(c_q)# + 0.01*SX.eye(n['x'])

# slack penalties
c_lin = 15
c_quad = 5

# add to mpc dictionary
mpc_cost = {'Qx':Qx, 'Qn':Qn, 'Ru':Ru}

# turn bounds into polyhedral constraints
Hx,hx,Hu,hu = utils.bound2poly(x_max,x_min,u_max,u_min,N)

# add to mpc dictionary
mpc_cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu}

# extract linearized dynamics at the origin
A = mod.dyn.A_nom(DM(n['x'],1),DM(n['u'],1))
B = mod.dyn.B_nom(DM(n['x'],1),DM(n['u'],1))

# use A and B as model in the MPC
model = {'A':A,'B':B}

# add to model
mod.makeMPC(N=N,cost=mpc_cost,cst=mpc_cst,model=model,p=p,options={'jac_tol':8,'solver':'qpoases','slack':True,'compile_jac':compile_jac,'compile_qp_sparse':compile_qp_sparse})


### UPPER LEVEL -----------------------------------------------------------

# upper-level horizon
T = 170

# create upper level
mod.makeUpperLevel(T=T)

# compute terminal cost initialization
p_init = vertcat(utils.dare2param(A,B,Q_true,R_true),1e-3)
mod.setInit({'p':p_init})

# extract closed-loop variables for upper level
params = mod.param
x_cl = vec(params['x_cl'])
u_cl = vec(params['u_cl'])

track_cost, cst_viol_l1, cst_viol_l2 = utils.quadCostAndBounds(Q_true,R_true,x_cl,u_cl,x_max,x_min)

# put together
cost = track_cost

# create upper-level constraints
Hx,hx,_,_ = utils.bound2poly(x_max,x_min,u_max,u_min,T+1)
_,_,Hu,hu = utils.bound2poly(x_max,x_min,u_max,u_min,T)
cst = vcat([Hx@vec(x_cl)-hx,Hu@vec(u_cl)-hu])

# store in upper-level
mod.setUpperLevelCost(cost,track_cost,cst)

# create algorithm
p = params['p']
Jp = params['Jp']
k = params['k']

# hyperparameters
rho = 0.0001
eta = 0.51

# create GD update rule
# p_next = p -(rho*log(k+1)/(k+1)**eta)*if_else(norm_2(Jp)>50000,Jp*50000/norm_2(Jp),Jp)
p_next = p -(rho*log(k+1)/(k+1)**eta)*Jp

# create update function
mod.setUpperLevelAlg(p_next)

# test derivatives
# out = tests.derivatives(mod)

# simulate with initial parameter
S,qp_data_sparse,_ = mod.simulate()

# create plot but do not show
plotter.plotTrajectory(S,options={'x':[0,1,2,3],'x_legend':['Position untrained','Velocity untrained','Angle untrained','Angular velocity untrained'],'u':[0],'u_legend':['Force untrained'],'color':'blue'},show=False)

# test closed loop
SIM,time_sparse,p_final = mod.closedLoop(options={'max_k':50})

# create plots
plotter.plotTrajectory(SIM[-1],options={'x':[0,1,2,3],'x_legend':['Position tuned','Velocity tuned','Angle tuned','Angular velocity tuned'],'u':[0],'u_legend':['Force tuned'],'color':'red'},show=False)

# create nonlinear solver
NLP = mod.makeTrajectoryOpt()

# create warm start trajectories
x_warm = SIM[-1].x_mat
u_warm = SIM[-1].u_mat

# solve
nlp_out,nlp_solved = NLP(x0,x_warm,u_warm)

# plot best solution
plotter.plotTrajectory(nlp_out,options={'x':[0,1,2,3],'x_legend':['Position best','Velocity best','Angle best','Angular velocity best'],'u':[0],'u_legend':['Force best'],'color':'orange'},show=True)

# after training, test speed of dense and sparse qp formulations
# start with sparse
SIM_sparse,time_sparse,p_final = mod.closedLoop(init={'p':p_final},options={'max_k':30,'mode':'Simulate'})

# printout
max_time = np.max(time_sparse['qp'])
mean_time = np.mean(time_sparse['qp'])
print(f'Max qp time (sparse): {max_time}, mean qp time (sparse): {mean_time}')

# create sparse QP
mod.makeDenseQP(p_final,solver='daqp')
SIM_dense,time_dense,p_final = mod.closedLoop(init={'p':p_final},options={'max_k':30,'mode':'dense'})

# printout
max_time = np.max(time_dense['qp'])
mean_time = np.mean(DM(time_dense['qp']))
print(f'Max qp time (sparse): {max_time}, mean qp time (sparse): {mean_time}')