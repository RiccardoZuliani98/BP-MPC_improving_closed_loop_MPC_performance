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
dyn = cart_pend.dynamics()

# store in model
mod.makeDynamics(dyn)

# extract dynamics for simplicity
n = mod.dim

# # check if model was recognized to be nonlinear
# if not mod.dyn.type == 'nonlinear':
#     raise Exception('There is an error in how you check if model is affine.')

# # check if nominal parameters match full parameters (they should not)
# if mod.dyn.param == mod.dyn.param_nominal:
#     raise Exception('There is an error in how you set the nominal parameters in the dynamics.')

# # check that param in mod are equal to param in dyn
# if not mod.param == mod.dyn.param:
#     raise Exception('There is an error in how you set the parameters in the model.')

# # check dimensions
# if mod.dim['x'] != n['x'] or mod.dim['u'] != n['u']:
#     raise Exception('There is an error in how you set the dimensions in the dynamics.')

# # check if dynamics are correct
# x_test = DM.rand(n['x'],1)
# u_test = DM.rand(n['u'],1)
# d_test = DM.rand(n['d'],1)
# w_test = DM.rand(n['w'],1)
# d_nom = DM(n['d'],1)
# w_nom = DM(n['w'],1)

# if mmax(fabs(mod.dyn.fc(x_test,u_test,d_test,w_test) - fc_temp(x_test,u_test,d_test))):
#     raise Exception('There is an error in how you set the continuous dynamics in the model.')
# if mmax(fabs(mod.dyn.f(x_test,u_test,d_test,w_test) - f_temp(x_test,u_test,d_test,w_test))):
#     raise Exception('There is an error in how you set the discrete dynamics in the model.')
# if mmax(fabs(mod.dyn.fc_nom(x_test,u_test) - fc_temp(x_test,u_test,d_nom))):
#     raise Exception('There is an error in how you set the continuous nominal dynamics in the model.')
# if mmax(fabs(mod.dyn.f_nom(x_test,u_test) - f_temp(x_test,u_test,d_nom,w_nom))):
#     raise Exception('There is an error in how you set the discrete nominal dynamics in the model.')

# # compute linearizations
# A = Function('A',[x,u,d,w],[jacobian(x_next,x)])
# B = Function('B',[x,u,d,w],[jacobian(x_next,u)])

# # check if linearizations are correct
# if mmax(fabs(mod.dyn.A(x_test,u_test,d_test,w_test) - A(x_test,u_test,d_test,w_test))):
#     raise Exception('There is an error in how you set the A matrix in the model.')
# if mmax(fabs(mod.dyn.B(x_test,u_test,d_test,w_test) - B(x_test,u_test,d_test,w_test))):
#     raise Exception('There is an error in how you set the B matrix in the model.')
# if mmax(fabs(mod.dyn.A_nom(x_test,u_test) - A(x_test,u_test,d_nom,w_nom))):
#     raise Exception('There is an error in how you set the nominal A matrix in the model.')
# if mmax(fabs(mod.dyn.B_nom(x_test,u_test) - B(x_test,u_test,d_nom,w_nom))):
#     raise Exception('There is an error in how you set the nominal B matrix in the model.')


# upper level cost
Q_true = 10*diag(vertcat(1,0.01,1,0.01))
R_true = 10*1e-2

# mpc horizon
N = 6

# constraints are simple bounds on state and input
x_max = vertcat(0.2,1.5,inf,inf)
x_min = -x_max
u_max = 3.85
u_min = -u_max

# get number of slack variables
n_eps = 2*N*int(sum1(x_max!=inf))

# first parameter => terminal state cost
c_qn = SX.sym('c_q',int(n['x']*(n['x']+1)/2),1)
# c_qn = SX.sym('c_qn',n['x'],1)

# second parameter => input cost
c_ru = SX.sym('c_ru',1,1)

# third parameter => constraint tightenings
c_eta = SX.sym('c_eta',n_eps,1)

# stage cost (state)
Qx = kron(SX.eye(N-1),Q_true)

# stage cost (input)
# Ru = kron(SX.eye(N),R_true + c_ru**2)
Ru = kron(SX.eye(N),R_true)

# MPC terminal cost
Qn = utils.param2terminalCost(c_qn) + SX.eye(n['x'])
# Qn = diag(c_qn**2) + SX.eye(n['x'])

# penalties on slack variables
s_lin = 10*45      # linear penalty
s_quad = 10*45     # quadratic penalty

# add to mpc dictionary
mpc_cost = {'Qx':Qx, 'Qn':Qn, 'Ru':Ru, 's_lin':s_lin, 's_quad':s_quad}

# turn bounds into polyhedral constraints
Hx,hx,Hu,hu = utils.bound2poly(x_max,x_min,u_max,u_min,N)

# add constraint tightenings
# hx = hx - c_eta**2

# add to mpc dictionary
mpc_cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu}

# construct parameter vector
# p = vertcat(c_qn,c_ru,c_eta)
p = c_qn

# add to model
mod.makeMPC(N=N,cost=mpc_cost,cst=mpc_cst,p=p,options={'jac_tol':8,'solver':'daqp'})


### PARAMETER INITIALIZATION -----------------------------------------------------------

# extract linearized dynamics at the origin
A = mod.dyn.A_nom(DM(n['x'],1),DM(n['u'],1))
B = mod.dyn.B_nom(DM(n['x'],1),DM(n['u'],1))

# compute terminal cost initialization
c_qn_init = utils.dare2param(A,B,Q_true,R_true)
# c_qn_init = sqrt(diag(Q_true))

# initial input cost
c_ru_init = 0

# initial tightenings
c_eta_init = 1e-5*DM.ones(n_eps,1)

# construct full initialization
# p1_stack = vertcat(c_qn_init,c_ru_init,c_eta_init)
p1_stack = vertcat(c_qn_init)

# initial condition
x0 = vertcat(0,0,-pi,0)

# initial linearization trajectory
y0_stack = vertcat(repmat(x0,N,1),1e-5*DM.ones(N*n['u'],1))
# y0_stack = 'optimal' #TODO

# upper-level horizon
T = 100

# create upper level
mod.makeUpperLevel(T=T)

# extract closed-loop variables for upper level
params = mod.param
x_cl = vec(params['x_cl'])
u_cl = vec(params['u_cl'])

track_cost, cst_viol_l1, cst_viol_l2 = utils.quadCostAndBounds(Q_true,R_true,x_cl,u_cl,x_max,x_min)

# put together
cost = track_cost + 100*cst_viol_l1

# create upper-level cost
mod.setUpperLevelCost(cost,track_cost,cst_viol_l1)

# run QP once to get better initialization
# lam_0,mu_0,y_all_0 = mod.QP.solve(mod.UpperLevel.idx['qp'](x0,y0_stack,p1_stack,DM(0,0),0))
# y0_x = y_all_0[mod.QP.idx['out']['x'][:-mod.dim['x']]]
# y0_u = y_all_0[mod.QP.idx['out']['u']]
# y0 = vertcat(x0,y0_x,y0_u)

# initial conditions
mod.setInit({'x': x0,'u':0,'w':DM(n['w'],1),'d':DM(n['d'],1),'p':p1_stack})
# mod.setInit({'x': x0,'p':p1_stack,'y_lin':y0})
# init = {'x': x0,'u':1e-5,'p':p1_stack}

# S, out_dict = mod.simulate()

# plotter.plotTrajectory(S,options={'x':[0,1,2,3],'x_legend':['Position 1','Velocity 1','Angle 1','Angular velocity 1'],'u':[0],'u_legend':['Force 1'],'color':'blue'},show=False)

# # mod.setInit({'x': x0,'p':0.1*p1_stack})
# init = {'x': x0,'u':1e-5,'p':0.1*p1_stack}

# S, out_dict = mod.simulate(init)

# plotter.plotTrajectory(S,options={'x':[0,1,2,3],'x_legend':['Position 2','Velocity 2','Angle 2','Angular velocity 2'],'u':[0],'u_legend':['Force 2'],'color':'orange'},show=True)

# # get closed-loop cost
# J = mod.upperLevel.cost(S)

# # get jacobian of closed-loop cost function
# J = mod.upperLevel.J_cost(S)

# with np.printoptions(precision=6,suppress=True,threshold=1e20,linewidth=1e20):
    # print(np.array(S.Jx))

# test derivatives
# out = tests.derivatives(mod,epsilon=1e-6,debug_qp=False)

# simOut,qp_data = mod.simulate(options={'debugQP':True,'epsilon':1e-8,'roundoff_QP':12})
# out = {'simOut':simOut,'qp_data':qp_data}

# J_qp_error_norm = []
# e_list = []

# for k in range(len(out['qp_data']['QP_debug'])):
#     # error = np.linalg.norm(out['qp_data']['QP_debug'][k]['J_QP']-out['qp_data']['QP_debug'][k]['J_num'],axis=0)
#     # rel_error = np.divide(error,np.linalg.norm(out['qp_data']['QP_debug'][k]['J_num'],axis=0))
#     # J_qp_error_norm.append(rel_error)
#     e_list.append(out['qp_data']['QP_debug'][k]['error_rel'])

# loop to test QP derivative accuracy
# worst_indx = None
# worst_error = inf

# for indx in range(len(out['qp_data']['QP_debug'])):

#     vec1 = np.array(out['qp_data']['QP_debug'][indx]['J_QP'])
#     vec2 = np.array(out['qp_data']['QP_debug'][indx]['J_num'])

#     all_errors = [vec1[:,k]-vec2[:,k] for k in range(vec1.shape[1])]
#     col = np.argmax([np.linalg.norm(err) for err in all_errors])

#     if np.linalg.norm(all_errors[col]) < worst_error:
#         worst_indx = indx
#         worst_error = np.linalg.norm(all_errors[col])
#         worst_col = col


# vec1 = np.array(out['qp_data']['QP_debug'][worst_indx]['J_QP'])
# vec2 = np.array(out['qp_data']['QP_debug'][worst_indx]['J_num'])

# with np.printoptions(precision=20,suppress=True,threshold=1e20,linewidth=1e20):
#     print(np.vstack((vec1[:,worst_col]-vec2[:,worst_col],vec2[:,worst_col])).T)

# extract symbolic variables from upper level

# create algorithm
p = params['p']
Jp = params['Jp']
k = params['k']

# hyperparameters
rho = 0.0001
eta = 0.6

# create GD update rule
p_next = p - (rho*log(k+1)/(k+1)**eta)*Jp

# create update function
mod.setUpperLevelAlg(p_next)

# test closed loop
# init_vec = {'x':[DM(n['x'],1),DM(n['x'],1),DM(n['x'],1)],'u':[1,0,0],'w':DM(n['w'],1),'d':DM(n['d'],1),'p':p1_stack}
mod.closedLoop()