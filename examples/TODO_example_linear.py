from BPMPC.scenario import scenario
import BPMPC.utils as utils
import BPMPC.tests as tests
from casadi import *

# create model
mod = scenario()

### CREATE DYNAMICS ------------------------------------------------------------------------

# define symbolic variables
x = SX.sym('x0',1,1)
u = SX.sym('u0',1,1)

# Construct a CasADi function for the ODE right-hand side
A = 0.8
B = 0.1
c = 0.01
x_dot = A*x + B*u + 0.001*x**(1.2) + c

# compute next state symbolically
x_next = x_dot

# create model dynamics
dyn = {'x':x, 'u':u, 'x_next':x_next, 'x_dot':x_dot}

# set dynamics
mod.makeDynamics(dyn)

# pass custom initialization
mod.setInit({'x':100,'f':1, 'u':0.1}) # f should be diregarded
# mod.setInit({'x':100,'w':1}) # this should fail

# # if d and f have been set in the initialization, there is an error
# if (mod.dyn.init['d'] is not None) or (mod.dyn.init['w'] is not None) or ('f' in mod.dyn.init) or (mod.dyn.init['x']!=100):
#     raise Exception('There is an error in your dynamics initialization routine. Please check the setInit method in the dynamics class.')

# # check if model was recognized to be affine
# if not mod.dyn.type == 'affine':
#     raise Exception('There is an error in how you check if model is affine.')

# # check if nominal parameters match full parameters
# if not mod.dyn.param == mod.dyn.param_nominal:
#     raise Exception('There is an error in how you set the nominal parameters in the dynamics.')

# # check that param in mod are equal to param in dyn
# if not mod.param == mod.dyn.param:
#     raise Exception('There is an error in how you set the parameters in the model.')

# # check dimensions
# if mod.dim['x'] != 1 or mod.dim['u'] != 1:
#     raise Exception('There is an error in how you set the dimensions in the dynamics.')

# # check if dynamics are correct
# x_test = DM.rand(1,1)
# u_test = DM.rand(1,1)
# x_dot_test = A*x_test + B*u_test + c
# if not mod.dyn.fc(x_test,u_test) == x_dot_test:
#     raise Exception('There is an error in how you set the continuous dynamics in the model.')
# if not mod.dyn.f(x_test,u_test) == x_dot_test:
#     raise Exception('There is an error in how you set the discrete dynamics in the model.')
# if not mod.dyn.fc_nom(x_test,u_test) == x_dot_test:
#     raise Exception('There is an error in how you set the continuous nominal dynamics in the model.')
# if not mod.dyn.f_nom(x_test,u_test) == x_dot_test:
#     raise Exception('There is an error in how you set the discrete nominal dynamics in the model.')

# # check if linearizations are correct
# if not mod.dyn.A(x_test,u_test) == A:
#     raise Exception('There is an error in how you set the A matrix in the model.')
# if not mod.dyn.B(x_test,u_test) == B:
#     raise Exception('There is an error in how you set the B matrix in the model.')
# if not mod.dyn.A_nom(x_test,u_test) == A:
#     raise Exception('There is an error in how you set the nominal A matrix in the model.')
# if not mod.dyn.B_nom(x_test,u_test) == B:
#     raise Exception('There is an error in how you set the nominal B matrix in the model.')


### CREATE MPC -----------------------------------------------------------------------------

# mpc horizon
N = 20
 
# constraints are simple bounds on state and input
x_max = 100
x_min = -x_max
u_max = 100
u_min = -u_max

# stage cost (state)
Qx = SX.eye(N-1)

# stage cost (input)
p = SX.sym('p',1,1)
Ru = SX.eye(N)*p+0.01

# MPC terminal cost
Qn = 1

# add to mpc dictionary
cost = {'Qx':Qx, 'Qn':Qn, 'Ru':Ru, 's_lin':10}
# cost = {'Qx':Qx, 'Qn':Qn, 'Ru':Ru}

# turn bounds into polyhedral constraints
Hx,hx,Hu,hu = utils.bound2poly(x_max,x_min,u_max,u_min,N)

# add to mpc dictionary
cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu}

# create MPC
mod.makeMPC(N=N,cost=cost,cst=cst,p=p,options={'solver':'qpoases'})

# choose upper level horizon
T = 50
n_x = 1

# create upper level
mod.makeUpperLevel(T=T)

# extract closed-loop variables for upper level
vars = mod.upperLevel.param
x_cl = vec(vars['x_cl'])
u_cl = vec(vars['u_cl'])
y_cl = vars['y_cl']
p_cl = vars['p']

# stack all constraints
x_max_stack = repmat(x_max,T+1,1)
x_min_stack = repmat(x_min,T+1,1)

# closed-loop cost
track_cost = x_cl.T@SX.eye(T+1)@x_cl + 0.01*u_cl.T@SX.eye(T)@u_cl
cst_viol_l1 = SX.ones(1,x_cl.shape[0])@fmax(x_cl-SX(x_max_stack),fmax(SX(x_min_stack)-x_cl,SX((T+1)*n_x,1)))

# put together
cost = track_cost + 100*cst_viol_l1 + y_cl[0,0]**2 + p

# create upper-level cost
mod.setUpperLevelCost(cost)

# initialize parameter
mod.setInit({'p':0.6})
# mod.setInit({'y_lin':repmat(vertcat(0,0),N,1)})

S, out_dict, _ = mod.simulate()

# get closed-loop cost
J = mod.upperLevel.cost(S)

# get jacobian of closed-loop cost function
J = mod.upperLevel.J_cost(S)

tests.derivatives(mod,epsilon=1e-5)