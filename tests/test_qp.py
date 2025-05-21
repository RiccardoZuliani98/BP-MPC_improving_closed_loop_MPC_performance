import sys
import os
import casadi as ca
import numpy as np
import pytest
import daqp
import piqp
from ctypes import *

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sample_elements import sample_mpc
from utils.cost_utils import bound2poly,quad_cost_and_bounds
from src.dynamics import Dynamics
from src.upper_level import UpperLevel
from src.scenario import Scenario
from src.ingredients import Ingredients
from src.qp import QP

# combinations that need to be tested
CONFIGURATIONS = [{'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':True,'use_pf':True,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':True,'use_pf':True,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':False,'use_pf':True,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':False,'use_pf':True,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':True,'use_pf':False,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':True,'use_pf':False,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':False,'use_pf':False,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':False,'use_pf':False,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':True,'use_pf':True,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':True,'use_pf':True,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':False,'use_pf':True,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':False,'use_pf':True,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':True,'use_pf':False,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':True,'use_pf':False,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':False,'use_pf':False,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':False,'use_pf':False,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':True,'use_pf':True,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':True,'use_pf':True,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':False,'use_pf':True,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':False,'use_pf':True,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':True,'use_pf':False,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':True,'use_pf':False,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':False,'use_pf':False,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':False,'use_pf':False,'use_slack':True,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':True,'use_pf':True,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':True,'use_pf':True,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':False,'use_pf':True,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':False,'use_pf':True,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':True,'use_pf':False,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':True,'use_pf':False,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':False,'use_pf':False,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':False,'use_pf':False,'use_slack':False,'linearization':'trajectory'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':True,'use_pf':True,'use_slack':True,'linearization':'initial_state'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':False,'use_pf':True,'use_slack':True,'linearization':'initial_state'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':True,'use_pf':False,'use_slack':True,'linearization':'initial_state'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':False,'use_pf':False,'use_slack':True,'linearization':'initial_state'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':True,'use_pf':True,'use_slack':False,'linearization':'initial_state'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':False,'use_pf':True,'use_slack':False,'linearization':'initial_state'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':True,'use_pf':False,'use_slack':False,'linearization':'initial_state'},
                  {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':False,'use_pf':False,'use_slack':False,'linearization':'initial_state'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':True,'use_pf':True,'use_slack':True,'linearization':'initial_state'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':False,'use_pf':True,'use_slack':True,'linearization':'initial_state'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':True,'use_pf':False,'use_slack':True,'linearization':'initial_state'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':False,'use_pf':False,'use_slack':True,'linearization':'initial_state'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':True,'use_pf':True,'use_slack':False,'linearization':'initial_state'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':False,'use_pf':True,'use_slack':False,'linearization':'initial_state'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':True,'use_pf':False,'use_slack':False,'linearization':'initial_state'},
                  {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':False,'use_pf':False,'use_slack':False,'linearization':'initial_state'}]

def test_problem():

    # define symbolic variables
    x = ca.SX.sym('x0',1,1)
    u = ca.SX.sym('u0',1,1)

    # Construct a CasADi function for the ODE right-hand side
    A = 1
    B = 1

    # initial condition
    x0 = 1

    # compute next state symbolically
    x_next = A*x + B*u

    # create sceanario
    dynamics = Dynamics({'x':x, 'u':u, 'x_next':x_next})

    # create constraints
    x_max = 1000
    x_min = -x_max
    u_max = 1000
    u_min = -u_max

    # horizon of MPC
    N = 2

    # create parameter
    p = ca.SX.sym('p',12,1)
    pf = ca.SX.sym('pf',12,1)

    # create parameters for MPC
    p_qp = ca.SX.sym('p_qp',4,1)
    pf_qp = ca.SX.sym('pf_qp',4,1)

    # create reference
    x_ref = ca.vertsplit(pf_qp[:2])
    u_ref = ca.vertsplit(pf_qp[2:])

    # MPC costs
    Qx = ca.SX(1)
    Ru = ca.SX(2)

    # add to mpc dictionary
    mpc_cost = {'Qx':Qx, 'Ru':Ru, 'x_ref':x_ref, 'u_ref':u_ref}

    # turn bounds into polyhedral constraints
    Hx,hx,Hu,hu = bound2poly(x_max,x_min,u_max,u_min)

    # add to mpc dictionary
    mpc_cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu}
    
    # create ingredients
    ingredients = Ingredients(horizon=N,dynamics=dynamics,cost=mpc_cost,constraints=mpc_cst)

    # create mpc
    mpc = QP(ingredients,p=p_qp,pf=pf_qp)

    # create index that selects pf for QP
    def idx_pf(t):
        return np.hstack([t,t+1,6+t,7+t],dtype=int)

    # construct upper-level
    upper_level = UpperLevel(p=p,pf=pf,idx_p=idx_pf,idx_pf=idx_pf,horizon=5,mpc=mpc)

    # create state reference by sampling random inputs
    u_ref = ca.DM.rand(6,1)
    x_ref = [x0]
    for t in range(6):
        x_ref.append( A*x_ref[-1] + B*u_ref[t] )
    x_ref = ca.vcat(x_ref)

    # extract closed-loop variables for upper level
    params = upper_level.param
    x_cl = ca.vec(params['x_cl'])
    u_cl = ca.vec(params['u_cl'])

    # create upper-level cost
    cost,_,_ = quad_cost_and_bounds(ca.SX(1),ca.SX(1),x_cl,u_cl,x_ref=x_ref[:-1],u_ref=u_ref[:-1])
    upper_level.set_cost(cost)

    # create scenario
    scenario = Scenario(dynamics,mpc,upper_level)

    # set pf
    pf_init = ca.vertcat(x_ref[1:],u_ref)
    scenario.set_init({'pf':pf_init,'p':pf_init,'x':x0})

    # simulate
    sim,_,_ = scenario.simulate(options={'mode':'simulate'})

    # check if closed-loop trajectory is close to reference
    assert ca.norm_2(sim.x.T - x_ref[:-1]) <= 1e-10 and  ca.norm_2(sim.u.T - u_ref[:-1]) <= 1e-10, 'Closed-loop trajectory does not match reference.'
    

def single_test_dynamics(mpc,params,x_list,u_list,configuration):

    # construct parameter
    p = ca.vcat(list(params.values()))

    # run qp sparse
    sparse_ingredients = mpc.qp_sparse(p=p) # p_qp_full -> ingredients for sparse problem
    
    # get number of slacks
    n_eps = mpc.dim['eps'] if 'eps' in mpc.dim else 0

    # get equality constraints
    n_eq = mpc.dim['eq']
    F = sparse_ingredients['a'][-n_eq:,:-n_eps] if n_eps > 0 else sparse_ingredients['a'][-n_eq:,:]
    f = sparse_ingredients['uba'][-n_eq:]

    assert ca.norm_2(F@ca.vcat(x_list+u_list)-f) <= 1e-8, 'Dynamics are not correct for configuration' + str(configuration)

    # make qp dense
    # mpc.make_dense_qp()
    # TODO: write test for dense QP

    # solve QP
    lam,mu,y = mpc.solve(p)

    # create dual problem
    dual_ingredients = mpc.dual_sparse(p=p)

    # solve dual problem
    P = np.array(dual_ingredients['H'],dtype=np.float64)
    c = np.array(dual_ingredients['h'],dtype=np.float64).squeeze()
    x_lb = np.array(np.hstack((np.zeros(mpc.dim['in']),-np.inf*np.ones(mpc.dim['eq']))),dtype=np.float64).squeeze()

    solver = piqp.DenseSolver()
    solver.settings.verbose = False

    solver.setup(P=P,c=c,A=None,b=None,G=None,h=None,x_lb=x_lb,x_ub=None)

    status = solver.solve()

    if status == 1:

        x_star = solver.result.x
        assert np.linalg.norm(x_star-np.hstack((lam,mu))) / np.linalg.norm(x_star) <= 0.01
        
    else:
        print('Dual failed')

    # A = np.array(np.eye(mpc.dim['z'])[:mpc.dim['in'],:],dtype=np.float64)
    # bu = np.array(np.inf*np.ones(mpc.dim['in']),dtype=np.float64).squeeze()
    # x_star,_,flag,_ = piqp.solve(H,f,A,bu,bl,2*np.ones(mpc.dim['z'],dtype=c_int).squeeze())
    

def single_test_symbolic_order(mpc,configuration,dynamics,ingredients,horizon,p,pf):

    # extract variables
    p_t = mpc.param['p_t'] if 'p_t' in mpc.param else ca.SX(0,0)
    p_qp = mpc.param['p_qp']
    p_qp_full = mpc.param['p_qp_full']
    y = mpc.param['y']

    # get indexing
    idx = mpc._ingredients._idx

    # expected values
    p_t_expected = p if configuration['use_p'] else ca.SX(0,0)
    p_qp_expected = [dynamics.param['x']]
    p_qp_full_expected = [dynamics.param['x']]
    if configuration['nonlinear']:
        assert 'y_lin' in ingredients.param, 'Linearization trajectory not present in dynamics.'
        p_qp_expected.append(ingredients.param['y_lin'])
        p_qp_full_expected.append(ingredients.param['y_lin'])
    if configuration['use_p']:
        p_qp_expected.append(p)
        p_qp_full_expected.append(p)
    if configuration['use_pf'] or configuration['use_theta']:
        p_qp_full_expected.append(pf)
    p_qp_expected = ca.vcat(p_qp_expected)
    p_qp_full_expected = ca.vcat(p_qp_full_expected)

    # check that parameters are passed correctly
    assert str(p_t_expected)==str(p_t), 'p_t does not match for configuration: ' + str(configuration)
    assert str(p_qp_expected)==str(p_qp), 'p_qp does not match for configuration: ' + str(configuration)
    assert str(p_qp_full_expected)==str(p_qp_full), 'p_qp_full does not match for configuration: ' + str(configuration)

    # check that indexing is correct
    assert str(p_qp_full[idx['in']['x']]) == str(dynamics.param['x']), 'x in idx does not match.'
    if configuration['nonlinear']:
        assert str(p_qp_full[idx['in']['y_lin']]) == str(ingredients.param['y_lin']), 'y_lin in idx does not match.'
    if configuration['use_p']:
        assert str(p_qp_full[idx['in']['p_t']]) == str(p), 'p in idx does not match.'

    n_slack = mpc.dim['slack'] if 'slack' in mpc.dim else 0

    # check that dimensions are correct
    assert y.shape[0] == horizon*(dynamics.dim['x']+dynamics.dim['u']) - n_slack, 'y dimension is wrong'  

def test_main():
    
    # loop through configurations
    for configuration in CONFIGURATIONS:
        
        # randomly sample the horizon
        horizon = 1 if configuration['linearization']=='initial_state' else np.random.randint(2,5)

        # generate dynamics and ingredients
        dynamics,ingredients,vars = sample_mpc(horizon=horizon,**configuration)

        # form pf and p
        pf = [ca.SX(0,0)]
        if configuration['use_pf']:
            pf.append(vars['pf'])
        if configuration['use_theta']:
            pf.append(vars['theta'])
        pf = ca.vcat(pf)
        p = vars['p']

        # generate random initial conditions
        x0 = ca.DM(np.random.rand(dynamics.dim['x'],1))

        # generate theta if required
        theta0 = ca.DM(np.random.rand(dynamics.dim['theta'],1)) if configuration['use_theta'] else None

        # generate random set of inputs and obtain successive states
        u_list = []
        x_list = []
        x_t = x0
        for _ in range(horizon):
            # generate random input
            u_t = ca.DM(np.random.rand(dynamics.dim['u'],1))
            u_list.append(u_t)
            # get next state
            fun_inputs = {'x':x_t,'u':u_t,'theta':theta0} if configuration['use_theta'] else {'x':x_t,'u':u_t}
            x_t = dynamics.f_nom.call(fun_inputs)['x_next']
            x_list.append(x_t)

        # generate random linearization trajectories (if needed)
        if configuration['nonlinear'] and configuration['linearization']=='trajectory':
            # add states excluding last one, then inputs
            y_lin0 = ca.vcat([x0] + x_list[:-1] + u_list)
        elif configuration['nonlinear'] and configuration['linearization']=='initial_state':
            # just first input
            y_lin0 = u_list[0]
        else:
            y_lin0 = None

        # generate random parameters
        p0 = ca.DM(np.random.rand(*p.shape)) if configuration['use_p'] else None
        if configuration['use_pf'] or configuration['use_theta']:
            pf0 = ca.DM(np.random.rand(*pf.shape))
            if configuration['use_theta']:
                pf0[-theta0.shape[0]:] = theta0
        else:
            pf0 = None
        
        # form qp inputs
        qp_inputs = {'x':x0,'y':y_lin0,'p':p0,'pf':pf0}
        qp_inputs_trimmed = {key:val for key,val in qp_inputs.items() if val is not None}

        # form MPC
        mpc = QP(ingredients=ingredients,p=p,pf=pf,options={'solver':'daqp'})

        single_test_symbolic_order(mpc,configuration,dynamics,ingredients,horizon,p,pf)
        single_test_dynamics(mpc,qp_inputs_trimmed,x_list,u_list,configuration)

if __name__ == '__main__':
    test_problem()
    test_main()