import sys
import os
import casadi as ca
import numpy as np

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.dynamics import random_linear
from src.dynamics import Dynamics
from src.scenario import Scenario
from src.ingredients import Ingredients
from src.qp import QP
from src.sim_var import SimVar
from src.upper_level import UpperLevel
from utils.cost_utils import param2terminal_cost,bound2poly,quad_cost_and_bounds
from utils.sys_id import ls,rls,rls_update_debug

def random_linear_scenario(
        noise:bool=True,
        noise_mag:float=0.5,
        n_x:int=3,
        theta_uncertainty_range:float=0.5,
        verbose:bool=False,
        upper_horizon:int=None,
        mpc_horizon:int=None
    ) -> Scenario:
    """
    Generates a random linear scenario for system identification and closed-loop MPC performance evaluation.

    This function creates a random linear system with configurable noise, uncertainty, and horizon parameters.
    It sets up the system dynamics, initial conditions, MPC problem, and upper-level optimization, returning
    a Scenario object that encapsulates all relevant components and initializations.

    Args:
        noise (bool, optional): Whether to add process noise to the scenario. Defaults to True.
        noise_mag (float, optional): Magnitude of the process noise. Defaults to 0.5.
        n_x (int, optional): Number of state variables for the system. Defaults to 3.
        theta_uncertainty_range (float, optional): Range of uncertainty for the model parameters. Defaults to 0.5.
        verbose (bool, optional): If True, prints additional information about the scenario setup. Defaults to False.
        upper_horizon (int, optional): Horizon length for the upper-level optimization. If None, randomly chosen.
        mpc_horizon (int, optional): Horizon length for the MPC controller. If None, randomly chosen.

    Returns:
        Scenario: An initialized Scenario object containing the system dynamics, MPC controller,
                    upper-level optimization, and initial conditions.
    """

    # check if horizons are passed
    if mpc_horizon is None:
        mpc_horizon = np.random.randint(5,10)
    if upper_horizon is None:
        upper_horizon = np.random.randint(20,30)

    # create dictionary with parameters of cart pendulum
    dyn_dict,true_theta = random_linear.dynamics(n_x=6,use_w=noise,pole_mag=[0.5,1])
    if verbose:
        print(true_theta)

    # model uncertainty parameter
    theta = dyn_dict['theta']

    # create dynamics object
    dyn = Dynamics(dyn_dict)

    # get state and input dimensions
    n_x, n_u = dyn.dim['x'], dyn.dim['u']

    # set initial conditions
    x0 = ca.DM.ones(n_x,1)#ca.DM( X0_MAG * (np.ones((n_x,1)) + 2*np.random.rand(n_x,1)) )
    theta_uncertainty = theta_uncertainty_range*(np.ones(theta.shape)+2*np.random.rand(*theta.shape))
    theta0 = ca.DM( np.multiply(theta_uncertainty,np.array(true_theta)) )
    if verbose:
        print(f'Initial condition: {x0}')
        print(f'Initial parameter estimate: {theta0}')

    # sample noise if requested
    if noise:
        w0 = ca.horzsplit(noise_mag*(2*np.random.rand(dyn.dim['w'],upper_horizon)-np.ones((dyn.dim['w'],upper_horizon))))

    ### CREATE MPC -----------------------------------------------------------------------------

    # upper level cost
    Q_true = 10*ca.DM.eye(n_x)
    R_true = 0.1

    # constraints are simple bounds on state and input
    x_max = 500*ca.DM.ones(n_x,1)
    x_min = -x_max
    u_max = 0.5
    u_min = -u_max

    # parameter = terminal state cost and input cost
    c_q = ca.SX.sym('c_q',int(n_x*(n_x+1)/2),1)
    c_r = ca.SX.sym('c_r',1,1)

    # stage cost (state)
    Qx = [Q_true] * (mpc_horizon-1)

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
    cost = {'Qx': Qx, 'Ru':Ru}

    # turn bounds into polyhedral constraints
    Hx,hx,Hu,hu = bound2poly(x_max,x_min,u_max,u_min)

    # add to mpc dictionary
    # cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu, 'Hx_e':ca.SX.eye(hx.shape[0])}
    cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu}

    # create QP ingredients
    ing = Ingredients(horizon=mpc_horizon,dynamics=dyn,cost=cost,constraints=cst)

    # create options
    qp_options = {'solver':'daqp'}

    # create MPC
    mpc = QP(ingredients=ing,p=p,pf=pf,options=qp_options)


    ### UPPER LEVEL -----------------------------------------------------------

    # create upper level
    upper_level = UpperLevel(p=p,pf=pf,horizon=upper_horizon,mpc=mpc)

    # compute terminal cost initialization
    p_init = ca.vertcat(ca.DM.ones(p.shape[0]-1,1)*1e-3,0.1)

    # extract closed-loop variables for upper level
    x_cl = ca.vec(upper_level.param['x_cl'])
    u_cl = ca.vec(upper_level.param['u_cl'])

    track_cost, _, _ = quad_cost_and_bounds(Q_true,R_true,x_cl,u_cl,x_max,x_min)

    # put together
    cost = track_cost

    # create upper-level constraints
    Hx,hx,_,_ = bound2poly(x_max,x_min,u_max,u_min,upper_horizon+1)
    _,_,Hu,hu = bound2poly(x_max,x_min,u_max,u_min,upper_horizon)
    cst_viol = ca.vcat([Hx@ca.vec(x_cl)-hx,Hu@ca.vec(u_cl)-hu])

    # store in upper-level
    upper_level.set_cost(cost,track_cost,cst_viol)

    ### CREATE SCENARIO -----------------------------------------------------------

    scenario = Scenario(dyn,mpc,upper_level)

    # initialize
    init_dict = {'p':p_init,'pf':theta0,'x': x0,'theta':theta0}
    if noise:
        init_dict['w'] = w0
    scenario.set_init(init_dict)

    return scenario

def test_rls_vs_ls_single():

    # generate random linear scenario
    scenario = random_linear_scenario()

    # noise:bool=True,
    # noise_mag:float=0.5,
    # n_x:int=3,
    # theta_uncertainty_range:float=0.5,
    # verbose:bool=False,
    # upper_horizon:int=None,
    # mpc_horizon:int=None

    # create system identification using rls
    rls_update, rls_init, phi = rls(
        dynamics=scenario.dyn,
        horizon=scenario.dim['T'],
        lam=0,
        theta0=ca.DM.zeros(scenario.init['theta'].shape[0],1),
        jit=False)
    
    # create system identification using ls
    ls_update, ls_init, phi = ls(
        dynamics=scenario.dyn,
        horizon=scenario.dim['T'],
        lam=0,
        theta0=ca.DM.zeros(scenario.init['theta'].shape[0],1),
        jit=False)
    
    # run simulation
    sim,out_dict,qp_failed = scenario.simulate(options={'mode':'simulate','solver':'daqp'})

    # run recursive least squares
    rls_param = rls_update(sim=sim,running_vars=rls_init(),k=0)['theta']

    # run least squares
    ls_param = ls_update(sim,ls_init(),0)['theta']

    # run rls using list comprehension
    rls_debug_param = rls_update_debug(scenario.dim['T'],phi,sim,rls_init(),0)

    e1 = ca.norm_2(rls_param-ls_param)
    e2 = ca.norm_2(rls_param-rls_debug_param)

    assert e1 <= 1e-8, 'LS and RLS do not match.'
    assert e2 <= 1e-8, 'RLS and list-comprehension RLS do not match.'

def test_rls_vs_ls_multiple():

    # generate random linear scenario
    scenario = random_linear_scenario()

    # noise:bool=True,
    # noise_mag:float=0.5,
    # n_x:int=3,
    # theta_uncertainty_range:float=0.5,
    # verbose:bool=False,
    # upper_horizon:int=None,
    # mpc_horizon:int=None

    # create system identification using rls
    rls_update, rls_init, phi = rls(
        dynamics=scenario.dyn,
        horizon=scenario.dim['T'],
        lam=0,
        theta0=ca.DM.zeros(scenario.init['theta'].shape[0],1),
        jit=False)
    
    # create system identification using ls
    ls_update, ls_init, phi = ls(
        dynamics=scenario.dyn,
        horizon=scenario.dim['T'],
        lam=0,
        theta0=ca.DM.zeros(scenario.init['theta'].shape[0],1),
        jit=False)
    
    # run simulation
    sim,out_dict,qp_failed = scenario.simulate(options={'mode':'simulate','solver':'daqp'})

    # run recursive least squares
    rls_param = rls_update(sim=sim,running_vars=rls_init(),k=0)['theta']

    # run least squares
    ls_param = ls_update(sim,ls_init(),0)['theta']

    # run rls using list comprehension
    rls_debug_param = rls_update_debug(scenario.dim['T'],phi,sim,rls_init(),0)

    e1 = ca.norm_2(rls_param-ls_param)
    e2 = ca.norm_2(rls_param-rls_debug_param)

    assert e1 <= 1e-8, 'LS and RLS do not match.'
    assert e2 <= 1e-8, 'RLS and list-comprehension RLS do not match.'

if __name__ == '__main__':
    test_rls_vs_ls_single()
    test_rls_vs_ls_multiple()