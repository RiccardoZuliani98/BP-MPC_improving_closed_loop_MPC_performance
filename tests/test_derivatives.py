import sys
import os
import casadi as ca
from numpy.random import randint, rand
import datetime

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dynamics import Dynamics
from src.scenario import Scenario
from src.qp import QP
from src.ingredients import Ingredients
from utils.sample_elements import sample_dynamics, sample_ingredients, sample_upper_level

def test_parallel_derivatives(mpc_horizon=None,upper_horizon=None,n_models=5,tol=1e-5):

    if mpc_horizon is None:
        mpc_horizon = randint(1,5)

    if upper_horizon is None:
        upper_horizon = randint(2,7)

    # generate noise free dynamics
    dynamics_dict, _ = sample_dynamics(use_d=False,use_w=False,use_theta=True,nonlinear=False)

    # extract theta
    theta = dynamics_dict['theta']

    # create dynamics
    dynamics = Dynamics(dynamics_dict)
    
    # create ingredients
    p, _, cost, constraints = sample_ingredients(dynamics.dim,p=True,horizon=mpc_horizon)
    ingredients = Ingredients(horizon=mpc_horizon,cost=cost,constraints=constraints,dynamics=dynamics)

    # create MPC
    mpc = QP(ingredients=ingredients,p=p,pf=theta)

    # create sample upper level
    upper_level = sample_upper_level(p=p,pf=theta,mpc=mpc,horizon=upper_horizon)

    # create scenario
    scenario = Scenario(dyn=dynamics,mpc=mpc,upper_level=upper_level)

    # get dimensions for simplicity
    n = scenario.dim

    # generate some thetas
    theta0 = ca.horzsplit(rand(n['theta'],n_models))

    # initialize
    init_dict = {'p':ca.DM(rand(n['p'],1)),'x': ca.DM(rand(n['x'],1)),'u': ca.DM(rand(n['u'],1)),'theta':theta0,'pf':theta0[0]}
    scenario.set_init(init_dict)

    # simulate with initial parameter
    sim,*_ = scenario.simulate(options={'simulate_parallel_models':True})
    # sim,*_ = scenario.simulate()

    # preallocate list of derivatives
    j_x = []

    # loop through all the thetas and compute individual derivatives
    for theta in theta0:

        # update initialization
        scenario.set_init({'theta':theta})

        # simulate
        sim_single,*_ = scenario.simulate(options={'simulate_parallel_models':False})

        # store result
        j_x.append(sim_single.j_x)

    # concatenate
    j_x = ca.hcat(j_x)

    # compute error
    error = ca.fabs(j_x-sim.j_x)
    error_max = ca.mmax(error)
    error_mean = ca.sum1(ca.sum2(error)) / error.numel()
    cos_sim = ca.vec(j_x).T@ca.vec(sim.j_x) / ( ca.norm_2(ca.vec(j_x)) * ca.norm_2(ca.vec(sim.j_x)) )

    if 1-cos_sim > tol:

        # save data for debug purposes
        data = {'init_dict':init_dict,'theta0':theta0,'p':p,'dynamics_dict':dynamics_dict,'upper_horizon':upper_horizon,'mpc_horizon':mpc_horizon}

        # get current date and time
        now = datetime.datetime.now()

    assert 1-cos_sim <= tol, 'Parallel derivative error is too large.'


if __name__ == "__main__":
    test_parallel_derivatives()