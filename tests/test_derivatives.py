import sys
import os
import casadi as ca
from numpy.random import randint, rand
from sample_elements import sample_dynamics, sample_ingredients, sample_upper_level

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dynamics import Dynamics
from src.scenario import Scenario
from src.qp import QP

# import pytest

def test_parallel_derivatives(mpc_horizon=1,upper_horizon=1,n_models=5):

    # generate noise free dynamics
    dynamics_dict, dynamics_matrices = sample_dynamics(use_d=False,use_w=False,use_theta=True,nonlinear=True)

    # extract theta
    theta = dynamics_dict['theta']

    # create dynamics
    dynamics = Dynamics(dynamics_dict)

    # create sample QP
    ingredients, p, _ = sample_ingredients(dynamics,p=True,horizon=mpc_horizon)
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
        scenario.set_init(init_dict | {'theta':theta})

        # simulate
        sim_single,*_ = scenario.simulate(options={'simulate_parallel_models':False})

        # store result
        j_x.append(sim_single.j_x)

    # concatenate
    j_x = ca.hcat(j_x)

    print('me')


if __name__ == "__main__":
    test_parallel_derivatives()