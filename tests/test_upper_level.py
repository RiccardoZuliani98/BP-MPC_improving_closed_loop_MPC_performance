import sys
import os
import casadi as ca
import numpy as np
import pytest

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sample_elements import sample_upper_level,sample_mpc
from src.dynamics import Dynamics
from src.ingredients import Ingredients
from src.qp import QP

def test_var_setup():
    single_test_var_setup(nonlinear=True,linearization='initial_state')
    single_test_var_setup(nonlinear=True,linearization='trajectory')
    single_test_var_setup(nonlinear=False)

def single_test_var_setup(nonlinear,linearization='trajectory'):

    # randomly choose horizon
    mpc_horizon = np.random.randint(2,5)
    upper_horizon = np.random.randint(5,10)
    
    # generate dynamics and ingredients
    dynamics,ingredients,vars = sample_mpc(horizon=mpc_horizon,use_d=False,use_w=False,
                                           use_theta=False,nonlinear=nonlinear,use_p=True,
                                           use_pf=True,use_slack=True,
                                           linearization=linearization)

    # extract pf and p
    pf_mpc = vars['pf']
    p_mpc = vars['p']

    # generate mpc
    mpc = QP(ingredients=ingredients,p=p_mpc,pf=pf_mpc)

    # generate p and pf for upper level
    p = ca.SX.sym('p',np.random.randint(5,10),1)
    pf = ca.SX.sym('pf',np.random.randint(5,10),1)

    # get dimensions
    n_p, n_pf = p.shape[0], pf.shape[0]

    # generate random indices
    idx_p_list = np.random.randint(low=0,high=n_p,size=(p_mpc.shape[0],upper_horizon))
    idx_pf_list = np.random.randint(low=0,high=n_pf,size=(pf_mpc.shape[0],upper_horizon))

    # create corresponding functions
    def idx_p(t):
        return idx_p_list[:,t]
    
    def idx_pf(t):
        return idx_pf_list[:,t]

    # generate upper level
    upper_level = sample_upper_level(p=p,mpc=mpc,pf=pf,horizon=upper_horizon,idx_p=idx_p,idx_pf=idx_pf)

    # check dimension of slack variable
    e_cl = upper_level.param['e_cl']

    assert mpc.dim['eps'] == e_cl.shape[0], 'Dimension of slack variable is wrong.'

    # get indices
    qp_var_setup = upper_level.idx['qp']

    # get random trajectory
    y = ca.DM(np.random.rand(mpc.dim['y']))

    # form linearization trajectory
    y_lin = y if linearization == 'trajectory' else y[mpc.idx['out']['u1']]

    # get random initial state
    x0 = ca.DM(np.random.rand(mpc.dim['x']))

    # get random p and pf
    p0 = ca.DM(np.random.rand(p.shape[0]))
    pf0 = ca.DM(np.random.rand(pf.shape[0]))

    # form dictionary
    var_in = {'x':x0,'y':y,'p':p0,'pf':pf0}

    # check correctness
    for t in range(upper_horizon):

        # expected value
        expected_var = ca.vertcat(x0,y_lin,p0[idx_p(t)],pf0[idx_pf(t)]) if nonlinear else ca.vertcat(x0,p0[idx_p(t)],pf0[idx_pf(t)])

        # compare
        assert ca.mmin(expected_var == qp_var_setup(var_in,t))==1, 'Indexing does not match.'

if __name__ == '__main__':
    test_var_setup()