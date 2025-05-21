import sys
import os
import casadi as ca
from numpy.random import randint
import pytest

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sample_elements import sample_mpc
from src.dynamics import Dynamics
from src.ingredients import Ingredients
from src.qp import QP

def test_symbolic_order():

    # combinations that need to be tested
    combinations = [{'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':True,'use_pf':True,'use_slack':True},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':True,'use_pf':True,'use_slack':True},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':False,'use_pf':True,'use_slack':True},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':False,'use_pf':True,'use_slack':True},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':True,'use_pf':False,'use_slack':True},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':True,'use_pf':False,'use_slack':True},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':False,'use_pf':False,'use_slack':True},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':False,'use_pf':False,'use_slack':True},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':True,'use_pf':True,'use_slack':False},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':True,'use_pf':True,'use_slack':False},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':False,'use_pf':True,'use_slack':False},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':False,'use_pf':True,'use_slack':False},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':True,'use_pf':False,'use_slack':False},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':True,'use_pf':False,'use_slack':False},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':False,'use_p':False,'use_pf':False,'use_slack':False},
                    {'use_d':True,'use_w':True,'use_theta':True,'nonlinear':True,'use_p':False,'use_pf':False,'use_slack':False},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':True,'use_pf':True,'use_slack':True},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':True,'use_pf':True,'use_slack':True},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':False,'use_pf':True,'use_slack':True},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':False,'use_pf':True,'use_slack':True},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':True,'use_pf':False,'use_slack':True},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':True,'use_pf':False,'use_slack':True},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':False,'use_pf':False,'use_slack':True},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':False,'use_pf':False,'use_slack':True},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':True,'use_pf':True,'use_slack':False},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':True,'use_pf':True,'use_slack':False},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':False,'use_pf':True,'use_slack':False},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':False,'use_pf':True,'use_slack':False},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':True,'use_pf':False,'use_slack':False},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':True,'use_pf':False,'use_slack':False},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':False,'use_p':False,'use_pf':False,'use_slack':False},
                    {'use_d':False,'use_w':False,'use_theta':False,'nonlinear':True,'use_p':False,'use_pf':False,'use_slack':False}]

    # loop through
    for configuration_single in combinations:

        dynamics,mpc,ingredients = [],[],[]

        # generate dynamics and ingredients
        dynamics,ingredients,vars = sample_mpc(horizon=randint(2,5),**configuration_single)

        # form pf and p
        pf = [ca.SX(0,0)]
        if configuration_single['use_pf']:
            pf.append(vars['pf'])
        if configuration_single['use_theta']:
            pf.append(vars['theta'])
        pf = ca.vcat(pf)
        p = vars['p']

        # form MPC
        mpc = QP(ingredients=ingredients,p=p,pf=pf)

        # extract variables
        p_t = mpc.param['p_t'] if 'p_t' in mpc.param else ca.SX(0,0)
        p_qp = mpc.param['p_qp']
        p_qp_full = mpc.param['p_qp_full']
        # y = mpc.param['y']

        # get indexing
        idx = mpc._ingredients._idx

        # expected values
        p_t_expected = p if configuration_single['use_p'] else ca.SX(0,0)
        p_qp_expected = [dynamics.param['x']]
        p_qp_full_expected = [dynamics.param['x']]
        if configuration_single['nonlinear']:
            assert 'y_lin' in ingredients.param, 'Linearization trajectory not present in dynamics.'
            p_qp_expected.append(ingredients.param['y_lin'])
            p_qp_full_expected.append(ingredients.param['y_lin'])
        if configuration_single['use_p']:
            p_qp_expected.append(p)
            p_qp_full_expected.append(p)
        if configuration_single['use_pf'] or configuration_single['use_theta']:
            p_qp_full_expected.append(pf)
        p_qp_expected = ca.vcat(p_qp_expected)
        p_qp_full_expected = ca.vcat(p_qp_full_expected)

        assert str(p_t_expected)==str(p_t), 'p_t does not match for configuration: ' + str(configuration_single)
        assert str(p_qp_expected)==str(p_qp), 'p_qp does not match for configuration: ' + str(configuration_single)
        assert str(p_qp_full_expected)==str(p_qp_full), 'p_qp_full does not match for configuration: ' + str(configuration_single)

        assert str(p_qp_full[idx['in']['x']]) == str(dynamics.param['x']), 'x in idx does not match.'
        
        if configuration_single['nonlinear']:
            assert str(p_qp_full[idx['in']['y_lin']]) == str(ingredients.param['y_lin']), 'y_lin in idx does not match.'
        if configuration_single['use_p']:
            assert str(p_qp_full[idx['in']['p_t']]) == str(p), 'p in idx does not match.'

if __name__ == '__main__':
    test_symbolic_order()