import sys
import os
import casadi as ca
from numpy.random import randint

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sample_elements import sample_dynamics, sample_ingredients
from src.dynamics import Dynamics
from src.ingredients import Ingredients

def test_parse_inputs_all_lists():

    # create dummy dynamics
    dynamics_dict, _ = sample_dynamics(use_d=True,use_w=True,use_theta=True,nonlinear=False)
    dynamics = Dynamics(dynamics_dict)

    # random horizon
    horizon = randint(1,5)

    # get model
    model = dynamics.linearize(horizon=horizon)[0]

    # create dictionary that can be passed to ingredients
    _,_,cost,constraints = sample_ingredients(dynamics.dim,p=False,horizon=horizon)
    ing_dict = cost | constraints | model

    out = Ingredients._parse_inputs(ing_dict,horizon)
    
    for key in out.keys():

        assert isinstance(out[key], list), 'Element ' + key + ' is not a list'
        assert key in ing_dict, 'Element ' + key + ' is not in the input dictionary'
        assert len(out[key]) == horizon, 'Element ' + key + ' does not have the correct length'

        if key in cost:
            assert all([ca.mmax(ca.fabs(elem1-elem2))==0 for elem1,elem2 in zip(cost[key],out[key])]), 'Element ' + key + ' does not match'
        elif key in constraints:
            assert all([ca.mmax(ca.fabs(elem1-elem2))==0 for elem1,elem2 in zip(constraints[key],out[key])]), 'Element ' + key + ' does not match'
        elif key in model:
            assert all([ca.mmax(ca.fabs(elem1-elem2))==0 for elem1,elem2 in zip(model[key],out[key])]), 'Element ' + key + ' does not match'

def test_parse_inputs_all_single():

    # create dummy dynamics
    dynamics_dict, _ = sample_dynamics(use_d=True,use_w=True,use_theta=True,nonlinear=False)
    dynamics = Dynamics(dynamics_dict)

    # random horizon
    horizon = randint(2,5)

    # get model
    model = dynamics.linearize(horizon=1)[0]

    # create dictionary that can be passed to ingredients
    _,_,cost,constraints = sample_ingredients(dynamics.dim,p=False,horizon=1)
    ing_dict = cost | constraints | model

    out = Ingredients._parse_inputs(ing_dict,horizon)
    
    for key in out.keys():

        assert isinstance(out[key], list), 'Element ' + key + ' is not a list'
        assert key in ing_dict, 'Element ' + key + ' is not in the input dictionary'
        assert len(out[key]) == horizon, 'Element ' + key + ' does not have the correct length'

        if key in cost:
            assert all([ca.mmax(ca.fabs(elem1-elem2))==0 for elem1,elem2 in zip(cost[key],out[key])]), 'Element ' + key + ' does not match'
        elif key in constraints:
            assert all([ca.mmax(ca.fabs(elem1-elem2))==0 for elem1,elem2 in zip(constraints[key],out[key])]), 'Element ' + key + ' does not match'
        elif key in model:
            assert all([ca.mmax(ca.fabs(elem1-elem2))==0 for elem1,elem2 in zip(model[key],out[key])]), 'Element ' + key + ' does not match'

def test_parse_inputs_some_single_lists():

    # create dummy dynamics
    dynamics_dict, _ = sample_dynamics(use_d=True,use_w=True,use_theta=True,nonlinear=False)
    dynamics = Dynamics(dynamics_dict)

    # random horizon
    horizon = randint(1,5)

    # get model
    model = dynamics.linearize(horizon=1)[0]

    # create dictionary that can be passed to ingredients
    _,_,cost,constraints = sample_ingredients(dynamics.dim,p=False,horizon=horizon)
    ing_dict = cost | constraints | model

    out = Ingredients._parse_inputs(ing_dict,horizon)
    
    for key in out.keys():

        assert isinstance(out[key], list), 'Element ' + key + ' is not a list'
        assert key in ing_dict, 'Element ' + key + ' is not in the input dictionary'
        assert len(out[key]) == horizon, 'Element ' + key + ' does not have the correct length'

        if key in cost:
            assert all([ca.mmax(ca.fabs(elem1-elem2))==0 for elem1,elem2 in zip(cost[key],out[key])]), 'Element ' + key + ' does not match'
        elif key in constraints:
            assert all([ca.mmax(ca.fabs(elem1-elem2))==0 for elem1,elem2 in zip(constraints[key],out[key])]), 'Element ' + key + ' does not match'
        elif key in model:
            assert all([ca.mmax(ca.fabs(elem1-elem2))==0 for elem1,elem2 in zip(model[key],out[key])]), 'Element ' + key + ' does not match'

def test_parse_inputs_some_single_elements():

    # create dummy dynamics
    dynamics_dict, _ = sample_dynamics(use_d=True,use_w=True,use_theta=True,nonlinear=False)
    dynamics = Dynamics(dynamics_dict)

    # random horizon
    horizon = randint(1,5)

    # get model
    model = dynamics.linearize(horizon=1)[0]
    model['A'] = model['A'][0]
    model['B'] = model['B'][0]
    model['c'] = model['c'][0]

    # create dictionary that can be passed to ingredients
    _,_,cost,constraints = sample_ingredients(dynamics.dim,p=False,horizon=horizon)
    ing_dict = cost | constraints | model

    out = Ingredients._parse_inputs(ing_dict,horizon)
    
    for key in out.keys():

        assert isinstance(out[key], list), 'Element ' + key + ' is not a list'
        assert key in ing_dict, 'Element ' + key + ' is not in the input dictionary'
        assert len(out[key]) == horizon, 'Element ' + key + ' does not have the correct length'

        if key in cost:
            assert all([ca.mmax(ca.fabs(elem1-elem2))==0 for elem1,elem2 in zip(cost[key],out[key])]), 'Element ' + key + ' does not match'
        elif key in constraints:
            assert all([ca.mmax(ca.fabs(elem1-elem2))==0 for elem1,elem2 in zip(constraints[key],out[key])]), 'Element ' + key + ' does not match'
        elif key in model:
            assert all([ca.mmax(ca.fabs(elem-model[key]))==0 for elem in out[key]]), 'Element ' + key + ' does not match'

# test to ensure that a quadratic penalty is imposed on slack if present

if __name__ == '__main__':

    test_parse_inputs_all_lists()
    test_parse_inputs_all_single()
    test_parse_inputs_some_single_lists()
    test_parse_inputs_some_single_elements()