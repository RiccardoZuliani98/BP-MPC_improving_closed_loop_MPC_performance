import sys
import os
import casadi as ca
from numpy.random import randint, rand
import pytest

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sample_elements import sample_dynamics
from src.dynamics import Dynamics

def test_affine():
    """
    Test if the dynamics correctly identify that the model is affine.
    """

    # generate affine dynamics
    dynamics_1 = Dynamics(sample_dynamics(use_d=True,use_w=True,use_theta=True)[0])
    dynamics_2 = Dynamics(sample_dynamics()[0])

    assert dynamics_1._is_affine, 'Model is not recognized to be affine.'
    assert dynamics_2._is_affine, 'Model is not recognized to be affine.'

    # generate nonlinear dynamics
    dynamics_nonlinear_1 = Dynamics(sample_dynamics(nonlinear=True)[0])
    dynamics_nonlinear_2 = Dynamics(sample_dynamics(use_d=True,use_w=True,use_theta=True,nonlinear=True)[0])

    assert not dynamics_nonlinear_1._is_affine, 'Model is not recognized to be nonlinear.'
    assert not dynamics_nonlinear_2._is_affine, 'Model is not recognized to be nonlinear.'

def test_nominal_and_derivatives():
    """
    Test correctness of nominal dynamics generation and their derivatives.
    """

    # generate affine dynamics
    dynamics_1_dict,dynamics_1_matrices = sample_dynamics(use_d=True,use_w=True,use_theta=True)
    dynamics_1 = Dynamics(dynamics_1_dict)
    dynamics_2_dict,dynamics_2_matrices = sample_dynamics()
    dynamics_2 = Dynamics(dynamics_2_dict)

    # initialization should contain None values in x and u
    assert len(dynamics_1.init) == 0, 'Init is not empty.'
    assert len(dynamics_2.init) == 0, 'Init is not empty.'

    # get dimensions
    n1 = dynamics_1.dim
    n2 = dynamics_2.dim

    # create initializations
    x1 = ca.DM(rand(n1['x'],1))
    u1 = ca.DM(rand(n1['u'],1))
    d1 = ca.DM(rand(n1['d'],1))
    w1 = ca.DM(rand(n1['w'],1))
    x2 = ca.DM(rand(n2['x'],1))
    u2 = ca.DM(rand(n2['u'],1))

    # create initialization dictionaries
    init1 = {'x':x1,'u':u1,'d':d1}
    init2 = {'x':x2,'u':u2}

    # pass initializations
    dynamics_1._set_init(init1)
    dynamics_2._set_init(init2)

    # helper function to check if two dictionaries containing DM variables coincide
    def compare_dicts(dict1,dict2):

        # check that keys are equal
        dict_equal = dict1.keys() == dict2.keys()

        # check that numerical values are equal
        dict_equal = dict_equal and all([ca.mmax(dict1[key] - dict2[key]) == 0 for key in dict1.keys()])

        return dict_equal

    # check that initializations are correct
    assert compare_dicts(dynamics_1.init,init1), 'Initialization is not handled correctly.'
    assert compare_dicts(dynamics_2.init,init2), 'Initialization is not handled correctly.'

    # add one more initialization
    dynamics_1._set_init({'w':w1})
    assert compare_dicts(dynamics_1.init,init1 | {'w':w1}), 'Passing initialization in two steps failed.'

    # try adding a wrong type
    test_passed = False
    try:
        dynamics_1._set_init({'ciao':1})
    except:
        test_passed = True

    assert test_passed, 'Wrong initialization was not rejected.'

    # check that f and fc coincide with non-noisy model
    assert ca.sum1(ca.fabs(ca.DM(ca.cse(dynamics_2.f_nom(dynamics_2.param_nom['x'],dynamics_2.param_nom['u']) - dynamics_2.x_next)))) == 0, 'Nominal and true dynamics should coincide'
    assert ca.sum1(ca.fabs(ca.DM(ca.cse(dynamics_2.f(dynamics_2.param_nom['x'],dynamics_2.param_nom['u']) - dynamics_2.x_next_nom)))) == 0, 'Nominal and true dynamics should coincide'
    
    # check derivatives
    assert ca.mmin(dynamics_2.A(x2,u2) == ca.DM(dynamics_2_matrices['A'])), 'x derivative does not match.'
    assert ca.mmin(dynamics_2.B(x2,u2) == ca.DM(dynamics_2_matrices['B'])), 'u derivative does not match.'
    assert ca.mmin(dynamics_2.B_nom(x2,u2) == ca.DM(dynamics_2_matrices['B'])), 'u derivative does not match.'
    assert ca.mmin(dynamics_2.A_nom(x2,u2) == ca.DM(dynamics_2_matrices['A'])), 'u derivative does not match.'

def test_nonlinear_if_d_nonzero():

    # create model where nonlinearity occurs only if d is not zero
    dynamics_dict,_ = sample_dynamics(use_d=True,use_w=False,use_theta=True,nonlinear=False)

    # add nonlinear term that is zero for d = 0
    d = dynamics_dict['d']
    x = dynamics_dict['x']
    A_d_x = ca.DM(rand(x.shape[0],x.shape[0]))
    dynamics_dict['x_next'] = dynamics_dict['x_next'] + (A_d_x@d[0])@(x**2)

    # create dynamics
    dynamics = Dynamics(dynamics_dict)
    
    # check that true model is recognized as nonlinear
    assert not dynamics._is_affine, 'Model was not recognized as nonlinear.'

    # check that nominal model is recognized as affine
    assert dynamics._is_nominal_affine, 'Nominal model was not recognized as affine'

    # run linearization and check that linearization method == 'affine'
    *_, linearization_method = dynamics.linearize(3, method='trajectory')
    assert linearization_method == 'affine', 'linearize method did not choose the affine linearization option'
    
def testlinearize_affine():
    """
    Test the linearize method for affine dynamics.
    """
    # Generate affine dynamics
    dynamics_dict, dynamics_matrices = sample_dynamics()
    dynamics = Dynamics(dynamics_dict)

    # Set horizon
    horizon = 5

    # Call linearize
    model, symbolic_vars, linearization_method = dynamics.linearize(horizon, method='trajectory')

    # Check linearization method
    assert linearization_method == 'affine', 'Linearization method should be affine.'

    # Check dimensions of A, B, and c
    assert len(model['A']) == horizon, 'Incorrect number of A matrices.'
    assert len(model['B']) == horizon, 'Incorrect number of B matrices.'
    assert len(model['c']) == horizon, 'Incorrect number of c vectors.'

    # Check that A, B, and c are consistent with dynamics matrices
    for a, b, c in zip(model['A'], model['B'], model['c']):
        assert ca.DM(ca.mmin(a == ca.DM(dynamics_matrices['A']))), 'A matrix does not match.'
        assert ca.DM(ca.mmin(b == ca.DM(dynamics_matrices['B']))), 'B matrix does not match.'
        assert ca.DM(ca.mmin(c == ca.DM(dynamics_matrices['c']))), 'c vector does not match.'


def testlinearize_initial_state():
    """
    Test the linearize method for initial_state linearization.
    """
    # Generate affine dynamics
    dynamics_dict, _ = sample_dynamics(nonlinear=True)
    dynamics = Dynamics(dynamics_dict)

    # get nominal dynamics
    f = dynamics.f_nom

    # Set horizon
    horizon = randint(2,5)

    # Call linearize
    model, symbolic_vars, linearization_method = dynamics.linearize(horizon, method='initial_state')

    # Check linearization method
    assert linearization_method == 'initial_state', 'Linearization method should be initial_state.'

    # Check dimensions of A, B, and c
    assert len(model['A']) == horizon, 'Incorrect number of A matrices.'
    assert len(model['B']) == horizon, 'Incorrect number of B matrices.'
    assert len(model['c']) == horizon, 'Incorrect number of c vectors.'

    # Check symbolic variable y_lin
    assert 'y_lin' in symbolic_vars.var, 'y_lin not found in symbolic variables.'

    # get variables required to run functions
    x = symbolic_vars.var['x']
    u = symbolic_vars.var['y_lin']

    # verify that derivatives are correct
    for A,B,c in zip(model['A'],model['B'],model['c']):

        # create casadi functions
        A_func = ca.Function('A',[x,u],[A])
        B_func = ca.Function('A',[x,u],[B])
        c_func = ca.Function('A',[x,u],[c])

        # random value for x and u
        x0 = ca.DM(rand(*x.shape))
        u0 = ca.DM(rand(*u.shape))

        # compute error
        error = f(x0,u0) - ( A_func(x0,u0)@x0 + B_func(x0,u0)@u0 + c_func(x0,u0) )
        assert ca.norm_2(error) <= 1e-12, 'Linearized dynamics are incorrect'


def testlinearize_trajectory():
    """
    Test the linearize method for trajectory linearization.
    """
    # Generate affine dynamics
    dynamics_dict, _ = sample_dynamics(nonlinear=True)
    dynamics = Dynamics(dynamics_dict)

    # Set horizon
    horizon = randint(2,5)

    # Call linearize
    model, symbolic_vars, linearization_method = dynamics.linearize(horizon, method='trajectory')

    # get nominal dynamics
    f = dynamics.f_nom

    # Check linearization method
    assert linearization_method == 'trajectory', 'Linearization method should be trajectory.'

    # Check dimensions of A, B, and c
    assert len(model['A']) == horizon, 'Incorrect number of A matrices.'
    assert len(model['B']) == horizon, 'Incorrect number of B matrices.'
    assert len(model['c']) == horizon, 'Incorrect number of c vectors.'

    # Check symbolic variable y_lin
    assert 'y_lin' in symbolic_vars.var, 'y_lin not found in symbolic variables.'

        # get variables required to run functions
    x = symbolic_vars.var['x']
    y_lin = symbolic_vars.var['y_lin']
    u = symbolic_vars.var['u']

    # verify that derivatives are correct
    for A,B,c in zip(model['A'],model['B'],model['c']):

        # create casadi functions
        A_func = ca.Function('A',[x,y_lin],[A])
        B_func = ca.Function('A',[x,y_lin],[B])
        c_func = ca.Function('A',[x,y_lin],[c])

        # random value for x and u
        x0 = ca.DM(rand(*x.shape))
        u0 = ca.DM(rand(*u.shape))

        # construct y_lin accordingly
        y_lin0 = ca.vertcat(ca.repmat(x0,horizon,1),ca.repmat(u0,horizon,1))

        # compute error
        error = f(x0,u0) - ( A_func(x0,y_lin0)@x0 + B_func(x0,y_lin0)@u0 + c_func(x0,y_lin0) )
        assert ca.norm_2(error) <= 1e-12, 'Linearized dynamics are incorrect'


def testlinearize_invalid_method():
    """
    Test the linearize method with an invalid method.
    """
    # Generate affine dynamics
    dynamics_dict, _ = sample_dynamics(nonlinear=True)
    dynamics = Dynamics(dynamics_dict)

    # Set horizon
    horizon = 3

    # Call linearize with an invalid method
    with pytest.raises(Exception, match='unknown linearization method'):
        dynamics.linearize(horizon, method='invalid_method')

if __name__ == '__main__':
    test_affine()
    test_nominal_and_derivatives()
    test_nonlinear_if_d_nonzero()
    testlinearize_affine()
    testlinearize_initial_state()
    testlinearize_trajectory()
    testlinearize_invalid_method()