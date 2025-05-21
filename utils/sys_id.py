import casadi as ca
import numpy as np
from src.sim_var import SimVar
from typing import Callable, Tuple
from src.dynamics import Dynamics
from typeguard import Union, Optional
from scipy.linalg import solve,lstsq
import time

def get_phi(dynamics:Dynamics, horizon:int, jit:Optional[bool]=False) -> ca.Function:
    """
    Constructs a CasADi function to compute the feature mapping phi(x, u) for system identification.
    
    Args:
        dynamics: An object representing the system dynamics, with symbolic CasADi expressions for the nominal model.
            Must contain 'param_nom' and 'init' dictionaries with keys 'theta', 'x', and 'u'.
        horizon (int): The number of time steps in the identification window.
        jit (bool): If True, compiles the feature mapping function for improved performance.
    
    Returns:
        phi (ca.Function): CasADi function representing the feature mapping phi(x, u).
    """

    # check if dynamics should be compiled
    if jit:
        jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
        compilation_options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
    else:
        compilation_options = {}

    # check that nominal model is not fully known
    assert 'theta' in dynamics.param_nom, 'Theta should be set as nominal parameter in dynamics.'

    # check that theta is initialized
    if theta0 is None:
        assert 'theta' in dynamics.init, 'Theta must be initialized.'
        theta0 = dynamics.init['theta']

    # extract parameters
    theta,x,u = dynamics.param_nom['theta'],dynamics.param_nom['x'],dynamics.param_nom['u']

    # represent the model as f(x,u) = theta.T@phi(x,u)
    phi_sym = ca.jacobian(dynamics.x_next_nom,theta)

    # check that jacobian does not depend on theta
    assert not ca.depends_on(phi_sym,theta), 'Model is not parameter affine.'

    # turn into function
    phi_single = ca.Function('phi',[x,u],[phi_sym])

    # map to accept entire trajectories
    phi = phi_single.map(horizon,[False,False],[False],compilation_options)

    return phi

def ls(dynamics:Dynamics,horizon:int,lam:float,theta0:Optional[ca.DM]=None,jit:Optional[bool]=False) -> Tuple[Callable, Callable, ca.Function]:
    """
    Least Squares (LS) system identification for parameter-affine models.
    This function constructs LS update and initialization routines for online system identification
    of parameter-affine dynamics using CasADi symbolic expressions. The model is assumed to be of the form:
        x_next = f(x, u) = theta.T @ phi(x, u)
    where theta is the parameter vector to be identified, and phi(x, u) is the feature vector.
    
    Args:
        dynamics: An object representing the system dynamics, with symbolic CasADi expressions for the nominal model.
            Must contain 'param_nom' and 'init' dictionaries with keys 'theta', 'x', and 'u'.
        horizon (int): The number of time steps in the identification window.
        lam (float): Regularization parameter (typically a large positive value) for the initial covariance matrix.
        theta0 (ca.DM, optional): Initial guess for the parameter vector theta. If None, uses dynamics.init['theta'].
        jit (bool, optional): If True, compiles the feature mapping function for improved performance.
    
    Returns:
        sys_id_update (function): Function to perform a single LS update step.
            Args:
                sim: Simulation object containing state and input trajectories (sim.x, sim.u)
                    or list of simulation objects.
                running_vars (dict): Dictionary containing current LS variables ('A', 'b').
                k (int): Current time index (not used in the update).
            Returns:
                dict: Updated LS variables, including the new parameter estimate ('theta').
        sys_id_init (function): Function to initialize the LS variables.
            Returns:
                dict: Initial LS variables ('A', 'b').
        phi (ca.Function): CasADi function representing the feature mapping phi(x, u).
    
    Raises:
        AssertionError: If the model is not parameter-affine, or required keys are missing in dynamics.
    """

    # obtain phi function
    phi_func = get_phi(dynamics,horizon,jit)

    # precompute dimension of theta
    n_theta = dynamics.param_nom['theta'].shape[0]

    def sys_id_update(sim:Union[list,SimVar],running_vars:dict,k:int) -> dict:

        if isinstance(sim,list):
            x = ca.vcat([elem.x[:,:-1] for elem in sim])
            u = ca.vcat([elem.u for elem in sim])
            z = np.array(ca.vcat([elem.x[:,1:] for elem in sim])).T
        else:
            x = sim.x[:,:-1]
            u = sim.u
            z = np.array(sim.x[:,1:]).T

        # compute feature vectors
        phi = np.array(phi_func(x,u)).T

        # form linear system
        LHS = phi.T@phi + lam*np.eye(n_theta)
        RHS = phi.T@z

        # compute new model
        start = time.time()
        theta = ca.solve(LHS,RHS)
        print(f'Casadi csparse, elapsed: {time.time()-start}')
        start = time.time()
        theta = lstsq(LHS,RHS)[0]
        print(f'Scipy lstsq, elapsed: {time.time()-start}')
        start = time.time()
        theta = solve(LHS,RHS,assume_a='pos')
        print(f'Scipy solve, elapsed: {time.time()-start}')
        
        # run through the horizon and perform the RLS updates
        new_psi = {'theta':theta}

        return sim.psi | new_psi
    
    def sys_id_init() -> dict:
        """
        Initializes and returns a dictionary containing system identification parameters.

        Returns:
            dict: A dictionary with the following keys:
                - 'A': A square diagonal matrix of size n_theta, scaled by lam (ca.DM.eye(n_theta) * lam).
                - 'b': The initial parameter vector theta0.
        """
        return {'A':ca.DM.eye(n_theta)*lam,'b':theta0}
    
    return sys_id_update, sys_id_init, phi_func
    

def rls(dynamics:Dynamics,horizon:int,lam:float,theta0:ca.DM=None,jit:bool=False) -> Tuple[Callable, Callable, ca.Function]:
    """
    Recursive Least Squares (RLS) system identification for parameter-affine models.
    This function constructs RLS update and initialization routines for online system identification
    of parameter-affine dynamics using CasADi symbolic expressions. The model is assumed to be of the form:
        x_next = f(x, u) = theta.T @ phi(x, u)
    where theta is the parameter vector to be identified, and phi(x, u) is the feature vector.
    
    Args:
        dynamics: An object representing the system dynamics, with symbolic CasADi expressions for the nominal model.
            Must contain 'param_nom' and 'init' dictionaries with keys 'theta', 'x', and 'u'.
        horizon (int): The number of time steps in the identification window.
        lam (float): Regularization parameter (typically a large positive value) for the initial covariance matrix.
        theta0 (ca.DM, optional): Initial guess for the parameter vector theta. If None, uses dynamics.init['theta'].
        jit (bool, optional): If True, compiles the feature mapping function for improved performance.
    
    Returns:
        sys_id_update (function): Function to perform a single RLS update step.
            Args:
                sim: Simulation object containing state and input trajectories (sim.x, sim.u).
                running_vars (dict): Dictionary containing current RLS variables ('A', 'b').
                k (int): Current time index (not used in the update).
            Returns:
                dict: Updated RLS variables, including the new parameter estimate ('theta').
        sys_id_init (function): Function to initialize the RLS variables.
            Returns:
                dict: Initial RLS variables ('A', 'b').
        phi (ca.Function): CasADi function representing the feature mapping phi(x, u).
    
    Raises:
        AssertionError: If the model is not parameter-affine, or required keys are missing in dynamics.
    """

    # obtain phi function
    phi = get_phi(dynamics,horizon,jit)

    # precompute dimension of theta
    n_theta = dynamics.param_nom['theta'].shape[0]

    def sys_id_update(sim:SimVar,running_vars:dict,k:int) -> dict:
        """
        Performs a recursive least squares (RLS) update for system identification.

        This function updates the system identification variables (A, b, theta) using the current simulation data.
        It computes feature vectors and output vectors from the simulation, reshapes them as needed, and applies
        the RLS update equations to refine the model parameters.

        Args:
            sim (SimVar): Simulation variable object containing state and input trajectories.
            running_vars (dict): Dictionary containing the current values of 'A' and 'b' for the RLS update.
            k (int): Current time step index (unused in the function body).

        Returns:
            dict: Updated dictionary of system identification variables, merging the previous `sim.psi` with the
                new values for 'A', 'b', and 'theta'.
        """

        # get past a and 
        a_k = running_vars['A']
        b_k = running_vars['b']

        # compute feature vectors
        phi_k = np.array(phi(sim.x[:,:-1],sim.u))

        # compute output vector
        z_k = np.array(sim.x[:,1:])

        # reshape to (horizon, *phi.shape)
        phi_reshaped = phi_k.reshape(phi_k.shape[0],-1,horizon,order='F').transpose(2,1,0)

        # update a and b
        a_k_1 = ca.DM(a_k + np.einsum('nij,njk->ik', phi_reshaped, phi_reshaped.transpose(0,2,1)))
        b_k_1 = ca.DM(b_k + np.atleast_2d(np.einsum('nij,nj->i', phi_reshaped, z_k.T)).T)

        # compute new model
        theta = ca.solve(a_k_1,b_k_1)

        # run through the horizon and perform the RLS updates
        new_psi = {'A':a_k_1,'b':b_k_1,'theta':theta}

        return sim.psi | new_psi
    
    def sys_id_init() -> dict:
        """
        Initializes and returns a dictionary containing system identification parameters.

        Returns:
            dict: A dictionary with the following keys:
                - 'A': A square diagonal matrix of size n_theta, scaled by lam (ca.DM.eye(n_theta) * lam).
                - 'b': The initial parameter vector theta0.
        """
        return {'A':ca.DM.eye(n_theta)*lam,'b':theta0}
    
    return sys_id_update, sys_id_init, phi

def rls_update_debug(results:dict,horizon:int,phi:ca.Function,sim:SimVar,running_vars:dict,k:int) -> None:
        """
        Debugs and verifies the system identification update step by comparing computed feature and output vectors,
        as well as their outer products, against reference results.

        Args:
            results (dict): Dictionary containing the updated feature matrix 'phi', and outer products 'A' and 'b'.
            horizon (int): The prediction or identification horizon (number of time steps).
            phi (ca.Function): CasADi function to compute feature vectors from state and input trajectories.
            sim (SimVar): Simulation variable object containing state (x) and input (u) trajectories.
            running_vars (dict): Dictionary containing the running (previous) values of 'A' and 'b'.
            k (int): Current time step index (unused in this function).
        
        Raises:
            AssertionError: If the reshaped feature vectors or their outer products do not match the reference results.
        """
        
        # extract results
        phi_reshaped = results['phi']
        a_k_1 = results['A']
        b_k_1 = results['b']
        
        # get past a and 
        a_k = running_vars['A']
        b_k = running_vars['b']
        
        # compute feature vectors
        phi_k = np.array(phi(sim.x[:,:-1],sim.u))

        # compute output vector
        z_k = np.array(sim.x[:,1:])

        # test against for loop
        phi_k_list = np.split(phi_k.T,horizon,axis=0)
        z_k_list = np.split(z_k,horizon,axis=1)

        # check that this is equal to the list above
        for idx, elem in enumerate(phi_k_list):
            assert np.allclose(elem,phi_reshaped[idx]), 'Reshaped phi_k does not match.'

        # preallocate outer products
        product_1 = a_k #np.zeros((phi_k_list[0].shape[0],phi_k_list[0].shape[0]))
        product_2 = b_k #np.zeros((phi_k_list[0].shape[0],z_k_list[0].shape[1]))

        # test against for loop
        for i in range(horizon):
            product_1 = product_1 + phi_k_list[i]@(phi_k_list[i].T)
            product_2 = product_2 + phi_k_list[i]@z_k_list[i]

        # check accuracy
        assert np.allclose(a_k_1,product_1), 'Outer product of phi_k does not match.'
        assert np.allclose(b_k_1,product_2), 'Outer product of phi_k does not match.'

        