import casadi as ca
from src.dynamics import Dynamics
from src.qp import QP
from src.upper_level import UpperLevel
from src.sim_var import SimVar
import time
from numpy.random import randint
from typeguard import typechecked
from src.options import Options
from src.symbolic_var import SymbolicVar
import numpy as np
from typing import Tuple, Optional, Union

"""
TODO: trajectory optimization should be a separate class!
TODO: inherit methods related to options and symbolic variables that are shared across classes
TODO: create multiple consecutive scenarios with the same variables and see what happens
"""

class Scenario:
    # TODO: add description

    _OPTIONS_ALLOWED_VALUES = {'shift_linearization': bool, 'warmstart_first_qp': bool, 'warmstart_shift': bool,
                               'epsilon': float, 'roundoff_qp': int, 'mode': ['optimize', 'simulate', 'dense'],
                               'gd_type': ['gd', 'sgd'], 'figures': bool, 'random_sampling': bool, 'debug_qp': bool,
                               'compute_qp_ingredients': bool, 'verbosity': [0, 1, 2], 'max_k': int,
                               'use_true_model': bool, 'simulate_parallel_models': bool,
                               'compile_mapped_dynamics':bool,'true_theta':np.ndarray}

    _OPTIONS_DEFAULT_VALUES = {'shift_linearization': True, 'warmstart_first_qp': True, 'warmstart_shift': True,
                               'epsilon': 1e-6, 'roundoff_qp': 10, 'mode': 'optimize', 'gd_type': 'gd',
                               'figures': False, 'random_sampling': False, 'debug_qp': False,
                               'compute_qp_ingredients': False, 'verbosity': 1, 'max_k': 200,
                               'use_true_model': True, 'simulate_parallel_models': False,
                               'compile_mapped_dynamics':False,'true_theta':np.zeros(1)}

    @typechecked
    def __init__(self,dyn:Dynamics,mpc:QP,upper_level:UpperLevel):
        """
        Initializes the scenario with the given dynamics, MPC controller, and optional upper-level controller.

        Args:
            dyn (Dynamics): The system dynamics object.
            mpc (QP): The model predictive controller (MPC) object.
            upper_level (UpperLevel, optional): An optional upper-level controller.

        Initializes internal properties and updates the scenario with the provided components.
        """

        # initialize properties
        self._sym = None
        self._dyn = None
        self._qp = None
        self._upper_level = None
        self._options = None
        self._trajectory_opt = None
        self._mapped = {}

        # run update
        self.update(dyn=dyn,qp=mpc,upper_level=upper_level)

    def update(self,**kwargs):
        """
        Update the scenario object with new components and properties.

        This method updates the internal properties of the scenario object based on the provided keyword arguments.
        It supports updating the following components:
            - 'dyn': The dynamics component.
            - 'qp': The quadratic programming component.
            - 'upper_level': The upper-level component.

        For each provided component, if the value is not None, the corresponding internal attribute is updated.
        The method also manages the symbolic variables and options associated with the scenario:
            - If symbolic variables or options are not already set, they are initialized.
            - The symbolic variables are updated to include those from the dynamics, QP, and (optionally) upper-level components.
            - The options are updated by combining the QP options with the current options.

        Args:
            **kwargs: Arbitrary keyword arguments for updating scenario components.
                Allowed keys: 'dyn', 'qp', 'upper_level'.

        Raises:
            AssertionError: If an invalid key is provided in kwargs.
        """

        # initialize properties
        for key, value in kwargs.items():
            assert key in ['dyn','qp','upper_level'], 'Wrong key value in update.'
            if value is not None:
                setattr(self, f"_{key}", value)

        # check if class already possesses symbols
        current_sym = self._sym if self._sym is not None else SymbolicVar()

        # check if class already possesses options
        current_options = self._options if self._options is not None else Options(self._OPTIONS_ALLOWED_VALUES, self._OPTIONS_DEFAULT_VALUES)

        # create symbols
        self._sym = self._dyn._sym + self._qp._sym + self._upper_level._sym + current_sym if self._upper_level is not None else self._dyn._sym + self._qp._sym + current_sym

        # create options
        self._options = self._qp.options + current_options

    @property
    def sym(self):
        return self._sym
    
    @property
    def dim(self):
        return self._sym.dim

    @property
    def options(self):
        return self._options

    @property
    def param(self):
        return self._sym.var

    @property
    def init(self):
        return self._sym.init
    
    def set_init(self, init):
        """
        Set the initial value for the symbolic variable.

        Parameters:
            init: The initial value to be set for the symbolic variable.
        """
        self._sym.set_init(init)

    @property
    def dyn(self):
        return self._dyn

    @property
    def qp(self):
        return self._qp

    @property
    def upper_level(self):
        return self._upper_level

    @property
    def trajectory_opt(self):
        return self._trajectory_opt
    
    def make_trajectory_opt(self,theta=None):
        """
        Creates and returns an optimal control trajectory solver for the system.
        This method formulates an optimal control problem using CasADi's Opti stack,
        based on the system dynamics and cost function defined in the class. The solver
        can be used to compute optimal state and control trajectories given an initial
        condition and optional warm-starts for the optimization variables.
        Args:
            theta (optional): Parameters for the system dynamics. If provided, the nominal
                dynamics function is parameterized by `theta`. Default is None.
        Returns:
            solver (function): A function that solves the optimal control problem.
                The solver has the following signature:
                    solver(x0_numeric, x_init=None, u_init=None)
                where:
                    x0_numeric (array-like): Initial state vector.
                    x_init (array-like, optional): Initial guess for the state trajectory.
                    u_init (array-like, optional): Initial guess for the control trajectory.
                The solver returns:
                    out (simVar): An object containing the optimal state and control trajectories,
                        as well as the optimal cost.
                    solved (bool): True if the optimization was successful, False otherwise.
        Notes:
            - The optimization problem enforces system dynamics and constraints at each time step.
            - The cost function and constraints are obtained from the upper-level cost method.
            - The solver uses IPOPT as the backend optimizer.
        """
  
        # extract system dynamics
        if theta is not None:
            f = lambda x,u: self.dyn.f_nom(x,u,theta)
        else:
            f = self.dyn.f_nom

        # extract dimensions
        n = self.dim
        
        # create opti object
        opti = ca.Opti()

        # create optimization variables
        x = opti.variable(n['x'],n['T']+1)
        u = opti.variable(n['u'],n['T'])

        # initial condition is a parameter
        x0 = opti.parameter(n['x'],1)

        # initial condition
        opti.subject_to( x[:,0] == x0)

        # loop for constraints and dynamics
        for t in range(1,n['T']+1):
        
            # dynamics
            opti.subject_to( x[:,t] == f(x[:,t-1],u[:,t-1]) )

        # to get cost, create fake simVar and pass it through the cost function
        s = SimVar(n)

        # add optimization variables (p and y are set to zero by default)
        s.x = x
        s.u = u

        # now get cost as a symbolic function of x and u
        _,cost,cst = self.upper_level.cost(s)
        
        # set constraints
        opti.subject_to(cst <= 0)

        # set objective
        opti.minimize( cost )

        # solver
        opts = dict()
        opts["print_time"] = False
        opts['ipopt'] = {"print_level": 0, "sb":'yes'}
        opti.solver('ipopt',opts)

        # create solver function
        def solver(x0_numeric,x_init=None,u_init=None):
            """
            Solves the optimization problem using CasADi's Opti interface, with optional warm-starting.
            Parameters:
                x0_numeric (array-like): The initial condition for the optimization variable x0.
                x_init (array-like, optional): Initial guess for the state variable x. If provided, used to warm-start the solver.
                u_init (array-like, optional): Initial guess for the control variable u. If provided, used to warm-start the solver.
            Returns:
                out (simVar): An object containing the solution variables:
                    - x (ca.DM): The optimized state trajectory.
                    - u (ca.DM): The optimized control trajectory.
                    - cost (ca.DM): The value of the cost function at the solution.
                solved (bool): True if the solver succeeded, False otherwise.
            Notes:
                Prints 'NLP failed' if the solver encounters an exception.
            """
            
            # set initial condition
            opti.set_value(x0,x0_numeric)
            
            # if initialization is passed, warmstart
            if x_init is not None:
                opti.set_initial(x,x_init)
            if u_init is not None:
                opti.set_initial(u,u_init)

            # solve problem
            solved = True
            try:
                opti.solve()
            except:
                print('NLP failed')
                solved = False

            # create output simVar
            out = SimVar(n)
            out.x = ca.DM(np.atleast_2d(opti.value(x)))
            out.u = ca.DM(np.atleast_2d(opti.value(u)))
            out.cost = ca.DM(opti.value(cost))

            return out,solved

        return solver

    def create_mapped_dynamics(self,n_models,jit=False):
        """
        Creates and stores mapped (vectorized or batched) versions of the system dynamics and cost
        Jacobian functions for multiple models. This method prepares the system's nominal dynamics
        matrices (`A_nom` and `B_nom`) and the upper-level cost Jacobian function for efficient
        evaluation over `n_models` instances, optionally using JIT compilation for performance.
        The mapped functions are stored in the `self._mapped` dictionary for later use.

        Args:
            n_models (int): The number of model instances to map the dynamics and cost Jacobian over.
            jit (bool, optional): If True, enables JIT compilation for the mapped functions to improve
                performance. Defaults to False.
        Side Effects:
            Updates `self._options` to enable compilation if `jit` is True.
            Populates `self._mapped` with mapped versions of the system dynamics (`A`, `B`) and the
                cost Jacobian (`j_cost`).
        Notes:
            - The mapping and compilation options are configured based on the `jit` argument and the
                `self._options['compile_mapped_dynamics']` flag.
            - The mapped cost Jacobian function (`j_cost`) internally computes the correct cost index
                and Jacobian components before evaluating the mapped function.
        """

        if jit:
            self._options.update({'compile_mapped_dynamics' : True})

        # check if dynamics should be compiled
        if self._options['compile_mapped_dynamics']:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            compilation_options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            compilation_options = {}

        # create mapped dynamics
        self._mapped['n_models'] = n_models
        self._mapped['A'] = self.dyn.A_nom.map(n_models, [True, True, False], [False], compilation_options)
        self._mapped['B'] = self.dyn.B_nom.map(n_models, [True, True, False], [False], compilation_options)

        # create mapped cost jacobian
        j_cost_func_temp_mapped = self.upper_level._j_cost_func_temp.map(n_models, [True, False, False, False], [False], compilation_options)

        def j_cost_func(s):

            # get true input cost
            cost_in_loc = self.upper_level._get_cost_idx(s.x,s.u,s.y,s.p)

            # get true Jacobian
            j_x,j_u,j_y = self.upper_level._get_cost_jacobian(s.j_x,s.j_u,s.j_y)

            return j_cost_func_temp_mapped(cost_in_loc,j_x,j_u,j_y)

        self._mapped['j_cost'] = j_cost_func

    def _get_init_parameters(self,init:dict=None):
        """
        Prepare and validate initialization parameters for the scenario.
        This method processes the initialization dictionary or vector, ensuring that all required
        parameters are present and correctly formatted for simulation or optimization. It handles
        special cases for parameters such as 'theta', 'w', and 'y_lin', supporting both single
        values and lists (trajectories), and adapts their structure as needed for the chosen
        linearization mode.

        Parameters
            init (dict, optional): Initialization values for the scenario parameters. If provided,
            it is used to set the initial values for the scenario variables (e.g., x, u, w, d, 
            theta, p, pf, y_lin). If None, uses the current self.init values.
        
        Returns
        
            p (casadi.DM, list, or None): Parameters for the scenario.
            pf (casadi.DM, list, or None): Fixed parameters for the scenario.
            w (casadi.DM, list, or None): Noise values for the scenario.
            d (casadi.DM, list, or None): Model uncertainty values for the scenario.
            theta (casadi.DM, list, or None): Nominal model parameters for the scenario.
            y (casadi.DM, list, or None): Linearization trajectory for the scenario.
            x (casadi.DM or list): Initial state(s) for the scenario.
        
        Raises
            AssertionError, If required parameters are missing or if lengths are inconsistent.
            Exception, If required parameters for simulation are missing or if an unsupported
                linearization mode is requested.
        
        Notes
            - Handles both single-step and multi-step (trajectory) initializations.
            - Adapts parameter shapes to match the required simulation or optimization format.
            - Special handling for 'theta', 'w', and 'y_lin' to support lists and matrices.
            - Ensures all required parameters are present based on the scenario configuration.
        """

        # pass the initialization and use the SymbolicVar.set_init function
        if init is not None:
            self.set_init(init)

        # now self.init contains initialization values that are either a single ca.DM vector
        # or a list of ca.DM vectors, each vector has the dimension of the associated variable,
        # i.e., the value contained in self.dim.
        init_values = self.init.copy()

        # theta is an exception: it can be a list or a list of lists. If this is the case,
        # it must be converted to either a single DM matrix or list of matrices.
        if 'theta' in init_values and isinstance(init_values['theta'], list):
            # verify if theta is a list of lists
            if isinstance(init_values['theta'][0],list):
                # if so, concatenate horizontally each element within the list
                init_values['theta'] = [ca.hcat(elem) for elem in init_values['theta']]
            else:
                # otherwise, concatenate theta directly
                init_values['theta'] = ca.hcat(init_values['theta'])

        # w must always be a list or a list of lists
        if 'w' in init_values:
            # check if it is a list
            assert isinstance(init_values['w'], list), 'w must be a list of length T'
            # check if it is a list of lists
            if isinstance(init_values['w'][0],list):
                # if so, concatenate horizontally each element within the list
                init_values['w'] = [ca.hcat(elem) for elem in init_values['w']]
            else:
                # otherwise, concatenate w directly
                init_values['w'] = ca.hcat(init_values['w'])

        # first check if at least one of the init values is a list
        lengths = [len(v) if isinstance(v,list) else 1 for v in init_values.values()]
        
        # if there are multiple nonzero lengths, check that they match
        assert len(set([item for item in lengths if item != 1])) <= 1, 'All parameters must have the same length.'
        
        # get final length
        max_length = max(lengths)

        # if there is at least one nonzero length, extend all "dynamics" parameters to that length
        if max_length > 1:
            
            # at least one parameter in "dynamics" is a list of a certain length
            for k,v in init_values.items():
                
                # only check among parameters that are inside dynamics subclass
                if k in self.dyn.param:
                    
                    # if the parameter is None, do nothing. If the parameter is a list,
                    # do nothing. Only extend to a list of appropriate length the
                    # parameters that are currently not lists
                    if (v is not None) and (not isinstance(v,list)):
                        init_values[k] = [v]*max_length

        # now "init_values" contains x,u,w,d,theta as lists of the same length if max_length > 1,
        # otherwise they are all vectors (or matrices). Note that w,d,theta are optional. Moreover,
        # p,pf,y_lin are up to now vectors if they are present.

        # under the "trajectory" linearization mode, we need y_lin to be a trajectory
        if self.options['linearization'] == 'trajectory':
            
            # if adaptive mode is used, copy x and u to create y_lin
            if ('y_lin' not in init_values) or (init_values['y_lin']=='adaptive'):
            
                # if u was not passed return an error
                assert 'u' in init_values, 'Either pass an input or a linearization trajectory.'
                
                # check if y_lin should be a list or a single value
                if max_length > 1:
                    init_values['y_lin'] = [ca.vertcat(ca.repmat(init_values['x'][i],self.dim['N'],1),ca.repmat(init_values['u'][i],self.dim['N'],1)) for i in range(max_length)]
                else:
                    init_values['y_lin'] = ca.vertcat(ca.repmat(init_values['x'],self.dim['N'],1),ca.repmat(init_values['u'],self.dim['N'],1))
            
            # check if optimal mode is used
            elif init_values['y_lin'] == 'optimal':
                raise Exception('Optimal linearization trajectory not implemented yet.')
                # TODO implement optimal linearization trajectory
            
            # last case is if y_lin was passed (either as a vector or as a list)
            else:
                # if it was not a list, make it a list
                if not isinstance(init_values['y_lin'],list) and max_length > 1:
                    init_values['y_lin'] = [init_values['y_lin']] * max_length

        # under the "initial_state" linearization mode, we need y_lin to be a single input
        if self.options['linearization'] == 'initial_state':
            
            # check if y_lin is not passed
            if 'y_lin' not in init_values:
                
                # if so, check if an input is passed
                if 'u' not in init_values:

                    # if none is passed, raise an exception
                    raise Exception('Either pass an input or a linearization trajectory.')
                
                # set equal to input (note that input is already either a list or a vector)
                init_values['y_lin'] = ca.DM.zeros(self.qp.dim['y'],1)
                init_values['y_lin'][self.upper_level.idx['y'](0)] = init_values['u']

        # initial condition
        assert 'x' in init_values, 'Initial state x is required to simulate the system.'
        
        if 'p' in self.param and 'p' not in init_values:
            raise Exception('Parameters p are required to simulate the system.')
        
        if 'pf' in self.param and 'pf' not in init_values:
            raise Exception('Fixed parameters pf are required to simulate the system.')
        
        if 'w' in self.param and 'w' not in init_values:
            raise Exception('Noise w is required to simulate the system.')
        
        if 'd' in self.param and 'd' not in init_values:
            raise Exception('Model uncertainty d is required to simulate the system.')

        # extract variables
        p = init_values['p'] if 'p' in init_values else None                # parameters
        pf = init_values['pf'] if 'pf' in init_values else None             # fixed parameters
        w = init_values['w'] if 'w' in init_values else None                # noise
        d = init_values['d'] if 'd' in init_values else None                # model uncertainty
        theta = init_values['theta'] if 'theta' in init_values else None    # nominal model
        y = init_values['y_lin'] if 'y_lin' in init_values else None        # linearization trajectory
        x = init_values['x']                                                # initial state

        return p,pf,w,d,theta,y,x

    def simulate(self,init:dict=None,options:dict=None) -> Union[SimVar,dict,bool]:
        """
        Simulates the system dynamics based on the provided initial parameters and options.
        Args:
            init (dict, optional): A dictionary containing initial parameters for the simulation.
                If None, default initial parameters will be used.
            options (dict, optional): A dictionary of options to update the simulation settings.
                If None, the current options will be used.
        Returns:
            tuple: A tuple containing:
                - s (object): The simulation result or state.
                - out_dict (dict): A dictionary containing additional output data from the simulation.
                - qp_failed (bool): A flag indicating whether the quadratic programming (QP) solver failed.
        """

        # get initial parameters
        p,pf,w,d,theta,y,x = self._get_init_parameters(init)

        # update options if provided
        if options is not None:
            self._options.update(options)

        # check if user wants to simulate several models in parallel
        if theta is not None and self._options['simulate_parallel_models']:

            # get number of models (columns of theta)
            # n_models = int(theta[0].shape[1]) if isinstance(theta,list) else theta.shape[1]
            n_models = theta.shape[1]

            # check that mapped (nominal) dynamics have been created
            if not self._mapped:
                self.create_mapped_dynamics(n_models,self._options['compile_mapped_dynamics'])
            else:
                # if they have been created, check that n_models matches
                assert self._mapped['n_models'] == n_models, 'The mapped dynamics do not have the correct n_models'

        else:
            n_models = 1

        # if more than one model, do not use true model
        if n_models > 1:
            self._options.update({'use_true_model':False})

        # create dictionary to run simulation
        input_vars = {'p':p,'pf':pf,'d':d,'w':w,'theta':theta,'x':x,'y':y}

        # simulate
        s, out_dict, qp_failed = self._simulate(input_vars,n_models)

        return s, out_dict, qp_failed

    def _simulate(
            self,
            var_in,
            n_models:int=1
        ) -> Tuple[SimVar,dict,bool]:
        """
        Simulates the closed-loop system using the specified QP-based controller for a given scenario.
        This method performs a simulation loop over the prediction horizon, solving a quadratic program
        (QP) at each time step to compute the optimal control input, propagating the system dynamics,
        and optionally computing Jacobians for sensitivity analysis. It supports both single and multiple
        model scenarios, warmstarting, dense/sparse QP solvers, and various debugging and optimization options.
        
        Parameters
            var_in (dict): Dictionary containing the initial conditions for the simulation. Expected keys:
                - 'x': Initial state vector.
                - 'y': Initial guess for optimization variables (optional).
                - 'theta': Model parameters (optional).
                - 'p': Parameters for the QP (optional).
                - 'pf': Final parameters for the QP (optional).
                - 'd': Disturbance vector (optional).
                - 'w': Process noise matrix (optional).
            n_models (int, optional): Number of models to simulate in parallel. Default is 1 (single model).
        
        Returns
            sim (simVar): Object containing the simulation results, including state, input, slack variables,
                and optionally Jacobians.
            out_dict (dict): Dictionary containing additional simulation information, such as:
                - 'qp_time': List of QP solver times per step.
                - 'jac_time': List of Jacobian computation times per step (if applicable).
                - 'qp_debug': Debug information for each QP (if enabled).
                - 'qp_ingredients': QP ingredients for each step (if enabled).
            qp_failed (bool): Flag indicating whether the QP solver failed during the simulation.
        
        Raises
            Exception If the QP solver fails at any time step, the simulation loop is broken and `qp_failed
                is set to True.
        
        Notes
            - The method supports both dense and sparse QP solvers, as well as warmstarting and shifting of
                optimization variables.
            - Jacobians are computed if the simulation is run in 'optimize' mode.
            - Additional debug and QP ingredient information can be collected based on options.
            - The simulation results are stacked at the end for easier post-processing.
        """

        # check if multiple models have been passed
        single_model = False if n_models > 1 else True

        # extract QP for simplicity
        qp = self.qp

        # extract dimensions for simplicity
        n = self.dim

        # flag to check if QP failed
        qp_failed = False

        # extract dynamics
        f = self.dyn.f

        # extract Jacobians of dynamics
        if single_model:
            A = self.dyn.A if self.options['use_true_model'] else self.dyn.A_nom
            B = self.dyn.B if self.options['use_true_model'] else self.dyn.B_nom
        else:
            A = self._mapped['A']
            B = self._mapped['B']

        # create simVar for current simulation
        sim = SimVar(n,n_models)

        # store p and pf if present
        sim.p = var_in['p'] if 'p' in var_in else None
        sim.pf = var_in['pf'] if 'pf' in var_in else None

        # set initial conditions
        x_t = var_in['x']
        sim.x.append(x_t)

        # extract parameter indexing
        idx_qp = self.upper_level.idx['qp']
        idx_jac = self.upper_level.idx['jac']

        # create qp variable. This variable is a "view" on the dictionary var_in.
        # This means that if var_in is modified, then var_in_qp will access the
        # modified variables. However, var_in_qp cannot overwrite var_in.
        var_in_qp = {key:var_in[key] for key in ['x','y','p','pf'] if key in var_in and var_in[key] is not None}

        # extract solver
        if self._options['mode'] == 'dense':

            # if in dense mode, choose dense solver
            solver = qp.dense_solve
        else:

            # otherwise, choose sparse solver
            solver = qp.solve
        
        # in optimize mode, initialize Jacobians
        if self._options['mode'] == 'optimize':
            # initialize Jacobians
            j_x_p_t = ca.DM(n['x'],n['p']*n_models) if single_model else np.zeros((n['x'],n['p'],n_models))
            j_y_p_t = ca.DM(n['y'],n['p']*n_models)
            sim.j_x.append(j_x_p_t)
        else:
            j_x_p_t, j_y_p_t = None, None

        # check if QP warmstart was passed
        if self._options['warmstart_first_qp']:

            # get qp parameter
            p_0 = idx_qp(var_in_qp,0)

            # run QP once to get better initialization
            lam_t,mu_t,y_all_t = qp.solve(p_0)

            # update y0
            y0_x = y_all_t[qp.idx['out']['x'][:-n['x']]]
            y0_u = y_all_t[qp.idx['out']['u']]
            var_in_qp['y'] = ca.vertcat(x_t,y0_x,y0_u)

            if self._options['mode'] == 'optimize':

                # extract jacobian of qp variables
                if single_model:
                    j_qp_p_t = qp.j_y_p(lam_t,mu_t,p_0,idx_jac(j_x_p_t.reshape((n['x'],n['p']*n_models)),j_y_p_t,0,n_models))
                else:
                    j_qp_p_t = qp.j_y_p(lam_t,mu_t,p_0,idx_jac(j_x_p_t.reshape((n['x'],n['p']*n_models),order='F'),j_y_p_t,0,n_models))

                # extract portion associated to y
                j_y_p_t = j_qp_p_t[qp.idx['out']['y'],:]

                # rearrange appropriately (note that the first entry of
                # y is x0)
                if single_model:
                    j_y_p_t = ca.vertcat(j_x_p_t.reshape((n['x'],n['p']*n_models)),j_y_p_t[qp.idx['out']['x'][:-n['x']],:],j_y_p_t[qp.idx['out']['u'],:])
                else:
                    j_y_p_t = ca.vertcat(j_x_p_t.reshape((n['x'],n['p']*n_models),order='F'),j_y_p_t[qp.idx['out']['x'][:-n['x']],:],j_y_p_t[qp.idx['out']['u'],:])
        else:
            lam_t, mu_t, y_all_t = None, None, None

        # get list of inputs to dynamics and to nominal dynamics
        var_in_fixed = {key:var_in[key] for key in ['x','d'] if var_in[key] is not None}
        var_in_nom_fixed = var_in_fixed if self._options['use_true_model'] else {key:var_in[key] for key in ['x','theta'] if var_in[key] is not None}

        # start counting the time taken to solve the QPs
        total_qp_time = []

        # start counting the time taken to compute the conservative Jacobians
        total_jac_time = []

        # create list to store debug information
        qp_debug = []

        # create list to store qp ingredients
        qp_ingredients = []

        # simulation loop
        for t in range(n['T']):
            
            # parameter to pass to the QP
            p_t = idx_qp(var_in_qp,t)

            # check if warm start should be shifted
            if self._options['warmstart_shift'] and t>0:
                y_all_t = y_all_t[qp.idx['out']['y_shift']]

            # solve QP
            try:
                # start counting time
                qp_time = time.time()
                # solve QP and get solution
                lam_t,mu_t,y_all_t = solver(p_t,y_all_t,lam_t,mu_t)
                # store time
                total_qp_time.append(time.time() - qp_time)
            except:
                print('QP solver failed!')
                qp_failed = True
                break

            # # if QP needs to be checked, compute full conservative Jacobian
            # if self._options['debug_qp']:

            #     # debug current QP
            #     qp_debug_out = QP.debug(lam,mu,p_t,self._options['epsilon'],self._options['roundoff_qp'],y_all)
                
            #     # pack results
            #     qp_debug.append(qp_debug_out)

            # if self._options['compute_qp_ingredients']:

            #     # compute qp ingredients
            #     qp_ingredients.append(qp._qp_sparse(p=p_t))

            # store optimization variables
            sim.add_opt_var(lam_t,mu_t,y_all_t[qp.idx['out']['y']],p_t)

            # get next linearization trajectory
            if self._options['shift_linearization']:
                # shift input trajectory
                x_qp_t = y_all_t[qp.idx['out']['x']]
                u_qp_t = y_all_t[qp.idx['out']['u']]
                var_in_qp['y'] = ca.vertcat(x_qp_t,u_qp_t[n['u']:],u_qp_t[-n['u']:])
            else:
                # do not shift
                var_in_qp['y'] = y_all_t[qp.idx['out']['y']]
            
            if 'eps' in qp.idx['out']:

                # get slack variables
                e_t = y_all_t[qp.idx['out']['eps']]

                # store slack
                sim.e.append(e_t)

            # get first input entry
            u_t = y_all_t[qp.idx['out']['u0']]

            # store input
            sim.u.append(u_t)

            # get current state and input
            current_var = {'x':x_t,'u':u_t}
            
            # update variables
            var_in_dyn = var_in_fixed | current_var
            var_in_dyn_nom = var_in_nom_fixed | current_var

            # check if noise is present
            if var_in['w'] is not None:
                var_in_dyn['w'] = var_in['w'][:,t]

            if self._options['mode'] == 'optimize':
            
                # count time for conservative Jacobians at this time-step
                cons_jac_time = time.time()

                # get conservative jacobian of optimal solution of QP with respect to parameter
                # vector p.
                if single_model:
                    j_qp_p_t = qp.j_y_p(lam_t,mu_t,p_t,idx_jac(j_x_p_t,j_y_p_t,t))
                else:
                    j_qp_p_t = qp.j_y_p(lam_t,mu_t,p_t)@idx_jac(j_x_p_t.reshape((n['x'],n['p']*n_models),order='F'),j_y_p_t,t,multiplier=n_models)

                # select entries associated to y
                if self._options['shift_linearization']:
                    j_x_qp_p_t = j_qp_p_t[qp.idx['out']['x'],:]
                    j_u_qp_p_t = j_qp_p_t[qp.idx['out']['u'],:]
                    j_y_p_t = ca.vertcat(j_x_qp_p_t,j_u_qp_p_t[n['u']:,:],j_u_qp_p_t[-n['u']:,:])
                else:
                    j_y_p_t = j_qp_p_t[qp.idx['out']['y'],:]

                if 'eps' in qp.idx['out']:
                    # select entries associated to slack variables and store them
                    j_eps_p_t = j_qp_p_t[qp.idx['out']['eps'],:]
                    sim.j_eps.append(j_eps_p_t)

                # select rows corresponding to first input u0
                j_u0_p_t = j_qp_p_t[qp.idx['out']['u0'],:]

                if single_model:

                    # propagate jacobian of closed loop state x
                    j_x_p_t = A.call(var_in_dyn_nom)['A']@j_x_p_t + B.call(var_in_dyn_nom)['B']@j_u0_p_t
                    
                    # store conservative jacobians of state and input
                    sim.add_sim_jac(j_x_p_t,j_u0_p_t,j_y_p_t)
                else:
                    j_x_p_t = np.einsum('mnr,ndr->mdr',
                                        np.array(A.call(var_in_dyn_nom)['A']).reshape((n['x'],n['x'],n_models),order='F'),
                                        j_x_p_t) \
                              + np.einsum('mnr,ndr->mdr',
                                          np.array(B.call(var_in_dyn_nom)['B']).reshape((n['x'],n['u'],n_models),order='F'),
                                          np.array(j_u0_p_t).reshape((n['u'],n['p'],n_models),order='F'))
                    
                    # store conservative jacobians of state and input
                    sim.add_sim_jac(j_x_p_t,j_u0_p_t,j_y_p_t)

                # store in total cons jac time
                total_jac_time.append(time.time() - cons_jac_time)

            # get next state
            x_t = f.call(var_in_dyn)['x_next']

            # update qp variable dictionary
            var_in_qp['x'] = x_t

            # store next state
            sim.x.append(x_t)

        # construct output dictionary
        out_dict = {'qp_time':total_qp_time,'jac_time':total_jac_time}

        if self._options['debug_qp']:
            out_dict = out_dict | {'qp_debug':qp_debug}

        if self._options['compute_qp_ingredients']:
            out_dict = out_dict | {'qp_ingredients':qp_ingredients}

        # stack all entries in sim
        sim.stack()

        return sim, out_dict, qp_failed

    def closed_loop(self,init=None,options=None):
        """
        Runs the closed-loop optimization algorithm.

        Args:
            init (dict, optional): Dictionary containing the initial conditions for the simulation. The dictionary can include:
                - x (ndarray): Initial state of the system (required).
                - p (ndarray): Parameters of the system (required if p is a parameter).
                - pf (ndarray): Fixed parameters of the system (required if pf is a parameter).
                - w (ndarray or list): Noise of the system. Can be:
                    - A single vector of shape (n_w, 1), repeated for all time steps.
                    - A matrix of shape (n_w, T), where each column represents noise at a given time step.
                    - A list of matrices of shape (n_w, T), where each element corresponds to noise for a given scenario.
                - d (ndarray): Model uncertainty of the system (required if d is a parameter).
                - y_lin (ndarray): Linearization trajectory of the system.
            options (dict, optional): Dictionary containing configuration options for the algorithm:
                - mode (str): Execution mode. Options are:
                    - 'optimize': Jacobians are computed.
                    - 'simulate': Jacobians are not computed.
                    - 'dense': Dense mode is used, and Jacobians are not computed.
                - shift_linearization : bool, default=True
                    Whether to shift the input-state trajectory used for linearization.
                - warmstart_first_qp (bool, optional, default=True): Whether the first QP should be solved twice (with sensitivity propagation).
                - debug_qp (bool, optional, default=False): Whether to store debug information about the QP.
                - epsilon (float, optional, default=1e-6): Perturbation magnitude for finite difference derivatives of QP.
                - roundoff_qp (int, optional, default=10): Number of digits below which QP derivative error is considered zero.
                - compute_qp_ingredients (bool, optional, default=False): Whether to save QP ingredients.
                - warmstart_shift (bool, optional, default=True): Whether the primal (or primal-dual) warmstart should be shifted.
                - gd_type (str, optional, default='gd'): Type of gradient descent update. Options are:
                    - 'gd': Gradient descent.
                    - 'sgd': Stochastic gradient descent.
                - batch_size (int, optional, default=1): Number of samples in each batch (only applicable if 'gd_type' is 'sgd').
                - figures (bool, optional, default=False): Whether to print debug figures.
                - random_sampling (bool, optional, default=False): Whether to randomly select samples from the dataset in each iteration.
                - verbosity (int, optional, default=1): Level of printout verbosity.
                - max_k (int, optional, default=200): Maximum number of closed-loop iterations.
                - use_true_model (bool, optional, default=True): Whether to use the true model for simulation.
        Returns:
            SIM (list): List of simulation results for each iteration.
            comp_time (dict): Dictionary containing computation times:
                - 'qp': List of QP computation times.
                - 'jac': List of Jacobian computation times.
                - 'iter': List of iteration times.
            p_best (ndarray): Best parameter values found during the optimization process.
        """

        # setup parameters
        p,pf,w,d,theta,y,x = self._get_init_parameters(init)

        # extract number of samples
        n_samples = len(w) if isinstance(w,list) else 1

        # if only one sample is passed, turn certain parameters to length-one lists (for compatibility)
        d, w, theta, x, y = [d], [w], [theta], [x], [y]

        # create running variable. This variable contains the value of each parameter for a single iteration
        # and it gets updated automatically by the sys_id and the parameter_update subroutines.
        running_vars = {'p':p,'pf':pf}

        # update options if provided
        if options is not None:
            self._options.update(options)

        # check if multiple models should be simulated
        if theta is not None and self._options['simulate_parallel_models']:

            # get number of models (columns of theta)
            n_models = int(theta[0].shape[1]) if isinstance(theta,list) else theta.shape[1]

            # check that mapped (nominal) dynamics have been created
            if not hasattr(self,'_mapped'):
                self.create_mapped_dynamics(n_models,self._options['compile_mapped_dynamics'])
            else:
                # if they have been created, check that n_models matches
                assert self._mapped['n_models'] == n_models, 'The mapped dynamics do not have the correct n_models'

        else:
            n_models = 1

        # if more than one model, do not use true model
        if n_models > 1:
            self._options.update({'use_true_model': False})

        # if gradient descent is used, the true number of iterations
        # is equal to max_k times the number of samples
        if self._options['gd_type'] == 'gd':
            batch_size = n_samples
        elif self._options['gd_type'] == 'sgd':
            batch_size = self._options['batch_size'] if 'batch_size' in options else 1

        # check that batch size does not exceed number of samples
        assert batch_size <= n_samples, 'Batch size cannot exceed number of samples.'
        
        # extract maximum number of iterations
        max_k = self._options['max_k'] * batch_size

        # extract cost function
        cost_f = self.upper_level.cost

        # extract gradient of cost function
        j_cost_f = self.upper_level.j_cost if n_models == 1 else self._mapped['j_cost']

        # start empty list
        sim = []

        # start counting time
        total_iter_time = []

        # list containing all QP times
        total_qp_time = []

        # list containing all Jacobian times
        total_jac_time = []

        # if number of iterations is too large, do not store derivatives
        # to save memory
        if max_k > 7500:
            save_memory = True
        else:
            save_memory = False

        # initialize best cost to infinity, and best iteration index to none
        best_cost = ca.inf
        p_best = p

        # check if sys_id should be performed
        if self.upper_level.sys_id_update is not None:
            sys_id = True
            running_vars = running_vars | self.upper_level.sys_id_init()
        else:
            sys_id = False

        # # check if NLP was solved
        # if self.opt['sol']['cost'] is None:
        #     print('Warning: NLP was not solved')

        # # print best cost
        # if self._options['verbosity'] > 0:
        #     cst = self.opt['sol']['cost']
        #     print(f'Best achievable cost: {cst}')

        # if self._options['figures']:
        #     plt.ion()
        #     fig1, ax1 = plt.subplots()
        #     line11, = ax1.plot([], [], 'r')
        #     line21, = ax1.plot([], [], 'b')
        #     fig2, ax2 = plt.subplots()
        #     line12, = ax2.plot([], [], 'r')
        #     line22, = ax2.plot([], [], 'b')
        #     fig3, ax3 = plt.subplots()
        #     line3, = ax3.plot([], [], 'r')

        # preallocate sys_id dictionary
        sys_id_vars = {}

        # outer loop
        for k in range(max_k):
            
            # start counting iteration time
            iter_time = time.time()

            # get index
            idx = randint(0,n_samples-1) if self._options['random_sampling'] else int(ca.fmod(k,batch_size))

            # update running vars with samples for iteration k
            running_vars = running_vars | {'d':d[idx], 'w':w[idx], 'x':x[idx], 'y':y[idx], 'theta':theta[idx]} | sys_id_vars

            # run simulation
            sim_k, qp_data, qp_failed = self._simulate(running_vars,n_models=n_models)

            # compute cost and constraint violation
            cost,track_cost,cst_viol = cost_f(sim_k)
            
            # store S into list
            sim.append(sim_k)

            # if qp failed, terminate
            if qp_failed:
                break

            # store QP and Jacobian times
            total_qp_time.append(qp_data['qp_time'])
            total_jac_time.append(qp_data['jac_time'])

            # store cost and constraint violation
            sim_k.cost = cost
            sim_k.cst = cst_viol

            # if in optimization mode, update parameters
            if self._options['mode'] == 'optimize':

                # if there is no constraint violation, and the cost has improved, save current parameter as best parameter
                if ca.sum1(cst_viol) == 0 and cost < best_cost:
                    best_cost, p_best = cost, p

                # compute gradient of upper-level cost function
                j_p = j_cost_f(sim_k)

                # store in simvar
                sim_k.j_p = j_p

                # on first iteration, initialize psi
                if k == 0:
                    running_vars = running_vars | self.upper_level.parameter_init(sim_k)

                # store psi in simvar
                sim_k.psi = running_vars

                # update parameter
                running_vars = running_vars | self.upper_level.parameter_update(sim_k,k)
                
            else:
                j_p = np.zeros((2,1)) # I need a vector for compatibility with the printout

            # run sys_id if needed
            sys_id_vars = sys_id_vars | self._upper_level.sys_id_update(sim_k,running_vars,k) if sys_id else {}

            if save_memory:
                sim_k.save_memory()

            # printout
            match self._options['verbosity']:
                case 0:
                    pass
                case 1:
                    to_print = f"Iteration: {k}, cost: {track_cost}, J: {ca.DM(np.linalg.norm(j_p,axis=0))}, e : {ca.sum1(ca.fmax(cst_viol,0))}"

                    if sys_id and self.options['true_theta'] is not np.zeros(1):

                        # compute estimation error
                        est_error = np.linalg.norm(running_vars['theta']-self.options['true_theta'])

                        to_print += f', Current estimation error: {est_error}'
                    
                    print(to_print)

            # if self._options['figures']:

            #     line11.set_data(np.linspace(0,S.X[::x['x']].shape[0],S.X[::n['x']].shape[0]),np.array(S.X[::n['x']]))
            #     line21.set_data(np.linspace(0,S.X[1::n['x']].shape[0],S.X[1::n['x']].shape[0]),np.array(S.X[1::n['x']]))
            #     ax1.set_xlim(0,n['N_opt'])
            #     ax1.set_ylim(float(mmin(vertcat(S.X[1::n['x']],S.X[::n['x']]))),float(mmax(vertcat(S.X[1::n['x']],S.X[::n['x']]))))

            #     line12.set_data(np.linspace(0,S.X[2::n['x']].shape[0],S.X[2::n['x']].shape[0]),np.array(S.X[2::n['x']]))
            #     line22.set_data(np.linspace(0,S.X[3::n['x']].shape[0],S.X[3::n['x']].shape[0]),np.array(S.X[3::n['x']]))
            #     ax2.set_xlim(0,n['N_opt'])
            #     ax2.set_ylim(float(mmin(vertcat(S.X[3::n['x']],S.X[2::n['x']]))),float(mmax(vertcat(S.X[3::n['x']],S.X[2::n['x']]))))

            #     line3.set_data(np.linspace(0,S.U.shape[0],S.U.shape[0]),np.array(S.U))
            #     ax3.set_xlim(0,n['N_opt'])
            #     ax3.set_ylim(float(mmin(S.U)),float(mmax(S.U)))

            #     plt.draw()  # Redraw the plot
            #     plt.pause(0.1)

            # get elapsed time
            total_iter_time.append(time.time()-iter_time)

        # if self._options['figures']:
        #     plt.ioff()

        # stack all computation times in one dictionary
        comp_time = dict()
        comp_time['qp'] = total_qp_time
        comp_time['jac'] = total_jac_time
        comp_time['iter'] = total_iter_time

        return sim, comp_time, p_best