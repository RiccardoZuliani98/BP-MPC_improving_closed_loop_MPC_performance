import casadi as ca
from src.dynamics import Dynamics
from src.qp import QP
from src.upper_level import UpperLevel
from src.sim_var import simVar
import time
from numpy.random import randint
from typeguard import typechecked
from src.options import Options
from src.symbolic_var import SymbolicVar
import numpy as np

"""
TODO:
* trajectory optimization should be a separate class!
"""

class Scenario:

    _OPTIONS_ALLOWED_VALUES = {'shift_linearization': bool, 'warmstart_first_qp': bool, 'warmstart_shift': bool,
                               'epsilon': float, 'roundoff_qp': int, 'mode': ['optimize', 'simulate', 'dense'],
                               'gd_type': ['gd', 'sgd'], 'figures': bool, 'random_sampling': bool, 'debug_qp': bool,
                               'compute_qp_ingredients': bool, 'verbosity': [0, 1, 2], 'max_k': int,
                               'use_true_model': bool, 'simulate_parallel_models': bool,
                               'compile_mapped_dynamics':bool}

    _OPTIONS_DEFAULT_VALUES = {'shift_linearization': True, 'warmstart_first_qp': True, 'warmstart_shift': True,
                               'epsilon': 1e-6, 'roundoff_qp': 10, 'mode': 'optimize', 'gd_type': 'gd',
                               'figures': False, 'random_sampling': False, 'debug_qp': False,
                               'compute_qp_ingredients': False, 'verbosity': 1, 'max_k': 200,
                               'use_true_model': True, 'simulate_parallel_models': False,
                               'compile_mapped_dynamics':False}

    @typechecked
    def __init__(self,dyn:Dynamics,mpc:QP,upper_level:UpperLevel):
        self.update(dyn=dyn,qp=mpc,upper_level=upper_level)

    def update(self,**kwargs):

        # initialize properties
        for key, value in kwargs.items():
            assert key in ['dyn','qp','upper_level'], 'Wrong key value in update.'
            if value is not None:
                setattr(self, f"_{key}", value)

        # check if class already possesses symbols
        current_sym = self._sym if hasattr(self,'_sym') else SymbolicVar()

        # check if class already possesses options
        current_options = self._options if hasattr(self,'_options') else Options(self._OPTIONS_ALLOWED_VALUES, self._OPTIONS_DEFAULT_VALUES)

        # create symbols
        self._sym = self._dyn._sym + self._qp._sym + self._upper_level._sym + current_sym

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
  
        # extract system dynamics
        if theta is not None:
            f = lambda state,input: self.dyn.f_nom(state,input,theta)
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
        s = simVar(n)

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
            out = simVar(n)
            out.x = ca.DM(np.atleast_2d(opti.value(x)))
            out.u = ca.DM(np.atleast_2d(opti.value(u)))
            out.cost = ca.DM(opti.value(cost))

            return out,solved

        return solver

    def _get_init_parameters(self,init=None):
        """
        Processes and validates the initialization parameters for the system simulation.
        Args:
            init (dict, optional): A dictionary containing initialization parameters. 
                If provided, it will be used to set the initial values for the system.
        Returns:
            tuple: A tuple containing the following elements:
                - p (array or None): Parameters for the system, if provided.
                - pf (array or None): Fixed parameters for the system, if provided.
                - w (array, list, or None): Noise values for the system, if provided.
                - d (array, list, or None): Model uncertainty values, if provided.
                - theta (array or None): Nominal model parameters, if provided.
                - y (array, list, or None): Linearization trajectory, if provided or computed.
                - x (array): Initial state of the system (required).
        Raises:
            AssertionError: If the lengths of initialization parameters are inconsistent.
            Exception: If required parameters (e.g., `x`, `p`, `pf`, `w`, `d`) are missing.
            Exception: If noise `w` dimensions do not match the prediction horizon.
            Exception: If `y_lin` is required but not provided or cannot be computed.
            Exception: If the "optimal" linearization trajectory mode is selected (not implemented).
        Notes:
            - If any parameter in the "dynamics" subclass is not a list and `max_length > 1`, 
                it will be extended to a list of appropriate length.
            - The function supports two linearization modes: "trajectory" and "initial_state".
            - Under the "trajectory" mode, `y_lin` is computed based on `x` and `u` if not provided.
            - Under the "initial_state" mode, `y_lin` defaults to `u` if not provided.
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
                init_values['y_lin'] = init_values['u']

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

    def create_mapped_dynamics(self,n_models,jit=False):

        if jit:
            self._options.update({'compile_mapped_dynamics' : True})

        # check if dynamics should be compiled
        if self._options['compile_mapped_dynamics']:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            compilation_options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            compilation_options = {}

        # create mapped dynamics
        self._mapped = {'n_models': n_models}
        self._mapped['A'] = self.dyn.A_nom.map(n_models, [True, True, False], [False], compilation_options)
        self._mapped['B'] = self.dyn.B_nom.map(n_models, [True, True, False], [False], compilation_options)

        # create mapped cost jacobian
        j_cost_func_temp_mapped = self.upper_level._j_cost_func_temp.map(n_models, [True, False, False, False], [False], compilation_options)

        def j_cost_func(s):

            # get true input cost
            cost_in_loc = self.upper_level._get_cost_idx(s.x,s.u,s.y,s.p)
            # cost_in = getCostIdx(S.x,S.u,S.y,S.p[:,-1])

            # get true Jacobian
            j_x,j_u,j_y = self.upper_level._get_cost_jacobian(s.j_x,s.j_u,s.j_y)

            return j_cost_func_temp_mapped(cost_in_loc,j_x,j_u,j_y)

        self._mapped['j_cost'] = j_cost_func

    def simulate(self,init=None,options=None):
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
            if not hasattr(self,'_mapped'):
                self.create_mapped_dynamics(n_models,self._options['compile_mapped_dynamics'])
            else:
                # if they have been created, check that n_models matches
                assert self._mapped['n_models'] == n_models, 'The mapped dynamics do not have the correct n_models'

        else:
            n_models = 1

        # if more than one model, do not use true model
        if n_models > 1:
            self._options.update({'use_true_model':False})

        # simulate
        s, out_dict, qp_failed = self._simulate(p,pf,w,d,theta,y,x,n_models)

        return s, out_dict, qp_failed

    def _simulate(self,p:ca.DM,pf:ca.DM,w:ca.DM,d:ca.DM,theta:ca.DM,y:ca.DM,x:ca.DM,n_models:int=1) -> simVar | dict | bool:
        """
        Simulates the system dynamics and solves a sequence of quadratic programs (QPs) 
        for a given set of parameters, initial conditions, and disturbances.
        This is a low-level simulation function that requires inputs to be passed 
        separately, as returned by `_get_init_parameters`.
        Args:
            p (ca.DM or None): Parameter vector for the simulation.
            pf (ca.DM or None): Final parameter vector for the simulation.
            w (list or None): List of disturbances for each time step. If None, 
                disturbances are assumed to be zero.
            d (ca.DM): Known disturbances for the system dynamics.
            y (ca.DM): Initial guess for the optimization variables.
            x (ca.DM): Initial state of the system.
        Returns:
            tuple:
                - S (simVar): Object containing simulation variables, including 
                  states, inputs, and optimization variables.
                - out_dict (dict): Dictionary containing timing information and 
                  optional debug information:
                    - 'qp_time': List of times taken to solve each QP.
                    - 'jac_time': List of times taken to compute conservative Jacobians.
                    - 'qp_debug' (optional): Debug information for QPs, if enabled.
                    - 'qp_ingredients' (optional): QP ingredients, if enabled.
                - qp_failed (bool): Flag indicating whether any QP solver failed 
                  during the simulation.
        Raises:
            Exception: If the QP solver fails at any time step.
        Notes:
            - The function supports different modes of operation, including 'dense', 
              'sparse', and 'optimize', which affect the solver and Jacobian computation.
            - Warm-starting and shifting of QP solutions are supported for improved 
              performance.
            - Debugging and computation of QP ingredients can be enabled via options.
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
        sim = simVar(n,n_models)

        # store p and pf if present
        sim.p = p if p is not None else None
        sim.pf = pf if pf is not None else None

        # set initial conditions
        sim.x.append(x)
        x_t, y_t = x, y

        # extract parameter indexing
        idx_qp = self.upper_level.idx['qp']
        idx_jac = self.upper_level.idx['jac']

        # extract solver
        if self._options['mode'] == 'dense':

            # if in dense mode, choose dense solver
            solver = qp._dense_solve
        else:

            # otherwise, choose sparse solver
            solver = qp.solve
        
        # in optimize mode, initialize Jacobians
        if self._options['mode'] == 'optimize':
            # initialize Jacobians
            j_x_p_t = ca.DM(n['x'],n['p']*n_models) if single_model else np.zeros((n['x'],n['p'],n_models))
            j_y_p_t = ca.DM(n['y'],n['p']*n_models)
            sim.j_x.append(j_x_p_t)

        # check if QP warmstart was passed
        if self._options['warmstart_first_qp']:

            # get qp parameter
            p_0 = idx_qp(x_t,y_t,p,pf,0)

            # run QP once to get better initialization
            lam_t,mu_t,y_all_t = qp.solve(p_0)

            # update y0
            y0_x = y_all_t[qp.idx['out']['x'][:-n['x']]]
            y0_u = y_all_t[qp.idx['out']['u']]
            y_t = ca.vertcat(x,y0_x,y0_u)

            if self._options['mode'] == 'optimize':

                # extract jacobian of qp variables
                j_qp_p_t = qp.J_y_p(lam_t,mu_t,p_0,idx_jac(j_x_p_t.reshape((n['x'],n['p']*n_models)),j_y_p_t,0,n_models))

                # extract portion associated to y
                j_y_p_t = j_qp_p_t[qp.idx['out']['y'],:]

                # rearrange appropriately (note that the first entry of
                # y is x0)
                j_y_p_t = ca.vertcat(j_x_p_t.reshape((n['x'],n['p']*n_models)),j_y_p_t[qp.idx['out']['x'][:-n['x']],:],j_y_p_t[qp.idx['out']['u'],:])
        else:
            lam_t = None
            mu_t = None
            y_all_t = None

        # get list of inputs to dynamics and to nominal dynamics
        var_in_fixed = {'d':d} if d is not None else {}
        if self._options['use_true_model']:
            var_in_nom_fixed = var_in_fixed
        else:
            var_in_nom_fixed = {'theta':theta} if theta is not None else {}

        # start counting the time taken to solve the QPs
        total_qp_time = []

        # start counting the time taken to compute the conservative Jacobians
        total_jac_time = []

        # create list to store debug information
        if self._options['debug_qp']:
            qp_debug = []

        # create list to store qp ingredients
        if self._options['compute_qp_ingredients']:
            qp_ingredients = []

        # simulation loop
        for t in range(n['T']):
            
            # replace first entry of state with current state
            y_lin_t = y_t

            # parameter to pass to the QP
            p_t = idx_qp(x_t,y_lin_t,p,pf,t)

            # check if warm start should be shifted
            if self._options['warmstart_shift']:
                if t > 0:
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
                y_t = ca.vertcat(x_qp_t,u_qp_t[n['u']:],u_qp_t[-n['u']:])
            else:
                # do not shift
                y_t = y_all_t[qp.idx['out']['y']]
            
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
            var_in = var_in_fixed | current_var
            var_in_nom = var_in_nom_fixed | current_var

            # check if noise is present
            if w is not None:
                var_in['w'] = w[:,t]

            if self._options['mode'] == 'optimize':
            
                # count time for conservative Jacobians at this time-step
                cons_jac_time = time.time()

                # get conservative jacobian of optimal solution of QP with respect to parameter
                # vector p.
                if single_model:
                    j_qp_p_t = qp.J_y_p(lam_t,mu_t,p_t,idx_jac(j_x_p_t,j_y_p_t,t))
                else:
                    j_qp_p_t = qp.J_y_p(lam_t,mu_t,p_t)@idx_jac(j_x_p_t.reshape((n['x'],n['p']*n_models)),j_y_p_t.reshape((n['y'],n['p']*n_models)),t,multiplier=n_models)

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
                    j_x_p_t = A.call(var_in_nom)['A']@j_x_p_t + B.call(var_in_nom)['B']@j_u0_p_t
                    
                    # store conservative jacobians of state and input
                    sim.add_sim_jac(j_x_p_t,j_u0_p_t,j_y_p_t)
                else:
                    j_x_p_t = np.einsum('mnr,ndr->mdr',
                                        np.array(A.call(var_in_nom)['A']).reshape((n['x'],n['x'],n_models)),
                                        np.array(j_x_p_t).reshape((n['x'],n['p'],n_models))) \
                              + np.einsum('ijk,ljk->ilk',
                                          np.array(B.call(var_in_nom)['B']).reshape((n['x'],n['u'],n_models)),
                                          np.array(j_u0_p_t).reshape((n['u'],n['p'],n_models)))
                    
                    # store conservative jacobians of state and input
                    sim.add_sim_jac(j_x_p_t,j_u0_p_t,j_y_p_t)

                # store in total cons jac time
                total_jac_time.append(time.time() - cons_jac_time)

            # get next state
            x_t = f.call(var_in)['x_next']

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

        if self._options['mode'] == 'optimize':

            # extract parameter update law
            parameter_init = self.upper_level.parameter_init
            parameter_update = self.upper_level.parameter_update

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

        # outer loop
        for k in range(max_k):
            
            # start counting iteration time
            iter_time = time.time()

            # get index
            idx = randint(0,n_samples-1) if self._options['random_sampling'] else int(ca.fmod(k,batch_size))

            # obain elements
            d_k, w_k, theta_k, x_k, y_k = d[idx], w[idx], theta[idx], x[idx], y[idx]

            # run simulation
            sim_k, qp_data, qp_failed = self._simulate(p,pf,w_k,d_k,theta_k,y_k,x_k,n_models=n_models)

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
                    psi = parameter_init(sim_k)

                # store psi in simvar
                sim_k.psi = psi

                # update parameter
                p,psi = parameter_update(sim_k,k)
                
            else:
                j_p = np.zeros((2,1)) # I need a vector for compatibility with the printout

            if save_memory:
                sim_k.save_memory()

            # printout
            match self._options['verbosity']:
                case 0:
                    pass
                case 1:
                    print(f"Iteration: {k}, cost: {track_cost}, J: {ca.DM(np.linalg.norm(j_p,axis=0))}, e : {ca.sum1(ca.fmax(cst_viol,0))}")#, slacks: {slack} ")

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