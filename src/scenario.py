import casadi as ca
from src.dynamics import Dynamics
from src.qp import QP
from src.upper_level import UpperLevel
from src.simVar import simVar
import time
from numpy.random import randint
from typeguard import typechecked
from src.options import Options
from src.symb import Symb

"""
TODO:
* descriptions
* trajectory optimization should be a separate class!
"""

class Scenario:

    _OPTIONS_ALLOWED_VALUES = {'shift_linearization': bool, 'warmstart_first_qp': bool, 'warmstart_shift': bool,
                               'epsilon': float, 'roundoff_qp': int, 'mode': ['optimize', 'simulate', 'dense'],
                               'gd_type': ['gd', 'sgd'], 'figures': bool, 'random_sampling': bool, 'debug_qp': bool,
                               'compute_qp_ingredients': bool, 'verbosity': [0, 1, 2], 'max_k': int,
                               'use_true_model': bool}

    _OPTIONS_DEFAULT_VALUES = {'shift_linearization': True, 'warmstart_first_qp': True, 'warmstart_shift': True,
                               'epsilon': 1e-6, 'roundoff_qp': 10, 'mode': 'optimize', 'gd_type': 'gd',
                               'figures': False, 'random_sampling': False, 'debug_qp': False,
                               'compute_qp_ingredients': False, 'verbosity': 1, 'max_k': 200,
                               'use_true_model': True}

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
        current_sym = self._sym if hasattr(self,'_sym') else Symb()

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

    ### NONLINEAR SOLVER FOR TRAJECTORY OPT PROBLEM ---------------------------

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
        S = simVar(n)

        # add optimization variables (p and y are set to zero by default)
        S._x = x
        S._u = u

        # now get cost as a symbolic function of x and u
        _,cost,cst = self.upper_level.cost(S)
        
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
            out._simVar__x = ca.DM(opti.value(ca.vec(x)))
            out._simVar__u = ca.DM(opti.value(ca.vec(u)))
            out._simVar__cost = ca.DM(opti.value(cost))

            return out,solved

        return solver


    ### SIMULATION FUNCTIONS ---------------------------------------------------

    def _get_init_parameters(self,init=None):

        if init is not None:
            self.set_init(init)

        # get initialization
        init_values = self.init

        # first check if at least one of the parameters is a list
        lengths = [len(v) if isinstance(v,list) else 1 for v in init_values.values()]
        
        # if there are multiple nonzero lengths, check that they match
        assert len(set([item for item in lengths if item != 1])) <= 1, 'All parameters must have the same length.'
        
        # get final length
        max_length = max(lengths)

        # if w is passed as a single vector (and it is not a list or None), repeat it
        if (init_values['w'] is not None) and (not isinstance(init_values['w'],list)) and (init_values['w'].shape[1] == 1):
            init_values['w'] = ca.repmat(init_values['w'],1,self.dim['T'])

        # check dimension of w
        if init_values['w'] is not None:

            # if w is a list, check that all elements have the same number of columns
            if isinstance(init_values['w'],list):
                if len(set([v.shape[1] for v in init_values['w']])) > 1:
                    raise Exception('All noise w must have the same number of columns.')
                if init_values['w'][0].shape[1] != self.dim['T']:
                    raise Exception('Noise w must have the same number of columns as the prediction horizon.')
            
            # otherwise, check that w has the same number of columns as the prediction horizon
            elif init_values['w'].shape[1] != self.dim['T']:
                raise Exception('Noise w must have the same number of columns as the prediction horizon.')

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

        # now "init_values" contains x,u,w,d as lists of the same length if max_length > 1,
        # otherwise they are all vectors. Moreover, p,pf,y_lin are always vectors.
        # Note that any one of these variables may also be None if it was not passed.

        # under the "trajectory" linearization mode, we need y_lin to be a trajectory
        if self.options['linearization'] == 'trajectory':
            
            # if adaptive mode is used, copy x and u to create y_lin
            if (init_values['y_lin'] is None) or (init_values['y_lin']=='adaptive'):
            
                # if u was not passed return an error
                if init_values['u'] is None:
                    raise Exception('Either pass an input or a linearization trajectory.')
                
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
            if init_values['y_lin'] is None:
                
                # if so, check if an input is passed
                if init_values['u'] is None:

                    # if none is passed, raise an exception
                    raise Exception('Either pass an input or a linearization trajectory.')
                
                # set equal to input (note that input is already either a list or a vector)
                init_values['y_lin'] = init_values['u']
                    
        # construct a dictionary to check the dimension. You need to concatenate horizontally
        # any list within init_values and leave all vectors as they are
        # init_values_not_none = {k:v for k,v in init_values.items() if v is not None}

        # init_values_concat = {k:ca.hcat(v) if isinstance(v,list) else v for k,v in init_values_not_none.items()}
        # _ = self.__checkInit(init_values_concat)

        # initial condition
        assert init_values['x'] is not None, 'Initial state x is required to simulate the system.'
        
        if 'p' in self.param and init_values['p'] is None:
            raise Exception('Parameters p are required to simulate the system.')
        
        if 'pf' in self.param and init_values['pf'] is None:
            raise Exception('Fixed parameters pf are required to simulate the system.')
        
        if 'w' in self.param and init_values['w'] is None:
            raise Exception('Noise w is required to simulate the system.')
        
        if 'd' in self.param and init_values['d'] is None:
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

        # simulate
        s, out_dict, qp_failed = self._simulate(p,pf,w,d,theta,y,x)

        return s, out_dict, qp_failed

    def _simulate(self,p,pf,w,d,theta,y,x):
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

        # extract QP for simplicity
        qp = self.qp

        # extract dimensions for simplicity
        n = self.dim

        # check if w is None
        if w is None:
            w = [None] * n['T']

        # flag to check if QP failed
        qp_failed = False

        # extract dynamics and linearization
        A = self.dyn.A if self.options['use_true_model'] else self.dyn.A_nom
        B = self.dyn.B if self.options['use_true_model'] else self.dyn.B_nom
        f = self.dyn.f

        # create simVar for current simulation
        S = simVar(n)

        # store p and pf if present
        if p is not None:
            S.p = p
        if pf is not None:
            S.pf = pf

        # set initial condition
        S.setState(0,x)

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
            j_x_p = ca.DM(n['x'],n['p'])
            j_y_p = ca.DM(n['y'],n['p'])
            S.setJx(0,j_x_p)
            # S.setJy(0,J_y_p)

        # check if QP warmstart was passed
        if self._options['warmstart_first_qp']:

            # get qp parameter
            p_0 = idx_qp(x,y,p,pf,0)

            # run QP once to get better initialization
            lam,mu,y_all = qp.solve(p_0)

            # update y0
            y0_x = y_all[qp.idx['out']['x'][:-n['x']]]
            y0_u = y_all[qp.idx['out']['u']]
            y = ca.vertcat(self.init['x'],y0_x,y0_u)

            if self._options['mode'] == 'optimize':

                # extract jacobian of qp variables
                j_qp_p = qp.J_y_p(lam,mu,p_0,idx_jac(j_x_p,j_y_p,0))

                # extract portion associated to y
                j_y_p = j_qp_p[qp.idx['out']['y'],:]

                # rearrange appropriately (note that the first entry of
                # y is x0)
                j_y_p = ca.vertcat(j_x_p,j_y_p[qp.idx['out']['x'][:-n['x']],:],j_y_p[qp.idx['out']['u'],:])
        else:
            lam = None
            mu = None
            y_all = None

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
            y_lin = y

            # parameter to pass to the QP
            p_t = idx_qp(x,y_lin,p,pf,t)

            # check if warm start should be shifted
            if self._options['warmstart_shift']:
                if t > 0:
                    y_all = y_all[qp.idx['out']['y_shift']]

            # solve QP
            try:
                # start counting time
                qp_time = time.time()
                # solve QP and get solution
                lam,mu,y_all = solver(p_t,y_all,lam,mu)
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
            S.setOptVar(t,lam,mu,y_all[qp.idx['out']['y']],p_t)

            # get next linearization trajectory
            if self._options['shift_linearization']:
                # shift input trajectory
                x_qp = y_all[qp.idx['out']['x']]
                u_qp = y_all[qp.idx['out']['u']]
                y = ca.vertcat(x_qp,u_qp[n['u']:],u_qp[-n['u']:])
            else:
                # do not shift
                y = y_all[qp.idx['out']['y']]
            
            if 'eps' in qp.idx['out']:

                # get slack variables
                e = y_all[qp.idx['out']['eps']]

                # store slack
                S.setSlack(t,e)

            # get first input entry
            u = y_all[qp.idx['out']['u0']]

            # store input
            S.setInput(t,u)

            # get current state and input
            current_var = {'x':x,'u':u}
            
            # update variables
            var_in = var_in_fixed | current_var
            var_in_nom = var_in_nom_fixed | current_var

            # check if noise is present
            if w is not None:
                var_in['w'] = w[t]

            if self._options['mode'] == 'optimize':
            
                # count time for conservative Jacobians at this time-step
                cons_jac_time = time.time()

                # get conservative jacobian of optimal solution of QP with respect to parameter
                # vector p.
                j_qp_p = qp.J_y_p(lam,mu,p_t,idx_jac(j_x_p,j_y_p,t))

                # select entries associated to y
                if self._options['shift_linearization']:
                    j_x_qp_p = j_qp_p[qp.idx['out']['x'],:]
                    j_u_qp_p = j_qp_p[qp.idx['out']['u'],:]
                    j_y_p = ca.vertcat(j_x_qp_p,j_u_qp_p[n['u']:,:],j_u_qp_p[-n['u']:,:])
                else:
                    j_y_p = j_qp_p[qp.idx['out']['y'],:]

                if 'eps' in qp.idx['out']:
                    # select entries associated to slack variables and store them
                    j_eps_p = j_qp_p[qp.idx['out']['eps'],:]
                    S.setJeps(t,j_eps_p)

                # select rows corresponding to first input u0
                j_u0_p = j_qp_p[qp.idx['out']['u0'],:]

                # propagate jacobian of closed loop state x
                j_x_p = A.call(var_in_nom)['A']@j_x_p + B.call(var_in_nom)['B']@j_u0_p

                # store in total cons jac time
                total_jac_time.append(time.time() - cons_jac_time)

                # store conservative jacobians of state and input
                S.setJx(t+1,j_x_p)
                S.setJu(t,j_u0_p)
                S.setJy(t,j_y_p)

            # get next state
            x = f.call(var_in)['x_next']

            # store next state
            S.setState(t+1,x)

        # construct output dictionary
        out_dict = {'qp_time':total_qp_time,'jac_time':total_jac_time}

        if self._options['debug_qp']:
            out_dict = out_dict | {'qp_debug':qp_debug}

        if self._options['compute_qp_ingredients']:
            out_dict = out_dict | {'qp_ingredients':qp_ingredients}

        return S, out_dict, qp_failed

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
        p,pf,W,D,THETA,Y,X = self._get_init_parameters(init)

        # extract number of samples
        n_samples = len(W) if isinstance(W,list) else 1

        # update options if provided
        if options is not None:
            self._options.update(options)

        # store dim in a variable with a shorter name
        n = self.dim

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
        J_cost_f = self.upper_level.j_cost

        if self._options['mode'] == 'optimize':

            # extract parameter update law
            alg = self.upper_level.alg
            p_next = alg['p_next']
            psi_next = alg['psi_next']
            psi_init = alg['psi_init']

        # start empty list
        SIM = []

        # # check if NLP was solved
        # if self.opt['sol']['cost'] is None:
        #     print('Warning: NLP was not solved')

        # # print best cost
        # if self._options['verbosity'] > 0:
        #     cst = self.opt['sol']['cost']
        #     print(f'Best achievable cost: {cst}')

        # start counting time
        total_iter_time = []

        # list containing all QP times
        total_qp_time = []

        # list containing all Jacobian times
        total_jac_time = []

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

        # if number of iterations is too large, do not store derivatives
        # to save memory
        if max_k > 7500:
            save_memory = True
        else:
            save_memory = False

        # initialize best cost to infinity, and best iteration index to none
        best_cost = ca.inf
        p_best = p

        # initialize full gradient of minibatch
        J_p_full = ca.DM(*p.shape)

        # outer loop
        for k in range(max_k):
            
            # start counting iteration time
            iter_time = time.time()

            # sample uncertain elements
            if n_samples > 1:
                if self._options['random_sampling']:
                    idx = randint(0,n_samples)
                else:
                    idx = int(ca.fmod(k,batch_size))
                d = D[idx]
                w = W[idx]
                theta = THETA[idx]
                x = X[idx]
                y = Y[idx]
            else:
                d = D
                w = W
                theta = THETA
                x = X
                y = Y

            # run simulation
            S, qp_data, qp_failed = self._simulate(p,pf,w,d,theta,y,x)
            
            # store S into list
            SIM.append(S)

            # if qp failed, terminate
            if qp_failed:
                break

            # store QP and Jacobian times
            total_qp_time.append(qp_data['qp_time'])
            total_jac_time.append(qp_data['jac_time'])

            # compute cost and constraint violation
            cost,track_cost,cst_viol = cost_f(S)

            # store them
            S.cost = cost
            S.cst = cst_viol

            # if in optimization mode, update parameters
            if self._options['mode'] == 'optimize':

                # if there is no constraint violation, and the cost has improved, save current parameter as best parameter
                if ca.sum1(cst_viol) == 0 and cost < best_cost:
                    best_cost = cost
                    p_best = p

                # compute gradient of upper-level cost function
                J_p = J_cost_f(S)

                # store in simvar
                S.Jp = J_p

                # update gradient of minibatch
                J_p_full = J_p_full + J_p

                # on first iteration, initialize psi
                if k == 0:

                    # initialize parameter
                    psi = psi_init(p,pf,J_p)

                if ca.fmod(k+1,batch_size) == 0:

                    # update parameter
                    p = p_next(p,pf,psi,k,J_p_full)
                    psi = psi_next(p,pf,psi,k,J_p_full)

                    # reset full gradient
                    J_p_full = ca.DM(*p.shape)
                
            else:
                J_p = 0

            if save_memory:
                S.saveMemory()

            # printout
            match self._options['verbosity']:
                case 0:
                    pass
                case 1:
                    print(f"Iteration: {k}, cost: {track_cost}, J: {ca.norm_2(J_p)}, e : {ca.sum1(ca.fmax(cst_viol,0))}")#, slacks: {slack} ")

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

        return SIM, comp_time, p_best