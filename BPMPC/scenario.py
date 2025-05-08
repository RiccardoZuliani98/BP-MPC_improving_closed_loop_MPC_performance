import casadi as ca
from BPMPC.dynamics import Dynamics
from BPMPC.QP import QP
from BPMPC.UpperLevel import UpperLevel
from BPMPC.simVar import simVar
import time
from numpy.random import randint
from typeguard import typechecked
from BPMPC.options import Options
from BPMPC.symb import Symb

"""
TODO:
* descriptions
* trajectory optimization should be a separate class!
* for now symb does not allow to have lists as init, should I change that?
"""

class Scenario:

    _OPTIONS_ALLOWED_VALUES = {'shift_linearization': bool, 'warmstart_first_qp': bool, 'warmstart_shift': bool,
                               'epsilon': float, 'roundoff_qp': int, 'mode': ['optimize', 'simulate', 'dense'],
                               'gd_type': ['gd', 'sgd'], 'figures': bool, 'random_sampling': bool, 'debug_qp': bool,
                               'compute_qp_ingredients': bool, 'verbosity': [0, 1, 2], 'max_k': int}

    _OPTIONS_DEFAULT_VALUES = {'shift_linearization': True, 'warmstart_first_qp': True, 'warmstart_shift': True,
                               'epsilon': 1e-6, 'roundoff_qp': 10, 'mode': 'optimize', 'gd_type': 'gd',
                               'figures': False, 'random_sampling': False, 'debug_qp': False,
                               'compute_qp_ingredients': False, 'verbosity': 1, 'max_k': 200}

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
    
    def make_trajectory_opt(self):
  
        # extract system dynamics
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
        init_values_not_none = {k:v for k,v in init_values.items() if v is not None}

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
        p = init_values['p'] if 'p' in init_values else None            # parameters
        pf = init_values['pf'] if 'pf' in init_values else None         # fixed parameters
        w = init_values['w'] if 'w' in init_values else None            # noise
        d = init_values['d'] if 'd' in init_values else None            # model uncertainty
        y = init_values['y_lin'] if 'y_lin' in init_values else None    # linearization trajectory
        x = init_values['x']                                            # initial state

        return p,pf,w,d,y,x

    def simulate(self,init=None,options=None):

        # get initial parameters
        p,pf,w,d,y,x = self._get_init_parameters(init)

        # simulate
        s, out_dict, qp_failed = self._simulate(p,pf,w,d,y,x,options)

        return s, out_dict, qp_failed

    def _simulate(self,p,pf,w,d,y,x,options=None):

        """
        Low-level simulate function, unlike simulate, this needs the inputs to be passed separately (as
        returned by _get_init_parameters).
        """

        # extract QP for simplicity
        qp = self.qp

        # extract dimensions for simplicity
        n = self.dim

        # update options if provided
        if options is not None:
            self._options.update(options)

        # check if w is None
        if w is None:
            w = [None] * n['T']

        # flag to check if QP failed
        qp_failed = False

        # extract dynamics and linearization
        A = self.dyn.A
        B = self.dyn.B
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

            # if QP needs to be checked, compute full conservative Jacobian
            if self._options['debug_qp']:

                # debug current QP
                qp_debug_out = QP.debug(lam,mu,p_t,self._options['epsilon'],self._options['roundoff_qp'],y_all)
                
                # pack results
                qp_debug.append(qp_debug_out)

            if self._options['compute_qp_ingredients']:

                # compute qp ingredients
                qp_ingredients.append(qp._qp_sparse(p=p_t))

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

            # get list of inputs to dynamics
            var_in = [x,u,d,w[t]]
            var_in = [var for var in var_in if var is not None]

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
                j_x_p = A(*var_in)@j_x_p + B(*var_in)@j_u0_p

                # store in total cons jac time
                total_jac_time.append(time.time() - cons_jac_time)

                # store conservative jacobians of state and input
                S.setJx(t+1,j_x_p)
                S.setJu(t,j_u0_p)
                S.setJy(t,j_y_p)

            # get next state
            x = f(*var_in)

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
        This function runs the closed-loop optimization algorithm. The inputs are

            - init: dictionary containing the initial conditions for the simulation. The dictionary
                    can contain the following keys:

                    - x: initial state of the system (required)
                    - p: parameters of the system (required if p is a parameter)
                    - pf: fixed parameters of the system (required if pf is a parameter)
                    - w: noise of the system (required if w is a parameter)
                    - d: model uncertainty of the system (required if d is a parameter)
                    - y_lin: linearization trajectory of the system
                
                    Note that w should be passed either as single vector of dimension (n_w,1), which
                    will be repeated for all time steps, as a matrix of dimension (n_w,T), where
                    each column represents the noise at a given time step, or as a list of matrices
                    of dimension (n_w,T), where each element of the list is the noise at each time 
                    step for a given scenario.

            - options: dictionary containing the following keys:

                    - mode: 'optimize' (jacobians are computed) or 'simulate' (jacobians are not computed) 
                            or 'dense' (dense mode is used and jacobians are not computed)
                    - shift_linearization: True (default) if the input-state trajectory used for 
                                           linearization should be shifted, False otherwise
                    - warmstart_first_qp: True (default) if the first QP should be solved twice (with
                                          propagation of the sensitivity)
                    - debug_qp: False (default), or True if debug information about the QP should be stored
                    - epsilon: perturbation magnitude used to compute finite difference derivatives of QP,
                               default is 1e-6
                    - roundoff_qp: number of digits below which QP derivative error is considered zero,
                                   default is 10
                    - compute_qp_ingredients:False (default), or True if QP ingredients should be saved
                    - warmstart_shift: True (default) if the primal (or primal-dual) warmstart should be shifted
                    - gd_type: type of update (options are 'sgd' and 'gd', default is 'gd')
                    - batch_size: number of samples in each batch (default is 1, only if 'gd_type' is 'sgd')
                    - figures: printout of debug figures (default is False)
                    - random_sampling: if True then samples are randomly selected from the dataset in each iteration
                    - verbosity: level of printout (default is 1)
                    - max_k: number of closed-loop iterations (default is 200)
        """

        # setup parameters
        p,pf,W,D,Y,X = self._get_init_parameters(init)

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
                x = X[idx]
                y = Y[idx]
            else:
                d = D
                w = W
                x = X
                y = Y

            # run simulation
            S, qp_data, qp_failed = self._simulate(p,pf,w,d,y,x)
            
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
                    psi = psi_init(p,J_p,pf)

                if ca.fmod(k+1,batch_size) == 0:

                    # update parameter
                    p = p_next(p,psi,k,J_p_full,pf)
                    psi = psi_next(p,psi,k,J_p_full,pf)

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