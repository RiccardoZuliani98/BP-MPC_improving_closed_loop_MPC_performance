from casadi import *
from BPMPC.dynamics import Dynamics
from BPMPC.QP import QP
from BPMPC.upperLevel import UpperLevel
from BPMPC.simVar import simVar
import time
from numpy.random import randint

"""
TODO:
* descriptions
"""

class Scenario:

    def __init__(self,MSX='SX'):

        # check type of symbolic variables
        if MSX == 'SX':
            self.__MSX = SX
        elif MSX == 'MX':
            self.__MSX = MX
        else:
            raise Exception('MSX must be either SX or MX.')

        # initialize properties
        self.__dim = {}
        self.__dyn = dynamics(MSX)
        self.__QP = QP(MSX)
        self.__upperLevel = upperLevel(MSX)
        self.__compTimes = {}

        # default options
        defaultClosedLoop = {'mode':'optimize','gd_type':'gd','figures':False,'random_sampling':False,'debug_qp':False,'compute_qp_ingredients':False,'verbosity':1,'max_k':200}
        defaultSimulate = {'mode':'optimize','shift_linearization':True,'warmstart_first_qp':True,'debug_qp':False,'compute_qp_ingredients':False,'warmstart_shift':True,'epsilon':1e-6,'roundoff_qp':10}
        self.__default_options = {'closedLoop':defaultClosedLoop,'simulate':defaultSimulate}

        pass

    @property
    def compTimes(self):
        return self.__compTimes
    
    @property
    def dim(self):
        return self.__dim
    
    def __addDim(self, dim):
        self.__dim = self.__dim | dim

    @property
    def options(self):
        return self.QP.options | self.upperLevel.options

    @property
    def param(self):
        return self.dyn.param | self.QP.param | self.upperLevel.param

    @property
    def init(self):
        return self.dyn.init | self.QP.init | self.upperLevel.init
    
    def setInit(self, init):

        # set initial values
        self.dyn._dynamics__setInit(init)
        self.QP._QP__setInit(init)
        self.upperLevel._upperLevel__setInit(init)

    def __checkInit(self,value):

        # preallocate output dictionary
        out = {}

        # call check init functions of subclasses
        out = out | self.dyn._dynamics__checkInit(value)
        out = out | self.QP._QP__checkInit(value)
        out = out | self.upperLevel._upperLevel__checkInit(value)

        return out

    @property
    def dyn(self):
        return self.__dyn

    @property
    def QP(self):
        return self.__QP

    @property
    def upperLevel(self):
        return self.__upperLevel
    



    ### NONLINEAR SOLVER FOR TRAJECTORY OPT PROBLEM ---------------------------

    @property
    def trajectoryOpt(self):
        return self.__trajectoryOpt
    
    def makeTrajectoryOpt(self):

        """
        This function creates a (possibly nonlinear) trajectory optimization solver for the full
        upper-level problem. The solver uses the tracking cost and the constraint violation of the
        upper-level combined with the nominal (possibly nonlinear) dynamics.

        This function returns a solver that takes the following inputs

            - x0: initial condition (n_x,1)
            - x_init: state trajectory warmstart (n_x,T+1)
            - u_init: input trajectory warmstart (n_u,T)

        and returns the following outputs

            - S: simVar object containing the solution (note that only S.x, S.u, and S.cost are nonzero)
            - solved: boolean indicating whether the problem was solved successfully
        """
  
        # extract system dynamics
        f = self.dyn.f_nom

        # extract dimensions
        n = self.dim
        
        # create opti object
        opti = Opti()

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
        S._simVar__x = x
        S._simVar__u = u

        # now get cost as a symbolic function of x and u
        _,cost,cst = self.upperLevel._upperLevel__cost(S)
        
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
            out._simVar__x = DM(opti.value(vec(x)))
            out._simVar__u = DM(opti.value(vec(u)))
            out._simVar__cost = DM(opti.value(cost))

            return out,solved

        return solver


    ### SIMULATION FUNCTIONS ---------------------------------------------------

    def __getInitParameters(self,init={}):

        """
        This function takes in a dictionary containing user-defined initial conditions
        and returns the following quantities in a list: p,pf,w,d,y,x, where each quantity
        is either a single DM vector or a list of DM vectors, depending on whether the
        user passed a single vector or a list of vectors in init.

        Note that w should be passed either as single vector of dimension (n_w,1), which
        will be repeated for all time steps, as a matrix of dimension (n_w,T), where
        each column represents the noise at a given time step, or as a list of matrices
        of dimension (n_w,T), where each element of the list is the noise at each time 
        step for a given scenario.

        Accepted keys in the dictionary are: p,pf,w,d,y_lin,x.
        """

        # create out vector
        out = self.init | init

        # first check if at least one of the parameters is a list
        lengths = [len(v) if isinstance(v,list) else 1 for v in out.values()]
        
        # if there are multiple nonzero lengths, check that they match
        if len(set([item for item in lengths if item != 1])) > 1:
            raise Exception('All parameters must have the same length.')
        
        # get final length
        max_length = max(lengths)

        # if w is passed as a single vector (and it is not a list or None), repeat it
        if (out['w'] is not None) and (not isinstance(out['w'],list)) and (out['w'].shape[1] == 1):
            out['w'] = repmat(out['w'],1,self.dim['T'])

        # check dimension of w
        if out['w'] is not None:

            # if w is a list, check that all elements have the same number of columns
            if isinstance(out['w'],list):
                if len(set([v.shape[1] for v in out['w']])) > 1:
                    raise Exception('All noise w must have the same number of columns.')
                if out['w'][0].shape[1] != self.dim['T']:
                    raise Exception('Noise w must have the same number of columns as the prediction horizon.')
            
            # otherwise, check that w has the same number of columns as the prediction horizon
            elif out['w'].shape[1] != self.dim['T']:
                raise Exception('Noise w must have the same number of columns as the prediction horizon.')

        # if there is at least one nonzero length, extend all "dynamics" parameters to that length
        if max_length > 1:
            
            # at least one parameter in "dynamics" is a list of a certain length
            for k,v in out.items():
                
                # only check among parameters that are inside dynamics subclass
                if k in self.dyn.param:
                    
                    # if the parameter is None, do nothing. If the parameter is a list,
                    # do nothing. Only extend to a list of appropriate length the
                    # parameters that are currently not lists
                    if (v is not None) and (not isinstance(v,list)):
                        out[k] = [v]*max_length

        # now "out" contains x,u,w,d as lists of the same length if max_length > 1,
        # otherwise they are all vectors. Moreover, p,pf,y_lin are always vectors.
        # Note that any one of these variables may also be None if it was not passed.

        # under the "trajectory" linearization mode, we need y_lin to be a trajectory
        if self.QP.options['linearization'] == 'trajectory':
            
            # if adaptive mode is used, copy x and u to create y_lin
            if (out['y_lin'] is None) or (out['y_lin']=='adaptive'):
            
                # if u was not passed return an error
                if out['u'] is None:
                    raise Exception('Either pass an input or a linearization trajectory.')
                
                # check if y_lin should be a list or a single value
                if max_length > 1:
                    out['y_lin'] = [vertcat(repmat(out['x'][i],self.dim['N'],1),repmat(out['u'][i],self.dim['N'],1)) for i in range(max_length)]
                else:
                    out['y_lin'] = vertcat(repmat(out['x'],self.dim['N'],1),repmat(out['u'],self.dim['N'],1))
            
            # check if optimal mode is used
            elif out['y_lin'] == 'optimal':
                raise Exception('Optimal linearization trajectory not implemented yet.')
                # TODO implement optimal linearization trajectory
            
            # last case is if y_lin was passed (either as a vector or as a list)
            else:
                # if it was not a list, make it a list
                if not isinstance(out['y_lin'],list):
                    out['y_lin'] = [out['y_lin']] * max_length

        # under the "initial_state" linearization mode, we need y_lin to be a single input
        if self.QP.options['linearization'] == 'initial_state':
            
            # check if y_lin is not passed
            if out['y_lin'] is None:
                
                # if so, check if an input is passed
                if out['u'] is None:

                    # if none is passed, raise an exception
                    raise Exception('Either pass an input or a linearization trajectory.')
                
                # set equal to input (note that input is already either a list or a vector)
                out['y_lin'] = out['u']
                    
        # construct a dictionary to check the dimension. You need to concatenate horizontally
        # any list within out and leave all vectors as they are
        out_not_none = {k:v for k,v in out.items() if v is not None}
        out_concat = {k:hcat(v) if isinstance(v,list) else v for k,v in out_not_none.items()}
        _ = self.__checkInit(out_concat)

        # initial condition
        if out['x'] is None:
            raise Exception('Initial state x is required to simulate the system.')
        
        if out['p'] is None and 'p' in self.param:
            raise Exception('Parameters p are required to simulate the system.')
        
        if out['pf'] is None and 'pf' in self.param:
            raise Exception('Fixed parameters pf are required to simulate the system.')
        
        if out['w'] is None and 'w' in self.param:
            raise Exception('Noise w is required to simulate the system.')
        
        if out['d'] is None and 'd' in self.param:
            raise Exception('Model uncertainty d is required to simulate the system.')

        # extract variables
        p = out['p']        # parameters
        pf = out['pf']      # fixed parameters
        w = out['w']        # noise
        d = out['d']        # model uncertainty
        y = out['y_lin']    # linearization trajectory
        x = out['x']        # initial state

        return p,pf,w,d,y,x

    def simulate(self,init={},options={}):

        """
        This function runs a single simulation of the closed-loop system and returns a list
        S, out_dict, qp_failed

            - S: simVar object containing the simulation results
            - out_dict: dictionary containing debug information about the QP calls, possible keys
                        could be 'qp_time', 'jac_time', 'qp_debug', 'qp_ingredients'
            - qp_failed: boolean indicating whether the QP failed (and simulation was interrupted)

        The function takes the following inputs:

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
                    - 'warmstart_first_qp':True (default) if the first QP should be solved twice (with
                                           propagation of the sensitivity)
                    - 'debug_qp': False (default), or True if debug information about the QP should be stored
                    - epsilon: perturbation magnitude used to compute finite difference derivatives of QP,
                               default is 1e-6
                    - roundoff_qp: number of digits below which QP derivative error is considered zero,
                                   default is 10
                    - 'compute_qp_ingredients':False (default), or True if QP ingredients should be saved
                    - 'warmstart_shift': True (default) if the primal (or primal-dual) warmstart should be shifted
        """

        # get initial parameters
        p,pf,w,d,y,x = self.__getInitParameters(init)

        # simulate
        S, out_dict, qp_failed = self.__simulate(p,pf,w,d,y,x,options)

        return S, out_dict, qp_failed

    def __simulate(self,p,pf,w,d,y,x,options={}):

        """
        Low-level simulate function, unlike simulate, this needs the inputs to be passed separately (as
        returned by __getInitParameters).
        """

        # extract QP for simplicity
        QP = self.QP

        # extract dimensions for simplicity
        n = self.dim

        # update options if provided
        options = self.__default_options['simulate'] | options

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
        idx_qp = self.upperLevel.idx['qp']
        idx_jac = self.upperLevel.idx['jac']

        # extract solver
        if options['mode'] == 'dense':

            # if in dense mode, choose dense solver
            solver = QP.denseSolve
        else:

            # otherwise, choose sparse solver
            solver = QP.solve
        
        # in optimize mode, initialize Jacobians
        if options['mode'] == 'optimize':
            # initialize Jacobians
            J_x_p = DM(n['x'],n['p'])
            J_y_p = DM(n['y'],n['p'])
            S.setJx(0,J_x_p)
            # S.setJy(0,J_y_p)

        # check if QP warmstart was passed
        if options['warmstart_first_qp']:

            # get qp parameter
            p_0 = idx_qp(x,y,p,pf,0)

            # run QP once to get better initialization
            lam,mu,y_all = QP.solve(p_0)

            # update y0
            y0_x = y_all[QP.idx['out']['x'][:-n['x']]]
            y0_u = y_all[QP.idx['out']['u']]
            y = vertcat(self.init['x'],y0_x,y0_u)

            if options['mode'] == 'optimize':

                # extract jacobian of QP variables
                J_QP_p = QP.J_y_p(lam,mu,p_0,idx_jac(J_x_p,J_y_p,0))

                # extract portion associated to y
                J_y_p = J_QP_p[QP.idx['out']['y'],:]

                # rearrange appropriately (note that the first entry of
                # y is x0)
                J_y_p = vertcat(J_x_p,J_y_p[QP.idx['out']['x'][:-n['x']],:],J_y_p[QP.idx['out']['u'],:])
        else:
            lam = None
            mu = None
            y_all = None

        # start counting the time taken to solve the QPs
        total_qp_time = []

        # start counting the time taken to compute the conservative Jacobians
        total_jac_time = []

        # create list to store debug information
        if options['debug_qp']:
            QP_debug = []

        # create list to store QP ingredients
        if options['compute_qp_ingredients']:
            QP_ingredients = []

        # simulation loop
        for t in range(n['T']):
            
            # replace first entry of state with current state
            y_lin = y

            # parameter to pass to the QP
            p_t = idx_qp(x,y_lin,p,pf,t)

            # check if warm start should be shifted
            if options['warmstart_shift']:
                if t > 0:
                    y_all = y_all[QP.idx['out']['y_shift']]

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
            if options['debug_qp']:

                # debug current QP
                qp_debug_out = QP.debug(lam,mu,p_t,options['epsilon'],options['roundoff_QP'],y_all)
                
                # pack results
                QP_debug.append(qp_debug_out)

            if options['compute_qp_ingredients']:

                # compute QP ingredients
                QP_ingredients.append(QP._QP__qp_sparse(p=p_t))

            # store optimization variables
            S.setOptVar(t,lam,mu,y_all[QP.idx['out']['y']],p_t)

            # get next linearization trajectory
            if options['shift_linearization']:
                # shift input trajectory
                x_qp = y_all[QP.idx['out']['x']]
                u_qp = y_all[QP.idx['out']['u']]
                y = vertcat(x_qp,u_qp[n['u']:],u_qp[-n['u']:])
            else:
                # do not shift
                y = y_all[QP.idx['out']['y']]
            
            if 'eps' in QP.idx['out']:

                # get slack variables
                e = y_all[QP.idx['out']['eps']]

                # store slack
                S.setSlack(t,e)

            # get first input entry
            u = y_all[QP.idx['out']['u0']]

            # store input
            S.setInput(t,u)

            # get list of inputs to dynamics
            var_in = [x,u,d,w[t]]
            var_in = [var for var in var_in if var is not None]

            if options['mode'] == 'optimize':
            
                # count time for conservative Jacobians at this time-step
                cons_jac_time = time.time()

                # get conservative jacobian of optimal solution of QP with respect to parameter
                # vector p.
                J_QP_p = QP.J_y_p(lam,mu,p_t,idx_jac(J_x_p,J_y_p,t))

                # select entries associated to y
                if options['shift_linearization']:
                    J_x_qp_p = J_QP_p[QP.idx['out']['x'],:]
                    J_u_qp_p = J_QP_p[QP.idx['out']['u'],:]
                    J_y_p = vertcat(J_x_qp_p,J_u_qp_p[n['u']:,:],J_u_qp_p[-n['u']:,:])
                else:
                    J_y_p = J_QP_p[QP.idx['out']['y'],:]

                if 'eps' in QP.idx['out']:
                    # select entries associated to slack variables and store them
                    J_eps_p = J_QP_p[QP.idx['out']['eps'],:]
                    S.setJeps(t,J_eps_p)

                # select rows corresponding to first input u0
                J_u0_p = J_QP_p[QP.idx['out']['u0'],:]

                # propagate jacobian of closed loop state x
                J_x_p = A(*var_in)@J_x_p + B(*var_in)@J_u0_p

                # store in total cons jac time
                total_jac_time.append(time.time() - cons_jac_time)

                # store conservative jacobians of state and input
                S.setJx(t+1,J_x_p)
                S.setJu(t,J_u0_p)
                S.setJy(t,J_y_p)

            # get next state
            x = f(*var_in)

            # store next state
            S.setState(t+1,x)

        # construct output dictionary
        out_dict = {'qp_time':total_qp_time,'jac_time':total_jac_time}

        if options['debug_qp']:
            out_dict = out_dict | {'qp_debug':QP_debug}

        if options['compute_qp_ingredients']:
            out_dict = out_dict | {'qp_ingredients':QP_ingredients}

        return S, out_dict, qp_failed

    def closedLoop(self,init={},options={}):

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
        p,pf,W,D,Y,X = self.__getInitParameters(init)

        # extract number of samples
        n_samples = len(W) if isinstance(W,list) else 1

        # pass default options
        options = self.__default_options['closedLoop'] | options

        # store dim in a variable with a shorter name
        n = self.dim

        # if gradient descent is used, the true number of iterations
        # is equal to max_k times the number of samples
        if options['gd_type'] == 'gd':
            batch_size = n_samples
        elif options['gd_type'] == 'sgd':
            batch_size = options['batch_size'] if 'batch_size' in options else 1

        # check that batch size does not exceed number of samples
        if batch_size > n_samples:
            raise Exception('Batch size cannot exceed number of samples.')
        
        # extract maximum number of iterations
        max_k = options['max_k'] * batch_size

        # extract cost function
        cost_f = self.upperLevel.cost

        # extract gradient of cost function
        J_cost_f = self.upperLevel.J_cost

        if options['mode'] == 'optimize':

            # extract parameter update law
            alg = self.upperLevel.alg
            p_next = alg['p_next']
            psi_next = alg['psi_next']
            psi_init = alg['psi_init']

        # start empty list
        SIM = []

        # # check if NLP was solved
        # if self.opt['sol']['cost'] is None:
        #     print('Warning: NLP was not solved')

        # # print best cost
        # if options['verbosity'] > 0:
        #     cst = self.opt['sol']['cost']
        #     print(f'Best achievable cost: {cst}')

        # start counting time
        total_iter_time = []

        # list containing all QP times
        total_qp_time = []

        # list containing all Jacobian times
        total_jac_time = []

        # if options['figures']:
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
        best_cost = inf
        p_best = p

        # initialize full gradient of minibatch
        J_p_full = DM(*p.shape)

        # outer loop
        for k in range(max_k):
            
            # start counting iteration time
            iter_time = time.time()

            # sample uncertain elements
            if n_samples > 1:
                if options['random_sampling']:
                    idx = randint(0,n_samples)
                else:
                    idx = int(fmod(k,batch_size))
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
            S, qp_data, qp_failed = self.__simulate(p,pf,w,d,y,x,options)
            
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
            if options['mode'] == 'optimize':

                # if there is no constraint violation, and the cost has improved, save current parameter as best parameter
                if sum1(cst_viol) == 0 and cost < best_cost:
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

                if fmod(k+1,batch_size) == 0:

                    # update parameter
                    p = p_next(p,psi,k,J_p_full,pf)
                    psi = psi_next(p,psi,k,J_p_full,pf)

                    # reset full gradient
                    J_p_full = DM(*p.shape)
                
            else:
                J_p = 0

            if save_memory:
                S.saveMemory()

            # printout
            match options['verbosity']:
                case 0:
                    pass
                case 1:
                    print(f"Iteration: {k}, cost: {track_cost}, J: {norm_2(J_p)}, e : {sum1(fmax(cst_viol,0))}")#, slacks: {slack} ")

            # if options['figures']:

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

        # if options['figures']:
        #     plt.ioff()

        # stack all computation times in one dictionary
        comp_time = dict()
        comp_time['qp'] = total_qp_time
        comp_time['jac'] = total_jac_time
        comp_time['iter'] = total_iter_time

        return SIM, comp_time, p_best