from casadi import *
from BPMPC.dynamics import dynamics
from BPMPC.QP import QP
from BPMPC.MPC import MPC
from BPMPC.upperLevel import upperLevel
from BPMPC.simVar import simVar
import time
from numpy.random import randint

class scenario:

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
        self.__mpc = MPC()
        self.__upperLevel = upperLevel(MSX)
        self.__compTimes = {}

        # default options
        defaultClosedLoop = {'mode':'optimize','gd_type':'gd','figures':False,'random_sampling':False,'debug_qp':False,'compute_qp_ingredients':False,'verbosity':1,'max_k':200}
        defaultSimulate = {'mode':'optimize','shift_linearization':True,'warmstart_first_qp':True,'debug_qp':False,'compute_qp_ingredients':False,'warmstart_shift':True,'epsilon':1e-6,'roundoff_qp':10}
        self.__default_options = {'closedLoop':defaultClosedLoop,'simulate':defaultSimulate}

        pass


    ### COMPUTATION TIMES --------------------------------------------------

    @property
    def compTimes(self):
        return self.__compTimes
    

    ### DIMENSIONS ---------------------------------------------------------

    @property
    def dim(self):
        return self.__dim
    
    def __addDim(self, dim):
        self.__dim = self.__dim | dim


    ### OPTIONS ------------------------------------------------------------

    @property
    def options(self):
        return self.mpc.options | self.upperLevel.options

    ### PARAMETERS ---------------------------------------------------------
    
    @property
    def param(self):
        return self.dyn.param | self.QP.param | self.upperLevel.param


    # INITIAL VALUES -------------------------------------------------------

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

    ### DYNAMICS -----------------------------------------------------------

    @property
    def dyn(self):
        return self.__dyn
    
    def makeDynamics(self, dyn, compile=False):

        """
        This function sets the dynamics of the scenario. It requires a dictionary as an input with keys:

            - 'x': symbolic state variable (n_x,1)
            - 'u': symbolic input variable (n_u,1)
            - 'x_dot': symbolic derivative of the state (n_x,1)
            - 'x_next': symbolic next state (n_x,1)
            - 'w': (optional) symbolic noise variable (n_w,1)
            - 'd': (optional) symbolic disturbance variable (n_d,1)
            - 'x0': (optional) initial state (n_x,1)
            - 'u0': (optional) initial input (n_u,1)
            - 'w0': (optional) nominal noise (n_w,1)
            - 'd0': (optional) nominal disturbance (n_d,1)

        """

        # compilation options
        if compile:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            options = {}

        # initialize dictionary containing all compilation times
        comp_time_dict = {}

        if not all(key in dyn for key in ['x', 'u', 'x_dot', 'x_next']):
            raise Exception('x,u,x_dot and x_next must be defined.')

        # store dynamics
        self.dyn._dynamics__set_x(dyn['x'])
        self.dyn._dynamics__set_u(dyn['u'])

        # check dimensions of x_next and x_dot
        if dyn['x_next'].shape != dyn['x'].shape:
            raise Exception('x_next must have the same dimensions as x.')
        if dyn['x_dot'].shape != dyn['x'].shape:
            raise Exception('x_dot must have the same dimensions as x.')

        # store next state and derivative
        self.dyn._dynamics__set_x_next(dyn['x_next'])
        self.dyn._dynamics__set_x_dot(dyn['x_dot'])

        # store noise and disturbance if present
        if 'w' in dyn:
            self.dyn._dynamics__set_w(dyn['w'])
        if 'd' in dyn:
            self.dyn._dynamics__set_d(dyn['d'])

        # compute dimensions of parameters
        self.__addDim({k: v.shape[0] for k, v in self.dyn.param.items()})

        # extract symbolic parameters and their names
        p_names,p = zip(*self.dyn.param.items())

        # turn into a list
        p = list(p)
        p_names = list(p_names)

        # create casadi function for the dynamics
        start = time.time()
        fc = Function('fc', p, [self.dyn.x_dot], p_names, ['x_dot'],options)
        comp_time_dict = comp_time_dict | {'fc':time.time()-start}
        start = time.time()
        f = Function('f', p, [self.dyn.x_next], p_names, ['x_next'],options)
        comp_time_dict = comp_time_dict | {'f':time.time()-start}

        # extract nominal symbolic parameters and their names
        p_nom_names,p_nom = zip(*self.dyn.param_nominal.items())
        
        # turn into a list
        p_nom = list(p_nom)
        p_nom_names = list(p_nom_names)

        # extract nominal values of nominal parameters
        p_init_nom = []
        if 'x0' not in dyn:
            p_init_nom.append(DM.rand(self.dim['x'],1))
        else:
            try:
                x0 = DM(dyn['x0'])
            except:
                raise Exception('Initial state x0 is not a valid numerical object.')
            if x0.shape[0] != self.dim['x']:
                raise Exception('Initial state x0 does not have the correct dimension or has wrong type.')
            self.dyn._dynamics__setInit({'x':x0})
            p_init_nom.append(x0)
        if 'u0' not in dyn:
            p_init_nom.append(DM.rand(self.dim['u'],1))
        else:
            try:
                u0 = DM(dyn['u0'])
            except:
                raise Exception('Initial input u0 is not a valid numerical object.')
            if u0.shape[0] != self.dim['u']:
                raise Exception('Initial input u0 does not have the correct dimension or has wrong type.')
            self.dyn._dynamics__setInit({'u':u0})
            p_init_nom.append(u0)

        # if d is present, add it to list of disturbance parameters
        dist_p = []
        if 'd' in self.dyn.param:
            if 'd0' not in dyn:
                print('Initialization of d was not passed, defaulting to zero.')
                self.dyn._dynamics__setInit({'d':DM(self.dim['d'],1)})
                dist_p.append(DM(self.dim['d'],1))
            else:
                try:
                    d0 = DM(dyn['d0'])
                except:
                    raise Exception('Initial disturbance d0 is not a valid numerical object.')
                if d0.shape[0] != self.dim['d']:
                    raise Exception('Nominal disturbance d0 does not have the correct dimension or has wrong type.')
                self.dyn._dynamics__setInit({'d':d0})
                dist_p.append(d0)
        # same for w
        if 'w' in self.dyn.param:
            if 'w0' not in dyn:
                print('Initialization of w was not passed, defaulting to zero.')
                self.dyn._dynamics__setInit({'w':DM(self.dim['w'],1)})
                dist_p.append(DM(self.dim['w'],1))
            else:
                try:
                    w0 = DM(dyn['w0'])
                except:
                    raise Exception('Initial noise w0 is not a valid numerical object.')
                if w0.shape[0] != self.dim['w']:
                    raise Exception('Nominal noise w0 does not have the correct dimension or has wrong type.')
                self.dyn._dynamics__setInit({'w':w0})
                dist_p.append(w0)

        # if d or w were passed, nominal and true models are different
        if len(dist_p) > 0:
            model_is_noisy = True
        else:
            model_is_noisy = False

        # extract values of all parameters
        p_init = p_nom + dist_p#[self.dyn.init[i] for i in p_names]
        
        # create nominal dynamics
        if model_is_noisy:

            # # store function arguments
            # args = p_nom + dist_p

            # compute nominal next symbolic state and derivative
            x_next_nom = f(*p_init)
            x_dot_nom = fc(*p_init)

            # save in dynamics
            start = time.time()
            fc_nom = Function('fc_nom',p_nom,[x_dot_nom],p_nom_names,['x_dot_nominal'],options)
            comp_time_dict = comp_time_dict | {'fc_nom':time.time()-start}
            start = time.time()
            f_nom = Function('f_nom',p_nom,[x_next_nom],p_nom_names,['x_next_nominal'],options)
            comp_time_dict = comp_time_dict | {'f_nom':time.time()-start}
        else:
            # otherwise, copy exact dynamics
            fc_nom = fc
            x_dot_nom = self.dyn.x_dot
            f_nom = f
            x_next_nom = self.dyn.x_next
        
        # save in dynamics
        self.__dyn._dynamics__set_fc(fc)
        self.__dyn._dynamics__set_f(f)
        self.__dyn._dynamics__set_fc_nom(fc_nom)
        self.__dyn._dynamics__set_x_dot_nom(x_dot_nom)
        self.__dyn._dynamics__set_f_nom(f_nom)
        self.__dyn._dynamics__set_x_next_nom(x_next_nom)

        # test that f works correctly
        try:
            x_next = f(*p_init)
        except:
            raise Exception('Function f is incompatible with the parameters x,u,d,w you passed.')
        if x_next.shape[0] != self.dim['x']:
            raise Exception('Function f does not return the correct dimension.')

        # test that fc works correctly
        try:
            x_dot = fc(*p_init)
        except:
            raise Exception('Function fc is incompatible with the parameters x,u,d,w you passed.')
        if x_dot.shape[0] != self.dim['x']:
            raise Exception('Function fc does not return the correct dimension.')

        # if nominal and real dynamics are different, test that f_nom and fc_nom work correctly
        if model_is_noisy:
            # test that f_nom works correctly
            try:
                x_next_nom = f_nom(*p_init_nom)
            except:
                raise Exception('Function f_nom is incompatible with the parameters x,u you passed.')
            if x_next_nom.shape[0] != self.dim['x']:
                raise Exception('Function f_nom does not return the correct dimension.')
            
            # test that fc_nom works correctly
            try:
                x_dot_nom = fc_nom(*p_init_nom)
            except:
                raise Exception('Function fc_nom is incompatible with the parameters x,u you passed.')
            if x_dot_nom.shape[0] != self.dim['x']:
                raise Exception('Function fc_nom does not return the correct dimension.')

        # compute jacobians symbolically
        df_dx = jacobian(self.dyn.x_next,self.dyn.x)
        df_du = jacobian(self.dyn.x_next,self.dyn.u)

        # check if df_dx and df_du are constant
        if jacobian(vcat([*symvar(df_dx),*symvar(df_du)]),vertcat(self.dyn.x,self.dyn.u)).is_zero():
            self.dyn._dynamics__set_type('affine')
        else:
            self.dyn._dynamics__set_type('nonlinear')

        # compute jacobians
        start = time.time()
        A = Function('A', p, [df_dx], p_names, ['A'], options)
        comp_time_dict = comp_time_dict | {'A':time.time()-start}
        start = time.time()
        B = Function('B', p, [df_du], p_names, ['B'], options)
        comp_time_dict = comp_time_dict | {'B':time.time()-start}

        # compute nominal jacobians
        if model_is_noisy:

            # compute jacobians symbolically
            df_dx_nom = jacobian(self.dyn.x_next_nom,self.dyn.x)
            df_du_nom = jacobian(self.dyn.x_next_nom,self.dyn.u)

            # compute jacobians
            start = time.time()
            A_nom = Function('A_nom', p_nom, [df_dx_nom], p_nom_names, ['A_nom'], options)
            comp_time_dict = comp_time_dict | {'A_nom':time.time()-start}
            start = time.time()
            B_nom = Function('B_nom', p_nom, [df_du_nom], p_nom_names, ['B_nom'], options)
            comp_time_dict = comp_time_dict | {'B_nom':time.time()-start}

            # save in dynamics
            self.dyn._dynamics__set_A_nom(A_nom)
            self.dyn._dynamics__set_B_nom(B_nom)
        else:
            # otherwise, copy exact dynamics
            A_nom = A
            B_nom = B

        # test that A and B work correctly
        try:
            A_test = A(*p_init)
            B_test = B(*p_init)
        except:
            raise Exception('Functions A and B are incompatible with the parameters x,u,d,w you passed.')
        try:
            A_test_nom = A_nom(*p_init_nom)
            B_test_nom = B_nom(*p_init_nom)
        except:
            raise Exception('Functions A_nom and B_nom are incompatible with the parameters x,u you passed.')
        if A_test.shape[0] != self.dim['x'] or A_test.shape[1] != self.dim['x']:
            raise Exception('Function A does not return the correct dimension.')
        if B_test.shape[0] != self.dim['x'] or B_test.shape[1] != self.dim['u']:
            raise Exception('Function B does not return the correct dimension.')
        if A_test_nom.shape[0] != self.dim['x'] or A_test_nom.shape[1] != self.dim['x']:
            raise Exception('Function A_nom does not return the correct dimension.')
        if B_test_nom.shape[0] != self.dim['x'] or B_test_nom.shape[1] != self.dim['u']:
            raise Exception('Function B_nom does not return the correct dimension.')

        # store in dynamics
        self.dyn._dynamics__set_A(A)
        self.dyn._dynamics__set_A_nom(A_nom)
        self.dyn._dynamics__set_B(B)
        self.dyn._dynamics__set_B_nom(B_nom)

        # store computation times (if compile is true)
        if compile:
            self.__compTimes = self.__compTimes | comp_time_dict


    ### MPC -----------------------------------------------------------------

    @property
    def mpc(self):
        return self.__mpc
    
    def makeMPC(self,N,cost,cst,p=None,pf=None,model=None,options={}):

        """
        This function allows the user to setup the MPC formulation.
        Inputs:

            - N: horizon of the MPC (natural number)
            
            - cost: dictionary containing the cost function. The dictionary must contain the keys
                
                - 'Qx': state cost matrix (n_x*(N-1),n_x*(N-1))
                - 'Qn': terminal state cost matrix (n_x,n_x)
                - 'Ru': input cost matrix (n_u*N,n_u*N)
                - 'x_ref': (optional, defaults to 0) reference state (n_x,1)
                - 'u_ref': (optional, defaults to 0) reference input (n_u,1)
                - 's_lin': (optional, defaults to 0) linear penalty on slack variables
                - 's_quad': (optional, defaults to 1) quadratic penalty on slack variables

              recall that the cost is (x-x_ref)'blkdiag(Qx,Qn)(x-x_ref) + (u-u_ref)'Ru(u-u_ref) + s_lin*e + s_quad*e^2
              Note that x and u are here of dimensions (n_x*N,1) and (n_u*N,1) respectively (i.e. they contain all time-steps).
            
            - cst: dictionary containing the constraints. The dictionary must contain the keys
            
                - 'Hx': state constraint matrix (=,n_x*N)
                - 'hx': state constraint vector (=,1)
                - 'Hx_e': (optional, defaults to identity) matrix that softens constraints (=,n_eps)
                - 'Hu': input constraint matrix (-,n_u*N)
                - 'hu': input constraint vector (-,1)
              
              recall that the constraints are Hx*x + Hx_e*e <= hx, Hu*u <= hu, where e are the slack variables.
              Note that x and u are here of dimensions (n_x*N,1) and (n_u*N,1) respectively (i.e. they contain all time-steps).

            - p: symbolic parameter used to set up the MPC at any given time-step. This is (part of) the decision variable
                 of the upper-level optimization problem.

            - pf: symbolic parameter used to set up the MPC at the first time-step. This is not a decision variable of the
                  upper-level function (e.g. the reference to be tracked).

            - model (optional): dictionary containing the linear model of the system. The dictionary must contain the keys
              A,B,c where the prediction model is x[t+1] = Ax[t] + Bu[t] + c. Note that c is optional and defaults to 0.

            - options (optional): dictionary containing the options. The dictionary can contain the following keys:

                - 'linearization': 'trajectory', 'state' or 'none' (default is 'trajectory')
                - 'slack': True or False (default is False)
                - 'qp_mode': 'stacked' or 'separate' (default is 'stacked')
                - 'solver': 'qpoases','osqp','cplex','gurobi','daqp','qrqp' (default is 'qpoases')
                - 'warmstart': 'x_lam_mu' (warmstart both primal and dual variables) or 'x' (warmstart only 
                            primal variables) (default is 'x_lam_mu')
                - 'jac_tol': tolerance below which multipliers are considered zero (default is 8)
                - 'jac_gamma': stepsize in optimality condition used to apply the IFT (default is 0.001)
                - 'compile_qp_sparse': True or False (default is False)
                - 'compile_jac': True or False (default is False)
                
        """

        # get symbolic variable type
        MSX = self.__MSX

        # add to model
        self.mpc._MPC__set_N(N)         # set function will handle checking if N > 0 and integer
        self.mpc._MPC__set_cost(cost)   # set function will remove extra entries
        self.mpc._MPC__set_cst(cst)     # set function will remove extra entries

        ### EXTRACT PARAMETERS ----------------------------------------------

        # add horizon of MPC to dimensions
        self.__addDim({'N': self.mpc.N})

        # check if linear slack penalty is passed
        if 's_lin' in self.mpc.cost:

            # if penalty is greater than 0, set slack mode to true
            if self.mpc.cost['s_lin'] > 0:
                self.mpc._MPC__updateOptions({'slack':True})
            
            # if penalty is negative, raise exception
            elif self.mpc.cost['s_lin'] < 0:
                raise Exception('Linear slack penalty must be nonnegative.')
            
        # check if quadratic slack penalty is passed
        if 's_quad' in self.mpc.cost:

            # if penalty is greater than 0, set slack mode to true
            if self.mpc.cost['s_quad'] > 0:
                self.mpc._MPC__updateOptions({'slack':True})

            # if penalty is negative, raise exception
            elif self.mpc.cost['s_quad'] < 0:
                raise Exception('Quadratic slack penalty must be nonnegative.')

        # extract options
        self.mpc._MPC__updateOptions(options)

        # if Hx_e is passed, set slack mode to true
        if 'Hx_e' in self.mpc.cst:
            self.mpc._MPC__updateOptions({'slack':True})
        
        # obtain number of slack variables
        if self.mpc.options['slack']:
            # check if slack matrix is passed
            if 'Hx_e' not in self.mpc.cst:
                # if not, set as identity matrix
                self.mpc._MPC__set_cst(self.mpc.cst | {'Hx_e':MSX.eye(self.mpc.cst['Hx'].shape[0])})
            # add number of slack variables
            self.__addDim({'eps': self.mpc.cst['Hx_e'].shape[0]})
        else:
            # if no constraint softening is used, set number of slack variables to zero
            self.__addDim({'eps': 0})
            # set Hx_e to None
            Hx_e = None

        # save dimensions in variable with shorter name for simplicity
        n = self.dim

        # extract symbolic variables
        x = self.param['x']
        self.QP._QP__set_x(x)

        # extract constraints
        Hx = MSX(self.mpc.cst['Hx'])
        hx = MSX(self.mpc.cst['hx'])
        Hu = MSX(self.mpc.cst['Hu'])
        hu = MSX(self.mpc.cst['hu'])

        # check dimensions
        if Hx.shape[1] != n['N']*n['x']:
            raise Exception('Hx must have as many columns as x.')
        if hx.shape[0] != Hx.shape[0]:
            raise Exception('hx must have as many rows as Hx.')
        if Hu.shape[1] != n['N']*n['u']:
            raise Exception('Hu must have as many columns as u.')
        if hu.shape[0] != Hu.shape[0]:
            raise Exception('hu must have as many rows as Hu.')

        if self.mpc.options['slack']:
            Hx_e = MSX(self.mpc.cst['Hx_e'])
            if Hx_e.shape[0] != Hx.shape[0]:
                raise Exception('Hx_e must have as many rows as Hx.')

        # extract cost
        Qx = MSX(self.mpc.cost['Qx'])
        Ru = MSX(self.mpc.cost['Ru'])
        Qn = MSX(self.mpc.cost['Qn'])

        # check dimensions
        if Qx.shape[0] != (n['N']-1)*n['x']:
            raise Exception('Qx must have as many rows as x.')
        if Qx.shape[1] != (n['N']-1)*n['x']:
            raise Exception('Qx must have as many columns as x.')
        if Ru.shape[0] != n['N']*n['u']:
            raise Exception('Ru must have as many rows as u.')
        if Ru.shape[1] != n['N']*n['u']:
            raise Exception('Ru must have as many columns as u.')
        if Qn.shape[0] != n['x']:
            raise Exception('Qn must have as many rows as x.')
        if Qn.shape[1] != n['x']:
            raise Exception('Qn must have as many columns as x.')
        
        # extract reference
        if 'x_ref' in self.mpc.cost:
            x_ref = self.mpc.cost['x_ref']
        else:
            x_ref = MSX(n['x']*n['N'],1)
        if 'u_ref' in self.mpc.cost:
            u_ref = self.mpc.cost['u_ref']
        else:
            u_ref = MSX(n['u']*n['N'],1)
        

        ### CREATE SPARSE MPC ----------------------------------------------

        # check if a model was passed
        if model is not None:
            self.mpc._MPC__set_model(model)

        # start by creating the linearizations
        A_list,B_list,c_list = self.__createMPCLinearizations()

        # create QP ingredients
        G,g,F,f,Q,Qinv,q,idx = self.__makeSparseMPC(A_list,B_list,c_list,Qx,Qn,Ru,x_ref,u_ref,Hx,Hu,hx,hu,Hx_e)

        # stack all constraints together to match CasADi's conic interface
        A = vcat([G,F])

        # equality constraints can be enforced by setting lba=uba
        uba = vcat([g,f])
        lba = vcat([-inf*MSX.ones(g.shape),f])

        # sparsify
        try:
            A = cse(sparsify(A))
            uba = cse(sparsify(uba))
            lba = cse(sparsify(lba))
        except:
            pass


        ### CREATE DENSE MPC ------------------------------------------------

        dense_qp = self.__makeDenseMPC(A_list,B_list,c_list,Qx,Qn,Ru,x_ref,u_ref,Hx,Hu,hx,hu)


        ### CREATE DUAL MPC FORMULATION -------------------------------------

        try:
            # define Hessian of dual
            H_11 = cse(sparsify(G@Qinv@G.T))
            H_12 = cse(sparsify(G@Qinv@F.T))
            H_21 = cse(sparsify(F@Qinv@G.T))
            H_22 = cse(sparsify(F@Qinv@F.T))
            H = cse(blockcat(H_11,H_12,H_21,H_22))

            # define linear term of dual
            h_1 = cse(sparsify(G@Qinv@q+g))
            h_2 = cse(sparsify(F@Qinv@q+f))
            h = cse(vcat([h_1,h_2]))

        except:

            # define Hessian of dual
            H_11 = G@Qinv@G.T
            H_12 = G@Qinv@F.T
            H_21 = F@Qinv@G.T
            H_22 = F@Qinv@F.T
            H = blockcat(H_11,H_12,H_21,H_22)

            # define linear term of dual
            h_1 = G@Qinv@q+g
            h_2 = F@Qinv@q+f
            h = vcat([h_1,h_2])

        # store in dictionary
        QP_dict = dict()
        QP_dict['Q'] = Q
        QP_dict['Qinv'] = Qinv
        QP_dict['q'] = q
        QP_dict['G'] = G
        QP_dict['g'] = g
        QP_dict['F'] = F
        QP_dict['f'] = f
        QP_dict['A'] = A
        QP_dict['lba'] = lba
        QP_dict['uba'] = uba
        QP_dict['H'] = H
        QP_dict['h'] = h
        QP_dict['dense'] = dense_qp


        ### STORE IN MPC -----------------------------------------------------

        # store dimensions of equality and inequality constraints
        self.__addDim({'in': G.shape[0]})
        self.__addDim({'eq': F.shape[0]})

        # store ingredients
        self.QP._QP__setIngredients(QP_dict)

        # store index
        self.QP._QP__updateIdx({'out':idx})

        # primal optimization variables
        self.QP._QP__set_y(MSX.sym('y',q.shape[0]-n['eps'],1))

        # dual optimization variables (inequality constraints)
        self.QP._QP__set_lam(MSX.sym('lam',g.shape[0],1))

        # dual optimization variables (equality constraints)
        self.QP._QP__set_mu(MSX.sym('mu',f.shape[0],1))

        # dual optimization variable (all constraints)
        self.QP._QP__set_z(vcat([self.QP.lam,self.QP.mu]))

        # add dimensions
        self.__addDim({k: v.shape[0] for k, v in self.QP.param.items()})

        # create QP
        self.__makeQP(p=p,pf=pf,mode=self.mpc.options['qp_mode'],solver=self.mpc.options['solver'],warmstart=self.mpc.options['warmstart'],compile=self.mpc.options['compile_qp_sparse'])

        # create conservative jacobian
        self.__makeConsJac(gamma=self.mpc.options['jac_gamma'],tol=self.mpc.options['jac_tol'],compile=self.mpc.options['compile_jac'])

    def __makeSparseMPC(self,A_list,B_list,c_list,Qx,Qn,Ru,x_ref,u_ref,Hx,Hu,hx,hu,Hx_e=None):

        """
        This function creates the sparse MPC formulation.
        
        The inputs are:
        
            - A_list, B_list, c_list: lists of matrices A, B, and c such that x[t+1] = A[t]@x[t] + B[t]@u[t] + c[t]
            - Qx, Qn, Ru, x_ref, u_ref: matrices defining the cost function (x-x_ref)'blkdiag(Qx,Qn)x(x-x_ref) + (u-u_ref)'Ru(u-u_ref)
            - Hx, Hu, hx, hu, Hx_e: polyhedral constraints Hx*x + Hx_e*e <= hx, Hu*u <= hu, where e are the slack variables
        
        Note that if Hx_e is not passed, but the slack option is enabled, it is set to the identity matrix.
        Note that x and u are of dimension (n_x*N,1) and (n_u*N,1) respectively (i.e. they contain all time-steps).
        
        This function additionally parses the slack penalties if in slack mode (default is quadratic penalty = 1 and linear 
        penalty = 0).
        
        The function returns the matrices G, g, F, f, Q, Qinv=inv(Q), and the dictionary idx. The matrices constitute the MPC as follows

            min 1/2 y'Qy + q'y
            s.t. Gy <= g
                 Fy = y
        
        whereas idx is a dictionary containing the indexing of the output optimization variables of the QP.
        This function sets up the following keys in idx:

            - 'u': range of all inputs
            - 'x': range of all states
            - 'y': range of all state-input variables
            - 'eps': range of all slack variables (if present)
            - 'u0': range of first input
            - 'u1': range of second input
            - 'x_shift': states shifted by one time-step (last state repeated)
            - 'u_shift': inputs shifted by one time-step (last input repeated)
            - 'y_shift': concatenation of x_shift and u_shift (and slacks shifted if present)

        """

        # get symbolic variable type
        MSX = self.__MSX

        # extract dimensions
        n = self.dim

        # extract options
        opt = self.mpc.options

        if opt['slack']:

            # if ('s_lin' not in self.mpc.cost) and ('s_quad' not in self.mpc.cost):
                # raise Exception('Slack variables are enabled, but no slack penalties are provided.')

            # linear penalty
            try:
                s_lin = MSX(self.mpc.cost['s_lin'])
            except:
                s_lin = MSX(0) # default value
                pass

            # quadratic penalty
            try:
                s_quad = MSX(self.mpc.cost['s_quad'])
            except:
                print('Quadratic slack penalty not provided, defaulting to 1.')
                s_quad = MSX(1) # default value
                pass

            # check dimensions
            if s_lin.shape[0] != 1:
                raise Exception('Linear slack penalty must be a scalar.')
            if s_quad.shape[0] != 1:
                raise Exception('Quadratic slack penalty must be a scalar.')

            # add columns associated to input and slack variables
            Hx = hcat([Hx,MSX(Hx.shape[0],n['N']*n['u']),-Hx_e])

            # add columns associated to state and slack variables
            Hu = hcat([MSX(Hu.shape[0],n['N']*n['x']),Hu,MSX(Hu.shape[0],n['eps'])])

            # add nonnegativity constraints on slack variables
            He = hcat([MSX(n['eps'],n['N']*(n['x']+n['u'])),-MSX.eye(n['eps'])])
            he = MSX(n['eps'],1)

            # create inequality constraint matrices
            try:
                G = cse(sparsify(vcat([Hx,Hu,He])))
                g = cse(sparsify(vcat([hx,hu,he])))
            except:
                G = vcat([Hx,Hu,He])
                g = vcat([hx,hu,he])

        else:

            # add columns associated to input and slack variables
            Hx = hcat([Hx,MSX(Hx.shape[0],n['N']*n['u'])])

            # add columns associated to state and slack variables
            Hu = hcat([MSX(Hu.shape[0],n['N']*n['x']),Hu])
                
            # create inequality constraint matrices
            try:
                G = cse(sparsify(vcat([Hx,Hu])))
                g = cse(sparsify(vcat([hx,hu])))
            except:
                G = vcat([Hx,Hu])
                g = vcat([hx,hu])

        
        ### CREATE EQUALITY CONSTRAINTS ------------------------------------

        # preallocate equality constraint matrices
        F = MSX(n['N']*n['x'],n['N']*(n['x']+n['u'])+n['eps'])
        f = MSX(n['N']*n['x'],1)

        # construct matrix
        for i in range(n['N']):
        
            # negative identity for next state
            F[i*n['x']:(i+1)*n['x'],i*n['x']:(i+1)*n['x']] = -MSX.eye(n['x'])

            # A matrix multiplying current state
            if i > 0:
                F[i*n['x']:(i+1)*n['x'],(i-1)*n['x']:i*n['x']] = A_list[i]

            # B matrix multiplying current input
            F[i*n['x']:(i+1)*n['x'],n['N']*n['x']+i*n['u']:n['N']*n['x']+(i+1)*n['u']] = B_list[i]

            # affine term 
            f[i*n['x']:(i+1)*n['x']] = c_list[i]

        # sparsify F and f
        try:
            F = cse(sparsify(F))
            f = cse(sparsify(f))
        except:
            pass


        ### CREATE COST -----------------------------------------------------

        # construct state cost by stacking Qx and Qn
        Q = blockcat(Qx,MSX((n['N']-1)*n['x'],n['x']),MSX(n['x'],(n['N']-1)*n['x']),Qn)

        # add input cost
        Q = blockcat(Q,MSX(n['N']*n['x'],n['N']*n['u']),MSX(n['N']*n['u'],n['N']*n['x']),Ru)

        # append cost applied to slack variable
        if opt['slack']:
            Q = blockcat(Q,MSX(Q.shape[0],n['eps']),MSX(n['eps'],Q.shape[0]),s_quad*MSX.eye(n['eps']))

        # inverse of quadratic cost matrix
        Qinv = inv(Q)

        # create linear part of the cost
        if opt['slack']:
            q = vcat([(-x_ref.T@blockcat(Qx,MSX(n['x']*(n['N']-1),n['x']),MSX(n['x'],n['x']*(n['N']-1)),Qn)).T,(-u_ref.T@Ru).T,s_lin*MSX.ones(n['eps'],1)])
        else:
            q = vcat([(-x_ref.T@blockcat(Qx,MSX(n['x']*(n['N']-1),n['x']),MSX(n['x'],n['x']*(n['N']-1)),Qn)).T,(-u_ref.T@Ru).T])

        # sparsify Q and q
        try:
            Q = cse(sparsify(Q))
            Qinv = cse(sparsify(Qinv))
            q = cse(sparsify(q))
        except:
            pass


        ### CREATE INDEX DICTIONARY ----------------------------------------

        # store output variable indices
        idx = dict()
        
        # range of all inputs
        idx['u'] = range(n['x']*n['N'],(n['x']+n['u'])*n['N'])
        
        # range of all states
        idx['x'] = range(0,n['x']*n['N'])
        
        # range of all state-input variables
        idx['y'] = range(0,(n['x']+n['u'])*n['N'])
        
        # range of all slack variables
        if opt['slack']:
            idx['eps'] = range((n['x']+n['u'])*n['N'],(n['x']+n['u'])*n['N']+n['eps'])
            idx_e = np.arange(n['eps']) + n['N'] * (n['x'] + n['u'])
            idx_e_shifted = np.hstack([idx_e[n['eps']:], idx_e[:n['eps']]])

        # first input
        idx['u0'] = range(n['x']*n['N'],n['x']*n['N']+n['u'])

        # second input
        idx['u1'] = range(n['x']*n['N']+n['u'],n['x']*n['N']+2*n['u'])

        # Generate indices for x and u in y
        idx_x = np.arange(n['N'] * n['x'])
        idx_u = np.arange(n['N'] * n['u']) + n['N'] * n['x']
        
        # Shift x and u indices
        idx_x_shifted = np.hstack([idx_x[n['x']:], idx_x[-n['x']:]])
        idx_u_shifted = np.hstack([idx_u[n['u']:], idx_u[-n['u']:]])
        
        # Combine the shifted indices
        if opt['slack']:
            idx_shifted = np.hstack([idx_x_shifted, idx_u_shifted, idx_e_shifted])
        else:
            idx_shifted = np.hstack([idx_x_shifted, idx_u_shifted])

        # create shifted indices
        idx['x_shift'] = idx_x_shifted
        idx['u_shift'] = idx_u_shifted
        idx['y_shift'] = idx_shifted
    
        return G,g,F,f,Q,Qinv,q,idx

    def __makeDenseMPC(self,A_list,B_list,c_list,Qx,Qn,Ru,x_ref,u_ref,Hx,Hu,hx,hu):
        """
        Create a dictionary with all the ingredients needed to solve the MPC problem in dense form.
        
        The inputs are:
            
            - A_list, B_list, c_list: lists of matrices A, B, and c such that x[t+1] = A[t]@x[t] + B[t]@u[t] + c[t]
            - Qx, Qn, Ru, x_ref, u_ref: matrices defining the cost function (x-x_ref)'Qx(x-x_ref) + (u-u_ref)'Ru(u-u_ref)
            - Hx, Hu, hx, hu: polyhedral constraints Hx*x <= hx, Hu*u <= hu
        
        The output dictionary has keys:
        
            - 'G_x', 'G_u', 'g_c': matrices satisfying x = G_x*x0 + G_u*u + g_c
            - 'Qx', 'Ru', 'x_ref', 'u_ref': cost function (x-x_ref)'Qx(x-x_ref) + (u-u_ref)'Ru(u-u_ref)
            - 'Hx', 'Hu', 'hx', 'hu': polyhedral constraints Hx*x <= hx, Hu*u <= hu
        """

        # get symbolic variable type
        MSX = self.__MSX

        # extract dimensions
        n = self.dim

        # start by constructing matrices G_x and G_u, and vector g_c such that
        # x = G_x*x_0 + G_u*u + g_c, where x = vec(x_1,x_2,...,x_N), 
        # u = vec(u_0,u_1,...,u_N-1).
        # To obtain g_c we need to multiply all the affine terms by a matrix
        # similar to G_u, which we call G_c.
        
        # first initialize G_u with the zero matrix
        G_u = MSX(n['x']*n['N'],n['u']*n['N'])

        # initialize G_c
        G_c = MSX(n['x']*n['N'],n['x']*n['N'])

        # we will need a tall matrix that will replace the columns of G_u
        # initially it is equal to a tall matrix full of zeros with an
        # identity matrix at the bottom.
        col = MSX.eye(n['N']*n['x'])[:,(n['N']-1)*n['x']:n['N']*n['x']]

        # loop through all columns of G_u (t ranges from N-1 to 1)
        for t in range(n['N']-1,0,-1):

            # get matrices A and at time-step t
            A_t = A_list[t]
            B_t = B_list[t]

            # update G_u matrix
            G_u[:,t*n['u']:(t+1)*n['u']] = col@B_t

            # update G_c matrix
            G_c[:,t*n['x']:(t+1)*n['x']] = col

            # update col by multiplying with A matrix and adding identity matrix
            col = col@A_t + MSX.eye(n['N']*n['x'])[:,(t-1)*n['x']:t*n['x']]

        # get linearized dynamics at time-step 0
        A_0 = A_list[0]
        B_0 = B_list[0]

        # correct first entry of c_list
        c_list[0] = c_list[0] + A_0@self.param['x']

        # now we only miss the left-most column (use x0 instead of x[:n['x']])
        G_u[:,:n['u']] = col@B_0

        # same for G_c
        G_c[:,:n['x']] = col

        # matrix G_x is simply col@A_0
        G_x = col@A_0

        # to create g_c concatenate vertically the entries in the list c_t_list
        # then multiply by G_c from the right
        c_t = -vcat(c_list)
        g_c = G_c@c_t

        # attach terminal cost to state cost matrix
        Qx = blockcat(Qx,MSX((n['N']-1)*n['x'],n['x']),MSX(n['x'],(n['N']-1)*n['x']),Qn)

        # create dictionary
        out = {'G_x':G_x,'G_u':G_u,'g_c':g_c,'Qx':Qx,'Ru':Ru,'x_ref':x_ref,'u_ref':u_ref,'Hx':Hx,'Hu':Hu,'hx':hx,'hu':hu}

        return out

    def __createMPCLinearizations(self):

        """
        This function constructs the prediction model for the MPC problem. There are multiple options:

            1. A custom time-invariant linear model can be passed (through self.mpc.model, which contains
               the matrices A, B, and c). In this case, opt['linearization'] is set to 'none'.

            2. If the model is affine, then A,B,c are the true nominal dynamics of the system, this happens
               if self.model.type == 'affine', and we set opt['linearization'] to 'none'.
               
            2. The model can be linearized around the initial state (opt['linearization'] = 'initial_state').
               In this case, the linearization trajectory is a single input u_lin.

            3. (default) The model can be linearized along a trajectory (opt['linearization'] = 'trajectory').
               In this case y_lin contains the state-input trajectory along which the dynamics are linearized.

        The function returns three list A_list,B_list,c_list, such that the linearized dynamics at time-step t
        are given by  x[t+1] = A_list[t]@x[t] + B_list[t]@u[t] + c_list[t].

        Note that c_list[0] contains additionally the effect -A_list[0]@x0 of the initial state x0.
        """

        # get symbolic variable type
        MSX = self.__MSX

        # extract dimensions
        n = self.dim

        # extract options
        opt = self.mpc.options

        # extract symbolic variables
        x = self.param['x']

        # extract dynamics
        fd = self.dyn.f_nom
        A = self.dyn.A_nom
        B = self.dyn.B_nom

        # if user passed a custom affine model, use it
        if self.mpc.model is not None:
            
            # extract matrices
            A_mat = self.mpc.model['A']
            B_mat = self.mpc.model['B']
            if 'c' in self.mpc.model:
                c_mat = self.mpc.model['c']
            else:
                c_mat = MSX(n['x'],1)

            # check dimensions
            if A_mat.shape[0] != n['x'] or A_mat.shape[1] != n['x']:
                raise Exception('A must have as many rows and columns as x.')
            if B_mat.shape[0] != n['x'] or B_mat.shape[1] != n['u']:
                raise Exception('B must have as many rows as x and as many columns as u.')
            if c_mat.shape[0] != n['x']:
                raise Exception('c must have as many rows as x.')

            # stack in list
            A_list = [A_mat] * n['N']
            B_list = [B_mat] * n['N']
            c_list = [c_mat] * n['N']

            # patch first entry
            c_list[0] = - A_mat@x

            self.mpc._MPC__updateOptions({'linearization':'none'})

        # if model is affine, compute exact dynamics
        elif self.dyn.type == 'affine':

            # extract nominal symbolic parameters and their names
            p_nom_names = self.dyn.param_nominal.keys()
            
            # extract nominal values of nominal parameters
            p_init_nom = [self.dyn.init[i] if self.dyn.init[i] is not None else DM(self.dim[i], 1) for i in p_nom_names]

            # get nominal state and input
            x_nom = self.dyn.init['x'] if self.dyn.init['x'] is not None else DM(self.dim['x'], 1)
            u_nom = self.dyn.init['u'] if self.dyn.init['u'] is not None else DM(self.dim['u'], 1)

            # create nominal dynamics f(x,u) = Ax + bu + c
            A_mat = A(*p_init_nom)
            B_mat = B(*p_init_nom)
            c_mat = -(fd(*p_init_nom) - A_mat@x_nom - B_mat@u_nom)

            # stack in list
            A_list = [A_mat] * n['N']
            B_list = [B_mat] * n['N']
            c_list = [c_mat] * n['N']

            # patch first entry of c_list
            c_list[0] = c_list[0] - A_mat@x

            self.mpc._MPC__updateOptions({'linearization':'none'})

        # if mode is 'initial_state', linearize around the initial state
        elif opt['linearization'] == 'initial_state':

            # linearization trajectory is a single input
            y_lin = MSX.sym('y_lin',n['u'],1)
            u_lin = y_lin

            # save in mpc
            self.QP._QP__set_y_lin(y_lin)

            # compute derivatives
            A_lin = A(x,u_lin)
            B_lin = B(x,u_lin)
            c_lin = - ( fd(x,u_lin) - A_lin@x - B_lin@u_lin )

            # stack in list
            A_list = [A_lin] * n['N']
            B_list = [B_lin] * n['N']
            c_list = [c_lin] * n['N']
                
        # if mode is 'trajectory', linearize along a trajectory (similar to real-time iteration)
        elif opt['linearization'] == 'trajectory':

            # create symbolic variable for linearization trajectory
            y_lin = MSX.sym('y_lin',(n['x']+n['u'])*n['N'],1)

            # save in mpc
            self.QP._QP__set_y_lin(y_lin)

            # extract linearization input and state
            x_lin = y_lin[:n['N']*n['x']]
            u_lin = y_lin[n['N']*n['x']:]

            # preallocate matrices
            A_list = []
            B_list = []
            c_list = []

            # construct matrix
            for i in range(n['N']):
            
                # extract current linearization point
                x_i = x_lin[i*n['x']:(i+1)*n['x']]
                u_i = u_lin[i*n['u']:(i+1)*n['u']]
                
                # distinguish between i=0 for initial condition
                if i > 0:
                    A_i = A(x_i,u_i)
                    A_list.append(A_i)
                    B_i = B(x_i,u_i)
                    B_list.append(B_i)
                    c_i = - ( fd(x_i,u_i) - A_i@x_i - B_i@u_i )
                    c_list.append(c_i)
                else:
                    A_i = A(x,u_i)
                    A_list.append(A_i)
                    B_i = B(x,u_i)
                    B_list.append(B_i)
                    c_i = - ( fd(x,u_i) - A_i@x - B_i@u_i ) - A_i@x
                    c_list.append(c_i)

        return A_list, B_list, c_list


    ### QP ------------------------------------------------------------------
    
    @property
    def QP(self):
        return self.__QP

    def __makeQP(self,p=None,pf=None,mode='stacked',solver='qpoases',warmstart='x_lam_mu',compile=False):
        
        """
        This function creates the functions necessary to solve the MPC problem in QP form. Specifically, this function
        sets the following properties of QP:

            - qp_sparse: this function takes in p_QP and returns the sparse ingredients, which are:
                
                - F,f,G,g,Q,q in the separate mode, where the QP is formulated as

                    min 1/2 y'Qy + q'y
                    s.t. Gy <= g
                         Fy = y

                - A,lba,uba,Q,q in the stacked mode, where the QP is formulated as

                    min 1/2 y'Qy + q'y
                    s.t. lba <= Ay <= uba

            - dual_sparse: this function takes in p_QP and returns the dual ingredients H,h, where the dual
              QP is formulated as:

                min 1/2 z'Hz + h'z
                s.t. z >= 0
            
            - qp_dense: this function takes in p_QP and returns the dense ingredients, which are:

                - 'G_x', 'G_u', 'g_c': matrices satisfying x = G_x*x0 + G_u*u + g_c
                - 'Qx', 'Ru', 'x_ref', 'u_ref': cost function (x-x_ref)'Qx(x-x_ref) + (u-u_ref)'Ru(u-u_ref)
                - 'Hx', 'Hu', 'hx', 'hu': polyhedral constraints Hx*x <= hx, Hu*u <= hu
              
              where the QP is formulated as

                min 1/2 u'(G_x'QxG_x + Ru)u + (G_x'Qx(G_x*x0 + g_c - x_ref) - Ru*u_ref)'u
                s.t. Hx*(G_x*x0 + G_u*u + g_c) <= hx
                     Hu*u <= hu

            - solve: this function takes in p_QP and returns the optimal solution of the QP problem.
              Based on the type of warmstarting (i.e. x_lam_mu or x), the function will take the following 
              inputs:

                - x_lam_mu: p_QP, x0=None, lam=None, mu=None
                - x: p_QP, x0=None

              the output is lam, mu, y.
                
        This function also sets up all the symbolic variables and their dimensions.

        Note that p_QP contains all necessary parameters required to setup the ingredients at any given time-step,
        e.g. x0,y_lin,p_t,pf_t). This is different from the p_QP stored as a parameter, which does not contain pf.

        Additionally, this functions sets up the 'in' entry of the idx dictionary of scenario.QP, which contains the
        indexing of all the input parameters in p_QP:

            - 'x0': initial state
            - 'y_lin': linearization trajectory
            - 'p_t': parameters that are optimized in the upper-level
        
        Note that not all parameters need to be present.

        """

        # compilation options
        if compile:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            options = {}

        # first construct the parameter vector required for the QP
        p_QP = []
        p_QP_names = []

        # the MPC problem always depends on the current state
        x = self.QP.x
        p_QP.append(x)
        p_QP_names.append('x')

        # there may be also a linearization trajectory
        if self.QP.y_lin is not None:

            # get linearization trajectory
            y_lin = self.QP.y_lin

            # append to list
            p_QP.append(y_lin)
            p_QP_names.append('y_lin')
        else:

            # if y_lin is not present, set it as a zero-dimensional SX
            y_lin = SX(0,0)

        # then add p
        if p is not None:

            # store in parameters of QP
            self.QP._QP__set_p_t(p)

            # append to list
            p_QP.append(p)
            p_QP_names.append('p_t')

        # create input index
        idx_in = dict()
        # initialize counter that reveals the beginning index of each variable
        running_idx = 0
        # loop through all the symbolic variables
        for i in range(len(p_QP)):
            # get length of current variable in p_d
            len_current_p_d = p_QP[i].shape[0]
            # store indexing of current variable
            idx_in[p_QP_names[i]] = range(running_idx,running_idx+len_current_p_d)
            # increment running index
            running_idx = running_idx + len_current_p_d

        # store in QP index dictionary
        self.QP._QP__updateIdx({'in':idx_in})

        # save full symbolic qp parameter (do not include pf even if present)
        self.QP._QP__set_p_qp(p_QP)

        # add dimension
        self.__addDim({'p_qp': vcat(p_QP).shape[0]})

        # add pf (i.e., fixed parameters that are not differentiated)
        if pf is not None:

            # append to list
            p_QP.append(pf)
            p_QP_names.append('pf')

            # save symbolic parameter
            self.QP._QP__set_pf_t(pf)
        
            # add dimension
            self.__addDim({'pf_qp': pf.shape[0]})
        
        else:
            # add dimension
            self.__addDim({'pf_qp': 0})

        # extract ingrediens
        Q = self.QP.ingredients['Q']
        q = self.QP.ingredients['q']
        F = self.QP.ingredients['F']
        f = self.QP.ingredients['f']
        G = self.QP.ingredients['G']
        g = self.QP.ingredients['g']
        A = self.QP.ingredients['A']
        lba = self.QP.ingredients['lba']
        uba = self.QP.ingredients['uba']
        H = self.QP.ingredients['H']
        h = self.QP.ingredients['h']

        # select outputs
        if mode == 'separate':
            QP_outs = [F,f,G,g,Q,q]
            QP_outs_names = ['F','f','G','g','Q','q']
            raise Exception('Separate mode not implemented yet.') #TODO implement separate mode
        elif mode == 'stacked':
            QP_outs = [A,lba,uba,Q,q]
            QP_outs_names = ['A','lba','uba','Q','q']

        # create function
        start = time.time()
        QP_func = Function('QP',[vcat(p_QP)],QP_outs,['p'],QP_outs_names,options)
        comp_time_dict = {'QP_func':time.time()-start}

        # save QP in model
        self.QP._QP__setQpSparse(QP_func)

        # create function
        start = time.time()
        dual_func = Function('dual',[vcat(p_QP)],[H,h],['p'],['H','h'],options)
        comp_time_dict['dual_func'] = time.time()-start

        # save dual in model
        self.QP._QP__setDualSparse(dual_func)

        # extract dense ingredients
        dense_qp = self.QP.ingredients['dense']

        # dense outputs
        QP_outs_dense_names = list(dense_qp.keys())
        QP_outs_dense = list(dense_qp.values())

        # create function
        start = time.time()
        QP_dense_func = Function('QP_dense',[vcat(p_QP)],QP_outs_dense,['p'],QP_outs_dense_names,options)
        comp_time_dict['QP_dense_func'] = time.time()-start

        # save in QP model
        self.QP._QP__setQpDense(QP_dense_func)

        # implement QP using conic interface to retrieve multipliers
        qp = {}
        qp['h'] = Q.sparsity()
        qp['a'] = A.sparsity()

        # vector specifying which constraints are equalities
        is_equality = [False]*self.dim['in']+[True]*self.dim['eq']

        # add existing options to compile function S
        start = time.time()
        match solver:
            case 'gurobi':
                S = conic('S','gurobi',qp,{'gurobi':{'OutputFlag':0},'equality':is_equality})
            case 'cplex':
                S = conic('S','cplex',qp,{'cplex':{'CPXPARAM_Simplex_Display': 0,'CPXPARAM_ScreenOutput': 0},'equality':is_equality})
            case 'qpoases':
                S = conic('S','qpoases',qp,options|{'printLevel':'none','equality':is_equality})
            case 'osqp':
                S = conic('S','osqp',qp,options|{'osqp':{'verbose':False},'equality':is_equality})
            case 'daqp':
                S = conic('S','daqp',qp,options)
            case 'qrqp':
                S = conic('S','qrqp',qp,options|{'print_iter':False,'equality':is_equality})
        comp_time_dict['S_sparse'] = time.time()-start

        if mode == 'stacked':

            # create local function setting up the qp
            match warmstart:
                
                # warmstart both primal and dual variables
                case 'x_lam_mu':
            
                    def local_qp(p_qp,x0=None,lam0=None,mu0=None):

                        # get data from qp_dense function
                        a,lba,uba,h,g = QP_func(p_qp)

                        # solve QP with warmstarting
                        if x0 is not None:
                            sol = S(h=h,a=a,g=g,lba=lba,uba=uba,x0=x0,lam_a0=vertcat(lam0,mu0))
                        
                        # if does not work, try without warmstarting
                        else:
                            sol = S(h=h,a=a,g=g,lba=lba,uba=uba)
                        
                        # return lambda, mu, y
                        return sol['lam_a'][:self.dim['in']],sol['lam_a'][self.dim['in']:],sol['x']
                    
                # warmstart only primal
                case 'x':
            
                    def local_qp(p_qp,x0=None):

                        # get data from qp_dense function
                        a,lba,uba,h,g = QP_func(p_qp)

                        # solve QP with warmstarting
                        if x0 is not None:
                            sol = S(h=h,a=a,g=g,lba=lba,uba=uba,x0=x0)
                        
                        # if does not work, try without warmstarting
                        else:
                            sol = S(h=h,a=a,g=g,lba=lba,uba=uba)
                        
                        # return lambda, mu, y
                        return sol['lam_a'][:self.dim['in']],sol['lam_a'][self.dim['in']:],sol['x']
                    
        else:
            raise Exception('Separate mode not implemented yet.')
        
        # save in model
        self.QP._QP__setSolver(local_qp)

        # store computation times (if compile is true)
        if compile:
            self.__compTimes = self.__compTimes | comp_time_dict

    def makeDenseQP(self,p,solver='qpoases',compile=False):

        """
        Given a numerical value of p, this function constructs the dense QP ingredients and sets up the QP solver,
        which can be accessed through QP.denseSolve.
        """

        # compilation options
        if compile:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            options = {}

        # get p_qp parameter
        p_qp_list = self.param['p_qp'].copy()

        # replace p with numerical value (if y lin is present, p is entry 3, otherwise 2)
        if self.QP.y_lin is not None:
            p_qp_list[2] = DM(p)
        else:
            p_qp_list[1] = DM(p)

        # get data from qp_dense function
        G_x,G_u,g_c,Qx,Ru,x_ref,u_ref,Hx,Hu,hx,hu = self.QP._QP__qp_dense(vcat(p_qp_list))

        # turn to DM
        Qx = DM(Qx)
        Ru = DM(Ru)
        x_ref = DM(x_ref)
        u_ref = DM(u_ref)
        Hx = DM(Hx)
        Hu = DM(Hu)
        hx = DM(hx)
        hu = DM(hu)

        # get initial state
        x = p_qp_list[0]

        # create propagation of initial state
        x_prop = G_x@x+g_c

        # create cost
        Q = G_u.T@Qx@G_u + Ru
        q = G_u.T@Qx@(x_prop - x_ref) - Ru@u_ref

        # create inequality constraint matrices
        G = vertcat(Hx@G_u,Hu)
        uba = vertcat(hx-Hx@x_prop,hu)
        lba = -inf*DM.ones(uba.shape)

        # initialize list of symbolic outputs
        out_list_symbolic = [Q,q,G,lba,uba,G_u,G_x,g_c]
        out_list_symbolic_names = ['Q','q','G','lba','uba','G_u','G_x','g_c']

        # re-create function
        start = time.time()
        QP_func = Function('QP_dense',[vcat(self.param['p_qp'])],out_list_symbolic,['p'],out_list_symbolic_names,options)
        comp_time_dict = {'QP_dense':time.time()-start}

        # implement QP using conic interface to retrieve multipliers
        qp = {}
        qp['h'] = Q.sparsity()
        qp['a'] = G.sparsity()

        # vector specifying which constraints are equalities
        is_equality = [False]*self.dim['in']

        # add existing options to compile function S
        start = time.time()
        match solver:
            case 'gurobi':
                S = conic('S','gurobi',qp,{'gurobi':{'OutputFlag':0},'equality':is_equality})
            case 'cplex':
                S = conic('S','cplex',qp,{'cplex':{'CPXPARAM_Simplex_Display': 0,'CPXPARAM_ScreenOutput': 0},'equality':is_equality})
            case 'qpoases':
                S = conic('S','qpoases',qp,options|{'printLevel':'none','equality':is_equality})
            case 'osqp':
                S = conic('S','osqp',qp,options|{'osqp':{'verbose':False},'equality':is_equality})
            case 'daqp':
                S = conic('S','daqp',qp,options)
            case 'qrqp':
                S = conic('S','qrqp',qp,options|{'print_iter':False,'equality':is_equality})
        comp_time_dict = comp_time_dict | {'QP_dense':time.time()-start}

        # define qp solver
        def local_qp(p_qp,x0=None,lam=None,mu=None):

            # get data from qp_dense function
            Q,q,G,lba,uba,G_u,G_x,g_c = QP_func(p_qp)

            # get initial condition
            x = p_qp[:self.dim['x']]

            # solve QP with warmstarting
            if x0 is not None:
                
                # extract input
                u0 = x0[self.dim['x']*self.dim['N']:]

                # solve
                u = S(h=Q,a=G,g=q,lba=lba,uba=uba,x0=u0)['x']
            
            # if does not work, try without warmstarting
            else:
                u = S(h=Q,a=G,g=q,lba=lba,uba=uba)['x']

            # create state prediction
            x = G_x@x + G_u@u + g_c
            
            # return lambda, mu, y
            return DM(self.dim['in'],1),DM(self.dim['eq'],1),vertcat(x,u,DM(self.dim['eps'],1))
        
        # save in model
        self.QP._QP__setDenseSolver(local_qp)

        # store computation times (if compile is true)
        if compile:
            self.__compTimes = self.__compTimes | comp_time_dict

    def __makeConsJac(self,gamma=0.001,tol=8,compile=False):
        
        """
        This function sets up the functions that compute the conservative jacobian of the QP solution.
        Specifically, two functions are set in the QP class:

            - J: this function takes in the dual variables lam, mu, and the parameters p_QP necessary 
                 to setup the QP (including pf_t), and returns the following quantities in a list:

                 - J_F_z: conservative jacobian of dual fixed point condition wrt z
                 - J_F_p: conservative jacobian of dual fixed point condition wrt p_t
                 - J_y_p: conservative jacobian of primal variable wrt p_t
                 - J_y_z_mat: conservative jacobian of primal variable wrt z

            - J_y_p: this function takes in lam, mu, p_QP (including pf_t), and optionally t (which
              defaults to 1), and returns the inner product between the conservative jacobian J_y_p
              of y wrt p_t and t.
        """

        # compilation options
        if compile:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            options = {}

        # get symbolic variable type
        MSX = self.__MSX
        
        # check if p was passed (required to compute conservative jacobian)
        if 'p_qp' not in self.QP.param:
            raise Exception('Parameter p_qp is required to compute conservative jacobian.')
            
        # turn to column vector
        p = vcat(self.QP.param['p_qp'])
        p_full = [p]

        # check if pf was passed
        if 'pf' in self.param:
            pf = vcat(self.param['pf_t'])
            # if so, add to p_full
            p_full.append(pf)

        # turn p_full into column vector
        p_full = vcat(p_full)        

        # extract multipliers
        lam = self.param['lam']
        mu = self.param['mu']

        # create list of inputs and their names
        dual_params = [lam,mu,p_full]
        dual_params_names = ['lam','mu','p']

        # extract dimensions
        n_z = self.dim['z']
        n_eq = self.dim['eq']

        # extract dual data
        H = self.QP.ingredients['H']
        h = self.QP.ingredients['h']

        # extract QP data
        Qinv = self.QP.ingredients['Qinv']
        q = self.QP.ingredients['q']
        F = self.QP.ingredients['F']
        G = self.QP.ingredients['G']

        # compute conservative jacobian of projector
        J_Pc = diag(vertcat(vec(sign(lam)),MSX.ones(n_eq,1)))
        J_F_z = J_Pc@(MSX.eye(n_z)-gamma*H)-MSX.eye(n_z)
        J_F_p = - gamma*J_Pc@( jacobian(H@vertcat(lam,mu)+h,p) )

        # compute conservative jacobian of primal variable
        y = -Qinv@(G.T@lam+F.T@mu+q)
        J_y_p = jacobian(y,p)
        J_y_z_mat = -Qinv@horzcat(G.T,F.T)

        # sparsify
        try:
            J_Pc = cse(sparsify(J_Pc))
            J_F_z = cse(sparsify(J_F_z))
            J_F_p = cse(sparsify(J_F_p))
            y = cse(sparsify(y))
            J_y_p = cse(sparsify(J_y_p))
            J_y_z_mat = cse(sparsify(J_y_z_mat))
        except:
            pass

        # stack all parameters
        dual_outs = [J_F_z,J_F_p,J_y_p,J_y_z_mat]
        dual_outs_names = ['J_F_z','J_F_p','J_y_p','J_y_z_mat']

        # turn into function
        start = time.time()
        J = Function('J',dual_params,dual_outs,dual_params_names,dual_outs_names,options)
        comp_time_dict = {'J':time.time()-start}

        def J_y_p(lam,mu,p_qp,t=1):

            # round lambda (always first entry in dual_params) to avoid numerical issues
            lam = np.round(np.array(fmax(lam,0)),tol)
            
            # get all conservative jacobian and matrices
            J_F_z,J_F_p,J_y_p,J_y_z_mat = J(lam,mu,p_qp)

            # get conservative jacobian of dual solution
            A = -solve(J_F_z,J_F_p@t,'csparse')

            # return conservative jacobian of primal
            return J_y_p@t+J_y_z_mat@A

        # save in QP
        self.QP._QP__set_J(J)
        self.QP._QP__set_J_y_p(J_y_p)

        # store computation times (if compile is true)
        if compile:
            self.__compTimes = self.__compTimes | comp_time_dict


    ### UPPER-LEVEL COST FUNCTION -------------------------------------------

    @property
    def upperLevel(self):
        return self.__upperLevel
    
    def makeUpperLevel(self,p=None,pf=None,idx_p=None,idx_pf=None,T=None):

        # get symbolic variable type
        MSX = self.__MSX
    
        if T is None:
            try:
                T = self.mpc.N
            except:
                raise Exception('Horizon of MPC was not set up.')
        else:
            if not isinstance(T,int):
                raise Exception('Horizon of upper level must be an integer.')
            
        # add to dimensions
        self.__addDim({'T': T})

        # check that necessary parameters have already been setup
        if p is None:
            if 'p_t' not in self.QP.param:
                raise Exception('Parameter p is required to compute upper-level cost function.')
            else:
                p = self.QP.param['p_t']
        if pf is None:
            if 'pf_t' in self.QP.param:
                pf = self.QP.param['pf_t']
        if 'u' not in self.dim:
            raise Exception('Parameter u is required to compute upper-level cost function.')
        if 'x' not in self.dim:
            raise Exception('Parameter x is required to compute upper-level cost function.')
        if 'y' not in self.dim:
            raise Exception('Parameter y is required to compute upper-level cost function.')
        
        # if idx_p was not passed, assume p if time-invariant
        if idx_p is None:
            idx_p = lambda t: range(0,p.shape[0])

        # setup parameters
        self.upperLevel._upperLevel__set_p(p)
        self.__addDim({'p': p.shape[0]})

        # create symbolic jacobian of cost function wrt p
        self.upperLevel._upperLevel__set_Jp(MSX.sym('J_p',self.dim['p'],1))

        # create symbolic variable representing the iteration number
        self.upperLevel._upperLevel__set_k(MSX.sym('k',1,1))

        # check that idx_p returns the correct dimension
        if self.param['p'][idx_p(0)].shape[0] != self.param['p_t'].shape[0]:
            raise Exception('Indexing function idx_p does not return the correct dimension.')
        
        # store in upperLevel
        self.upperLevel._upperLevel__updateIdx({'p':idx_p})
        
        if pf is not None:
            self.upperLevel._upperLevel__set_pf(pf)
            self.__addDim({'pf': pf.shape[0]})
            # if idx_pf was not passed, assume pf if time-invariant
            if idx_pf is None:
                idx_pf = lambda t: range(0,pf.shape[0])
            
            # store in upperLevel
            self.upperLevel._upperLevel__updateIdx({'pf':idx_pf})

            # if pf_t is present also in QP, check that idx_p returns the correct dimension
            if 'pf_t' in self.QP.param:
                if self.param['pf'][idx_pf(0)].shape[0] != self.param['pf_t'].shape[0]:
                    raise Exception('Indexing function idx_p does not return the correct dimension.')

        # get indices of y required for next MPC call
        if self.mpc.options['linearization'] == 'trajectory':
            y_idx = lambda t: self.QP.idx['out']['y']
            self.upperLevel._upperLevel__updateIdx({'y_next':y_idx})
        elif self.mpc.options['linearization'] == 'initial_state':
            y_idx = lambda t: self.QP.idx['out']['u1']
            self.upperLevel._upperLevel__updateIdx({'y_next':y_idx})
        
        # create function that sets up the necessary inputs to the QP
        def QPVarSetup(x,y,p,pf,t):

            # get optional input list
            inputs = [y,p,pf]
            input_names = ['y_next','p','pf']

            # output list
            out = [x]
            
            # loop through inputs
            for k in range(len(inputs)):
                
                # if an idx range has been passed, it means
                # that the k-th optional input is needed
                if input_names[k] in self.upperLevel.idx:

                    # all inputs should be column vectors
                    out.append(DM(inputs[k])[self.upperLevel.idx[input_names[k]](t)])

            return vcat(out)
        
        def JacVarSetup(J_x_p,J_y_p,t):
            
            # get entries of p
            J_p = DM.eye(self.dim['p'])[self.upperLevel.idx['p'](t),:]

            # get entries o y
            if self.mpc.options['linearization'] == 'trajectory' or self.mpc.options['linearization'] == 'initial_state':
                J_y = J_y_p[self.upperLevel.idx['y_next'](t),:]
            else:
                J_y = DM(0,self.dim['p'])

            return vertcat(J_x_p,J_y,J_p)
        
        # save in upperLevel
        self.upperLevel._upperLevel__updateIdx({'qp':QPVarSetup,'jac':JacVarSetup})
        self.upperLevel._upperLevel__set_x_cl(MSX.sym('x_cl',self.dim['x'],T+1))
        self.upperLevel._upperLevel__set_u_cl(MSX.sym('u_cl',self.dim['u'],T))
        self.upperLevel._upperLevel__set_y_cl(MSX.sym('y_cl',self.dim['y'],T))
        self.upperLevel._upperLevel__set_e_cl(MSX.sym('y_cl',self.dim['eps'],T))

    def setUpperLevelCost(self,cost,track_cost=None,cst_viol=None):

        # get symbolic variable type
        MSX = self.__MSX

        # check if tracking cost function is passed, if not, set it equal to cost
        if track_cost is None:
            track_cost = cost
        
        # check if constraint violation function is passed, if not, set to zero
        if cst_viol is None:
            cst_viol = MSX(1,1)

        # helper function to check if symbolic variables are correct
        def symvar_str(expr):
            return [str(v) for v in symvar(expr)]

        # check that the variables appearing in track_cost and cst_viol are
        # contained within the variables appearing in cost
        if len(set(symvar_str(track_cost)) - set(symvar_str(cost))) > 0:
            raise Exception('Variables in tracking cost are not contained in full cost.')
        if len(set(symvar_str(cst_viol)) - set(symvar_str(cost))) > 0:
            raise Exception('Variables in constraint violation are not contained in full cost.')

        # check if cost is a scalar symbolic expression
        if not isinstance(cost,MSX):
            raise Exception('Cost must be a symbolic expression.')
        if cost.shape[0] != 1 or cost.shape[1] != 1:
            raise Exception('Cost must be scalar.')
        
        # extract upper-level parameters
        p = self.upperLevel.p
        x_cl = self.upperLevel.x_cl
        u_cl = self.upperLevel.u_cl
        y_cl = self.upperLevel.y_cl
        
        # check if cost contains variables that are not not p,x_cl,u_cl,y_cl
        if len(set(symvar_str(cost)) - set(symvar_str(vcat([p,vec(x_cl),vec(u_cl),vec(y_cl)])))) > 0:
            raise Exception('Cost contains variables that are not p,x_cl,u_cl,y_cl.')

        # get number of parameters that are being differentiated
        n_p = self.dim['p']

        # initialize list of parameters that enter the cost, and their names
        param_in = []

        # initialize list of indices representing entries of the parameters
        # that enter the cost
        param_idx = []
        
        # check if x_cl appears in the cost
        try:
            # this one works with SX variables
            x_cl_cost_idx_temp = DM(sum1(jacobian(vcat(symvar(cost)),vec(x_cl)))).T
            x_cl_cost_idx = np.array(x_cl_cost_idx_temp).nonzero()[0]
        except:
            # this one works with MX variables
            if 'x_cl' in symvar_str(cost):
                x_cl_cost_idx = np.arange(0,vec(x_cl).shape[0])
            else:
                x_cl_cost_idx = []
            pass

        if len(x_cl_cost_idx) > 0:

            # extract only the relevant entries of x_cl
            x_cl_cost = vec(x_cl)[x_cl_cost_idx]

            # compute jacobian of cost with respect to x_cl
            J_cost_x = jacobian(cost,vec(x_cl_cost))

            # create symbolic variable for jacobian of x_cl_cost wrt p
            J_x_p = MSX.sym('J_x_p',x_cl_cost.shape[0],n_p)

            # add to list of parameters entering the cost
            param_in.append(x_cl_cost)
            param_idx.append(x_cl_cost_idx)

        else:

            # if no x_cl entries appear in the cost, set J_cost_x and x_cl_cost as an empty matrix
            J_cost_x = MSX(1,n_p)

            # create symbolic variable for jacobian of x_cl_cost wrt p (needed for compatibility)
            J_x_p = MSX.sym('J_x_p',1,1)

            # add a None to the parameter list
            param_in.append(None)

        # check if u_cl appears in the cost
        try:
            # this one works with SX variables
            u_cl_cost_idx_temp = DM(sum1(jacobian(vcat(symvar(cost)),vec(u_cl)))).T
            u_cl_cost_idx = np.array(u_cl_cost_idx_temp).nonzero()[0]
        except:
            # this one works with MX variables
            if 'u_cl' in symvar_str(cost):
                u_cl_cost_idx = range(0,vec(u_cl).shape[0])
            else:
                u_cl_cost_idx = []
            pass

        if len(u_cl_cost_idx) > 0:

            # extract only the relevant entries of u_cl
            u_cl_cost = vec(u_cl)[u_cl_cost_idx]

            # compute jacobian of cost with respect to u_cl
            J_cost_u = jacobian(cost,vec(u_cl_cost))

            # create symbolic variable for jacobian of u_cl_cost wrt p
            J_u_p = MSX.sym('J_u_p',u_cl_cost.shape[0],n_p)

            # add to list of parameters entering the cost
            param_in.append(u_cl_cost)
            param_idx.append(u_cl_cost_idx)

        else:

            # if no u_cl entries appear in the cost, set J_cost_u and u_cl_cost as an empty matrix
            J_cost_u = MSX(1,n_p)

            # create symbolic variable for jacobian of u_cl_cost wrt p (needed for compatibility)
            J_u_p = MSX.sym('J_u_p',1,1)

            # add a None to the parameter list
            param_in.append(None)

        # check what entries of y_cl appear in the cost
        try:
            # this one works with SX variables
            y_cl_cost_idx_temp = DM(sum1(jacobian(vcat(symvar(cost)),vec(y_cl)))).T
            y_cl_cost_idx = np.array(y_cl_cost_idx_temp).nonzero()[0]
        except:
            # this one works with MX variables
            if 'y_cl' in symvar_str(cost):
                y_cl_cost_idx = range(0,vec(y_cl).shape[0])
            else:
                y_cl_cost_idx = []
            pass

        if len(y_cl_cost_idx) > 0:

            # extract only the relevant entries of y_cl
            y_cl_cost = vec(y_cl)[y_cl_cost_idx]

            # compute jacobian of cost with respect to y_cl
            J_cost_y = jacobian(cost,vec(y_cl_cost))

            # create symbolic variable for jacobian of y_cl_cost wrt p
            J_y_p = MSX.sym('J_y_p',y_cl_cost.shape[0],n_p)

            # add to list of parameters entering the cost
            param_in.append(y_cl_cost)
            param_idx.append(y_cl_cost_idx)

        else:

            # if no y_cl entries appear in the cost, set J_cost_y and y_cl_cost as an empty matrix
            J_cost_y = MSX(1,n_p)

            # create symbolic variable for jacobian of y_cl_cost wrt p (needed for compatibility)
            J_y_p = MSX.sym('J_y_p',1,1)

            # add a None to the parameter list
            param_in.append(None)

        # check if p appears in the cost
        try:
            # this one works with SX variables
            p_cost_idx_temp = DM(sum1(jacobian(vcat(symvar(cost)),vec(p)))).T
            p_cost_idx = np.array(p_cost_idx_temp).nonzero()[0]
        except:
            # this one works with MX variables
            if 'p_t' in symvar_str(cost):
                p_cost_idx = range(0,vec(p).shape[0])
            else:
                p_cost_idx = []
            pass

        if len(p_cost_idx) > 0:
            
            # extract only the relevant entries of p
            p_cost = vec(p)[p_cost_idx]

            # compute jacobian of cost with respect to p
            J_cost_p = jacobian(cost,vec(p_cost))

            # add to list of parameters entering the cost
            param_in.append(p_cost)
            param_idx.append(p_cost_idx)

        else:
            
            # if no p entries appear in the cost, set J_cost_p and p_cost as an empty matrix
            J_cost_p = MSX(1,n_p)

            # add a None to the parameter list
            param_in.append(None)
        
        # create function that retrieves only the indices that enter the cost given
        # the full vectors
        def getCostIdx(x_cl,u_cl,y_cl,p):

            # get input list
            inputs = [x_cl,u_cl,y_cl,p]
            
            # initialize output list
            out = []

            # loop through all parameters
            for i in range(len(inputs)):

                # check if the parameter is empty
                if param_in[i] is not None:

                    # extract the relevant indices
                    out.append(vec(inputs[i])[param_idx[i]])

            # return as list
            return vcat(out)
        
        # create function that retrieves the jacobians that are needed to compute the
        # full jacobian of the cost function, given the full jacobian
        def getCostJacobian(J_x_p,J_u_p,J_y_p):

            # get input list
            inputs = [J_x_p,J_u_p,J_y_p]

            # initialize output list
            out = []

            # loop through all parameters
            for i in range(len(inputs)):

                # check if the parameter is empty
                if param_in[i] is not None:

                    # extract the relevant jacobians
                    out.append(inputs[i][param_idx[i],:])

                    # TODO: I think this fails if the parameter is a scalar

                else:

                    # add a None to the parameter list
                    out.append(0)

            # return as list
            return out

        # parameters that are necessary for cost
        cost_in = vcat([vec(item) for item in param_in if item is not None])

        # quick test to see if things are working
        try:
            if sum1(getCostIdx(x_cl,u_cl,y_cl,p) - cost_in) != 0:
                raise Exception('Error in getCostIdx function.')
        except:
            # TODO make a test for this in MX mode
            pass

        # create cost functions in two steps
        cost_func_temp = Function('cost',[cost_in],[cost,track_cost,cst_viol])
        def cost_func(S):
            # return cost_func_temp(getCostIdx(S.x,S.u,S.y,S.p[:,-1]))
            return cost_func_temp(getCostIdx(S.x,S.u,S.y,S.p))

        # create full jacobian functions in two steps
        J_cost = J_cost_p + J_cost_x@J_x_p + J_cost_u@J_u_p + J_cost_y@J_y_p
        J_cost_func_temp = Function('J_cost',[cost_in,J_x_p,J_u_p,J_y_p],[J_cost.T])
        def J_cost_func(S):

            # get true input cost
            cost_in = getCostIdx(S.x,S.u,S.y,S.p)
            # cost_in = getCostIdx(S.x,S.u,S.y,S.p[:,-1])

            # get true jacobians
            J_x_p,J_u_p,J_y_p = getCostJacobian(S.Jx,S.Ju,S.Jy)

            return J_cost_func_temp(cost_in,J_x_p,J_u_p,J_y_p)
        
        # store in upper level
        self.upperLevel._upperLevel__set_cost(cost_func)
        self.upperLevel._upperLevel__set_J_cost(J_cost_func)

    def setUpperLevelAlg(self,p_next,psi_init=None,psi_next=None,psi=None):

        # get symbolic variable type
        MSX = self.__MSX

        # parse inputs
        if psi_init is None:
            psi_init = MSX(0)
        elif not isinstance(psi_init,MSX):
            raise Exception('psi_init if of the wrong symbolic type.')
        if psi_next is None:
            psi_next = MSX(0)
        elif not isinstance(psi_next,MSX):
            raise Exception('psi_next if of the wrong symbolic type.')
        if psi is None:
            psi = MSX.sym('psi',1,1)
        elif not isinstance(psi,MSX):
            raise Exception('psi if of the wrong symbolic type.')

        # check that p_next returns a vector with the same dimension as p
        if p_next.shape != self.param['p'].shape:
            raise Exception('Parameters p and p_next must have the same dimension.')
        
        # check if pf is present
        if 'pf' not in self.param:
            pf = MSX.sym('pf',1,1)
        else:
            pf = self.param['pf']

        # construct list of parameters on which p_next is allowed to depend
        param_p_next = [self.param['p'],pf,psi,self.param['k'],self.param['Jp']]

        # helper function to check if symbolic variables are correct
        def symvar_str(expr):
            return [str(v) for v in symvar(expr)]

        # check if p_next is a function of p, pf, psi, k, and Jp
        if len(set(symvar_str(p_next)) - set(symvar_str(vcat(param_p_next)))) > 0:
            raise Exception('Parameter p_next must depend on p, pf, psi, k, and Jp.')
        
        # check that psi_init and psi_next have the same dimension as psi
        if psi_init.shape != psi.shape:
            raise Exception('Initial value of psi must have the same dimension as psi.')
        if psi_next.shape != psi.shape:
            raise Exception('Next value of psi must have the same dimension as psi.')
        
        # check that psi_next is a function of p, pf, psi, k, and Jp
        if len(set(symvar_str(psi_next)) - set(symvar_str(vcat(param_p_next)))) > 0:
            raise Exception('Parameter p_next must depend on p, pf, psi, k, and Jp.')
        
        # check that psi is a function of p, pf, and Jp
        if len(set(symvar_str(psi_init)) - set(symvar_str(vcat([self.param['p'],pf,self.param['Jp']])))):
            raise Exception('Initial value of psi must depend on p, pf, and Jp.')
        
        # create casadi function
        psi_next_func = Function('psi_next',[self.param['p'],pf,psi,self.param['k'],self.param['Jp']],[psi_next],['p','pf','psi','k','Jp'],['psi_next'])
        psi_init_func = Function('psi_init',[self.param['p'],pf,self.param['Jp']],[psi_init],['p','pf','Jp'],['psi_init'])
        p_next_func = Function('p_next',[self.param['p'],pf,psi,self.param['k'],self.param['Jp']],[p_next],['p','pf','psi','k','Jp'],['p_next'])

        # if pf is not passed, wrap a python function around that defaults pf to 0
        if 'pf' not in self.param:
            def p_next_func_py(p,psi,k,Jp,pf):
                if pf is None:
                    pf = 0
                return p_next_func(p,pf,psi,k,Jp)
            def psi_next_func_py(p,psi,k,Jp,pf):
                if pf is None:
                    pf = 0
                return psi_next_func(p,pf,psi,k,Jp)
            def psi_init_func_py(p,Jp,pf):
                if pf is None:
                    pf = 0
                return psi_init_func(p,pf,Jp)
        else:
            p_next_func_py = p_next_func
            psi_next_func_py = psi_next_func
            psi_init_func_py = psi_init_func

        # store in upperLevel
        self.upperLevel._upperLevel__setAlg({'psi_next':psi_next_func_py,'psi_init':psi_init_func_py,'p_next':p_next_func_py})


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
        if self.mpc.options['linearization'] == 'trajectory':
            
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
        if self.mpc.options['linearization'] == 'initial_state':
            
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
                    - warmstart_first_qp:True (default) if the first QP should be solved twice (with
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