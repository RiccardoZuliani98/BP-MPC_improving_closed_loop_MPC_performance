from casadi import *
from BPMPC.dynamics import dynamics
from BPMPC.QP import QP
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
        return self.QP.options | self.upperLevel.options

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

    ### MPC -----------------------------------------------------------------

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

        # check if linearization option was passed
        if 'linearization' not in options:
            options['linearization'] = 'trajectory'

        # check if a model was passed
        if model is None:
            A_list, B_list, c_list, y_lin = self.linearize(N,linearization=options['linearization'])
            model = {'A':A_list,'B':B_list,'c':c_list,'y_lin':y_lin,'x0':self.param['x']}
            # check if model is affine
            if y_lin is None:
                options['linearization'] = 'none'

        # extract y_lin from model if present
        if 'y_lin' in model:
            y_lin = model['y_lin']
        else:
            y_lin = None

        # create MPC class
        if self.__MSX == SX:
            MSX = 'SX'
        if self.__MSX == MX:
            MSX = 'MX'
        mpc = MPC(N,model,cost,cst,MSX)
        
        # create QP ingredients
        G,g,F,f,Q,Qinv,q,idx,denseQP = mpc.MPC2QP()

        # create QP
        self.makeQP(G,g,F,f,Q,Qinv,q,idx,y_lin,denseQP,p,pf,options)


    ### QP ------------------------------------------------------------------
    
    @property
    def QP(self):
        return self.__QP


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
        if self.QP.options['linearization'] == 'trajectory':
            y_idx = lambda t: self.QP.idx['out']['y']
            self.upperLevel._upperLevel__updateIdx({'y_next':y_idx})
        elif self.QP.options['linearization'] == 'initial_state':
            y_idx = lambda t: self.QP.idx['out']['u1']
            self.upperLevel._upperLevel__updateIdx({'y_next':y_idx})
        
        # create function that sets up the necessary inputs to the QP
        def QPVarSetup(x,y,p,pf,t):

            # get optional input list
            inputs = [y,p,pf]
            input_names = ['y_next','p','pf']

            # output list
            out = [x]

            # faster with list comprehension
            # out.append([DM(input)[self.upperLevel.idx[name](t)] for input, name in zip(inputs,input_names) if name in self.upperLevel.idx])
            
            # loop through inputs
            for input, name in zip(inputs,input_names):
                
                # if an idx range has been passed, it means
                # that the k-th optional input is needed
                if name in self.upperLevel.idx:

                    # all inputs should be column vectors
                    out.append(DM(input)[self.upperLevel.idx[name](t)])

            return vcat(out)
        
        def JacVarSetup(J_x_p,J_y_p,t):
            
            # get entries of p
            J_p = DM.eye(self.dim['p'])[self.upperLevel.idx['p'](t),:]

            # get entries o y
            if self.QP.options['linearization'] == 'trajectory' or self.QP.options['linearization'] == 'initial_state':
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