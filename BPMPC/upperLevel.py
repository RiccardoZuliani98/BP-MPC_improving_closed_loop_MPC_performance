import casadi as ca
from BPMPC.QP import QP
from typing import Callable, Union
import numpy as np
from BPMPC.symb import Symb

class UpperLevel:

    # def __init__(self):

    #     # initialize variables
    #     self.__p = None
    #     self.__pf = None
    #     self.__Jp = None
    #     self.__k = None
    #     self.__x_cl = None
    #     self.__u_cl = None
    #     self.__y_cl = None
    #     self.__e_cl = None
    #     self.__cost = None
    #     self.__J_cost = None
    #     self.__init = {'p':None, 'pf':None}
    #     self.__idx = {}
    #     self.__alg = None
    #     pass

    def __init__(self,
                 p: ca.SX,
                 T: int,
                 MPC: QP,
                 pf: ca.SX = None,
                 idx_p: Callable[[int], Union[range, np.ndarray]] = None,
                 idx_pf: Callable[[int], Union[range, np.ndarray]] = None):

        # create symbolic variable
        self._sym = Symb()
            
        # add to dimensions
        self._sym.addDim({'T': T})

        # add p to set of symbolic variables
        self._sym.addVar('p',p)

        # create symbolic jacobian of cost function wrt p
        self._sym.addVar('J_p',ca.SX.sym('J_p',self.dim['p'],1))

        # create symbolic variable representing the iteration number
        self._sym.addVar('k',ca.SX.sym('k',1,1))

        # save all symbolic variables
        self._sym.addVar('x_cl',ca.SX.sym('x_cl',MPC.dim['x'],T+1))
        self._sym.addVar('u_cl',ca.SX.sym('u_cl',MPC.dim['u'],T))
        self._sym.addVar('y_cl',ca.SX.sym('y_cl',MPC.dim['y'],T))
        if 'eps' in MPC.dim:
            self._sym.addVar('e_cl',ca.SX.sym('e_cl',MPC.dim['eps'],T))

        # if idx_p was not passed, assume p if time-invariant
        if idx_p is None:
            idx_p = lambda t: range(0,p.shape[0])

        # get dimension of qp parameter p
        n_p_qp_t = MPC.param['p_t'].shape[0]

        # check that idx_p returns the correct dimension
        assert all([self.param['p'][idx_p(t)].shape[0] == n_p_qp_t for t in range(T)]), 'Indexing function idx_p does not return the correct dimension.'
        
        # store in upperLevel
        self._idx = {'p':idx_p}
        
        # check if pf is passed
        if pf is not None:

            # store symbolic variable
            self._sym.addVar('pf',pf)

            # if idx_pf was not passed, assume pf if time-invariant
            if idx_pf is None:
                idx_pf = lambda t: range(0,pf.shape[0])
            
            # store in upperLevel
            self._idx = self._idx | {'pf':idx_pf}

            # if pf_t is present also in QP, check that idx_p returns the correct dimension
            if 'pf_t' in MPC.param:

                # get dimension of qp parameter pf_t
                n_pf_qp_t = MPC.param['pf_t'].shape[0]

                # check that idx_p returns the correct dimension
                assert all([self.param['pf'][idx_pf(t)].shape[0] == n_pf_qp_t for t in range(T)]), 'Indexing function idx_pf does not return the correct dimension.'

        # flag saying that a linearization trajectory is needed
        y_idx = False

        # get linearization trajectory index if one is present in MPC
        if 'y_next' in MPC.idx['out']:

            # add index to indices of UpperLevel
            self._idx = self._idx | {'y_next',MPC.idx['y']}

            # set flag to true
            y_idx = True
        
        # create function that sets up the necessary inputs to the QP
        def QPVarSetup(x,y,p,pf,t):

            # get optional input list
            inputs = [y,p,pf]
            input_names = ['y_next','p','pf']

            # output list
            out = [x]

            # loop through inputs
            for input, name in zip(inputs,input_names):
                
                # if an idx range has been passed, it means
                # that the k-th optional input is needed
                if name in self._idx:

                    # all inputs should be column vectors
                    out.append(ca.DM(input)[self._idx[name](t)])

            return ca.vcat(out)
        
        def JacVarSetup(J_x_p,J_y_p,t):
            
            # get entries of p
            J_p = ca.DM.eye(self.dim['p'])[self._idx['p'](t),:]

            # get entries o y
            J_y = J_y_p[self.upperLevel.idx['y_next'](t),:] if y_idx else ca.DM(0,self.dim['p'])

            return ca.vertcat(J_x_p,J_y,J_p)
        
        # save in upperLevel
        self._idx = self._idx | {'qp':QPVarSetup,'jac':JacVarSetup}

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

    @property
    def alg(self):
        return self.__alg
    
    def __setAlg(self,value):
        self.__alg = value

    @property
    def p(self):
        return self.__p
    
    def __set_p(self, value):
        if type(value) is not self.__MSX:
            raise Exception('p is of the wrong symbolic type.')
        self.__p = value

    @property
    def pf(self):
        return self.__pf
    
    def __set_pf(self, value):
        if type(value) is not self.__MSX:
            raise Exception('pf is of the wrong symbolic type.')
        self.__pf = value

    @property
    def Jp(self):
        return self.__Jp
    
    def __set_Jp(self, value):
        if type(value) is not self.__MSX:
            raise Exception('Jp is of the wrong symbolic type.')
        self.__Jp = value

    @property
    def k(self):
        return self.__k
    
    def __set_k(self, value):
        if type(value) is not self.__MSX:
            raise Exception('k is of the wrong symbolic type.')
        self.__k = value

    @property
    def x_cl(self):
        return self.__x_cl
    
    def __set_x_cl(self, value):
        if type(value) is not self.__MSX:
            raise Exception('x_cl is of the wrong symbolic type.')
        self.__x_cl = value

    @property
    def u_cl(self):
        return self.__u_cl

    def __set_u_cl(self, value):
        if type(value) is not self.__MSX:
            raise Exception('u_cl is of the wrong symbolic type.')
        self.__u_cl = value

    @property
    def y_cl(self):
        return self.__y_cl

    def __set_y_cl(self, value):
        if type(value) is not self.__MSX:
            raise Exception('y_cl is of the wrong symbolic type.')
        self.__y_cl = value

    @property
    def e_cl(self):
        return self.__e_cl

    def __set_e_cl(self, value):
        if type(value) is not self.__MSX:
            raise Exception('e_cl is of the wrong symbolic type.')
        self.__e_cl = value

    @property
    def cost(self):
        return self.__cost
    
    def __set_cost(self, value):
        self.__cost = value

    @property
    def J_cost(self):
        return self.__J_cost
    
    def __set_J_cost(self, value):
        self.__J_cost = value

    @property
    def init(self):
        return {k: v for k, v in self.__init.items()}
    
    def __setInit(self,value):
        self.__init = self.__init | self.__checkInit(value)
    
    def __checkInit(self, value):

        # preallocate output
        out = {}

        if 'p' in value:

            if 'p' not in self.param:
                raise Exception('Define parameter p before setting its initial value.')

            # turn into DM
            p_init = DM(value['p'])

            if p_init.shape == self.p.shape:
                out = out | {'p':p_init}
            else:
                raise Exception('p must have the same shape as the initial parameter.')
            
        if 'pf' in value:

            if 'pf' not in self.param:
                raise Exception('Define parameter pf before setting its initial value.')

            # turn into DM
            pf_init = DM(value['pf'])

            if pf_init.shape == self.pf.shape:
                out = out | {'pf':pf_init}
            else:
                raise Exception('pf must have the same shape as the final parameter.')
            
        return out

    @property   
    def idx(self):
        return self.__idx

    def __updateIdx(self, idx):
        self.__idx = self.__idx | idx

    @property
    def param(self):
        return {k: v for k, v in {
            'p': self.__p,
            'pf':self.__pf,
            'Jp':self.__Jp,
            'k':self.__k,
            'x_cl': self.__x_cl,
            'u_cl': self.__u_cl,
            'y_cl': self.__y_cl,
            'e_cl': self.__e_cl,
        }.items() if v is not None}
    
    # overwrite the __dir__ method
    def __dir__(self):
        return [attr for attr in super().__dir__() if not attr.startswith('_UpperLevel__')]