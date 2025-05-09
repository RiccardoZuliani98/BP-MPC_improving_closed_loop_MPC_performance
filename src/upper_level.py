import casadi as ca
from src.qp import QP
from typing import Callable, Union, Optional
import numpy as np
from src.symb import Symb
from typeguard import typechecked

"""
TODO
* are JacVarSetup and QPVarSetup slow?
* descriptions
"""


class UpperLevel:

    @typechecked
    def __init__(
            self,
            p: ca.SX,
            horizon: int,
            mpc: QP,
            pf: Optional[ca.SX] = None,
            idx_p: Optional[Callable[[int], Union[range, np.ndarray]]] = None,
            idx_pf: Optional[Callable[[int], Union[range, np.ndarray]]] = None
        ):

        # create symbolic variable
        self._sym = Symb()
            
        # add to dimensions
        self._sym.addDim('T',horizon)

        # add p to set of symbolic variables
        self._sym.addVar('p',p)

        # create symbolic jacobian of cost function wrt p
        self._sym.addVar('J_p',ca.SX.sym('J_p',self.dim['p'],1))

        # create symbolic variable representing the iteration number
        self._sym.addVar('k',ca.SX.sym('k',1,1))

        # save all symbolic variables
        self._sym.addVar('x_cl',ca.SX.sym('x_cl',mpc.dim['x'],horizon+1))
        self._sym.addVar('u_cl',ca.SX.sym('u_cl',mpc.dim['u'],horizon))
        self._sym.addVar('y_cl',ca.SX.sym('y_cl',mpc.dim['y'],horizon))
        if 'eps' in mpc.dim:
            self._sym.addVar('e_cl',ca.SX.sym('e_cl',mpc.dim['eps'],horizon))

        # if idx_p was not passed, assume p if time-invariant
        if idx_p is None:
            idx_p = lambda t: range(0,p.shape[0])

        # get dimension of qp parameter p
        n_p_qp_t = mpc.param['p_t'].shape[0]

        # check that idx_p returns the correct dimension
        assert all([self.param['p'][idx_p(t)].shape[0] == n_p_qp_t for t in range(horizon)]), 'Indexing function idx_p does not return the correct dimension.'
        
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
            if 'pf_t' in mpc.param:

                # get dimension of qp parameter pf_t
                n_pf_qp_t = mpc.param['pf_t'].shape[0]

                # check that idx_p returns the correct dimension
                assert all([self.param['pf'][idx_pf(t)].shape[0] == n_pf_qp_t for t in range(horizon)]), 'Indexing function idx_pf does not return the correct dimension.'

        # flag saying that a linearization trajectory is needed
        y_idx = False

        # get linearization trajectory index if one is present in mpc
        if 'y_next' in mpc.idx['out']:

            # add index to indices of UpperLevel
            self._idx = self._idx | {'y_next':mpc.idx['out']['y_next']}

            # set flag to true
            y_idx = True
        
        # create function that sets up the necessary inputs to the QP
        def qp_var_setup(x,y,p_loc,pf_loc,t):

            # get optional input list
            inputs = [y,p_loc,pf_loc]
            input_names = ['y_next','p','pf']

            # output list
            out = [x]

            # loop through inputs
            for inp, name in zip(inputs,input_names):
                
                # if an idx range has been passed, it means
                # that the k-th optional input is needed
                if name in self._idx:

                    # all inputs should be column vectors
                    out.append(ca.DM(inp)[self._idx[name](t)])

            return ca.vcat(out)
        
        def jac_var_setup(j_x_p,j_y_p,t):
            
            # get entries of p
            j_p = ca.DM.eye(self.dim['p'])[self._idx['p'](t),:]

            # get entries o y
            j_y = j_y_p[self.idx['y_next'](t),:] if y_idx else ca.DM(0,self.dim['p'])

            return ca.vertcat(j_x_p,j_y,j_p)
        
        # save in upperLevel
        self._idx = self._idx | {'qp':qp_var_setup,'jac':jac_var_setup}

        # setup more properties
        self._cost = None
        self._J_cost = None
        self._alg = None

    @typechecked
    def set_cost(
            self,
            cost:ca.SX,
            track_cost:Optional[ca.SX]=None,
            cst_viol:Optional[ca.SX]=None
        ):

        # check if tracking cost function is passed, if not, set it equal to cost
        if track_cost is None:
            track_cost = cost
        
        # check if constraint violation function is passed, if not, set to zero
        if cst_viol is None:
            cst_viol = ca.SX(1,1)

        # helper function to check if symbolic variables are correct
        def symvar_str(expr):
            return [str(v) for v in ca.symvar(expr)]

        # check that the variables appearing in track_cost and cst_viol are
        # contained within the variables appearing in cost
        assert len(set(symvar_str(track_cost)) - set(symvar_str(cost))) == 0, 'Variables in tracking cost are not contained in full cost.'
        assert len(set(symvar_str(cst_viol)) - set(symvar_str(cost))) == 0, 'Variables in constraint violation are not contained in full cost.'

        # check if cost is a scalar symbolic expression
        assert cost.shape == (1,1), 'Cost must be scalar.'
        
        # extract upper-level parameters
        p_cl, x_cl, u_cl, y_cl = self.param['p'], self.param['x_cl'], self.param['u_cl'], self.param['y_cl']
        
        # check if cost contains variables that are not p_cl,x_cl,u_cl,y_cl
        assert len(set(symvar_str(cost)) - set(symvar_str(ca.vcat([p_cl,ca.vec(x_cl),ca.vec(u_cl),ca.vec(y_cl)])))) == 0, 'Cost contains variables that are not p,x_cl,u_cl,y_cl.'

        # get number of parameters that are being differentiated
        n_p = self.dim['p']

        # initialize list of parameters that enter the cost, and their names
        param_in = []

        # initialize list of indices representing entries of the parameters
        # that enter the cost
        param_idx = []

        # list containing all Jacobians
        j_param = []
        j_cost = []
        
        # loop through all possible parameters that can enter the cost
        for param, jac_name in zip([x_cl,u_cl,y_cl,p_cl],['J_x_p','J_u_p','J_y_p','J_p_p']):
        
            # check if parameter enters the cost
            param_cost_idx = np.array(ca.DM(ca.sum1(ca.jacobian(ca.vcat(ca.symvar(cost)),ca.vec(param)))).T).nonzero()[0]

            # # this one works with MX variables
            # if 'x_cl' in symvar_str(cost):
            #     x_cl_cost_idx = np.arange(0,ca.vec(x_cl).shape[0])
            # else:
            #     x_cl_cost_idx = []
            # pass

            # if so, get parameters entering the cost
            if len(param_cost_idx) > 0:

                # extract only the relevant entries of x_cl
                param_cost = ca.vec(param)[param_cost_idx]

                # compute jacobian of cost with respect to x_cl
                j_cost.append(ca.jacobian(cost,ca.vec(param_cost)))

                # create symbolic variable for jacobian of x_cl_cost wrt p
                j_param.append(ca.SX.sym(jac_name,param_cost.shape[0],n_p))

                # add to list of parameters entering the cost
                param_in.append(param_cost)
                param_idx.append(param_cost_idx)

            else:

                # if no param entries appear in the cost, set J_cost and param_cost to an empty matrix
                j_cost.append(ca.SX(1,n_p))

                # create symbolic variable for jacobian of param_cost wrt p (needed for compatibility)
                j_param.append(ca.SX.sym(jac_name,1,1))

                # add a None to the parameter list
                param_in.append(None)

        # assign Jacobians
        j_x_p, j_u_p, j_y_p, _ = j_param
        j_cost_x, j_cost_u, j_cost_y, j_cost_p = j_cost

        # create function that retrieves only the indices that enter the cost given
        # the full vectors
        def get_cost_idx(x,u,y,p):

            # get input list
            inputs = [x,u,y,p]
            
            # initialize output list
            out = []

            # loop through all parameters
            for i in range(len(inputs)):

                # check if the parameter is empty
                if param_in[i] is not None:

                    # extract the relevant indices
                    out.append(ca.vec(inputs[i])[param_idx[i]])

            # return as list
            return ca.vcat(out)
        
        # create function that retrieves the Jacobian that are needed to compute the
        # full jacobian of the cost function, given the full jacobian
        def get_cost_jacobian(j_x,j_u,j_y):

            # get input list
            inputs = [j_x,j_u,j_y]

            # initialize output list
            out = []

            # loop through all parameters
            for i in range(len(inputs)):

                # check if the parameter is empty
                if param_in[i] is not None:

                    # extract the relevant Jacobian
                    out.append(inputs[i][param_idx[i],:])

                    # TODO: I think this fails if the parameter is a scalar

                else:

                    # add a None to the parameter list
                    out.append(0)

            # return as list
            return out

        # parameters that are necessary for cost
        cost_in = ca.vcat([ca.vec(item) for item in param_in if item is not None])

        # quick test to see if things are working
        assert ca.sum1(get_cost_idx(x_cl,u_cl,y_cl,p_cl) - cost_in) == 0, 'Error in getCostIdx function.'

        # create cost functions in two steps
        cost_func_temp = ca.Function('cost',[cost_in],[cost,track_cost,cst_viol])
        def cost_func(s):
            # return cost_func_temp(getCostIdx(S.x,S.u,S.y,S.p[:,-1]))
            return cost_func_temp(get_cost_idx(s.x,s.u,s.y,s.p))

        # create full jacobian functions in two steps
        j_cost = j_cost_p + j_cost_x@j_x_p + j_cost_u@j_u_p + j_cost_y@j_y_p
        j_cost_func_temp = ca.Function('J_cost',[cost_in,j_x_p,j_u_p,j_y_p],[j_cost.T])
        def j_cost_func(s):

            # get true input cost
            cost_in_loc = get_cost_idx(s.x,s.u,s.y,s.p)
            # cost_in = getCostIdx(S.x,S.u,S.y,S.p[:,-1])

            # get true Jacobian
            j_x,j_u,j_y = get_cost_jacobian(s.Jx,s.Ju,s.Jy)

            return j_cost_func_temp(cost_in_loc,j_x,j_u,j_y)
        
        # store in upper level
        self._cost = cost_func
        self._J_cost = j_cost_func

    @typechecked
    def set_alg(self,
            p_next,
            psi_init:Optional[ca.SX]=ca.SX(0),
            psi_next:Optional[ca.SX]=ca.SX(0),
            psi:Optional[ca.SX]=ca.SX.sym('psi',1,1)
        ):
        
        # check that p_next returns a vector with the same dimension as p
        assert p_next.shape == self.param['p'].shape, 'Parameters p and p_next must have the same dimension.'
        
        # check if pf is present
        pf = self.param['pf'] if 'pf' in self.param else ca.SX.sym('pf',1,1)

        # construct list of parameters on which p_next is allowed to depend
        param_p_next = [self.param['p'],pf,psi,self.param['k'],self.param['J_p']]

        # helper function to check if symbolic variables are correct
        def symvar_str(expr):
            return [str(v) for v in ca.symvar(expr)]

        # check if p_next is a function of p, pf, psi, k, and Jp
        assert len(set(symvar_str(p_next)) - set(symvar_str(ca.vcat(param_p_next)))) == 0, 'Parameter p_next must depend on p, pf, psi, k, and Jp.'
        
        # check that psi_init and psi_next have the same dimension as psi
        assert psi_init.shape == psi.shape, 'Initial value of psi must have the same dimension as psi.'
        assert psi_next.shape == psi.shape, 'Next value of psi must have the same dimension as psi.'
        
        # check that psi_next is a function of p, pf, psi, k, and Jp
        assert len(set(symvar_str(psi_next)) - set(symvar_str(ca.vcat(param_p_next)))) == 0, 'Parameter p_next must depend on p, pf, psi, k, and Jp.'
        
        # check that psi is a function of p, pf, and Jp
        assert len(set(symvar_str(psi_init)) - set(symvar_str(ca.vcat([self.param['p'],pf,self.param['J_p']])))) == 0, 'Initial value of psi must depend on p, pf, and Jp.'
        
        # create casadi function
        psi_next_func = ca.Function('psi_next',[self.param['p'],pf,psi,self.param['k'],self.param['J_p']],[psi_next],['p','pf','psi','k','Jp'],['psi_next'])
        psi_init_func = ca.Function('psi_init',[self.param['p'],pf,self.param['J_p']],[psi_init],['p','pf','Jp'],['psi_init'])
        p_next_func = ca.Function('p_next',[self.param['p'],pf,psi,self.param['k'],self.param['J_p']],[p_next],['p','pf','psi','k','Jp'],['p_next'])

        # if pf is not passed, wrap a python function around that defaults pf to 0
        if 'pf' not in self.param:
            def p_next_func_py(p,pf_loc,psi_loc,k,j_p):
                if pf_loc is None:
                    pf_loc = 0
                return p_next_func(p,pf_loc,psi_loc,k,j_p)
            def psi_next_func_py(p,pf_loc,psi_loc,k,j_p):
                if pf_loc is None:
                    pf_loc = 0
                return psi_next_func(p,pf_loc,psi_loc,k,j_p)
            def psi_init_func_py(p,pf_loc,j_p):
                if pf_loc is None:
                    pf_loc = 0
                return psi_init_func(p,pf_loc,j_p)
        else:
            p_next_func_py = p_next_func
            psi_next_func_py = psi_next_func
            psi_init_func_py = psi_init_func

        # store in upperLevel
        self._alg = {'psi_next':psi_next_func_py,'psi_init':psi_init_func_py,'p_next':p_next_func_py}

    @property
    def alg(self):
        return self._alg

    @property
    def cost(self):
        return self._cost

    @property
    def j_cost(self):
        return self._J_cost

    @property
    def idx(self):
        return self._idx

    @property
    def param(self):
        return self._sym.var
    
    @property
    def dim(self):
        return self._sym.dim

    @property
    def init(self):
        return {key: val for key, val in self._sym.init.items() if val is not None}

    def _set_init(self, data):
        self._sym.set_init(data)
    
    # # overwrite the __dir__ method
    # def __dir__(self):
    #     return [attr for attr in super().__dir__() if not attr.startswith('_UpperLevel__')]