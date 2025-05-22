import casadi as ca
from src.qp import QP
from typing import Callable, Union, Optional
import numpy as np
from src.symbolic_var import SymbolicVar
from typeguard import typechecked

class UpperLevel:
    """
    UpperLevel class for managing upper-level optimization in a hierarchical MPC framework.
    This class encapsulates the symbolic variables, parameter indexing, cost function setup, and algorithmic hooks
    required for upper-level optimization, typically in a bi-level or hierarchical MPC structure.
    
    Methods:
        __init__(p, horizon, mpc, pf=None, idx_p=None, idx_pf=None):
        set_cost(cost, track_cost=None, cst_viol=None):
            Performs checks for variable consistency, scalar cost, and allowed dependencies, and constructs helper functions for cost and Jacobian evaluation.
        set_alg(parameter_update, parameter_init=None, sys_id=None):
            Sets the algorithmic functions for parameter initialization, parameter update, and system identification.
    
    Properties:
        parameter_init: Returns the parameter initialization function.
        parameter_update: Returns the parameter update function.
        sys_id: Returns the system identification function.
        alg: Returns the algorithm function.
        cost: Returns the cost evaluation function.
        j_cost: Returns the cost Jacobian evaluation function
        idx: Returns the dictionary of indexing functions.
        param: Returns the dictionary of symbolic variables.
        dim: Returns the dictionary of variable dimensions.
        init: Returns the dictionary of initial values for symbolic variables.
    
    Usage:
        This class is intended to be used as the upper-level controller in a hierarchical MPC setup, where it manages
        symbolic representations, parameter passing, cost evaluation, and algorithmic updates for the upper-level optimization.
    """
    
    @typechecked
    def __init__(
            self,
            p: ca.SX,
            horizon: int,
            mpc: QP,
            pf: Optional[ca.SX] = None,
            idx_p: Optional[Callable[[int], Union[range, np.ndarray]]] = None,
            idx_pf: Optional[Callable[[int], Union[range, np.ndarray]]] = None
        ) -> None:
        """
        Initializes the UpperLevel class with symbolic variables, dimensions, and indexing functions for parameters and optional parameters.
        Args:
            p (ca.SX): Symbolic parameter vector for the QP.
            horizon (int): Prediction horizon length.
            mpc (QP): Instance of the lower-level QP controller, providing dimensions and parameters.
            pf (Optional[ca.SX], optional): Optional symbolic parameter vector for additional QP parameters. Defaults to None.
            idx_p (Optional[Callable[[int], Union[range, np.ndarray]]], optional): Function to index into `p` at each time step. If None, assumes `p` is time-invariant. Defaults to None.
            idx_pf (Optional[Callable[[int], Union[range, np.ndarray]]], optional): Function to index into `pf` at each time step. If None, assumes `pf` is time-invariant. Defaults to None.
        Raises:
            AssertionError: If the indexing functions `idx_p` or `idx_pf` do not return the correct dimensions for the QP parameters.
        Attributes: TODO
        """

        # create symbolic variable
        self._sym = SymbolicVar()

        # add to dimensions
        self._sym.add_dim('T',horizon)

        # add p to set of symbolic variables
        self._sym.add_var('p',p)

        # create symbolic jacobian of cost function wrt p
        self._sym.add_var('J_p',ca.SX.sym('J_p',self.dim['p'],1))

        # create symbolic variable representing the iteration number
        self._sym.add_var('k',ca.SX.sym('k',1,1))

        # save all symbolic variables
        self._sym.add_var('x_cl',ca.SX.sym('x_cl',mpc.dim['x'],horizon+1))
        self._sym.add_var('u_cl',ca.SX.sym('u_cl',mpc.dim['u'],horizon))
        self._sym.add_var('y_cl',ca.SX.sym('y_cl',mpc.dim['y'],horizon))
        if 'eps' in mpc.dim:
            self._sym.add_var('e_cl',ca.SX.sym('e_cl',mpc.dim['eps'],horizon))

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
            self._sym.add_var('pf',pf)

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
        if 'y_lin' in mpc.idx['out']:

            # add index to indices of UpperLevel
            self._idx = self._idx | {'y':mpc.idx['out']['y_lin']}

            # set flag to true
            y_idx = True

        # create local index function
        local_idx = self._idx

        # add state
        local_idx['x'] = lambda t: range(0,mpc.dim['x'])

        # create function that sets up the necessary inputs to the QP
        def qp_var_setup(var_in:dict,t:int) -> ca.DM:
            # var_in is a dictionary set up by scenario when calling the function _simulate.
            # It contains all the variables necessary to set up the QP. The QP always requires
            # the current state x, everything else is optional. Optional variables are: p, pf,
            # y_lin, theta. Variables that are not required are not present in var_in.
            # Note that the variables in var_in are ordered as follows: [x,y,p,pf].

            # preallocate output
            out = []

            # loop through variables in var_in
            for key,val in var_in.items():
                
                # convert input to DM in case it is not DM already, and read the required entries
                # according to the indexing function
                if key in local_idx:
                    out.append(ca.DM(val)[local_idx[key](t)])

            return ca.vcat(out)

        # create function that sets up the necessary inputs to the Jacobian
        def jac_var_setup(
                j_x_p:Union[np.ndarray,ca.DM],
                j_y_p:ca.DM,
                t:int,
                multiplier:int=1
            ) -> ca.DM:
            # j_x_p represents the jacobian of x with respect to p at time t, j_y_p represents the
            # jacobian of the optimization variables y with respect to p at time t. "multiplier"
            # denotes the number of times the jacobian should be "copied" in case multiple models
            # are being backpropagated.
            
            # get entries of p
            j_p = ca.repmat(ca.DM.eye(self.dim['p']),1,multiplier)[self._idx['p'](t),:]

            # get entries of y (if y_idx is not present it means that the sensitivities of y are
            # not being backpropagated).
            j_y = j_y_p[self.idx['y'](t),:] if y_idx else ca.DM(0,self.dim['p']*multiplier)

            # stack results vertically
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
        ) -> None:
        """
        Set the cost function and its associated tracking and constraint violation costs for the upper-level optimization problem.
        This method performs several checks and setups:
        - Ensures that the tracking cost and constraint violation cost are compatible with the main cost function.
        - Validates that all symbolic variables in the tracking and constraint violation costs are contained within the main cost.
        - Ensures the cost is a scalar symbolic expression.
        - Checks that the cost only depends on the allowed variables (`p_cl`, `x_cl`, `u_cl`, `y_cl`).
        - Identifies which parameters enter the cost and computes their Jacobians.
        - Constructs helper functions to extract relevant indices and Jacobians for the cost.
        - Creates callable CasADi functions for evaluating the cost and its Jacobian, storing them as attributes for later use.
        Args:
            cost (ca.SX): The main cost function as a scalar CasADi symbolic expression.
            track_cost (Optional[ca.SX], optional): The tracking cost function. If not provided, defaults to `cost`.
            cst_viol (Optional[ca.SX], optional): The constraint violation cost function. If not provided, defaults to a zero scalar.
        Raises:
            AssertionError: If the tracking cost or constraint violation cost contains variables not present in the main cost.
            AssertionError: If the cost is not a scalar expression.
            AssertionError: If the cost contains variables other than `p_cl`, `x_cl`, `u_cl`, or `y_cl`.
            AssertionError: If the helper function for extracting cost indices does not match the expected output.
        Side Effects: TODO (what does this function set?)
        """

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

        # store in upper level
        self._cost = cost_func

        # create full jacobian functions in two steps
        j_cost = j_cost_p + j_cost_x@j_x_p + j_cost_u@j_u_p + j_cost_y@j_y_p
        j_cost_func_temp = ca.Function('J_cost',[cost_in,j_x_p,j_u_p,j_y_p],[j_cost.T])

        # save indices
        self._get_cost_idx = get_cost_idx
        self._get_cost_jacobian = get_cost_jacobian

        # save cost temporary jacobian function
        self._j_cost_func_temp = j_cost_func_temp

        def j_cost_func(s):

            # get true input cost
            cost_in_loc = get_cost_idx(s.x,s.u,s.y,s.p)

            # get true Jacobian
            j_x,j_u,j_y = get_cost_jacobian(s.j_x,s.j_u,s.j_y)

            return j_cost_func_temp(cost_in_loc,j_x,j_u,j_y)
        
        # store in upper level
        self._J_cost = j_cost_func

    def set_alg(
            self,parameter_update:callable,
            parameter_init:Optional[callable]=None,
            sys_id_update:Optional[callable]=None,
            sys_id_init:Optional[callable]=None
        ) -> None:
        """
        Sets the algorithmic functions for parameter initialization, parameter update, and system identification.
        Args:
            parameter_update (callable): Function to update parameters during simulation. Must accept a simulation object as input.
            parameter_init (callable, optional): Function to initialize parameters before simulation starts. Must accept a simulation object as input. Defaults to a no-op function if not provided.
            sys_id (callable, optional): Function for system identification during simulation. Must accept a simulation object as input. Defaults to a no-op function if not provided.
        """

        # TODO: write tests here!!! For example idx_pf in sys_id

        self._parameter_init = parameter_init if parameter_init is not None else lambda sim: None
        self._parameter_update = parameter_update
        self._sys_id_update = sys_id_update if sys_id_update is not None else None
        self._sys_id_init = sys_id_init if sys_id_init is not None else None
    
    @property
    def parameter_init(self):
        return self._parameter_init
    
    @property
    def parameter_update(self):
        return self._parameter_update
    
    @property
    def sys_id_update(self):
        return self._sys_id_update
    
    @property
    def sys_id_init(self):
        return self._sys_id_init

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