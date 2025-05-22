from casadi import hcat,vcat,DM
from numpy import hstack,vstack,ndarray
from typing import Union

class SimVar:
    """
    simVar is a class designed to store and manage the results of closed-loop simulations in model predictive 
    control (MPC) frameworks. It provides structured storage for system states, inputs, slack variables, 
    optimization variables, Lagrange multipliers, and their corresponding Jacobians across simulation time 
    steps and potentially multiple models.

    Attributes:
        dim (dict): Dictionary containing the dimensions of various variables, such as state ('x'), 
            input ('u'), parameter ('p'), fixed parameter ('pf'), slack ('eps'), and primal optimization 
            variables ('y'). Also includes the number of models ('n_models').
        x: Closed-loop state trajectories, typically of shape (n_x*(T+1), 1).
        u: Closed-loop input trajectories, typically of shape (n_u*T, 1).
        e: Closed-loop slack variables, typically of shape (n_eps*T, 1).
        j_x: Jacobian of state trajectories with respect to parameters, typically of shape (n_x*(T+1), n_p).
        j_u: Jacobian of input trajectories with respect to parameters, typically of shape (n_u*T, n_p).
        j_eps: Jacobian of slack variables with respect to parameters, typically of shape (n_eps*T, n_p).
        y: Primal optimization variables, typically of shape (n_y, T).
        mu: Multipliers of equality constraints, typically of shape (n_eq, T).
        lam: Multipliers of inequality constraints, typically of shape (n_in, T).
        p_qp: Parameters for QP setup at each time-step, typically of shape (n_p_qp+pf_qp, T).
        j_y: Jacobian of optimization variables with respect to parameters, typically of shape (n_y*T, n_p).
        p: Closed-loop design parameter, if present in dim.
        j_p: Jacobian of closed-loop cost with respect to design parameter, if present in dim.
        pf: Closed-loop fixed parameter (e.g., reference trajectory), if present in dim.
        psi: Auxiliary parameter for closed-loop optimization.
        cost: Closed-loop cost, typically a scalar.    
        cst: Closed-loop constraint violation, typically a scalar.
    
    Methods:
        __init__(dim: dict, n_models: int = 1):
            Initializes the simVar object with specified variable dimensions and number of models.
        stack():
            Aggregates and concatenates the lists of simulation variables and Jacobians into single arrays
            or matrices for further processing. Uses horizontal or vertical concatenation depending on the 
            variable type.
        add_sim_jac(j_x, j_u, j_y):
            Appends simulation Jacobian matrices to their respective lists.
        add_opt_var(lam, mu, y, p):
            Appends optimization variables to their respective lists.
        save_memory():
            Frees up memory by setting large internal attributes (such as Jacobians and optimization 
            variables) to None when they are no longer needed.
    
    Usage:
        This class is typically used in closed-loop MPC simulations to collect, organize, and process simulation
        results and sensitivities for analysis, visualization, or further optimization.
    """

    _VAR_LIST = ['x','u','e','y','mu','lam','p_qp']
    _JAC_LIST = ['j_x','j_u','j_eps','j_y']

    def __init__(self,dim:dict,n_models:int=1) -> None:
        """
        Initializes the simulation variable class to store results of closed-loop simulations.
        Args:
            dim (dict): A dictionary containing the dimensions of various variables:
                - 'x': State dimension.
                - 'u': Input dimension.
                - 'p' (optional): Parameter dimension.
                - 'pf' (optional): Fixed parameter dimension.
                - 'eps': Slack dimension.
                - 'y': Primal optimization variables dimension (excluding slacks).
            n_models (int, optional): Number of models. Defaults to 1.
        Attributes:
            dim (dict): Stores the dimensions of the variables.
            x (list): Closed-loop state (n_x*(T+1), 1).
            u (list): Closed-loop input (n_u*T, 1).
            e (list): Closed-loop slack (n_eps*T, 1).
            j_x (list): Jacobian of state (n_x*(T+1), n_p).
            j_u (list): Jacobian of input (n_u*T, n_p).
            j_eps (list): Jacobian of slack (n_eps*T, n_p).
            y (list): Primal optimization variables (n_y, T).
            mu (list): Multipliers of equality constraints (n_eq, T).
            lam (list): Multipliers of inequality constraints (n_in, T).
            p_qp (list): Parameters setting up QP at each time-step (n_p_qp+pf_qp, T).
            j_y (list): Jacobian of optimization variables (n_y*T, n_p).
            p (list, optional): Closed-loop design parameter (n_p, 1), if 'p' is in dim.
            j_p (list, optional): Jacobian of closed-loop cost with respect to design parameter (n_p, 1), if 'p' is in dim.
            pf (list, optional): Closed-loop fixed parameter (e.g., reference to track), if 'pf' is in dim.
            cost (None): Closed-loop cost (scalar).
            cst (None): Closed-loop constraint violation (scalar).
        """
        
        # first set all dimensions
        self.dim = {}
        self.dim['x'] = dim['x']              # state dimension
        self.dim['u'] = dim['u']              # input dimension
        if 'p' in dim:
            self.dim['p'] = dim['p']          # parameter dimension
        if 'pf' in dim:
            self.dim['pf'] = dim['pf']        # fixed parameter dimension
        self.dim['eps'] = dim['eps']          # slack dimension
        self.dim['y'] = dim['y']              # primal optimization variables dimension (no slacks)
        self.dim['n_models'] = n_models        # number of models

        # then initialize all variables
        self.x = []         # closed-loop state (n_x*(T+1),1)
        self.u = []         # closed-loop input (n_u*T,1)
        self.e = []         # closed-loop slack (n_eps*T,1)
        self.j_x = []       # Jacobian of state (n_x*(T+1),n_p)
        self.j_u = []       # Jacobian of input (n_u*T,n_p)
        self.j_eps = []     # Jacobian of slack (n_eps*T,n_p)
        self.y = []         # primal optimization variables (n_y,T)
        self.mu = []        # multipliers of equality constraints (n_eq,T)
        self.lam = []       # multipliers of inequality constraints (n_in,T)
        self.p_qp = []      # parameters setting up QP at each time-step (n_p_qp+pf_qp,T)
        self.j_y = []       # Jacobian of optimization variables (n_y*T,n_p)
        self.p = None       # closed-loop design parameter (n_p,1)
        self.j_p = None     # Jacobian of closed-loop cost wrt design parameter (n_p,1)
        self.pf = None      # closed-loop fixed parameter (e.g. reference to track)
        self.psi = None     # auxiliary parameter for closed-loop optimization
        self.cost = None    # closed-loop cost
        self.cst = None     # closed-loop constraint violation

    def stack(self) -> None:
        """
        Stacks elements of variables listed in _VAR_LIST and _JAC_LIST.
        For each variable name in _VAR_LIST:
            - If the attribute is a non-empty list:
                - If the first element is a CasADi DM object, horizontally concatenates the list using hcat.
                - Otherwise, horizontally stacks the list using numpy's hstack, reshapes, and converts to DM.
            - The concatenated result replaces the original attribute.
        For each variable name in _JAC_LIST:
            - If the attribute is a non-empty list:
                - If the first element is a CasADi DM object, vertically concatenates the list using vcat.
                - Otherwise, vertically stacks the list using numpy's vstack, reshapes, and converts to DM.
            - The concatenated result replaces the original attribute.
        This method is typically used to aggregate simulation variables and Jacobians for further processing.
        """
        
        for elem in self._VAR_LIST:

            var = getattr(self,elem)
            
            if isinstance(var,list) and len(var) > 0:
                
                concat_var = hcat(var) if isinstance(var[0],DM) else DM(hstack(var).reshape((var[0].shape[0],-1,1)).squeeze())
                
                setattr(self,elem,concat_var)

        for elem in self._JAC_LIST:

            var = getattr(self,elem)
            
            if isinstance(var,list) and len(var) > 0:
                
                concat_var = vcat(var) if isinstance(var[0],DM) else DM(vstack(var).reshape((-1,var[0].shape[1]*var[0].shape[2]),order='F'))
                
                setattr(self,elem,concat_var)

    def add_sim_jac(
            self,j_x:Union[DM,ndarray],
            j_u:Union[DM,ndarray],
            j_y:Union[DM,ndarray]
        ) -> None:
        """
        Appends simulation Jacobian matrices to the respective lists.

        Parameters:
            j_x: The Jacobian matrix with respect to the state variables.
            j_u: The Jacobian matrix with respect to the input variables.
            j_y: The Jacobian matrix with respect to the output variables.

        Returns:
            None
        """
        self.j_x.append(j_x)
        self.j_u.append(j_u)
        self.j_y.append(j_y)

    def add_opt_var(self,
            lam:Union[DM,ndarray],
            mu:Union[DM,ndarray],
            y:Union[DM,ndarray],
            p:Union[DM,ndarray]
        ) -> None:
        """
        Adds optimization variables to the corresponding lists.

        Parameters:
            lam: The value to append to the 'lam' list, typically representing Lagrange multipliers.
            mu: The value to append to the 'mu' list, typically representing dual variables.
            y: The value to append to the 'y' list, typically representing primal variables or outputs.
            p: The value to append to the 'p_qp' list, typically representing parameters or problem data for the QP.

        """
        self.lam.append(lam)
        self.mu.append(mu)
        self.y.append(y)
        self.p_qp.append(p)
    
    def save_memory(self) -> None:
        """
        Frees up memory by setting various internal attributes to None.

        This method clears the following attributes:
            - j_x: State cost or related data
            - j_u: Input cost or related data
            - j_eps: Epsilon cost or related data
            - y: Output or measurement data
            - mu: Dual variables or multipliers
            - lam: Lagrange multipliers or related data
            - p_qp: Quadratic programming parameters or results
            - j_y: Output cost or related data

        Use this method to release memory when these attributes are no longer needed.
        """
        self.j_x = None
        self.j_u = None
        self.j_eps = None
        self.y = None
        self.mu = None
        self.lam = None
        self.p_qp = None
        self.j_y = None