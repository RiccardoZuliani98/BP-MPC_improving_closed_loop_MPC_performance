from casadi import hcat,vcat,DM
from numpy import hstack,vstack

class simVar:

    _VAR_LIST = ['x','u','e','y','mu','lam','p_qp']
    _JAC_LIST = ['j_x','j_u','j_eps','j_y']

    def __init__(self,dim,n_models=1):
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

    def stack(self):
        
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

    def add_sim_jac(self,j_x,j_u,j_y):
        self.j_x.append(j_x)
        self.j_u.append(j_u)
        self.j_y.append(j_y)

    def add_opt_var(self,lam,mu,y,p):
        self.lam.append(lam)
        self.mu.append(mu)
        self.y.append(y)
        self.p_qp.append(p)
    
    def save_memory(self):
        self.j_x = None
        self.j_u = None
        self.j_eps = None
        self.y = None
        self.mu = None
        self.lam = None
        self.p_qp = None
        self.j_y = None