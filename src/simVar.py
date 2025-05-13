from casadi import hcat,vcat,DM
from numpy import hstack,vstack

class simVar:

    _VAR_LIST = ['x','u','e','y','mu','lam','p_qp']
    _JAC_LIST = ['j_x','j_u','j_eps','j_y']

    def __init__(self,dim,n_models=1):

        """
        Class that contains results of closed-loop simulations. Contains the following variables:

            - x: closed-loop state (n_x*(T+1),1)
            - u: closed-loop input (n_u*T,1)
            - e: closed-loop slack (n_eps*T,1)
            - Jx: Jacobian of state (n_x*(T+1),n_p)
            - Ju: Jacobian of input (n_u*T,n_p)
            - Jeps: Jacobian of slack (n_eps*T,n_p)
            - y: primal optimization variables (n_y,T)
            - mu: multipliers of equality constraints (n_eq,T)
            - lam: multipliers of inequality constraints (n_in,T)
            - p_qp: parameters setting up QP at each time-step, including pf (n_p_qp+pf_qp,T)
            - Jy: Jacobian of optimization variables (n_y*T,n_p)
            - p: closed-loop design parameter (n_p,1)
            - Jp: Jacobian of closed-loop cost wrt design parameter (n_p,1)
            - pf: closed-loop fixed parameter (e.g. reference to track) (n_pf,1)
            - cost: closed-loop cost (scalar)
            - cst: closed-loop constraint violation (scalar)

        Additionally, the class offers the following methods:

            - setOptVar: set optimization variables at time-step t, with inputs:

                    - t: time-step
                    - lam: multipliers of inequality constraints at time-step t (n_in,1)
                    - mu: multipliers of equality constraints at time-step t (n_eq,1)
                    - y: primal optimization variables at time-step t (n_y,1)
                    - p: parameters setting up QP at time-step t, including pf (n_p_qp+pf_qp,1)

            - getOptVar: input the time-step t and get lam,mu,y,p at the corresponding time-step

            - saveMemory: call this function to save memory for long closed-loop simulations

            - x_mat, u_mat, e_mat, y_mat: return the reshaped version of x, u, e, y, where
                each column corresponds to a time-step.
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
        self.x = []        # closed-loop state (n_x*(T+1),1)
        self.u = []        # closed-loop input (n_u*T,1)
        self.e = []        # closed-loop slack (n_eps*T,1)
        self.j_x = []       # Jacobian of state (n_x*(T+1),n_p)
        self.j_u = []       # Jacobian of input (n_u*T,n_p)
        self.j_eps = []     # Jacobian of slack (n_eps*T,n_p)
        self.y = []        # primal optimization variables (n_y,T)
        self.mu = []       # multipliers of equality constraints (n_eq,T)
        self.lam = []      # multipliers of inequality constraints (n_in,T)
        self.p_qp = []     # parameters setting up QP at each time-step (n_p_qp+pf_qp,T)
        self.j_y = []       # Jacobian of optimization variables (n_y*T,n_p)
        if 'p' in dim:
            self.p = []    # closed-loop design parameter (n_p,1)
            self.j_p = []   # Jacobian of closed-loop cost wrt design parameter (n_p,1)
        if 'pf' in dim:
            self.pf = []   # closed-loop fixed parameter (e.g. reference to track)
        self.cost = None   # closed-loop cost
        self.cst = None    # closed-loop constraint violation

    def stack(self):
        
        for elem in self._VAR_LIST:

            var = getattr(self,elem)
            
            if isinstance(var,list) and len(var) > 0:
                
                concat_var = hcat(var) if isinstance(var[0],DM) else DM(hstack(var).reshape((var[0].shape[0],-1,1)).squeeze())
                
                setattr(self,elem,concat_var)

        for elem in self._JAC_LIST:

            var = getattr(self,elem)
            
            if isinstance(var,list) and len(var) > 0:
                
                concat_var = vcat(var) if isinstance(var[0],DM) else DM(vstack(var).reshape((-1,var[0].shape[1]*var[0].shape[2])))
                
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

    # @property
    # def x_mat(self):
    #     return reshape(self.x,self.nx,-1)
    
    # @property
    # def u_mat(self):
    #     return reshape(self.u,self.nu,-1)
    
    # @property
    # def e_mat(self):
    #     return reshape(self.e,self.neps,-1)
    
    # @property
    # def y_mat(self):
    #     return reshape(self.y,self.ny,-1)

    # set and get method for optimization variables
    def setOptVar(self,t,lam,mu,y,p):
        self._y[:,t] = y
        self._lam[:,t] = lam
        self._mu[:,t] = mu
        self._p_qp[:,t] = p
    
    def getOptVar(self,t):
        return self.lam[:,t],self.mu[:,t],self.y[:,t],self.p_qp[:,t]
    
    # you can call this function to save memory for long closed-loop simulations
    def saveMemory(self):
        self._Jx = None
        self._Ju = None
        self._Jeps = None
        self._y = None
        self._mu = None
        self._lam = None
        self._p_qp = None
        self._Jy = None