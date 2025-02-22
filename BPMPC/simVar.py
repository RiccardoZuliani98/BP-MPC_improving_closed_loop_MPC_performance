from casadi import *

class simVar:
    def __init__(self,dim):

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
        self.nx = dim['x']              # state dimension
        self.nu = dim['u']              # input dimension
        self.np = dim['p']              # parameter dimension
        self.neps = dim['eps']          # slack dimension
        self.ny = dim['y']              # primal optimization variables dimension (no slacks)

        # then initialize all variables
        self.__x = DM(dim['x']*(dim['T']+1),1)                # closed-loop state (n_x*(T+1),1)
        self.__u = DM(dim['u']*dim['T'],1)                    # closed-loop input (n_u*T,1)
        self.__e = DM(dim['eps']*dim['T'],1)                  # closed-loop slack (n_eps*T,1)
        self.__Jx = DM(dim['x']*(dim['T']+1),dim['p'])        # Jacobian of state (n_x*(T+1),n_p)
        self.__Ju = DM(dim['u']*dim['T'],dim['p'])            # Jacobian of input (n_u*T,n_p)
        self.__Jeps = DM(dim['eps']*dim['T'],dim['p'])        # Jacobian of slack (n_eps*T,n_p)
        self.__y = DM(dim['y'],dim['T'])                      # primal optimization variables (n_y,T)
        self.__mu = DM(dim['eq'],dim['T'])                    # multipliers of equality constraints (n_eq,T)
        self.__lam = DM(dim['in'],dim['T'])                   # multipliers of inequality constraints (n_in,T)
        self.__p_qp = DM(dim['p_qp']+dim['pf_qp'],dim['T'])   # parameters setting up QP at each time-step (n_p_qp+pf_qp,T)
        self.__Jy = DM((dim['y'])*dim['T'],dim['p'])          # Jacobian of optimization variables (n_y*T,n_p)
        if 'p' in dim:
            self.__p = DM(dim['p'],1)                           # closed-loop design parameter (n_p,1)
            self.__Jp = DM(dim['p'],1)                          # Jacobian of closed-loop cost wrt design parameter (n_p,1)
        if 'pf' in dim:
            self.__pf = DM(dim['pf'],1)                         # closed-loop fixed parameter (e.g. reference to track)
        self.__cost = None                                    # closed-loop cost
        self.__cst = None                                     # closed-loop constraint violation

    @property
    def x_mat(self):
        return reshape(self.x,self.nx,-1)
    
    @property
    def u_mat(self):
        return reshape(self.u,self.nu,-1)
    
    @property
    def e_mat(self):
        return reshape(self.e,self.neps,-1)
    
    @property
    def y_mat(self):
        return reshape(self.y,self.ny,-1)

    # set and get method for optimization variables
    def setOptVar(self,t,lam,mu,y,p):
        self.__y[:,t] = y
        self.__lam[:,t] = lam
        self.__mu[:,t] = mu
        self.__p_qp[:,t] = p
    
    def getOptVar(self,t):
        return self.lam[:,t],self.mu[:,t],self.y[:,t],self.p_qp[:,t]
    
    # you can call this function to save memory for long closed-loop simulations
    def saveMemory(self):
        self.__Jx = None
        self.__Ju = None
        self.__Jeps = None
        self.__y = None
        self.__mu = None
        self.__lam = None
        self.__p_qp = None
        self.__Jy = None

    @property
    def p(self):
        return self.__p
    
    @p.setter
    def p(self,p):
        self.__p = p

    @property
    def pf(self):
        return self.__pf
    
    @pf.setter
    def pf(self,pf):
        self.__pf = pf

    @property
    def Jp(self):
        return self.__Jp
    
    @Jp.setter
    def Jp(self,Jp):
        self.__Jp = Jp

    @property
    def cost(self):
        return self.__cost
    
    @cost.setter
    def cost(self,cost):
        self.__cost = cost

    @property
    def cst(self):
        return self.__cst
    
    @cst.setter
    def cst(self,cst):
        self.__cst = cst

    @property
    def x(self):
        return self.__x
    # set and get method for closed-loop state
    def setState(self,t,x):
        self.__x[self.nx*t:self.nx*(t+1)] = x
    def getState(self,t):
        return self.x[self.nx*t:self.nx*(t+1)]
    
    @property
    def u(self):
        return self.__u
    # set and get method for closed-loop input
    def setInput(self,t,u):
        self.__u[self.nu*t:self.nu*(t+1)] = u
    def getInput(self,t):
        return self.u[self.nu*t:self.nu*(t+1)]
    
    @property
    def e(self):
        return self.__e
    # set and get method for closed-loop slack
    def setSlack(self,t,e):
        self.__e[self.neps*t:self.neps*(t+1)] = e
    def getSlack(self,t):
        return self.e[self.neps*t:self.neps*(t+1)]
    
    @property
    def Jx(self):
        return self.__Jx
    # set and get method for Jacobian of state
    def setJx(self,t,J):
        self.__Jx[self.nx*t:self.nx*(t+1),:] = J
    def getJx(self,t):
        return self.Jx[self.nx*t:self.nx*(t+1),:]
    
    @property
    def Ju(self):
        return self.__Ju
    # set and get method for Jacobian of input
    def setJu(self,t,J):
        self.__Ju[self.nu*t:self.nu*(t+1),:] = J
    def getJu(self,t):
        return self.Ju[self.nu*t:self.nu*(t+1),:]
    
    @property
    def Jeps(self):
        return self.__Jeps
    # set and get method for Jacobian of slack
    def setJeps(self,t,J):
        self.__Jeps[self.neps*t:self.neps*(t+1),:] = J
    def getJeps(self,t):
        return self.Jeps[self.neps*t:self.neps*(t+1),:]
    
    @property
    def Jy(self):
        return self.__Jy
    # set and get method for Jacobian of optimization variables
    def setJy(self,t,J):
        self.__Jy[self.ny*t:self.ny*(t+1),:] = J
    def getJy(self,t):
        return self.Jy[self.ny*t:self.ny*(t+1),:]
    
    @property
    def y(self):
        return self.__y
    
    @property
    def lam(self):
        return self.__lam
    
    @property
    def mu(self):
        return self.__mu
    
    @property
    def p_qp(self):
        return self.__p_qp