from casadi import *

class MPC:

    def __init__(self):

        # horizon of the MPC (positive integer)
        self.__N = None

        """
        Cost dictionary. Can contain the following entries (provided by the user):
            
            - Qx: stage cost for the state (n_x*(N-1),n_x*(N-1))
            - Qn: terminal cost for the state (n_x,n_x)
            - Ru: stage cost for the input (n_u*N,n_u*N)
            - s_lin: linear slack cost (nonnegative scalar)
            - s_quad: quadratic slack cost (nonnegative scalar, defaults to 1 as we need 
                      a positive definite cost)
        
        note that the cost is defined as (x-x_ref)'blkdiag(Qx,Qn)(x-x_ref) + (u-u_ref)'Ru(u-u_ref),
        where x and u have dimension (n_x*N,1) and (n_u*N,1), respectively.
        """
        self.__cost = {}

        """
        Constraints dictionary. Can contain the following entries (provided by the user):

            - 'Hx': state constraint matrix (=,n_x*N)
            - 'hx': state constraint vector (=,1)
            - 'Hx_e': (optional, defaults to identity) matrix that softens constraints (=,n_eps)
            - 'Hu': input constraint matrix (-,n_u*N)
            - 'hu': input constraint vector (-,1)

        recall that the constraints are Hx*x + Hx_e*e <= hx, Hu*u <= hu, where e are the slack variables.
        Note that x and u are here of dimensions (n_x*N,1) and (n_u*N,1) respectively (i.e. they contain all time-steps).
        """
        self.__cst = {}

        """
        Options dictionary, can contain the following entries (provided by the user):

            - 'linearization': 'trajectory', 'state' or 'none' (default is 'trajectory')
            - 'slack': True or False (default is False)
            - 'qp_mode': 'stacked' or 'separate' (default is 'stacked')
            - 'solver': 'qpoases','osqp','cplex','gurobi','daqp','qrqp' (default is 'qpoases')
            - 'warmstart': 'x_lam_mu' (warmstart both primal and dual variables) or 'x' (warmstart only 
                           primal variables) (default is 'x_lam_mu')
            - 'jac_tol': tolerance below which multipliers are considered zero (default is 8)
            - 'jac_gamma': stepsize in optimality condition used to apply the IFT (default is 0.001)
        """
        self.__options = {'linearization':'trajectory','slack':False,
                          'qp_mode':'stacked','solver':'qpoases',
                          'warmstart':'x_lam_mu',
                          'jac_tol':8,'jac_gamma':0.001,
                          'compile_qp_sparse':False,
                          'compile_jac':False}
        
        """
        Dictionary containing used defined time-invariant linear model for the dynamics.
        The allowed entries are A,B,c, where the prediction model is x[t+1] = Ax[t] + Bu[t] + c.
        Note that c is optional and defaults to zero.
        If this entry is None, it means that the model is obtained by linearizing the nominal
        dynamics.
        """
        self.__model = None

        # lists of keys that the user is allowed to set in cost, cst, model, and options
        allowed_keys_cost = ['Qx','Qn','Ru','x_ref','u_ref','s_lin','s_quad']
        allowed_keys_cst = ['Hx','hx','Hu','hu','Hx_e']
        allowed_keys_model = ['A','B','c']
        allowed_keys_options = ['linearization','slack','qp_mode','solver','warmstart','jac_tol','jac_gamma']

        # we store them in a private dictionary and use them in the set functions
        self.__allowed_keys = {'cost':allowed_keys_cost,'cst':allowed_keys_cst,'model':allowed_keys_model,'options':allowed_keys_options}
        pass

    @property
    def N(self):
        return self.__N
    
    def __set_N(self, value):
        if (not isinstance(value,int)) or value <= 0:
            raise Exception('N must be an positive integer.')
        self.__N = value

    @property
    def cost(self):
        return self.__cost
    
    def __set_cost(self, value):

        # check if value is a dictionary
        if not isinstance(value, dict):
            raise Exception('Cost must be a dictionary.')
        
        # remove keys that are not allowed
        value = {k:v for k,v in value.items() if k in self.__allowed_keys['cost']}

        # update cost dictionary
        self.__cost = value

    @property
    def cst(self):
        return self.__cst
    
    def __set_cst(self, value):

        # check if value is a dictionary
        if (not isinstance(value, dict)):
            raise Exception('Constraints must be a dictionary.')
        
        # remove keys that are not allowed
        value = {k:v for k,v in value.items() if k in self.__allowed_keys['cst']}

        # update constraints dictionary
        self.__cst = value

    @property
    def options(self):
        return self.__options
    
    def __updateOptions(self, value):

        # check if value is a dictionary
        if not isinstance(self.__cost, dict):
            raise Exception('Options must be a dictionary.')
        
        # remove keys that are not allowed
        value = {k:v for k,v in value.items() if k in self.__allowed_keys['options']}

        # update options dictionary
        self.__options = self.__options | value

    @property
    def model(self):
        return self.__model

    def __set_model(self, value):

        # check if value is a dictionary
        if not isinstance(value, dict):
            raise Exception('Model must be a dictionary.')
        
        # remove keys that are not allowed
        value = {k:v for k,v in value.items() if k in self.__allowed_keys['model']}

        # update model dictionary
        self.__model = value

    # overwrite the __dir__ method
    def __dir__(self):
        return [attr for attr in super().__dir__() if not attr.startswith('_MPC__')]