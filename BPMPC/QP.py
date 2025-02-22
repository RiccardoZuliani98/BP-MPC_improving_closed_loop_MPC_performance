from casadi import *

class QP:

    def __init__(self,MSX):

        # check type of symbolic variables
        if MSX == 'SX':
            self.__MSX = SX
        elif MSX == 'MX':
            self.__MSX = MX
        else:
            raise Exception('MSX must be either SX or MX.')

        """
        Symbolic variables needed or manipulated in the QP, all in SX or MX.
        """
        self.__x = None         # initial state (n_x,1)
        self.__y = None         # primal optimization variables (n_y,1)
        self.__lam = None       # dual multipliers of the inequality constraints (n_in,1)
        self.__mu = None        # dual multipliers of the equality constraints (n_eq,1)
        self.__z = None         # dual multipliers (n_in + n_eq,1)
        self.__y_lin = None     # linearization trajectory (n_y_lin,1)
        self.__p_t = None       # design parameter of QP at time t (dimension not stored)
        self.__pf_t = None      # fixed parameter of QP at time t (e.g. reference) (n_pf_t,1)
        self.__p_qp = None      # symbolic parameters needed to setup QP at time t (n_p_qp,1)
                                # likely contains x,y_lin,p_t,pf_t

        """
        QP elements, setup through call to scenario.__makeQP. It is a dictionary containing the following keys:

            - 'Q': Hessian of QP (n_y,n_y)
            - 'Qinv': inverse of Hessian of QP (n_y,n_y)
            - 'q': gradient of cost of QP (n_y,1)
            - 'G': linear inequality constraint matrix (n_in,n_y)
            - 'g': linear inequality constraint vector (n_in,1)
            - 'H': Hessian of dual problem (n_z,n_z)
            - 'h': gradient of cost of dual problem (n_z,1)
            - 'A': stacked inequality constraint matrix for casadi's conic interface,
                   specifically, A = vertcat(G,F) (n_in+n_eq,n_y)
            - 'lba': lower bound of inequality constraints, lba = (-inf,f) (n_in+n_eq,1)
            - 'uba': upper bound of inequality constraints, uba = (g,f) (n_in+n_eq,1)
            - 'dense_qp: this is another dictionary containing the elements of the dense
                version of the QP, it is created by the function scenario.__makeDenseMPC,
                and it contains the keys
                    - 'G_x', 'G_u', 'g_c': matrices satisfying x = G_x*x0 + G_u*u + g_c
                    - 'Qx', 'Ru', 'x_ref', 'u_ref': cost function (x-x_ref)'Qx(x-x_ref) + (u-u_ref)'Ru(u-u_ref)
                    - 'Hx', 'Hu', 'hx', 'hu': polyhedral constraints Hx*x <= hx, Hu*u <= hu
    
        remember that the primal problem is a QP with the following structure:
    
            min 1/2 y'Qy + q'y
            s.t. Gy <= g
                 Fy = y
    
        and the dual problem is a QP with the following structure:
    
            min 1/2 z'Hz + h'z
            s.t. z=(lam,mu)
                 lam >= 0
        """
        self.__ingredients = {}


        """
        idx is a dictionary containing the indexing of the input and output optimization variables of the QP.
        It contains keys setup by different functions. Specifically, calling scenario.__makeSparseMPC creates
        a key 'out' which contains the index of the output QP variables. idx['out'] is itself a dictionary
        with keys

            - 'u': range of all inputs
            - 'x': range of all states
            - 'y': range of all state-input variables
            - 'eps': range of all slack variables (if present)
            - 'u0': range of first input
            - 'u1': range of second input
            - 'x_shift': states shifted by one time-step (last state repeated)
            - 'u_shift': inputs shifted by one time-step (last input repeated)
            - 'y_shift': concatenation of x_shift and u_shift (and slacks shifted if present)

        the second entry is 'in', which contains the index of the input QP variables (i.e. p_QP, not including
        pf_t). idx['in'] is itself a dictionary set up by __makeQP with keys

            - 'x0': initial state
            - 'y_lin': linearization trajectory
            - 'p_t': parameters that are optimized in the upper-level

        Note that not all parameters need to be present.
        """
        self.__idx = {}


        """
        this function takes in p_QP and returns the optimal solution of the QP problem. Based on the type
        of warmstarting (i.e. x_lam_mu or x), the function will take the following inputs:

            - x_lam_mu: p_QP (including pf_t), x0=None, lam=None, mu=None
            - x: p_QP (including pf_t), x0=None

        the output is lam, mu, y.
        """
        # sparse version
        self.__solve = None
        # dense version
        self.__denseSolve = None
        

        """
        This function takes in p_QP and returns the sparse ingredients, which are:
                
            - F,f,G,g,Q,q in the separate mode, where the QP is formulated as

                min 1/2 y'Qy + q'y
                s.t. Gy <= g
                     Fy = y

            - A,lba,uba,Q,q in the stacked mode, where the QP is formulated as

                min 1/2 y'Qy + q'y
                s.t. lba <= Ay <= uba
        """
        self.__qp_sparse = None


        """
        this function takes in p_QP (including pf_t) and returns the dense ingredients, which are:

            - 'G_x', 'G_u', 'g_c': matrices satisfying x = G_x*x0 + G_u*u + g_c
            - 'Qx', 'Ru', 'x_ref', 'u_ref': cost function (x-x_ref)'Qx(x-x_ref) + (u-u_ref)'Ru(u-u_ref)
            - 'Hx', 'Hu', 'hx', 'hu': polyhedral constraints Hx*x <= hx, Hu*u <= hu
              
        where the QP is formulated as

            min 1/2 u'(G_x'QxG_x + Ru)u + (G_x'Qx(G_x*x0 + g_c - x_ref) - Ru*u_ref)'u
            s.t. Hx*(G_x*x0 + G_u*u + g_c) <= hx
                 Hu*u <= hu
        """
        self.__qp_dense = None


        """
        This function takes in p_QP (including pf_t) and returns the dual ingredients H,h, where the dual
        QP is formulated as:

            min 1/2 z'Hz + h'z
            s.t. z >= 0
        """
        self.__dual_sparse = None


        """
        this function takes in the dual variables lam, mu, and the parameters p_QP necessary to setup the
        QP (including pf_t), and returns the following quantities in a list:

            - J_F_z: conservative jacobian of dual fixed point condition wrt z
            - J_F_p: conservative jacobian of dual fixed point condition wrt p_t
            - J_y_p: conservative jacobian of primal variable wrt p_t
            - J_y_z_mat: conservative jacobian of primal variable wrt z
        """
        self.__J = None


        """
        this function takes in lam, mu, p_QP (including pf_t), and optionally t (which defaults to 1), and
        returns the inner product between the conservative jacobian J_y_p of y wrt p_t and t.
        """
        self.__J_y_p = None


        """
        init contains the initial value of the QP variables, it can be set through __setInit.
        """
        self.__init = {'y_lin':None}
        pass

    @property
    def x(self):
        return self.__x
    
    def __set_x(self, value):
        if type(value) is not self.__MSX:
            raise Exception('x is of the wrong symbolic type.')
        self.__x = value

    @property
    def y(self):
        return self.__y
    
    def __set_y(self, value):
        if type(value) is not self.__MSX:
            raise Exception('y is of the wrong symbolic type.')
        self.__y = value

    @property
    def lam(self):
        return self.__lam
    
    def __set_lam(self, value):
        if type(value) is not self.__MSX:
            raise Exception('lam is of the wrong symbolic type.')
        self.__lam = value

    @property
    def mu(self):
        return self.__mu
    
    def __set_mu(self, value):
        if type(value) is not self.__MSX:
            raise Exception('mu is of the wrong symbolic type.')
        self.__mu = value

    @property
    def z(self):
        return self.__z
    
    def __set_z(self, value):
        if type(value) is not self.__MSX:
            raise Exception('z is of the wrong symbolic type.')
        self.__z = value

    @property
    def y_lin(self):
        return self.__y_lin
    
    def __set_y_lin(self, value):
        if type(value) is not self.__MSX:
            raise Exception('y_lin is of the wrong symbolic type.')
        self.__y_lin = value

    @property
    def ingredients(self):
        return self.__ingredients
    
    def __setIngredients(self, value):
        self.__ingredients = value

    @property
    def idx(self):
        return self.__idx
    
    def __updateIdx(self, value):
        self.__idx = self.idx | value

    @property
    def solve(self):
        return self.__solve

    def __setSolver(self, value):
        self.__solve = value

    @property
    def denseSolve(self):
        return self.__denseSolve

    def __setDenseSolver(self, value):
        self.__denseSolve = value

    @property
    def qp_sparse(self):
        return self.__qp_sparse
    
    def __setQpSparse(self, value):
        self.__qp_sparse = value

    @property
    def qp_dense(self):
        return self.__qp_dense
    
    def __setQpDense(self, value):
        self.__qp_dense = value

    @property
    def dual_sparse(self):
        return self.__dual_sparse
    
    def __setDualSparse(self, value):
        self.__dual_sparse = value

    @property
    def p_t(self):
        return self.__p_t
    
    def __set_p_t(self, value):
        self.__p_t = value

    @property
    def pf_t(self):
        return self.__pf_t
    
    def __set_pf_t(self, value):
        self.__pf_t = value

    @property
    def p_qp(self):
        return self.__p_qp
    
    def __set_p_qp(self, value):
        self.__p_qp = value

    @property
    def J(self):
        return self.__J

    def __set_J(self, value):
        self.__J = value

    @property
    def J_y_p(self):
        return self.__J_y_p

    def __set_J_y_p(self, value):
        self.__J_y_p = value

    @property
    def param(self):
        return {k: v for k, v in {
            'x': self.__x,
            'y': self.__y,
            'z': self.__z,
            'lam': self.__lam,
            'mu': self.__mu,
            'y_lin': self.__y_lin,
            'p_t': self.__p_t,
            'pf_t': self.__pf_t,
            'p_qp': self.__p_qp
        }.items() if v is not None}
    
    @property
    def init(self):
        return {k:v for k,v in self.__init.items()}
    
    def __setInit(self, value):
        self.__init = self.__init | self.__checkInit(value)
        

    def __checkInit(self, value):

        # preallocate output dictionary
        out = {}

        # check if input dictionary contains 'y_lin' key
        if 'y_lin' in value:

            if 'y_lin' not in self.param:
                raise Exception('Current MPC does not require a linearization trajectory.')

            # if so, extract y_lin
            y_lin = value['y_lin']

            # check if y_lin has correct dimension (unless the 'adpative' or the 'optimal' option is passed)
            if (y_lin != 'adaptive' or y_lin != 'optimal') and (y_lin.shape[0] != self.param['y_lin'].shape[0]):
                raise Exception('y_lin has incorrect dimension.')

            # add new initial linearization
            out = {'y_lin':y_lin}

        return out

    # overwrite the __dir__ method
    def __dir__(self):
        return [attr for attr in super().__dir__() if not attr.startswith('_QP__')]
    
    def debug(self,lam,mu,p_t,epsilon=1e-6,roundoff=10,y_all=None):

        # get full derivative
        J_QP = QP.J_y_p(lam,mu,p_t)
        
        # prepare vector containing derivative
        J_num = DM(*J_QP.shape)
        
        # check against numerical differences
        j = 0
        for v in np.eye(p_t.shape[0]):
        
            # compute y_all for small perturbations around p_t
            _,_,y_all_forward = QP.solve(p_t+DM(v*epsilon),y_all,lam,mu)
            _,_,y_all_backward = QP.solve(p_t-DM(v*epsilon),y_all,lam,mu)
            
            # get finite difference estimate 
            dy_num = (y_all_forward-y_all_backward)/(2*epsilon)
            
            # store in J_num
            J_num[:,j] = dy_num
            
            # increment counter
            j = j + 1
        
        # compute error
        error_abs = np.round(np.linalg.norm(J_QP-J_num,axis=0),roundoff)
        error_rel = np.where(np.linalg.norm(J_num, axis=0) == 0, 0, error_abs / np.linalg.norm(J_num, axis=0))

        return {'J_num':J_num,'J_QP':J_QP,'error_abs':error_abs,'error_rel':error_rel}