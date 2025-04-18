import casadi as ca
from time import time

"""
TODO

* better init function?

"""

class QP:
    
    # Allowed option keys
    __ALLOWED_OPTION_KEYS = ['linearization','slack','qp_mode','solver','warmstart','jac_tol','jac_gamma']

    # default values of options dictionary
    __DEFAULT_OPTIONS = {'linearization':'trajectory','slack':False,'qp_mode':'stacked','solver':'qpoases',
                         'warmstart':'x_lam_mu','jac_tol':8,'jac_gamma':0.001,'compile_qp_sparse':False,
                         'compile_jac':False}

    def __init__(self,ingredients,p=None,pf=None,options={}):

        # copy dimensions and variables inside ingredients
        self.__sym = ingredients._Ingredients__sym.copy()

        # copy ingredients
        self.__ingredients = ingredients

        # if p is passed, store it in parameters of QP
        if p is not None:
            assert isinstance(p,ca.SX), 'p must be of type SX'
            self.__sym.addVar('p_t',p)

        # TODO: check if order of variables in p_QP is correct

        # construct the parameter vector required for the QP
        p_QP = self.__sym.var.values()
        p_QP_names = self.__sym.var.keys()

        # save full symbolic qp parameter (do not include pf even if present)
        # self.__sym.var.addVar('p_QP',ca.vcat(p_QP))
        self.__sym.var.addDim('p_qp',ca.vcat(p_QP).shape[0])

        # create input index
        idx_in = dict()
        # initialize counter that reveals the beginning index of each variable
        running_idx = 0
        # loop through all the symbolic variables
        for i in range(len(p_QP)):
            # get length of current variable in p_d
            len_current_p_d = p_QP[i].shape[0]
            # store indexing of current variable
            idx_in[p_QP_names[i]] = range(running_idx,running_idx+len_current_p_d)
            # increment running index
            running_idx = running_idx + len_current_p_d

        # store in QP index dictionary
        self.__ingredients.__sparse['idx']['in'] = idx_in

        # if pf is passed, store it in parameters of QP
        if pf is not None:
            assert isinstance(pf,ca.SX), 'p must be of type SX'
            self.__sym.addVar('pf_t',pf)
        else:
            # add dimension
            self.__sym.addDim('pf_t',0)

        # store dimensions of equality and inequality constraints
        self.__sym.addDim('in',ingredients.sparse['G'].shape[0])
        self.__sym.addDim('eq',ingredients.sparse['F'].shape[0])

        # primal optimization variables
        self.__sym.addVar('y',ca.SX.sym('y',ingredients.sparse['q'].shape[0]-self.dim['eps'],1))

        # dual optimization variables (inequality constraints)
        self.__sym.addVar('lam',ca.SX.sym('lam',ingredients.sparse['g'].shape[0],1))

        # dual optimization variables (equality constraints)
        self.__sym.addVar('mu',ca.SX.sym('mu',ingredients.sparse['f'].shape[0],1))

        # dual optimization variable (all constraints)
        self.__sym.addVar('z',ca.vcat([self.__sym.var['lam'],self.__sym.var['mu']]))

        # create QP
        self.__makeSparseQP(p=p,pf=pf,mode=self.QP.options['qp_mode'],solver=self.QP.options['solver'],warmstart=self.QP.options['warmstart'],compile=self.QP.options['compile_qp_sparse'])

        # create conservative jacobian
        self.__makeConsJac(gamma=self.QP.options['jac_gamma'],tol=self.QP.options['jac_tol'],compile=self.QP.options['compile_jac'])

        __solve = None
        __denseSolve = None
        __qp_sparse = None
        __qp_dense = None
        __dual_sparse = None
        __J = None
        __J_y_p = None

    def __makeSparseQP(self,p=None,pf=None,solver='qpoases',warmstart='x_lam_mu',compile=False):

        # compilation options
        if compile:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            options = {}

        # extract ingrediens
        Q = self.ingredients['Q']
        q = self.ingredients['q']
        A = self.ingredients['A']
        lba = self.ingredients['lba']
        uba = self.ingredients['uba']
        H = self.ingredients['H']
        h = self.ingredients['h']

        # form QP ingredients
        QP_outs = [A,lba,uba,Q,q]
        QP_outs_names = ['A','lba','uba','Q','q']

        # create function
        start = time.time()
        QP_func = ca.Function('QP',[ca.vcat(p_QP)],QP_outs,['p'],QP_outs_names,options)
        comp_time_dict = {'QP_func':time.time()-start}

        # save QP in model
        self.QP._QP__setQpSparse(QP_func)

        # create function
        start = time.time()
        dual_func = ca.Function('dual',[ca.vcat(p_QP)],[H,h],['p'],['H','h'],options)
        comp_time_dict['dual_func'] = time.time()-start

        # save dual in model
        self.QP._QP__setDualSparse(dual_func)

        # extract dense ingredients
        dense_qp = self.QP.ingredients['dense']

        # dense outputs
        QP_outs_dense_names = list(dense_qp.keys())
        QP_outs_dense = list(dense_qp.values())

        # create function
        start = time.time()
        QP_dense_func = ca.Function('QP_dense',[ca.vcat(p_QP)],QP_outs_dense,['p'],QP_outs_dense_names,options)
        comp_time_dict['QP_dense_func'] = time.time()-start

        # save in QP model
        self.QP._QP__setQpDense(QP_dense_func)

        # implement QP using conic interface to retrieve multipliers
        qp = {}
        qp['h'] = Q.sparsity()
        qp['a'] = A.sparsity()

        # vector specifying which constraints are equalities
        is_equality = [False]*self.dim['in']+[True]*self.dim['eq']

        # add existing options to compile function S
        start = time.time()
        match solver:
            case 'gurobi':
                S = conic('S','gurobi',qp,{'gurobi':{'OutputFlag':0},'equality':is_equality})
            case 'cplex':
                S = conic('S','cplex',qp,{'cplex':{'CPXPARAM_Simplex_Display': 0,'CPXPARAM_ScreenOutput': 0},'equality':is_equality})
            case 'qpoases':
                S = conic('S','qpoases',qp,options|{'printLevel':'none','equality':is_equality})
            case 'osqp':
                S = conic('S','osqp',qp,options|{'osqp':{'verbose':False},'equality':is_equality})
            case 'daqp':
                S = conic('S','daqp',qp,options)
            case 'qrqp':
                S = conic('S','qrqp',qp,options|{'print_iter':False,'equality':is_equality})
        comp_time_dict['S_sparse'] = time.time()-start

        if mode == 'stacked':

            # create local function setting up the qp
            match warmstart:
                
                # warmstart both primal and dual variables
                case 'x_lam_mu':
            
                    def local_qp(p_qp,x0=None,lam0=None,mu0=None):

                        # get data from qp_dense function
                        a,lba,uba,h,g = QP_func(p_qp)

                        # solve QP with warmstarting
                        if x0 is not None:
                            sol = S(h=h,a=a,g=g,lba=lba,uba=uba,x0=x0,lam_a0=vertcat(lam0,mu0))
                        
                        # if does not work, try without warmstarting
                        else:
                            sol = S(h=h,a=a,g=g,lba=lba,uba=uba)
                        
                        # return lambda, mu, y
                        return sol['lam_a'][:self.dim['in']],sol['lam_a'][self.dim['in']:],sol['x']
                    
                # warmstart only primal
                case 'x':
            
                    def local_qp(p_qp,x0=None):

                        # get data from qp_dense function
                        a,lba,uba,h,g = QP_func(p_qp)

                        # solve QP with warmstarting
                        if x0 is not None:
                            sol = S(h=h,a=a,g=g,lba=lba,uba=uba,x0=x0)
                        
                        # if does not work, try without warmstarting
                        else:
                            sol = S(h=h,a=a,g=g,lba=lba,uba=uba)
                        
                        # return lambda, mu, y
                        return sol['lam_a'][:self.dim['in']],sol['lam_a'][self.dim['in']:],sol['x']
                    
        else:
            raise Exception('Separate mode not implemented yet.')
        
        # save in model
        self.QP._QP__setSolver(local_qp)

        # store computation times (if compile is true)
        if compile:
            self.__compTimes = self.__compTimes | comp_time_dict

    def __makeDenseQP(self,p,solver='qpoases',compile=False):

        """
        Given a numerical value of p, this function constructs the dense QP ingredients and sets up the QP solver,
        which can be accessed through QP.denseSolve.
        """

        # compilation options
        if compile:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            options = {}

        # get p_qp parameter
        p_qp_list = self.param['p_qp'].copy()

        # replace p with numerical value (if y lin is present, p is entry 3, otherwise 2)
        if self.QP.y_lin is not None:
            p_qp_list[2] = DM(p)
        else:
            p_qp_list[1] = DM(p)

        # get data from qp_dense function
        G_x,G_u,g_c,Qx,Ru,x_ref,u_ref,Hx,Hu,hx,hu = self.QP._QP__qp_dense(vcat(p_qp_list))

        # turn to DM
        Qx = DM(Qx)
        Ru = DM(Ru)
        x_ref = DM(x_ref)
        u_ref = DM(u_ref)
        Hx = DM(Hx)
        Hu = DM(Hu)
        hx = DM(hx)
        hu = DM(hu)

        # get initial state
        x = p_qp_list[0]

        # create propagation of initial state
        x_prop = G_x@x+g_c

        # create cost
        Q = G_u.T@Qx@G_u + Ru
        q = G_u.T@Qx@(x_prop - x_ref) - Ru@u_ref

        # create inequality constraint matrices
        G = vertcat(Hx@G_u,Hu)
        uba = vertcat(hx-Hx@x_prop,hu)
        lba = -inf*DM.ones(uba.shape)

        # initialize list of symbolic outputs
        out_list_symbolic = [Q,q,G,lba,uba,G_u,G_x,g_c]
        out_list_symbolic_names = ['Q','q','G','lba','uba','G_u','G_x','g_c']

        # re-create function
        start = time.time()
        QP_func = Function('QP_dense',[vcat(self.param['p_qp'])],out_list_symbolic,['p'],out_list_symbolic_names,options)
        comp_time_dict = {'QP_dense':time.time()-start}

        # implement QP using conic interface to retrieve multipliers
        qp = {}
        qp['h'] = Q.sparsity()
        qp['a'] = G.sparsity()

        # vector specifying which constraints are equalities
        is_equality = [False]*self.dim['in']

        # add existing options to compile function S
        start = time.time()
        match solver:
            case 'gurobi':
                S = conic('S','gurobi',qp,{'gurobi':{'OutputFlag':0},'equality':is_equality})
            case 'cplex':
                S = conic('S','cplex',qp,{'cplex':{'CPXPARAM_Simplex_Display': 0,'CPXPARAM_ScreenOutput': 0},'equality':is_equality})
            case 'qpoases':
                S = conic('S','qpoases',qp,options|{'printLevel':'none','equality':is_equality})
            case 'osqp':
                S = conic('S','osqp',qp,options|{'osqp':{'verbose':False},'equality':is_equality})
            case 'daqp':
                S = conic('S','daqp',qp,options)
            case 'qrqp':
                S = conic('S','qrqp',qp,options|{'print_iter':False,'equality':is_equality})
        comp_time_dict = comp_time_dict | {'QP_dense':time.time()-start}

        # define qp solver
        def local_qp(p_qp,x0=None,lam=None,mu=None):

            # get data from qp_dense function
            Q,q,G,lba,uba,G_u,G_x,g_c = QP_func(p_qp)

            # get initial condition
            x = p_qp[:self.dim['x']]

            # solve QP with warmstarting
            if x0 is not None:
                
                # extract input
                u0 = x0[self.dim['x']*self.dim['N']:]

                # solve
                u = S(h=Q,a=G,g=q,lba=lba,uba=uba,x0=u0)['x']
            
            # if does not work, try without warmstarting
            else:
                u = S(h=Q,a=G,g=q,lba=lba,uba=uba)['x']

            # create state prediction
            x = G_x@x + G_u@u + g_c
            
            # return lambda, mu, y
            return DM(self.dim['in'],1),DM(self.dim['eq'],1),vertcat(x,u,DM(self.dim['eps'],1))
        
        # save in model
        self.QP._QP__setDenseSolver(local_qp)

        # store computation times (if compile is true)
        if compile:
            self.__compTimes = self.__compTimes | comp_time_dict

    def __makeConsJac(self,gamma=0.001,tol=8,compile=False):
        
        """
        This function sets up the functions that compute the conservative jacobian of the QP solution.
        Specifically, two functions are set in the QP class:

            - J: this function takes in the dual variables lam, mu, and the parameters p_QP necessary 
                 to setup the QP (including pf_t), and returns the following quantities in a list:

                 - J_F_z: conservative jacobian of dual fixed point condition wrt z
                 - J_F_p: conservative jacobian of dual fixed point condition wrt p_t
                 - J_y_p: conservative jacobian of primal variable wrt p_t
                 - J_y_z_mat: conservative jacobian of primal variable wrt z

            - J_y_p: this function takes in lam, mu, p_QP (including pf_t), and optionally t (which
              defaults to 1), and returns the inner product between the conservative jacobian J_y_p
              of y wrt p_t and t.
        """

        # compilation options
        if compile:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            options = {}

        # get symbolic variable type
        MSX = self.__MSX
        
        # check if p was passed (required to compute conservative jacobian)
        if 'p_qp' not in self.QP.param:
            raise Exception('Parameter p_qp is required to compute conservative jacobian.')
            
        # turn to column vector
        # p = vcat(self.QP.param['p_qp'])
        # p_full = p
        p_full = self.QP.param['p_qp']

        # check if pf was passed
        if 'pf_t' in self.param:
            # pf = vcat(self.param['pf_t'])
            # if so, add to p_full
            # p_full.append(pf)
            pf = self.param['pf_t']
            p = vcat([param for param in p_full if not(depends_on(param, pf))])
        else:
            p = vcat(p_full)

        # turn p_full into column vector
        p_full = vcat(p_full)        

        # extract multipliers
        lam = self.param['lam']
        mu = self.param['mu']

        # create list of inputs and their names
        dual_params = [lam,mu,p_full]
        dual_params_names = ['lam','mu','p']

        # extract dimensions
        n_z = self.dim['z']
        n_eq = self.dim['eq']

        # extract dual data
        H = self.QP.ingredients['H']
        h = self.QP.ingredients['h']

        # extract QP data
        Qinv = self.QP.ingredients['Qinv']
        q = self.QP.ingredients['q']
        F = self.QP.ingredients['F']
        G = self.QP.ingredients['G']

        # compute conservative jacobian of projector
        J_Pc = diag(vertcat(vec(sign(lam)),MSX.ones(n_eq,1)))
        J_F_z = J_Pc@(MSX.eye(n_z)-gamma*H)-MSX.eye(n_z)
        J_F_p = - gamma*J_Pc@( jacobian(H@vertcat(lam,mu)+h,p) )

        # compute conservative jacobian of primal variable
        y = -Qinv@(G.T@lam+F.T@mu+q)
        J_y_p = jacobian(y,p)
        J_y_z_mat = -Qinv@horzcat(G.T,F.T)

        # sparsify
        try:
            J_Pc = cse(sparsify(J_Pc))
            J_F_z = cse(sparsify(J_F_z))
            J_F_p = cse(sparsify(J_F_p))
            y = cse(sparsify(y))
            J_y_p = cse(sparsify(J_y_p))
            J_y_z_mat = cse(sparsify(J_y_z_mat))
        except:
            pass

        # stack all parameters
        dual_outs = [J_F_z,J_F_p,J_y_p,J_y_z_mat]
        dual_outs_names = ['J_F_z','J_F_p','J_y_p','J_y_z_mat']

        # turn into function
        start = time.time()
        J = Function('J',dual_params,dual_outs,dual_params_names,dual_outs_names,options)
        comp_time_dict = {'J':time.time()-start}

        def J_y_p(lam,mu,p_qp,t=1):

            # round lambda (always first entry in dual_params) to avoid numerical issues
            lam = np.round(np.array(fmax(lam,0)),tol)
            
            # get all conservative jacobian and matrices
            J_F_z,J_F_p,J_y_p,J_y_z_mat = J(lam,mu,p_qp)

            # get conservative jacobian of dual solution
            A = -solve(J_F_z,J_F_p@t,'csparse')

            # return conservative jacobian of primal
            return J_y_p@t+J_y_z_mat@A

        # save in QP
        self.QP._QP__set_J(J)
        self.QP._QP__set_J_y_p(J_y_p)

        # store computation times (if compile is true)
        if compile:
            self.__compTimes = self.__compTimes | comp_time_dict

    @property
    def ingredients(self):
        return self.__ingredients

    @property
    def idx(self):
        return self.__idx

    @property
    def solve(self):
        return self.__solve

    @property
    def denseSolve(self):
        return self.__denseSolve

    @property
    def qp_sparse(self):
        return self.__qp_sparse
    
    def __setQpSparse(self, value):
        self.__qp_sparse = value

    @property
    def qp_dense(self):
        return self.__qp_dense

    @property
    def dual_sparse(self):
        return self.__dual_sparse

    @property
    def J(self):
        return self.__J

    @property
    def J_y_p(self):
        return self.__J_y_p

    @property
    def param(self):
        # dictionary containing all symbolic variables of the class
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
    def options(self):
        return self.__options

    def __updateOptions(self, value):
        """
        Update the options of the class.
        """

        # check if value is a dictionary
        if not isinstance(value, dict):
            raise Exception('Options must be a dictionary.')
        
        # remove keys that are not allowed
        value = {k:v for k,v in value.items() if k in self.__allowed_options_keys}

        # update options dictionary
        self.__options = self.__options | value
    
    @property
    def init(self):
        return {k:v for k,v in self.__init.items()}
    
    def __setInit(self, value):
        self.__init = self.__init | self.__checkInit(value)

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