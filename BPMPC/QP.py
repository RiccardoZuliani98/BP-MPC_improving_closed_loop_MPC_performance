from casadi import *

class QP:
    
    # Allowed option keys
    __ALLOWED_OPTION_KEYS = ['linearization','slack','qp_mode','solver','warmstart','jac_tol','jac_gamma']

    # default values of options dictionary
    __DEFAULT_OPTIONS = {'linearization':'trajectory','slack':False,'qp_mode':'stacked','solver':'qpoases',
                         'warmstart':'x_lam_mu','jac_tol':8,'jac_gamma':0.001,'compile_qp_sparse':False,
                         'compile_jac':False}

    def __init__(self,MSX):

        __x = None
        __y = None
        __lam = None
        __mu = None
        __z = None
        __y_lin = None
        __p_t = None
        __pf_t = None
        __p_qp = None
        __init = {'y_lin':None}

        __ingredients = {}
        __idx = {}


        __solve = None
        __denseSolve = None
        __qp_sparse = None
        __qp_dense = None
        __dual_sparse = None
        __J = None
        __J_y_p = None
        

        # check type of symbolic variables
        assert MSX in ['SX','MX'], 'MSX must be either SX or MX'
        self.__MSX = SX if MSX == 'SX' else MX

    def update(self,x,G,g,F,f,Q,Qinv,q,idx,y_lin,denseQP,p=None,pf=None,options={}):

        # update options
        self.__updateOptions(options)

        # add initial state to QP variables
        self.__x = x

        # add y_lin to QP variables
        if y_lin is not None:
            self.__y_lin = y_lin
        else:
            self.__updateOptions({'linearization':'none'})

        # check if slack variables are present
        if 'eps' in idx:
            self.__addDim({'eps': len(idx['eps'])})
        else:
            self.__addDim({'eps': 0})
        
        # add horizon of MPC to dimensions
        self.__addDim({'N': int(len(idx['x']) / self.dim['x'])})

        # save dimensions in variable with shorter name for simplicity
        n = self.dim

        #TODO: add dimension checks

        # store in dictionary
        QP_dict = dict()
        QP_dict['Q'] = Q
        QP_dict['Qinv'] = Qinv
        QP_dict['q'] = q
        QP_dict['G'] = G
        QP_dict['g'] = g
        QP_dict['F'] = F
        QP_dict['f'] = f
        QP_dict['A'] = A
        QP_dict['lba'] = lba
        QP_dict['uba'] = uba
        QP_dict['H'] = H
        QP_dict['h'] = h
        QP_dict['dense'] = denseQP


        ### STORE IN MPC -----------------------------------------------------

        # store dimensions of equality and inequality constraints
        self.__addDim({'in': G.shape[0]})
        self.__addDim({'eq': F.shape[0]})

        # store ingredients
        self.QP._QP__setIngredients(QP_dict)

        # store index
        self.QP._QP__updateIdx({'out':idx})

        # primal optimization variables
        self.QP._QP__set_y(self.__MSX.sym('y',q.shape[0]-n['eps'],1))

        # dual optimization variables (inequality constraints)
        self.QP._QP__set_lam(self.__MSX.sym('lam',g.shape[0],1))

        # dual optimization variables (equality constraints)
        self.QP._QP__set_mu(self.__MSX.sym('mu',f.shape[0],1))

        # dual optimization variable (all constraints)
        self.QP._QP__set_z(vcat([self.QP.lam,self.QP.mu]))

        # add dimensions
        self.__addDim({k: v.shape[0] for k, v in self.QP.param.items()})

        # create QP
        self.__makeQP(p=p,pf=pf,mode=self.QP.options['qp_mode'],solver=self.QP.options['solver'],warmstart=self.QP.options['warmstart'],compile=self.QP.options['compile_qp_sparse'])

        # create conservative jacobian
        self.__makeConsJac(gamma=self.QP.options['jac_gamma'],tol=self.QP.options['jac_tol'],compile=self.QP.options['compile_jac'])

    def __makeSparseQP(self,p=None,pf=None,mode='stacked',solver='qpoases',warmstart='x_lam_mu',compile=False):
        
        """
        This function creates the functions necessary to solve the MPC problem in QP form. Specifically, this function
        sets the following properties of QP:

            - qp_sparse: this function takes in p_QP and returns the sparse ingredients, which are:
                
                - F,f,G,g,Q,q in the separate mode, where the QP is formulated as

                    min 1/2 y'Qy + q'y
                    s.t. Gy <= g
                         Fy = f

                - A,lba,uba,Q,q in the stacked mode, where the QP is formulated as

                    min 1/2 y'Qy + q'y
                    s.t. lba <= Ay <= uba

            - dual_sparse: this function takes in p_QP and returns the dual ingredients H,h, where the dual
              QP is formulated as:

                min 1/2 z'Hz + h'z
                s.t. z >= 0
            
            - qp_dense: this function takes in p_QP and returns the dense ingredients, which are:

                - 'G_x', 'G_u', 'g_c': matrices satisfying x = G_x*x0 + G_u*u + g_c
                - 'Qx', 'Ru', 'x_ref', 'u_ref': cost function (x-x_ref)'Qx(x-x_ref) + (u-u_ref)'Ru(u-u_ref)
                - 'Hx', 'Hu', 'hx', 'hu': polyhedral constraints Hx*x <= hx, Hu*u <= hu
              
              where the QP is formulated as

                min 1/2 u'(G_x'QxG_x + Ru)u + (G_x'Qx(G_x*x0 + g_c - x_ref) - Ru*u_ref)'u
                s.t. Hx*(G_x*x0 + G_u*u + g_c) <= hx
                     Hu*u <= hu

            - solve: this function takes in p_QP and returns the optimal solution of the QP problem.
              Based on the type of warmstarting (i.e. x_lam_mu or x), the function will take the following 
              inputs:

                - x_lam_mu: p_QP, x0=None, lam=None, mu=None
                - x: p_QP, x0=None

              the output is lam, mu, y.
                
        This function also sets up all the symbolic variables and their dimensions.

        Note that p_QP contains all necessary parameters required to setup the ingredients at any given time-step,
        e.g. x0,y_lin,p_t,pf_t). This is different from the p_QP stored as a parameter, which does not contain pf.

        Additionally, this functions sets up the 'in' entry of the idx dictionary of scenario.QP, which contains the
        indexing of all the input parameters in p_QP:

            - 'x0': initial state
            - 'y_lin': linearization trajectory
            - 'p_t': parameters that are optimized in the upper-level
        
        Note that not all parameters need to be present.

        """

        # compilation options
        if compile:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            options = {}

        # first construct the parameter vector required for the QP
        p_QP = []
        p_QP_names = []

        # the MPC problem always depends on the current state
        x = self.QP.x
        p_QP.append(x)
        p_QP_names.append('x')

        # there may be also a linearization trajectory
        if self.QP.y_lin is not None:

            # get linearization trajectory
            y_lin = self.QP.y_lin

            # append to list
            p_QP.append(y_lin)
            p_QP_names.append('y_lin')
        else:

            # if y_lin is not present, set it as a zero-dimensional SX
            y_lin = SX(0,0)

        # then add p
        if p is not None:

            # store in parameters of QP
            self.QP._QP__set_p_t(p)

            # append to list
            p_QP.append(p)
            p_QP_names.append('p_t')

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
        self.QP._QP__updateIdx({'in':idx_in})

        # save full symbolic qp parameter (do not include pf even if present)
        self.QP._QP__set_p_qp(p_QP)

        # add dimension
        self.__addDim({'p_qp': vcat(p_QP).shape[0]})

        # add pf (i.e., fixed parameters that are not differentiated)
        if pf is not None:

            # append to list
            p_QP.append(pf)
            p_QP_names.append('pf')

            # save symbolic parameter
            self.QP._QP__set_pf_t(pf)
        
            # add dimension
            self.__addDim({'pf_qp': pf.shape[0]})
        
        else:
            # add dimension
            self.__addDim({'pf_qp': 0})

        # extract ingrediens
        Q = self.QP.ingredients['Q']
        q = self.QP.ingredients['q']
        F = self.QP.ingredients['F']
        f = self.QP.ingredients['f']
        G = self.QP.ingredients['G']
        g = self.QP.ingredients['g']
        A = self.QP.ingredients['A']
        lba = self.QP.ingredients['lba']
        uba = self.QP.ingredients['uba']
        H = self.QP.ingredients['H']
        h = self.QP.ingredients['h']

        # select outputs
        if mode == 'separate':
            QP_outs = [F,f,G,g,Q,q]
            QP_outs_names = ['F','f','G','g','Q','q']
            raise Exception('Separate mode not implemented yet.') #TODO implement separate mode
        elif mode == 'stacked':
            QP_outs = [A,lba,uba,Q,q]
            QP_outs_names = ['A','lba','uba','Q','q']

        # create function
        start = time.time()
        QP_func = Function('QP',[vcat(p_QP)],QP_outs,['p'],QP_outs_names,options)
        comp_time_dict = {'QP_func':time.time()-start}

        # save QP in model
        self.QP._QP__setQpSparse(QP_func)

        # create function
        start = time.time()
        dual_func = Function('dual',[vcat(p_QP)],[H,h],['p'],['H','h'],options)
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
        QP_dense_func = Function('QP_dense',[vcat(p_QP)],QP_outs_dense,['p'],QP_outs_dense_names,options)
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
    def x(self):
        # initial state (n_x,1)
        return self.__x

    @property
    def y(self):
        # primal optimization variables (n_y,1)
        return self.__y

    @property
    def lam(self):
        # dual multipliers of the inequality constraints (n_in,1)
        return self.__lam

    @property
    def mu(self):
        # dual multipliers of the equality constraints (n_eq,1)
        return self.__mu

    @property
    def z(self):
        # dual multipliers (n_in + n_eq,1)
        return self.__z
    
    @property
    def y_lin(self):
        # linearization trajectory (n_y_lin,1)
        return self.__y_lin

    @property
    def ingredients(self):
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
        return self.__ingredients

    @property
    def idx(self):
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
        return self.__idx

    @property
    def solve(self):
        """
        this function takes in p_QP and returns the optimal solution of the QP problem. Based on the type
        of warmstarting (i.e. x_lam_mu or x), the function will take the following inputs:

            - x_lam_mu: p_QP (including pf_t), x0=None, lam=None, mu=None
            - x: p_QP (including pf_t), x0=None

        the output is lam, mu, y.
        """
        return self.__solve

    @property
    def denseSolve(self):
        """
        this function takes in p_QP and returns the optimal solution of the QP problem in dense form.
        Based on the type of warmstarting (i.e. x_lam_mu or x), the function will take the following inputs:

            - x_lam_mu: p_QP (including pf_t), x0=None, lam=None, mu=None
            - x: p_QP (including pf_t), x0=None

        the output is lam, mu, y.
        """
        return self.__denseSolve

    @property
    def qp_sparse(self):
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
        return self.__qp_sparse
    
    def __setQpSparse(self, value):
        self.__qp_sparse = value

    @property
    def qp_dense(self):
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
        return self.__qp_dense

    @property
    def dual_sparse(self):
        """
        This function takes in p_QP (including pf_t) and returns the dual ingredients H,h, where the dual
        QP is formulated as:

            min 1/2 z'Hz + h'z
            s.t. z >= 0
        """
        return self.__dual_sparse

    @property
    def p_t(self):
        # design parameter of QP at time t (dimension not stored)
        return self.__p_t

    @property
    def pf_t(self):
        # fixed parameter of QP at time t (e.g. reference) (n_pf_t,1)
        return self.__pf_t

    @property
    def p_qp(self):
        # symbolic parameters needed to setup QP at time t (n_p_qp,1)
        return self.__p_qp

    @property
    def J(self):
        """
        this function takes in the dual variables lam, mu, and the parameters p_QP necessary to setup the
        QP (including pf_t), and returns the following quantities in a list:

            - J_F_z: conservative jacobian of dual fixed point condition wrt z
            - J_F_p: conservative jacobian of dual fixed point condition wrt p_t
            - J_y_p: conservative jacobian of primal variable wrt p_t
            - J_y_z_mat: conservative jacobian of primal variable wrt z
        """
        return self.__J

    @property
    def J_y_p(self):
        """
        this function takes in lam, mu, p_QP (including pf_t), and optionally t (which defaults to 1), and
        returns the inner product between the conservative jacobian J_y_p of y wrt p_t and t.
        """
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
        """
        Options dictionary. Possible keys are:

            - 'linearization': 'trajectory', 'state' or 'none' (default is 'trajectory')
            - 'slack': True or False (default is False)
            - 'qp_mode': 'stacked' or 'separate' (default is 'stacked')
            - 'solver': 'qpoases','osqp','cplex','gurobi','daqp','qrqp' (default is 'qpoases')
            - 'warmstart': 'x_lam_mu' (warmstart both primal and dual variables) or 'x' (warmstart only 
                        primal variables) (default is 'x_lam_mu')
            - 'jac_tol': tolerance below which multipliers are considered zero (default is 8)
            - 'jac_gamma': stepsize in optimality condition used to apply the IFT (default is 0.001)
            - 'compile_qp_sparse': True or False (default is False)
            - 'compile_jac': True or False (default is False)
        """
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
        """
        init contains the initial value of the QP variables, it can be set through __setInit.
        """
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