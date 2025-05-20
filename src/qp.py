import casadi as ca
import time
from src.ingredients import Ingredients
from src.options import Options
import numpy as np
from copy import copy
from numpy.linalg import lstsq

class QP:
    """
    QP: Quadratic Program Solver Class
    This class provides a flexible interface for constructing, compiling, and solving quadratic programming (QP)
    problems using symbolic and numerical tools, primarily leveraging CasADi. It supports both sparse and dense
    QP formulations, warmstarting, multiple solver backends, and automatic Jacobian computation for sensitivity
    analysis. 
    Attributes:
        _OPTIONS_ALLOWED_VALUES (dict): Allowed keys and their valid values/types for solver options.
        _OPTIONS_DEFAULT_VALUES (dict): Default values for solver options.
        _options (Options): Instance managing current solver options.
        _sym (SymbolManager): Symbolic variable manager (copied from Ingredients).
        _ingredients (Ingredients): Problem data and symbolic variables.
        _qp_sparse (ca.Function): Compiled CasADi function for sparse QP.
        _dual_sparse (ca.Function): Compiled CasADi function for dual problem.
        _solve (callable): Local QP solver function for sparse QP.
        _denseSolve (callable): Local QP solver function for dense QP.
        _J (ca.Function): CasADi function for conservative Jacobian evaluation.
        _J_y_p (callable): Function for computing conservative Jacobian of the primal variable.
        _compTimes (dict): Compilation and setup timing information.
    Methods:
        __init__(self, ingredients, p=None, pf=None, options=None)
            Initializes the QP object with problem data, symbolic variables, parameters, and options.
        _makeSparseQP(self)
            Constructs and compiles the sparse QP and dual functions, sets up the solver interface, and defines the local QP solver.
        make_dense_qp(self, p)
            Constructs and compiles a dense QP solver using CasADi, based on current model parameters and options.
        _makeConsJac(self)
            Constructs and compiles conservative Jacobian functions for QP constraints and primal variables.
        debug(self, lam, mu, p_t, epsilon=1e-6, roundoff=10, y_all=None)
            Numerically checks the computed Jacobian against finite differences for debugging and validation.
    Properties:
        ingredients: Returns the Ingredients instance.
        idx: Returns the input index dictionary.
        solve: Returns the local sparse QP solver function.
        denseSolve: Returns the local dense QP solver function.
        qp_sparse: Returns the compiled sparse QP function.
        qp_dense: Returns the compiled dense QP function.
        dual_sparse: Returns the compiled dual function.
        J: Returns the CasADi Jacobian function.
        J_y_p: Returns the conservative Jacobian computation function.
        param: Returns the symbolic parameter dictionary.
        dim: Returns the symbolic dimensions dictionary.
        options: Returns the current options.
        init: Returns the dictionary of initialized symbolic variables.
    Usage:
        - Instantiate with problem Ingredients and optional parameters/options.
        - Call `solve` or `denseSolve` to solve the QP.
        - Use Jacobian functions for sensitivity analysis.
        - Use `debug` to validate Jacobian computations.
        - Requires CasADi and NumPy.
        - Supports multiple QP solvers (e.g., qpoases, daqp, gurobi, cplex, osqp, qrqp).
        - Designed for use in model predictive control (MPC) and related optimization tasks.
    """
    
    # Allowed option keys
    _OPTIONS_ALLOWED_VALUES = {'solver':['qpoases','daqp','osqp'],'dense_solver':['qpoases','daqp','osqp'],
                               'warmstart':['x_lam_mu','x'],'jac_tol':int,'jac_gamma':float,
                               'compile_qp_sparse':bool,'compile_qp_dense':bool,'compile_jac':bool,
                               'ls_solver':['numpy','casadi']}

    # default values of options dictionary
    _OPTIONS_DEFAULT_VALUES = {'solver':'qpoases','dense_solver':'qpoases',
                               'warmstart':'x_lam_mu','jac_tol':8,'jac_gamma':0.001,
                               'compile_qp_sparse':False,'compile_qp_dense':False,'compile_jac':False,
                               'ls_solver':'casadi'}

    def __init__(self,ingredients,p=None,pf=None,options=None):
        """
        Initializes the QP (Quadratic Program) object with the provided ingredients, parameters, and options.
        Parameters:
            ingredients (Ingredients): An instance of the Ingredients class containing all necessary problem data and symbolic variables.
            p (ca.SX, optional): Optional symbolic parameter vector to be included in the QP. Must be of type casadi.SX.
            pf (ca.SX, optional): Optional terminal parameter vector to be included in the QP. Must be of type casadi.SX.
            options (dict, optional): Dictionary of user-specified options to override default and ingredient options.
        Raises:
            AssertionError: If `ingredients` is not an instance of Ingredients.
            AssertionError: If `p` or `pf` are provided and are not of type casadi.SX.
        Notes:
            - Copies symbolic variables and options from the provided ingredients.
            - Constructs and stores the full parameter vector required for the QP.
            - Sets up indexing for input variables.
            - Initializes symbolic variables for primal and dual optimization variables.
            - Prepares the sparse QP structure and conservative Jacobian.
        """

        # check if options is not passed
        if options is None:
            options = {}

        assert isinstance(ingredients,Ingredients), 'Ingredients must be of a class instance of type Ingredients'

        # create options
        self._options = Options(self._OPTIONS_ALLOWED_VALUES,self._OPTIONS_DEFAULT_VALUES) + ingredients.options

        # add user-specified options
        self._options.update(options)

        # copy dimensions and variables inside ingredients
        self._sym = ingredients._sym.copy()

        # copy ingredients
        self._ingredients = ingredients

        # if p is passed, store it in parameters of QP
        if p is not None:
            assert isinstance(p,ca.SX), 'p must be of type SX'
            self._sym.add_var('p_t',p)

        # TODO: check if order of variables in p_QP is correct

        # construct the parameter vector required for the QP
        p_QP = list(self._sym.var.values())
        p_QP_names = list(self._sym.var.keys())

        # save full symbolic qp parameter (do not include pf even if present)
        # self._sym.add_dim('p_qp',ca.vcat(p_QP).shape[0])
        self._sym.add_var('p_qp',ca.vcat(p_QP))

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
        self._ingredients._idx['in'] = idx_in

        # if pf is passed, store it in parameters of QP
        if pf is not None:
            assert isinstance(pf,ca.SX), 'p must be of type SX'
            self._sym.add_var('pf_t',pf)
            p_QP.append(pf)
        else:
            # add dimension
            self._sym.add_dim('pf_t',0)

        # store full qp parameter
        self._sym.add_var('p_qp_full',ca.vcat(p_QP))

        # store dimensions of equality and inequality constraints
        self._sym.add_dim('in',ingredients.sparse['G'].shape[0])
        self._sym.add_dim('eq',ingredients.sparse['F'].shape[0])

        # primal optimization variables
        self._sym.add_var('y',ca.SX.sym('y',ingredients.sparse['q'].shape[0]-self._sym.dim['eps'],1))

        # dual optimization variables (inequality constraints)
        self._sym.add_var('lam',ca.SX.sym('lam',ingredients.sparse['g'].shape[0],1))

        # dual optimization variables (equality constraints)
        self._sym.add_var('mu',ca.SX.sym('mu',ingredients.sparse['f'].shape[0],1))

        # dual optimization variable (all constraints)
        self._sym.add_var('z',ca.vcat([self._sym.var['lam'],self._sym.var['mu']]))

        # create sparse QP
        self._makeSparseQP()

        # create conservative jacobian
        self._makeConsJac()

    def _makeSparseQP(self):
        """
        Constructs and compiles sparse Quadratic Programming (QP) and dual functions for the model,
        sets up the QP solver interface, and defines a local QP solver function with optional warmstarting.
        This method performs the following steps:
        1. Sets compilation options for just-in-time (JIT) compilation if enabled.
        2. Extracts sparse QP and dual problem ingredients from the model's ingredients.
        3. Checks that all symbolic outputs are a subset of the provided symbolic inputs.
        4. Creates CasADi functions for the QP and dual problem using the extracted ingredients.
        5. Sets up the QP solver interface using the specified solver (e.g., gurobi, cplex, qpoases, osqp, daqp, qrqp),
           configuring solver-specific options and equality constraints.
        6. Defines a local QP solver function (`local_qp`) that prepares the QP problem, applies warmstarting if requested,
           solves the QP, and returns the solution (primal and dual variables).
        7. Stores the compiled QP and dual functions, the local QP solver, and computation times in the model.
        Raises:
            AssertionError: If the QP ingredients depend on more symbolic inputs than provided.
        Side Effects:
            - Sets `self._qp_sparse` to the compiled QP function.
            - Sets `self._dual_sparse` to the compiled dual function.
            - Sets `self._solve` to the local QP solver function.
            - Sets `self._compTimes` to a dictionary of compilation times.
        Note:
            This method is intended for internal use and assumes that the model's ingredients and symbolic variables
            have been properly initialized.
        """

        # compilation options
        if self._options['compile_qp_sparse']:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            options = {}

        # extract ingrediens
        Q = self.ingredients.sparse['Q']
        q = self.ingredients.sparse['q']
        A = self.ingredients.sparse['A']
        lba = self.ingredients.sparse['lba']
        uba = self.ingredients.sparse['uba']
        H = self.ingredients.dual['H']
        h = self.ingredients.dual['h']

        # form QP ingredients
        QP_outs = [A,lba,uba,Q,q]
        QP_outs_names = ['a','lba','uba','h','g']

        # set of symbolic outputs
        sym_outputs = set(ca.symvar(ca.vcat([ca.vcat(ca.symvar(elem)) for elem in QP_outs])))

        # set of symbolic inputs
        sym_inputs = set(ca.symvar(self._sym.var['p_qp_full']))

        assert sym_outputs.issubset(sym_inputs), 'The QP ingredients depend on more inputs than the one you provided. Did you forget about p or pf? Missing symbols: '
        # sym_outputs.difference(sym_inputs)

        # create function
        start = time.time()
        QP_func = ca.Function('QP',[self._sym.var['p_qp_full']],QP_outs,['p'],QP_outs_names,options)
        comp_time_dict = {'QP_func':time.time()-start}

        # save QP in model
        self._qp_sparse = QP_func

        # create function
        start = time.time()
        dual_func = ca.Function('dual',[self._sym.var['p_qp_full']],[H,h],['p'],['H','h'],options)
        comp_time_dict['dual_func'] = time.time()-start

        # save dual in model
        self._dual_sparse = dual_func

        # implement QP using conic interface to retrieve multipliers
        qp = {}
        qp['h'] = Q.sparsity()
        qp['a'] = A.sparsity()

        # vector specifying which constraints are equalities
        is_equality = [False]*self._sym.dim['in']+[True]*self._sym.dim['eq']

        # add existing options to compile function S
        start = time.time()
        match self._options['solver']:
            case 'gurobi':
                S = ca.conic('S','gurobi',qp,{'gurobi':{'OutputFlag':0},'equality':is_equality})
            case 'cplex':
                S = ca.conic('S','cplex',qp,{'cplex':{'CPXPARAM_Simplex_Display': 0,'CPXPARAM_ScreenOutput': 0},'equality':is_equality})
            case 'qpoases':
                S = ca.conic('S','qpoases',qp,options | {'printLevel':'none','equality':is_equality})
                # S = ca.conic('S','qpoases',qp,options | {'equality':is_equality})
            case 'osqp':
                S = ca.conic('S','osqp',qp,options | {'osqp':{'verbose':False},'equality':is_equality})
                # S = ca.conic('S','osqp',qp,options | {'equality':is_equality})
            case 'daqp':
                S = ca.conic('S','daqp',qp,options)
            case 'qrqp':
                S = ca.conic('S','qrqp',qp,options | {'print_iter':False,'equality':is_equality})
        comp_time_dict['S_sparse'] = time.time()-start

        # create local function setting up the qp
        match self._options['warmstart']:
            
            # warmstart both primal and dual variables
            case 'x_lam_mu':
        
                def local_qp(p_qp,x0=None,lam0=None,mu0=None):

                    # check if warmstart is passed
                    warm_start = {'x0':x0,'lam_a0':ca.vertcat(lam0,mu0)} if x0 is not None else {}

                    # get data from qp_dense function
                    qp_ingredients = QP_func(p=p_qp)

                    # solve QP
                    sol = S.call(qp_ingredients | warm_start)

                    # get outputs
                    # lam_a_out = np.array(sol['lam_a'])
                    # x_out = np.array(sol['x'])
                    lam_a_out = sol['lam_a']
                    x_out = sol['x']
                    
                    # return lambda, mu, y
                    return lam_a_out[:self.dim['in']],lam_a_out[self.dim['in']:],x_out
                
            # warmstart only primal
            case 'x':
        
                def local_qp(p_qp,x0=None):

                    # check if warmstart is passed
                    warm_start = {'x0':x0} if x0 is not None else {}

                    # get data from qp_dense function
                    qp_ingredients = QP_func(p=p_qp)

                    # solve QP
                    sol = S(qp_ingredients | warm_start)

                    # get outputs
                    # lam_a_out = np.array(sol['lam_a'])
                    # x_out = np.array(sol['x'])
                    lam_a_out = sol['lam_a']
                    x_out = sol['x']
                    
                    # return lambda, mu, y
                    return lam_a_out[:self.dim['in']],lam_a_out[self.dim['in']:],x_out
        
        # save in model
        self._solve = local_qp

        # store computation times (if compile is true)
        self._comp_times = comp_time_dict

    def make_dense_qp(self,p):
        """
        Constructs and compiles a dense Quadratic Program (QP) solver using CasADi, based on the current model
        parameters and options. This method performs the following steps:
        1. Sets up compilation and JIT options for the QP function if enabled.
        2. Extracts dense QP ingredients from the model.
        3. Creates a CasADi function to compute dense QP outputs symbolically.
        4. Substitutes symbolic parameters with provided numerical values.
        5. Computes the QP matrices (Q, q, G, lba, uba) and other required ingredients.
        6. Re-creates a CasADi function for the QP with the computed outputs.
        7. Sets up the QP solver using the specified backend (e.g., gurobi, cplex, osqp, etc.).
        8. Defines a local QP solver function that supports warmstarting and returns the solution and multipliers.
        9. Stores the local QP solver and computation times in the model.
        
        Parameters:
            p (array-like or CasADi DM): The numerical value for the QP parameter vector, used to substitute symbolic parameters.
        
        Side Effects:
            - Sets `self._dense_solve` to the local QP solver function.
            - Updates `self._compTimes` with timing information for function compilation and solver setup.
        
        Notes:
            - The method supports multiple QP solvers via CasADi's conic interface.
            - Warmstarting is supported if an initial guess is provided to the local QP solver.
            - The function assumes that model parameters and dimensions are properly initialized.
        """

        # compilation options
        if self._options['compile_qp_dense']:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            options = {}

        # extract dense ingredients
        dense_qp = self.ingredients.dense

        # dense outputs
        QP_outs_dense_names = list(dense_qp.keys())
        QP_outs_dense = list(dense_qp.values())

        # create function
        start = time.time()
        QP_dense_func = ca.Function('QP_dense',[self._sym.var['p_qp_full']],QP_outs_dense,['p'],QP_outs_dense_names,options)
        self._comp_times['QP_dense_func'] = time.time()-start

        # get p_qp_full parameter
        p_qp_full = copy(self.param['p_qp_full'])

        # get symbolic value of p
        p_symb = self.param['p_t']

        # replace symbolic value of p with its numerical value (if provided)
        p_qp_full = ca.substitute(p_qp_full,p_symb,p)

        # get data from qp_dense function
        ingredients = QP_dense_func(p=p_qp_full)

        # get ingredients
        G_x = ingredients['G_x']
        G_u = ingredients['G_u']
        g_c = ingredients['g_c']

        # turn to DM
        Qx = ca.DM(ingredients['Qx'])
        Ru = ca.DM(ingredients['Ru'])
        Hx = ca.DM(ingredients['Hx'])
        Hu = ca.DM(ingredients['Hu'])
        hx = ca.DM(ingredients['hx'])
        hu = ca.DM(ingredients['hu'])

        # get initial state
        x = self.param['x']

        # create propagation of initial state
        x_prop = G_x@x+g_c

        # create cost
        Q = G_u.T@Qx@G_u + Ru
        if 'x_ref' in ingredients:
            x_ref = ca.DM(ingredients['x_ref'])
            q = G_u.T @ Qx @ (x_prop - x_ref)
        else:
            q = G_u.T @ Qx @ x_prop
        if 'u_ref' in ingredients:
            u_ref = ca.DM(ingredients['u_ref'])
            q = q - Ru@u_ref

        # create inequality constraint matrices
        G = ca.vertcat(Hx@G_u,Hu)
        uba = ca.vertcat(hx-Hx@x_prop,hu)
        lba = -ca.inf*ca.DM.ones(uba.shape)

        # initialize list of symbolic outputs
        out_list_symbolic = [Q,q,G,lba,uba,G_u,G_x,g_c]
        out_list_symbolic_names = ['Q','q','G','lba','uba','G_u','G_x','g_c']

        # re-create function
        start = time.time()
        QP_func = ca.Function('QP_dense',[self.param['p_qp_full']],out_list_symbolic,['p'],out_list_symbolic_names,options)
        comp_time_dict = {'QP_dense':time.time()-start}

        # implement QP using conic interface to retrieve multipliers
        qp = {}
        qp['h'] = Q.sparsity()
        qp['a'] = G.sparsity()

        # vector specifying which constraints are equalities
        is_equality = [False]*self.dim['in']

        # add existing options to compile function S
        start = time.time()
        match self._options['solver']:
            case 'gurobi':
                S = ca.conic('S','gurobi',qp,{'gurobi':{'OutputFlag':0},'equality':is_equality})
            case 'cplex':
                S = ca.conic('S','cplex',qp,{'cplex':{'CPXPARAM_Simplex_Display': 0,'CPXPARAM_ScreenOutput': 0},'equality':is_equality})
            case 'qpoases':
                S = ca.conic('S','qpoases',qp,options|{'printLevel':'none','equality':is_equality})
            case 'osqp':
                S = ca.conic('S','osqp',qp,options|{'osqp':{'verbose':False},'equality':is_equality})
            case 'daqp':
                S = ca.conic('S','daqp',qp,options)
            case 'qrqp':
                S = ca.conic('S','qrqp',qp,options|{'print_iter':False,'equality':is_equality})
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
            return ca.DM(self.dim['in'],1),ca.DM(self.dim['eq'],1),ca.vertcat(x,u,ca.DM(self.dim['eps'],1))
        
        # save in model
        self._dense_solve = local_qp

        # store computation times (if compile is true)
        self._comp_times = self._comp_times | comp_time_dict

    def _makeConsJac(self):
        """
        Constructs and compiles the conservative Jacobian functions for the QP constraints and primal variables.
        This method generates symbolic Jacobians using CasADi for the constraint projector and the primal variable
        with respect to the QP parameters and dual variables. It supports both NumPy and CasADi-based least-squares solvers
        for evaluating the Jacobians. The resulting functions are stored as attributes for later use.
        Steps performed:
            - Configures compilation options for CasADi functions based on user settings.
            - Extracts relevant parameters, dual variables, and QP data from class attributes.
            - Computes conservative Jacobians for the constraint projector and the primal variable.
            - Applies common subexpression elimination and sparsification to optimize symbolic expressions.
            - Stacks outputs and creates a CasADi function for efficient evaluation.
            - Defines a function to compute the conservative Jacobian of the primal variable with respect to parameters,
              using either NumPy or CasADi solvers as specified.
            - Stores the generated functions and computation times in class attributes.
        Attributes Set:
            self._J: CasADi Function for evaluating Jacobians.
            self._J_y_p: Function for computing the conservative Jacobian of the primal variable.
            self._compTimes: Dictionary updated with computation times for Jacobian generation.
        Raises:
            None
        Notes:
            - This method assumes that all required parameters and data structures are already initialized in the class.
            - The method is intended for internal use within the QP solver class.
        """

        # compilation options
        if self._options['compile_jac']:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            options = {}

        # get parameters that will be differentiated
        p = self.param['p_qp']

        # get all parameters including those that won't be differentiated
        p_full = self.param['p_qp_full']

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
        H = self.ingredients.dual['H']
        h = self.ingredients.dual['h']

        # extract QP data
        Qinv = self.ingredients.sparse['Qinv']
        q = self.ingredients.sparse['q']
        F = self.ingredients.sparse['F']
        G = self.ingredients.sparse['G']

        # compute conservative jacobian of projector
        J_Pc = ca.diag(ca.vertcat(ca.vec(ca.sign(lam)),ca.SX.ones(n_eq,1)))
        J_F_z = J_Pc@(ca.SX.eye(n_z)-self._options['jac_gamma']*H)-ca.SX.eye(n_z)
        J_F_p = -self._options['jac_gamma']*J_Pc@( ca.jacobian(H@ca.vertcat(lam,mu)+h,p) )

        # compute conservative jacobian of primal variable
        y = -Qinv@(G.T@lam+F.T@mu+q)
        J_y_p = ca.jacobian(y,p)
        J_y_z_mat = -Qinv@ca.horzcat(G.T,F.T)

        # sparsify
        J_Pc = ca.cse(ca.sparsify(J_Pc))
        J_F_z = ca.cse(ca.sparsify(J_F_z))
        J_F_p = ca.cse(ca.sparsify(J_F_p))
        y = ca.cse(ca.sparsify(y))
        J_y_p = ca.cse(ca.sparsify(J_y_p))
        J_y_z_mat = ca.cse(ca.sparsify(J_y_z_mat))

        # stack all parameters
        dual_outs = [J_F_z,J_F_p,J_y_p,J_y_z_mat]
        dual_outs_names = ['J_F_z','J_F_p','J_y_p','J_y_z_mat']

        # start counting time
        start = time.time()

        # turn into function
        J = ca.Function('J',dual_params,dual_outs,dual_params_names,dual_outs_names,options)

        # stop counting time
        comp_time_dict = {'J':time.time()-start}

        # function to generate jacobians
        if self._options['ls_solver'] == 'numpy':

            def J_y_p(lam,mu,p_qp,t=1):

                # round lambda (always first entry in dual_params) to avoid numerical issues
                lam = np.round(np.array(ca.fmax(lam,0)),self._options['jac_tol'])
                
                # get all conservative jacobian and matrices
                J_F_z,J_F_p,J_y_p,J_y_z_mat = J(lam,mu,p_qp)

                # get conservative jacobian of dual solution
                A = -lstsq(np.array(J_F_z),np.array(J_F_p@t))[0]

                # return conservative jacobian of primal
                return J_y_p@t+J_y_z_mat@A
        
        elif self._options['ls_solver'] == 'casadi':

                def J_y_p(lam,mu,p_qp,t=1):

                    # round lambda (always first entry in dual_params) to avoid numerical issues
                    lam = np.round(np.array(ca.fmax(lam,0)),self._options['jac_tol'])
                    
                    # get all conservative jacobian and matrices
                    J_F_z,J_F_p,J_y_p,J_y_z_mat = J(lam,mu,p_qp)

                    # get conservative jacobian of dual solution
                    A = -ca.solve(J_F_z,J_F_p@t,'csparse')

                    # return conservative jacobian of primal
                    return J_y_p@t+J_y_z_mat@A

        # save in QP
        self._j = J
        self._j_y_p = J_y_p

        # store computation times
        self._comp_times = self._comp_times | comp_time_dict

    @property
    def ingredients(self):
        return self._ingredients

    @property
    def idx(self):
        return self._ingredients.idx

    @property
    def solve(self):
        return self._solve

    @property
    def dense_solve(self):
        return self._dense_solve

    @property
    def qp_sparse(self):
        return self._qp_sparse

    @property
    def qp_dense(self):
        return self._qp_dense

    @property
    def dual_sparse(self):
        return self._dual_sparse

    @property
    def j(self):
        return self._j

    @property
    def j_y_p(self):
        return self._j_y_p

    @property
    def param(self):
        return self._sym.var
    
    @property
    def dim(self):
        return self._sym.dim
    
    @property
    def options(self):
        return self._options
    
    @property
    def init(self):
        return {key:val for key,val in self._sym.init.items() if val is not None}
    
    def _set_init(self, data):
        # TODO: improve the set_init function
        self._sym.set_init(data)

    def debug(self,lam,mu,p_t,epsilon=1e-6,roundoff=10,y_all=None):
        #TODO: update this

        # get full derivative
        J_QP = QP.J_y_p(lam,mu,p_t)
        
        # prepare vector containing derivative
        J_num = ca.DM(*J_QP.shape)
        
        # check against numerical differences
        j = 0
        for v in np.eye(p_t.shape[0]):
        
            # compute y_all for small perturbations around p_t
            _,_,y_all_forward = QP.solve(p_t+ca.DM(v*epsilon),y_all,lam,mu)
            _,_,y_all_backward = QP.solve(p_t-ca.DM(v*epsilon),y_all,lam,mu)
            
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