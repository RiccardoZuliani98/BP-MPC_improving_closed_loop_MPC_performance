import casadi as ca
import time
from BPMPC.Ingredients import Ingredients
from BPMPC.options import Options
import numpy as np
from copy import copy

"""
TODO
* better init function?
* descriptions
* comp times
"""

class QP:
    
    # Allowed option keys
    _OPTIONS_ALLOWED_VALUES = {'solver':['qpoases','daqp'],'dense_solver':['qpoases','daqp'],
                               'warmstart':['x_lam_mu','x'],'jac_tol':int,'jac_gamma':float,
                               'compile_qp_sparse':bool,'compile_qp_dense':bool,'compile_jac':bool}

    # default values of options dictionary
    _OPTIONS_DEFAULT_VALUES = {'solver':'qpoases','dense_solver':'qpoases',
                               'warmstart':'x_lam_mu','jac_tol':8,'jac_gamma':0.001,
                               'compile_qp_sparse':False,'compile_qp_dense':False,'compile_jac':False}

    def __init__(self,ingredients,p=None,pf=None,options=None):

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
            self._sym.addVar('p_t',p)

        # TODO: check if order of variables in p_QP is correct

        # construct the parameter vector required for the QP
        p_QP = list(self._sym.var.values())
        p_QP_names = list(self._sym.var.keys())

        # save full symbolic qp parameter (do not include pf even if present)
        # self._sym.addDim('p_qp',ca.vcat(p_QP).shape[0])
        self._sym.addVar('p_qp',ca.vcat(p_QP))

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
            self._sym.addVar('pf_t',pf)
            p_QP.append(pf)
        else:
            # add dimension
            self._sym.addDim('pf_t',0)

        # store full qp parameter
        self._sym.addVar('p_qp_full',ca.vcat(p_QP))

        # store dimensions of equality and inequality constraints
        self._sym.addDim('in',ingredients.sparse['G'].shape[0])
        self._sym.addDim('eq',ingredients.sparse['F'].shape[0])

        # primal optimization variables
        self._sym.addVar('y',ca.SX.sym('y',ingredients.sparse['q'].shape[0]-self._sym.dim['eps'],1))

        # dual optimization variables (inequality constraints)
        self._sym.addVar('lam',ca.SX.sym('lam',ingredients.sparse['g'].shape[0],1))

        # dual optimization variables (equality constraints)
        self._sym.addVar('mu',ca.SX.sym('mu',ingredients.sparse['f'].shape[0],1))

        # dual optimization variable (all constraints)
        self._sym.addVar('z',ca.vcat([self._sym.var['lam'],self._sym.var['mu']]))

        # create sparse QP
        self._makeSparseQP()

        # create conservative jacobian
        self._makeConsJac()

    def _makeSparseQP(self):

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
        QP_outs_names = ['A','lba','uba','Q','q']

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
            case 'osqp':
                S = ca.conic('S','osqp',qp,options | {'osqp':{'verbose':False},'equality':is_equality})
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

                    # get data from qp_dense function
                    a,lba,uba,h,g = QP_func(p_qp)

                    # solve QP with warmstarting
                    if x0 is not None:
                        sol = S(h=h,a=a,g=g,lba=lba,uba=uba,x0=x0,lam_a0=ca.vertcat(lam0,mu0))
                    
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
        
        # save in model
        self._solve = local_qp

        # store computation times (if compile is true)
        self._compTimes = comp_time_dict

    def make_dense_qp(self,p):

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
        self._compTimes['QP_dense_func'] = time.time()-start

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
        self._compTimes = self._compTimes | comp_time_dict

    def _makeConsJac(self):

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

        # turn into function
        start = time.time()
        J = ca.Function('J',dual_params,dual_outs,dual_params_names,dual_outs_names,options)
        comp_time_dict = {'J':time.time()-start}

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
        self._J = J
        self._J_y_p = J_y_p

        # store computation times
        self._compTimes = self._compTimes | comp_time_dict

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
    def denseSolve(self):
        return self._denseSolve

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
    def J(self):
        return self._J

    @property
    def J_y_p(self):
        return self._J_y_p

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
        self._sym.set_init(data)

    def debug(self,lam,mu,p_t,epsilon=1e-6,roundoff=10,y_all=None):

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