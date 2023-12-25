from casadi import *
from myDiffQP.MPC2QP import *

def setup_qp(QP_data,p,p_names,options):
    """
    This function takes a description of a QP in the form
    
    min. 0.5*y.T*Q*y+q.T*y  s.t. F*y=f, G*y<=g
    
    where the elements Q,q,F,f,G,g are parameterized with some parameter p,
    and returns a function that returns matrices that can be directly passed
    to CasADi's Conic interface to solve the QP.

    INPUTS ------------------------------------------------------------------------

    * QP_data: dict with entries
      - Q,q: symbolic cost of QP problem (cost is 0.5*y.T*Q*y+q.T*y)
      - Qinv: inverse of Q
      - F,f: symbolic equality constraints of QP problem (constraints are F*y=f)
      - G,g: symbolic inequality constraints of QP problem (constraints are G*y<=g)
    
    * p: list containing all parameters affecting the QP (one paramater per entry)
    
    * p_names: list containing the names of the entries in p
    
    * options: dict to be passed to all CasADi functions containing options (e.g. 
               jit compilation)

    OUTPUTS -----------------------------------------------------------------------

    * qp_data: dict containing information about the QP

    * qp / qp_dense: CasADi functions returning the matrices and vectors describing
                     the QP (qp_dense only returns those elements necessary to set
                     up the Conic interface)
    
    * dim: dict containing dimensions of the QP problem
    """

    # extract QP data
    Q = QP_data['Q']
    Qinv = QP_data['Qinv']
    q = QP_data['q']
    G = QP_data['G']
    g = QP_data['g']
    F = QP_data['F']
    f = QP_data['f']
    
    # stack all constraints together
    A = vertcat(G,F)
    uba = vertcat(g,f)
    lba = vertcat(-inf*SX.ones(g.shape),f)
    
    # stack all parameters
    qp_outs = [Q,Qinv,q,G,g,F,f,lba,uba,A]
    qp_outs_names = ['Q','Qinv','q','G','g','F','f','lba','uba','A']
    
    # stack all parameters, only dense form
    qp_outs_dense = [Q,q,lba,uba,A]
    qp_outs_dense_names = ['H','g','lba','uba','A']
    
    # store QP data
    qp_data = {'Q':Q,'Qinv':Qinv,'q':q,'G':G,'g':g,'F':F,'f':f,'lba':lba,'uba':uba,'A':A}
    
    # store qp function
    qp = Function('QP',p,qp_outs,p_names,qp_outs_names,options)
    qp_dense = Function('QP_dense',[vcat(p)],qp_outs_dense,['p'],qp_outs_dense_names,options)

    # setup dimensions dictionary using dimensions of QP matrices
    dim = {'n_y':qp_data['Q'].shape[0],   # number of primal variables
           'n_eq':qp_data['F'].shape[0],  # number of equality constraints
           'n_in':qp_data['G'].shape[0]}  # number of inequality constraints

    # number of dual variables
    dim['n_z'] = dim['n_eq'] + dim['n_in']

    # return
    return qp_data,qp,qp_dense,dim

def setup_qp_solver(qp_data,qp_dense,options,dim):
    """
    This function returns a function that can be used to solve the QP problem

    min. 0.5*y.T*Q*y+q.T*y  s.t. F*y=f, G*y<=g      (1)

    The problem data (i.e., the matrices Q,F,G and the vectors q,f,g) are obtained
    by calling the function qp_dense. The solver uses CasADi's Conic interface. 
    Options can be passed to Conic through the input "options".

    Note that Conic requires problem (1) to be written as

    min. 0.5*y.T*Q*y+q.T*y  s.t. lba <= A*y <= uba,

    where A = [F G].T, lba = [-f -inf].T, uba = [f 0].T

    INPUTS ------------------------------------------------------------------------

    * qp_data: dict with keys
      - Q / q / G / g / F / f: symbolic elements in the primal QP
      - Qinv: inverse of Q
      - lba / uba: lower and upper bounds on constraints
      - A: matrix containing both F and G to setup constraints in Conic

    OUTPUTS -----------------------------------------------------------------------

    * qp solver (see function local_qp below)

    """
    
    # don't print when solving
    solv_options = {'printLevel':'none'}
    
    # implement QP using conic interface to retrieve multipliers
    qp = {}
    qp['h'] = qp_data['Q'].sparsity()
    qp['a'] = qp_data['A'].sparsity()

    # add existing options to compile function S
    S = conic('S','qpoases',qp,{**options, **solv_options})
    
    # get dimensions
    n_y = dim['n_y']        # primal variable
    n_eps = dim['n_eps']    # number of slack variables in y
    n_in = dim['n_in']      # number of inequality constraints
    n_eq = dim['n_eq']      # number of equality constraints
    n_y_no_eps = n_y-n_eps  # number of primal variables that are not slack

    # create local function setting up the qp
    def local_qp(p,x0=DM.zeros(n_y),lam0=DM.zeros(n_in),mu0=DM.zeros(n_eq)):
      """
      This function solves the QP problem

      min. 0.5*y.T*Q*y+q.T*y  s.t. F*y=f, G*y<=g
    
      where the elements Q,q,F,f,G,g are parameterized with some parameter p.
      To obtain the matrices in the QP, this function uses the qp_dense function.

      INPUTS ------------------------------------------------------------------------

      * p: paramater vector

      * x0 / lam0 / mu0: warm start for primal / dual variables (lam=inequality,
                         mu=equality)

      OUTPUTS -----------------------------------------------------------------------

      * [1] => lambda (multiplier of inequality constraints)

      * [2] => mu (multiplier of equality constraints)

      * [3] => y (primal variable without slack variables)

      * [4] => epsilon (slack variables)
      """

      # get data from qp_dense function
      QP_data = qp_dense(p=p)

      # rearrange initial guesses, since we are in an MPC setting, the initial guess is
      # usually the past MPC solution (primal and dual). Therefore, we need to shift
      # the initial guess by one entry and repeat the final entry twice.
      # x0 = vertcat(x0[1:],x0[-1])
      # lam0 = vertcat(lam0[1:],lam0[-1])
      # mu0 = vertcat(mu0[1:],mu0[-1])

      # solve QP with warmstarting
      try:
        sol = S(h=QP_data['H'],a=QP_data['A'],g=QP_data['g'],
                lba=QP_data['lba'],uba=QP_data['uba'],
                x0=x0,lam_a0=vertcat(mu0,lam0))
      
      # if does not work, try without warmstarting
      except:
        sol = S(h=QP_data['H'],a=QP_data['A'],g=QP_data['g'],
                lba=QP_data['lba'],uba=QP_data['uba'])
      
      # return lambda, mu, y, eps
      return sol['lam_a'][:dim['n_in']],sol['lam_a'][dim['n_in']:],sol['x'][:n_y_no_eps],sol['x'][n_y_no_eps:]
    
    # return
    return local_qp

def setup_dual(qp_data):
    """
    This function computes the symbolic representation of the dual problem
    given the primal problem (both problems are QPs). Specifically, let
    the primal be

    min. 0.5*y.T*Q*y+q.T*y  s.t. F*y=f, G*y<=g

    then the dual is

    min. 0.5*z.T*H*z + h.T*z  s.t. z=(lam,mu), lam>=0

    This function returns a dictionary containing H and h.

    INPUTS ------------------------------------------------------------------------

    * qp_data: dict with keys
      - Q / q / G / g / F / f: symbolic elements in the primal QP
      - Qinv: inverse of Q

    OUTPUTS -----------------------------------------------------------------------

    * dict with entries H and h (symbolic elements of dual QP)
    """

    # extract QP data. Run sparsify to ensure sparse structure, and cse to reuse
    # common sub-expressions (to speed up computations).
    G = cse(sparsify(qp_data['G']))
    g = cse(sparsify(qp_data['g']))
    F = cse(sparsify(qp_data['F']))
    f = cse(sparsify(qp_data['f']))
    Qinv = cse(sparsify(qp_data['Qinv']))
    q = cse(sparsify(qp_data['q']))

    # define Hessian of dual
    H_11 = cse(sparsify(G@Qinv@G.T))
    H_12 = cse(sparsify(G@Qinv@F.T))
    H_21 = cse(sparsify(F@Qinv@G.T))
    H_22 = cse(sparsify(F@Qinv@F.T))
    H = cse(blockcat(H_11,H_12,H_21,H_22))

    # define linear term of dual
    h_1 = cse(sparsify(G@Qinv@q+g))
    h_2 = cse(sparsify(F@Qinv@q+f))
    h = cse(vertcat(h_1,h_2))

    # return
    return {'H':H,'h':h}

def setup_dual_solver(H,h,p,n_in,n_eq,options):
    """
    This function sets up a CasADi Conic interface that solves the dual QP problem

    min. 0.5*z.T*H*z + h.T*z  s.t. z=(lam,mu), lam>=0

    where H and h depend on a parameter p.

    INPUTS ------------------------------------------------------------------------

    * H / h: matrices describing the dual

    * p: parameter vector

    * n_in, n_eq: dimensions of lambda and mu respectively

    * options: options passed to Conic (for jit compilation)    

    OUTPUTS -----------------------------------------------------------------------

    * dual qp solver function
    """
    # solver options
    solv_options = {'printLevel':'none'}
    
    # create functions returning numerical value of H and h
    H_func = Function('H_func',[vcat(p)],[H],options)
    h_func = Function('h_func',[vcat(p)],[h],options)

    # implement dual qp using conic interface
    dual_qp = {}
    dual_qp['h'] = H.sparsity()
    dual_qp['a'] = H.sparsity()
    S = conic('S','qpoases',dual_qp,{**options, **solv_options})
    
    # create local function setting up the dual qp
    def local_dual_qp(p):
      
      # unpack H and h
      H = H_func(p)
      h = h_func(p)

      # set bounds
      lbx = vertcat(DM.zeros(n_in,1),-inf*DM.ones(n_eq,1))
      ubx = inf*DM.ones(n_in+n_eq,1)
      
      # solve QP
      sol = S(h=H,g=h,lbx=lbx,ubx=ubx)

      # return solution
      return sol['x']
    
    # return
    return local_dual_qp


class diffQP:
  """
  Class that provides interfaces to solve the primal and the dual QP problems, and to
  compute the conservative Jacobian of the primal solution.

  METHODS ------------------------------------------------------------------------

  * opt_solve: solve closed-loop problem optimally (potentially using a nonlinear solver)

  * vars: dictionary containing all the relevant symbolic variables, namely
    - p: parameter vector
    - lam: Lagrange multiplier associated to the inequality constraints
    - mu: Lagrange multiplier associated to the equality constraints
    - gamma: stepsize used in the fixed point condition of the dual problem
  
  * idx: TODO

  * dim: TODO

  * qp_data: see function setup_qp()

  * qp_dense: see function setup_qp()

  * qp: see function setup_qp()

  * solve: see function setup_qp_solver()

  * dual_data: see function setup_dual()

  * dual_solve: see function setup_dual_solver()

  * J: see method set_cons_jac()

  """
  
  # store jit options for all functions
  jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
  options = {"jit": True, "compiler": "shell", "jit_options": jit_options}

  def __init__(self,problem):
    
    # check problem type (either hard of soft state constraints)
    if problem['type'] == 'hard':
      # form QP (no slack)
      qp_data = MPC2QP_hard(problem)
    elif problem['type'] == 'slack':
      # form QP and optimal QP (slack)
      qp_data = MPC2QP_slack(problem)

    # optimal MPC
    self.opt_solve = optiMPC(problem)

    # extract parameters
    p = problem['param']['p']
    p_names = problem['param']['names']

    # save parameters in class instance
    self.vars = dict()
    self.vars['p'] = vcat(p)
    self.var_names = p_names

    # save indices
    self.idx = problem['param']['idx']

    # setup qp
    qp_data_full,qp,qp_dense,dim = setup_qp(qp_data,p,p_names,self.options)

    # save default variables
    self.vars['lam'] = SX.sym('lam',dim['n_in'],1)
    self.vars['mu'] = SX.sym('mu',dim['n_eq'],1)
    self.vars['gamma'] = SX.sym('gamma',1,1)

    # save dimensions
    self.dim = {**dim, **problem['dim']}

    # setup qp solver
    qp_solver = setup_qp_solver(qp_data_full,qp_dense,self.options,self.dim)

    # store all qp information
    self.qp_data = qp_data_full
    self.qp = qp
    self.qp_dense = qp_dense
    self.solve = qp_solver

    # setup dual
    self.dual_data = setup_dual(qp_data)
    
    # setup dual solver
    self.dual_solve = setup_dual_solver(self.dual_data['H'],self.dual_data['h'],p,dim['n_in'],dim['n_eq'],self.options)

    # setup conservative jacobian
    self.set_cons_jac()     

  def set_cons_jac(self):
    """
    This function defines the method "J" of diffQP, which is a function that computes multiple matrices.
    We consider a QP where y is the primal variable and z is the dual variable. The optimality conditions
    of the dual are described using the fixed point condition F(z,p)=0 where p is a parameter. 
    
    J is a CasADi function that returns

    * J_F_z: conservative Jacobian of the fixed point condition F(z,p)=0 wrt z

    * J_F_p: conservative Jacobian of the fixed point condition F(z,p)=0 wrt p

    * J_y_p: conservative Jacobian of y wrt p (where y is intended as a function of p and z)

    * J_y_z_mat: conservative Jacobian of y wrt z (where y is intended as a function of p and z)
    """

    # extract variables
    lam = self.vars['lam']
    mu = self.vars['mu']
    p = self.vars['p']
    gamma = self.vars['gamma']

    # extract dimensions
    n_z = self.dim['n_z']
    n_eq = self.dim['n_eq']

    # extract dual data
    H = self.dual_data['H']
    h = self.dual_data['h']

    # extract QP data
    Qinv = self.qp_data['Qinv']
    q = self.qp_data['q']
    F = self.qp_data['F']
    G = self.qp_data['G']

    # compute conservative jacobian of projector
    J_Pc = cse(diag(vertcat(vec(sign(lam)),SX.ones(n_eq,1))))
    J_F_z = cse(J_Pc@(SX.eye(n_z)-gamma*H)-SX.eye(n_z))
    J_F_p = cse( - gamma*J_Pc@( jacobian(H@vertcat(lam,mu)+h,p) ))

    # compute conservative jacobian of primal variable
    y = -Qinv@(G.T@lam+F.T@mu+q)
    J_y_p = cse(simplify(jacobian(y,p)))
    J_y_z_mat = cse(simplify(-Qinv@horzcat(G.T,F.T)))

    # stack all parameters
    dual_params = [lam,mu,gamma,p]
    dual_params_names = ['lam','mu','gamma','p']
    dual_outs = [J_F_z,J_F_p,J_y_p,J_y_z_mat]
    dual_outs_names = ['J_F_z','J_F_p','J_y_p','J_y_z_mat']

    # turn into function
    self.J = Function('J',dual_params,dual_outs,dual_params_names,dual_outs_names,self.options)

  def J_y(self,lam,mu,gamma,p,tol=12):
    """
    This function computes the conservative Jacobian of the primal optimizer y of a QP in the 
    form

    min. 0.5*y.T*Q*y+q.T*y  s.t. F*y=f, G*y<=g

    First, it calls the method "J" to obtain the conservative Jacobians of the fixed point
    condition F(z,p)=0 (of the dual problem), then it computes the conservative Jacobian of
    the dual optimizer z wrt variations of p by solving a linear system of equations, finally
    it returns the conservative Jacobian of y wrt variations of p.

    INPUTS ------------------------------------------------------------------------

    * lam: Lagrange multiplier of inequality constraints

    * mu: Lagrange multiplier of equality constraints

    * gamma: stepsize used in the fixed point condition F(z,p)=0

    * p: parameter vector

    * tol: tolerance when computing the sign of the inequality multipliers (default = 12)

    OUTPUTS -----------------------------------------------------------------------

    * conservative Jacobian of primal optimizer y wrt p

    """
    # get all conservative jacobian and matrices
    J = self.J(lam=np.round(np.array(fmax(lam,0)),tol),mu=mu,gamma=gamma,p=p)

    # get conservative jacobian of dual solution
    A = -solve(J['J_F_z'],J['J_F_p'],'csparse')

    # return conservative jacobian of primal
    return J['J_y_p']+J['J_y_z_mat']@A