from casadi import *
from datetime import datetime
import pickle
import time
import os

def closedLoop(QP,problem,verbose=1):

  # get type of simulation
  type = problem['type']
  penalize_violations = problem['penalize_violations']

  # extract initial conditions
  x0 = problem['init']['x0'][0]
  y0 = problem['init']['y0'][0]
  p = problem['init']['p0'][0]

  # extract algorithm options
  alg = problem['alg']

  # list all options and their default value
  list_of_options = ['max_it', 'max_k', 'tol', 'grad_tol', 'reg', 'gamma']
  default_values = [50, 500, 1e-5, 1e-6, 0.05, 0.2]

  # check if some options were not passed, if so, use default value
  for option, default_value in zip(list_of_options, default_values):
     if option not in alg:
        alg[option] = default_value

  # extract iterations
  max_k = alg['max_k']
  max_it = alg['max_it']

  # extract tolerances
  tol = alg['tol']
  grad_tol = alg['grad_tol']

  # stepsize in conservative Jacobian
  gamma = alg['gamma']

  # algorithm type
  algorithm = problem['algorithm']

  # stepsize in algorithm (if gd)
  if algorithm == 'gd':
    alpha = alg['alpha']
    eta = alg['eta']

  # regularization in Gauss-Newton update
  rho = alg['rho']

  # extract dimensions
  n_x = problem['dim']['n_x']
  n_u = problem['dim']['n_u']
  n_p = problem['dim']['n_p']
  N = problem['dim']['N']
  n_y = QP.dim['n_y']
  n_eps = problem['dim']['n_eps']
  n_eps_tot = max_it*n_eps

  # extract dynamics and linearization
  A = problem['sys']['A']
  B = problem['sys']['B']
  f = problem['sys']['f']

  # extract cost
  Ru = problem['cost']['Ru']

  # start empty list
  SIM = []

  # list for closed-loop cost
  cost = []

  # get optimal solution with qp_opt
  opt_sol = QP.opt_solve(x0)
  opt_cost = opt_sol[2]
  
  # print best cost
  if verbose > 0:
    print(f'Best achievable cost: {opt_cost}')

  # start counting time
  start = time.time()

  # outer loop
  for k in range(max_k):

      # create simVar for current simulation
      S = simVar(n_x,n_u,n_p,max_it,N)

      # initial state
      x = x0
      S.setState(0,x)

      # initial linearization trajectory
      y = y0

      # initialize Jacobians
      J_x_p = DM.zeros(n_x,n_p)
      J_y_p = DM.zeros(n_y-n_eps,n_p)
      S.setJx(0,J_x_p)

      # set initial guess for warm start
      mu = DM.ones(QP.dim['n_eq'])
      lam = DM.ones(QP.dim['n_in'])

      # violation (sum)
      violation = 0

      # simulation loop
      for t in range(max_it):
          
          # replace first entry of state with current state
          y_lin = y

          # solve QP
          try: 
            lam,mu,y,e = QP.solve(vertcat(x,y,p),y,lam,mu)
          except:
             print('QP solver failed!')
        
          # store violation
          violation = violation + sum1(fabs(e))
          
          # get inputs
          u = y[QP.idx['out']['u0']]

          # store input
          S.setInput(t,u)
          
          # get conservative jacobian of optimization variable
          J_y = QP.J_y(lam,mu,gamma,vertcat(x,y_lin,p))
          
          # get conservative jacobian of u0 wrt all parameters
          J_u0_x0 = J_y[QP.idx['out']['u0'],QP.idx['in']['x0']]
          J_u0_y0 = J_y[QP.idx['out']['u0'],QP.idx['in']['y0']]
          J_u0_p = J_y[QP.idx['out']['u0'],QP.idx['in']['p']]

          # get conservative jacobian of u1 wrt all parameters
          J_y0_x0 = J_y[:,QP.idx['in']['x0']]
          J_y0_y0 = J_y[:,QP.idx['in']['y0']]
          par_y0_p = J_y[:,QP.idx['in']['p']]
          
          # get conservative jacobian of u1 wrt parameter
          J_y_p_temp = J_y0_x0@J_x_p + J_y0_y0@J_y_p + par_y0_p

          # get conservative jacobian of next state wrt parameter
          J_x_p = A(x,u)@J_x_p + B(x,u)@( J_u0_x0@J_x_p + J_u0_y0@J_y_p + J_u0_p )

          # replace temporary variable
          J_y_p = J_y_p_temp[QP.idx['out']['y'],:]

          # if problem has slack variables, get jacobian of slack and store it
          if type == 'slack':
            J_eps_p = J_y_p_temp[QP.idx['out']['eps'],:]
            S.setJeps(t,J_eps_p)
          
          # store conservative jacobian
          S.setJx(t+1,J_x_p)
          
          # store optimization variables
          S.addOptVar(mu,lam,y,J_y)

          # get next state
          x = f(x,u)

          # store next state
          S.setState(t+1,x)
          
          # if the system is very close to the origin, end simulation
          if norm_2(x) < tol:
              break

      # compute gradient of closed-loop objective function
      if not penalize_violations:
        J_xcl_p = (S.X.T@S.Jx).T
      else:
        J_xcl_p = (S.X.T@S.Jx).T + 100*(DM.ones(1,n_eps_tot)@S.Jeps).T
      
      # update parameter
      match algorithm:
        case 'gd':
          p = p - alpha(k,J_xcl_p)*J_xcl_p
        case 'gn':
          p = p - inv(J_xcl_p@J_xcl_p.T+rho*(k+1)*DM.eye(n_p))@J_xcl_p
      
      # printout
      match verbose:
          case 0:
            pass
          case 1:
            print(f"Iteration: {k}, cost: {S.X.T@S.X}, J: {norm_2(J_xcl_p)}, e : {violation}")
          case 2:
            print(f"Iteration: {k}, cost: {S.X.T@S.X}, p: {p}, J: {J_xcl_p}")

      # # check value of the gradient, if too small, break
      # if norm_2(J_xcl_p) < grad_tol:
      #     break
      
      # store cost
      cost.append(S.X.T@S.X + Ru*S.U.T@S.U)

      # store S into list
      SIM.append(S)

  # get computation time
  end = time.time()
  comp_time = (end-start)/(k+1)

  # return
  return SIM, cost, p, opt_cost, opt_sol, comp_time

def pack(simOut,problem):
  
  # get time and date
  time_str = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")

  # create simulations directory
  if not os.path.exists('Simulations'): 
    os.makedirs('Simulations') 

  # file name
  file_name = 'Simulations/' + problem['name'] + '.pkl'

  # output dictionary
  data = dict()

  # save tame of save
  data['time'] = time_str

  # extract simulation output
  SIM = simOut[0]

  # extract optimal output
  opt = simOut[4]
  data['x1_opt'] = opt[0][:,0]
  data['x2_opt'] = opt[0][:,1]
  data['u_opt'] = opt[1]

  # get state / input trajectory on first iteration
  data['X0'] = SIM[0].X
  data['U0'] = SIM[0].U

  # get state / input trajectory on last iteration
  data['XM'] = SIM[-1].X
  data['UM'] = SIM[-1].U

  # store cost and optimal cost
  data['cost'] = simOut[1]
  data['opt_cost'] = simOut[3]

  # store tuning parameters of update rule
  data['rho'] = problem['alg']['rho']
  if problem['algorithm'] == 'gd':
    data['eta'] = problem['alg']['eta']

  # store constraints
  cst = dict()
  cst['u_min'] = problem['sys']['u_min']
  cst['u_max'] = problem['sys']['u_max']
  cst['x_min'] = problem['sys']['x_min']
  cst['x_max'] = problem['sys']['x_max']
  data['cst'] = cst

  # get computation time
  data['comp_time'] = simOut[5]

  # get final parameter
  data['p_final'] = simOut[2]

  # save using pickle
  with open(file_name, 'wb') as fp:
    pickle.dump(data, fp)
    print('Dictionary saved successfully to file')

class simVar:
  def __init__(self,nx,nu,np,max_it,N):
    self.nx = nx
    self.nu = nu
    self.np = np
    self.neps = 2*nx*N
    self.X = DM.zeros(nx*(max_it+1),1)
    self.U = DM.zeros(nu*max_it,1)
    self.Jx = DM.zeros(nx*(max_it+1),np)
    self.Jeps = DM.zeros(self.neps*(max_it),np)
    self.optVar_y = []
    self.optVar_mu = []
    self.optVar_lam = []
    self.optVar_J = []

  def setState(self,t,x):
    self.X[self.nx*t:self.nx*(t+1)] = x
  def getState(self,t):
    return self.X[self.nx*t:self.nx*(t+1)]
  def setInput(self,t,u):
    self.U[self.nu*t:self.nu*(t+1)] = u
  def getInput(self,t):
    return self.U[self.nu*t:self.nu*(t+1)]
  def setJx(self,t,J):
    self.Jx[self.nx*t:self.nx*(t+1),:] = J
  def setJeps(self,t,J):
    self.Jeps[self.neps*t:self.neps*(t+1),:] = J
  def getJx(self,t):
    return self.Jx[self.nx*t:self.nx*(t+1),:]
  def addOptVar(self,mu,lam,y,J):
    self.optVar_y.append(y)
    self.optVar_mu.append(mu)
    self.optVar_lam.append(lam)
    self.optVar_J.append(J)