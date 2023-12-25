from casadi import *

def optiMPC(p):
  
  # extract system dynamics
  f = p['sys']['f']

  # extract constraints
  x_max = p['sys']['x_max']
  x_min = p['sys']['x_min']
  u_max = p['sys']['u_max']
  u_min = p['sys']['u_min']

  # extract cost
  Q = DM(p['cost']['Qx'])
  R = DM(p['cost']['Ru'])

  # extract dimensions
  n_x = p['dim']['n_x']
  n_u = p['dim']['n_u']
  N_opt = p['dim']['N_opt']

  # create opti object
  opti = Opti()

  # define variables
  x = opti.variable(n_x,N_opt)
  u = opti.variable(n_u,N_opt-1)

  x0 = opti.parameter(n_x,1)

  # initialize cost
  cost = x[:,0].T@Q@x[:,0]

  # initial condition
  opti.subject_to( x[:,0] == x0)

  # loop for constraints and dynamics
  for t in range(1,N_opt):
     
    # state and input constraints
    opti.subject_to( [x_min <= x[:,t], x[:,t] <= x_max] )
    opti.subject_to( [u_min <= u[:,t-1], u[:,t-1] <= u_max] )
     
    # dynamics
    opti.subject_to( x[:,t] == f(x[:,t-1],u[:,t-1]) )

    # cost
    cost = cost + x[:,t].T@Q@x[:,t] + u[:,t-1].T@R@u[:,t-1]

  # setup problem
  opti.minimize( cost )

  # solver
  opts = dict()
  opts["print_time"] = False
  opts['ipopt'] = {"print_level": 0, "sb":'yes'}
  opti.solver('ipopt',opts)

  # turn to function
  S_opt = opti.to_function('S_opt',[x0],[x.T,u.T,cost])

  # return solver
  return S_opt

def MPC2QP_slack(p):

  # extract symbolic variables
  x0 = p['param']['vars']['x0']
  y0 = p['param']['vars']['y0']
  Qn = p['param']['vars']['Qn']
  Qn_inv = p['param']['vars']['Qn_inv']

  # extract system dynamics
  fd = p['sys']['f']
  A = p['sys']['A']
  B = p['sys']['B']

  # extract constraints
  x_max = p['sys']['x_max']
  x_min = p['sys']['x_min']
  u_max = p['sys']['u_max']
  u_min = p['sys']['u_min']

  # extract cost
  Qx = p['cost']['Qx']
  Ru = p['cost']['Ru']
  c1 = p['cost']['c1']
  c2 = p['cost']['c2']

  # extract dimensions
  n_x = p['dim']['n_x']
  n_u = p['dim']['n_u']
  N = p['dim']['N']

  # preallocate equality constraint matrices
  F = SX.zeros(N*n_x,N*(n_x+n_u)+2*N*n_x)
  f = SX.zeros(N*n_x,1)

  # extract linearization input and state
  x = y0[:N*n_x]
  u = y0[N*n_x:]

  # construct matrix
  for i in range(N):
      
      # extract current linearization point
      x_i = x[i*n_x:(i+1)*n_x]
      u_i = u[i*n_u:(i+1)*n_u]
      
      # negative identity for next state
      F[i*n_x:(i+1)*n_x,i*n_x:(i+1)*n_x] = -SX.eye(n_x)
      
      # distinguish between i=0 for initial condition
      if i > 0:
          # A_i
          F[i*n_x:(i+1)*n_x,(i-1)*n_x:i*n_x] = A(x_i,u_i)
          # c_i
          f[i*n_x:(i+1)*n_x] = - ( fd(x_i,u_i) - A(x_i,u_i)@x_i - B(x_i,u_i)@u_i )
      else:
          # -A_0x0 - c_0
          f[i*n_x:(i+1)*n_x] = - ( fd(x0,u_i) - A(x0,u_i)@x0 - B(x0,u_i)@u_i ) - A(x_i,u_i)@x0
      
      # B_i
      F[i*n_x:(i+1)*n_x,N*n_x+i*n_u:N*n_x+(i+1)*n_u] = B(x_i,u_i)

  # sparsify F and f
  F = cse(sparsify(F))
  f = cse(sparsify(f))

  # get number of slack variables
  n_eps = 2*n_x*N

  # preallocate inequality constraint matrices (state)
  Hx = kron(SX.eye(N),vertcat(SX.eye(n_x),-SX.eye(n_x)))
  Hx = horzcat(Hx,SX.zeros(Hx.shape[0],N*n_u),-SX.eye(n_eps))
  hx = repmat(vertcat(x_max,-x_min),N,1)

  # preallocate inequality constraint matrices (input)
  Hu = kron(SX.eye(N),vertcat(SX.eye(n_u),-SX.eye(n_u)))
  Hu = horzcat(SX.zeros(Hu.shape[0],N*n_x),Hu,SX.zeros(2*n_u*N,n_eps))
  hu = repmat(vertcat(u_max,-u_min),N,1)

  # inequality constraint matrix (slack)
  He = horzcat(SX.zeros(n_eps,N*(n_x+n_u)),-SX.eye(n_eps))
  he = SX.zeros(n_eps,1)

  # create inequality constraint matrices
  G = cse(sparsify(vertcat(Hx,Hu,He)))
  g = cse(sparsify(vertcat(hx,hu,he)))
  
  # cost: first allocate everything that is not parameterized
  Qinv = blockcat(kron(SX.eye(N),inv(Qx)),SX.zeros(N*n_x,N*n_u),SX.zeros(N*n_u,N*n_x),kron(SX.eye(N),inv(Ru)))
  Q = blockcat(kron(SX.eye(N),Qx),SX.zeros(N*n_x,N*n_u),SX.zeros(N*n_u,N*n_x),kron(SX.eye(N),Ru))

  # get number of optimization variables (not slack)
  n_y = Q.shape[0]
  
  # append cost applied to slack variable
  Q = blockcat(Q,SX.zeros(n_y,n_eps),SX.zeros(n_eps,n_y),c2*SX.eye(n_eps))
  Qinv = blockcat(Qinv,SX.zeros(n_y,n_eps),SX.zeros(n_eps,n_y),1/c2*SX.eye(n_eps))

  # create linear part of the cost
  q = vertcat(SX.zeros(n_y,1),c1*SX.ones(n_eps,1))

  # then change the parameteric part
  Qinv[(N-1)*n_x:N*n_x,(N-1)*n_x:N*n_x] = Qn_inv
  Q[(N-1)*n_x:N*n_x,(N-1)*n_x:N*n_x] = Qn

  # sparsify Q and q
  Q = cse(sparsify(Q))
  Qinv = cse(sparsify(Qinv))
  q = cse(sparsify(q))

  # return
  QP = dict()
  QP['Q'] = Q
  QP['Qinv'] = Qinv
  QP['q'] = q
  QP['G'] = G
  QP['g'] = g
  QP['F'] = F
  QP['f'] = f

  return QP

def MPC2QP_hard(p):

  # extract symbolic variables
  x0 = p['param']['vars']['x0']
  y0 = p['param']['vars']['y0']
  Qn = p['param']['vars']['Qn']
  Qn_inv = p['param']['vars']['Qn_inv']

  # extract system dynamics
  fd = p['sys']['f']
  A = p['sys']['A']
  B = p['sys']['B']

  # extract constraints
  x_max = p['sys']['x_max']
  x_min = p['sys']['x_min']
  u_max = p['sys']['u_max']
  u_min = p['sys']['u_min']

  # extract cost
  Qx = p['cost']['Qx']
  Ru = p['cost']['Ru']

  # extract dimensions
  n_x = p['dim']['n_x']
  n_u = p['dim']['n_u']
  N = p['dim']['N']

  # preallocate equality constraint matrices
  F = SX.zeros(N*n_x,N*(n_x+n_u))
  f = SX.zeros(N*n_x,1)

  # extract linearization input and state
  x = y0[:N*n_x]
  u = vertcat(y0[N*n_x+n_u:],y0[-1-n_u:-1])

  # construct matrix
  for i in range(N):
      
      # extract current linearization point
      x_i = x[i*n_x:(i+1)*n_x]
      u_i = u[i*n_u:(i+1)*n_u]
      
      # negative identity for next state
      F[i*n_x:(i+1)*n_x,i*n_x:(i+1)*n_x] = -SX.eye(n_x)
      
      # distinguish between i=0 for initial condition
      if i > 0:
          # A_i
          F[i*n_x:(i+1)*n_x,(i-1)*n_x:i*n_x] = A(x_i,u_i)
          # c_i
          f[i*n_x:(i+1)*n_x] = - ( fd(x_i,u_i) - A(x_i,u_i)@x_i - B(x_i,u_i)@u_i )
          # B_i
          F[i*n_x:(i+1)*n_x,N*n_x+i*n_u:N*n_x+(i+1)*n_u] = B(x_i,u_i)
      else:
          # -A_0x0 - c_0
          f[i*n_x:(i+1)*n_x] = - ( fd(x0,u_i) - A(x0,u_i)@x0 - B(x0,u_i)@u_i ) - A(x0,u_i)@x0
          # B_i
          F[i*n_x:(i+1)*n_x,N*n_x+i*n_u:N*n_x+(i+1)*n_u] = B(x0,u_i)
      


  # sparsify F and f
  F = cse(sparsify(F))
  f = cse(sparsify(f))

  # preallocate inequality constraint matrices (state)
  Hx = kron(SX.eye(N),vertcat(SX.eye(n_x),-SX.eye(n_x)))
  Hx = horzcat(Hx,SX.zeros(Hx.shape[0],N*n_u))
  hx = repmat(vertcat(x_max,-x_min),N,1)

  # preallocate inequality constraint matrices (input)
  Hu = kron(SX.eye(N),vertcat(SX.eye(n_u),-SX.eye(n_u)))
  Hu = horzcat(SX.zeros(Hu.shape[0],N*n_x),Hu)
  hu = repmat(vertcat(u_max,-u_min),N,1)

  # create inequality constraint matrices
  G = cse(sparsify(vertcat(Hx,Hu)))
  g = cse(sparsify(vertcat(hx,hu)))

  # cost: first allocate everything that is not parameterized
  Qinv = blockcat(kron(SX.eye(N),inv(Qx)),SX.zeros(N*n_x,N*n_u),SX.zeros(N*n_u,N*n_x),kron(SX.eye(N),inv(Ru)))
  Q = blockcat(kron(SX.eye(N),Qx),SX.zeros(N*n_x,N*n_u),SX.zeros(N*n_u,N*n_x),kron(SX.eye(N),Ru))
  q = SX.zeros(Q.shape[0],1)

  # then change the parameteric part
  Qinv[(N-1)*n_x:N*n_x,(N-1)*n_x:N*n_x] = Qn_inv
  Q[(N-1)*n_x:N*n_x,(N-1)*n_x:N*n_x] = Qn

  # sparsify Q
  Q = cse(sparsify(Q))
  Qinv = cse(sparsify(Qinv))

  # return
  QP = dict()
  QP['Q'] = Q
  QP['Qinv'] = Qinv
  QP['q'] = q
  QP['G'] = G
  QP['g'] = g
  QP['F'] = F
  QP['f'] = f

  return QP