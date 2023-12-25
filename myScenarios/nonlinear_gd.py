from casadi import *
from myDiffQP import *

def scenario():

    # choose type of problem
    problem = {'type': 'hard', 'penalize_violations': False,
               'algorithm':'gd', 'name':'nonlinear_gd'}

    ### PART 1: problem definition --------------------------------------------------
    
    # define dimensions of the problem
    n_x = 2
    n_u = 1
    n_p = 3
    N = 3
    N_opt = 75

    # initial condition
    x0 = SX.sym('x0',n_x,1)

    # linearization trajectory
    y0 = SX.sym('y',N*(n_x+n_u),1)

    # input for symbolic functions
    u0 = SX.sym('u0',n_u,1)

    # parameterized terminal cost (state dependent)
    c = SX.sym('c',n_p,1)
    Qn = SX.zeros(n_x,n_x)
    Qn[0,0] = c[0]
    Qn[0,1] = c[1]
    Qn[1,0] = c[1]
    Qn[1,1] = c[2]
    Qn = Qn.T@Qn

    # inverse of terminal cost
    det_Qn = Qn[0,0]*Qn[1,1]-Qn[0,1]*Qn[1,0]
    adj_Qn = vertcat(horzcat(Qn[1,1],-Qn[0,1]),horzcat(-Qn[1,0],Qn[0,0]))
    Qn_inv = adj_Qn/det_Qn

    # define dynamics (nonlinear)
    x1 = x0[0]
    x2 = x0[1]
    x1_next = x1+0.4*x2
    x2_next = (0.56+0.1*x1)*x2 + 0.4*u0 + 0.9*x1*exp(-x1)
    x_next = vertcat(x1_next,x2_next)

    # create function
    f = Function('f',[x0,u0],[x_next])

    # compute linearized dynamics
    A = Function('A',[x0,u0],[jacobian(x_next,x0)])
    B = Function('A',[x0,u0],[jacobian(x_next,u0)])

    # penalty on state and input
    Qx = SX.eye(n_x)
    Ru = 0.0001
    
    # define box constraints
    x_max = vertcat(10,5)
    x_min = vertcat(-2,-5)
    u_max = 2
    u_min = -2

    # create initial conditions and parameters
    x0_stack = [vertcat(8,0)]
    p0_stack = [vertcat(0.1,0,0.1)]
    y0_stack = [vertcat(repmat(vertcat(10,0),N,1),DM.zeros(N*n_u,1))]

    ### PART 2: pack and export -----------------------------------------------------

    # system properties
    sys = dict()
    sys['A'] = A
    sys['B'] = B
    sys['f'] = f
    sys['x_max'] = x_max
    sys['x_min'] = x_min
    sys['u_max'] = u_max
    sys['u_min'] = u_min

    # algorithm options
    alg = dict()
    alg['max_k'] = 200
    alg['max_it'] = N_opt
    alg['rho'] = 0.2
    alg['eta'] = 1
    alg['alpha'] = lambda k,J : 0.2*log(k+1)/(k+1)

    # cost options
    cost = dict()
    cost['Qx'] = Qx
    cost['Ru'] = Ru

    # dimensions
    dim = dict()
    dim['n_x'] = n_x
    dim['n_u'] = n_u
    dim['n_p'] = n_p
    dim['n_eps'] = 0
    dim['N'] = N
    dim['N_opt'] = N_opt

    # stack all parameters
    param = dict()
    param['vars'] = {'x0':x0,'y0':y0,'Qn':Qn,'Qn_inv':Qn_inv}
    param['p'] = [x0,y0,c]
    param['names'] = ['x0','y0','c']

    # store input variable indices
    var_in = dict()
    var_in['x0'] = range(0,n_x)
    var_in['y0'] = range(n_x,n_x+N*(n_x+n_u))
    var_in['p'] = range(n_x+N*(n_x+n_u),n_x+N*(n_x+n_u)+n_p)

    # store output variable indices
    var_out = dict()
    var_out['u0'] = range(n_x*N,n_x*N+n_u)
    var_out['y'] = range(0,(n_x+n_u)*N)

    # pack all indices
    idx = {'in':var_in, 'out':var_out}
    param['idx'] = idx

    # stack initial conditions
    init_stack = dict()
    init_stack['x0'] = x0_stack
    init_stack['p0'] = p0_stack
    init_stack['y0'] = y0_stack

    # plotting options
    plot = dict()
    plot['iter_len'] = 200
    plot['iter_step'] = 1
    plot['time_len'] = 34
    plot['time_step'] = 1
    plot['plot_constraints'] = False
    plot['x_range'] = [0,35]

    # stack together
    problem['sys'] = sys
    problem['alg'] = alg
    problem['cost'] = cost
    problem['dim'] = dim
    problem['param'] = param
    problem['init'] = init_stack
    problem['plot'] = plot

    # return
    return problem