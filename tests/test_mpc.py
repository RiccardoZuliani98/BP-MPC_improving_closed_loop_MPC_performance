def MPC():

    # define symbolic variables
    x = SX.sym('x0',1,1)
    u = SX.sym('u0',1,1)

    # Construct a CasADi function for the ODE right-hand side
    A = 1
    B = 1

    # initial condition
    x0 = 1

    # compute next state symbolically
    x_next = A*x + B*u

    # create model dynamics
    dyn = {'x':x, 'u':u, 'x_next':x_next, 'x_dot':SX(0), 'x0':x0}

    # create sceanario
    mod = scenario()
    mod.makeDynamics(dyn)

    # create constraints
    x_max = 1000
    x_min = -1000
    u_max = 1000
    u_min = -1000

    # horizon of MPC
    N = 2

    # create parameter
    p = SX.sym('p',12,1)
    pf = SX.sym('pf',12,1)

    # create parameters for MPC
    p_qp = SX.sym('p_qp',4,1)
    pf_qp = SX.sym('pf_qp',4,1)

    # create reference
    x_ref = vertsplit(pf_qp[:2])
    u_ref = vertsplit(pf_qp[2:])

    # MPC costs
    Qx = SX(1)
    Ru = SX(2)
    Qn = SX(1)

    # add to mpc dictionary
    mpc_cost = {'Qx':Qx, 'Qn':Qn, 'Ru':Ru, 'x_ref':x_ref, 'u_ref':u_ref}

    # turn bounds into polyhedral constraints
    Hx,hx,Hu,hu = utils.bound2poly(x_max,x_min,u_max,u_min)

    # add to mpc dictionary
    mpc_cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu}

    # create mpc
    mod.makeMPC(N=N,cost=mpc_cost,cst=mpc_cst,p=p_qp,pf=pf_qp,options={'slack':False})

    # create index that selects pf for QP
    def idx_pf(t):
        return np.hstack([t,t+1,6+t,7+t],dtype=int)

    # construct upper-level
    mod.makeUpperLevel(p=p,pf=pf,idx_p=idx_pf,idx_pf=idx_pf,T=5)

    # create state reference by sampling random inputs
    u_ref = DM.rand(6,1)
    x_ref = [x0]
    for t in range(6):
        x_ref.append( A*x_ref[-1] + B*u_ref[t] )
    x_ref = vcat(x_ref)

    # extract closed-loop variables for upper level
    params = mod.param
    x_cl = vec(params['x_cl'])
    u_cl = vec(params['u_cl'])

    # create upper-level cost
    cost,_,_ = utils.quadCostAndBounds(SX(1),SX(1),x_cl,u_cl,x_ref=x_ref[:-1],u_ref=u_ref[:-1])
    mod.setUpperLevelCost(cost)

    pf_init = vertcat(x_ref[1:],u_ref)

    # set initial value of pf
    mod.setInit({'pf':pf_init,'p':pf_init})

    # simulate
    S,_,_ = mod.simulate(options={'mode':'Simulate'})

    # check if closed-loop trajectory is close to reference
    if norm_2(S.x - x_ref[:-1]) > 1e-10 or norm_2(S.u - u_ref[:-1]) > 1e-10:
        raise Exception('Closed-loop trajectory does not match reference.')
    
    # repeat with p instead of pf