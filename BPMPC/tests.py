from BPMPC.scenario import scenario
from casadi import *
from BPMPC import utils

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

def derivatives(scenario,p=None,pf=None,x=None,d=None,w=None,epsilon=1e-6,debug_qp=False):

    # # check if p is not passed
    # if p is None:
    #     if scenario.init['p'] is None:
    #         raise Exception('Parameter vector p is not defined.')
    #     p = scenario.init['p']

    # # create init dictionary
    # init = {'p':p}
    # if x0_in is not None:
    #     init['x'] = x0_in
    # if d_in is not None:
    #     init['d'] = d_in
    # if w_in is not None:
    #     init['w'] = w_in

    # create dictionary of initial conditions
    init = {'p':p,'pf':pf,'x':x,'d':d,'w':w}

    # remove entries that are None
    init = {k: v for k, v in init.items() if v is not None}

    # setup parameters
    p,pf,W,D,Y,X = scenario.getInitParameters(init)

    # extract cost function
    cost_f = scenario.upperLevel.cost

    # extract gradient of cost function
    J_cost_f = scenario.upperLevel.J_cost

    # main simulation loop
    if debug_qp:
        S,qp_data,_ = scenario._scenario__simulate(p,pf,W,D,Y,X,options={'debugQP':True,'epsilon':1e-8,'roundoff_QP':12})
    else:
        S,qp_data,_ = scenario._scenario__simulate(p,pf,W,D,Y,X)

    # get conservative jacobian
    J_x_p = S.Jx
    J_y_p = S.Jy
    # J_u_p = simOut.Ju
    # J_e_p = simOut.Jeps

    # compute closed-loop cost and closed-loop cost jacobian
    cost,track_cost,cst_viol = cost_f(S)
    J_cost = J_cost_f(S)

    # initialize derivatives
    DX = DM(*J_x_p.shape)
    DY = DM(*J_y_p.shape)
    Dcost = DM(*J_cost.shape)

    # check against finite differences
    k = 0
    for v in np.eye(p.shape[0]):

        # simulate with perturbed parameter
        S_p,_,_ = scenario._scenario__simulate(p + DM(v*epsilon),pf,W,D,Y,X, options={'mode':'Simulate'})

        # simulate with perturbed parameter
        S_p_2,_,_ = scenario._scenario__simulate(p - DM(v*epsilon),pf,W,D,Y,X, options={'mode':'Simulate'})

        # get closed-loop trajectory
        X_p = S_p.x
        Y_p = vec(S_p.y)
        # U_p = S_p.u
        # E_p = S_p.e
        X_p_2 = S_p_2.x
        Y_p_2 = vec(S_p_2.y)
        # U_p_2 = S_p_2.u
        # E_p_2 = S_p_2.e

        # get cost
        cost_p,_,_ = cost_f(S_p)
        cost_p_2,_,_ = cost_f(S_p_2)

        # compute numerical derivative
        dx_num = (X_p-X_p_2)/(2*epsilon)
        dy_num = (Y_p-Y_p_2)/(2*epsilon)
        # du_num = (U_p-U_p_2)/(2*epsilon)
        # de_num = (E_p-E_p_2)/(2*epsilon)
        dcost_num = (cost_p-cost_p_2)/(2*epsilon)
        
        # store result
        DX[:,k] = dx_num
        DY[:,k] = dy_num
        Dcost[k] = dcost_num

        # increase index
        k = k + 1

        # printout
        print(f'Done {k} out of {p.shape[0]}, current error (relative) {norm_2(dx_num - J_x_p@DM(v))/norm_2(dx_num)}, current error (absolute) {norm_2(dx_num - J_x_p@DM(v))}, true magnitude: {norm_2(dx_num)}')

    # print difference in cost
    print(f'Difference in cost: {Dcost-J_cost}')

    return {'dx':J_x_p, 'dx_num':DX, 'dy':J_y_p, 'dy_num':DY, 'd_cost_num':Dcost, 'd_cost':J_cost, 'qp_data':qp_data}

def initialization():

    # define symbolic variables
    x = SX.sym('x0',2,1)
    u = SX.sym('u0',1,1)

    # Construct a CasADi function for the ODE right-hand side
    A = 0.8
    B = vertcat(0.1,1)
    c = vertcat(0.01,0)
    x_dot = A*x + B*u + c

    # compute next state symbolically
    x_next = x_dot

    # create model dynamics
    dyn = {'x':x, 'u':u, 'x_next':x_next, 'x_dot':x_dot}

    # create sceanario
    mod = scenario()
    mod.makeDynamics(dyn)

    # initialization should contain None values in x and u
    if mod.init['x'] is not None or mod.init['u'] is not None:
        raise Exception('Initialization should contain None values in x and u.')

    # initializations
    x0 = vertcat(1,1)
    u0 = 1

    # add initial values of x and u
    dyn = dyn | {'x0':x0, 'u0':u0}

    # create scenario
    mod = scenario()
    mod.makeDynamics(dyn)

    # now initialization should contain values in x and u
    if sum1(fabs(mod.init['x']-x0)) or sum1(fabs(mod.init['u']-u0)):
        raise Exception('Initialization process failed.')

    # now try to add some variables that should not appear
    dyn = dyn | {'w0':1, 'd0':1, 'f':1}

    # create sceanario
    mod = scenario()
    mod.makeDynamics(dyn)

    # check that initialization is correct
    if 'f' in mod.init or mod.init['w'] is not None or mod.init['d'] is not None:
        raise Exception('Initialization should not contain values for w, d or f.')
    
    # now test setInit
    dyn = {'x':x, 'u':u, 'x_next':x_next, 'x_dot':x_dot}
    mod = scenario()
    mod.makeDynamics(dyn)

    # set initial conditions    
    mod.setInit({'x':x0, 'u':u0})    
    
    # now initialization should contain values in x and u
    if sum1(fabs(mod.init['x']-x0)) or sum1(fabs(mod.init['u']-u0)):
        raise Exception('Initialization process failed.')
    
    # now try to add some variables that should not appear
    mod.setInit({'f':1})

    # check that initialization is correct
    if 'f' in mod.init:
        raise Exception('Initialization should not contain w.')
    
    # now repeat with noisy model
    d = SX.sym('d',1,1)
    w = SX.sym('w',1,1)
    x_dot = A*x + B*u + c + vertcat(w,d)
    x_next = x_dot

    # create model dynamics
    dyn = {'x':x, 'u':u, 'w':w, 'd':d, 'x_next':x_next, 'x_dot':x_dot, 'w0':1, 'd0':1}

    # create sceanario
    mod = scenario()
    mod.makeDynamics(dyn)

    # initialization should contain None values in x and u
    if mod.init['x'] is not None or mod.init['u'] is not None:
        raise Exception('Initialization should contain None values in x and u.')
    
    # check initialization of w and d
    if mod.init['w'] != 1 or mod.init['d'] != 1:
        raise Exception('Initialization of w and d is wrong.')
    
    # now test setInit
    dyn = {'x':x, 'u':u, 'w':w, 'd':d, 'x_next':x_next, 'x_dot':x_dot}
    mod = scenario()
    mod.makeDynamics(dyn)

    # set initial conditions    
    mod.setInit({'d':1,'w':1})
    
    # now initialization should contain values in x and u
    if mod.init['d']!=1 or mod.init['w']!=1:
        raise Exception('Initialization process failed.')

    # last test: pass wrong dimension of x_next
    dyn = {'x':x, 'u':u, 'w':w, 'd':d, 'x_next':SX(1,1), 'x_dot':SX(2,1)}
    mod = scenario()
    test_failed = False
    failed_tests = []
    try:
        mod.makeDynamics(dyn)
    except Exception as e:
        if str(e)!='x_next must have the same dimensions as x.':
            test_failed = True
            failed_tests.append('x_next')
    # test x_dot
    dyn = {'x':x, 'u':u, 'w':w, 'd':d, 'x_next':SX(2,1), 'x_dot':SX(1,1)}
    mod = scenario()
    try:
        mod.makeDynamics(dyn)
    except Exception as e:
        if str(e)!='x_dot must have the same dimensions as x.':
            test_failed = True
            failed_tests.append('x_dot')
    # initialization of x
    try:
        mod.setInit({'x':SX(1,1)})
    except Exception as e:
        if str(e)!='x has incorrect dimension.':
            test_failed = True
            failed_tests.append('x')
    # initialization of u
    try:
        mod.setInit({'u':SX(2,1)})
    except Exception as e:
        if str(e)!='u has incorrect dimension.':
            test_failed = True
            failed_tests.append('u')
    # initialization of w
    dyn = {'x':x, 'u':u, 'w':w, 'd':d, 'x_next':SX(2,1), 'x_dot':SX(2,1)}
    mod = scenario()
    mod.makeDynamics(dyn)
    try:
        mod.setInit({'w':SX(2,1)})
    except Exception as e:
        if str(e)!='w has incorrect dimension.':
            test_failed = True
            failed_tests.append('w')
    # initialization of d
    try:
        mod.setInit({'d':SX(2,1)})
    except Exception as e:
        if str(e)!='d has incorrect dimension.':
            test_failed = True
            failed_tests.append('d')
    # if one test failed, raise exception
    if test_failed:
        raise Exception('makeDynamics did not spot a variable with wrong dimension. Failed tests: ' + str(failed_tests))

    # check that f and fc coincide with non-noisy model
    x_dot = A*x + B*u + c
    x_next = x_dot
    dyn = {'x':x, 'u':u, 'x_next':x_next, 'x_dot':x_dot}
    mod = scenario()
    mod.makeDynamics(dyn)
    if sum1(fabs(DM(mod.dyn.fc(x,u)-x_dot))) or sum1(fabs(DM(mod.dyn.fc_nom(x,u)-x_dot))):
        raise Exception('fc or fc_nom does not coincide with x_dot.')
    if sum1(fabs(DM(mod.dyn.f(x,u)-x_next))) or sum1(fabs(DM(mod.dyn.f_nom(x,u)-x_next))):
        raise Exception('f does not coincide with x_next.')
    
    # check if model was recognized to be affine
    if not mod.dyn.type == 'affine':
        raise Exception('Model was not recognized as affine.')
    
    # check derivatives
    if not mmax(fabs(mod.dyn.A(x,u) == A)):
        raise Exception('A matrix is incorrect.') 
    if not mmax(fabs(mod.dyn.B(x,u) == B)):
        raise Exception('B matrix is incorrect.')
    
    # create model where nonlinearity occurs only if d is not zero
    x_next = A*x + B*u + c + d*x**2
    dyn = {'x':x, 'u':u, 'd':d, 'x_next':x_next, 'x_dot':x_dot, 'd0':0}
    mod = scenario()
    mod.makeDynamics(dyn)

    # check that model is recognized as nonlinear
    if not mod.dyn.type == 'nonlinear':
        raise Exception('Model was not recognized as nonlinear.')
    
    # check correctness of nominal dynamics
    if sum1(fabs(DM(cse(mod.dyn.f_nom(x,u)-(A*x + B*u + c))))):
        raise Exception('f_nom does not coincide with x_next nominal.')