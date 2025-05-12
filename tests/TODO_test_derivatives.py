
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

