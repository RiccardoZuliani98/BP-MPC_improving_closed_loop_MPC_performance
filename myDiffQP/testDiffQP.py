from casadi import *
from myDiffQP.simVar import *

def testOptQPvsQP(QP,x0,p0,N,n_x):
    
    # get optimal solution with qp_opt
    sol = QP.opt_solve(x0)

    # get linearization trajectory
    y0 = vertcat(reshape(sol[0][:-1,:].T,N*n_x,1),sol[1])

    # solve QP
    lam,mu,y = QP.solve(vertcat(x0,y0,p0))

    # compare
    y_opt = vertcat(reshape(sol[0][1:,:].T,N*n_x,1),sol[1])
    print(sum1(fabs(y-y_opt)))

def testDual(QP,problem):

    # extract init stacks
    x0_stack = problem['init']['x0']
    y0_stack = problem['init']['y0']
    p0_stack = problem['init']['p0']

    # initialize erro vector
    error = []

    # loop through all initial conditions
    for x0 in x0_stack:

        # loop through all initial trajectories
        for y0 in y0_stack:

            # loop through all parameters
            for p0 in p0_stack:
                
                # create stacked parameter
                p = vertcat(x0,y0,p0)

                # solve dual
                z = QP.dual_solve(p)

                # solve QP
                lam,mu,y,eps = QP.solve(p)

                # compare
                error.append(sum1(fabs(z-vertcat(lam,mu))))

    # printout
    print(f'Max dual error is: {max(error)}')


def testConsJac(diffQP,problem):

    # extract init stacks
    x0_stack = problem['init']['x0']
    y0_stack = problem['init']['y0']
    p0_stack = problem['init']['p0']

    # get number of parameters
    n_p = p0_stack[0].shape[0]
    n_x = x0_stack[0].shape[0]
    n_y = y0_stack[0].shape[0]

    # initialize dp_stack
    dp_stack = []

    # create list of all the dp
    for i in range(n_p+n_x+n_y):
        dp_stack.append( repmat(0,n_p+n_x+n_y,1) )
        dp_stack[-1][i] = 0.001

    # loop through all initial conditions
    for x0 in x0_stack:

        # loop through all linearization trajectories
        for y0 in y0_stack:

            # loop through all parameters
            for p0 in p0_stack:

                # create stacked parameter
                p = vertcat(x0,y0,p0)
        
                # initialize lists containing variations
                dy_error = []

                for dp in dp_stack:

                    # evaluate QP to get two different solutions
                    lam_1,mu_1,y_1,temp = diffQP.solve(p)
                    lam_2,mu_2,y_2,temp = diffQP.solve(p+dp)

                    # compute conservative jacobian
                    J_y = diffQP.J_y(lam_1,mu_1,0.2,p)

                    # get primal variation
                    measured_dy = (y_2-y_1)/norm_2(dp)
                    predicted_dy = J_y@dp/norm_2(dp)

                    # store in error list
                    if norm_2(measured_dy) > 1e-12:
                        dy_error.append(sum1(fabs(measured_dy-predicted_dy)) / norm_2(measured_dy))

                # printout
                print(f'max y error (%): {max(dy_error)}')

def testConsJacClosedLoop(QP,problem):

    # extract init stacks
    x0_stack = problem['init']['x0']
    y0_stack = problem['init']['y0']
    p0_stack = problem['init']['p0']

    # get number of parameters
    n_p = p0_stack[0].shape[0]

    # initialize dp_stack
    dp_stack = []

    # create list of all the dp
    for i in range(n_p):
        dp_stack.append( repmat(0,n_p,1) )
        dp_stack[-1][i] = 0.0001

    # set max_it to 1
    problem['alg']['max_k'] = 1

    # loop through all initial conditions
    for x0 in x0_stack:

        # loop through all initial trajectories
        for y0 in y0_stack:

            # loop through all parameters
            for p0 in p0_stack:
                
                # initialize lists containing errors
                dx_error = []

                # define initialization
                init = {'x0':[x0],'y0':[y0],'p0':[p0]}

                # redefine problem initial condition
                problem['init'] = init

                # simulate without perturbation
                SIM_0, cost, p, opt_cost, opt_sol, comp_time = closedLoop(QP,problem,0)

                # loop through all perturbations
                for dp in dp_stack:

                    # compute full p
                    p = p0 + dp

                    # define initialization
                    init = {'x0':[x0],'y0':[y0],'p0':[p]}

                    # redefine problem initial condition
                    problem['init'] = init

                    # run closed-loop simulation
                    SIM, cost, p, opt_cost, opt_sol, comp_time = closedLoop(QP,problem,0)

                    # get Jacobian and closed-loop state
                    J_x = SIM[0].Jx
                    X = SIM[0].X

                    # compute predicted variation
                    predicted_dx = J_x@dp/norm_2(dp)

                    # compute measured variation
                    measured_dx = (SIM[0].X-SIM_0[0].X)/norm_2(dp)

                    # store in error list
                    dx_error.append(sum1(fabs(predicted_dx-measured_dx)) / norm_2(measured_dx))

                print(f'max closed-loop x error (%): {max(dx_error)}')


def testClosedLoop(QP,problem):

    # extract init stacks
    x0_stack = problem['init']['x0']
    y0_stack = problem['init']['y0']
    p0_stack = problem['init']['p0']

    # get total number of experiments
    tot_exp = len(x0_stack)*len(y0_stack)*len(p0_stack)
    
    # start counter for number of experiments done
    done_exp = 0

    # loop over initial conditions here
    for x0 in x0_stack:
        
        # loop over initial trajectories
        for y0 in y0_stack:

            # loop over initial parameter values
            for p0 in p0_stack:

                # define initialization
                init = {'x0':[x0],'y0':[y0],'p0':[p0]}

                # redefine problem initial condition
                problem['init'] = init

                # run closed-loop simulation
                SIM, cost, p, opt_cost, opt_sol, comp_time = closedLoop(QP,problem,0)
                # increment the counter
                done_exp = done_exp + 1
                # printout
                print(f"{done_exp} \\ {tot_exp}, cost: {cost[-1]}, best cost: {opt_cost}, improvement: {cost[-1]-cost[0]}, p0: {p0}, pM: {p}, x0: {x0}")