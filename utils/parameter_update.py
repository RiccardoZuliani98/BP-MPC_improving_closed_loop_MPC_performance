import casadi as ca
import numpy as np

def average_gradient_descent(rho,eta,log=True):

    def parameter_update(sim,k):

        # average all Jacobians
        j_p = ca.sum2(sim.j_p) / sim.j_p.shape[1]

        # gradient step
        p_next = sim.p - (rho*ca.log(k+2)/(k+1)**eta)*j_p if log else sim.p - (rho/(k+1)**eta)*j_p

        return {'p':p_next}

    return parameter_update, lambda sim: {}

def robust_gradient_descent(rho,eta,n_models,n_p,log=True,jit=False,verbose=False):

    # compilation options
    if jit:
        jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
        options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
    else:
        options = {}

    if not verbose:
        options = options | {'osqp':{'verbose':False}}

    # create optimization variables
    d = ca.SX.sym('d',n_p,1)
    epsilon = ca.SX.sym('epsilon',1,1)

    # create constraint functions
    g1 = -ca.repmat(d,n_models,1) + ca.SX.ones(n_models*n_p,1)*epsilon
    g2 = ca.repmat(d,n_models,1) + ca.SX.ones(n_models*n_p,1)*epsilon

    # form objective
    f = epsilon**2

    # form QP solver
    S = ca.qpsol('S','osqp',{'x':ca.vertcat(epsilon,d),'f':f,'g':ca.vertcat(g1,g2)},options)

    def parameter_update(sim,k):

        # get gradient matrix and form lower-bound
        j_p = ca.reshape(ca.DM(sim.j_p),-1,1)

        # solve
        sol = S(lbg=ca.vertcat(-j_p,j_p))['x']

        # get direction
        d = sol[1:]

        # run GD update
        p_next = sim.p - (rho*ca.log(k+2)/(k+1)**eta)*d if log else sim.p - (rho/(k+1)**eta)*d

        return {'p':p_next}

    # no initialization for psi
    parameter_init = lambda sim: {}

    return parameter_update,parameter_init

def gradient_descent(rho,eta=1,log=True):
    
    def update_log(sim,k):
        return {'p': sim.p - (rho*ca.log(k+2)/(k+1)**eta)*sim.j_p}
    
    def update_simple(sim,k):
        return {'p': sim.p - (rho/(k+1)**eta)*sim.j_p}
        
    parameter_update = update_log if log else update_simple
        
    return parameter_update, lambda sim: {}

def minibatch_descent(rho,eta=1,log=True,batch_size=1):

    def parameter_update(sim,k):
        
        # check if number of steps has been reached
        if ca.fmod(k+1,batch_size) == 0:

            # construct average gradient
            j_p = (sim.psi['j_p'] + sim.j_p) / batch_size

            # zero the running gradient
            psi = {'j_p':ca.DM.zeros(*j_p.shape)}
            
            # run update
            p = sim.p - (rho*ca.log(k+2)/(k+1)**eta)*j_p if log else sim.p - (rho/(k+1)**eta)*j_p

        # else update gradient
        else:
            psi = sim.psi['j_p'] + sim.j_p
            p = sim.p

        return {'p':p,'psi':psi}
    
    def parameter_init(sim):
        return {'j_p':ca.DM.zeros(*sim.j_p.shape)}

    return parameter_update, parameter_init