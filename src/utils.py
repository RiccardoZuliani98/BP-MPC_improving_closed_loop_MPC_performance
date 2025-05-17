import casadi as ca
import numpy as np
import os, glob

#TODO: add descriptions

def rls(psi_init):

    def sys_id_update(sim,_):

        # get parameters
        A_k = sim.psi['A']
        b_k = sim.psi['b']
        theta = sim.psi['theta']

        # run through the horizon and perform the RLS updates
        new_psi = {'A':A_k,'b':b_k,'theta':theta}

        return sim.psi | new_psi
    
    def sys_id_init():
        return psi_init
    
    return sys_id_update, sys_id_init

def average_gradient_descent(rho,eta,log=True):

    def parameter_update(sim,k):

        # average all Jacobians
        j_p = ca.sum2(sim.j_p) / sim.j_p.shape[1]

        # gradient step
        p_next = sim.p - (rho*ca.log(k+2)/(k+1)**eta)*j_p if log else sim.p - (rho/(k+1)**eta)*j_p

        return p_next, None

    return parameter_update, lambda sim: None

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

        # no auxiliary parameter
        psi = None

        return p_next,psi

    # no initialization for psi
    parameter_init = lambda sim: None

    return parameter_update,parameter_init

def gradient_descent(rho,eta=1,log=True):
    
    def update_log(sim,k):
        return sim.p - (rho*ca.log(k+2)/(k+1)**eta)*sim.j_p, None
    
    def update_simple(sim,k):
        return sim.p - (rho/(k+1)**eta)*sim.j_p, None
        
    parameter_update = update_log if log else update_simple
        
    return parameter_update, lambda sim: None

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

        return p,psi
    
    def parameter_init(sim):
        return {'j_p':ca.DM.zeros(*sim.j_p.shape)}

    return parameter_update, parameter_init
    
def quad_cost_and_bounds(Q,R,x_cl,u_cl,x_max=None,x_min=None,x_ref=None,u_ref=None):

    # get symbolic type
    msx = type(x_cl)

    # ensure that x_cl and u_cl are column vectors
    x_cl = ca.vec(x_cl)
    u_cl = ca.vec(u_cl)

    # get state dimension
    n_x = Q.shape[0]

    # get closed-loop horizon
    T = int(x_cl.shape[0]/n_x) - 1

    # stack all constraints
    x_max_stack = ca.repmat(x_max,T+1,1) if x_max is not None else None
    x_min_stack = ca.repmat(x_min,T+1,1) if x_min is not None else None

    if x_ref is None:
        x_ref = msx(*x_cl.shape)
    else:
        if x_ref.shape[0] != x_cl.shape[0]:
            raise Exception('Inconsistent dimensions for x_ref.')
    if u_ref is None:
        u_ref = msx(*u_cl.shape)
    else:
        if u_ref.shape[0] != u_cl.shape[0]:
            raise Exception('Inconsistent dimensions for u_ref.')

    # closed-loop tracking cost
    track_cost = (x_cl-x_ref).T@ca.kron(msx.eye(T+1),Q)@(x_cl-x_ref) + (u_cl-u_ref).T@ca.kron(msx.eye(T),R)@(u_cl-u_ref)

    # sparsify if symbolic type is SX
    if msx == ca.SX:
        track_cost = ca.cse(ca.sparsify(track_cost))

    # constraint violation (l2 and l1 norm)
    if x_max is not None:    
        cst_viol_l1 = msx.ones(1,x_cl.shape[0])@ca.fmax(x_cl-msx(x_max_stack),ca.fmax(msx(x_min_stack)-x_cl,msx((T+1)*n_x,1)))
        cst_viol_l2 = ca.fmax(x_cl-msx(x_max_stack),ca.fmax(msx(x_min_stack)-x_cl,msx((T+1)*n_x,1))).T@ca.fmax(x_cl-msx(x_max_stack),ca.fmax(msx(x_min_stack)-x_cl,msx((T+1)*n_x,1)))
        if msx == ca.SX:
            cst_viol_l1 = ca.cse(ca.sparsify(cst_viol_l1))
            cst_viol_l2 = ca.cse(ca.sparsify(cst_viol_l2))
    else:
        cst_viol_l1 = None
        cst_viol_l2 = None
                    
    return track_cost, cst_viol_l1, cst_viol_l2

def param_2_terminal_cost(p):

    # get symbolic type
    msx = type(p)

    # get state dimension
    n_x = int(0.5*(ca.sqrt(8*p.shape[0]+1)-1))

    # construct Cholesky decomposition Qn = LL.T of terminal cost by
    # rearranging the entries in the parameter vector c_qx. First
    # preallocate L
    L = msx(n_x,n_x)

    # construct L row by row
    length = 0
    for i in range(n_x):
        length = length + i
        L[i,0:i+1] = p[length:length+i+1]

    if isinstance(p,ca.SX):
        out = ca.cse(ca.sparsify(L@L.T))
    else:
        out = L@L.T

    return out

def dare2param(A,B,Q,R):

    # imports
    from scipy.linalg import solve_discrete_are as dare
    from scipy.linalg import cholesky

    # turn to numpy array
    A = np.array(A)
    B = np.array(B)
    Q = np.array(Q)
    R = np.array(R)

    # obtain solution of dare
    P = dare(A,B,Q,R)

    # turn into parameter
    P_half = ca.DM(cholesky(P,lower=True))

    # helper function to unpack P_half into a parameter vector
    def P2p(P):
        n = P.shape[0]
        p = []
        for i in range(n):
            for j in range(i+1):
                p.append(P[i,j])
        return ca.vcat(p)

    return P2p(P_half)

def bound2poly(x_max,x_min,u_max,u_min,N=1):

    # turn to DM
    x_max = ca.DM(x_max)
    x_min = ca.DM(x_min)
    u_max = ca.DM(u_max)
    u_min = ca.DM(u_min)

    # get dimensions
    n_x = x_max.shape[0]
    n_u = u_max.shape[0]

    # check dimension of x min and u min
    if x_min.shape[0] != n_x:
        raise Exception('Inconsistent dimensions for x_min.')
    if u_min.shape[0] != n_u:
        raise Exception('Inconsistent dimensions for u_min.')

    # preallocate inequality constraint matrices (state)
    hx = ca.repmat(ca.vertcat(x_max,-x_min),N,1)
    Hx = ca.kron(ca.DM.eye(N),ca.vertcat(ca.DM.eye(n_x),-ca.DM.eye(n_x)))

    # get indices associated to inf entries in hx
    idx_hx = []
    for idx in range(0,hx.shape[0]):
        if hx[idx] == ca.inf:
            idx_hx.append(idx)

    # remove indices from hx and Hx
    hx.remove(idx_hx,[])
    Hx.remove(idx_hx,[])

    # preallocate inequality constraint matrices (input)
    Hu = ca.kron(ca.DM.eye(N),ca.vertcat(ca.DM.eye(n_u),-ca.DM.eye(n_u)))
    hu = ca.repmat(ca.vertcat(u_max,-u_min),N,1)

    # get indices associated to inf entries in hu
    idx_hu = []
    for idx in range(0,hu.shape[0]):
        if hu[idx] == ca.inf:
            idx_hu.append(idx)

    # remove indices from Hx and hx
    Hu.remove(idx_hu,[])
    hu.remove(idx_hu,[])

    return Hx,hx,Hu,hu

# matrix generating a block diagonal matrix starting from the individual blocks
def matrixify(M_list):

    # get dimensions
    N = len(M_list)
    n_col = M_list[0].shape[1]
    n_row = M_list[0].shape[0]

    # pad matrices with zeros
    M_list_pad = [ ca.vcat([ca.SX(i*n_row,n_col),M_list[i],ca.SX((N-i-1)*n_row,n_col)]) for i in range(N) ]

    # stack horizontally
    return ca.hcat(M_list_pad)

def cleanup():
    
    # get all jit files
    list_of_files = glob.glob("./jit*")

    # get all tmp files
    list_of_files.extend(glob.glob("./tmp*"))

    # eliminate all
    for filename in list_of_files:
        os.remove(filename)