import casadi as ca
import numpy as np

def quadCostAndBounds(Q,R,x_cl,u_cl,x_max=None,x_min=None,x_ref=None,u_ref=None):

    # get symbolic type
    MSX = type(x_cl)

    # ensure that x_cl and u_cl are column vectors
    x_cl = ca.vec(x_cl)
    u_cl = ca.vec(u_cl)

    # get state dimension
    n_x = Q.shape[0]

    # get closed-loop horizon
    T = int(x_cl.shape[0]/n_x) - 1

    # stack all constraints
    if x_max is not None:
        x_max_stack = ca.repmat(x_max,T+1,1)
    if x_min is not None:
        x_min_stack = ca.repmat(x_min,T+1,1)

    if x_ref is None:
        x_ref = MSX(*x_cl.shape)
    else:
        if x_ref.shape[0] != x_cl.shape[0]:
            raise Exception('Inconsistent dimensions for x_ref.')
    if u_ref is None:
        u_ref = MSX(*u_cl.shape)
    else:
        if u_ref.shape[0] != u_cl.shape[0]:
            raise Exception('Inconsistent dimensions for u_ref.')

    # closed-loop tracking cost
    track_cost = (x_cl-x_ref).T@ca.kron(MSX.eye(T+1),Q)@(x_cl-x_ref) + (u_cl-u_ref).T@ca.kron(MSX.eye(T),R)@(u_cl-u_ref)

    try:
        track_cost = ca.cse(ca.sparsify(track_cost))
    except:
        pass

    # constraint violation (l2 and l1 norm)
    if x_max is not None:    
        cst_viol_l1 = MSX.ones(1,x_cl.shape[0])@ca.fmax(x_cl-MSX(x_max_stack),ca.fmax(MSX(x_min_stack)-x_cl,MSX((T+1)*n_x,1)))
        cst_viol_l2 = ca.fmax(x_cl-MSX(x_max_stack),ca.fmax(MSX(x_min_stack)-x_cl,MSX((T+1)*n_x,1))).T@ca.fmax(x_cl-MSX(x_max_stack),ca.fmax(MSX(x_min_stack)-x_cl,MSX((T+1)*n_x,1)))
        try:
            cst_viol_l1 = ca.cse(ca.sparsify(cst_viol_l1))
            cst_viol_l2 = ca.cse(ca.sparsify(cst_viol_l2))
        except:
            pass
    else:
        cst_viol_l1 = None
        cst_viol_l2 = None
                    
    return track_cost, cst_viol_l1, cst_viol_l2

def param2terminalCost(p):

    # get symbolic type
    MSX = type(p)

    # get state dimension
    n_x = int(0.5*(ca.sqrt(8*p.shape[0]+1)-1))

    # construct cholesky decomposition Qn = LL.T of terminal cost by
    # rearranging the entries in the parameter vector c_qx. First
    # preallocate L
    L = MSX(n_x,n_x)

    # construct L row by row
    len = 0
    for i in range(n_x):
        len = len + i
        L[i,0:i+1] = p[len:len+i+1]

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

# imports
import os, glob

def cleanup():
    
    # get all jit files
    list_of_files = glob.glob("./jit*")

    # get all tmp files
    list_of_files.extend(glob.glob("./tmp*"))

    # eliminate all
    for filename in list_of_files:
        os.remove(filename)