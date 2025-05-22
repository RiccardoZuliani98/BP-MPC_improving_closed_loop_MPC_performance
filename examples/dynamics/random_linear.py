import casadi as ca
import numpy as np
from scipy.linalg import expm,eig

def dynamics(Ts:float=0.1,n_x:int=2,pole_mag:list=[0.5,1.2],use_theta:bool=True,use_w:bool=True) -> dict:

    assert pole_mag[1] >= pole_mag[0], 'Pole magnitude bounds should be given as [mag_min, mag_max] with mag_min <= mag_max.'

    # always one input
    n_u = 1

    # define state and input symbolic variables
    x = ca.SX.sym('x',n_x,1)
    u = ca.SX.sym('u',n_u,1)

    # generate random continuous-time poles
    poles = np.random.rand(n_x)*(pole_mag[1]-pole_mag[0]) + np.ones(n_x)*pole_mag[0]
    
    # put ones in the off-diagonal of A
    A_cont = np.diag(np.ones(n_x-1),k=1)

    # substitute last row of A with characteristic polynomial
    A_cont[-1,:] = -np.array(np.flip(np.poly(poles)[1:]))

    # B matrix in controllable canonical form
    B_cont = ca.vcat([ca.DM(n_x-1,1),1])

    # check eigenvalues of A_cont
    eig_a_cont = eig(A_cont)[0]
    
    # check that poles match
    assert np.allclose(np.sort(eig_a_cont),np.sort(poles),rtol=1e-12), 'Poles do not match'

    # # Euler
    # A_euler = ca.SX.eye(n_x) + Ts*ca.SX(A_cont)
    # B_euler = Ts*B_cont

    # # second order
    # A_second_order = ca.SX.eye(n_x) + Ts*ca.SX(A_cont) + Ts**2/2*ca.SX(A_cont@A_cont)

    # discretize
    A = ca.cse(ca.sparsify(ca.SX(expm(Ts*A_cont))))
    B = ca.cse(ca.sparsify(ca.SX(ca.pinv(A_cont)@(A-ca.SX.eye(n_x))@B_cont)))

    # check eigenvalues of A
    eig_a = eig(expm(Ts*A_cont))[0]
    print(f'Eigenvalues of A: {np.absolute(eig_a)}')

    # create output dictionary
    out = {'x':x, 'u':u}
    
    # nominal model is the entire A and B matrices
    if use_theta:

        # create symbolic variable
        theta = ca.SX.sym('theta',n_x*(n_x+n_u),1)

        # append to dictionaries
        out['theta'] = theta

        # create nominal dynamics
        A_nom = ca.reshape(theta[:n_x*n_x],n_x,n_x)
        B_nom = ca.reshape(theta[n_x*n_x:],n_x,n_u)

    # otherwise nominal and true models coincide
    else:
        A_nom = A
        B_nom = B

    # create successor state and nominal successor state
    x_next = ca.cse(ca.sparsify(A@x + B@u))
    x_next_nom = ca.cse(ca.sparsify(A_nom@x + B_nom@u))

    # print true theta
    true_theta = ca.DM(ca.vec(ca.jacobian(x_next,ca.vertcat(x,u))))

    # noise if required
    if use_w:

        # generate w
        w = ca.SX.sym('w',n_x,1)
        out['w'] = w

        # add to true dynamics
        x_next = x_next + w

    # add to dictionaries
    out['x_next'] = x_next
    out['x_next_nom'] = x_next_nom

    return out,true_theta