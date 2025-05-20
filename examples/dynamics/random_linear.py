import casadi as ca
import numpy as np

def dynamics(n_x:int=2,pole_mag:list=[0.8,1.2],use_theta:bool=True,use_w:bool=True) -> dict:

    assert pole_mag[1] >= pole_mag[0], 'Pole magnitude bounds should be given as [mag_min, mag_max] with mag_min <= mag_max.'

    # always one input
    n_u = 1

    # define state and input symbolic variables
    x = ca.SX.sym('x',n_x,1)
    u = ca.SX.sym('u',n_u,1)

    # generate random floats between -1 and 1
    random_units = 2*np.random.rand(1,n_x)-np.ones((1,n_x))

    # adapt to specified range
    poles = random_units*(pole_mag[1]-pole_mag[0]) + np.ones((1,n_x))*pole_mag[0]
    
    # put ones in the off-diagonal of A
    A = ca.SX(np.diag(np.ones(n_x-1),k=1))

    # substitute last row of A with randomly generated poles (with flipped sign)
    A[-1,:] = -ca.DM(poles)

    # B matrix in controllable canonical form
    B = ca.vcat([ca.SX(n_x-1,1),1])

    # create output dictionary
    out = {'x':x, 'u':u}
    func_in = [x,u]
    func_in_nom = [x,u]
    
    # nominal model is the entire A and B matrices
    if use_theta:

        # create symbolic variable
        theta = ca.SX.sym('theta',n_x*(n_x+n_u),1)

        # append to dictionaries
        out['theta'] = theta
        func_in_nom.append(theta)

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