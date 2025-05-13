import casadi as ca

def dynamics(dt=0.05,use_theta:bool=False,use_d:bool=True,use_w:bool=True) -> dict: #l_p=0.0712,m_p=0.0218,m_c=0.2832,

    # define dimensions of the problem
    n_x = 4               # number of states
    n_u = 1               # number of inputs
    n_d = 3               # number of uncertain parameters
    n_w = 2               # number of disturbances

    # define symbolic variables
    x = ca.SX.sym('x0',n_x,1)         # state
    u = ca.SX.sym('u0',n_u,1)         # input

    # create output dictionary
    out = {'x':x, 'u':u}
    func_in = [x,u]
    func_in_nom = [x,u]

    # gravitational acceleration
    g = 9.804 #9.81

    # extract parameters
    m_nom = 0.305 #m_p + m_c
    mu_nom = 1.5492*1e-3 #m_p*l_p
    j_nom = 1.47*1e-4 #4/3*m_p*l_p**2
    
    # check if model uncertainty is present
    if use_d:
        d_m = ca.SX.sym('d_m',1,1)
        d_mu = ca.SX.sym('d_mu',1,1)
        d_J = ca.SX.sym('d_J',1,1)
        d = ca.vcat([d_m,d_mu,d_J])
        out['d'] = d
        func_in.append(d)
        func_in_nom.append(ca.SX(*d.shape))

    # add uncertainty if required
    m = m_nom*(1+d_m) if use_d else m_nom
    mu = mu_nom*(1+d_mu) if use_d else mu_nom
    j = j_nom if use_d else j_nom

    # extract states
    p_dot = x[1]
    theta = x[2]
    theta_dot = x[3]

    # extract inputs
    u = u[0]

    # construct derivatives
    denom = m*j - mu**2*ca.cos(theta)**2 
    theta_ddot = (m*mu*g*ca.sin(theta) - mu*ca.cos(theta)*(u + mu*theta_dot**2*ca.sin(theta))) / denom 
    p_ddot = (j*(u + mu*theta_dot**2*ca.sin(theta)) - mu**2*g*ca.sin(theta)*ca.cos(theta)) / denom 

    # Construct a CasADi function for the ODE right-hand side
    x_dot = ca.vertcat(p_dot,p_ddot,theta_dot,theta_ddot)

    # create function
    fc = ca.Function('fc', func_in, [x_dot])

    # my own RK4
    def rk4(fc,T,x,*args):
        k1 = fc(x, *args)
        k2 = fc(x + T/2*k1, *args)
        k3 = fc(x + T/2*k2, *args)
        k4 = fc(x + T*k3, *args)
        return x + T/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    # # my own Euler
    # def euler(fc,x,u,d,T):
    #     return x + T*fc(x,u,d)

    # noise if required
    if use_w:
        w = ca.SX.sym('w',n_w,1)
        out['w'] = w

    # compute next state symbolically
    x_next = rk4(fc,dt,*func_in) + ca.vcat([0,w[0],0,w[1]]) if use_w else rk4(fc,dt,*func_in)
    out['x_next'] = x_next

    # check if nominal model needs to be shipped too
    if use_theta:

        # compute nominal state
        theta = ca.SX.sym('theta',n_d,1)

        # add to output dictionary
        out = out | {'theta':theta}

    # create nominal model
    x_next_nom = rk4(fc,dt,x,u,theta) if use_theta else rk4(fc,dt,*func_in_nom)
    out['x_next_nom'] = x_next_nom

    return out