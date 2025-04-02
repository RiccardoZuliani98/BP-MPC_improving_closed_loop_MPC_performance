# type of symbolic variable used (either SX or MX)
__MSX = None

"""
Dimension dictionary with keys

    - N: horizon of the MPC, positive integer
    - u: number of inputs, positive integer
    - x: number of states, positive integer
    - eps: number of slack variables, positive integer [optional, defaults to 0]

"""
__dim = {}

"""
Model dictionary with entries

    - A: list of length N of matrices (n_x,n_x)
    - B: list of length N of matrices (n_x,n_u)
    - x0: symbolic variable representing the initial state (n_x,1)
    - c: list of length N of matrices (n_x,1) [optional, defaults to 0]
        
where the dynamics are given by x[t+1] = A[t]x[t] + B[t]u[t] + c[t], with x[0] = x0.
"""
__model = {}

"""
Cost dictionary with keys
            
    - 'Qx': state stage cost, list of length N of matrices (n_x,n_x)
    - 'Ru': input stage cost, list of length N of matrices (n_u,n_u)
    - 'x_ref': state reference, list of length N of vectors (n_x,1) [optional, defaults to 0]
    - 'u_ref': reference input, list of length N of vectors (n_u,1) [optional, defaults to 0]
    - 's_lin': linear penalty on slack variables, nonnegative scalar [optional, defaults to 0]
    - 's_quad': quadratic penalty on slack variables, positive scalar [optional, defaults to 0]

where the stage cost is given by
    
    (x[t]-x_ref[t])'Qx[t](x[t]-x_ref[t]) + (u[t]-u_ref[t])'Ru[t](u[t]-u_ref[t]) + s_lin*e[t] + s_quad*e[t]**2
"""
__cost = {}

"""
    Constraints dictionary with keys

    - 'Hx': list of length N of matrices (=,n_x)
    - 'hx': list of length N of vectors (=,1)
    - 'Hx_e': list of length N of matrices (=,n_eps) [optional, defaults to zero]
    - 'Hu': list of length N of matrices (-,n_u)
    - 'hu': list of length N of vectors (-,1)
    
where the constraints at each time-step are
    
    Hx[t]*x[t] <= hx[t] - Hx_e[t]*e[t],
    Hu[t]*u[t] <= hu[t],
        
where e[t] denotes the slack variables.
"""
__cst = {}

# keys allowed in dictionaries
__allowed_keys = {'dim':['N','u','x','eps','cst_x','cst_u'],
                    'model':['A','B','c'],
                    'cost':['Qx','Ru','x_ref','u_ref','s_lin','s_quad'],
                    'cst':['Hx','Hx_e','Hu','hx','hu']}

# expected dimensions
__expected_dimensions = {'model':{'A':['x','x'],'B':['x','u'],'c':['x','one']},
                            'cost':{'Qx':['x','x'],'Ru':['u','u'],'x_ref':['x','one'],'u_ref':['u','one'],'s_lin':['one','one'],'s_quad':['one','one']},
                            'cst':{'Hx':['cst_x','x'],'Hx_e':['cst_x','eps'],'Hu':['cst_u','u'],'hx':['cst_x','one'],'hu':['cst_u','one']}}

# allowed inputs to __init__ and updateMPC
allowed_inputs = ['model','cost','cst']