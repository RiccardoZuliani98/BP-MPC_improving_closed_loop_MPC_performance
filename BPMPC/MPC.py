from casadi import *

class MPC:

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

        - A: list of matrices (of length N) or a single matrix (n_x,n_x)
        - B: list of matrices (of length N) or a single matrix (n_x,n_u)
        - c: list of matrices (of length N) or a single matrix (n_x,1) [optional, defaults to 0]
            
    where the dynamics are given by x[t+1] = A[t]x[t] + B[t]u[t] + c[t], or (in the time-invariant case) by
    x[t+1] = Ax[t] + Bu[t] + c.
    """
    __model = {}

    """
    Cost dictionary with keys
                
        - 'Qx': state stage cost, list of matrices (of length N) or single matrix (n_x,n_x)
        - 'Qn': terminal state cost matrix (n_x,n_x) [optional, defaults to Qx]
        - 'Ru': input stage cost, list of matrices (of length N) or single matrix (n_u,n_u)
        - 'x_ref': state reference, list of vectors (of length N) or single vector (n_x,1) [optional, defaults to 0]
        - 'u_ref': reference input, list of vectors (of length N) or single vector (n_u,1) [optional, defaults to 0]
        - 's_lin': linear penalty on slack variables, nonnegative scalar [optional, defaults to 0]
        - 's_quad': quadratic penalty on slack variables, positive scalar [optional, defaults to 0]

    where the stage cost is given by
        
        (x[t]-x_ref[t])'Qx[t](x[t]-x_ref[t]) + (u[t]-u_ref[t])'Ru[t](u[t]-u_ref[t]) + s_lin*e[t] + s_quad*e[t]**2
        
    Note: Qn is disregarded if Qx is a list.
    """
    __cost = {}

    """
     Constraints dictionary with keys
    
        - 'Hx': list of matrices (length N) or single matrix (=,n_x)
        - 'hx': list of vectors (length N) or single vector (=,1)
        - 'Hx_e': list of matrices (length N) or single matrix (=,n_eps) [optional, defaults to zero]
        - 'Hu': list of matrices (length N) or single matrix (-,n_u)
        - 'hu': list of vectors (length N) or single vector (-,1)
        
    where the constraints at each time-step are
        
        Hx[t]*x[t] <= hx[t] - Hx_e[t]*e[t],
        Hu[t]*u[t] <= hu[t],
            
    where e are the slack variables.
    """
    __cst = {}

    # keys allowed in dictionaries
    __allowed_keys = {'dim':['N','u','x','eps','cst_x','cst_u'],
                      'model':['A','B','c'],
                      'cost':['Qx','Ru','Se','qx','ru','se'],
                      'cst':['Hx','Hx_e','Hu','hx','hu']}
    
    # expected dimensions
    __expected_dimensions = {'model':{'A':['x','x'],'B':['x','u'],'c':['x','one']},
                             'cost':{'Qx':['x','x'],'Ru':['u','u'],'Qn':['x','x'],'x_ref':['x','one'],'u_ref':['u','one']},
                             'cst':{'Hx':['cst_x','x'],'Hx_e':['cst_x','eps'],'Hu':['cst_u','u'],'hx':['cst_x','one'],'hu':['cst_u','one']}}
    
    # allowed inputs to __init__ and updateMPC
    allowed_inputs = ['N','model','cost','cst']


    def __init__(self,N=None,model=None,cost=None,cst=None,MSX=SX):

        """
        Constructor of the MPC class.

        INPUTS (all optional):

            * N: horizon of the MPC, positive integer

            * model: model definition, dictionary with keys

                - A: list of matrices (of length N) or a single matrix (n_x,n_x)
                - B: list of matrices (of length N) or a single matrix (n_x,n_u)
                - c: list of matrices (of length N) or a single matrix (n_x,1) [optional, defaults to 0]
            
                where the dynamics are given by x[t+1] = A[t]x[t] + B[t]u[t] + c[t], or (in the time-invariant case) by
                x[t+1] = Ax[t] + Bu[t] + c.

            * cost: cost definition, dictionary with keys
                
                - 'Qx': state stage cost, list of matrices (of length N) or single matrix (n_x,n_x)
                - 'Qn': terminal state cost matrix (n_x,n_x) [optional, defaults to Qx]
                - 'Ru': input stage cost, list of matrices (of length N) or single matrix (n_u,n_u)
                - 'x_ref': state reference, list of vectors (of length N) or single vector (n_x,1) [optional, defaults to 0]
                - 'u_ref': reference input, list of vectors (of length N) or single vector (n_u,1) [optional, defaults to 0]
                - 's_lin': linear penalty on slack variables, nonnegative scalar [optional, defaults to 0]
                - 's_quad': quadratic penalty on slack variables, positive scalar [optional, defaults to 0]

              where the stage cost is given by
              
              (x[t]-x_ref[t])'Qx[t](x[t]-x_ref[t]) + (u[t]-u_ref[t])'Ru[t](u[t]-u_ref[t]) + s_lin*e[t] + s_quad*e[t]**2
              
              Note: Qn is disregarded if Qx is a list.
            
            * cst: constraints definition, dictionary with keys
            
                - 'Hx': list of matrices (length N) or single matrix (=,n_x)
                - 'hx': list of vectors (length N) or single vector (=,1)
                - 'Hx_e': list of matrices (length N) or single matrix (=,n_eps) [optional, defaults to zero]
                - 'Hu': list of matrices (length N) or single matrix (-,n_u)
                - 'hu': list of vectors (length N) or single vector (-,1)
              
                where the constraints at each time-step are
              
                    Hx[t]*x[t] <= hx[t] - Hx_e[t]*e[t],
                    Hu[t]*u[t] <= hu[t],
                    
                where e are the slack variables.

            * MSX: type of symbolic variable to use (either SX or MX) [optional, defaults to SX]

        """

        # check that MSX is of the appropriate type
        if MSX == 'SX':
            self.__MSX = SX
        elif MSX == 'MX':
            self.__MSX = MX
        else:
            raise Exception('MSX must be either SX or MX.')
        
        # gather inputs that are not None (except MSX)
        inputs = {k:v for k,v in locals().items() if k in self.allowed_inputs and v is not None}

        # call update function
        self.updateMPC(**inputs)
        
    def updateMPC(self,N=None,model=None,cost=None,cst=None):

        """
        Updates the MPC ingredients contained in the MPC class object.

        INPUTS:

            * N: horizon of the MPC, positive integer

            * model: model definition, dictionary containing entries

                - A: list of matrices (of length N) or a single matrix (n_x,n_x)
                - B: list of matrices (of length N) or a single matrix (n_x,n_u)
                - c: list of matrices (of length N) or a single matrix (n_x,1) [optional, defaults to 0]
            
                where the dynamics are given by x[t+1] = A[t]x[t] + B[t]u[t] + c[t], or (in the time-invariant case) by
                x[t+1] = Ax[t] + Bu[t] + c.

            * cost: cost definition, dictionary with keys
                
                - 'Qx': state stage cost, list of matrices (of length N) or single matrix (n_x,n_x)
                - 'Qn': terminal state cost matrix (n_x,n_x) [optional, defaults to Qx]
                - 'Ru': input stage cost, list of matrices (of length N) or single matrix (n_u,n_u)
                - 'x_ref': state reference, list of vectors (of length N) or single vector (n_x,1) [optional, defaults to 0]
                - 'u_ref': reference input, list of vectors (of length N) or single vector (n_u,1) [optional, defaults to 0]
                - 's_lin': linear penalty on slack variables, nonnegative scalar [optional, defaults to 0]
                - 's_quad': quadratic penalty on slack variables, positive scalar [optional, defaults to 0]

              where the stage cost is given by
              
              (x[t]-x_ref[t])'Qx[t](x[t]-x_ref[t]) + (u[t]-u_ref[t])'Ru[t](u[t]-u_ref[t]) + s_lin*e[t] + s_quad*e[t]**2
              
              Note: Qn is disregarded if Qx is a list.
            
            * cst: constraints definition, dictionary with keys
            
                - 'Hx': list of matrices (length N) or single matrix (=,n_x)
                - 'hx': list of vectors (length N) or single vector (=,1)
                - 'Hx_e': list of matrices (length N) or single matrix (=,n_eps) [optional, defaults to zero]
                - 'Hu': list of matrices (length N) or single matrix (-,n_u)
                - 'hu': list of vectors (length N) or single vector (-,1)
              
                where the constraints at each time-step are
              
                    Hx[t]*x[t] <= hx[t] - Hx_e[t]*e[t],
                    Hu[t]*u[t] <= hu[t],
                    
                where e are the slack variables.

        """

        # set all attributes

        if N is not None:
            self.__set_N(N)

        if model is not None:
            self.__set_model(model)

        if cost is not None:
            self.__set_cost(cost)

        if cst is not None:
            self.__set_cst(cst)

        # check that dimensions match
        self.__checkDimensions()
        
    def __checkDimensions(self):

        """
        This function checks if the dimensions of the properties: dynamics, cost, csts are consistent.
        If not, it throws an error. This function is called automatically whenever dynamics, cost, csts
        are updated. It also updates the dimension dictionary.
        """

        # get all nonempty attributes
        nonempty_attr = [v for v in self.allowed_inputs if getattr(self,v)]

        # loop through nonempty attributes
        for v in nonempty_attr:
            
            # loop through keys within each attribute
            for w in self.__allowed_keys[v]:

                # extract what dimensions are expected
                expected_dim = self.__expected_dimensions[v][w]

                # check correctness of dimensions
                for dim, val in zip(expected_dim,getattr(self,v)[w].shape):

                    # val should be a list of length N
                    if isinstance(val,list):
                        if len(val) != self.N:
                            raise Exception('Attribute {} must have a list of length N.'.format(v))
                    else:
                        raise Exception('Attribute {} must have a list of length N.'.format(v))

                    # check if dimension should be one
                    if dim == 'one':
                        if all([v != 1 for v in val]):
                            raise Exception('Attribute {} must have a scalar value.'.format(v))

                    # check if dimension is present
                    elif dim not in self.dim:
                        
                        # if not, add it
                        self.__add_to_dim({dim:val})

                    # otherwise, check if it matches existing dimensions
                    else:
                        if all([v != val for v in self.dim[dim]]):
                            raise Exception('Attribute {} must have the right dimensions.'.format(v))
    
    @property
    def N(self):
        return self.__N
    
    def __set_N(self, value):

        # convert to int
        try:
            value = int(value)
        except:
            raise Exception('Conversion failed, N must be an integer.')

        # check if value is positive
        if value <= 0:
            raise Exception('N must be an positive integer.')
        
        # assign
        self.__N = value

    @property
    def dim(self):
        return self.__dim
    
    def __add_to_dim(self, value):

        # check if value is a dictionary
        if not isinstance(self.__cost, dict):
            raise Exception('Dimensions must be passed as a dictionary.')
        
        # remove keys that are not allowed
        value = {k:v for k,v in value.items() if k in self.__allowed_keys['dim']}

        # update options dictionary
        self.__dim = self.__dim | value

    @property
    def model(self):
        return self.__model

    def __MPC_to_sparse_QP(self,A_list,B_list,c_list,Qx,Qn,Ru,Hx,hx,Hu,hu,Hx_e=None,x_ref=None,u_ref=None,s_lin=None,s_quad=None):

        """
        INTERNAL FUNCTION, NOT TO BE CALLED BY THE USER.

        This function takes in the ingredients of a problem of the form

        minimize      1/2 (x_N-x_r_N)^T Q_n (x_N-x_r_N)
         x,u,e      + 1/2 \sum_{t=0}^{N-1} [ (x_t-x_r_t)^T Q_x (x_t-x_r_t) + (u_t-u_r_t)^T R_u (u_t-u_r_t) ]
                    + 1/2 \sum_{t=0}^{N} [ c_lin e_t + c_quad e_t^T e_t ]

        subject to  x_{t+1} = A_t x_t + B_t u_t + c_t,  t = 0,...,N-1,
                    H_x x_t <= h_x - H_e e_t,           t = 0,...,N,
                    H_u u_t <= h_u,                     t = 0,...,N-1,
                    e_t >= 0,                           t = 0,...,N,
                    x_0 = x,
        
        where x_t,u_t denote state and input at time t, N is the horizon of the MPC, e_t is a slack variable (which only affects certain)
        states, as encoded in matrix H_e, and x is the current state of the system. The output are the ingredients of a QP of the form

        minimize    1/2 y'Qy + q'y
            
        subject to  Gy <= g
                    Fy = f

        where y = col(x,u,e).

        More specifically, the inputs are:

            - A_list, B_list, c_list: lists of matrices A, B, and c such that x[t+1] = A[t]@x[t] + B[t]@u[t] + c[t]
            - Qx, Qn, Ru, x_ref, u_ref: matrices defining the cost function (x-x_ref)'blkdiag(Qx,Qn)x(x-x_ref) + (u-u_ref)'Ru(u-u_ref)
            - Hx, Hu, hx, hu, Hx_e: polyhedral constraints Hx*x + Hx_e*e <= hx, Hu*u <= hu, where e are the slack variables
            - c_lin, c_quad: linear and quadratic penalties on slack (default is 0 and 1 respectively)
        
        Note that if Hx_e is not passed, but the slack option is enabled, it is set to the identity matrix.

        And the outputs are the matrices G, g, F, f, Q, Qinv=inv(Q), and the dictionary idx containing the indexing of the output optimization
        variables of the QP. This function sets up the following keys in idx:

            - 'u': range of all inputs
            - 'x': range of all states
            - 'y': range of all state-input variables
            - 'eps': range of all slack variables (if present)
            - 'u0': range of first input
            - 'u1': range of second input
            - 'x_shift': states shifted by one time-step (last state repeated)
            - 'u_shift': inputs shifted by one time-step (last input repeated)
            - 'y_shift': concatenation of x_shift and u_shift (and slacks shifted if present)

        """

        # get symbolic variable type
        MSX = self.__MSX

        # initialize slack to False
        slack = False
        
        # check if linear slack penalty is passed
        if s_lin is not None:

            # if penalty is greater than 0, set slack mode to true
            if self.cost['s_lin'] > 0:
                slack = True
            
            # if penalty is negative, raise exception
            elif self.cost['s_lin'] <= 0:
                raise Exception('Linear slack penalty must be positive.')
            
        # check if quadratic slack penalty is passed
        if s_quad is not None:

            # if penalty is greater than 0, set slack mode to true
            if self.cost['s_quad'] > 0:
                slack = True

            # if penalty is negative, raise exception
            elif self.cost['s_quad'] <= 0:
                raise Exception('Quadratic slack penalty must be nonnegative.')

        # check if Hx_e was passed 
        if Hx_e is not None:
            slack = True
            Hx_e = MSX(Hx_e)

        # if slack is passed, then ensure Hx_e is not None
        if slack and (Hx_e is None):
            Hx_e = MSX.eye(Hx.shape[0])

        # add slack dimension to dimension vector
        self.__add_to_dim({'eps':Hx_e.shape[1]})

        # extract dimensions
        n = self.dim

        if slack:

            # linear penalty
            try:
                s_lin = MSX(self.mpc.cost['s_lin'])
            except:
                s_lin = MSX(0) # default value
                pass

            # quadratic penalty
            try:
                s_quad = MSX(self.mpc.cost['s_quad'])
            except:
                print('Quadratic slack penalty not provided, defaulting to 1.')
                s_quad = MSX(1) # default value
                pass

            # check dimensions
            if s_lin.shape[0] != 1:
                raise Exception('Linear slack penalty must be a scalar.')
            if s_quad.shape[0] != 1:
                raise Exception('Quadratic slack penalty must be a scalar.')

            # add columns associated to input and slack variables
            Hx = hcat([Hx,MSX(Hx.shape[0],n['N']*n['u']),-Hx_e])

            # add columns associated to state and slack variables
            Hu = hcat([MSX(Hu.shape[0],n['N']*n['x']),Hu,MSX(Hu.shape[0],n['eps'])])

            # add nonnegativity constraints on slack variables
            He = hcat([MSX(n['eps'],n['N']*(n['x']+n['u'])),-MSX.eye(n['eps'])])
            he = MSX(n['eps'],1)

            # create inequality constraint matrices
            try:
                G = cse(sparsify(vcat([Hx,Hu,He])))
                g = cse(sparsify(vcat([hx,hu,he])))
            except:
                G = vcat([Hx,Hu,He])
                g = vcat([hx,hu,he])

        else:

            # add columns associated to input and slack variables
            Hx = hcat([Hx,MSX(Hx.shape[0],n['N']*n['u'])])

            # add columns associated to state and slack variables
            Hu = hcat([MSX(Hu.shape[0],n['N']*n['x']),Hu])
                
            # create inequality constraint matrices
            try:
                G = cse(sparsify(vcat([Hx,Hu])))
                g = cse(sparsify(vcat([hx,hu])))
            except:
                G = vcat([Hx,Hu])
                g = vcat([hx,hu])

        
        ### CREATE EQUALITY CONSTRAINTS ------------------------------------

        # preallocate equality constraint matrices
        F = MSX(n['N']*n['x'],n['N']*(n['x']+n['u'])+n['eps'])
        f = MSX(n['N']*n['x'],1)

        # construct matrix
        for i in range(n['N']):
        
            # negative identity for next state
            F[i*n['x']:(i+1)*n['x'],i*n['x']:(i+1)*n['x']] = -MSX.eye(n['x'])

            # A matrix multiplying current state
            if i > 0:
                F[i*n['x']:(i+1)*n['x'],(i-1)*n['x']:i*n['x']] = A_list[i]

            # B matrix multiplying current input
            F[i*n['x']:(i+1)*n['x'],n['N']*n['x']+i*n['u']:n['N']*n['x']+(i+1)*n['u']] = B_list[i]

            # affine term 
            f[i*n['x']:(i+1)*n['x']] = c_list[i]

        # sparsify F and f
        try:
            F = cse(sparsify(F))
            f = cse(sparsify(f))
        except:
            pass


        ### CREATE COST -----------------------------------------------------

        # construct state cost by stacking Qx and Qn
        Q = blockcat(Qx,MSX((n['N']-1)*n['x'],n['x']),MSX(n['x'],(n['N']-1)*n['x']),Qn)

        # add input cost
        Q = blockcat(Q,MSX(n['N']*n['x'],n['N']*n['u']),MSX(n['N']*n['u'],n['N']*n['x']),Ru)

        # append cost applied to slack variable
        if slack:
            Q = blockcat(Q,MSX(Q.shape[0],n['eps']),MSX(n['eps'],Q.shape[0]),s_quad*MSX.eye(n['eps']))

        # inverse of quadratic cost matrix
        Qinv = inv_minor(Q) #inv(Q)

        # create linear part of the cost
        if slack:
            q = vcat([(-x_ref.T@blockcat(Qx,MSX(n['x']*(n['N']-1),n['x']),MSX(n['x'],n['x']*(n['N']-1)),Qn)).T,(-u_ref.T@Ru).T,s_lin*MSX.ones(n['eps'],1)])
        else:
            q = vcat([(-x_ref.T@blockcat(Qx,MSX(n['x']*(n['N']-1),n['x']),MSX(n['x'],n['x']*(n['N']-1)),Qn)).T,(-u_ref.T@Ru).T])

        # sparsify Q and q
        try:
            Q = cse(sparsify(Q))
            Qinv = cse(sparsify(Qinv))
            q = cse(sparsify(q))
        except:
            pass


        ### CREATE INDEX DICTIONARY ----------------------------------------

        # store output variable indices
        idx = dict()
        
        # range of all inputs
        idx['u'] = range(n['x']*n['N'],(n['x']+n['u'])*n['N'])
        
        # range of all states
        idx['x'] = range(0,n['x']*n['N'])
        
        # range of all state-input variables
        idx['y'] = range(0,(n['x']+n['u'])*n['N'])
        
        # range of all slack variables
        if slack:
            idx['eps'] = range((n['x']+n['u'])*n['N'],(n['x']+n['u'])*n['N']+n['eps'])
            idx_e = np.arange(n['eps']) + n['N'] * (n['x'] + n['u'])
            idx_e_shifted = np.hstack([idx_e[n['eps']:], idx_e[:n['eps']]])

        # first input
        idx['u0'] = range(n['x']*n['N'],n['x']*n['N']+n['u'])

        # second input
        idx['u1'] = range(n['x']*n['N']+n['u'],n['x']*n['N']+2*n['u'])

        # Generate indices for x and u in y
        idx_x = np.arange(n['N'] * n['x'])
        idx_u = np.arange(n['N'] * n['u']) + n['N'] * n['x']
        
        # Shift x and u indices
        idx_x_shifted = np.hstack([idx_x[n['x']:], idx_x[-n['x']:]])
        idx_u_shifted = np.hstack([idx_u[n['u']:], idx_u[-n['u']:]])
        
        # Combine the shifted indices
        if slack:
            idx_shifted = np.hstack([idx_x_shifted, idx_u_shifted, idx_e_shifted])
        else:
            idx_shifted = np.hstack([idx_x_shifted, idx_u_shifted])

        # create shifted indices
        idx['x_shift'] = idx_x_shifted
        idx['u_shift'] = idx_u_shifted
        idx['y_shift'] = idx_shifted

        return G,g,F,f,Q,Qinv,q,idx

    def generate_linear_MPC(self):

        # if user passed a custom affine model, use it
        if model is not None:
            
            # extract matrices
            A_mat = model['A']
            B_mat = model['B']
            if 'c' in model:
                c_mat = model['c']
            else:
                c_mat = MSX(n_x,1)

            # check dimensions
            if A_mat.shape[0] != n_x or A_mat.shape[1] != n_x:
                raise Exception('A must have as many rows and columns as x.')
            if B_mat.shape[0] != n_x or B_mat.shape[1] != n_u:
                raise Exception('B must have as many rows as x and as many columns as u.')
            if c_mat.shape[0] != n_x:
                raise Exception('c must have as many rows as x.')

            # stack in list
            A_list = [A_mat] * N
            B_list = [B_mat] * N
            c_list = [c_mat] * N

            # patch first entry
            c_list[0] = - A_mat@x

        pass

    # overwrite the __dir__ method
    def __dir__(self):
        return [attr for attr in super().__dir__() if not attr.startswith('_MPC__')]