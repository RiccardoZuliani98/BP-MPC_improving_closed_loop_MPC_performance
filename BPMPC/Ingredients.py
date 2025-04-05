from casadi import *
from BPMPC.utils import matrixify

class Ingredients:
    """
    QP ingredients, contains the following keys

        - 'Q': Hessian of QP (n_y,n_y)
        - 'Qinv': inverse of Hessian of QP (n_y,n_y)
        - 'q': gradient of cost of QP (n_y,1)
        - 'G': linear inequality constraint matrix (n_in,n_y)
        - 'g': linear inequality constraint vector (n_in,1)
        - 'H': Hessian of dual problem (n_z,n_z)
        - 'h': gradient of cost of dual problem (n_z,1)
        - 'A': stacked inequality constraint matrix for casadi's conic interface,
               specifically, A = vertcat(G,F) (n_in+n_eq,n_y)
        - 'lba': lower bound of inequality constraints, lba = (-inf,f) (n_in+n_eq,1)
        - 'uba': upper bound of inequality constraints, uba = (g,f) (n_in+n_eq,1)

    remember that the primal problem is a QP with the following structure:

        min 1/2 y'Qy + q'y
        s.t. Gy <= g
             Fy = y

    and the dual problem is a QP with the following structure:

        min 1/2 z'Hz + h'z
        s.t. z=(lam,mu)
             lam >= 0
    """

    __required_keys = ['A','B','Qx','Ru','Hx','hx','Hu','hu','x']

    __allowed_keys = ['A','B','c',
                      'Qx','Ru','x_ref','u_ref','s_lin','s_quad',
                      'Hx','Hx_e','Hu','hx','hu']
    
    __expected_dimensions = {'x':['x'],
                             'A':['x','x'],'B':['x','u'],'c':['x','one'],
                             'Qx':['x','x'],'Ru':['u','u'],'x_ref':['x','one'],'u_ref':['u','one'],'s_lin':['one','one'],'s_quad':['one','one'],
                             'Hx':['cst_x','x'],'Hx_e':['cst_x','eps'],'Hu':['cst_u','u'],'hx':['cst_x','one'],'hu':['cst_u','one']}
    
    __all_dimensions = ['x','u','one','cst_x','eps','cst_u']


    def __init__(self,data,MSX='SX'):

        """
        This function converts an MPC problem of the form

        minimize      1/2 (x_N-x_r_N)^T Qx[t] (x_N-x_r_N)
         x,u,e      + 1/2 \sum_{t=0}^{N-1} [ (x_t-x_r_t)^T Qx[t] (x_t-x_r_t) + (u_t-u_r_t)^T Ru[t] (u_t-u_r_t) ]
                    + 1/2 \sum_{t=0}^{N} [ c_lin e_t + c_quad e_t^T e_t ]

        subject to  x_{t+1} = A_t x_t + B_t u_t + c_t,  t = 0,...,N-1,
                    H_x x_t <= h_x - H_e e_t,           t = 0,...,N,
                    H_u u_t <= h_u,                     t = 0,...,N-1,
                    e_t >= 0,                           t = 0,...,N,
                    x_0 = x,
        
        where x_t,u_t denote state and input at time t, N is the horizon of the MPC, e_t is a slack variable (which only affects certain)
        states, as encoded in matrix H_e, and x is the current state of the system, to a QP of the form

        minimize    1/2 y'Qy + q'y
            
        subject to  Gy <= g
                    Fy = f

        where y = col(x,u,e). The outputs are the matrices G, g, F, f, Q, Qinv=inv(Q), the dictionary idx containing the indexing of the output optimization variables of the QP. This function sets up the following keys in idx:

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

        # set symbolic variable type
        self.__MSX = SX if MSX is 'SX' else MX
        MSX = self.__MSX

        # parse inputs
        processed_data,N = self.__parseInputs(data)

        # check if dimensions are correct
        self.__checkDimensions(processed_data)

        # check if slacks are passed correctly
        slack = self.__checkSlack(processed_data)
        
        # extract model
        A_list = processed_data['A']
        B_list = processed_data['B']

        # extract initial condition
        x = data['x']

        # get dimensions
        N = len(A_list)
        n_x,n_u = B_list[0].shape

        # check if affine term is present
        if 'c' in processed_data:
            c_list = processed_data['c']
        else:
            c_list = [MSX(n_x,1)]*N

        # patch first affine term
        c_list[0] = c_list[0] - A_list[0]@x

        # extract cost
        Qx_list = processed_data['Qx']
        Ru_list = processed_data['Ru']

        # check if reference is passed
        if 'x_ref' in processed_data:
            x_ref = vcat(processed_data['x_ref'])
        else:
            x_ref = MSX(n_x*N,1)
        if 'u_ref' in processed_data:
            u_ref = vcat(processed_data['u_ref'])
        else:
            u_ref = MSX(n_u*N,1)

        # extract constraints
        Hx_list = processed_data['Hx']
        Hu_list = processed_data['Hu']
        hx_list = processed_data['hx']
        hu_list = processed_data['hu']
        
        # convert cost into single matrix
        Qx = matrixify(Qx_list,MSX)
        Ru = matrixify(Ru_list,MSX)

        # convert constraints into single matrix / vector
        Hx = matrixify(Hx_list,MSX)
        Hu = matrixify(Hu_list,MSX)
        hx = vcat(hx_list)
        hu = vcat(hu_list)

        # check if Hx_e was passed 
        if 'Hx_e' in processed_data:

            # set slack to true
            slack = True

            # convert into a single matrix
            Hx_e = matrixify(processed_data['Hx_e'],MSX)

            # get slack dimension
            n_eps = Hx_e.shape[1]

            # linear penalty
            s_lin = vcat(processed_data['s_lin']) if 's_lin' in processed_data else MSX(int(n_eps*N))

            # quadratic penalty
            s_quad = vcat(processed_data['s_quad'])

            # add columns associated to input and slack variables
            Hx_u = hcat([Hx,MSX(Hx.shape[0],N*n_u),-Hx_e])

            # add columns associated to state and slack variables
            Hu_x = hcat([MSX(Hu.shape[0],N*n_x),Hu,MSX(Hu.shape[0],n_eps)])

            # add nonnegativity constraints on slack variables
            He = hcat([MSX(n_eps,N*(n_x+n_u)),-MSX.eye(n_eps)])
            he = MSX(n_eps,1)

            # create inequality constraint matrices
            try:
                G = cse(sparsify(vcat([Hx_u,Hu_x,He])))
                g = cse(sparsify(vcat([hx,hu,he])))
            except:
                G = vcat([Hx_u,Hu_x,He])
                g = vcat([hx,hu,he])

        else:

            # no slacks
            n_eps = 0

            # add columns associated to input and slack variables
            Hx_u = hcat([Hx,MSX(Hx.shape[0],N*n_u)])

            # add columns associated to state and slack variables
            Hu_x = hcat([MSX(Hu.shape[0],N*n_x),Hu])
                
            # create inequality constraint matrices
            try:
                G = cse(sparsify(vcat([Hx_u,Hu_x])))
                g = cse(sparsify(vcat([hx,hu])))
            except:
                G = vcat([Hx_u,Hu_x])
                g = vcat([hx,hu])

        
        ### CREATE EQUALITY CONSTRAINTS ------------------------------------

        # preallocate equality constraint matrices
        F = MSX(N*n_x,N*(n_x+n_u)+n_eps)
        f = MSX(N*n_x,1)

        # construct matrix
        for i in range(N):
        
            # negative identity for next state
            F[i*n_x:(i+1)*n_x,i*n_x:(i+1)*n_x] = -MSX.eye(n_x)

            # A matrix multiplying current state
            if i > 0:
                F[i*n_x:(i+1)*n_x,(i-1)*n_x:i*n_x] = A_list[i]

            # B matrix multiplying current input
            F[i*n_x:(i+1)*n_x,N*n_x+i*n_u:N*n_x+(i+1)*n_u] = B_list[i]

            # affine term 
            f[i*n_x:(i+1)*n_x] = c_list[i]

        # sparsify F and f
        try:
            F = cse(sparsify(F))
            f = cse(sparsify(f))
        except:
            pass


        ### CREATE COST -----------------------------------------------------

        # add input cost
        Q = blockcat(Qx,MSX(N*n_x,N*n_u),MSX(N*n_u,N*n_x),Ru)

        # append cost applied to slack variable
        if slack:
            Q = blockcat(Q,MSX(Q.shape[0],n_eps),MSX(n_eps,Q.shape[0]),diag(s_quad))

        # inverse of quadratic cost matrix
        Qinv = inv_minor(Q)
        # Qinv = inv(Q)

        # create linear part of the cost
        q = vcat([-Qx@x_ref,-Ru@u_ref,s_lin]) if slack else vcat([-Qx@x_ref,-Ru@u_ref])

        # sparsify Q and q
        try:
            Q = cse(sparsify(Q))
            Qinv = cse(sparsify(Qinv))
            q = cse(sparsify(q))
        except:
            pass

        sparse = {'G':G, 'g':g, 'F':F, 'f':f, 'Q':Q, 'Qinv':Qinv, 'q':q}

        ### DENSE QP

        # start by constructing matrices G_x and G_u, and vector g_c such that
        # x = G_x*x_0 + G_u*u + g_c, where x = vec(x_1,x_2,...,x_N), 
        # u = vec(u_0,u_1,...,u_N-1).
        # To obtain g_c we need to multiply all the affine terms by a matrix
        # similar to G_u, which we call G_c.
        
        # first initialize G_u with the zero matrix
        G_u = MSX(n_x*N,n_u*N)

        # initialize G_c
        G_c = MSX(n_x*N,n_x*N)

        # we will need a tall matrix that will replace the columns of G_u
        # initially it is equal to a tall matrix full of zeros with an
        # identity matrix at the bottom.
        col = MSX.eye(N*n_x)[:,(N-1)*n_x:N*n_x]

        # loop through all columns of G_u (t ranges from N-1 to 1)
        for t in range(N-1,0,-1):

            # get matrices A and at time-step t
            A_t = A_list[t]
            B_t = B_list[t]

            # update G_u matrix
            G_u[:,t*n_u:(t+1)*n_u] = col@B_t

            # update G_c matrix
            G_c[:,t*n_x:(t+1)*n_x] = col

            # update col by multiplying with A matrix and adding identity matrix
            col = col@A_t + MSX.eye(N*n_x)[:,(t-1)*n_x:t*n_x]

        # get linearized dynamics at time-step 0
        A_0 = A_list[0]
        B_0 = B_list[0]

        # correct first entry of c_list
        c_list[0] = c_list[0] + A_0@x

        # now we only miss the left-most column (use x0 instead of x[:n['x']])
        G_u[:,:n_u] = col@B_0

        # same for G_c
        G_c[:,:n_x] = col

        # matrix G_x is simply col@A_0
        G_x = col@A_0

        # to create g_c concatenate vertically the entries in the list c_t_list
        # then multiply by G_c from the right
        c_t = -vcat(c_list)
        g_c = G_c@c_t

        # create dictionary
        dense = {'G_x':G_x,'G_u':G_u,'g_c':g_c,'Qx':Qx,'Ru':Ru,'x_ref':x_ref,'u_ref':u_ref,'Hx':Hx,'Hu':Hu,'hx':hx,'hu':hu}


        ### CREATE INDEX DICTIONARY ----------------------------------------

        # store output variable indices
        idx = dict()
        
        # range of all inputs
        idx['u'] = range(n_x*N,(n_x+n_u)*N)
        
        # range of all states
        idx['x'] = range(0,n_x*N)
        
        # range of all state-input variables
        idx['y'] = range(0,(n_x+n_u)*N)
        
        # range of all slack variables
        if slack:
            idx['eps'] = range((n_x+n_u)*N,(n_x+n_u)*N+n_eps)
            idx_e = np.arange(n_eps) + N * (n_x + n_u)
            idx_e_shifted = np.hstack([idx_e[n_eps:], idx_e[:n_eps]])

        # first input
        idx['u0'] = range(n_x*N,n_x*N+n_u)

        # second input
        idx['u1'] = range(n_x*N+n_u,n_x*N+2*n_u)

        # Generate indices for x and u in y
        idx_x = np.arange(N * n_x)
        idx_u = np.arange(N * n_u) + N * n_x
        
        # Shift x and u indices
        idx_x_shifted = np.hstack([idx_x[n_x:], idx_x[-n_x:]])
        idx_u_shifted = np.hstack([idx_u[n_u:], idx_u[-n_u:]])
        
        # Combine the shifted indices
        if slack:
            idx_shifted = np.hstack([idx_x_shifted, idx_u_shifted, idx_e_shifted])
        else:
            idx_shifted = np.hstack([idx_x_shifted, idx_u_shifted])

        # create shifted indices
        idx['x_shift'] = idx_x_shifted
        idx['u_shift'] = idx_u_shifted
        idx['y_shift'] = idx_shifted

        # store
        self.sparse = sparse
        self.dense = dense
        self.idx = idx

    def update(self,Q=None,q=None,G=None,g=None,F=None,f=None):
        #TODO
        pass

    def __parseInputs(self, data):

        # get length of all elements
        N_list = [len(elem) for elem in data.values() if isinstance(elem,list)]

        # check if N was passed
        if 'N' in data:
            N_list.append(data['N'])

        # check that all dimensions match
        assert len(N_list) > 0, 'Please either pass a list of qp ingredients or set horizon N.'
        assert N_list.count(N_list[0]) == len(N_list), 'Dimensions of passed ingredients do not match.'

        # get dimension
        N = N_list[0]

        # for key,val in data.items():
        #     if val is not None and not isinstance(val,list):
        #         pass

        # turn all elements into list of appropriate dimension
        processed_data = data \
                         | {key : [self.__MSX(val)]*N for key,val in data.items() if val is not None and not isinstance(val,list)} \
                         | {key : [self.__MSX(val) for val in val_list] for key,val_list in data.items() if isinstance(val_list,list)}

        # check that required entries were passed
        assert all([processed_data[key] is not None for key in self.__required_keys] ), 'Some required inputs are missing.'

        # strip unwanted inputs
        processed_data_stripped = {key : val for key,val in processed_data.items() if key in self.__allowed_keys}

        return processed_data_stripped,N_list[0]

    def __checkDimensions(self,data):
        """
        This function checks if the dimensions of the properties: dynamics, cost, csts are consistent.
        If not, it throws an error.
        """

        # initialize dictionary containing all allowed keys
        dimension_dict = {key:[] for key in self.__all_dimensions}

        # specify 'one' entry
        dimension_dict['one'] = [1]

        # loop through nonempty attributes
        for key,val in data.items():

            # get expected dimensions
            expected_dimension = self.__expected_dimensions[key]
            
            # get dimensions of each element of list and turn to set
            actual_dimension_set = set([entry.shape for entry in val])

            # check that all elements have the same dimension
            assert len(actual_dimension_set) == 1, 'Please provide a list where all elements have the same dimension.'

            actual_dimension = actual_dimension_set.pop()

            # append to the list stored in dimension_dict
            dimension_dict[expected_dimension[0]].append(actual_dimension[0])
            dimension_dict[expected_dimension[1]].append(actual_dimension[1])

        # strip empty lists
        dimension_dict_stripped = {key:val for key,val in dimension_dict.items() if len(val)>0}

        # now check that all dimensions match
        for key,val in dimension_dict_stripped.items():
            assert val.count(val[0]) == len(val), 'Wrong dimension detected for key: ' + key

    def __checkSlack(self,data):

        if 'Hx_e' in data:
            assert 's_quad' in data and all([data['s_quad'] > 0]), 'Please pass a positive quadratic penalty.'

        if 's_quad' in data or 's_lin' in data:
            assert 'Hx_e' in data, 'Please pass a matrix specifying slack constraints.'

        if 's_lin' in data:
            assert all(data['s_lin']>0), 'The linear slack penalty must be positive'

    @property
    def G(self):
        return self.__G
    
    @property
    def g(self):
        return self.__g
    
    @property
    def F(self):
        return self.__F
    
    @property
    def f(self):
        return self.__f
    
    @property
    def Q(self):
        return self.__Q
    
    @property
    def q(self):
        return self.__q