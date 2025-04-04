from casadi import *
from utils import matrixify

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

    __required_keys = ['A','B','Qx','Ru','Hx','hx','Hu','hu']

    __allowed_keys = ['A','B','c',
                      'Qx','Ru','x_ref','u_ref','s_lin','s_quad',
                      'Hx','Hx_e','Hu','hx','hu']
    
    __expected_dimensions = {'A':['x','x'],'B':['x','u'],'c':['x','one'],
                             'Qx':['x','x'],'Ru':['u','u'],'x_ref':['x','one'],'u_ref':['u','one'],'s_lin':['one','one'],'s_quad':['one','one'],
                             'Hx':['cst_x','x'],'Hx_e':['cst_x','eps'],'Hu':['cst_u','u'],'hx':['cst_x','one'],'hu':['cst_u','one']}
    
    __all_dimensions = ['x','u','one','cst_x','eps','cst_u']


    def __init__(self):
        #TODO
        pass

    def update(self,Q=None,q=None,G=None,g=None,F=None,f=None):
        #TODO
        pass

    def __parseInputs(self, data, N):

        # get length of all elements
        N_list = [len(elem) for elem in data.values() if elem is not None]
        if N is not None:
            N_list.append(N)

        # check that all dimensions match
        assert N_list.count(N_list[0]) == len(N_list), 'Dimensions of passed ingredients do not match.'

        # turn all elements into list of appropriate dimension
        processed_data = data \
                         | {key : [self.__MSX(val)]*N for key,val in data.items() if val is not None and ~isinstance(val,list)} \
                         | {key : [self.__MSX(val) for val in val_list] for key,val_list in data.items() if isinstance(val_list,list)}

        # check that required entries were passed
        assert all([processed_data[key] for key in self.__required_keys] is not None), 'Some required inputs are missing.'

        # strip unwanted inputs
        processed_data_stripped = {key : val for key,val in processed_data if key in self.__allowed_keys}

        return processed_data_stripped,N_list[0]

    def __checkDimensions(self,data):
        """
        This function checks if the dimensions of the properties: dynamics, cost, csts are consistent.
        If not, it throws an error.
        """

        # initialize dictionary containing all allowed keys
        dimension_dict = {key:[] for key in self.__all_dimensions}

        # specify 'one' entry
        dimension_dict['one'] = 1

        # loop through nonempty attributes
        for key,val in data.items():

            # get expected dimensions
            expected_dimension = self.__expected_dimensions[key]
            
            # get dimensions of each element of list and turn to set
            actual_dimension = set([entry.shape for entry in val])

            # check that all elements have the same dimension
            assert len(set) == 1, 'Please provide a list where all elements have the same dimension.'

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

    def MPC2QP(self):

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

        # check that all ingredients are present
        if not all([v in self.__cost for v in ['Qx','Ru']]):
            raise Exception('Cost must be fully specified.')
        
        if not all([v in self.__cst for v in ['Hx','hx','Hu','hu']]):
            raise Exception('Constraints must be fully specified.')
        
        if not all([v in self.__model for v in ['A','B','c']]):
            raise Exception('Model must be fully specified.')
        
        # extract ingredients
        A_list = self.__model['A']
        B_list = self.__model['B']
        c_list = self.__model['c']
        Qx = self.__cost['Qx']
        Ru = self.__cost['Ru']
        if 'x_ref' in self.__cost:
            x_ref = self.__cost['x_ref']
        else:
            x_ref = None
        if 'u_ref' in self.__cost:
            u_ref = self.__cost['u_ref']
        else:
            u_ref = None
        Hx = self.__cst['Hx']
        Hu = self.__cst['Hu']
        hx = self.__cst['hx']
        hu = self.__cst['hu']
        if 'Hx_e' in self.__cst:
            Hx_e = self.__cst['Hx_e']
        else:
            Hx_e = None
        if 's_lin' in self.__cost:
            s_lin = self.__cost['s_lin']
        else:
            s_lin = None
        if 's_quad' in self.__cost:
            s_quad = self.__cost['s_quad']
        else:
            s_quad = None
        
        # get symbolic variable type
        MSX = self.__MSX

        # convert into matrices
        Qx = matrixify(Qx,MSX)
        Ru = matrixify(Ru,MSX)
        Hx = matrixify(Hx,MSX)
        Hu = matrixify(Hu,MSX)
        hx = vcat(hx)
        hu = vcat(hu)

        if x_ref is not None:
            x_ref = vcat(x_ref)
        if u_ref is not None:
            u_ref = vcat(u_ref)

        # initialize slack to False
        slack = False
        
        # check if linear slack penalty is passed
        if s_lin is not None:

            # check that all entries in s_lin are equal and turn s_lin to a scalar
            if len(set(s_lin)) !=1:
                raise Exception('Linear slack penalty must be a scalar.')
            s_lin = s_lin[0]

            # if penalty is greater than 0, set slack mode to true
            if s_lin > 0:
                slack = True
            
            # if penalty is negative, raise exception
            elif s_lin <= 0:
                raise Exception('Linear slack penalty must be positive.')
            
        # check if quadratic slack penalty is passed
        if s_quad is not None:

            # check that all entries in s_quad are equal and turn s_quad to a scalar
            if len(set(s_quad)) != 1:
                raise Exception('Quadratic slack penalty must be a scalar.')
            s_quad = s_quad[0]

            # if penalty is greater than 0, set slack mode to true
            if s_quad > 0:
                slack = True

            # if penalty is negative, raise exception
            elif s_quad <= 0:
                raise Exception('Quadratic slack penalty must be nonnegative.')

        # check if Hx_e was passed 
        if Hx_e is not None:
            slack = True
            Hx_e = matrixify(Hx_e,MSX)
            Hx_e = MSX(Hx_e)

        # if slack is passed, then ensure Hx_e is not None
        if slack and (Hx_e is None):
            Hx_e = MSX.eye(Hx.shape[0])

        # add slack dimension to dimension vector
        if slack:
            self.__add_to_dim({'eps':Hx_e.shape[1]})
        else:
            self.__add_to_dim({'eps':0})

        # extract dimensions
        n = self.dim

        if slack:

            # linear penalty
            try:
                s_lin = MSX(s_lin)
            except:
                s_lin = MSX(0) # default value
                pass

            # quadratic penalty
            try:
                s_quad = MSX(s_quad)
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
            Hx_u = hcat([Hx,MSX(Hx.shape[0],n['N']*n['u']),-Hx_e])

            # add columns associated to state and slack variables
            Hu_x = hcat([MSX(Hu.shape[0],n['N']*n['x']),Hu,MSX(Hu.shape[0],n['eps'])])

            # add nonnegativity constraints on slack variables
            He = hcat([MSX(n['eps'],n['N']*(n['x']+n['u'])),-MSX.eye(n['eps'])])
            he = MSX(n['eps'],1)

            # create inequality constraint matrices
            try:
                G = cse(sparsify(vcat([Hx_u,Hu_x,He])))
                g = cse(sparsify(vcat([hx,hu,he])))
            except:
                G = vcat([Hx_u,Hu_x,He])
                g = vcat([hx,hu,he])

        else:

            # add columns associated to input and slack variables
            Hx_u = hcat([Hx,MSX(Hx.shape[0],n['N']*n['u'])])

            # add columns associated to state and slack variables
            Hu_x = hcat([MSX(Hu.shape[0],n['N']*n['x']),Hu])
                
            # create inequality constraint matrices
            try:
                G = cse(sparsify(vcat([Hx_u,Hu_x])))
                g = cse(sparsify(vcat([hx,hu])))
            except:
                G = vcat([Hx_u,Hu_x])
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
        Q = Qx

        # add input cost
        Q = blockcat(Q,MSX(n['N']*n['x'],n['N']*n['u']),MSX(n['N']*n['u'],n['N']*n['x']),Ru)

        # append cost applied to slack variable
        if slack:
            Q = blockcat(Q,MSX(Q.shape[0],n['eps']),MSX(n['eps'],Q.shape[0]),s_quad*MSX.eye(n['eps']))

        # inverse of quadratic cost matrix
        Qinv = inv_minor(Q)
        # Qinv = inv(Q)

        # create linear part of the cost
        if slack:
            q = vcat([-Qx@x_ref,-Ru@u_ref,s_lin*MSX.ones(n['eps'],1)])
        else:
            q = vcat([-Qx@x_ref,-Ru@u_ref])

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

        # create dense QP
        denseQP = self.__makeDenseMPC(A_list,B_list,c_list,Qx,Ru,x_ref,u_ref,Hx,Hu,hx,hu)

        return G,g,F,f,Q,Qinv,q,idx,denseQP

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