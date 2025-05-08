import casadi as ca
from BPMPC.utils import matrixify
from BPMPC.dynamics import Dynamics
from copy import copy
import numpy as np
from BPMPC.options import Options

"""
TODO:
* add descriptions
* add update function
* create a separate function for checking dimensions?
* allow symb to contain list of variables? In this case the dimension must be provided manually perhaps
"""

class Ingredients:

    _REQUIRED_KEYS = ['A','B','Qx','Ru','Hx','hx','Hu','hu']

    _ALLOWED_KEYS = ['A','B','c',
                      'Qx','Ru','x_ref','u_ref','s_lin','s_quad',
                      'Hx','Hx_e','Hu','hx','hu']
    
    _EXPECTED_DIMENSIONS = {'x':['x'],
                             'A':['x','x'],'B':['x','u'],'c':['x','one'],
                             'Qx':['x','x'],'Ru':['u','u'],'x_ref':['x','one'],'u_ref':['u','one'],'s_lin':['one','one'],'s_quad':['one','one'],
                             'Hx':['cst_x','x'],'Hx_e':['cst_x','eps'],'Hu':['cst_u','u'],'hx':['cst_x','one'],'hu':['cst_u','one']}
    
    _ALL_DIMENSIONS = ['x','u','one','cst_x','eps','cst_u']

    _OPTIONS_ALLOWED_VALUES = {'linearization':['trajectory','initial_state'], 'slack':bool}
    
    _OPTIONS_DEFAULT_VALUES = {'linearization':'trajectory', 'slack':False}

    def __init__(self,N,dynamics,cost,constraints,options=None):

        if options is None:
            options = {}

        assert isinstance(dynamics,Dynamics), 'The system dynamics must be an instance of class Dynamics'
        assert isinstance(options,dict), 'Options must be passed as a dictionary'
        assert isinstance(N,int) and N>0, 'N must be a positive integer'

        # copy dynamics
        dynamics_copy = copy(dynamics)

        # generate options dictionary
        self._options = Options(self._OPTIONS_ALLOWED_VALUES,self._OPTIONS_DEFAULT_VALUES)

        # add user-specified options
        self._options.update(options)

        # check if user passed a special option for linearization
        used_linearization_method = dynamics_copy._linearize(horizon=N,method=self._options['linearization'])

        # retrieve prediction model
        model = dynamics_copy.model

        # store linearization method that was used
        self._options['linearization']  = used_linearization_method

        # retrieve symbolic variables in model
        self._sym = dynamics_copy._sym.copy(['x','y_lin','theta'])

        # add horizon
        self._sym.addDim('N',N)

        # add input dimensions
        self._sym.addDim('u',dynamics_copy._sym.dim['u'])

        # merge into dictionary
        data = model | cost | constraints

        # parse inputs
        processed_data = self._parseInputs(data)

        # check if dimensions are correct
        self._checkDimensions(processed_data)

        # check if slacks are passed correctly
        slack_true = self._checkSlack(processed_data)
        self._options['slack'],n_eps = slack_true

        # save slack dimension in symbolic variable
        self._sym.addDim('eps',n_eps)

        # create sparse QP
        self._sparse = self._makeSparseQP(processed_data)

        # create index
        self._idx = {'out': self._makeIdx()}

        # create dense QP if requested
        self._dense = self._makeDenseQP(processed_data)

        # create dual QP
        self._dual = self._makeDualQP()
    

    def update(self,Q=None,q=None,G=None,g=None,F=None,f=None):
        #TODO remember to remove the dense and to update the dual
        pass

    def _makeSparseQP(self,processed_data):

        # extract initial condition
        x = self.param['x']

        # extract model
        A_list = processed_data['A']
        B_list = processed_data['B']

        # get dimensions
        n_x = self.dim['x']
        n_u = self.dim['u']
        n_eps = self.dim['eps']
        N = self.dim['N']


        # check if affine term is present
        if 'c' in processed_data:
            c_list = processed_data['c']
        else:
            c_list = [ca.SX(n_x,1)]*N

        # patch first affine term
        c_list[0] = c_list[0] - A_list[0]@x

        # extract cost
        Qx_list = processed_data['Qx']
        Ru_list = processed_data['Ru']

        # check if reference is passed
        if 'x_ref' in processed_data:
            x_ref = ca.vcat(processed_data['x_ref'])
        else:
            x_ref = ca.SX(n_x*N,1)
        if 'u_ref' in processed_data:
            u_ref = ca.vcat(processed_data['u_ref'])
        else:
            u_ref = ca.SX(n_u*N,1)

        # extract constraints
        Hx_list = processed_data['Hx']
        Hu_list = processed_data['Hu']
        hx_list = processed_data['hx']
        hu_list = processed_data['hu']
        
        # convert cost into single matrix
        Qx = matrixify(Qx_list)
        Ru = matrixify(Ru_list)

        # convert constraints into single matrix / vector
        Hx = matrixify(Hx_list)
        Hu = matrixify(Hu_list)
        hx = ca.vcat(hx_list)
        hu = ca.vcat(hu_list)

        # check if Hx_e was passed 
        if self._options['slack']:

            # convert into a single matrix
            Hx_e = matrixify(processed_data['Hx_e'])

            # get slack dimension
            n_eps = self.dim['eps']

            # linear penalty
            s_lin = ca.vcat(processed_data['s_lin']) if 's_lin' in processed_data else ca.SX(int(n_eps*N))

            # quadratic penalty
            s_quad = ca.vcat(processed_data['s_quad'])

            # add columns associated to input and slack variables
            Hx_u = ca.hcat([Hx,ca.SX(Hx.shape[0],N*n_u),-Hx_e])

            # add columns associated to state and slack variables
            Hu_x = ca.hcat([ca.SX(Hu.shape[0],N*n_x),Hu,ca.SX(Hu.shape[0],n_eps)])

            # add nonnegativity constraints on slack variables
            He = ca.hcat([ca.SX(n_eps,N*(n_x+n_u)),-ca.SX.eye(n_eps)])
            he = ca.SX(n_eps,1)

            # create inequality constraint matrices
            G = ca.cse(ca.sparsify(ca.vcat([Hx_u,Hu_x,He])))
            g = ca.cse(ca.sparsify(ca.vcat([hx,hu,he])))

        else:

            # add columns associated to input and slack variables
            Hx_u = ca.hcat([Hx,ca.SX(Hx.shape[0],N*n_u)])

            # add columns associated to state and slack variables
            Hu_x = ca.hcat([ca.SX(Hu.shape[0],N*n_x),Hu])
                
            # create inequality constraint matrices
            G = ca.cse(ca.sparsify(ca.vcat([Hx_u,Hu_x])))
            g = ca.cse(ca.sparsify(ca.vcat([hx,hu])))

            
        ### CREATE EQUALITY CONSTRAINTS ------------------------------------

        # preallocate equality constraint matrices
        F = ca.SX(N*n_x,N*(n_x+n_u)+n_eps)
        f = ca.SX(N*n_x,1)

        # construct matrix
        for i in range(N):
        
            # negative identity for next state
            F[i*n_x:(i+1)*n_x,i*n_x:(i+1)*n_x] = -ca.SX.eye(n_x)

            # A matrix multiplying current state
            if i > 0:
                F[i*n_x:(i+1)*n_x,(i-1)*n_x:i*n_x] = A_list[i]

            # B matrix multiplying current input
            F[i*n_x:(i+1)*n_x,N*n_x+i*n_u:N*n_x+(i+1)*n_u] = B_list[i]

            # affine term 
            f[i*n_x:(i+1)*n_x] = c_list[i]

        # sparsify F and f
        F = ca.cse(ca.sparsify(F))
        f = ca.cse(ca.sparsify(f))


        ### CREATE COST -----------------------------------------------------

        # add input cost
        Q = ca.blockcat(Qx,ca.SX(N*n_x,N*n_u),ca.SX(N*n_u,N*n_x),Ru)

        # append cost applied to slack variable
        if self._options['slack']:
            Q = ca.blockcat(Q,ca.SX(Q.shape[0],n_eps),ca.SX(n_eps,Q.shape[0]),ca.diag(s_quad))

        # inverse of quadratic cost matrix
        Qinv = ca.inv_minor(Q)
        # Qinv = ca.inv(Q)

        # create linear part of the cost
        q = ca.vcat([-Qx@x_ref,-Ru@u_ref,s_lin]) if self._options['slack'] else ca.vcat([-Qx@x_ref,-Ru@u_ref])

        # sparsify Q and q
        Q = ca.cse(ca.sparsify(Q))
        Qinv = ca.cse(ca.sparsify(Qinv))
        q = ca.cse(ca.sparsify(q))

        # stack all constraints together to match CasADi's conic interface
        A = ca.cse(ca.sparsify(ca.vcat([G,F])))

        # equality constraints can be enforced by setting lba=uba
        uba = ca.cse(ca.sparsify(ca.vcat([g,f])))
        lba = ca.cse(ca.sparsify(ca.vcat([-ca.inf*ca.SX.ones(g.shape),f])))

        return {'G':G, 'g':g, 'F':F, 'f':f, 'Q':Q, 'Qinv':Qinv, 'q':q, 'A':A, 'uba':uba, 'lba':lba}

    def _makeDenseQP(self,processed_data):

        # get horizon
        N = self.dim['N']

        # extract initial condition
        x = self.param['x']

        # extract model
        A_list = processed_data['A']
        B_list = processed_data['B']

        # get x and u dimensions
        n_x = self.dim['x']
        n_u = self.dim['u']

        # check if affine term is present
        if 'c' in processed_data:
            c_list = processed_data['c']
        else:
            c_list = [ca.SX(n_x,1)]*N

        # start by constructing matrices G_x and G_u, and vector g_c such that
        # x = G_x*x_0 + G_u*u + g_c, where x = vec(x_1,x_2,...,x_N), 
        # u = vec(u_0,u_1,...,u_N-1).
        # To obtain g_c we need to multiply all the affine terms by a matrix
        # similar to G_u, which we call G_c.
        
        # first initialize G_u with the zero matrix
        G_u = ca.SX(n_x*N,n_u*N)

        # initialize G_c
        G_c = ca.SX(n_x*N,n_x*N)

        # we will need a tall matrix that will replace the columns of G_u
        # initially it is equal to a tall matrix full of zeros with an
        # identity matrix at the bottom.
        col = ca.SX.eye(N*n_x)[:,(N-1)*n_x:N*n_x]

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
            col = col@A_t + ca.SX.eye(N*n_x)[:,(t-1)*n_x:t*n_x]

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
        c_t = -ca.vcat(c_list)
        g_c = G_c@c_t

        # create dictionary
        return {'G_x':G_x,'G_u':G_u,'g_c':g_c}

    def _makeDualQP(self):

        # extract ingredients
        Qinv = self.sparse['Qinv']
        q = self.sparse['q']
        F = self.sparse['F']
        f = self.sparse['f']
        G = self.sparse['G']
        g = self.sparse['g']

        # define Hessian of dual
        H_11 = ca.cse(ca.sparsify(G@Qinv@G.T))
        H_12 = ca.cse(ca.sparsify(G@Qinv@F.T))
        H_21 = ca.cse(ca.sparsify(F@Qinv@G.T))
        H_22 = ca.cse(ca.sparsify(F@Qinv@F.T))
        H = ca.cse(ca.blockcat(H_11,H_12,H_21,H_22))

        # define linear term of dual
        h_1 = ca.cse(ca.sparsify(G@Qinv@q+g))
        h_2 = ca.cse(ca.sparsify(F@Qinv@q+f))
        h = ca.cse(ca.vcat([h_1,h_2]))

        return {'H':H, 'h':h}

    def _makeIdx(self):

        # extract dimensions
        n_x = self.dim['x']
        n_u = self.dim['u']
        N = self.dim['N']
        n_eps = self.dim['eps']

        # store output variable indices
        idx = dict()
        
        # range of all inputs
        idx['u'] = range(n_x*N,(n_x+n_u)*N)
        
        # range of all states
        idx['x'] = range(0,n_x*N)
        
        # range of all state-input variables
        idx['y'] = range(0,(n_x+n_u)*N)
        
        # range of all slack variables
        if n_eps > 0:
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
        if n_eps > 0:
            idx_shifted = np.hstack([idx_x_shifted, idx_u_shifted, idx_e_shifted])
        else:
            idx_shifted = np.hstack([idx_x_shifted, idx_u_shifted])

        # create shifted indices
        idx['x_shift'] = idx_x_shifted
        idx['u_shift'] = idx_u_shifted
        idx['y_shift'] = idx_shifted
        
        # if a linearization trajectory is used, add entry to idx
        if self._options['linearization'] == 'trajectory':
            idx['y_next'] = lambda time: idx['y']
        elif self._options['linearization'] == 'initial_state':
            idx['y_next'] = lambda time: idx['u1']

        return idx

    def _parseInputs(self, data):

        # get horizon
        N = self.dim['N']

        # get length of all elements
        N_list = [len(elem) for elem in data.values() if isinstance(elem,list)]

        # append N
        N_list.append(N)

        # check that all dimensions match
        assert len(N_list) > 0, 'Please either pass a list of qp ingredients or set horizon N.'
        assert N_list.count(N_list[0]) == len(N_list), 'Dimensions of passed ingredients do not match.'

        # turn all elements into list of appropriate dimension
        processed_data = data \
                         | {key : [ca.SX(val)]*N for key,val in data.items() if val is not None and not isinstance(val,list)} \
                         | {key : [ca.SX(val) for val in val_list] for key,val_list in data.items() if isinstance(val_list,list)}

        # check that required entries were passed
        assert all([processed_data[key] is not None for key in self._REQUIRED_KEYS] ), 'Some required inputs are missing.'

        # strip unwanted inputs
        processed_data_stripped = {key : val for key,val in processed_data.items() if key in self._ALLOWED_KEYS}

        return processed_data_stripped

    def _checkDimensions(self,data):
        """
        This function checks if the dimensions of the properties: dynamics, cost, csts are consistent.
        If not, it throws an error.
        """

        # initialize dictionary containing all allowed keys
        dimension_dict = {key:[] for key in self._ALL_DIMENSIONS}

        # specify 'one' entry
        dimension_dict['one'] = [1]

        # loop through nonempty attributes
        for key,val in data.items():

            # get expected dimensions
            expected_dimension = self._EXPECTED_DIMENSIONS[key]
            
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

    def _checkSlack(self,data):

        slack = False
        n_eps = 0

        if 'Hx_e' in data:
            assert 's_quad' in data and all([data['s_quad'] > 0]), 'Please pass a positive quadratic penalty.'
            slack = True
            n_eps = data['Hx_e'].shape[1]

        if 's_quad' in data or 's_lin' in data:
            assert 'Hx_e' in data, 'Please pass a matrix specifying slack constraints.'
            slack = True

        if 's_lin' in data:
            assert all(data['s_lin']>0), 'The linear slack penalty must be positive'
            slack = True

        return slack,n_eps

    @property
    def sparse(self):
        return self._sparse
    
    @property
    def dense(self):
        return self._dense
    
    @property
    def dual(self):
        return self._dual
    
    @property
    def idx(self):
        return self._idx
    
    @property
    def param(self):
        return self._sym.var
    
    @property
    def options(self):
        return self._options
    
    @property
    def dim(self):
        return self._sym.dim