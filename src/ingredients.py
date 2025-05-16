import casadi as ca
from src.utils import matrixify
from src.dynamics import Dynamics
from copy import copy
import numpy as np
from src.options import Options
from typeguard import typechecked
from typing import Optional

class Ingredients:
    """
    Ingredients class for Model Predictive Control (MPC) problem formulation.
    This class encapsulates the construction and validation of all ingredients required to formulate
    a (possibly constrained) MPC problem, including system dynamics, cost, constraints, and optional
    configuration such as slack variables and linearization methods. It provides methods to assemble
    sparse, dense, and dual QP representations compatible with CasADi, and ensures all data is
    dimensionally consistent and properly structured.

    Attributes:
        _REQUIRED_KEYS (list): List of keys required in the input data for a valid MPC problem.
        _ALLOWED_KEYS (list): List of all allowed keys for model, cost, and constraint data.
        _EXPECTED_DIMENSIONS (dict): Dictionary mapping each key to its expected dimensions.
        _ALL_DIMENSIONS (list): List of all possible dimension names used in the problem.
        _OPTIONS_ALLOWED_VALUES (dict): Allowed values for configuration options.
        _OPTIONS_DEFAULT_VALUES (dict): Default values for configuration options.

    Methods:
        __init__(horizon, dynamics, cost, constraints, options=None):
            Initializes the Ingredients object, validates and processes all inputs, and constructs
            the symbolic and numeric representations of the MPC problem.
        update(Q=None, q=None, G=None, g=None, F=None, f=None):
            Placeholder for updating QP matrices after initialization.
        _makeSparseQP(processed_data):
            Constructs the sparse QP matrices (G, g, F, f, Q, Qinv, q, A, uba, lba) for the MPC problem.
        _makeDenseQP(processed_data):
            Constructs the dense (condensed) QP matrices (G_x, G_u, g_c, Hx, hx, Hu, hu, Qx, Ru, x_ref, u_ref).
        _makeDualQP():
            Constructs the dual QP matrices (H, h) from the sparse QP representation.
        _makeIdx():
            Generates and returns a dictionary of index ranges for state, input, slack, and combined variables.
        _parseInputs(data):
            Parses and processes input data, ensuring correct dimensions and presence of required keys.
        _checkDimensions(data):
            Checks the consistency of dimensions for all model, cost, and constraint data.
        _checkSlack(data):
            Checks and validates the presence and configuration of slack variables and their penalties.
    
    Properties:
        sparse: Returns the dictionary of sparse QP matrices.
        dense: Returns the dictionary of dense QP matrices.
        dual: Returns the dictionary of dual QP matrices.
        idx: Returns the dictionary of index ranges for variables.
        param: Returns the symbolic parameter variables.
        options: Returns the options dictionary.
        dim: Returns the dictionary of variable dimensions.
        - The class is designed for use with CasADi symbolic expressions and supports both standard and
          slack-variable-augmented MPC formulations.
        - All input data is validated for dimensional consistency and completeness.
        - The class supports different linearization strategies for the system dynamics.
    """

    _REQUIRED_KEYS = ['A','B','Qx','Ru','Hx','hx','Hu','hu']

    _ALLOWED_KEYS = ['A','B','c',
                      'Qx','Ru','x_ref','u_ref','s_lin','s_quad',
                      'Hx','Hx_e','Hu','hx','hu']
    
    _EXPECTED_DIMENSIONS = {'x':['x'],
                             'A':['x','x'],'B':['x','u'],'c':['x','one'],
                             'Qx':['x','x'],'Ru':['u','u'],'x_ref':['x','one'],'u_ref':['u','one'],'s_lin':['one','one'],'s_quad':['one','one'],
                             'Hx':['cst_x','x'],'Hx_e':['cst_x','eps'],'Hu':['cst_u','u'],'hx':['cst_x','one'],'hu':['cst_u','one']}
    
    _ALL_DIMENSIONS = ['x','u','one','cst_x','eps','cst_u']

    _OPTIONS_ALLOWED_VALUES = {'linearization':['trajectory','initial_state','affine'], 'slack':bool}
    
    _OPTIONS_DEFAULT_VALUES = {'linearization':'trajectory', 'slack':False}

    @typechecked
    def __init__(
            self,
            horizon:int,
            dynamics:Dynamics,
            cost:dict,
            constraints:dict,
            options:Optional[dict]=None
        ) -> None:
        """
        Initializes the object with the specified prediction horizon, system dynamics, cost, constraints, and optional configuration options.
        Args:
            horizon (int): The prediction horizon. Must be a positive integer.
            dynamics (Dynamics): The system dynamics object to be used.
            cost (dict): Dictionary specifying the cost function parameters.
            constraints (dict): Dictionary specifying the constraints.
            options (Optional[dict], optional): Additional configuration options. Defaults to None.
        Raises:
            AssertionError: If the horizon is not a positive integer.
        Notes:
            - Copies the provided dynamics object to avoid side effects.
            - Initializes and updates options with user-specified values.
            - Performs model linearization based on the specified or default method.
            - Sets up symbolic variables and their dimensions.
            - Merges model, cost, and constraints into a single data structure.
            - Parses and checks input data for correctness, including dimensions and slack variables.
            - Constructs sparse, dense, and dual quadratic programming (QP) representations as needed.
        """

        # TODO: horizon should be optional as it can be inferred from the length of the passed ingredients

        if options is None:
            options = {}

        assert horizon > 0, 'horizon must be a positive integer'

        # copy dynamics
        dynamics_copy = copy(dynamics)

        # generate options dictionary
        self._options = Options(self._OPTIONS_ALLOWED_VALUES,self._OPTIONS_DEFAULT_VALUES)

        # add user-specified options
        self._options.update(options)

        # check if user passed a special option for linearization
        model, symbolic_vars, used_linearization_method = dynamics_copy._linearize(horizon=horizon,method=self._options['linearization'])

        # store linearization method that was used
        self._options['linearization']  = used_linearization_method

        # retrieve symbolic variables in model
        self._sym = symbolic_vars.copy(['x','y_lin'])

        # add horizon
        self._sym.add_dim('N',horizon)

        # add input dimensions
        self._sym.add_dim('u',symbolic_vars.dim['u'])

        # merge into dictionary
        data = model | cost | constraints

        # parse inputs
        processed_data = self._parse_inputs(data,horizon)

        # check if dimensions are correct
        self._check_dimensions(processed_data)

        # check if slacks are passed correctly
        slack_true = self._check_slack(processed_data)
        self._options['slack'],n_eps = slack_true

        # save slack dimension in symbolic variable
        self._sym.add_dim('eps',n_eps)

        # create sparse QP
        self._sparse = self._make_sparse_qp(processed_data)

        # create index
        self._idx = {'out': self._make_idx()}

        # create dense QP if requested
        self._dense = self._make_dense_qp(processed_data)

        # create dual QP
        self._dual = self._make_dual_qp()
    

    def update(self,Q=None,q=None,G=None,g=None,F=None,f=None):
        #TODO remember to remove the dense and to update the dual
        pass

    def _make_sparse_qp(self,processed_data:dict) -> dict:
        """
        Constructs the sparse Quadratic Program (QP) matrices for a Model Predictive Control (MPC) problem using 
        the provided processed data. This method extracts system dynamics, cost, and constraint information from
        the input `processed_data` and assembles the matrices required to define a sparse QP suitable for use
        with CasADi's conic solver interface. It supports both standard and slack-variable-augmented formulations.

        Parameters:
        
            processed_data (dict): Dictionary containing all necessary processed model data, including system 
                matrices (A, B, c), cost matrices (Qx, Ru), constraint matrices (Hx, Hu, hx, hu), references 
                (x_ref, u_ref), and optionally slack-related data (Hx_e, s_lin, s_quad).
        
        Returns: 
            dict: Dictionary containing the following keys:
                - 'G': Inequality constraint matrix (CasADi SX or sparse SX).
                - 'g': Inequality constraint vector.
                - 'F': Equality constraint matrix.
                - 'f': Equality constraint vector.
                - 'Q': Quadratic cost matrix.
                - 'Qinv': Pseudoinverse of the quadratic cost matrix.
                - 'q': Linear cost vector.
                - 'A': Combined constraint matrix for CasADi's conic interface.
                - 'uba': Upper bound vector for constraints.
                - 'lba': Lower bound vector for constraints.
        
        Notes:
            - The method supports both standard and slack-variable-augmented QP formulations, depending on the `self._options['slack']` flag.
            - All matrices are constructed to be compatible with CasADi's sparse matrix operations.
            - Equality constraints are enforced by setting `lba=uba` for the corresponding rows.
        """

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
        c_list = processed_data['c'] if 'c' in processed_data else [ca.SX(n_x,1)]*N

        # patch first affine term
        c_list[0] = c_list[0] + A_list[0]@x

        # extract cost
        Qx_list = processed_data['Qx']
        Ru_list = processed_data['Ru']

        # check if reference is passed
        x_ref = ca.vcat(processed_data['x_ref']) if 'x_ref' in processed_data else ca.SX(n_x*N,1)
        u_ref = ca.vcat(processed_data['u_ref']) if 'u_ref' in processed_data else ca.SX(n_u*N,1)

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
            f[i*n_x:(i+1)*n_x] = -c_list[i]

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
        Q = ca.sparsify(ca.cse(Q))
        Qinv = ca.pinv(Q)
        # Qinv = ca.inv_minor(Q)
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

    def _make_dense_qp(self,processed_data:dict) -> dict:
        """
        Constructs the dense QP (Quadratic Program) matrices for a given processed system model and parameters.
        This method builds the condensed system dynamics and cost matrices required for solving a dense QP 
        in the context of Model Predictive Control (MPC). It computes matrices G_x, G_u, and g_c such that 
        the predicted state trajectory x can be written as:
            x = G_x * x_0 + G_u * u + g_c
        where x is the stacked vector of predicted states, u is the stacked vector of control inputs, and x_0 
        is the initial state.

        Args:
            processed_data (dict): Dictionary containing the linearized system matrices and other problem data.
                Required keys:
                    - 'A': List of state transition matrices (A_t) for each time step.
                    - 'B': List of input matrices (B_t) for each time step.
                    - 'Hx', 'hx': State constraint matrices and vectors.
                    - 'Hu', 'hu': Input constraint matrices and vectors.
                    - 'Qx': State cost matrix.
                    - 'Ru': Input cost matrix.
                Optional keys:
                    - 'c': List of affine terms for each time step (default: zeros).
                    - 'x_ref': List of state reference vectors (optional).
                    - 'u_ref': List of input reference vectors (optional).
                    - 'Hx_e': Matrix specifying slack constraints (optional).
                    - 's_quad': Quadratic penalty for slack variables (optional).
                    - 's_lin': Linear penalty for slack variables (optional).
        
        Returns:
            dict: Dictionary containing the condensed QP matrices:
                - 'G_x': Matrix mapping initial state to predicted states.
                - 'G_u': Matrix mapping control inputs to predicted states.
                - 'g_c': Vector of affine terms in the predicted states.
                - 'Hx', 'hx': State constraint matrices and vectors.
                - 'Hu', 'hu': Input constraint matrices and vectors.
                - 'Qx': State cost matrix.
                - 'Ru': Input cost matrix.
                - 'x_ref': (optional) Stacked state reference vector.
                - 'u_ref': (optional) Stacked input reference vector.
        """

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
        c_list = processed_data['c'] if 'c' in processed_data else [ca.SX(n_x,1)]*N

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
        c_list[0] = c_list[0] - A_0@x

        # now we only miss the left-most column (use x0 instead of x[:n['x']])
        G_u[:,:n_u] = col@B_0

        # same for G_c
        G_c[:,:n_x] = col

        # matrix G_x is simply col@A_0
        G_x = col@A_0

        # to create g_c concatenate vertically the entries in the list c_t_list
        # then multiply by G_c from the right
        c_t = ca.vcat(c_list)
        g_c = G_c@c_t

        # create output dictionary
        out = {'G_x':G_x,'G_u':G_u,'g_c':g_c,
               'Hx':matrixify(processed_data['Hx']),'hx':ca.vcat(processed_data['hx']),
               'Hu':matrixify(processed_data['Hu']),'hu':ca.vcat(processed_data['hu']),
               'Qx':matrixify(processed_data['Qx']),'Ru':matrixify(processed_data['Ru'])}

        # add references if present
        if 'x_ref' in processed_data:
            out = out | {'x_ref':ca.vcat(processed_data['x_ref'])}
        if 'u_ref' in processed_data:
            out = out | {'u_ref': ca.vcat(processed_data['u_ref'])}

        return out

    def _make_dual_qp(self) -> dict:
        """
        Constructs the dual Quadratic Program (QP) ingredients for the optimization problem.
        This method extracts the necessary matrices and vectors from the `self.sparse` dictionary,
        computes the Hessian and linear term of the dual QP using CasADi operations, and returns
        them in a dictionary.

        Returns:
            dict: A dictionary containing:
                - 'H': The Hessian matrix of the dual QP (CasADi sparse matrix).
                - 'h': The linear term vector of the dual QP (CasADi vector).
        """

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

    def _make_idx(self) -> dict:
        """
        Generate and return a dictionary of index ranges for state, input, slack, and combined variables used 
        in the optimization problem. This method computes index ranges for:
            - All states (`x`)
            - All inputs (`u`)
            - All state-input variables (`y`)
            - Slack variables (`eps`), if present
            - The first and second input variables (`u0`, `u1`)
            - Shifted indices for states, inputs, and combined variables (`x_shift`, `u_shift`, `y_shift`)
            - Optionally, a function for the next step indices (`y_next`) depending on the linearization option
        Returns:
            dict: A dictionary containing index ranges and shifted indices for use in the optimization problem.
        """

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

    @staticmethod
    def _parse_inputs(data:dict,N:int=None) -> dict:
        """
        Parses and processes input data for the MPC ingredient setup.
        This method ensures that all input data elements have consistent dimensions
        with the specified prediction horizon `N`, converts scalar values to lists
        of CasADi SX symbols of length `N`, and validates the presence of required keys.
        It also strips out any keys not allowed by the class configuration.
        
        Args:
            data (dict): Dictionary containing input data for the MPC problem. Values can be
                scalars or lists. Keys must correspond to allowed/required ingredient names.
            N (int,optional): The prediction horizon. Must be a positive integer.

        Returns:
            dict: Processed and dimension-checked dictionary containing only allowed keys,
                with all values as lists of CasADi SX symbols of length `N`.
        
        Raises:
            AssertionError: If no input lists are provided or the horizon `N` is not set.
            AssertionError: If the dimensions of the input ingredients do not match.
            AssertionError: If any required input is missing.
        """

        # get length of all elements
        N_list = [len(elem) for elem in data.values() if isinstance(elem,list) if len(elem) > 1]

        # check if N is passed
        if N is not None:
            
            # if so, check that it is positive
            assert N > 0, 'Please set a positive horizon N.'

            # and append to list of lengths
            N_list.append(N)

        # check that all dimensions match
        assert len(N_list) > 0, 'Please either pass a list of qp ingredients or set horizon N.'
        assert N_list.count(N_list[0]) == len(N_list), 'Dimensions of passed ingredients do not match.'

        # define N in case it was not passed
        N = N_list[0]

        # turn all elements into list of appropriate dimension
        processed_data = data \
                         | {key : [ca.SX(val)]*N for key,val in data.items() if val is not None and not isinstance(val,list)} \
                         | {key : [ca.SX(val) for val in val_list] for key,val_list in data.items() if isinstance(val_list,list) and len(val_list) > 1} \
                         | {key : [ca.SX(val_list[0])]*N for key,val_list in data.items() if isinstance(val_list,list) and len(val_list) == 1}

        # check that required entries were passed
        assert all([processed_data[key] is not None for key in Ingredients._REQUIRED_KEYS] ), 'Some required inputs are missing.'

        # strip unwanted inputs
        processed_data_stripped = {key : val for key,val in processed_data.items() if key in Ingredients._ALLOWED_KEYS}

        return processed_data_stripped

    @staticmethod
    def _check_dimensions(data:dict) -> None:
        """
        Checks the consistency of dimensions for the properties: dynamics, cost, and csts within the provided data 
        dictionary. This method verifies that:
        - All elements in each list associated with a property have the same shape.
        - The dimensions of each property match the expected dimensions defined in self._EXPECTED_DIMENSIONS.
        - All collected dimensions for each expected key are consistent across the data.

        Parameters:
            data (dict): A dictionary where keys correspond to property names (e.g., 'dynamics', 'cost', 'constraints') 
            and values are lists of numpy arrays or similar objects with a .shape attribute.
        
        Raises:
            AssertionError: If any list contains elements with differing shapes, or if the dimensions do not match
            the expected configuration for any property.
        """

        # initialize dictionary containing all allowed keys
        dimension_dict = {key:[] for key in Ingredients._ALL_DIMENSIONS}

        # specify 'one' entry
        dimension_dict['one'] = [1]

        # loop through nonempty attributes
        for key,val in data.items():

            # get expected dimensions
            expected_dimension = Ingredients._EXPECTED_DIMENSIONS[key]
            
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

    def _check_slack(self,data:dict) -> bool | int:
        """
        Checks and validates the presence and configuration of slack variables and their penalties in the
        provided data dictionary.
        
        Parameters:
            data (dict): A dictionary containing potential keys for slack constraints and their penalties
                - 'Hx_e': Matrix specifying slack constraints.
                - 's_quad': Quadratic penalty for slack variables (must be positive).
                - 's_lin': Linear penalty for slack variables (must be positive).
        
        Returns:
            slack (bool): True if slack variables are detected and properly configured, False otherwise.
            n_eps (int): The number of slack variables (columns in 'Hx_e') if present, otherwise 0.
        
        Raises:
            AssertionError: If required keys are missing or penalties are not positive as expected.
        """

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