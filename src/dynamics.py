import casadi as ca
import time
from src.symbolic_var import SymbolicVar

class Dynamics:
    """
    Dynamics Class
    This class represents a system's dynamics and provides methods for symbolic representation, 
    linearization, and compilation of the dynamics using CasADi.

    Attributes:
        x_next (ca.SX): Symbolic expression for the next state of the system.
        x_next_nom (ca.SX): Symbolic expression for the nominal next state of the system.
        f (ca.Function): CasADi function representing the system dynamics.
        A (ca.Function): CasADi function representing the Jacobian of the dynamics with respect to the state.
        B (ca.Function): CasADi function representing the Jacobian of the dynamics with respect to the control input.
        f_nom (ca.Function): CasADi function representing the nominal system dynamics.
        A_nom (ca.Function): CasADi function representing the Jacobian of the nominal dynamics with respect to the state.
        B_nom (ca.Function): CasADi function representing the Jacobian of the nominal dynamics with respect to the control input.
        param (dict): Dictionary of symbolic parameters for the system dynamics.
        param_nom (dict): Dictionary of symbolic parameters for the nominal system dynamics.
        dim (dict): Dictionary containing the dimensions of the state and input variables.
        model (dict): Dictionary containing the linearized dynamics matrices (A, B, c).
        compTimes (dict): Dictionary containing the computation times for various compiled CasADi functions.
        init (dict): Dictionary containing the initial values of symbolic variables.
    
    Methods:
        __init__(dyn: dict, compile: bool = False):
            Initializes the Dynamics class with the provided symbolic dynamics and compilation options.
        _linearize(horizon: int, method: str = 'trajectory'):
            Constructs the prediction model for the MPC problem using linearization. 
            Supports affine dynamics, linearization around the initial state, or along a trajectory.
        _set_init(data):
            Sets the initial values for symbolic variables.
    
    Properties:
        x_next: Returns the symbolic expression for the next state.
        x_next_nom: Returns the symbolic expression for the nominal next state.
        f: Returns the CasADi function for the system dynamics.
        A: Returns the CasADi function for the Jacobian with respect to the state.
        B: Returns the CasADi function for the Jacobian with respect to the control input.
        f_nom: Returns the CasADi function for the nominal system dynamics.
        A_nom: Returns the CasADi function for the Jacobian of the nominal dynamics with respect to the state.
        B_nom: Returns the CasADi function for the Jacobian of the nominal dynamics with respect to the control input.
        param: Returns the dictionary of symbolic parameters for the system dynamics.
        param_nom: Returns the dictionary of symbolic parameters for the nominal system dynamics.
        dim: Returns the dimensions of the state and input variables.
        model: Returns the dictionary containing the linearized dynamics matrices (A, B, c).
        compTimes: Returns the dictionary containing computation times for compiled CasADi functions.
        init: Returns the dictionary containing the initial values of symbolic variables.
    """

    def __init__(self,dyn:dict,jit:bool=False) -> None:
        """
        Class constructor.

        Args:
            dyn (dict): Dictionary containing the system dynamics in symbolic form. 
                Required keys:
                    - 'x': state variable
                    - 'u': control input
                    - 'x_next': next state expression
                Optional keys:
                    - 'd': disturbance
                    - 'w': process noise
                    - 'theta': parameters
                    - 'x_next_nom': nominal next state
            jit (bool, optional): If True, compile the dynamics using CasADi's JIT compilation.
                Defaults to False.
        """

        # compilation options
        if jit:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            options = {}

        # check that all keys were passed
        assert all(key in dyn for key in ['x', 'u', 'x_next']), 'x, u, and x_next must be defined.'

        # check symbolic type
        assert all( isinstance(elem,ca.SX) for elem in dyn.values()), 'Dynamics dictionary should contain variables of type SX.'

        # store dynamics
        self._x_next = dyn['x_next']

        # determine nominal dynamics
        self._x_next_nom = dyn['x_next_nom'] if 'x_next_nom' in dyn else dyn['x_next']

        # check dimensions of x_next and x_next_nom
        assert self._x_next.shape == dyn['x'].shape, 'x_next must have the same dimensions as x.'
        assert self._x_next_nom.shape == dyn['x'].shape, 'x_next must have the same dimensions as x.'

        # create dictionary containing symbolic variables
        self._sym = SymbolicVar()

        # store state and input
        self._sym.add_var('x', dyn['x'])
        self._sym.add_var('u', dyn['u'])

        # save variables in parameter vectors
        self._param = {'x':dyn['x'],'u':dyn['u']}
        self._param_nom = self._param.copy()

        # store noise and disturbance if present
        if 'd' in dyn:
            self._sym.add_var('d', dyn['d'])
            self._param = self._param | {'d':dyn['d']}
        if 'w' in dyn:
            self._sym.add_var('w', dyn['w'])
            self._param = self._param | {'w':dyn['w']}

        # store nominal model if present
        if 'theta' in dyn:
            self._sym.add_var('theta', dyn['theta'])
            self._param_nom = self._param_nom | {'theta':dyn['theta']}

        # extract symbolic parameters and their names
        p_names, p = map(list,zip(*[(key, val) for key, val in self.param.items()]))

        # extract nominal symbolic parameters and their names
        p_nom_names,p_nom = map(list,zip(*[(key, val) for key, val in self._param_nom.items()]))

        # check that x_next only depends on provided symbols
        assert set(ca.symvar(self._x_next)).issubset(set(ca.symvar(ca.vcat(p)))), 'x_next must depend on the symbolic parameters x,u,d,w'

        # check that x_next_nom only depends on provided symbols
        assert set(ca.symvar(self._x_next_nom)).issubset(set(ca.symvar(ca.vcat(p_nom)))), 'x_next_nom must depend on the symbolic parameters x,u'
        
        # initialize dictionary containing all compilation times
        comp_time_dict = {}

        # create casadi function for the dynamics
        start = time.time()
        f = ca.Function('f', p, [self._x_next], p_names, ['x_next'], options)
        comp_time_dict = comp_time_dict | {'f':time.time()-start}

        # if d or w were passed, nominal and true models are different
        model_is_noisy = False if set(ca.symvar(ca.vcat(p))) == set(ca.symvar(ca.vcat(p_nom))) else True

        # create nominal dynamics
        if model_is_noisy:
            start = time.time()
            f_nom = ca.Function('f_nom',p_nom,[self._x_next_nom],p_nom_names,['x_next_nominal'],options)
            comp_time_dict = comp_time_dict | {'f_nom':time.time()-start}
        else:
            # otherwise, copy exact dynamics
            f_nom = f
            comp_time_dict = comp_time_dict | {'f_nom':comp_time_dict['f']}
        
        # save in dynamics
        self._f = f
        self._f_nom = f_nom

        # compute Jacobians symbolically
        df_dx = ca.jacobian(self.x_next,self.param['x'])
        df_du = ca.jacobian(self.x_next,self.param['u'])

        # check if df_dx and df_du are constant
        if ca.jacobian(ca.vcat([*ca.symvar(df_dx),*ca.symvar(df_du)]),ca.vertcat(self.param['x'],self.param['u'])).is_zero():
            self._is_affine = True
        else:
            self._is_affine = False

        # compute Jacobians
        start = time.time()
        a = ca.Function('A', p, [df_dx], p_names, ['A'], options)
        comp_time_dict = comp_time_dict | {'A':time.time()-start}
        start = time.time()
        b = ca.Function('B', p, [df_du], p_names, ['B'], options)
        comp_time_dict = comp_time_dict | {'B':time.time()-start}

        # compute nominal jacobians
        if model_is_noisy:

            # compute jacobians symbolically
            df_dx_nom = ca.jacobian(self.x_next_nom,self.param['x'])
            df_du_nom = ca.jacobian(self.x_next_nom,self.param['u'])

            # compute jacobians
            start = time.time()
            a_nom = ca.Function('A_nom', p_nom, [df_dx_nom], p_nom_names, ['A'], options)
            comp_time_dict = comp_time_dict | {'A_nom':time.time()-start}
            start = time.time()
            b_nom = ca.Function('B_nom', p_nom, [df_du_nom], p_nom_names, ['B'], options)
            comp_time_dict = comp_time_dict | {'B_nom':time.time()-start}

            # save in dynamics
            self._A_nom = a_nom
            self._B_nom = b_nom
        else:
            # otherwise, copy exact dynamics
            a_nom = a
            b_nom = b
            comp_time_dict = comp_time_dict | {'A_nom':comp_time_dict['A'],'B_nom':comp_time_dict['B']}
        
        # store in dynamics
        self._A = a
        self._A_nom = a_nom
        self._B = b
        self._B_nom = b_nom

        # store computation times
        self._compTimes = comp_time_dict

        # store empty model
        self._model = {}

    def _linearize(self,horizon:int,method='trajectory') -> dict | SymbolicVar | str:
        """
        Linearizes the system dynamics based on the specified method and horizon.
        
        Args:
            horizon (int):The prediction horizon for the linearization.
            method (str, optional):The linearization method to use. Options are:
                - 'trajectory': Linearizes along a trajectory (default).
                - 'initial_state': Linearizes around the initial state.
                - If the model is affine, computes exact dynamics.
        Returns:
            model (dict): A dictionary containing the linearized system matrices:
                - 'A': List of state transition matrices for each time step.
                - 'B': List of input matrices for each time step.
                - 'c': List of offset vectors for each time step.
        symbolic_vars (SymbolicVar): The symbolic variables used during the linearization
            process as well as the symbolic variables of the dynamics class object.
        linearization_method (str): The method used for linearization ('affine', 
            'trajectory', or 'initial_state').
        
        Notes:
            - If the model is affine, the dynamics are computed exactly as f(x, u) = Ax + Bu + c.
            - For 'trajectory' mode, the linearization is performed along a trajectory, 
                splitting the input and state vectors for each time step.
            - For 'initial_state' mode, the linearization is performed around the initial state.
        """


        # extract symbolic variables
        x = self.param_nom['x']
        u = self.param_nom['u']

        # get state and input dimensions
        n_x = self.dim['x']
        n_u = self.dim['u']

        # extract dynamics
        fd = self.f_nom
        a = self.A_nom
        b = self.B_nom

        # extract nominal parameters
        param_nom = self.param_nom

        # extract symbolic variables
        symbolic_vars = self._sym.copy()

        # if model is affine, compute exact dynamics
        if self._is_affine:

            # create nominal dynamics f(x,u) = Ax + bu + c
            a_mat = ca.sparsify(ca.cse(list(a.call(param_nom).values())[0]))
            b_mat = ca.sparsify(ca.cse(list(b.call(param_nom).values())[0]))
            c_mat = ca.sparsify(ca.cse( list(fd.call(param_nom).values())[0] - a_mat@x - b_mat@u ))

            # substitute x and u (sometimes casadi does not recognize that c_mat is constant)
            c_mat_num_1 = ca.substitute(c_mat,x,ca.SX.zeros(*x.shape))
            c_mat_num_2 = ca.substitute(c_mat_num_1,u,ca.SX.zeros(*u.shape))

            # stack in list
            a_list = [a_mat] * horizon
            b_list = [b_mat] * horizon
            c_list = [c_mat_num_2] * horizon

        # if mode is 'initial_state', linearize around the initial state
        elif method == 'initial_state':

            # linearization trajectory is a single input
            y_lin = ca.SX.sym('y_lin',n_u,1)
            u_lin = y_lin

            # substitute u_lin in param_nom
            param_nom['u'] = u_lin

            # compute derivatives
            a_lin = list(a.call(param_nom).values())[0]
            b_lin = list(b.call(param_nom).values())[0]
            c_lin = list(fd.call(param_nom).values())[0] - a_lin@x - b_lin@u_lin

            # stack in list
            a_list = [a_lin] * horizon
            b_list = [b_lin] * horizon
            c_list = [c_lin] * horizon

            # store y_lin
            symbolic_vars.add_var('y_lin', y_lin)
                
        # if mode is 'trajectory', linearize along a trajectory (similar to real-time iteration)
        elif method == 'trajectory':

            # create symbolic variable for linearization trajectory
            y_lin = ca.SX.sym('y_lin',(n_x+n_u)*horizon,1)

            # extract linearization input and state
            x_lin = y_lin[:horizon*n_x]
            u_lin = y_lin[horizon*n_x:]

            # preallocate matrices
            a_list, b_list, c_list = [], [], []

            # extract linearization points by splitting the x_lin and u_lin vectors
            x_lin_list = ca.vertsplit(x_lin,n_x)
            u_lin_list = ca.vertsplit(u_lin,n_u)

            # the first state should be the true system state
            x_lin_list[0] = x

            # construct matrices
            for x_i,u_i in zip(x_lin_list,u_lin_list):

                # substitute values of x and u
                param_nom['x'] = x_i
                param_nom['u'] = u_i

                # evaluate jacobians
                a_i = list(a.call(param_nom).values())[0]
                a_list.append(a_i)
                b_i = list(b.call(param_nom).values())[0]
                b_list.append(b_i)

                # evaluate linear part
                c_i = list(fd.call(param_nom).values())[0] - a_i@x_i - b_i@u_i
                c_list.append(c_i)

            # store y_lin
            symbolic_vars.add_var('y_lin', y_lin)

        else:
            raise Exception('unknown linearization method')

        # store output dictionary
        model = {'A':a_list, 'B':b_list, 'c':c_list}

        # determine linearization method that was used
        linearization_method = 'affine' if self._is_affine else method

        # return used linearization method
        return model, symbolic_vars, linearization_method
    
    @property
    def x_next(self):
        return self._x_next

    @property
    def x_next_nom(self):
        return self._x_next_nom
    
    @property
    def f(self):
        return self._f
    
    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B
    
    @property
    def f_nom(self):
        return self._f_nom

    @property
    def A_nom(self):
        return self._A_nom
    
    @property
    def B_nom(self):
        return self._B_nom

    @property
    def param(self):
        return self._param

    @property
    def param_nom(self):
        return self._param_nom
    
    @property
    def dim(self):
        return self._sym.dim
    
    @property
    def model(self):
        return self._model
    
    @property
    def comp_times(self):
        return self._compTimes
    
    @property
    def init(self):
        return {key:val for key,val in self._sym.init.items() if val is not None}
    
    def _set_init(self,data:dict):
        self._sym.set_init(data)