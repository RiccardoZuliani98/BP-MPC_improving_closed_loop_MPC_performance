import casadi as ca
import time
from src.symb import Symb

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

    def __init__(self,dyn:dict,compile:bool=False):
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
            compile (bool, optional): If True, compile the dynamics using CasADi's JIT compilation. 
                Defaults to False.
        """

        # compilation options
        if compile:
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
        self._sym = Symb()

        # store state and input
        self._sym.addVar('x', dyn['x'])
        self._sym.addVar('u', dyn['u'])

        # save variables in parameter vectors
        self._param = {'x':dyn['x'],'u':dyn['u']}
        self._param_nom = self._param.copy()

        # store noise and disturbance if present
        if 'd' in dyn:
            self._sym.addVar('d', dyn['d'])
            self._param = self._param | {'d':dyn['d']}
        if 'w' in dyn:
            self._sym.addVar('w', dyn['w'])
            self._param = self._param | {'w':dyn['w']}

        # store nominal model if present
        if 'theta' in dyn:
            self._sym.addVar('theta', dyn['theta'])
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
        model_is_noisy = True if len(p) > len(p_nom) else False

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

        # compute jacobians symbolically
        df_dx = ca.jacobian(self.x_next,self.param['x'])
        df_du = ca.jacobian(self.x_next,self.param['u'])

        # check if df_dx and df_du are constant
        if ca.jacobian(ca.vcat([*ca.symvar(df_dx),*ca.symvar(df_du)]),ca.vertcat(self.param['x'],self.param['u'])).is_zero():
            self._is_affine = True
        else:
            self._is_affine = False

        # compute jacobians
        start = time.time()
        A = ca.Function('A', p, [df_dx], p_names, ['A'], options)
        comp_time_dict = comp_time_dict | {'A':time.time()-start}
        start = time.time()
        B = ca.Function('B', p, [df_du], p_names, ['B'], options)
        comp_time_dict = comp_time_dict | {'B':time.time()-start}

        # compute nominal jacobians
        if model_is_noisy:

            # compute jacobians symbolically
            df_dx_nom = ca.jacobian(self.x_next_nom,self.param['x'])
            df_du_nom = ca.jacobian(self.x_next_nom,self.param['u'])

            # compute jacobians
            start = time.time()
            A_nom = ca.Function('A_nom', p_nom, [df_dx_nom], p_nom_names, ['A_nom'], options)
            comp_time_dict = comp_time_dict | {'A_nom':time.time()-start}
            start = time.time()
            B_nom = ca.Function('B_nom', p_nom, [df_du_nom], p_nom_names, ['B_nom'], options)
            comp_time_dict = comp_time_dict | {'B_nom':time.time()-start}

            # save in dynamics
            self._A_nom = A_nom
            self._B_nom = B_nom
        else:
            # otherwise, copy exact dynamics
            A_nom = A
            B_nom = B
            comp_time_dict = comp_time_dict | {'A_nom':comp_time_dict['A'],'B_nom':comp_time_dict['B']}
        
        # store in dynamics
        self._A = A
        self._A_nom = A_nom
        self._B = B
        self._B_nom = B_nom

        # store computation times
        self._compTimes = comp_time_dict

        # store empty model
        self._model = {}

    def _linearize(self,horizon:int,method='trajectory'):
        """
        Construct the prediction model for the MPC problem using linearization.

        There are multiple options:

            1. If the model is affine (`self._is_affine == True`), then `A`, `B`, and `c` represent 
            the true nominal dynamics of the system.

            2. Linearize around the initial state (`method='state'`). 
            In this case, the linearization trajectory uses a single input `u_lin`.

            3. Linearize along a trajectory (`method='trajectory'`, default).
            In this case, `y_lin` contains the state-input trajectory used for linearization.

        The function returns three lists: `A_list`, `B_list`, and `c_list`, such that the linearized 
        dynamics at time step `t` are given by:

            x[t+1] = A_list[t] @ x[t] + B_list[t] @ u[t] + c_list[t]

        It also returns `y_lin`, the symbolic parameter used in the linearization.

        Note:
            `c_list[0]` includes the effect of the initial state `x0`, specifically the term `-A_list[0] @ x0`.

        Args:
            horizon (int): Horizon of the MPC.
            method (str, optional): Type of linearization. Options are:
                - 'trajectory' (default)
                - 'state'
        """

        # extract symbolic variables
        x = self.param_nom['x']
        u = self.param_nom['u']

        # get state and input dimensions
        n_x = self.dim['x']
        n_u = self.dim['u']

        # extract dynamics
        fd = self.f_nom
        A = self.A_nom
        B = self.B_nom

        # extract nominal parameters
        param_nom = self.param_nom

        # if model is affine, compute exact dynamics
        if self._is_affine:

            # create nominal dynamics f(x,u) = Ax + bu + c
            A_mat = list(A(param_nom).values())[0]
            B_mat = list(B(param_nom).values())[0]
            c_mat = -(list(fd(param_nom).values())[0] - A_mat@x - B_mat@u)

            # stack in list
            A_list = [A_mat] * horizon
            B_list = [B_mat] * horizon
            c_list = [c_mat] * horizon

        # if mode is 'initial_state', linearize around the initial state
        elif method == 'initial_state':

            # linearization trajectory is a single input
            y_lin = ca.SX.sym('y_lin',n_u,1)
            u_lin = y_lin

            # substitute u_lin in param_nom
            param_nom['u'] = u_lin

            # compute derivatives
            A_lin = list(A(param_nom).values())[0]
            B_lin = list(B(param_nom).values())[0]
            c_lin = - ( list(fd(param_nom).values())[0] - A_lin@x - B_lin@u_lin )

            # stack in list
            A_list = [A_lin] * horizon
            B_list = [B_lin] * horizon
            c_list = [c_lin] * horizon

            # store y_lin
            self._sym.addVar('y_lin', y_lin)
                
        # if mode is 'trajectory', linearize along a trajectory (similar to real-time iteration)
        elif method == 'trajectory':

            # create symbolic variable for linearization trajectory
            y_lin = ca.SX.sym('y_lin',(n_x+n_u)*horizon,1)

            # extract linearization input and state
            x_lin = y_lin[:horizon*n_x]
            u_lin = y_lin[horizon*n_x:]

            # preallocate matrices
            A_list, B_list, c_list = [], [], []

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
                A_i = list(A.call(param_nom).values())[0]
                A_list.append(A_i)
                B_i = list(B.call(param_nom).values())[0]
                B_list.append(B_i)

                # evaluate linear part
                c_i = - ( list(fd.call(param_nom).values())[0] - A_i@x_i - B_i@u_i )
                c_list.append(c_i)

            # store y_lin
            self._sym.addVar('y_lin', y_lin)

        else:
            raise Exception('unknown linearization method')

        # store output dictionary
        self._model = {'A':A_list, 'B':B_list, 'c':c_list}

        # return used linearization method
        return 'affine' if self._is_affine else method
    
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
    def compTimes(self):
        return self._compTimes
    
    @property
    def init(self):
        return {key:val for key,val in self._sym.init.items() if val is not None}
    
    def _set_init(self,data:dict):
        self._sym.set_init(data)