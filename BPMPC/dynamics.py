from casadi import *

class dynamics:

    def __init__(self,MSX='SX'):

        # check type of symbolic variables
        if MSX == 'SX':
            self.__MSX = SX
        elif MSX == 'MX':
            self.__MSX = MX
        else:
            raise Exception('MSX must be either SX or MX.')

        # initialize all variables
        self.__x = None
        self.__u = None
        self.__d = None
        self.__w = None
        self.__x_next = None
        self.__x_next_nom = None
        self.__x_dot = None
        self.__x_dot_nom = None
        self.__init = {'x':None,'u':None,'d':None,'w':None}
        self.__A = None
        self.__B = None
        self.__f = None
        self.__fc = None
        self.__A_nom = None
        self.__B_nom = None
        self.__f_nom = None
        self.__fc_nom = None
        self.__type = None
        pass

    @property
    def x(self):
        return self.__x
    
    def __set_x(self, value):
        assert type(value) is self.__MSX, 'x is of the wrong symbolic type.'
        self.__x = value

    @property
    def u(self):
        return self.__u
    
    def __set_u(self, value):
        assert type(value) is self.__MSX, 'u is of the wrong symbolic type.'
        self.__u = value

    @property
    def d(self):
        return self.__d
    
    def __set_d(self, value):
        assert type(value) is self.__MSX, 'd is of the wrong symbolic type.'
        self.__d = value

    @property
    def w(self):
        return self.__w
    
    def __set_w(self, value):
        assert type(value) is self.__MSX, 'w is of the wrong symbolic type.'
        self.__w = value

    @property
    def x_next(self):
        return self.__x_next
    
    def __set_x_next(self, value):
        if type(value) is not self.__MSX:
            raise Exception('x_next is of the wrong symbolic type.')
        self.__x_next = value

    @property
    def x_next_nom(self):
        return self.__x_next_nom
    
    def __set_x_next_nom(self, value):
        if type(value) is not self.__MSX:
            raise Exception('x_next_nom is of the wrong symbolic type.')
        self.__x_next_nom = value
    
    @property
    def x_dot(self):
        return self.__x_dot
    
    def __set_x_dot(self, value):
        if type(value) is not self.__MSX:
            raise Exception('x_dot is of the wrong symbolic type.')
        self.__x_dot = value

    @property
    def x_dot_nom(self):
        return self.__x_dot_nom
    
    def __set_x_dot_nom(self, value):
        if type(value) is not self.__MSX:
            raise Exception('x_dot_nom is of the wrong symbolic type.')
        self.__x_dot_nom = value

    @property
    def param(self):
        return {k: v for k, v in {
            'x': self.__x,
            'u': self.__u,
            'd': self.__d,
            'w': self.__w
        }.items() if v is not None}

    @property
    def param_nominal(self):
        return {k: v for k, v in {
            'x': self.__x,
            'u': self.__u
        }.items() if v is not None}
    
    @property
    def init(self):
        return {k: v for k, v in self.__init.items()}
    
    def __setInit(self, init):
        out = self.__checkInit(init)
        for key in ['x','u','d']:
            if key in out and out[key].shape[1]!=1:
                raise Exception('Initial value must be a column vector.')
        self.__init = self.__init | out
    
    def __checkInit(self, value):

        # create output dictionary
        out = {}

        # check if input dictionary contains 'x' key
        if 'x' in value:

            assert 'x' in self.param, 'Define the state x before defining its initial value.'

            # if so, extract x
            try:
                x = DM(value['x'])
            except:
                raise Exception('Initial value of x must be a numeric value.')

            # check if x has correct dimension
            assert x.shape[0] == self.param['x'].shape[0], 'x has incorrect dimension.'

            # add new initial linearization
            out = out | {'x':x}

        # check if input dictionary contains 'u' key
        if 'u' in value:

            assert 'u' in self.param, 'Define the input u before defining its initial value.'

            # if so, extract u
            try:
                u = DM(value['u'])
            except:
                raise Exception('Initial value of u must be a numeric value.')

            # check if u has correct dimension
            assert u.shape[0] == self.param['u'].shape[0], 'u has incorrect dimension.'

            # add new initial linearization
            out = out | {'u':u}

        # check if input dictionary contains 'udw' key
        if 'd' in value:

            assert 'd' in self.param, 'Define the disturbance d before defining its initial value.'

            # if so, extract d
            try:              
                d = DM(value['d'])
            except:
                raise Exception('Initial value of d must be a numeric value.')

            # check if d has correct dimension
            assert d.shape[0] == self.param['d'].shape[0], 'd has incorrect dimension.'

            # add new initial linearization
            out = out | {'d':d}

        # check that number of columns of variables that are not None match
        if len(set([v.shape[1] for v in out.values() if v is not None])) > 1:
            raise Exception('All initial values except w must have the same number of columns.')
        
        # check if input dictionary contains 'w' key
        if 'w' in value:

            assert 'w' in self.param, 'Define the noise w before defining its initial value.'

            # if so, extract w
            try:
                w = DM(value['w'])
            except:
                raise Exception('Initial value of w must be a numeric value.')

            # check if w has correct dimension
            assert w.shape[0] == self.param['w'].shape[0], 'w has incorrect dimension.'

            # add new initial linearization
            out = out | {'w':w}

        return out

    @property
    def A(self):
        return self.__A
    
    def __set_A(self, value):
        self.__A = value

    @property
    def B(self):
        return self.__B
    
    def __set_B(self, value):
        self.__B = value

    @property
    def f(self):
        return self.__f
    
    def __set_f(self, value):
        self.__f = value

    @property
    def fc(self):
        return self.__fc
    
    def __set_fc(self, value):
        self.__fc = value

    @property
    def A_nom(self):
        return self.__A_nom
    
    def __set_A_nom(self, value):
        self.__A_nom = value

    @property
    def B_nom(self):
        return self.__B_nom
    
    def __set_B_nom(self, value):
        self.__B_nom = value

    @property
    def f_nom(self):
        return self.__f_nom
    
    def __set_f_nom(self, value):
        self.__f_nom = value

    @property
    def fc_nom(self):
        return self.__fc_nom
    
    def __set_fc_nom(self, value):
        self.__fc_nom = value

    @property
    def type(self):
        return self.__type
    
    def __set_type(self, value):
        self.__type = value

    def linearize(self,N,linearization='trajectory'):

        """
        This function constructs the prediction model for the MPC problem. There are multiple options:

            1. If the model is affine, then A,B,c are the true nominal dynamics of the system, this happens
               if self.model.type == 'affine'.
               
            2. The model can be linearized around the initial state (linearization = 'state').
               In this case, the linearization trajectory is a single input u_lin.

            3. (default) The model can be linearized along a trajectory (linearization = 'trajectory').
               In this case y_lin contains the state-input trajectory along which the dynamics are linearized.

        The function returns three list A_list,B_list,c_list, such that the linearized dynamics at time-step t
        are given by  x[t+1] = A_list[t]@x[t] + B_list[t]@u[t] + c_list[t]. It also returns y_lin, the symbolic
        parameter used in the linearization.

        Note that c_list[0] contains additionally the effect -A_list[0]@x0 of the initial state x0.

        The inputs are

            * N: horizon of the MPC

            * linearization: type of chosen linearization (default is 'trajectory')

        """

        # get symbolic variable type
        MSX = self.__MSX

        # extract symbolic variables
        x = self.param['x']

        # get state and input dimensions
        n_x = x.shape[0]
        n_u = self.param['u'].shape[0]

        # extract dynamics
        fd = self.f_nom
        A = self.A_nom
        B = self.B_nom

        # if model is affine, compute exact dynamics
        if self.type == 'affine':

            # extract nominal symbolic parameters and their names
            p_nom_names = self.param_nominal.keys()
            
            # extract nominal values of nominal parameters
            p_init_nom = [self.init[i] if self.init[i] is not None else DM(*self.param[i].shape) for i in p_nom_names]

            # get nominal state and input
            x_nom = self.init['x'] if self.init['x'] is not None else DM(*self.param['x'].shape)
            u_nom = self.init['u'] if self.init['u'] is not None else DM(*self.param['u'].shape)

            # create nominal dynamics f(x,u) = Ax + bu + c
            A_mat = A(*p_init_nom)
            B_mat = B(*p_init_nom)
            c_mat = -(fd(*p_init_nom) - A_mat@x_nom - B_mat@u_nom)

            # stack in list
            A_list = [A_mat] * N
            B_list = [B_mat] * N
            c_list = [c_mat] * N

            # patch first entry of c_list
            c_list[0] = c_list[0] - A_mat@x

            # no linearization here
            y_lin = None

        # if mode is 'initial_state', linearize around the initial state
        elif linearization == 'initial_state':

            # linearization trajectory is a single input
            y_lin = MSX.sym('y_lin',n_u,1)
            u_lin = y_lin

            # compute derivatives
            A_lin = A(x,u_lin)
            B_lin = B(x,u_lin)
            c_lin = - ( fd(x,u_lin) - A_lin@x - B_lin@u_lin )

            # stack in list
            A_list = [A_lin] * N
            B_list = [B_lin] * N
            c_list = [c_lin] * N

            # patch first entry of c_list
            c_list[0] = c_list[0] - A_lin@x
                
        # if mode is 'trajectory', linearize along a trajectory (similar to real-time iteration)
        elif linearization == 'trajectory':

            # create symbolic variable for linearization trajectory
            y_lin = MSX.sym('y_lin',(n_x+n_u)*N,1)

            # extract linearization input and state
            x_lin = y_lin[:N*n_x]
            u_lin = y_lin[N*n_x:]

            # preallocate matrices
            A_list, B_list, c_list = [], [], []

            # extract linearization points by splitting the x_lin and u_lin vectors
            x_lin_list = vertsplit(x_lin,n_x)
            u_lin_list = vertsplit(u_lin,n_u)

            # the first state should be the true system state
            x_lin_list[0] = x

            # construct matrices
            for x_i,u_i in zip(x_lin_list,u_lin_list):
                A_i = A(x_i,u_i)
                A_list.append(A_i)
                B_i = B(x_i,u_i)
                B_list.append(B_i)
                c_i = - ( fd(x_i,u_i) - A_i@x_i - B_i@u_i )
                c_list.append(c_i)

        return A_list, B_list, c_list, y_lin

    # overwrite the __dir__ method
    def __dir__(self):
        return [attr for attr in super().__dir__() if not attr.startswith('_dynamics__')]