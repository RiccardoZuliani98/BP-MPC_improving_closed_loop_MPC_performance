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
        if type(value) is not self.__MSX:
            raise Exception('x is of the wrong symbolic type.')
        self.__x = value
        # self.__setInit({'x':DM(value.shape[0],1)})

    @property
    def u(self):
        return self.__u
    
    def __set_u(self, value):
        if type(value) is not self.__MSX:
            raise Exception('u is of the wrong symbolic type.')
        self.__u = value
        # self.__setInit({'u':DM(value.shape[0],1)})

    @property
    def d(self):
        return self.__d
    
    def __set_d(self, value):
        if type(value) is not self.__MSX:
            raise Exception('d is of the wrong symbolic type.')
        self.__d = value
        # self.__setInit({'d':DM(value.shape[0],1)})

    @property
    def w(self):
        return self.__w
    
    def __set_w(self, value):
        if type(value) is not self.__MSX:
            raise Exception('w is of the wrong symbolic type.')
        self.__w = value
        # self.__setInit({'w':DM(value.shape[0],1)})

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

            if 'x' not in self.param:
                raise Exception('Define the state x before defining its initial value.')

            # if so, extract x
            try:
                x = DM(value['x'])
            except:
                raise Exception('Initial value of x must be a numeric value.')

            # check if x has correct dimension
            if x.shape[0] != self.param['x'].shape[0]:
                raise Exception('x has incorrect dimension.')

            # add new initial linearization
            out = out | {'x':x}

        # check if input dictionary contains 'u' key
        if 'u' in value:

            if 'u' not in self.param:
                raise Exception('Define the input u before defining its initial value.')

            # if so, extract u
            try:
                u = DM(value['u'])
            except:
                raise Exception('Initial value of u must be a numeric value.')

            # check if u has correct dimension
            if u.shape[0] != self.param['u'].shape[0]:
                raise Exception('u has incorrect dimension.')

            # add new initial linearization
            out = out | {'u':u}

        # check if input dictionary contains 'udw' key
        if 'd' in value:

            if 'd' not in self.param:
                raise Exception('Define the disturbance d before defining its initial value.')

            # if so, extract d
            try:              
                d = DM(value['d'])
            except:
                raise Exception('Initial value of d must be a numeric value.')

            # check if d has correct dimension
            if d.shape[0] != self.param['d'].shape[0]:
                raise Exception('d has incorrect dimension.')

            # add new initial linearization
            out = out | {'d':d}

        # check that number of columns of variables that are not None match
        if len(set([v.shape[1] for v in out.values() if v is not None])) > 1:
            raise Exception('All initial values except w must have the same number of columns.')
        
        # check if input dictionary contains 'w' key
        if 'w' in value:

            if 'w' not in self.param:
                raise Exception('Define the noise w before defining its initial value.')

            # if so, extract w
            try:
                w = DM(value['w'])
            except:
                raise Exception('Initial value of w must be a numeric value.')

            # check if w has correct dimension
            if w.shape[0] != self.param['w'].shape[0]:
                raise Exception('w has incorrect dimension.')

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

    # overwrite the __dir__ method
    def __dir__(self):
        return [attr for attr in super().__dir__() if not attr.startswith('_Dynamics__')]