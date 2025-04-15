import casadi as ca
import time
from BPMPC.symb import Symb

class dynamics:

    def __init__(self,dyn,compile=False):

        # compilation options
        if compile:
            jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
            options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        else:
            options = {}

        # check that all keys were passed
        assert all(key in dyn for key in ['x', 'u', 'x_dot', 'x_next']), 'x,u,x_dot and x_next must be defined.'

        # check dimensions of x_next and x_dot
        assert dyn['x_next'].shape == dyn['x'].shape, 'x_next must have the same dimensions as x.'
        assert dyn['x_dot'].shape == dyn['x'].shape, 'x_dot must have the same dimensions as x.'

        # determine type
        self.__MSX = type(dyn['x'])
        MSX = self.__MSX
        assert MSX in [ca.SX,ca.MX], 'Supported symbolic types are SX and MX'

        # create dictionary containing symbolic variables
        self.__sym = Symb('SX') if type(MSX) is ca.SX else Symb('MX')

        # store state and input
        self.__sym.addVar('x', dyn['x'])
        self.__sym.addVar('u', dyn['u'])

        # store next state and derivative
        self.__x_next = dyn['x_next']
        self.__x_dot = dyn['x_dot']

        # store noise and disturbance if present
        if 'd' in dyn:
            self.__sym.addVar('d', dyn['d'])
        if 'w' in dyn:
            self.__sym.addVar('w', dyn['w'])

        # initialize dictionary containing all compilation times
        comp_time_dict = {}

        # extract symbolic parameters and their names
        p_names, p = zip(*[(key, val) for key, val in self.__sym.var.items() if key in ['x', 'u', 'd', 'w']])

        # turn into a list
        p = list(p)
        p_names = list(p_names)

        # create casadi function for the dynamics
        start = time.time()
        fc = ca.Function('fc', p, [self.__sym.var['x_dot']], p_names, ['x_dot'], options)
        comp_time_dict = comp_time_dict | {'fc':time.time()-start}
        start = time.time()
        f = ca.Function('f', p, [self.__sym.var['x_next']], p_names, ['x_next'], options)
        comp_time_dict = comp_time_dict | {'f':time.time()-start}

        # extract nominal symbolic parameters and their names
        p_nom_names,p_nom = zip(*[(key, val) for key, val in self.__sym.var.items() if key in ['x', 'u']])
        
        # turn into a list
        p_nom = list(p_nom)
        p_nom_names = list(p_nom_names)

        # extract nominal values of nominal parameters
        p_init_nom = []

        if 'x0' not in dyn:
            p_init_nom.append(ca.DM.rand(self.dim['x'],1))
        else:
            try:
                x0 = ca.DM(dyn['x0'])
            except:
                raise Exception('Initial state x0 is not a valid numerical object.')
            
            assert x0.shape[0] == self.dim['x'], 'Initial state x0 does not have the correct dimension or has wrong type.'

            # set initialization
            self.__setInit({'x':x0})
            p_init_nom.append(x0)
        
        if 'u0' not in dyn:
            p_init_nom.append(ca.DM.rand(self.dim['u'],1))
        else:
            try:
                u0 = ca.DM(dyn['u0'])
            except:
                raise Exception('Initial input u0 is not a valid numerical object.')
            
            assert u0.shape[0] == self.dim['u'], 'Initial input u0 does not have the correct dimension or has wrong type.'

            # set initialization
            self.__setInit({'u':u0})
            p_init_nom.append(u0)

        # if d is present, add it to list of disturbance parameters
        dist_p = []
        if 'd' in self.param:

            if 'd0' not in dyn:
                print('Initialization of d was not passed, defaulting to zero.')
                self.__setInit({'d':ca.DM(self.dim['d'],1)})
                dist_p.append(ca.DM(self.dim['d'],1))
            else:
                try:
                    d0 = ca.DM(dyn['d0'])
                except:
                    raise Exception('Initial disturbance d0 is not a valid numerical object.')
                
                assert d0.shape[0] == self.dim['d'], 'Nominal disturbance d0 does not have the correct dimension or has wrong type.'

                # set initialization
                self.__setInit({'d':d0})
                dist_p.append(d0)
        
        # same for w
        if 'w' in self.param:

            if 'w0' not in dyn:
                print('Initialization of w was not passed, defaulting to zero.')
                self.__setInit({'w':ca.DM(self.dim['w'],1)})
                dist_p.append(ca.DM(self.dim['w'],1))
            else:
                try:
                    w0 = ca.DM(dyn['w0'])
                except:
                    raise Exception('Initial noise w0 is not a valid numerical object.')
                
                assert w0.shape[0] == self.dim['w'], 'Nominal noise w0 does not have the correct dimension or has wrong type.'

                # set initialization
                self.__setInit({'w':w0})
                dist_p.append(w0)

        # if d or w were passed, nominal and true models are different
        model_is_noisy = True if len(dist_p) > 0 else False

        # extract values of all parameters
        p_init = p_nom + dist_p
        
        # create nominal dynamics
        if model_is_noisy:

            # compute nominal next symbolic state and derivative
            x_next_nom = f(*p_init)
            x_dot_nom = fc(*p_init)

            # save in dynamics
            start = time.time()
            fc_nom = ca.Function('fc_nom',p_nom,[x_dot_nom],p_nom_names,['x_dot_nominal'],options)
            comp_time_dict = comp_time_dict | {'fc_nom':time.time()-start}
            start = time.time()
            f_nom = ca.Function('f_nom',p_nom,[x_next_nom],p_nom_names,['x_next_nominal'],options)
            comp_time_dict = comp_time_dict | {'f_nom':time.time()-start}
        else:
            # otherwise, copy exact dynamics
            fc_nom = fc
            x_dot_nom = self.x_dot
            f_nom = f
            x_next_nom = self.x_next
        
        # save in dynamics
        self.__fc = fc
        self.__f = f
        self.__fc_nom = fc_nom
        self.__x_dot_nom = x_dot_nom
        self.__f_nom = f_nom
        self.__x_next_nom = x_next_nom

        # test that f works correctly
        try:
            x_next = f(*p_init)
        except:
            raise Exception('Function f is incompatible with the parameters x,u,d,w you passed.')
        
        assert x_next.shape[0] == self.dim['x'], 'Function f does not return the correct dimension.'

        # test that fc works correctly
        try:
            x_dot = fc(*p_init)
        except:
            raise Exception('Function fc is incompatible with the parameters x,u,d,w you passed.')
        
        assert x_dot.shape[0] == self.dim['x'], 'Function fc does not return the correct dimension.'

        # if nominal and real dynamics are different, test that f_nom and fc_nom work correctly
        if model_is_noisy:
            # test that f_nom works correctly
            try:
                x_next_nom = f_nom(*p_init_nom)
            except:
                raise Exception('Function f_nom is incompatible with the parameters x,u you passed.')
            
            assert x_next_nom.shape[0] == self.dim['x'], 'Function f_nom does not return the correct dimension.'
            
            # test that fc_nom works correctly
            try:
                x_dot_nom = fc_nom(*p_init_nom)
            except:
                raise Exception('Function fc_nom is incompatible with the parameters x,u you passed.')
            assert x_dot_nom.shape[0] == self.dim['x'], 'Function fc_nom does not return the correct dimension.'

        # compute jacobians symbolically
        df_dx = ca.jacobian(self.param['x_next'],self.param['x'])
        df_du = ca.jacobian(self.param['x_next'],self.param['u'])

        # check if df_dx and df_du are constant
        if ca.jacobian(ca.vcat([*ca.symvar(df_dx),*ca.symvar(df_du)]),ca.vertcat(self.param['x'],self.param['u'])).is_zero():
            self.__type = 'affine'
        else:
            self.__type = 'nonlinear'

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
            df_dx_nom = ca.jacobian(self.x_next_nom,self.x)
            df_du_nom = ca.jacobian(self.x_next_nom,self.u)

            # compute jacobians
            start = time.time()
            A_nom = ca.Function('A_nom', p_nom, [df_dx_nom], p_nom_names, ['A_nom'], options)
            comp_time_dict = comp_time_dict | {'A_nom':time.time()-start}
            start = time.time()
            B_nom = ca.Function('B_nom', p_nom, [df_du_nom], p_nom_names, ['B_nom'], options)
            comp_time_dict = comp_time_dict | {'B_nom':time.time()-start}

            # save in dynamics
            self.__A_nom = A_nom
            self.__B_nom = B_nom
        else:
            # otherwise, copy exact dynamics
            A_nom = A
            B_nom = B

        # test that A and B work correctly
        try:
            A_test = A(*p_init)
            B_test = B(*p_init)
        except:
            raise Exception('Functions A and B are incompatible with the parameters x,u,d,w you passed.')
        try:
            A_test_nom = A_nom(*p_init_nom)
            B_test_nom = B_nom(*p_init_nom)
        except:
            raise Exception('Functions A_nom and B_nom are incompatible with the parameters x,u you passed.')
        
        assert A_test.shape[0] == self.dim['x'] and A_test.shape[1] == self.dim['x'], 'Function A does not return the correct dimension.'
        assert B_test.shape[0] == self.dim['x'] and B_test.shape[1] == self.dim['u'], 'Function B does not return the correct dimension.'
        assert A_test_nom.shape[0] == self.dim['x'] and A_test_nom.shape[1] == self.dim['x'], 'Function A_nom does not return the correct dimension.'
        assert B_test_nom.shape[0] == self.dim['x'] and B_test_nom.shape[1] == self.dim['u'], 'Function B_nom does not return the correct dimension.'

        # store in dynamics
        self.__A = A
        self.__A_nom = A_nom
        self.__B = B
        self.__B_nom = B_nom

        # store computation times (if compile is true)
        if compile:
            self.__compTimes = self.__compTimes | comp_time_dict

    @property
    def param(self):
        return {k: v for k, v in self.__sym.var.items() if k in ['x','u','d','w']}

    @property
    def param_nominal(self):
        return {k: v for k, v in self.__sym.var.items() if k in ['x','u']}
    
    @property
    def dim(self):
        return self.__sym.dim

    @property
    def init(self):
        return {k: v for k, v in self.__init.items()}
    
    def __setInit(self, init):
        self.__sym.setInit(init)

    def linearize(self,N,method='trajectory'):

        """
        This function constructs the prediction model for the MPC problem. There are multiple options:

            1. If the model is affine, then A,B,c are the true nominal dynamics of the system, this happens
               if self.type == 'affine'.
               
            2. The model can be linearized around the initial state (method = 'state').
               In this case, the linearization trajectory is a single input u_lin.

            3. (default) The model can be linearized along a trajectory (method = 'trajectory').
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
        elif method == 'initial_state':

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
        elif method == 'trajectory':

            # create symbolic variable for linearization trajectory
            y_lin = MSX.sym('y_lin',(n_x+n_u)*N,1)

            # extract linearization input and state
            x_lin = y_lin[:N*n_x]
            u_lin = y_lin[N*n_x:]

            # preallocate matrices
            A_list, B_list, c_list = [], [], []

            # extract linearization points by splitting the x_lin and u_lin vectors
            x_lin_list = ca.vertsplit(x_lin,n_x)
            u_lin_list = ca.vertsplit(u_lin,n_u)

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

        # create output dictionary
        out = {'A':A_list, 'B':B_list, 'c':c_list, 'y_lin':y_lin, 'x':x}

        return out

    @property
    def x(self):
        return self.__x

    @property
    def u(self):
        return self.__u

    @property
    def d(self):
        return self.__d

    @property
    def w(self):
        return self.__w

    @property
    def x_next(self):
        return self.__x_next

    @property
    def x_next_nom(self):
        return self.__x_next_nom
    
    @property
    def x_dot(self):
        return self.__x_dot

    @property
    def x_dot_nom(self):
        return self.__x_dot_nom
    
    @property
    def A(self):
        return self.__A

    @property
    def B(self):
        return self.__B

    @property
    def f(self):
        return self.__f

    @property
    def fc(self):
        return self.__fc

    @property
    def A_nom(self):
        return self.__A_nom
    
    @property
    def B_nom(self):
        return self.__B_nom

    @property
    def f_nom(self):
        return self.__f_nom

    @property
    def fc_nom(self):
        return self.__fc_nom

    @property
    def type(self):
        return self.__type
    
    @property
    def compTimes(self):
        return self.__compTimes

    # overwrite the __dir__ method
    def __dir__(self):
        return [attr for attr in super().__dir__() if not attr.startswith('_dynamics__')]