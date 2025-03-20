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

        - A: list of length N of matrices (n_x,n_x)
        - B: list of length N of matrices (n_x,n_u)
        - x0: symbolic variable representing the initial state (n_x,1)
        - c: list of length N of matrices (n_x,1) [optional, defaults to 0]
            
    where the dynamics are given by x[t+1] = A[t]x[t] + B[t]u[t] + c[t], with x[0] = x0.
    """
    __model = {}

    """
    Cost dictionary with keys
                
        - 'Qx': state stage cost, list of length N of matrices (n_x,n_x)
        - 'Ru': input stage cost, list of length N of matrices (n_u,n_u)
        - 'x_ref': state reference, list of length N of vectors (n_x,1) [optional, defaults to 0]
        - 'u_ref': reference input, list of length N of vectors (n_u,1) [optional, defaults to 0]
        - 's_lin': linear penalty on slack variables, nonnegative scalar [optional, defaults to 0]
        - 's_quad': quadratic penalty on slack variables, positive scalar [optional, defaults to 0]

    where the stage cost is given by
        
        (x[t]-x_ref[t])'Qx[t](x[t]-x_ref[t]) + (u[t]-u_ref[t])'Ru[t](u[t]-u_ref[t]) + s_lin*e[t] + s_quad*e[t]**2
    """
    __cost = {}

    """
     Constraints dictionary with keys
    
        - 'Hx': list of length N of matrices (=,n_x)
        - 'hx': list of length N of vectors (=,1)
        - 'Hx_e': list of length N of matrices (=,n_eps) [optional, defaults to zero]
        - 'Hu': list of length N of matrices (-,n_u)
        - 'hu': list of length N of vectors (-,1)
        
    where the constraints at each time-step are
        
        Hx[t]*x[t] <= hx[t] - Hx_e[t]*e[t],
        Hu[t]*u[t] <= hu[t],
            
    where e[t] denotes the slack variables.
    """
    __cst = {}

    # keys allowed in dictionaries
    __allowed_keys = {'dim':['N','u','x','eps','cst_x','cst_u'],
                      'model':['A','B','c'],
                      'cost':['Qx','Ru','x_ref','u_ref','s_lin','s_quad'],
                      'cst':['Hx','Hx_e','Hu','hx','hu']}
    
    # expected dimensions
    __expected_dimensions = {'model':{'A':['x','x'],'B':['x','u'],'c':['x','one']},
                             'cost':{'Qx':['x','x'],'Ru':['u','u'],'x_ref':['x','one'],'u_ref':['u','one'],'s_lin':['one','one'],'s_quad':['one','one']},
                             'cst':{'Hx':['cst_x','x'],'Hx_e':['cst_x','eps'],'Hu':['cst_u','u'],'hx':['cst_x','one'],'hu':['cst_u','one']}}
    
    # allowed inputs to __init__ and updateMPC
    allowed_inputs = ['model','cost','cst']


    def __init__(self,N=None,model=None,cost=None,cst=None,MSX='SX'):

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
        inputs = {k:v for k,v in locals().items() if k in ['N','model','cost','cst'] and v is not None}

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
                - x0: symbolic variable representing the initial state (n_x,1)
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

                if w not in getattr(self,v):
                    continue

                # extract what dimensions are expected
                expected_dim = self.__expected_dimensions[v][w]

                # check correctness of dimensions
                for i in range(2):

                    # extract expected dimension (string)
                    dim = expected_dim[i]

                    # extract actual dimension (list of integers)
                    val = [elem.shape[i] for elem in getattr(self,v)[w]]

                    # val should be a list of length N
                    if isinstance(val,list):
                        if len(val) != self.dim['N']:
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
                        self.__add_to_dim({dim:val[0]})

                    # otherwise, check if it matches existing dimensions
                    else:
                        if all([v != self.dim[dim] for v in val]):
                            raise Exception('Attribute {} must have the right dimensions.'.format(v))

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
        self.__add_to_dim({'N':value})

    @property
    def model(self):
        return self.__model
    
    def __set_model(self, model):

        """
        This function sets the cost dictionary. The input must be a dictionary with keys

            - A: list of matrices (of length N) or a single matrix (n_x,n_x)
            - B: list of matrices (of length N) or a single matrix (n_x,n_u)
            - x0: symbolic variable representing the initial state (n_x,1)
            - c: list of matrices (of length N) or a single matrix (n_x,1) [optional, defaults to 0]

        """

        # extract matrices
        A_mat = model['A']
        B_mat = model['B']

        # check if a list is passed
        if isinstance(A_mat,list):

            # update horizon of the MPC
            self.__set_N(len(A_mat))

            # convert to chosen symbolic variable type
            try:
                A_mat = [self.__MSX(A) for A in A_mat]
            except:
                raise Exception('A must be a list of matrices')
            
            # update state dimension
            self.__add_to_dim({'x':A_mat[0].shape[0]})

        else:

            # convert to chosen symbolic variable type
            try:
                A_mat = self.__MSX(A_mat)
            except:
                raise Exception('A must be a matrix.')
            
            # update state dimension
            self.__add_to_dim({'x':A_mat.shape[0]})

            # if A is passed as a single value then the MPC must have a horizon
            if 'N' not in self.dim:
                raise Exception('N must be passed if A is a single matrix.')

            # create list of A matrices
            A_mat = [A_mat] * self.dim['N']

        # check if B is passed as list
        if isinstance(B_mat,list):

            # check that length is correct
            if len(B_mat) != self.dim['N']:
                raise Exception('B must be a list of length N.')
            try:
                B_mat = [self.__MSX(B) for B in B_mat]
            except:
                raise Exception('B must be a list of matrices')
            
            # update input dimension
            self.__add_to_dim({'u':B_mat[0].shape[1]})

        else:

            # convert to chosen symbolic variable type
            try:
                B_mat = self.__MSX(B_mat)
            except:
                raise Exception('B must be a matrix.')
            B_mat = [B_mat] * self.dim['N']

            # update input dimension
            self.__add_to_dim({'u':B_mat[0].shape[1]})

        # check if c is passed
        if 'c' in model:

            # extract c
            c_mat = model['c']

            # check if c is passed as list
            if isinstance(c_mat,list):

                # check that length is correct
                if len(c_mat) != self.dim['N']:
                    raise Exception('c must be a list of length N.')
                try:
                    c_mat = [self.__MSX(c) for c in c_mat]
                except:
                    raise Exception('c must be a list of vectors')
            
            else:
                try:
                    c_mat = self.__MSX(c_mat)
                except:
                    raise Exception('c must be a vector.')
                c_mat = [c_mat] * self.dim['N']

        else:
            c_mat = [self.__MSX(self.dim['x'],1)] * self.dim['N']

        # extract initial state
        if 'x0' in model:
            x0 = model['x0']

            # check dimension
            if x0.shape[0] != self.dim['x']:
                raise Exception('Initial state must have the right dimension.')
        else:
            raise Exception('Initial state must be passed.')

        # patch first entry
        # c_mat[0] = - A_mat[0]@x0

        # store matrices
        self.__model = {'A':A_mat,'B':B_mat,'c':c_mat,'x0':x0}

    @property
    def cost(self):
        return self.__cost
    
    def __set_cost(self, value):

        """
        This function sets the cost dictionary. The input must be a dictionary with specific combinations of keys.
        It must contain either

            - 'Qx': stage cost, matrix (n_x,n_x)
            - 'Qn': terminal cost, matrix (n_x,n_x) [optional, defaults to Qx]
            - 'Ru': input cost, matrix (n_u,n_u)
            - 'x_ref': state reference, vector (n_x,1) [optional, defaults to 0]
            - 'u_ref': input reference, vector (n_u,1) [optional, defaults to 0]
            - 'c_lin': linear penalty on slack variables, scalar [optional]
            - 'c_quad': quadratic penalty on slack variables, scalar [optional]
        
        or

            - 'Qx': stage cost, list of matrices (of length N)
            - 'Ru': input cost, list of matrices (of length N)
            - 'x_ref': state reference, list of vectors (of length N) [optional, defaults to 0]
            - 'u_ref': input reference, list of vectors (of length N) [optional, defaults to 0]
            - 'c_lin': linear penalty on slack variables, list of scalars (of length N) [optional]
            - 'c_quad': quadratic penalty on slack variables, list of scalars (of length N) [optional]

        """
        
        # check that value is a dictionary
        if not isinstance(value, dict):
            raise Exception('Cost must be passed as a dictionary.')
        
        # check if Qx is passed as value or as list
        if 'Qx' in value:
            Qx = value['Qx']
            if isinstance(Qx,list):

                # update horizon of the MPC
                self.__set_N(len(Qx))

                # if Qx is passed as list, Qn should not be passed
                if 'Qn' in value:
                    raise Exception('Qn must not be passed if Qx is a list.')
                
                # convert Qx to chosen symbolic variable type
                try:
                    Qx = [self.__MSX(Q) for Q in Qx]
                except:
                    raise Exception('Qx must be a list of matrices')
                
                # update state dimension
                self.__add_to_dim({'x':Qx[0].shape[0]})

            else:

                # convert Qx to chosen symbolic variable type
                try:
                    Qx = self.__MSX(Qx)
                except:
                    raise Exception('Qx must be a matrix.')
                
                # update state dimension
                self.__add_to_dim({'x':Qx.shape[0]})

                # if Qx is passed as a single value then the MPC must have a horizon
                if 'N' not in self.dim:
                    raise Exception('N must be passed if Qx is a single matrix.')

                # check if Qn is passed
                if 'Qn' in value:
                    
                    # Qn must be a matrix
                    Qn = value['Qn']
                    try:
                        Qn = self.__MSX(Qn)
                    except:
                        raise Exception('Qn must be a matrix.')

                    # create list of Qx matrices including terminal cost
                    Qx = [Qx] * (self.dim['N']-1)
                    Qx.append(Qn)
                
                # if Qn is not passed assume Qn = Qx
                else:
                    Qx = [Qx] * self.dim['N']

        else:
            raise Exception('Qx must be passed.')

        pass
        
        if 'Ru' in value:
            Ru = value['Ru']
            if isinstance(Ru,list):

                # check that length is correct
                if len(Ru) != self.dim['N']:
                    raise Exception('Ru must be a list of length N.')
                try:
                    Ru = [self.__MSX(R) for R in Ru]
                except:
                    raise Exception('Ru must be a list of matrices')
                
                # update input dimension
                self.__add_to_dim({'u':Ru[0].shape[0]})

            else:
                try:
                    Ru = self.__MSX(Ru)
                except:
                    raise Exception('Ru must be a matrix.')
                Ru = [Ru] * self.dim['N']

                # update input dimension
                self.__add_to_dim({'u':Ru[0].shape[0]})
        
        else:
            raise Exception('Ru must be passed.')

        # update cost
        self.__cost = {'Qx':Qx,'Ru':Ru}

        # check if x_ref is passed
        if 'x_ref' in value:
            x_ref = value['x_ref']
            if isinstance(x_ref,list):
                if len(x_ref) != self.dim['N']:
                    raise Exception('x_ref must be a list of length N.')
                try:
                    x_ref = [self.__MSX(x) for x in x_ref]
                except:
                    raise Exception('x_ref must be a list of vectors')
            else:
                try:
                    x_ref = self.__MSX(x_ref)
                except:
                    raise Exception('x_ref must be a vector.')
                x_ref = [x_ref] * self.dim['N']

        else:
            x_ref = [self.__MSX(self.dim['x'],1)] * self.dim['N']

        # check if u_ref is passed
        if 'u_ref' in value:
            u_ref = value['u_ref']
            if isinstance(u_ref,list):
                if len(u_ref) != self.dim['N']:
                    raise Exception('u_ref must be a list of length N.')
                try:
                    u_ref = [self.__MSX(u) for u in u_ref]
                except:
                    raise Exception('u_ref must be a list of vectors')
            else:
                try:
                    u_ref = self.__MSX(u_ref)
                except:
                    raise Exception('u_ref must be a vector.')
                u_ref = [u_ref] * self.dim['N']
        
        else:
            u_ref = [self.__MSX(self.dim['u'],1)] * self.dim['N']

        # update cost
        self.__cost = self.__cost | {'x_ref':x_ref,'u_ref':u_ref}

        # check if s_lin is passed
        if 's_lin' in value:
            s_lin = value['s_lin']
            try:
                s_lin = self.__MSX(s_lin)
            except:
                raise Exception('s_lin must be a scalar.')
            
            s_lin = [s_lin] * self.dim['N']
            
            # update cost
            self.__cost = self.__cost | {'s_lin':s_lin}

        # check if s_quad is passed
        if 's_quad' in value:
            s_quad = value['s_quad']
            try:
                s_quad = self.__MSX(s_quad)
            except:
                raise Exception('s_quad must be a scalar.')
            
            s_quad = [s_quad] * self.dim['N']
            
            # update cost
            self.__cost = self.__cost | {'s_quad':s_quad}
            
    @property
    def cst(self):
        return self.__cst
    
    def __set_cst(self, value):
        
        """
        This functions sets up the constraint dictionary. The input must be a dictionary with keys
            
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

        # empty existing constraints
        self.__cst = {}

        # check that value is a dictionary
        if not isinstance(value, dict):
            raise Exception('Constraints must be passed as a dictionary.')
        
        # check if Hx is passed as value or as list
        if 'Hx' in value:

            # extract matrices
            Hx = value['Hx']

            # check if a list is passed
            if isinstance(Hx,list):

                # update horizon of the MPC
                self.__set_N(len(Hx))

                # convert to chosen symbolic variable type
                try:
                    Hx = [self.__MSX(H) for H in Hx]
                except:
                    raise Exception('Hx must be a list of matrices')
                
                # update state dimension
                self.__add_to_dim({'x':Hx[0].shape[1]})

                # check if hx is passed
                if 'hx' not in value:
                    raise Exception('hx must be passed with Hx.')
                
                # check that hx is a list
                if not isinstance(value['hx'],list):
                    raise Exception('hx must be a list of vectors.')
                
                # check that list has correct dimension
                if len(value['hx']) != self.N:
                    raise Exception('hx must be a list of length N.')
                
                # extract vectors
                hx = value['hx']

                # check that hx can be converted to correct symbolic type
                try:
                    hx = [self.__MSX(h) for h in hx]
                except:
                    raise Exception('hx must be a list of vectors')

            else:

                # convert to chosen symbolic variable type
                try:
                    Hx = self.__MSX(Hx)
                except:
                    raise Exception('Hx must be a matrix.')
                
                # update state dimension
                self.__add_to_dim({'x':Hx.shape[1]})

                # if Hx is passed as a single value then the MPC must have a horizon
                if 'N' not in self.dim:
                    raise Exception('N must be passed if Hx is a single matrix.')

                # create list of Hx matrices
                Hx = [Hx] * self.dim['N']

                # check if hx is passed
                if 'hx' not in value:
                    raise Exception('hx must be passed with Hx.')
                
                # check that hx is a single matrix
                try:
                    hx = self.__MSX(value['hx'])
                except:
                    raise Exception('hx must be a vector.')
                
                hx = [hx] * self.dim['N']
                
            # store matrices
            self.__cst = self.__cst | {'Hx':Hx,'hx':hx}

        # check if Hx_e is passed
        if 'Hx_e' in value:

            # extract matrices
            Hx_e = value['Hx_e']

            # check if a list is passed
            if isinstance(Hx_e,list):

                # update horizon of the MPC
                self.__set_N(len(Hx_e))

                # convert to chosen symbolic variable type
                try:
                    Hx_e = [self.__MSX(H) for H in Hx_e]
                except:
                    raise Exception('Hx_e must be a list of matrices')
                
                # update state dimension
                self.__add_to_dim({'eps':Hx_e[0].shape[1]})

            else:

                # convert to chosen symbolic variable type
                try:
                    Hx_e = self.__MSX(Hx_e)
                except:
                    raise Exception('Hx_e must be a matrix.')
                
                # update state dimension
                self.__add_to_dim({'eps':Hx_e.shape[1]})

                # if Hx is passed as a single value then the MPC must have a horizon
                if 'N' not in self.dim:
                    raise Exception('N must be passed if Hx_e is a single matrix.')

                # create list of Hx matrices
                Hx_e = [Hx_e] * self.dim['N']

            # store matrices
            self.__cst = self.__cst | {'Hx_e':Hx_e}

        # check if Hu is passed
        if 'Hu' in value:
            
            # extract matrices
            Hu = value['Hu']

            # check if a list is passed
            if isinstance(Hu,list):

                # update horizon of the MPC
                self.__set_N(len(Hu))

                # convert to chosen symbolic variable type
                try:
                    Hu = [self.__MSX(H) for H in Hu]
                except:
                    raise Exception('Hu must be a list of matrices')
                
                # update input dimension
                self.__add_to_dim({'u':Hu[0].shape[1]})

                # check if hu is passed
                if 'hu' not in value:
                    raise Exception('hu must be passed with Hu.')
                
                # check that hu is a list
                if not isinstance(value['hu'],list):
                    raise Exception('hu must be a list of vectors.')
                
                # check that list has correct dimension
                if len(value['hu']) != self.dim['N']:
                    raise Exception('hu must be a list of length N.')
                
                # extract vectors
                hu = value['hu']

                # check that hu can be converted to correct symbolic type
                try:
                    hu = [self.__MSX(h) for h in hu]
                except:
                    raise Exception('hu must be a list of vectors')

            else:

                # convert to chosen symbolic variable type
                try:
                    Hu = self.__MSX(Hu)
                except:
                    raise Exception('Hu must be a matrix.')
                
                # update input dimension
                self.__add_to_dim({'u':Hu.shape[1]})

                # if Hu is passed as a single value then the MPC must have a horizon
                if 'N' not in self.dim:
                    raise Exception('N must be passed if Hu is a single matrix.')

                # create list of Hu matrices
                Hu = [Hu] * self.dim['N']

                # check if hu is passed
                if 'hu' not in value:
                    raise Exception('hu must be passed with Hu.')
                
                # check that hu is a single matrix
                try:
                    hu = self.__MSX(value['hu'])
                except:
                    raise Exception('hu must be a vector.')
                
                hu = [hu] * self.dim['N']
                
            # store matrices
            self.__cst = self.__cst | {'Hu':Hu,'hu':hu}

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
        
        # call internal function
        return self.__MPC2QP(A_list,B_list,c_list,Qx,Ru,Hx,hx,Hu,hu,Hx_e,x_ref,u_ref,s_lin,s_quad)

    def __MPC2QP(self,A_list,B_list,c_list,Qx,Ru,Hx,hx,Hu,hu,Hx_e=None,x_ref=None,u_ref=None,s_lin=None,s_quad=None):

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

        And the outputs are the matrices G, g, F, f, Q, Qinv=inv(Q), the dictionary idx containing the indexing of the output optimization
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

        #TODO: check dimensions

        def matrixify(M_list,MSX):

            # get dimensions
            N = len(M_list)
            n_col = M_list[0].shape[1]
            n_row = M_list[0].shape[0]

            # pad matrices with zeros
            M_list_pad = [ vcat([MSX(i*n_row,n_col),M_list[i],MSX((N-i-1)*n_row,n_col)]) for i in range(N) ]

            # stack horizontally
            return hcat(M_list_pad)
        
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

            # if penalty is greater than 0, set slack mode to true
            if all([elem > 0 for elem in s_lin]):
                slack = True
            
            # if penalty is negative, raise exception
            elif any([elem <= 0 for elem in s_lin]):
                raise Exception('Linear slack penalty must be positive.')
            
        # check if quadratic slack penalty is passed
        if s_quad is not None:

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
        # Qinv = inv_minor(Q)
        Qinv = inv(Q)

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
    
    def __makeDenseMPC(self,A_list,B_list,c_list,Qx,Ru,x_ref,u_ref,Hx,Hu,hx,hu):
        """
        Create a dictionary with all the ingredients needed to solve the MPC problem in dense form.
        
        The inputs are:
            
            - A_list, B_list, c_list: lists of matrices A, B, and c such that x[t+1] = A[t]@x[t] + B[t]@u[t] + c[t]
            - Qx, Qn, Ru, x_ref, u_ref: matrices defining the cost function (x-x_ref)'Qx(x-x_ref) + (u-u_ref)'Ru(u-u_ref)
            - Hx, Hu, hx, hu: polyhedral constraints Hx*x <= hx, Hu*u <= hu
        
        The output dictionary has keys:
        
            - 'G_x', 'G_u', 'g_c': matrices satisfying x = G_x*x0 + G_u*u + g_c
            - 'Qx', 'Ru', 'x_ref', 'u_ref': cost function (x-x_ref)'Qx(x-x_ref) + (u-u_ref)'Ru(u-u_ref)
            - 'Hx', 'Hu', 'hx', 'hu': polyhedral constraints Hx*x <= hx, Hu*u <= hu
        """

        # get symbolic variable type
        MSX = self.__MSX

        # extract dimensions
        n = self.dim

        # extract initial condition
        x0 = self.__model['x0']

        # start by constructing matrices G_x and G_u, and vector g_c such that
        # x = G_x*x_0 + G_u*u + g_c, where x = vec(x_1,x_2,...,x_N), 
        # u = vec(u_0,u_1,...,u_N-1).
        # To obtain g_c we need to multiply all the affine terms by a matrix
        # similar to G_u, which we call G_c.
        
        # first initialize G_u with the zero matrix
        G_u = MSX(n['x']*n['N'],n['u']*n['N'])

        # initialize G_c
        G_c = MSX(n['x']*n['N'],n['x']*n['N'])

        # we will need a tall matrix that will replace the columns of G_u
        # initially it is equal to a tall matrix full of zeros with an
        # identity matrix at the bottom.
        col = MSX.eye(n['N']*n['x'])[:,(n['N']-1)*n['x']:n['N']*n['x']]

        # loop through all columns of G_u (t ranges from N-1 to 1)
        for t in range(n['N']-1,0,-1):

            # get matrices A and at time-step t
            A_t = A_list[t]
            B_t = B_list[t]

            # update G_u matrix
            G_u[:,t*n['u']:(t+1)*n['u']] = col@B_t

            # update G_c matrix
            G_c[:,t*n['x']:(t+1)*n['x']] = col

            # update col by multiplying with A matrix and adding identity matrix
            col = col@A_t + MSX.eye(n['N']*n['x'])[:,(t-1)*n['x']:t*n['x']]

        # get linearized dynamics at time-step 0
        A_0 = A_list[0]
        B_0 = B_list[0]

        # correct first entry of c_list
        c_list[0] = c_list[0] + A_0@x0

        # now we only miss the left-most column (use x0 instead of x[:n['x']])
        G_u[:,:n['u']] = col@B_0

        # same for G_c
        G_c[:,:n['x']] = col

        # matrix G_x is simply col@A_0
        G_x = col@A_0

        # to create g_c concatenate vertically the entries in the list c_t_list
        # then multiply by G_c from the right
        c_t = -vcat(c_list)
        g_c = G_c@c_t

        # create dictionary
        out = {'G_x':G_x,'G_u':G_u,'g_c':g_c,'Qx':Qx,'Ru':Ru,'x_ref':x_ref,'u_ref':u_ref,'Hx':Hx,'Hu':Hu,'hx':hx,'hu':hu}

        return out

    # overwrite the __dir__ method
    def __dir__(self):
        return [attr for attr in super().__dir__() if not attr.startswith('_MPC__')]