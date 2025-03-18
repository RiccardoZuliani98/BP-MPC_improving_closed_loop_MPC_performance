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
                      'cost':['Qx','Ru','Se','qx','ru','se'],
                      'cst':['Hx','Hx_e','Hu','hx','hu']}
    
    # expected dimensions
    __expected_dimensions = {'model':{'A':['x','x'],'B':['x','u'],'c':['x','one']},
                             'cost':{'Qx':['x','x'],'Ru':['u','u'],'Qn':['x','x'],'x_ref':['x','one'],'u_ref':['u','one']},
                             'cst':{'Hx':['cst_x','x'],'Hx_e':['cst_x','eps'],'Hu':['cst_u','u'],'hx':['cst_x','one'],'hu':['cst_u','one']}}
    
    # allowed inputs to __init__ and updateMPC
    allowed_inputs = ['N','model','cost','cst']


    def __init__(self,N=None,model=None,cost=None,cst=None,MSX=SX):

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
        inputs = {k:v for k,v in locals().items() if k in self.allowed_inputs and v is not None}

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

                # extract what dimensions are expected
                expected_dim = self.__expected_dimensions[v][w]

                # check correctness of dimensions
                for dim, val in zip(expected_dim,getattr(self,v)[w].shape):

                    # val should be a list of length N
                    if isinstance(val,list):
                        if len(val) != self.N:
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
                        self.__add_to_dim({dim:val})

                    # otherwise, check if it matches existing dimensions
                    else:
                        if all([v != val for v in self.dim[dim]]):
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
                A_mat = [self.MSX(A) for A in A_mat]
            except:
                raise Exception('A must be a list of matrices')
            
            # update state dimension
            self.__add_to_dim({'x':A_mat[0].shape[0]})

        else:

            # convert to chosen symbolic variable type
            try:
                A_mat = self.MSX(A_mat)
            except:
                raise Exception('A must be a matrix.')
            
            # update state dimension
            self.__add_to_dim({'x':A_mat.shape[0]})

            # if A is passed as a single value then the MPC must have a horizon
            if 'N' not in self.dim:
                raise Exception('N must be passed if A is a single matrix.')

            # create list of A matrices
            A_mat = [A_mat] * self.N

        # check if B is passed as list
        if isinstance(B_mat,list):

            # check that length is correct
            if len(B_mat) != self.N:
                raise Exception('B must be a list of length N.')
            try:
                B_mat = [self.MSX(B) for B in B_mat]
            except:
                raise Exception('B must be a list of matrices')
            
            # update input dimension
            self.__add_to_dim({'u':B_mat[0].shape[1]})

        else:

            # convert to chosen symbolic variable type
            try:
                B_mat = self.MSX(B_mat)
            except:
                raise Exception('B must be a matrix.')
            B_mat = [B_mat] * self.N

            # update input dimension
            self.__add_to_dim({'u':B_mat.shape[1]})

        # check if c is passed
        if 'c' in model:

            # extract c
            c_mat = model['c']

            # check if c is passed as list
            if isinstance(c_mat,list):

                # check that length is correct
                if len(c_mat) != self.N:
                    raise Exception('c must be a list of length N.')
                try:
                    c_mat = [self.MSX(c) for c in c_mat]
                except:
                    raise Exception('c must be a list of vectors')
            
            else:
                try:
                    c_mat = self.MSX(c_mat)
                except:
                    raise Exception('c must be a vector.')
                c_mat = [c_mat] * self.N

        else:
            c_mat = [self.MSX(self.dim['x'],1)] * self.N

        # extract initial state
        if 'x0' in model:
            x0 = model['x0']

            # check dimension
            if x0.shape[0] != self.dim['x']:
                raise Exception('Initial state must have the right dimension.')
        else:
            raise Exception('Initial state must be passed.')

        # patch first entry
        c_mat[0] = - A_mat@x0

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
            - 'c_lin': linear penalty on slack variables, scalar [optional]
            - 'c_quad': quadratic penalty on slack variables, scalar [optional]

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
                    Qx = [self.MSX(Q) for Q in Qx]
                except:
                    raise Exception('Qx must be a list of matrices')
                
                # update state dimension
                self.__add_to_dim({'x':Qx[0].shape[0]})

            else:

                # convert Qx to chosen symbolic variable type
                try:
                    Qx = self.MSX(Qx)
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
                        Qn = self.MSX(Qn)
                    except:
                        raise Exception('Qn must be a matrix.')

                    # create list of Qx matrices including terminal cost
                    Qx = [Qx] * (self.N-1)
                    Qx.append(Qn)
                
                # if Qn is not passed assume Qn = Qx
                else:
                    Qx = [Qx] * self.N

        else:
            raise Exception('Qx must be passed.')

        pass
        
        if 'Ru' in value:
            Ru = value['Ru']
            if isinstance(Ru,list):

                # check that length is correct
                if len(Ru) != self.N:
                    raise Exception('Ru must be a list of length N.')
                try:
                    Ru = [self.MSX(R) for R in Ru]
                except:
                    raise Exception('Ru must be a list of matrices')
                
                # update input dimension
                self.__add_to_dim({'u':Ru[0].shape[0]})

            else:
                try:
                    Ru = self.MSX(Ru)
                except:
                    raise Exception('Ru must be a matrix.')
                Ru = [Ru] * self.N

                # update input dimension
                self.__add_to_dim({'u':Ru.shape[0]})
        
        else:
            raise Exception('Ru must be passed.')
        
        # check if x_ref is passed
        if 'x_ref' in value:
            x_ref = value['x_ref']
            if isinstance(x_ref,list):
                if len(x_ref) != self.N:
                    raise Exception('x_ref must be a list of length N.')
                try:
                    x_ref = [self.MSX(x) for x in x_ref]
                except:
                    raise Exception('x_ref must be a list of vectors')
            else:
                try:
                    x_ref = self.MSX(x_ref)
                except:
                    raise Exception('x_ref must be a vector.')
                x_ref = [x_ref] * self.N

        else:
            x_ref = [self.MSX(self.dim['x'],1)] * self.N

        # check if u_ref is passed
        if 'u_ref' in value:
            u_ref = value['u_ref']
            if isinstance(u_ref,list):
                if len(u_ref) != self.N:
                    raise Exception('u_ref must be a list of length N.')
                try:
                    u_ref = [self.MSX(u) for u in u_ref]
                except:
                    raise Exception('u_ref must be a list of vectors')
            else:
                try:
                    u_ref = self.MSX(u_ref)
                except:
                    raise Exception('u_ref must be a vector.')
                u_ref = [u_ref] * self.N
        
        else:
            u_ref = [self.MSX(self.dim['u'],1)] * self.N

        # check if s_lin is passed
        if 's_lin' in value:
            s_lin = value['s_lin']
            try:
                s_lin = self.MSX(s_lin)
            except:
                raise Exception('s_lin must be a scalar.')

        # check if s_quad is passed
        if 's_quad' in value:
            s_quad = value['s_quad']
            try:
                s_quad = self.MSX(s_quad)
            except:
                raise Exception('s_quad must be a scalar.')

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
                    Hx = [self.MSX(H) for H in Hx]
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
                    hx = [self.MSX(h) for h in hx]
                except:
                    raise Exception('hx must be a list of vectors')

            else:

                # convert to chosen symbolic variable type
                try:
                    Hx = self.MSX(Hx)
                except:
                    raise Exception('Hx must be a matrix.')
                
                # update state dimension
                self.__add_to_dim({'x':Hx.shape[1]})

                # if Hx is passed as a single value then the MPC must have a horizon
                if 'N' not in self.dim:
                    raise Exception('N must be passed if Hx is a single matrix.')

                # create list of Hx matrices
                Hx = [Hx] * self.N

                # check if hx is passed
                if 'hx' not in value:
                    raise Exception('hx must be passed with Hx.')
                
                # check that hx is a single matrix
                try:
                    hx = self.MSX(value['hx'])
                except:
                    raise Exception('hx must be a vector.')

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
                    Hx_e = [self.MSX(H) for H in Hx_e]
                except:
                    raise Exception('Hx_e must be a list of matrices')
                
                # update state dimension
                self.__add_to_dim({'eps':Hx_e[0].shape[1]})

            else:

                # convert to chosen symbolic variable type
                try:
                    Hx_e = self.MSX(Hx_e)
                except:
                    raise Exception('Hx_e must be a matrix.')
                
                # update state dimension
                self.__add_to_dim({'eps':Hx_e.shape[1]})

                # if Hx is passed as a single value then the MPC must have a horizon
                if 'N' not in self.dim:
                    raise Exception('N must be passed if Hx_e is a single matrix.')

                # create list of Hx matrices
                Hx_e = [Hx_e] * self.N

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
                    Hu = [self.MSX(H) for H in Hu]
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
                if len(value['hu']) != self.N:
                    raise Exception('hu must be a list of length N.')
                
                # extract vectors
                hu = value['hu']

                # check that hu can be converted to correct symbolic type
                try:
                    hu = [self.MSX(h) for h in hu]
                except:
                    raise Exception('hu must be a list of vectors')

            else:

                # convert to chosen symbolic variable type
                try:
                    Hu = self.MSX(Hu)
                except:
                    raise Exception('Hu must be a matrix.')
                
                # update input dimension
                self.__add_to_dim({'u':Hu.shape[1]})

                # if Hu is passed as a single value then the MPC must have a horizon
                if 'N' not in self.dim:
                    raise Exception('N must be passed if Hu is a single matrix.')

                # create list of Hu matrices
                Hu = [Hu] * self.N

                # check if hu is passed
                if 'hu' not in value:
                    raise Exception('hu must be passed with Hu.')
                
                # check that hu is a single matrix
                try:
                    hu = self.MSX(value['hu'])
                except:
                    raise Exception('hu must be a vector.')


    def __MPC_to_sparse_QP(self,A_list,B_list,c_list,Qx,Qn,Ru,Hx,hx,Hu,hu,Hx_e=None,x_ref=None,u_ref=None,s_lin=None,s_quad=None):

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

        And the outputs are the matrices G, g, F, f, Q, Qinv=inv(Q), and the dictionary idx containing the indexing of the output optimization
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

        # get symbolic variable type
        MSX = self.__MSX

        # initialize slack to False
        slack = False
        
        # check if linear slack penalty is passed
        if s_lin is not None:

            # if penalty is greater than 0, set slack mode to true
            if self.cost['s_lin'] > 0:
                slack = True
            
            # if penalty is negative, raise exception
            elif self.cost['s_lin'] <= 0:
                raise Exception('Linear slack penalty must be positive.')
            
        # check if quadratic slack penalty is passed
        if s_quad is not None:

            # if penalty is greater than 0, set slack mode to true
            if self.cost['s_quad'] > 0:
                slack = True

            # if penalty is negative, raise exception
            elif self.cost['s_quad'] <= 0:
                raise Exception('Quadratic slack penalty must be nonnegative.')

        # check if Hx_e was passed 
        if Hx_e is not None:
            slack = True
            Hx_e = MSX(Hx_e)

        # if slack is passed, then ensure Hx_e is not None
        if slack and (Hx_e is None):
            Hx_e = MSX.eye(Hx.shape[0])

        # add slack dimension to dimension vector
        self.__add_to_dim({'eps':Hx_e.shape[1]})

        # extract dimensions
        n = self.dim

        if slack:

            # linear penalty
            try:
                s_lin = MSX(self.mpc.cost['s_lin'])
            except:
                s_lin = MSX(0) # default value
                pass

            # quadratic penalty
            try:
                s_quad = MSX(self.mpc.cost['s_quad'])
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
            Hx = hcat([Hx,MSX(Hx.shape[0],n['N']*n['u']),-Hx_e])

            # add columns associated to state and slack variables
            Hu = hcat([MSX(Hu.shape[0],n['N']*n['x']),Hu,MSX(Hu.shape[0],n['eps'])])

            # add nonnegativity constraints on slack variables
            He = hcat([MSX(n['eps'],n['N']*(n['x']+n['u'])),-MSX.eye(n['eps'])])
            he = MSX(n['eps'],1)

            # create inequality constraint matrices
            try:
                G = cse(sparsify(vcat([Hx,Hu,He])))
                g = cse(sparsify(vcat([hx,hu,he])))
            except:
                G = vcat([Hx,Hu,He])
                g = vcat([hx,hu,he])

        else:

            # add columns associated to input and slack variables
            Hx = hcat([Hx,MSX(Hx.shape[0],n['N']*n['u'])])

            # add columns associated to state and slack variables
            Hu = hcat([MSX(Hu.shape[0],n['N']*n['x']),Hu])
                
            # create inequality constraint matrices
            try:
                G = cse(sparsify(vcat([Hx,Hu])))
                g = cse(sparsify(vcat([hx,hu])))
            except:
                G = vcat([Hx,Hu])
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
        Q = blockcat(Qx,MSX((n['N']-1)*n['x'],n['x']),MSX(n['x'],(n['N']-1)*n['x']),Qn)

        # add input cost
        Q = blockcat(Q,MSX(n['N']*n['x'],n['N']*n['u']),MSX(n['N']*n['u'],n['N']*n['x']),Ru)

        # append cost applied to slack variable
        if slack:
            Q = blockcat(Q,MSX(Q.shape[0],n['eps']),MSX(n['eps'],Q.shape[0]),s_quad*MSX.eye(n['eps']))

        # inverse of quadratic cost matrix
        Qinv = inv_minor(Q) #inv(Q)

        # create linear part of the cost
        if slack:
            q = vcat([(-x_ref.T@blockcat(Qx,MSX(n['x']*(n['N']-1),n['x']),MSX(n['x'],n['x']*(n['N']-1)),Qn)).T,(-u_ref.T@Ru).T,s_lin*MSX.ones(n['eps'],1)])
        else:
            q = vcat([(-x_ref.T@blockcat(Qx,MSX(n['x']*(n['N']-1),n['x']),MSX(n['x'],n['x']*(n['N']-1)),Qn)).T,(-u_ref.T@Ru).T])

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

        return G,g,F,f,Q,Qinv,q,idx

    # overwrite the __dir__ method
    def __dir__(self):
        return [attr for attr in super().__dir__() if not attr.startswith('_MPC__')]