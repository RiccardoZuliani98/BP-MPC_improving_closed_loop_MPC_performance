import casadi as ca

"""
TODO:

* descriptions

"""

class Symb:

    def __init__(self):
        self.__var = {}
        self.__dim = {}
        self.__init = {}

    @property
    def dim(self):
        return self.__dim
    
    @property
    def var(self):
        return self.__var

    @property
    def init(self):
        # return {key:val for key,val in self.init.items() if val is not None}
        return self.__init
    
    def setInit(self,data):

        assert isinstance(data,dict), 'Pass a dictionary to initialize variables'

        for name,value in data.items():
        
            assert name in self.__var, 'Cannot initialize variable that does not exist'

            try:
                value = ca.DM(value)
            except:
                raise Exception('Cannot convert init value to DM')

            assert self.var[name].shape == value.shape, 'Dimension of initialization does not match dimension of symbolic variable'

            try:
                self.__init[name] = ca.DM(value)
            except:
                raise Exception('Provided type cannot be converted to DM')
       
    def getVar(self,name):
        return self.var[name],self.dim[name],self.init[name]
    
    def addVar(self,name,var,init=None):

        if init is not None:

            assert var.shape == init.shape, 'Initialization must have the same dimension as the symbolic variable'
            assert isinstance(init,ca.DM), 'Initialization must of type casadi.DM'

        self.__var[name] = var
        self.__dim[name] = var.shape[0] if var.shape[1] == 1 else var.shape
        self.__init[name] = init

    def __add__(self,other):

        assert self.type is other.type, 'Type of addends must match'

        # create copy of class
        self_copy = self.__class__(self.__typeString)

        self_copy.__dim = self.dim | other.dim
        self_copy.__var = self.var | other.var
        self_copy.__init = self.init | other.init

        return self_copy

    def __iadd__(self,other):

        assert type(self.type()) is type(other.type()), 'Type of addends must match'

        self.__dim = self.dim | other.dim
        self.__var = self.var | other.var
        self.__init = self.init | other.init

        return self
    
    def copy(self,vars2keep=None):

        # create copy of class
        self_copy = self.__class__()

        if vars2keep is not None:
            self_copy.__dim = {key:val for key,val in self.dim.items() if key in vars2keep}
            self_copy.__var = {key:val for key,val in self.var.items() if key in vars2keep}
            self_copy.__init = {key:val for key,val in self.init.items() if key in vars2keep}
        else:
            self_copy.__dim = self.dim
            self_copy.__var = self.var
            self_copy.__init = self.init

        return self_copy