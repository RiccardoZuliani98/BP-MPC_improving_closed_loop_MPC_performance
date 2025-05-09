import casadi as ca
from copy import copy

"""
TODO:
* descriptions
"""

class Symb:

    def __init__(self):
        self._var = {}
        self._dim = {}
        self._init = {}

    @property
    def dim(self):
        return self._dim
    
    @property
    def var(self):
        return self._var

    @property
    def init(self):
        # return {key:val for key,val in self.init.items() if val is not None}
        return self._init
    
    def set_init(self,data):

        assert isinstance(data,dict), 'Pass a dictionary to initialize variables'

        for name,value in data.items():
        
            assert name in self._var, 'Cannot initialize variable that does not exist'

            if isinstance(value,list):

                try:
                    value = [ca.DM(item) for item in value]
                except:
                    raise Exception('Cannot convert init value to DM')

                assert all([self.var[name].shape == item.shape for item in value]), 'Dimension of initialization does not match dimension of symbolic variable'

            else:

                try:
                    value = ca.DM(value)
                except:
                    raise Exception('Cannot convert init value to DM')

                assert self.var[name].shape == value.shape, 'Dimension of initialization does not match dimension of symbolic variable'

            # add to initialization
            self._init[name] = ca.DM(value)
       
    def get_var(self,name):
        return self.var[name],self.dim[name],self.init[name]
    
    def addVar(self,name,var,init=None):

        if init is not None:

            assert var.shape == init.shape, 'Initialization must have the same dimension as the symbolic variable'
            assert isinstance(init,ca.DM), 'Initialization must of type casadi.DM'

        try:
            var_SX = ca.SX(var)
        except:
            raise Exception('Variable cannot be converted to SX')

        self._var[name] = var
        self._dim[name] = var.shape[0] if var.shape[1] == 1 else var.shape
        self._init[name] = init

    def addDim(self,name,val):

        assert isinstance(name,str), 'Name of variable must be a string'
        assert isinstance(val,int) and val>=0, 'Value of dimension must be a nonnegative integer'

        self._dim[name] = val

    def __add__(self,other):

        assert isinstance(other,Symb), 'Type of addends must match'

        # create copy of class
        self_copy = copy(self)

        self_copy._dim = self.dim | other.dim
        self_copy._var = self.var | other.var
        self_copy._init = self.init | other.init

        return self_copy

    def __iadd__(self,other):

        assert type(self.type()) is type(other.type()), 'Type of addends must match'

        self._dim = self.dim | other.dim
        self._var = self.var | other.var
        self._init = self.init | other.init

        return self
    
    def copy(self,vars2keep=None):

        # create copy of class
        self_copy = self.__class__()

        if vars2keep is not None:
            self_copy._dim = {key:val for key,val in self.dim.items() if key in vars2keep}
            self_copy._var = {key:val for key,val in self.var.items() if key in vars2keep}
            self_copy._init = {key:val for key,val in self.init.items() if key in vars2keep}
        else:
            self_copy._dim = self.dim
            self_copy._var = self.var
            self_copy._init = self.init

        return self_copy