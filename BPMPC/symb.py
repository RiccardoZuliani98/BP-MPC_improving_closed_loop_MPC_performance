import casadi as ca

class Symb:

    def __init__(self,type='SX'):

        self.__setType(type)
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
    
    @property
    def type(self):
        return self.__type
    
    def getVar(self,name):
        return self.var[name],self.dim[name],self.init[name]
    
    def addVar(self,name,var,init=None):

        if init is not None:

            assert var.shape == init.shape, 'Initialization must have the same dimension as the symbolic variable'
            assert isinstance(init,ca.DM), 'Initialization must of type casadi.DM'

        self.__var[name] = var
        self.__dim[name] = var.shape[0] if var.shape[1] == 1 else var.shape
        self.__init[name] = init

    def __setType(self,type):

        assert type in ['SX','MX'], 'Supper types are SX and MX'

        self.__type = ca.SX if type == 'SX' else ca.MX

    def __add__(self,other):

        assert type(self.type()) is type(other.type()), 'Type of addends must match'

        self.__dim = self.dim | other.dim
        self.__var = self.var | other.var
        self.__init = self.init | other.init

        return self

    def __iadd__(self,other):

        assert type(self.type()) is type(other.type()), 'Type of addends must match'

        self.__dim = self.dim | other.dim
        self.__var = self.var | other.var
        self.__init = self.init | other.init

        return self