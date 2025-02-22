from casadi import *

class upperLevel:

    def __init__(self,MSX):

        # check type of symbolic variables
        if MSX == 'SX':
            self.__MSX = SX
        elif MSX == 'MX':
            self.__MSX = MX
        else:
            raise Exception('MSX must be either SX or MX.')

        # initialize variables
        self.__p = None
        self.__pf = None
        self.__Jp = None
        self.__k = None
        self.__x_cl = None
        self.__u_cl = None
        self.__y_cl = None
        self.__e_cl = None
        self.__cost = None
        self.__J_cost = None
        self.__init = {'p':None, 'pf':None}
        self.__idx = {}
        self.__alg = None
        pass

    @property
    def alg(self):
        return self.__alg
    
    def __setAlg(self,value):
        self.__alg = value

    @property
    def p(self):
        return self.__p
    
    def __set_p(self, value):
        if type(value) is not self.__MSX:
            raise Exception('p is of the wrong symbolic type.')
        self.__p = value

    @property
    def pf(self):
        return self.__pf
    
    def __set_pf(self, value):
        if type(value) is not self.__MSX:
            raise Exception('pf is of the wrong symbolic type.')
        self.__pf = value

    @property
    def Jp(self):
        return self.__Jp
    
    def __set_Jp(self, value):
        if type(value) is not self.__MSX:
            raise Exception('Jp is of the wrong symbolic type.')
        self.__Jp = value

    @property
    def k(self):
        return self.__k
    
    def __set_k(self, value):
        if type(value) is not self.__MSX:
            raise Exception('k is of the wrong symbolic type.')
        self.__k = value

    @property
    def x_cl(self):
        return self.__x_cl
    
    def __set_x_cl(self, value):
        if type(value) is not self.__MSX:
            raise Exception('x_cl is of the wrong symbolic type.')
        self.__x_cl = value

    @property
    def u_cl(self):
        return self.__u_cl

    def __set_u_cl(self, value):
        if type(value) is not self.__MSX:
            raise Exception('u_cl is of the wrong symbolic type.')
        self.__u_cl = value

    @property
    def y_cl(self):
        return self.__y_cl

    def __set_y_cl(self, value):
        if type(value) is not self.__MSX:
            raise Exception('y_cl is of the wrong symbolic type.')
        self.__y_cl = value

    @property
    def e_cl(self):
        return self.__e_cl

    def __set_e_cl(self, value):
        if type(value) is not self.__MSX:
            raise Exception('e_cl is of the wrong symbolic type.')
        self.__e_cl = value

    @property
    def cost(self):
        return self.__cost
    
    def __set_cost(self, value):
        self.__cost = value

    @property
    def J_cost(self):
        return self.__J_cost
    
    def __set_J_cost(self, value):
        self.__J_cost = value

    @property
    def init(self):
        return {k: v for k, v in self.__init.items()}
    
    def __setInit(self,value):
        self.__init = self.__init | self.__checkInit(value)
    
    def __checkInit(self, value):

        # preallocate output
        out = {}

        if 'p' in value:

            if 'p' not in self.param:
                raise Exception('Define parameter p before setting its initial value.')

            # turn into DM
            p_init = DM(value['p'])

            if p_init.shape == self.p.shape:
                out = out | {'p':p_init}
            else:
                raise Exception('p must have the same shape as the initial parameter.')
            
        if 'pf' in value:

            if 'pf' not in self.param:
                raise Exception('Define parameter pf before setting its initial value.')

            # turn into DM
            pf_init = DM(value['pf'])

            if pf_init.shape == self.pf.shape:
                out = out | {'pf':pf_init}
            else:
                raise Exception('pf must have the same shape as the final parameter.')
            
        return out

    @property   
    def idx(self):
        return self.__idx

    def __updateIdx(self, idx):
        self.__idx = self.__idx | idx

    @property
    def param(self):
        return {k: v for k, v in {
            'p': self.__p,
            'pf':self.__pf,
            'Jp':self.__Jp,
            'k':self.__k,
            'x_cl': self.__x_cl,
            'u_cl': self.__u_cl,
            'y_cl': self.__y_cl,
            'e_cl': self.__e_cl,
        }.items() if v is not None}
    
    # overwrite the __dir__ method
    def __dir__(self):
        return [attr for attr in super().__dir__() if not attr.startswith('_UpperLevel__')]