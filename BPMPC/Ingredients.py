from casadi import *

class Ingredients:
    """
    QP ingredients, contains the following keys

        - 'Q': Hessian of QP (n_y,n_y)
        - 'Qinv': inverse of Hessian of QP (n_y,n_y)
        - 'q': gradient of cost of QP (n_y,1)
        - 'G': linear inequality constraint matrix (n_in,n_y)
        - 'g': linear inequality constraint vector (n_in,1)
        - 'H': Hessian of dual problem (n_z,n_z)
        - 'h': gradient of cost of dual problem (n_z,1)
        - 'A': stacked inequality constraint matrix for casadi's conic interface,
               specifically, A = vertcat(G,F) (n_in+n_eq,n_y)
        - 'lba': lower bound of inequality constraints, lba = (-inf,f) (n_in+n_eq,1)
        - 'uba': upper bound of inequality constraints, uba = (g,f) (n_in+n_eq,1)

    remember that the primal problem is a QP with the following structure:

        min 1/2 y'Qy + q'y
        s.t. Gy <= g
             Fy = y

    and the dual problem is a QP with the following structure:

        min 1/2 z'Hz + h'z
        s.t. z=(lam,mu)
             lam >= 0
    """

    __Q = None
    __q = None
    __F = None
    __f = None
    __G = None
    __g = None

    @property
    def G(self):
        return self.__G
    
    @property
    def g(self):
        return self.__g
    
    @property
    def F(self):
        return self.__F
    
    @property
    def f(self):
        return self.__f
    
    @property
    def Q(self):
        return self.__Q
    
    @property
    def q(self):
        return self.__q
    
    def update(self,Q=None,q=None,G=None,g=None,F=None,f=None):
        #TODO
        pass

    