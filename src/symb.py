from __future__ import annotations
import casadi as ca
from copy import copy
from typeguard import typechecked
from typing import Optional,Union

class SymbolicVar:

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
        return self._init

    @staticmethod
    def _check_and_convert(value:Union[list,ca.DM],expected_shape:Union[int,list[int]]) -> Union[list,ca.DM]:
        """
        Recursively checks and converts the input value to a CasADi DM (Dense Matrix) object 
        while ensuring it matches the expected shape.
        Args:
            value (list or numeric): The input value to be checked and converted. It can be a 
                numeric value, a list of numeric values, or a nested list structure.
            expected_shape (tuple or int): The expected shape of the resulting CasADi DM object. 
                If the input is a column vector, this can be an integer representing the number 
                of rows.
        Returns:
            ca.DM: The converted CasADi DM object with the expected shape.
        Raises:
            AssertionError: If the shape of the converted value does not match the expected shape.
        """

        # check if value is a list
        if isinstance(value, list):

            # convert value: if length is greater than 1, run this function recursively on each element
            # of the list, if length is 1, extract its unique element and run this function on it.
            value_converted = [SymbolicVar._check_and_convert(elem, expected_shape) for elem in value] if len(
                value) > 1 else SymbolicVar._check_and_convert(value[0], expected_shape)

        else:

            # convert to DM
            value_converted = ca.DM(value)

            # check dimension
            if value_converted.shape[1] == 1:
                assert expected_shape == value_converted.shape[0], 'Dimension of initialization does not match dimension of symbolic variable'
            else:
                assert expected_shape == value_converted.shape, 'Dimension of initialization does not match dimension of symbolic variable'

        return value_converted

    @typechecked
    def set_init(self,data:dict):
        """
        Initialize symbolic variables with given values.
        This method initializes the symbolic variables of the object using the 
        provided dictionary. It ensures that the variables being initialized 
        exist and that the dimensions of the provided values match the dimensions 
        of the symbolic variables.
        Args:
            data (dict): A dictionary where keys are variable names (strings) 
                            and values are the initialization values. The values 
                            can be:
                            - A list of elements (converted to CasADi DM objects).
                            - A single-element list (converted to a CasADi DM object).
                            - Any entity that can be converted to a CasADi DM object.
        Raises:
            AssertionError: If `data` is not a dictionary.
            AssertionError: If a variable name in `data` does not exist in the 
                            symbolic variables.
            AssertionError: If the dimensions of the initialization values do not 
                            match the dimensions of the symbolic variables.
            Exception: If a value cannot be converted to a CasADi DM object.
        Notes:
            - If a list with more than one element is provided, each element is 
                converted to a CasADi DM object, and their dimensions are checked.
            - If a single-element list is provided, it is converted directly to 
                a CasADi DM object.
            - The `_init` attribute contains all the initialized values as
                CasADi DM objects.
        """

        for name,value in data.items():
        
            assert name in self._var, 'Cannot initialize variable that does not exist'

            # run function that checks dimension and stores
            converted_value = self._check_and_convert(value,self._dim[name])

            # add to initialization
            self._init[name] = converted_value

    def get_var(self,name):
        return self.var[name],self.dim[name],self.init[name]
    
    @typechecked
    def add_var(self,name:str,var:ca.SX,init=None):
        """
        Adds a symbolic variable to the internal variable dictionary with its associated dimension
        and optionally initializes it.
        Args:
            name (str): The name of the variable to be added.
            var (ca.SX): The CasADi SX symbolic variable to be added.
            init (optional): The initial value for the variable. If provided, the variable will be
                             initialized using the `set_init` method.
        Behavior:
            - The variable is stored in the `_var` dictionary using the provided name as the key.
            - The dimension of the variable is determined based on its shape:
                - If the variable is a vector (second dimension is 1), the dimension is set to the
                  length of the vector.
                - Otherwise, the dimension is set to the full shape of the variable.
            - If the user provides an initial value, the `set_init` method is called to initialize the variable.
        """

        # create symbolic variable using provided name and CasADi SX
        self._var[name] = var

        # if the variable is a vector, set the dimension to the length of the vector
        # otherwise, set the dimension to the shape of the variable
        self._dim[name] = var.shape[0] if var.shape[1] == 1 else var.shape

        # run initialization routine if init is not None
        if init is not None:
            self.set_init({name:init})

    @typechecked
    def add_dim(self,name:str,val:Union[int,list]):
        """
        Adds a dimension to the internal dimension dictionary.
        Parameters:
            name (str): The name of the dimension to add.
            val (Union[int, list]): The value of the dimension. It can be either:
                - An integer representing a single non-negative dimension.
                - A list of integers (up to two elements) representing multidimensional values,
                  where each element must be a non-negative integer.
        Raises:
            AssertionError: If `val` is a list and contains elements that are not non-negative integers.
            AssertionError: If `val` is a list with more than two elements.
            AssertionError: If `val` is an integer and is negative.
        """

        # check that the dimension is strictly positive
        if isinstance(val,list):
            assert all([isinstance(v,int) and v >= 0 for v in val]), 'Value of dimension must be a non-negative integer'
            assert len(list) <= 2, 'Init values must not be more than two-dimensional'
        else:
            assert val >= 0, 'Value of dimension must be a non-negative integer'

        self._dim[name] = val

    @typechecked
    def __add__(self,other:SymbolicVar):

        # create copy of class
        self_copy = copy(self)

        self_copy._dim = self.dim | other.dim
        self_copy._var = self.var | other.var
        self_copy._init = self.init | other.init

        return self_copy

    @typechecked
    def __iadd__(self,other:SymbolicVar):

        self._dim = self.dim | other.dim
        self._var = self.var | other.var
        self._init = self.init | other.init

        return self
    
    def copy(self,vars2keep:Optional[list]=None):
        """
        Create a copy of the current object, optionally retaining only specified variables.
        Parameters:
            vars2keep (list, optional): A collection of variable names to retain in the copied object.
                If None, all variables from the original object are retained.
        Returns:
            self_copy: A new instance of the same class as the original object, containing either all
                variables or only the specified subset of variables.
        """

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