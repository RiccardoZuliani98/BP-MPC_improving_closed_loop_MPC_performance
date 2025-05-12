from __future__ import annotations
from copy import copy
from typing import Optional,Union

class Options:

    def __init__(self,allowed_options:dict,default_options:Optional[dict]=None) -> None:
        """
        Initializes the options handler with allowed and default options.
        Args:
            allowed_options (dict): A dictionary specifying the allowed options 
                and their corresponding valid values or constraints.
            default_options (Optional[dict]): A dictionary specifying the default 
                options to initialize with. Defaults to an empty dictionary if not provided.
        Raises:
            AssertionError: If `allowed_options` is not a dictionary.
        """

        if default_options is None:
            default_options = {}

        assert isinstance(allowed_options,dict), 'Allowed options must be a dictionary'

        self._allowed_options = allowed_options
        self.update(default_options)

    def __add__(self,other:Options) -> Options:

        assert isinstance(other,Options), 'You can only sum two options objects'

        # create a copy of the current Object
        out = copy(self)

        # add other object
        out._allowed_options = out._allowed_options | other._allowed_options
        out.update(other)

        return out
    
    def __getitem__(self, key:str):
        # Allow dictionary-like access: obj['key']
        return self.__dict__[key]
    
    def keys(self):
        # Support dict-like keys() method
        return [k for k in self.__dict__.keys() if not k.startswith('_')]
    
    def values(self):
        # Support dict-like values() method
        return [self.__dict__[k] for k in self.keys()]
    
    def items(self):
        # Support dict-like items() method
        return [(k, self.__dict__[k]) for k in self.keys()]

    def __setitem__(self, key:str, value):
        # Allow dictionary-like assignment: obj['key'] = value
        self.__dict__[key] = value

    def update(self,dict_in:Union[dict,Options]) -> None:
        """
        Updates the options with the provided dictionary.
        This method allows updating the current options by passing a dictionary
        or another `Options` instance. It validates the keys and values against
        the allowed options and their constraints.
        Args:
            dict_in (dict): A dictionary or an `Options` instance containing the
                            options to update.
        Raises:
            AssertionError: If `dict_in` is not a dictionary or an `Options` instance.
            AssertionError: If a key in `dict_in` is not in the allowed options.
            AssertionError: If a value in `dict_in` does not match the constraints
                            of the corresponding allowed option. Constraints can
                            either be a finite list of allowed values or a type
                            that the value must match.
        """

        assert isinstance(dict_in,(dict,Options)), 'A dictionary or another Options class must be passed to update options'

        for key,val in dict_in.items():

            assert key in self._allowed_options.keys(), 'The option you passed is not allowed'

            # check if allowed_options[key] is a finite list
            if isinstance(self._allowed_options[key],list):
                assert val in self._allowed_options[key], 'You passed an illegal value for option' + key
            # otherwise, it must be a type
            else:
                assert isinstance(val,self._allowed_options[key]), 'You passed an illegal value for option' + key

            self.__setitem__(key,val)