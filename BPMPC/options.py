class Options:

    def __init__(self,allowed_options,default_options={}):

        assert isinstance(allowed_options,dict), 'Allowed options must be a dictionary'

        self._allowed_options = allowed_options
        self.update(default_options)
    
    def __getitem__(self, key):
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

    def __setitem__(self, key, value):
        # Allow dictionary-like assignment: obj['key'] = value
        self.__dict__[key] = value

    def update(self,dict_in):

        assert isinstance(dict_in,dict), 'A dictionary must be passed to update options'

        for key,val in dict_in.items():

            assert key in self._allowed_options.keys(), 'The option you passed is not allowed'
            assert val in self._allowed_options[key], 'You passed an illegal value for option' + key

            self.__setitem__(key,val)