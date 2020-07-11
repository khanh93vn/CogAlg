"""
A base class for all composite structure.
"""

from numbers import Number


NoneType = type(None)

_methods_template = '''
def pack(self{param_args}):
    {assignments}
    
def unpack(self):
    return ({param_vals})

def accumulate(self, **kwargs):
    {accumulations}

def __contains__(self, item):
    return (item in {params})

def __delattr__(self, item):
    raise AttributeError("cannot delete attribute from "
                         "'{typename}' object")

def __repr__(self):
    return "{typename}({repr_fmt})" % ({param_vals})

'''

class MetaComposite(type):

    def __new__(mcs, typename, bases, attrs):  # called right before a new class is created
        params = tuple(attr for attr in attrs if not attr.startswith('_'))
        numeric_params = tuple(param for param in params
                               if issubclass(attrs[param], Number))

        methods_definitions = _methods_template.format(
            typename=typename,
            params=str(params),
            param_vals=', '.join(f'self.{param}'
                                 for param in params),
            param_args=', '.join(param for param in ('', *params)),
            assignments='; '.join(f'self.{param} = {param}'
                                  for param in params)
                        if params else 'pass',
            accumulations='; '.join(f"self.{param} += "
                                    f"kwargs.get('{param}', 0)"
                                    for param in numeric_params)
                          if params else 'pass',
            repr_fmt=', '.join(f'{param}=%r' for param in params),
        ); print(methods_definitions)
        namespace = dict(print=print)
        exec(methods_definitions, namespace)
        namespace.pop('__builtins__')
        attrs.update(namespace)
        for param in params:
            attrs['_'+param] = attrs.pop(param)
        attrs['params'] = params
        attrs['numeric_params'] = numeric_params

        attrs['__slots__'] = ('_id', *params)
        cls = super().__new__(mcs, typename, bases, attrs)
        cls._instances = []
        return cls

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)

        for k in cls.__slots__:
            if k != '_id':
                setattr(instance, k,
                        kwargs.get(k, getattr(cls, '_' + k)()))

        instance._id = len(cls._instances)
        cls._instances.append(instance)

        return instance


class CompositeStructure(metaclass=MetaComposite):
    def __init__(self, **kwargs):
        pass

if __name__ == "__main__":  # for debugging
    from sys import getsizeof as size
    pass

