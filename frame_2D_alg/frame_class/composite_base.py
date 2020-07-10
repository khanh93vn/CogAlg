"""
A base class for all composite structure
"""

class MetaComposite(type):
    def __new__(mcs, name, bases, attrs):  # called right before a new class is created
        cls = super().__new__(mcs, name, bases, attrs)
        cls.instances = []
        return cls

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.id = len(cls.instances)
        cls.instances.append(instance)
        return instance


class CompositeStructure(metaclass=MetaComposite):
    def unpack(self):
        return self.__dict__.values()


if __name__ == "__main__":  # for debugging
    pass
