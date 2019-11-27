class AbortRunError(Exception):
    pass


class InvalidTypeReferenceError(Exception):
    pass


class TypeNameCollisionError(Exception):
    pass


class MultipleInstanceError(Exception):
    def __init__(self):
        super(MultipleInstanceError, self).__init__('It is not permitted to create multiple instances')


class InvalidConfigurationError(Exception):
    def __init__(self):
        super(InvalidConfigurationError, self).__init__('This configuration is not supported!')


class DependencyInjectionError(Exception):
    def __init__(self, message: str = None):
        super(DependencyInjectionError, self).__init__(
            'Object is not callable due to unresolved variable dependency. '
            'Verify your configurations and be aware that class ordering in configs matters, '
            'due to instantiation order! {}'.format(message))


class TypeInstantiationError(Exception):
    pass
