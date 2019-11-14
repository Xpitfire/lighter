from lighter.decorator import register, context


class App:
    @context
    @register(type='envs.defaults.Environment', property='env')
    def __init__(self):
        pass


if __name__ == '__main__':
    app = App()
    if app.env is None or not hasattr(app.env, 'step'):
        raise ReferenceError()
