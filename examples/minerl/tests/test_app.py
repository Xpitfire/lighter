from lighter.decorator import register, context


class App:
    @context
    @register(type='examples.minerl.envs.defaults.Environment', property='env')
    def __init__(self):
        pass


if __name__ == '__main__':
    app = App()
    if app.env is not None and hasattr(app.env, 'step'):
        pass
    else:
        raise NotImplementedError()
