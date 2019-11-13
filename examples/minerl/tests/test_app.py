from lighter.decorator import register, context


class App:
    @context
    @register(type='examples.minerl.envs.defaults.Environment', property='env')
    def __init__(self):
        pass


if __name__ == '__main__':
    app = App()
    app.env.connect()
    env = app.env.make(app.config.env.env_name)
    if env is not None:
        pass
