from lighter.decorator import context, config
from examples.minerl.envs.env_client import RemoteGym


class Environment(RemoteGym):
    @context
    @config(path='envs/config.json', property='env')
    def __init__(self):
        super(Environment, self).__init__(host=self.config.env.host, port=self.config.env.port)
        self.env.connect()
        self.env = self.env.make(self.config.env.env_name)

    def __get__(self, instance, owner):
        return self.env
