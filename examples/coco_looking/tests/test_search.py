from searches.defaults import ParameterSearchRegistration
from lighter.parser import ConfigParser
from lighter.context import Context


if __name__ == '__main__':
    context = Context.create(auto_instantiate_types=False)
    psr = ParameterSearchRegistration()
    cp = ConfigParser(experiment=psr)
    cp.parse()
