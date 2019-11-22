from searches.defaults import ParameterSearchRegistration
from lighter.parser import ConfigParser
from lighter.context import Context


if __name__ == '__main__':
    context = Context.create()
    se = ParameterSearchRegistration()
    cp = ConfigParser(experiment=se)
    cp.parse()
