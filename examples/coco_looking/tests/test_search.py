from searches.defaults import ParameterSearchRegistration
from lighter.parser import ConfigParser
from lighter.context import Context


if __name__ == '__main__':
    # configs are used for the initial configurations of the fixed parameters
    context = Context.create()
    # instantiate types since the SearchExperiment has not strategy decorator defined
    context.instantiate_types(context.registry.types)
    se = ParameterSearchRegistration()
    cp = ConfigParser(experiment=se)
    cp.parse()
