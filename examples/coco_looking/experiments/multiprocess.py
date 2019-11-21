from lighter.decorator import references
from lighter.experiment import DefaultExperiment


class SearchExperiment(DefaultExperiment):
    @references
    def __init__(self):
        super(SearchExperiment, self).__init__(epochs=50)
