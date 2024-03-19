from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()
        self.results_dir = './results/'
        self.isTrain = False
        self.which_epoch = 'latest'
