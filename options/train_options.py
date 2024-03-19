from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        self.epoch_count = 1
        self.which_epoch = 'latest'
        self.niter = 100
        self.niter_decay = 100
        self.beta1 = 0.5
        self.lr = 0.0002
        self.trans_lr_coef = 1
        self.no_lsgan = True
        self.lambda_A = 10.0
        self.lambda_B = 10.0
        self.lambda_f = 0.9
        self.lambda_identity = 0
        self.pool_size = 50
        self.lambda_f = 0.9
        self.lr_policy = 'lambda'
        self.lr_decay_iters = 50
        self.lambda_vgg = 1
        self.vgg_layer = 2
        self.lambda_adv = 1.0
        self.continue_train = False
        self.isTrain = True
