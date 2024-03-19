class BaseOptions():
    def __init__(self):
        self.inputs = ''
        self.target = ''
        self.batchSize = 1
        self.input_nc = ''
        self.output_nc = 1
        self.ngf = 64
        self.ndf = 64
        self.which_model_netD = 'basic'
        self.which_model_netG = 'resvit'
        self.n_layers_D = 3
        self.gpu_ids = 0
        self.name = 'experiment_name'
        self.model = 'resvit_one'
        self.checkpoints_dir = './checkpoints'
        self.norm = 'instance'
        self.init_type = 'normal'
        self.vit_name = 'Res-ViT-B_16'
        self.pre_trained_path = 'Res-ViT-B_16'
        self.pre_trained_transformer = True
        self.pre_trained_resnet = False
        self.fineSize = 256
        self.no_dropout = True
        self.which_direction = 'AtoB'

