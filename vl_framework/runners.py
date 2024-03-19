import os.path

from options.test_options import TestOptions
from options.train_options import TrainOptions
from vl_framework.test import test
from vl_framework.train import train


def train_resvit(experiment_name, checkpoints_dir, dataset, input_nc, gpu=0):
    ### Pretrain the ART blocks
    opt = TrainOptions()
    opt.name = os.path.join(experiment_name, 'art_pretain')
    opt.gpu_ids = gpu
    opt.model = 'resvit_one'
    opt.which_model_netG = 'res_cnn'
    opt.lambda_A = 100
    opt.lambda_adv = 1
    opt.norm = 'batch'
    opt.pool_size = 0
    opt.niter = 50
    opt.niter_decay = 50
    opt.checkpoints_dir = checkpoints_dir
    opt.lr = 0.0002
    opt.input_nc = input_nc
    opt.output_nc = 1
    train(opt, dataset)

    ### refine with transformers
    opt.name = os.path.join(experiment_name, 'resvit')
    opt.which_model_netG = 'resvit'
    opt.lr = 0.001
    opt.niter = 25
    opt.niter_decay = 25
    opt.pre_trained_transformer = 1
    opt.pre_trained_resnet = 1
    opt.pre_trained_path = os.path.join(checkpoints_dir, experiment_name, 'art_pretain', 'latest_net_G.pth')
    train(opt, dataset)

def test_resvit(experiment_name, checkpoints_dir, dataset, input_nc, deployment=False, gpu=0):
    opt = TestOptions()
    opt.name = os.path.join(experiment_name, 'resvit')
    opt.gpu_ids = gpu
    opt.model = 'resvit_one'
    opt.which_model_netG = 'resvit'
    opt.norm = 'batch'
    opt.pool_size = 0
    opt.niter = 50
    opt.niter_decay = 50
    opt.checkpoints_dir = checkpoints_dir
    opt.input_nc = input_nc
    opt.output_nc = 1
    test(opt, dataset, deployment)
    
def train_unet(experiment_name, checkpoints_dir, dataset, input_nc, output_nc, gpu=0):
    opt = TrainOptions()
    opt.name = os.path.join(experiment_name, 'unet')
    opt.gpu_ids = gpu
    opt.model = 'resvit_one'
    opt.which_model_netG = 'unet_256'
    opt.lambda_adv = 0
    opt.lambda_A = 100
    opt.norm = 'batch'
    opt.pool_size = 0
    opt.niter = 50
    opt.niter_decay = 50
    opt.checkpoints_dir = checkpoints_dir
    opt.lr = 0.0002
    opt.input_nc = input_nc
    opt.output_nc = output_nc
    train(opt, dataset)
    
def test_unet(experiment_name, checkpoints_dir, dataset, input_nc, output_nc, gpu=0):
    opt = TestOptions()
    opt.name = os.path.join(experiment_name, 'unet')
    opt.gpu_ids = gpu
    opt.model = 'resvit_one'
    opt.which_model_netG = 'unet_256'
    opt.norm = 'batch'
    opt.pool_size = 0
    opt.niter = 0
    opt.niter_decay = 0
    opt.checkpoints_dir = checkpoints_dir
    opt.input_nc = input_nc
    opt.output_nc = output_nc
    test(opt, dataset)
    
    
def train_cgan(experiment_name, checkpoints_dir, dataset, input_nc, gpu=0):
    opt = TrainOptions()
    opt.name = os.path.join(experiment_name, 'cgan')
    opt.gpu_ids = gpu
    opt.model = 'resvit_one'
    opt.which_model_netG = 'unet_256'
    opt.lambda_adv = 1
    opt.lambda_A = 100
    opt.norm = 'batch'
    opt.pool_size = 0
    opt.niter = 50
    opt.niter_decay = 50
    opt.checkpoints_dir = checkpoints_dir
    opt.lr = 0.0002
    opt.input_nc = input_nc
    opt.output_nc = 1
    train(opt, dataset)
    
def test_cgan(experiment_name, checkpoints_dir, dataset, input_nc, gpu=0):
    opt = TestOptions()
    opt.name = os.path.join(experiment_name, 'cgan')
    opt.gpu_ids = gpu
    opt.model = 'resvit_one'
    opt.which_model_netG = 'unet_256'
    opt.norm = 'batch'
    opt.pool_size = 0
    opt.niter = 0
    opt.niter_decay = 0
    opt.checkpoints_dir = checkpoints_dir
    opt.input_nc = input_nc
    opt.output_nc = 1
    test(opt, dataset)