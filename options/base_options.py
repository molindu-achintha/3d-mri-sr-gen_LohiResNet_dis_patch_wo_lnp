import argparse
import os
from pathlib import Path
from util import util
import torch


def _default_dinov3_repo_path() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root.parent / 'dinov3'
    return str(candidate) if candidate.exists() else ''


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', required=True, help='path to images (trainA/trainB etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='crop size')
        self.parser.add_argument('--depthSize', type=int, default=256, help='depth for 3d images')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resunet_3d',
                                 help='generator name; legacy values are ignored and resunet_3d is used')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                                 help='execution device. auto uses CUDA when available and gpu_ids is not -1')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='dataset loading mode')
        self.parser.add_argument('--model', type=str, default='pix2pix3d',
                                 help='chooses which model to use.')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--lr_subdir', type=str, default='',
                                 help='optional LR subdirectory under dataroot (e.g. LR)')
        self.parser.add_argument('--hr_subdir', type=str, default='',
                                 help='optional HR subdirectory under dataroot (e.g. HR)')
        self.parser.add_argument('--allow_unmatched_lr', action='store_true',
                                 help='skip unmatched LR files instead of raising an error')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='single visdom pane columns')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='maximum # of samples')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling/cropping mode')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip images')
        self.parser.add_argument('--patch_size', type=int, default=64,
                                 help='TorchIO cube patch size for training; use 0 to train on full volumes')
        self.parser.add_argument('--patch_overlap', type=int, default=0,
                                 help='TorchIO grid patch overlap in voxels')
        self.parser.add_argument('--scale_factor', type=int, default=1,
                                 help='legacy SR scale; forced to 1 because resunet_3d outputs input size')
        self.parser.add_argument('--n_fmdrb', type=int, default=6,
                                 help='legacy no-op kept for old command compatibility')
        self.parser.add_argument('--skip_compress_ratio', type=float, default=0.5,
                                 help='legacy no-op kept for old command compatibility')
        self.parser.add_argument('--wgan_gp', action='store_true', help='use WGAN-GP loss and critic')
        self.parser.add_argument('--gp_lambda', type=float, default=10.0, help='gradient penalty weight')
        self.parser.add_argument('--n_critic', type=int, default=5, help='critic steps per generator step')
        self.parser.add_argument('--lambda_gan', type=float, default=1.0,
                                 help='weight for adversarial generator loss term')
        self.parser.add_argument('--research_profile', type=str, default='none',
                                 choices=['none', 'balanced_mri_sr_v1'],
                                 help='apply research-backed default hyperparameters')
        self.parser.add_argument('--use_perceptual_loss', action='store_true',
                                 help='enable perceptual feature loss in generator training')
        self.parser.add_argument('--perceptual_backbone', type=str, default='swinunetr',
                                 choices=['swinunetr', 'dinov3'],
                                 help='perceptual feature backbone')
        self.parser.add_argument('--lambda_perceptual', type=float, default=0.1,
                                 help='weight for perceptual loss term')
        self.parser.add_argument('--perceptual_distance', type=str, default='l2',
                                 choices=['l1', 'l2'],
                                 help='feature distance for perceptual loss')
        self.parser.add_argument('--perceptual_model_arch', type=str, default='dinov3_vitb16',
                                 help='backbone architecture id (local DINOv3 factory name or timm model name)')
        self.parser.add_argument('--perceptual_model_ckpt', type=str, default='',
                                 help='optional checkpoint path for perceptual backbone weights')
        self.parser.add_argument('--perceptual_dinov3_repo', type=str, default=_default_dinov3_repo_path(),
                                 help='optional local DINOv3 repo path used for dinov3_* architectures')
        self.parser.add_argument('--perceptual_pretrained', dest='perceptual_pretrained', action='store_true',
                                 help='try to load default pretrained weights when available')
        self.parser.add_argument('--no_perceptual_pretrained', dest='perceptual_pretrained', action='store_false',
                                 help='disable loading default pretrained weights')
        self.parser.add_argument('--perceptual_dino_input_size', type=int, default=224,
                                 help='input resolution for slice-wise dinov3 perceptual features')
        self.parser.add_argument('--perceptual_swin_feature_layers', type=str,
                                 default='encoder1,encoder2,encoder3,encoder4',
                                 help='comma-separated SwinUNETR module names used for feature taps')
        self.parser.set_defaults(perceptual_pretrained=True)

        self.initialized = True

    def _apply_research_profile_defaults(self):
        preview_opt, _ = self.parser.parse_known_args()
        profile = getattr(preview_opt, 'research_profile', 'none')
        if profile == 'none':
            return

        if profile == 'balanced_mri_sr_v1':
            self.parser.set_defaults(
                which_model_netG='resunet_3d',
                ngf=48,
                which_model_netD='n_layers',
                ndf=48,
                n_layers_D=3,
                wgan_gp=True,
                gp_lambda=10.0,
                n_critic=5,
                lambda_A=100.0,
                lambda_gan=0.001,
                use_perceptual_loss=True,
                perceptual_backbone='dinov3',
                lambda_perceptual=0.01,
                perceptual_distance='l2',
                scale_factor=1,
            )

    def parse(self):
        if not self.initialized:
            self.initialize()
        self._apply_research_profile_defaults()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain

        if self.opt.which_model_netG != 'resunet_3d':
            print("Generator '%s' is deprecated; using 'resunet_3d'." % self.opt.which_model_netG)
            self.opt.which_model_netG = 'resunet_3d'
        if self.opt.scale_factor != 1:
            print("resunet_3d outputs the input spatial size; forcing scale_factor=1.")
            self.opt.scale_factor = 1
        if self.opt.patch_size < 0:
            raise ValueError('--patch_size must be >= 0.')
        if self.opt.patch_overlap < 0:
            raise ValueError('--patch_overlap must be >= 0.')
        if self.opt.patch_size > 0 and self.opt.patch_overlap >= self.opt.patch_size:
            raise ValueError('--patch_overlap must be smaller than --patch_size.')

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            gid = int(str_id)
            if gid >= 0:
                self.opt.gpu_ids.append(gid)

        if self.opt.device == 'cpu':
            self.opt.gpu_ids = []
            self.opt.device = 'cpu'
        elif self.opt.device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested via --device cuda, but CUDA is not available.')
            if len(self.opt.gpu_ids) == 0:
                self.opt.gpu_ids = [0]
            torch.cuda.set_device(self.opt.gpu_ids[0])
            self.opt.device = 'cuda'
        else:
            if torch.cuda.is_available() and len(self.opt.gpu_ids) > 0:
                torch.cuda.set_device(self.opt.gpu_ids[0])
                self.opt.device = 'cuda'
            else:
                self.opt.gpu_ids = []
                self.opt.device = 'cpu'

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
