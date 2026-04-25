import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks3d as networks
from .perceptual_loss import PerceptualLoss3D, build_perceptual_extractor


class Pix2Pix3dModel(BaseModel):
    def name(self):
        return 'Pix2Pix3dModel'

    def initialize(self, opt, perceptual_model=None):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.depthSize, opt.fineSize, opt.fineSize)
        # For SR models, ground-truth B is at the upscaled resolution
        requested_scale = getattr(opt, 'scale_factor', 1)
        if requested_scale != 1:
            print("resunet_3d outputs the input spatial size; forcing scale_factor=1.")
        self.scale_factor = 1
        hr_depth = opt.depthSize * self.scale_factor
        hr_size = opt.fineSize * self.scale_factor
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   hr_depth, hr_size, hr_size)

        # load/define networks
        self.gp_lambda = getattr(opt, 'gp_lambda', 10.0)
        self.n_critic = getattr(opt, 'n_critic', 5)
        self.use_wgan_gp = getattr(opt, 'wgan_gp', False)
        self.use_perceptual_loss = getattr(opt, 'use_perceptual_loss', False)
        self.lambda_perceptual = getattr(opt, 'lambda_perceptual', 0.1)
        self.lambda_gan = getattr(opt, 'lambda_gan', 1.0)
        self.perceptual_loss_fn = None

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm,
                                      not opt.no_dropout, self.gpu_ids,
                                      scale_factor=self.scale_factor)
        if self.isTrain:
            use_sigmoid = False  # not used with WGAN-GP
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid,
                                          self.gpu_ids, wgan_gp=self.use_wgan_gp)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.loss_G_Perc = self.input_A.new_tensor(0.0)

            if self.use_perceptual_loss:
                device = next(self.netG.parameters()).device
                extractor = build_perceptual_extractor(
                    opt=opt,
                    perceptual_model=perceptual_model,
                    device=device,
                )
                self.perceptual_loss_fn = PerceptualLoss3D(
                    extractor=extractor,
                    distance=getattr(opt, 'perceptual_distance', 'l2'),
                ).to(device)

            # initialize optimizers
            betas = (0.0, 0.9) if self.use_wgan_gp else (opt.beta1, 0.999)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=betas)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=betas)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        #self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return "blksdf"
        #return self.image_paths

    def _gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, 1).to(real_data.device)
        alpha = alpha.expand_as(real_data)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self.netD(interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(disc_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda
        return gradient_penalty

    def backward_D(self):
        real_A_for_D = self.real_A
        real_AB = torch.cat((real_A_for_D, self.real_B), 1)
        fake_AB = torch.cat((real_A_for_D, self.fake_B.detach()), 1)

        pred_real = self.netD(real_AB)
        pred_fake = self.netD(fake_AB)

        if self.use_wgan_gp:
            gp = self._gradient_penalty(real_AB.data, fake_AB.data)
            self.loss_D = pred_fake.mean() - pred_real.mean() + gp
            self.loss_D_real = -pred_real.mean()
            self.loss_D_fake = pred_fake.mean()
            self.loss_D.backward()
        else:
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()

    def backward_G(self):
        real_A_for_D = self.real_A
        fake_AB = torch.cat((real_A_for_D, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        if self.use_wgan_gp:
            self.loss_G_GAN = -pred_fake.mean()
        else:
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        if self.use_perceptual_loss and self.perceptual_loss_fn is not None:
            raw_perceptual = self.perceptual_loss_fn(self.fake_B, self.real_B)
            self.loss_G_Perc = raw_perceptual * self.lambda_perceptual
        else:
            self.loss_G_Perc = self.fake_B.new_tensor(0.0)

        self.loss_G = (self.lambda_gan * self.loss_G_GAN) + self.loss_G_L1 + self.loss_G_Perc
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # Critic updates
        for _ in range(self.n_critic if self.use_wgan_gp else 1):
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # Generator update
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('G_Perc', self.loss_G_Perc.item()),
                            ('D_Real', self.loss_D_real.item()),
                            ('D_Fake', self.loss_D_fake.item())
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im3d(self.real_A.data)
        fake_B = util.tensor2im3d(self.fake_B.data)
        real_B = util.tensor2im3d(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
