import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks3d as networks
from .perceptual_loss import PerceptualLoss3D, build_perceptual_extractor


class LohiResNet_dis_patch_wo_lnp(BaseModel):
    def name(self):
        return 'LohiResNet_dis_patch_wo_lnp'

    def initialize(self, opt, perceptual_model=None):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self._initialize_input_tensors(opt)
        self._configure_training_flags(opt)
        self._build_networks(opt)
        self._load_networks_if_needed(opt)
        if self.isTrain:
            self._build_training_components(opt, perceptual_model)
        self._print_networks()

    def _initialize_input_tensors(self, opt):
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.depthSize, opt.fineSize, opt.fineSize)
        requested_scale = getattr(opt, 'scale_factor', 1)
        if requested_scale != 1:
            print("resunet_3d outputs the input spatial size; forcing scale_factor=1.")
        self.scale_factor = 1
        hr_depth = opt.depthSize * self.scale_factor
        hr_size = opt.fineSize * self.scale_factor
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   hr_depth, hr_size, hr_size)

    def _configure_training_flags(self, opt):
        self.gp_lambda = getattr(opt, 'gp_lambda', 10.0)
        self.n_critic = getattr(opt, 'n_critic', 5)
        self.use_wgan_gp = getattr(opt, 'wgan_gp', False)
        self.use_perceptual_loss = getattr(opt, 'use_perceptual_loss', False)
        self.lambda_perceptual = getattr(opt, 'lambda_perceptual', 0.1)
        self.lambda_gan = getattr(opt, 'lambda_gan', 1.0)
        self.perceptual_loss_fn = None

    def _build_networks(self, opt):
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm,
                                      not opt.no_dropout, self.gpu_ids,
                                      scale_factor=self.scale_factor,
                                      input_size=opt.depthSize)
        if self.isTrain:
            use_sigmoid = False  # not used with WGAN-GP
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid,
                                          self.gpu_ids, wgan_gp=self.use_wgan_gp)

    def _load_networks_if_needed(self, opt):
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

    def _build_training_components(self, opt, perceptual_model=None):
        self.fake_AB_pool = ImagePool(opt.pool_size)
        self.old_lr = opt.lr
        self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
        self.criterionL1 = torch.nn.L1Loss()
        self.loss_G_Perc = self.input_A.new_tensor(0.0)
        self.perceptual_loss_fn = self._build_perceptual_loss(opt, perceptual_model)
        self._build_optimizers(opt)

    def _build_perceptual_loss(self, opt, perceptual_model=None):
        if not self.use_perceptual_loss:
            return None
        device = next(self.netG.parameters()).device
        extractor = build_perceptual_extractor(
            opt=opt,
            perceptual_model=perceptual_model,
            device=device,
        )
        return PerceptualLoss3D(
            extractor=extractor,
            distance=getattr(opt, 'perceptual_distance', 'l2'),
        ).to(device)

    def _build_optimizers(self, opt):
        betas = (0.0, 0.9) if self.use_wgan_gp else (opt.beta1, 0.999)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=opt.lr, betas=betas)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=betas)

    def _print_networks(self):
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

    def _discriminator_pairs(self, detach_fake: bool):
        real_A_for_D = self.real_A
        real_AB = torch.cat((real_A_for_D, self.real_B), 1)
        fake_B = self.fake_B.detach() if detach_fake else self.fake_B
        fake_AB = torch.cat((real_A_for_D, fake_B), 1)
        return real_AB, fake_AB

    def _backward_wgan_discriminator(self, real_AB, fake_AB, pred_real, pred_fake):
        gp = self._gradient_penalty(real_AB.data, fake_AB.data)
        self.loss_D = pred_fake.mean() - pred_real.mean() + gp
        self.loss_D_real = -pred_real.mean()
        self.loss_D_fake = pred_fake.mean()
        self.loss_D.backward()

    def _backward_gan_discriminator(self, pred_real, pred_fake):
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_D(self):
        real_AB, fake_AB = self._discriminator_pairs(detach_fake=True)
        pred_real = self.netD(real_AB)
        pred_fake = self.netD(fake_AB)

        if self.use_wgan_gp:
            self._backward_wgan_discriminator(real_AB, fake_AB, pred_real, pred_fake)
        else:
            self._backward_gan_discriminator(pred_real, pred_fake)

    def _generator_gan_loss(self, pred_fake):
        if self.use_wgan_gp:
            return -pred_fake.mean()
        return self.criterionGAN(pred_fake, True)

    def _generator_reconstruction_loss(self):
        return self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

    def _generator_perceptual_loss(self):
        if self.use_perceptual_loss and self.perceptual_loss_fn is not None:
            raw_perceptual = self.perceptual_loss_fn(self.fake_B, self.real_B)
            return raw_perceptual * self.lambda_perceptual
        return self.fake_B.new_tensor(0.0)

    def backward_G(self):
        _real_AB, fake_AB = self._discriminator_pairs(detach_fake=False)
        pred_fake = self.netD(fake_AB)

        self.loss_G_GAN = self._generator_gan_loss(pred_fake)
        self.loss_G_L1 = self._generator_reconstruction_loss()
        self.loss_G_Perc = self._generator_perceptual_loss()
        self.loss_G = (self.lambda_gan * self.loss_G_GAN) + self.loss_G_L1 + self.loss_G_Perc
        self.loss_G.backward()

    def _run_discriminator_step(self):
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def _run_generator_step(self):
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()

        for _ in range(self.n_critic if self.use_wgan_gp else 1):
            self._run_discriminator_step()

        self._run_generator_step()

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
