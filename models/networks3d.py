import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight') and m.weight is not None:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm3d') != -1 and hasattr(m, 'weight') and m.weight is not None:
        m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch',
             use_dropout=False, gpu_ids=[], scale_factor=4, n_fmdrb=6,
             skip_compress_ratio=0.5):
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert torch.cuda.is_available()

    if which_model_netG != 'resunet_3d':
        print("Generator '%s' is deprecated; using 'resunet_3d'." % which_model_netG)
    if scale_factor != 1:
        print("ResUNetGenerator3D outputs the input spatial size; ignoring scale_factor=%s." % scale_factor)

    netG = ResUNetGenerator3D(input_nc, output_nc, base_filters=ngf)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[],
             wgan_gp=False):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert torch.cuda.is_available()
    if which_model_netD == 'basic':
        if wgan_gp:
            netD = Critic3D(input_nc, ndf, n_layers=3, norm_layer=norm_layer,
                             gpu_ids=gpu_ids)
        else:
            netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        if wgan_gp:
            netD = Critic3D(input_nc, ndf, n_layers_D, norm_layer=norm_layer,
                             gpu_ids=gpu_ids)
        else:
            netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# GAN loss: LSGAN (MSE) or vanilla GAN (BCE)
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# ---------------------------------------------------------------------------
#  128^3 ResUNet generator
# ---------------------------------------------------------------------------


class DownResBlock3D(nn.Module):
    """Residual downsampling block for 3D volumes."""

    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True):
        super().__init__()

        norm1 = nn.BatchNorm3d(out_ch) if use_norm else nn.Identity()
        norm2 = nn.BatchNorm3d(out_ch) if use_norm else nn.Identity()
        skip_norm = nn.BatchNorm3d(out_ch) if use_norm else nn.Identity()

        self.main = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1,
                      bias=not use_norm),
            norm1,
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1,
                      bias=not use_norm),
            norm2,
        )

        self.skip = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=2, padding=0,
                      bias=not use_norm),
            skip_norm,
        )

        self.out_act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.main(x) + self.skip(x))


class UpResBlock3D(nn.Module):
    """Residual upsampling block for 3D volumes."""

    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False):
        super().__init__()

        self.main = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose3d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm3d(out_ch),
        )

        self.skip = nn.Sequential(
            nn.ConvTranspose3d(
                in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=False
            ),
            nn.BatchNorm3d(out_ch),
        )

        self.dropout = nn.Dropout3d(0.5) if dropout else nn.Identity()
        self.out_act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x) + self.skip(x)
        x = self.out_act(x)
        x = self.dropout(x)
        return x


class ResUNetGenerator3D(nn.Module):
    """3D U-Net generator with residual blocks for same-size 128^3 outputs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 64):
        super().__init__()

        f = base_filters

        self.enc1 = DownResBlock3D(in_channels, f, use_norm=False)
        self.enc2 = DownResBlock3D(f, f * 2)
        self.enc3 = DownResBlock3D(f * 2, f * 4)
        self.enc4 = DownResBlock3D(f * 4, f * 8)
        self.enc5 = DownResBlock3D(f * 8, f * 8)
        self.enc6 = DownResBlock3D(f * 8, f * 8)
        # The 128^3 path reaches 1x1x1 here, so normalization must stay off.
        self.enc7 = DownResBlock3D(f * 8, f * 8, use_norm=False)

        self.dec1 = UpResBlock3D(f * 8, f * 8, dropout=True)
        self.dec2 = UpResBlock3D(f * 16, f * 8, dropout=True)
        self.dec3 = UpResBlock3D(f * 16, f * 8, dropout=True)
        self.dec4 = UpResBlock3D(f * 16, f * 4, dropout=False)
        self.dec5 = UpResBlock3D(f * 8, f * 2, dropout=False)
        self.dec6 = UpResBlock3D(f * 4, f, dropout=False)

        self.final = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose3d(f * 2, out_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        d1 = self.dec1(e7)
        d1 = torch.cat([d1, e6], dim=1)

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e5], dim=1)

        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e4], dim=1)

        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e3], dim=1)

        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e2], dim=1)

        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e1], dim=1)

        return self.final(d6)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class Critic3D(nn.Module):
    """PatchGAN-style 3D critic (no sigmoid) for WGAN-GP."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, gpu_ids=[]):
        super(Critic3D, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
