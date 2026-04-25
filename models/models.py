def create_model(opt, perceptual_model=None):
    from .LohiResNet_dis_patch_wo_lnp import LohiResNet_dis_patch_wo_lnp
    model = LohiResNet_dis_patch_wo_lnp()
    model.initialize(opt, perceptual_model=perceptual_model)
    return model
