def create_model(opt, perceptual_model=None):
    from .pix2pix3d_model import Pix2Pix3dModel
    model = Pix2Pix3dModel()
    model.initialize(opt, perceptual_model=perceptual_model)
    return model
