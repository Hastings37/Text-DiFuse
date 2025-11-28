import logging

logger = logging.getLogger("base")


def create_model(opt):
    model = opt["model"]


    if model == "sde":
        pass
        # from .sde_model import SDEModel as M 这种情况你如何操作；
    elif model == 'latent':
        from .latent_model import LatentModel as M
    elif model == "Fusion":
        from .fusion_model import FusionModel as M
    elif model == "latent_denoising":
        from .latent_denoising_model import DenoisingModel as M
    elif model=='my_latent':
        from .my_latent_model import LatentModel as M
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format(model))
    m = M(opt)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m
