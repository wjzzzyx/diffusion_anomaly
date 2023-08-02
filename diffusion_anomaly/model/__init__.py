from . import denoising_diffusion


def get_model(config):
    if config.model == 'denoising_diffusion':
        return denoising_diffusion.Trainer(config)
    else:
        raise NotImplementedError()