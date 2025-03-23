import logging
import torch
from stylesynth_diffusion_model import StyleSynth_DiffusionModel


class StyleSynth:
    '''
    StyleSynth is an AI system that uses a custom diffusion model to generate
    images of new articles of clothing!
    '''

    def __init__(self):
        # get logger
        self.logger = logging.getLogger('StyleSynth_Logger')
        self.logger.setLevel(logging.INFO)

        # set device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, epochs=10, batch_size=64, vis_interval=2,
              num_imgs_to_generate=3):
        # train U-Net to predict noise in an image
        self.diffusion_model = StyleSynth_DiffusionModel(
            T=150, w=2.0, upper_beta=0.02, device=self.device,
            logger=self.logger)
        self.diffusion_model.train(
            epochs=epochs, batch_size=batch_size,
            vis_interval=vis_interval,
            num_imgs_to_generate=num_imgs_to_generate)

    def generate(self, clothing_types):
        # generate images of new clothing using the diffusion model
        return self.diffusion_model.generate_images(clothing_types)
