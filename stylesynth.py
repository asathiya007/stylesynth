import logging
import torch
from stylesynth_diffusion_model import StyleSynth_DiffusionModel, \
    CLOTHING_TYPES


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

    def train(self, img_size=24, epochs=10, batch_size=64, vis_interval=2):
        # train U-Net to predict noise in an image
        self.diffusion_model = StyleSynth_DiffusionModel(
            img_size=img_size, T=200, w=3.0, upper_beta=0.02,
            device=self.device, logger=self.logger)
        self.diffusion_model.train(
            epochs=epochs, batch_size=batch_size, vis_interval=vis_interval)

    def generate(self, clothing_types):
        # check clothing types
        for clothing_type in clothing_types:
            if clothing_type not in CLOTHING_TYPES:
                err_msg = f'Invalid clothing type: {clothing_type}. Expected '\
                    + f'one of: {CLOTHING_TYPES}'
                self.logger.error(err_msg)
                raise ValueError(err_msg)

        # generate images of new clothing using the diffusion model
        return self.diffusion_model.generate_images(clothing_types)

    def save(self, save_path='./stylesynth_diffusion_model.pt'):
        if not hasattr(self, 'diffusion_model'):
            self.logger.warning(
                'No diffusion model has been trained, nothing to save')
        else:
            self.diffusion_model.save(save_path)
            self.logger.info(
                f'Saved StyleSynth diffusion model to {save_path}')

    def load(self, load_path):
        model_dict = torch.load(load_path)
        self.diffusion_model = StyleSynth_DiffusionModel(
            img_size=model_dict['img_size'], T=model_dict['T'],
            w=model_dict['w'], upper_beta=model_dict['upper_beta'],
            device=self.device, logger=self.logger)
        self.diffusion_model.load(model_dict)
        self.logger.info('Loaded StyleSynth diffusion model')
