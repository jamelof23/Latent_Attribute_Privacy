"""
This file has been updated from the InterFaceGAN repository (https://github.com/genforce/interfacegan), 
and modified to use in PrivAI (https://github.com/jamelof23/PrivAI). 
The modifications include enabling gradient-based updates and adding support for multi-GPU training.
"""

# python3.8
"""Contains the generator class of StyleGAN.

Basically, this class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import os
import numpy as np

import torch
import torch.nn as nn  # Make sure to import nn

from . import model_settings
from .stylegan_generator_model import StyleGANGeneratorModel
from .base_generator import BaseGenerator

__all__ = ['StyleGANGenerator']


class StyleGANGenerator(BaseGenerator):
    """Defines the generator class of StyleGAN.

    Different from conventional GAN, StyleGAN introduces a disentangled latent
    space (i.e., W space) besides the normal latent space (i.e., Z space). Then,
    the disentangled latent code, w, is fed into each convolutional layer to
    modulate the `style` of the synthesis through AdaIN (Adaptive Instance
    Normalization) layer. Normally, the w's fed into all layers are the same. But,
    they can actually be different to make different layers get different styles.
    Accordingly, an extended space (i.e. W+ space) is used to gather all w's
    together. Taking the official StyleGAN model trained on FF-HQ dataset as an
    instance, there are
    (1) Z space, with dimension (512,)
    (2) W space, with dimension (512,)
    (3) W+ space, with dimension (18, 512)
    """

    def __init__(self, model_name, logger=None):
        self.truncation_psi = model_settings.STYLEGAN_TRUNCATION_PSI
        self.truncation_layers = model_settings.STYLEGAN_TRUNCATION_LAYERS
        self.randomize_noise = model_settings.STYLEGAN_RANDOMIZE_NOISE
        self.model_specific_vars = ['truncation.truncation']
        super().__init__(model_name, logger)
        self.num_layers = (int(np.log2(self.resolution)) - 1) * 2
        assert self.gan_type == 'stylegan'

    def build(self):
        self.check_attr('w_space_dim')
        self.check_attr('fused_scale')
        self.model = StyleGANGeneratorModel(
            resolution=self.resolution,
            w_space_dim=self.w_space_dim,
            fused_scale=self.fused_scale,
            output_channels=self.output_channels,
            truncation_psi=self.truncation_psi,
            truncation_layers=self.truncation_layers,
            randomize_noise=self.randomize_noise)

    def load(self):
        self.logger.info(f'Loading pytorch model from `{self.model_path}`.')
        state_dict = torch.load(self.model_path)
        for var_name in self.model_specific_vars:
            state_dict[var_name] = self.model.state_dict()[var_name]
        self.model.load_state_dict(state_dict)
        self.logger.info(f'Successfully loaded!')
        # Handle DataParallel wrapping when accessing model attributes
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        self.lod = model.synthesis.lod.to(self.cpu_device).tolist()
        self.logger.info(f'  `lod` of the loaded model is {self.lod}.')

    def convert_tf_model(self, test_num=10):
        import sys
        import pickle
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        sys.path.append(model_settings.BASE_DIR + '/stylegan_tf_official')

        self.logger.info(f'Loading tensorflow model from `{self.tf_model_path}`.')
        tf.InteractiveSession()
        with open(self.tf_model_path, 'rb') as f:
            _, _, tf_model = pickle.load(f)
        self.logger.info(f'Successfully loaded!')

        self.logger.info(f'Converting tensorflow model to pytorch version.')
        tf_vars = dict(tf_model.__getstate__()['variables'])
        tf_vars.update(
            dict(tf_model.components.mapping.__getstate__()['variables']))
        tf_vars.update(
            dict(tf_model.components.synthesis.__getstate__()['variables']))
        state_dict = self.model.state_dict()
        for pth_var_name, tf_var_name in self.model.pth_to_tf_var_mapping.items():
            if 'ToRGB_lod' in tf_var_name:
                lod = int(tf_var_name[len('ToRGB_lod')])
                lod_shift = 10 - int(np.log2(self.resolution))
                tf_var_name = tf_var_name.replace(f'{lod}', f'{lod - lod_shift}')
            if tf_var_name not in tf_vars:
                self.logger.debug(f'Variable `{tf_var_name}` does not exist in '
                                  f'tensorflow model.')
                continue
            self.logger.debug(f'  Converting `{tf_var_name}` to `{pth_var_name}`.')
            var = torch.from_numpy(np.array(tf_vars[tf_var_name]))
            if 'weight' in pth_var_name:
                if 'dense' in pth_var_name:
                    var = var.permute(1, 0)
                elif 'conv' in pth_var_name:
                    var = var.permute(3, 2, 0, 1)
            state_dict[pth_var_name] = var
        self.logger.info(f'Successfully converted!')

        self.logger.info(f'Saving pytorch model to `{self.model_path}`.')
        for var_name in self.model_specific_vars:
            del state_dict[var_name]
        torch.save(state_dict, self.model_path)
        self.logger.info(f'Successfully saved!')

        self.load()

        # Official tensorflow model can only run on GPU.
        if test_num <= 0 or not tf.test.is_built_with_cuda():
            return
        self.logger.info(f'Testing conversion results.')
        self.model.eval().to(self.run_device)
        total_distance = 0.0
        for i in range(test_num):
            latent_code = self.easy_sample(1)
            tf_output = tf_model.run(latent_code,  # latents_in
                                     None,  # labels_in
                                     truncation_psi=self.truncation_psi,
                                     truncation_cutoff=self.truncation_layers,
                                     randomize_noise=self.randomize_noise)
            pth_output = self.synthesize(latent_code)['image']
            distance = np.average(np.abs(tf_output - pth_output))
            self.logger.debug(f'  Test {i:03d}: distance {distance:.6e}.')
            total_distance += distance
        self.logger.info(f'Average distance is {total_distance / test_num:.6e}.')

    def sample(self, num, latent_space_type='Z'):
        """Samples latent codes randomly.

        Args:
          num: Number of latent codes to sample. Should be positive.
          latent_space_type: Type of latent space from which to sample latent code.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

        Returns:
          A `numpy.ndarray` as sampled latent codes.

        Raises:
          ValueError: If the given `latent_space_type` is not supported.
        """
        latent_space_type = latent_space_type.upper()
        if latent_space_type == 'Z':
            latent_codes = np.random.randn(num, self.latent_space_dim)
        elif latent_space_type == 'W':
            latent_codes = np.random.randn(num, self.w_space_dim)
        elif latent_space_type == 'WP':
            latent_codes = np.random.randn(num, self.num_layers, self.w_space_dim)
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        return latent_codes.astype(np.float32)

    def preprocess(self, latent_codes, latent_space_type='Z'):
        """Preprocesses the input latent codes if needed."""
        if isinstance(latent_codes, np.ndarray):
            latent_codes = torch.from_numpy(latent_codes).float().to(self.run_device)
        elif isinstance(latent_codes, torch.Tensor):
            latent_codes = latent_codes.float().to(self.run_device)
        else:
            raise ValueError('Latent codes should be either a NumPy array or a PyTorch tensor!')

        latent_space_type = latent_space_type.upper()
        if latent_space_type == 'Z':
            latent_codes = latent_codes.view(-1, self.latent_space_dim)
        elif latent_space_type == 'W':
            latent_codes = latent_codes.view(-1, self.w_space_dim)
        elif latent_space_type == 'WP':
            latent_codes = latent_codes.view(-1, self.num_layers, self.w_space_dim)
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        return latent_codes

    def easy_sample(self, num, latent_space_type='Z'):
        return self.preprocess(self.sample(num, latent_space_type),
                               latent_space_type)

    def synthesize(self, latent_codes, latent_space_type='Z', generate_style=False, generate_image=True, **kwargs):
        """Synthesizes images with given latent codes.

        Args:
            latent_codes: Input latent codes for image synthesis.
            latent_space_type: Type of latent space to which the latent codes belong.
                Only ['Z', 'W', 'WP'] are supported. Case insensitive. (default: 'Z')
            generate_style: Whether to generate the layer-wise style codes. (default: False)
            generate_image: Whether to generate the final image synthesis. (default: True)

        Returns:
            A dictionary whose values are raw outputs from the generator.
        """
        if isinstance(latent_codes, np.ndarray):
            # Convert NumPy array to PyTorch tensor
            latent_codes = torch.from_numpy(latent_codes).float().to(self.run_device)
        elif isinstance(latent_codes, torch.Tensor):
            # Ensure tensor is on the correct device and of the right type
            latent_codes = latent_codes.float().to(self.run_device)
        else:
            raise ValueError('Latent codes should be either a NumPy array or a PyTorch tensor!')

        results = {}

        # Handle DataParallel wrapping
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        latent_space_type = latent_space_type.upper()
        latent_codes_shape = latent_codes.shape
        # Generate from Z space
        if latent_space_type == 'Z':
            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[1] == self.latent_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'latent_space_dim], where '
                                 f'latent_space_dim == {self.latent_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            zs = latent_codes
            ws = model.mapping(zs)
            wps = model.truncation(ws)
            results['z'] = zs
            results['w'] = ws
            results['wp'] = wps
        # Generate from W space
        elif latent_space_type == 'W':
            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[1] == self.w_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'w_space_dim], where '
                                 f'w_space_dim == {self.w_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            ws = latent_codes
            wps = model.truncation(ws)
            results['w'] = ws
            results['wp'] = wps
        # Generate from W+ space
        elif latent_space_type == 'WP':
            if not (len(latent_codes_shape) == 3 and
                    latent_codes_shape[1] == self.num_layers and
                    latent_codes_shape[2] == self.w_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'num_layers, w_space_dim], where '
                                 f'num_layers == {self.num_layers}, and w_space_dim == {self.w_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            wps = latent_codes
            results['wp'] = wps
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        if generate_style:
            for i in range(self.num_layers):
                style = model.synthesis.__getattr__(
                    f'layer{i}').epilogue.style_mod.dense(wps[:, i, :])
                results[f'style{i:02d}'] = style

        if generate_image:
            images = model.synthesis(wps)
            results['image'] = images

        return results
