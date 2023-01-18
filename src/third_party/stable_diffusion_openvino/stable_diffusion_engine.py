###############################################################################################
# This source code was edited by kasys1422 based on stable_diffusion.openvino                 #
# (https://github.com/bes-dev/stable_diffusion.openvino). The license of the                  #
# original repository is Apache-2.0 license. The full license is described in                 #
# the "LICENSE" file.                                                                         #
#                                                                                             #
# [Changes]                                                                                   #
# - Modified to use offline files instead of downloading models from huggingface_hub.         #
# - Reduced memory usage by dynamically loading models.                                       #
# - Support for diffusers 0.11.1                                                              #
#                                                                                             #
# [Original file]                                                                             #
# https://github.com/bes-dev/stable_diffusion.openvino/blob/master/stable_diffusion_engine.py #
###############################################################################################

import inspect
import numpy as np
# openvino
import openvino.runtime as ov
# tokenizer
from transformers import CLIPTokenizer
# utils
from tqdm import tqdm
#from huggingface_hub import hf_hub_download
from diffusers import LMSDiscreteScheduler, PNDMScheduler
import cv2
import torch


def result(var):
    return next(iter(var.values()))

class StableDiffusionEngine:
    def __init__(
            self,
            scheduler,
            tokenizer,
            model="./res/model/stable-diffusion-v1-4-openvino-int8",
            device="CPU"
    ):
        self.model = model
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
        self.scheduler = scheduler
        # models
        self.core = ov.Core()
        self.core.set_property("CPU", ov.properties.intel_cpu.denormals_optimization(True))
        
        # encoder
        self._vae_encoder = self.core.read_model(
            self.model + "/" + "vae_encoder.xml",
            self.model + "/" + "vae_encoder.bin"
        )
        self.init_image_shape = tuple(self._vae_encoder.inputs[0].shape)[2:]

    def _preprocess_mask(self, mask):
        h, w = mask.shape
        if h != self.init_image_shape[0] and w != self.init_image_shape[1]:
            mask = cv2.resize(
                mask,
                (self.init_image_shape[1], self.init_image_shape[0]),
                interpolation = cv2.INTER_NEAREST
            )
        mask = cv2.resize(
            mask,
            (self.init_image_shape[1] // 8, self.init_image_shape[0] // 8),
            interpolation = cv2.INTER_NEAREST
        )
        mask = mask.astype(np.float32) / 255.0
        mask = np.tile(mask, (4, 1, 1))
        mask = mask[None].transpose(0, 1, 2, 3)
        mask = 1 - mask
        return mask

    def _preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[1:]
        if h != self.init_image_shape[0] and w != self.init_image_shape[1]:
            image = cv2.resize(
                image,
                (self.init_image_shape[1], self.init_image_shape[0]),
                interpolation=cv2.INTER_LANCZOS4
            )
        # normalize
        image = image.astype(np.float32) / 255.0
        image = 2.0 * image - 1.0
        # to batch
        image = image[None].transpose(0, 3, 1, 2)
        return image

    def _encode_image(self, init_image):
        moments = result(self.vae_encoder.infer_new_request({
            "init_image": self._preprocess_image(init_image)
        }))
        mean, logvar = np.split(moments, 2, axis=1)
        std = np.exp(logvar * 0.5)
        latent = (mean + std * np.random.randn(*mean.shape)) * 0.18215
        return latent

    def __call__(
            self,
            prompt,
            init_image = None,
            mask = None,
            strength = 0.5,
            num_inference_steps = 32,
            guidance_scale = 7.5,
            eta = 0.0
    ):
        # text features
        self._text_encoder = self.core.read_model(
            self.model + "/" + "text_encoder.xml",
            self.model + "/" + "text_encoder.bin"
        )
        self.text_encoder = self.core.compile_model(self._text_encoder, self.device)
        del self._text_encoder
        # extract condition
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        ).input_ids
        text_embeddings = result(
            self.text_encoder.infer_new_request({"tokens": np.array([tokens])})
        )

        # do classifier free guidance
        if guidance_scale > 1.0:
            tokens_uncond = self.tokenizer(
                "",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True
            ).input_ids
            uncond_embeddings = result(
                self.text_encoder.infer_new_request({"tokens": np.array([tokens_uncond])})
            )
            text_embeddings = np.concatenate((uncond_embeddings, text_embeddings), axis=0)

        del self.text_encoder, self.tokenizer



        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # diffusion
        self._unet = self.core.read_model(
            self.model + "/" + "unet.xml",
            self.model + "/" + "unet.bin"
        )
        
        self.latent_shape = tuple(self._unet.inputs[0].shape)[1:]

        # initialize latent latent
        if init_image is None:
            latents = np.random.randn(*self.latent_shape)
            init_timestep = num_inference_steps
        else:
            self.vae_encoder = self.core.compile_model(self._vae_encoder, self.device)
            init_latents = self._encode_image(init_image)
            del self.vae_encoder
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)
            timesteps = np.array([[self.scheduler.timesteps[-init_timestep]]]).astype(np.long)
            noise = np.random.randn(*self.latent_shape)
            latents = self.scheduler.add_noise(torch.from_numpy(init_latents.astype(np.float32)).clone(), torch.from_numpy(noise.astype(np.float32)).clone(), torch.from_numpy(timesteps.astype(np.int64)).clone())[0]

        del self.latent_shape, self._vae_encoder

        if init_image is not None and mask is not None:
            mask = self._preprocess_mask(mask)
        else:
            mask = None

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if init_image is None:
            latents = latents * self.scheduler.init_noise_sigma.to('cpu').detach().numpy().copy()
        else:
            latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        t_start = max(num_inference_steps - init_timestep + offset, 0)

        self.unet = self.core.compile_model(self._unet, self.device)
        del self._unet

        for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.stack([latents, latents], 0) if guidance_scale > 1.0 else latents[None]
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = result(self.unet.infer_new_request({
                "latent_model_input": latent_model_input,
                "t": t,
                "encoder_hidden_states": text_embeddings
            }))

            # perform guidance
            if guidance_scale > 1.0:
                noise_pred = noise_pred[0] + guidance_scale * (noise_pred[1] - noise_pred[0])

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, torch.from_numpy(latents.astype(np.float32)).clone(), **extra_step_kwargs).prev_sample.to('cpu').detach().numpy().copy()

            # masking for inapinting
            if mask is not None:
                init_latents_proper = self.scheduler.add_noise(init_latents, noise, t)
                latents = ((init_latents_proper * mask) + (latents * (1 - mask)))[0]

        del self.unet

        # decoder
        self._vae_decoder = self.core.read_model(
            self.model + "/" + "vae_decoder.xml",
            self.model + "/" + "vae_decoder.bin"
        )
        self.vae_decoder = self.core.compile_model(self._vae_decoder, self.device)

        image = result(self.vae_decoder.infer_new_request({
            "latents": np.expand_dims(latents, 0)
        }))

        del self.vae_decoder

        # convert tensor to opencv's image format
        image = (image / 2 + 0.5).clip(0, 1)
        image = (image[0].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
        return image
