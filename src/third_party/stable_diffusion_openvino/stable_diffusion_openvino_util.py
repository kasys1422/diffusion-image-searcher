###############################################################################################
# This source code was edited by kasys1422 based on stable_diffusion.openvino                 #
# (https://github.com/bes-dev/stable_diffusion.openvino). The license of the                  #
# original repository is Apache-2.0 license. The full license is described in                 #
# the "LICENSE" file.                                                                         #
#                                                                                             #
# [Changes]                                                                                   #
# - The main function in demo.py was used as the base for the GenerateImage function.         #
# - Changed the contents of "Args" to be input individually.                                  #
# - Changed so that no image is output if output is "None".                                   #
# - Specified "image" as return value.                                                        #
#                                                                                             #
# [Based file]                                                                                #
# https://github.com/bes-dev/stable_diffusion.openvino/blob/master/demo.py                    #
###############################################################################################
# engine
from .stable_diffusion_engine import StableDiffusionEngine
# scheduler
from diffusers import LMSDiscreteScheduler, PNDMScheduler
# utils
import cv2
import numpy as np

# Generate image function
def GenerateImage(model="./res/model/stable-diffusion-v1-4-openvino-fp16",
                  device="CPU",
                  seed=None,
                  beta_start=0.00085,
                  beta_end=0.012,
                  beta_schedule="scaled_linear",
                  num_inference_steps=32,
                  guidance_scale=7.5,
                  eta=0.0,
                  tokenizer="openai/clip-vit-large-patch14",
                  prompt="Street-art painting of Emilia Clarke in style of Banksy, photorealism",
                  init_image_path=None,
                  strength=0.5,
                  mask_image_path=None,
                  output=None):
    if seed is not None:
        np.random.seed(seed)
    if init_image_path is None:
        scheduler = LMSDiscreteScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            #tensor_format="np"
        )
    else:
        scheduler = PNDMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            skip_prk_steps = True,
            #tensor_format="np"
        )
    engine = StableDiffusionEngine(
        model = model,
        scheduler = scheduler,
        tokenizer = tokenizer,
        device = device
    )
    image = engine(
        prompt = prompt,
        init_image = None if init_image_path is None else cv2.imread(init_image_path),
        mask = None if mask_image_path is None else cv2.imread(mask_image_path, 0),
        strength = strength,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
        eta = eta
    )
    if output != None:
        cv2.imwrite(output, image)
    return image