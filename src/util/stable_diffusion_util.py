import src.third_party.stable_diffusion_openvino.stable_diffusion_openvino_util as sd_opevino
import cv2
import numpy as np
import torch
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from src.util.util import MessageboxError
from src.inference.inference import CheckCanUseDevice
from pathlib import Path

def GenerateImage(model_data, prompt,num_inference_steps, output,init_image_path):
    # [Default] use OpenVINO and CPU
    if model_data["type"] == "openvino":
        image = sd_opevino.GenerateImage(prompt=prompt,
                                         device=CheckCanUseDevice(model_data["device"]),
                                         model=model_data["path"],
                                         num_inference_steps=int(num_inference_steps),
                                         init_image_path=init_image_path)
        pass

    # [Experimental] use Torch and cuda
    # If you want to use torch, you must install a version that is compatible with cuda. Please edit the requirements.txt file or install it manually.
    elif model_data["type"] == "torch":
        if init_image_path == None:
            pipeline = StableDiffusionPipeline
        else:
            pipeline = StableDiffusionImg2ImgPipeline
        if model_data["device"] == "cuda_fp16" or model_data["device"] == "CUDA_FP16" :
            pipe = pipeline.from_pretrained(
                model_data["path"],
                torch_dtype=torch.float16,
                revision="fp16",
            ).to("cuda")
            try:
                if model_data["enable_sequential_cpu_offload"] == True:
                    pipe.enable_sequential_cpu_offload()
            except:
                pass
            try:
                if model_data["enable_attention_slicing"] == True:
                    pipe.enable_attention_slicing(1)
            except:
                pass
            try:
                image = cv2.cvtColor(np.array(pipe(prompt,num_inference_steps=int(num_inference_steps)).images[0], dtype=np.uint8), cv2.COLOR_RGB2BGR)
            except torch.cuda.OutOfMemoryError as e:
                MessageboxError("OutOfMemoryError", e)
                return
            except RuntimeError as e:
                MessageboxError("RuntimeError", e)
                return
            except:
                MessageboxError("UnexpectedError", "Unexpected error.")
                return
        else:
            pipe = pipeline.from_pretrained(
                model_data["path"],
                num_inference_steps=int(num_inference_steps),
            ).to("cpu")
            try:
                if model_data["enable_attention_slicing"] == True:
                    pipe.enable_attention_slicing(1)
            except:
                pass
            try:
                image = cv2.cvtColor(np.array(pipe(prompt,num_inference_steps=int(num_inference_steps)).images[0], dtype=np.uint8), cv2.COLOR_RGB2BGR)
            except RuntimeError as e:
                MessageboxError("RuntimeError", e)
                return
            except:
                MessageboxError("UnexpectedError", "Unexpected error.")
                return

    if output != None:
        print("save img to " + output)
        path = Path(output)
        if not os.path.exists(str(path.parent)):
            os.mkdir(str(path.parent))
        cv2.imwrite(str(path), image)
    return image