import src.third_party.stable_diffusion_openvino.stable_diffusion_openvino_util as sd_opevino
import cv2
import datetime

def GenerateImage(device, prompt,num_inference_steps, output):
    if device == "CPU":
        image = sd_opevino.GenerateImage(prompt=prompt,
                              num_inference_steps=num_inference_steps,
                              output="./img/"+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+" "+ prompt+".png")
        pass
    elif device == "CUDA":
        pass

    if output != None:
        cv2.imwrite(output, image)
    return image