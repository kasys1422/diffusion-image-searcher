import src.third_party.stable_diffusion_openvino.stable_diffusion_openvino_util as sd_opevino
import cv2
import datetime

def GenerateImage(model_data, prompt,num_inference_steps, output,init_image_path):
    
    print(model_data)
    if model_data["type"] == "openvino":
        image = sd_opevino.GenerateImage(prompt=prompt,
                                         device=model_data["device"],
                                         model="./res/model/" + model_data["path"],
                                         num_inference_steps=int(num_inference_steps),
                                         init_image_path=init_image_path,
                                         output="./img/"+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+" "+ prompt+".png")
        pass
    elif model_data["type"] == "torch":
        pass

    if output != None:
        cv2.imwrite(output, image)
    return image