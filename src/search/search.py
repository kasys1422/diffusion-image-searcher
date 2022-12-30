import datetime
import math
from pathlib import Path
import re
from xml.dom import minicompat
from scipy.spatial import distance
import numpy as np
import cv2
import dearpygui.dearpygui as dpg
from src.util.stable_diffusion_util import GenerateImage


CURRENT_IMAGE = np.full((256, 256, 3), (37, 37, 37),np.uint8)
SEARCH_ONECE = False

# Calculate cosine similarity
def GetCosineSimilarity(vec1, vec2):
    return np.sum(vec1 * vec2) / (math.sqrt(np.sum(vec1 * vec1)) * math.sqrt(np.sum(vec2 * vec2)))

def DpgSetImage(image, tag, width=300, height=300):
    image_height, image_width, _ = image.shape[:3]
    if image_height <= image_width:
        padding_v = int((image_width - image_height) / 2)
        padding_h = 0
    else:
        padding_v = 0
        padding_h = int((image_height - image_width) / 2)
    image = cv2.copyMakeBorder(image, padding_v, padding_v, padding_h, padding_h, cv2.BORDER_CONSTANT, (37, 37, 39, 0))
    image = cv2.resize(image , (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    dpg.set_value(tag, image/255)

def DpgSetImageOld(image, tag, width=300, height=300):
    image_height, image_width, _ = image.shape[:3]
    if image_height <= image_width:
        padding_v = int((image_width - image_height) / 2)
        padding_h = 0
    else:
        padding_v = 0
        padding_h = int((image_height - image_width) / 2)
    image = cv2.copyMakeBorder(image, padding_v, padding_v, padding_h, padding_h, cv2.BORDER_CONSTANT, (37, 37, 39, 0))
    image = cv2.resize(image , (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    dpg.set_value(tag, image/255)

# Convert DearPyGUI to handle OpenCV images
def ConvertImageOpenCVToDearPyGUI(image):
    data = np.flip(image, 2)
    data = data.ravel()
    data = np.asfarray(data, dtype='f')
    return np.true_divide(data, 255.0)

def DpgSetImage(image, tag, parent, width=None, height=None, min_width=256, min_height=256):
    # Resize frame and show it in window

    if width == None:
        width = dpg.get_item_width(parent)
    if height == None:
        height = dpg.get_item_height(parent)
    img_height, img_width = image.shape[:2]
    input_mag = max(width,min_width) / max(height,min_height)
    img_mag = img_width / img_height
    if input_mag <= img_mag:
        padding_v = int((img_height * img_mag / input_mag - img_height) / 2)
        padding_h = 0
        img_height = img_height * img_mag / input_mag 
    else:
        padding_v = 0
        padding_h = int((img_width / img_mag * input_mag - img_width) / 2)
        img_width = img_width / img_mag * input_mag
    image = cv2.copyMakeBorder(image, padding_v, padding_v, padding_h, padding_h, cv2.BORDER_CONSTANT, (37, 37, 39))
    print(dpg.get_item_height("MainWindow"))
    print(dpg.get_item_width("MainWindow"))
    frame_height = max(height, min_height)
    frame_width = max(width, min_width)
    if frame_width < frame_height * (img_width / img_height):
        frame_height = frame_width * (img_height / img_width)
    else:
        frame_width = frame_height * (img_width / img_height)
    buffer_frame = ConvertImageOpenCVToDearPyGUI(cv2.resize(image, (int(frame_width), int(frame_height))))
    dpg.delete_item(parent, children_only=True)
    dpg.delete_item(tag)
    with dpg.texture_registry(show=False):      
        dpg.add_raw_texture(int(frame_width), int(frame_height), buffer_frame, tag=tag, format=dpg.mvFormat_Float_rgb)
        dpg.add_image(tag, parent=parent)
    dpg.configure_item(tag, width=int(frame_width), height=int(frame_height))
    dpg.set_value(tag, buffer_frame)

def DpgSetImageListBoxCallback(sender, app_data, user_data):
    print(app_data)
    #DpgSetImage(user_data[int(app_data.lstrip("[").split("]")[0]) - 1][2], "DynamicTexture")
    DpgSetImage(user_data[int(app_data.lstrip("[").split("]")[0]) - 1][2], "DynamicTexture", "DynamicTextureWindow")

def UpdateResultArea():
    if SEARCH_ONECE == True:
        x = int(dpg.get_item_width("MainWindow") / 3 * 2)
        y = int(dpg.get_item_height("MainWindow"))
        dpg.delete_item("DynamicTextureWindow", children_only=True)
        if x > y:
            dpg.add_group(horizontal=True, tag="ResultPicture", parent="DynamicTextureWindow")
            width = int(dpg.get_item_width("MainWindow")/3)
            height = int(dpg.get_item_height("MainWindow")-240)
        else:
            dpg.add_group(horizontal=False, tag="ResultPicture", parent="DynamicTextureWindow")
            width = int(dpg.get_item_width("MainWindow")/3*2-60)
            height = int(dpg.get_item_height("MainWindow")/2.5)
        DpgSetImage(CURRENT_IMAGE, "DynamicTexture", "ResultPicture", width=width, height=height)
        dpg.add_group(horizontal=False, tag="ResultInfo", parent="ResultPicture")
        dpg.add_text("test", parent="ResultInfo")

def DpgSetImageCallback(sender, app_data, user_data):
    global CURRENT_IMAGE
    CURRENT_IMAGE = user_data
    UpdateResultArea()


def ImageSearch(inference_data, pictures_dir_path, prompt, search_only=False):
    
    index = 1
    if search_only == False:
        generated_image = GenerateImage(device="CPU",
                                        prompt=prompt,
                                        num_inference_steps=8,
                                        output="./img/"+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+" "+ prompt+".png")
        inference_data.StartInferenceAsync(generated_image)
    else:
        inference_data.StartInferenceAsync(cv2.imread(str("./img/20221215144044ramen on the wood table.png")))
    
    image_list = sorted([p for p in Path(pictures_dir_path).glob('**/*') if re.search('/*\\.(jpg|jpeg|png|gif|bmp|tiff)', str(p))])
    #print(image_list)

    searched_images = []
    searched_images_names = []

    dpg.delete_item("Result", children_only=True)
    
    while 1:
        if inference_data.exec_net.requests[0].wait(-1) == 0:
            generated_image_vector = inference_data.GetInferenceDataAsync()
            break
    
    for image_path in image_list:
        image_frame = cv2.imread(str(image_path))
        image_vec = inference_data.InferenceData(image_frame)
        #cos_sim = GetCosineSimilarity(image_vec, generated_image_vector)
        cos_sim = distance.cdist(image_vec, generated_image_vector, 'cosine')[0]
        print([image_path, cos_sim])
        if cos_sim <= 0.35:
            searched_images.append([image_path, cos_sim, image_frame])
            searched_images_names.append("[" + str(index) + "] " + str(image_path.name))
            dpg.add_group(parent="Result", horizontal=True, tag="item_" + str(index))
            DpgSetImage(image_frame,"img_" + str(index),"item_" + str(index), width=24, height = 24,min_width=24,min_height = 24)
            dpg.add_button(label=image_path.name,parent="item_" + str(index), callback=DpgSetImageCallback,user_data=searched_images[index-1][2])
            index += 1
    #dpg.add_listbox(searched_images_names,parent="Result",callback=DpgSetImageListBoxCallback,user_data=searched_images,width=540,num_items=16)
    #DpgSetImage(searched_images[0][2], "DynamicTexture", 300,300)
    global CURRENT_IMAGE, SEARCH_ONECE
    CURRENT_IMAGE = searched_images[0][2]
    SEARCH_ONECE = True
    UpdateResultArea()
    return searched_images