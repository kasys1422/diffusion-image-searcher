import datetime
import math
import os
from pathlib import Path
from pickletools import float8
import re
from tokenize import tabsize
from scipy.spatial import distance
import numpy as np
import cv2
import dearpygui.dearpygui as dpg
from src.util.stable_diffusion_util import GenerateImage
from src.util.util import MessageboxWarn, _
import psutil

CURRENT_IMAGE = np.full((256, 256, 3), (37, 37, 37),np.uint8)
SEARCH_ONECE = False

# Calculate cosine similarity
def GetCosineSimilarity(vec1, vec2):
    return np.sum(vec1 * vec2) / (math.sqrt(np.sum(vec1 * vec1)) * math.sqrt(np.sum(vec2 * vec2)))

'''
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
    '''
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
    if dpg.does_item_exist("LoadingWindow"):
        dpg.set_item_pos("LoadingWindow", [dpg.get_item_width("MainWindow")/2-62,dpg.get_item_height("MainWindow")/2-75])
def DpgSetImageCallback(sender, app_data, user_data):
    global CURRENT_IMAGE
    CURRENT_IMAGE = user_data
    UpdateResultArea()

def ShowLoadingWindow():
    if not dpg.does_item_exist("LoadingWindow"):
        with dpg.window(label="", tag="LoadingWindow",pos=[dpg.get_item_width("MainWindow")/2-62,dpg.get_item_height("MainWindow")/2-75],width=124,height=150, no_resize=True, no_move=True, no_close=True, modal=True, menubar=False, no_scrollbar=True, no_background=True, no_title_bar=True):
            dpg.add_loading_indicator(radius=6, color=(48,172,255), secondary_color=(48,172,255))
                
        
def HideLoadingWindow():
    if dpg.does_item_exist("LoadingWindow"):
        dpg.delete_item("LoadingWindow")

def ImageSearch(settings, inference_data, pictures_dir_path, prompt, base_image_path=None, search_only=False):
    # Check params
    model_data = settings.stable_diffusion_models[[d['name'] for d in settings.stable_diffusion_models].index(settings.stable_diffusion_model_name)]
    # - about ram
    if model_data["min_ram"] > psutil.virtual_memory().total / (2**30):
        MessageboxWarn(_("Warning"), _("Insufficient memory to run the image generation model. You need {}GB to run. Please change the model or add more memory.").format(model_data["min_ram"]))
        HideLoadingWindow()
        return 0 
    if str(pictures_dir_path) == "None" or os.path.isdir(pictures_dir_path) == False:
        MessageboxWarn(_("Warning"), _("Please enter the correct path to the folder you wish to search."))
        HideLoadingWindow()
        return 0
    if str(base_image_path) == "" or os.path.exists(str(base_image_path)) == False and str(base_image_path) != "None":
        MessageboxWarn(_("Warning"), _("Please enter the correct path to the image you wish to search for."))
        HideLoadingWindow()
        return 0

    
    # Get model's threshold 
    try:
        threshold = float(model_data["threshold"])
    except:
        threshold = 0.35

    index = 1
    ShowLoadingWindow()
    
    if search_only == False:
        generated_image = GenerateImage(model_data=model_data,
                                        prompt=prompt,
                                        num_inference_steps=settings.num_inference_steps,
                                        init_image_path=base_image_path,
                                        output=None if settings.save_inferenced_image == True else "./img/"+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+" "+ prompt+".png")
        inference_data.StartInferenceAsync(generated_image)
    else:
        try:
            base_image = cv2.imread(base_image_path)
        except:
            MessageboxWarn(_("Warning"), _("File read error."))
            HideLoadingWindow()
            return 0
        inference_data.StartInferenceAsync(base_image)
    
    image_list = sorted([p for p in Path(pictures_dir_path).glob('**/*') if re.search('/*\\.(jpg|jpeg|png|gif|bmp|tiff)', str(p))])

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
        if cos_sim <= threshold:
            searched_images.append([image_path, cos_sim, image_frame])
            searched_images_names.append("[" + str(index) + "] " + str(image_path.name))
            dpg.add_group(parent="Result", horizontal=True, tag="item_" + str(index))
            DpgSetImage(image_frame,"img_" + str(index),"item_" + str(index), width=24, height = 24,min_width=24,min_height = 24)
            dpg.add_button(label=image_path.name,parent="item_" + str(index), callback=DpgSetImageCallback,user_data=searched_images[index-1][2])
            index += 1
    global CURRENT_IMAGE, SEARCH_ONECE
    CURRENT_IMAGE = searched_images[0][2]
    SEARCH_ONECE = True
    UpdateResultArea()
    HideLoadingWindow()
    return searched_images