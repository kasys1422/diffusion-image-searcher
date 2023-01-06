import datetime
import math
import os
from pathlib import Path
import time
import re
import numpy as np
import cv2
import dearpygui.dearpygui as dpg
from src.util.stable_diffusion_util import GenerateImage
from src.util.util import MessageboxWarn, _, GetImageEgifTags, OpenInExplorerCallback, ImRead
import psutil
import pyperclip

CURRENT_IMAGE = np.full((256, 256, 3), (37, 37, 37),np.uint8)
CURRENT_IMAGE_DATA = None
SEARCH_ONECE = False
INDEX = 1
CONSOLE_TEXT = ""

# Calculate cosine similarity
def GetCosineSimilarity(vec1, vec2):
    return np.sum(vec1 * vec2) / (math.sqrt(np.sum(vec1 * vec1)) * math.sqrt(np.sum(vec2 * vec2)))

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
    DpgSetImage(user_data[int(app_data.lstrip("[").split("]")[0]) - 1][2], "DynamicTexture", "DynamicTextureWindow")

def DpgAddCopyToClipBoardButton(label, parent, value, tag=None, width=None):
    if tag==None:
        tag = parent + "Button"
    dpg.add_button(label=label, tag=tag, width=width, parent=parent, callback=lambda:pyperclip.copy(value))
    with dpg.tooltip(parent=tag):
        dpg.add_text(_("Copy values to clipboard"))
    pass

def DpgAddPictureInfo(parent, tag, button_width,width, label ,value):
    dpg.add_group(horizontal=True, tag=tag, parent=parent)
    DpgAddCopyToClipBoardButton(label=label, parent=tag, width=button_width, value=value)
    dpg.add_input_text(readonly=True, default_value=value, parent=tag,width=width)

def UpdateResultArea(): 
    win_width = dpg.get_item_width("MainWindow")
    win_height = dpg.get_item_height("MainWindow")
    if SEARCH_ONECE == True:
        x = int(win_width / 3 * 1.8)
        y = int(win_height)
        dpg.delete_item("DynamicTextureWindow", children_only=True)
        button_width = 100
        if x > y:
            dpg.add_group(horizontal=True, tag="ResultPicture", parent="DynamicTextureWindow")
            img_width = int(win_width/3)
            img_height = int(win_height-240)
            info_text_width = img_width-190
            info_button_width = img_width-82
        else:
            dpg.add_group(horizontal=False, tag="ResultPicture", parent="DynamicTextureWindow")
            img_width = int(win_width/3*2-60)
            img_height = int(win_height/2.5)
            info_text_width = img_width-134
            info_button_width = img_width-26
        
        for i in range(INDEX-1):
            dpg.set_item_width(item="button_" + str(i + 1), width=max(int(win_width / 3)-70,250))

        # Show picture data
        exif_tags = GetImageEgifTags(str(CURRENT_IMAGE_DATA[0]), ["DateTime", "Model", "GPSTag"], _("No information"))
        DpgSetImage(CURRENT_IMAGE, "DynamicTexture", "ResultPicture", width=img_width, height=img_height)
        dpg.add_child_window(autosize_x =True,autosize_y =True ,horizontal_scrollbar=True, tag="ResultInfo", parent="ResultPicture")
        DpgAddPictureInfo(parent="ResultInfo", tag="ResultPicutureName", button_width=button_width, width=info_text_width, label=_("Name"), value=CURRENT_IMAGE_DATA[0].name)
        DpgAddPictureInfo(parent="ResultInfo", tag="ResultPicuturePath", button_width=button_width, width=info_text_width, label=_("Path"), value=str(CURRENT_IMAGE_DATA[0]))
        DpgAddPictureInfo(parent="ResultInfo", tag="ResultPicutureRes", button_width=button_width, width=info_text_width, label=_("Resolution"), value=str(CURRENT_IMAGE_DATA[2].shape[1])+"x"+str(CURRENT_IMAGE_DATA[2].shape[0]))
        DpgAddPictureInfo(parent="ResultInfo", tag="ResultPicutureDate", button_width=button_width, width=info_text_width, label=_("Date"), value=str(exif_tags[0]))
        DpgAddPictureInfo(parent="ResultInfo", tag="ResultPicutureDevice", button_width=button_width, width=info_text_width, label=_("Device"), value=str(exif_tags[1]))
        DpgAddPictureInfo(parent="ResultInfo", tag="ResultPicutureGPS", button_width=button_width, width=info_text_width, label=_("GPS"), value=str(exif_tags[2]))
        dpg.add_button(parent="ResultInfo", label=_("Open in explorer"), width=info_button_width, callback=OpenInExplorerCallback, user_data=CURRENT_IMAGE_DATA[0].parent)

    if dpg.does_item_exist("LoadingWindow"):
        dpg.set_item_pos("LoadingWindow", [int(dpg.get_item_width("MainWindow")/2-(dpg.get_item_width("MainWindow")/3)),int(dpg.get_item_height("MainWindow")/2-76)])
        dpg.set_item_width("LoadingWindow", int(dpg.get_item_width("MainWindow")/1.5))
        dpg.set_item_height("LoadingWindow", int(dpg.get_item_height("MainWindow")/2))
        dpg.set_item_indent("LoadingIcon", int(dpg.get_item_width("MainWindow")/3-60))

def DpgSetImageCallback(sender, app_data, user_data):
    global CURRENT_IMAGE, CURRENT_IMAGE_DATA
    CURRENT_IMAGE = user_data[2]
    CURRENT_IMAGE_DATA = user_data
    UpdateResultArea()

def ShowLoadingWindow(show_info = False):
    if not dpg.does_item_exist("LoadingWindow"):
        with dpg.window(label="", tag="LoadingWindow",pos=[int(dpg.get_item_width("MainWindow")/2-(dpg.get_item_width("MainWindow")/3)),int(dpg.get_item_height("MainWindow")/2-76)],width=int(dpg.get_item_width("MainWindow")/1.5),height=int(dpg.get_item_height("MainWindow")/2), no_resize=True, no_move=True, no_close=True, modal=True, menubar=False, no_scrollbar=True, no_background=True, no_title_bar=True):
            dpg.add_loading_indicator(tag="LoadingIcon", radius=6, color=(48,172,255), secondary_color=(48,172,255),indent=int(dpg.get_item_width("MainWindow")/3-60))
            if show_info == True:
                with dpg.child_window(tag='LoadingConsoleWindow', autosize_x =True,autosize_y =True ,horizontal_scrollbar=True):
                        dpg.add_text(CONSOLE_TEXT, tag="LoadingConsole")
        
def HideLoadingWindow():
    if dpg.does_item_exist("LoadingWindow"):
        dpg.delete_item("LoadingWindow")

def ImageSearch(settings, inference_data, pictures_dir_path, prompt, base_image_path=None, not_create_image=False):
    # Get start time
    start_time = time.time()

    # Check params
    model_data = settings.stable_diffusion_models[[d['name'] for d in settings.stable_diffusion_models].index(settings.stable_diffusion_model_name)]
    # Error check
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
    if prompt == "" and not_create_image == False:
        MessageboxWarn(_("Warning"), _("Please enter a prompt."))
        HideLoadingWindow()
        return 0

    # Get model's threshold 
    if settings.override_threshold == 0:
        try:
            threshold = float(model_data["threshold"])
        except:
            threshold = 0.65
    else:
        threshold = settings.override_threshold

    global INDEX
    INDEX = 1

    ShowLoadingWindow(settings.show_info_when_search)

    if not_create_image == False:
        print("[Info] Start image generate")
        generated_image = GenerateImage(model_data=model_data,
                                        prompt=prompt,
                                        num_inference_steps=settings.num_inference_steps,
                                        init_image_path=base_image_path,
                                        output=None if settings.save_inferenced_image == False else "./img/"+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+" "+ prompt+".png")
        if str(type(generated_image)) != "<class 'numpy.ndarray'>":
            HideLoadingWindow()
            return 0
        inference_data.StartInferenceAsync(generated_image)
        print("[Info] Complete image output")
    else:
        try:
            base_image = ImRead(base_image_path)
        except:
            MessageboxWarn(_("Warning"), _("File read error."))
            HideLoadingWindow()
            return 0
        inference_data.StartInferenceAsync(base_image)
    
    image_generate_time = time.time()

    print("[Info] Start image searching")
    image_list = sorted([p for p in Path(pictures_dir_path).glob('**/*') if re.search('/*\\.(jpg|jpeg|png|gif|bmp|tiff)', str(p))])

    searched_images = []
    searched_images_names = []

    dpg.delete_item("Result", children_only=True)
    
    while 1:
        if inference_data.exec_net.requests[0].wait(-1) == 0:
            generated_image_vector = inference_data.GetInferenceDataAsync()
            break
    
    num = 0

    for image_path in image_list:
        image_frame = ImRead(str(image_path))
        if image_frame is None:
            continue
        image_vec = inference_data.InferenceData(image_frame)
        cos_sim = GetCosineSimilarity(image_vec, generated_image_vector)
        print("[Info] value:" + str(cos_sim) + ", file:"  + str(image_path))
        if cos_sim >= threshold:
            searched_images.append([image_path, cos_sim, image_frame])
            searched_images_names.append("[" + str(INDEX) + "] " + str(image_path.name))
            dpg.add_group(parent="Result", horizontal=True, tag="item_" + str(INDEX))
            DpgSetImage(image_frame,"img_" + str(INDEX),"item_" + str(INDEX), width=24, height = 24,min_width=24,min_height = 24)
            dpg.add_button(label=image_path.name,tag="button_" + str(INDEX),parent="item_" + str(INDEX), callback=DpgSetImageCallback,user_data=searched_images[INDEX-1])
            INDEX += 1
        num += 1

    print("[Info] Complete image searching")
    print("[Info] Number of checked images = " + str(num))
    print("[Info] Number of similar images = " + str(INDEX - 1))
    if not_create_image == False:
        print("[Info] Image generate times = " + str(image_generate_time - start_time) + "(s)")
        print("[Info] Image searching times = " + str(time.time() - image_generate_time) + "(s)")
    print("[Info] Elapsed times = " + str(time.time() - start_time) + "(s)")
    if INDEX != 1:
        DpgSetImageCallback(None,None,searched_images[0])
        global SEARCH_ONECE
        SEARCH_ONECE = True
        UpdateResultArea()
        HideLoadingWindow()
        return searched_images
    else:
        UpdateResultArea()
        HideLoadingWindow()
        dpg.add_text(_("No file"), parent="Result")