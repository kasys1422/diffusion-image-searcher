import dearpygui.dearpygui as dpg

import src.util.util as Util
from src.util.util import _ , InitLogging
from src.search.search import ImageSearch, UpdateResultArea, CONSOLE_TEXT
import numpy as np
import sys
import os
import psutil
import re

# Global variable
VERSION = "0.0.2"
INIT_WINDOW = False

def Logging(input_text):
    if input_text == "\n":
        return
    global CONSOLE_TEXT
    input_text.replace("\n", "")
    CONSOLE_TEXT = CONSOLE_TEXT + input_text + "\n"
    if CONSOLE_TEXT.count('\n') > 200:
        CONSOLE_TEXT = CONSOLE_TEXT[CONSOLE_TEXT.find('\n', 0) + 1:]
    if INIT_WINDOW == True:
        if dpg.does_item_exist("Console"):
            dpg.set_value(item="Console",value=CONSOLE_TEXT)
            try: 
                if dpg.get_y_scroll_max('console_window') - 100.0 <= dpg.get_y_scroll('console_window') or (dpg.get_y_scroll_max('console_window') >= 1.0 and dpg.get_y_scroll_max('console_window') <= 20.0):
                    dpg.set_y_scroll('console_window', dpg.get_y_scroll_max('console_window') + 13.0)
            except SystemError:
                pass
        if dpg.does_item_exist("LoadingConsole"):
            dpg.set_value(item="LoadingConsole",value=CONSOLE_TEXT)
            dpg.set_y_scroll('LoadingConsoleWindow', dpg.get_y_scroll_max('LoadingConsoleWindow') + 13.0)

InitLogging(Logging, "./log/last_log.txt")

def CreateSettingWindow(sender, app_data, user_data):
    def SaveValues():
        user_data.stable_diffusion_model_name = dpg.get_value("SelectModel")
        user_data.num_inference_steps = int((str(re.search(r'\d+\.?\d*', dpg.get_value("NumInferenceSteps")).group())))
        user_data.override_threshold = float((str(re.search(r'\d+\.?\d*', dpg.get_value("OverrideThreshold")).group())))
        user_data.save_inferenced_image = dpg.get_value("SaveInferencedImage")
        user_data.show_info_when_search = dpg.get_value("ShowInfoWhenSearch")
        user_data.sort_result = dpg.get_value("SortResult")
        user_data.Save()
        dpg.delete_item("SettingWindow")

    if dpg.does_item_exist("SettingWindow"):
        #print("window already exists")
        pass
    else:
        with dpg.window(tag="SettingWindow", label=_("Settings"),pos=[dpg.get_item_width("MainWindow")/2-300,dpg.get_item_height("MainWindow")/2-250],height=500,width=600,on_close=OnWindowClose):
            with dpg.child_window(autosize_x =True,autosize_y =True ,horizontal_scrollbar=True):
                dpg.add_text(_("Image Generation Model"))
                list_buf = [ d['name'] for d in user_data.stable_diffusion_models]
                dpg.add_combo(list_buf,tag="SelectModel",width=500,default_value=user_data.stable_diffusion_model_name)
                with dpg.tooltip(parent="SelectModel"):
                    dpg.add_text(_("Trained model used for image generation."))
                dpg.add_separator()
                dpg.add_text(_("Num inference steps"))
                dpg.add_input_text(tag="NumInferenceSteps", decimal=True,default_value=user_data.num_inference_steps)
                with dpg.tooltip(parent="NumInferenceSteps"):
                    dpg.add_text(_("Number of steps of inference. The more steps, the better the accuracy, but the longer the inference time."))
                dpg.add_separator()
                dpg.add_text(_("Override threshold"))
                dpg.add_input_text(tag="OverrideThreshold", decimal=True,default_value=user_data.override_threshold)
                with dpg.tooltip(parent="OverrideThreshold"):
                    dpg.add_text(_("Overrides the threshold, which, if set to 0, automatically selects a threshold. The closer the value is to 1, the greater the similarity to the image."))
                dpg.add_separator()
                dpg.add_text(_("Save the inferred image"))
                dpg.add_checkbox(tag="SaveInferencedImage",default_value=user_data.save_inferenced_image)
                with dpg.tooltip(parent="SaveInferencedImage"):
                    dpg.add_text(_("Automatically stores inferred images under img/ ."))
                dpg.add_separator()
                dpg.add_text(_("Display detailed information during search"))
                dpg.add_checkbox(tag="ShowInfoWhenSearch",default_value=user_data.show_info_when_search)
                with dpg.tooltip(parent="ShowInfoWhenSearch"):
                    dpg.add_text(_("Displays command line information during search."))
                dpg.add_separator()
                dpg.add_text(_("Sort search results by similarity"))
                dpg.add_checkbox(tag="SortResult",default_value=user_data.sort_result)
                with dpg.tooltip(parent="SortResult"):
                    dpg.add_text(_("Sort the list of search results based on similarity to the search target."))
                dpg.add_separator()
                dpg.add_button(label=_("Save"), callback=SaveValues)
                pass

def CreateInfoWindow():
    if dpg.does_item_exist("InfoWindow"):
        #print("window already exists")
        pass
    else:
        with dpg.window(tag="InfoWindow", label=_("Information"),pos=[dpg.get_item_width("MainWindow")/2-300,dpg.get_item_height("MainWindow")/2-250],width=600,height=500, on_close=OnWindowClose):
            with dpg.child_window(autosize_x =True,autosize_y =True ,horizontal_scrollbar=True):
                dpg.add_text(_('Diffusion Image Searcher') + ' ' +_('version') + ' ' + VERSION)
                dpg.add_text(_('[website]') + ' ' + 'https://github.com/kasys1422/diffusion-image-searcher')
                dpg.add_text(_('[console]'))
                with dpg.child_window(tag='console_window', autosize_x =True,autosize_y =True ,horizontal_scrollbar=True):
                    dpg.add_text(CONSOLE_TEXT, tag="Console")

def CreateAboutWindow():
    if dpg.does_item_exist("TPLWindow"):
        #print("window already exists")
        pass
    else:
        f = open("./third_party_licenses.txt", encoding="utf-8")
        licenses = f.read()
        f.close()
        with dpg.window(tag="TPLWindow", label=_("Third party license"),pos=[dpg.get_item_width("MainWindow")/2-315,dpg.get_item_height("MainWindow")/2-250],height=500,width=630, on_close=OnWindowClose):
            with dpg.child_window(autosize_x =True,autosize_y =True ,horizontal_scrollbar=True):
                dpg.add_text(licenses)

def OnWindowClose(sender):
    # workaround
    children_dict = dpg.get_item_children(sender)
    for key in children_dict.keys():
        for child in children_dict[key]:
            dpg.delete_item(child)
    #print("window closed")
    dpg.delete_item(sender)

def DpgSyncValue(value, tag_list):
    for tag in tag_list:
        dpg.set_value(tag, value)

def SavePicturePath(value, tag_list, settings, settings_type=None):
    if value != "":
        DpgSyncValue(value, tag_list)
        if settings_type == None: 
            settings.pictures_path = value
        elif settings_type == "search_pictures_path":
            settings.search_pictures_path = value

def Main():
    # Load settings
    settings = Util.Settings()
    # alart about ram
    MIN_RAM = min([d['min_ram'] for d in settings.stable_diffusion_models])
    mem_bytes = psutil.virtual_memory().total / (2**30)
    limited_mode_text = ""
    print(f'Memory total: {mem_bytes} GiB')
    if mem_bytes < MIN_RAM:
        Util.MessageboxWarn(_("Not enough RAM"), str(_("A minimum of {} GB of RAM is required to use the full functionality of this software. Some functions are limited due to lack of RAM.").format(MIN_RAM)))
        limited_mode_text = _("[Limited mode]") + " "
    # Setup DearPyGUI
    dpg.create_context()
    dpg.create_viewport(title=limited_mode_text + _('Diffusion Image Searcher'), width=1280, height=720,min_width=720,min_height=640)
    dpg.setup_dearpygui()
    global INIT_WINDOW
    INIT_WINDOW = True



    # Setup window
    with dpg.font_registry():
        with dpg.font(file="./res/font/Mplus1-Medium.ttf", size = 18) as default_font:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese)
        dpg.bind_font(default_font)

    with dpg.texture_registry():
        img = np.full((300, 300, 4), (37, 37, 39, 0),np.uint8)
        dpg.add_dynamic_texture(300, 300, img, tag="DynamicTexture")

    with dpg.window(label="MainWindow", tag="MainWindow", horizontal_scrollbar=True):
        with dpg.menu_bar():
            
            dpg.add_menu_item(label=_("Settings"), callback=CreateSettingWindow, user_data=settings)
            with dpg.menu(label=_("Help")):
                dpg.add_menu_item(label=_("Information"), callback=CreateInfoWindow)
                if os.path.isfile("./third_party_licenses.txt"):
                    dpg.add_menu_item(label=_("Third party license"), callback=CreateAboutWindow)
        with dpg.child_window(autosize_x =True,autosize_y =True ,horizontal_scrollbar=True, tag="wb"):
            with dpg.tab_bar(label="TabBar", tag="TabBar"):
                if mem_bytes >= MIN_RAM:
                    with dpg.tab(label=_("Search by text"), tag="Text2Img"):
                        with dpg.group(horizontal=True):
                            dpg.add_text(_("Folder to search"))
                            dpg.add_input_text(tag="T2IFolder", default_value=settings.pictures_path, callback=lambda:SavePicturePath(dpg.get_value("T2IFolder"), ["T2IFolder", "ISFolder"], settings))
                            dpg.add_button(label=_("Select Folder"), callback=lambda:SavePicturePath(Util.OpenFolder("./img"), ["T2IFolder", "ISFolder"], settings))
                        with dpg.group(horizontal=True):
                            dpg.add_text(_('Prompt'))
                            dpg.add_input_text(label="", tag="T2IPrompt")
                        dpg.add_button(label=_("Search"), callback=lambda:ImageSearch(settings, dpg.get_value("T2IFolder"), dpg.get_value("T2IPrompt")))
                    '''
                    with dpg.tab(label="Search by text and image", tag="Img2Img"):
                        with dpg.group(horizontal=True):
                            dpg.add_text(_("Folder to search"))
                            dpg.add_input_text(tag="I2IFolder", default_value=settings.pictures_path, callback=lambda:SavePicturePath(dpg.get_value("I2IFolder"), ["T2IFolder", "I2IFolder", "ISFolder"], settings))
                            dpg.add_button(label=_("Select Folder"), callback=lambda:SavePicturePath(Util.OpenFolder("./img"), ["T2IFolder", "I2IFolder", "ISFolder"], settings))
                        with dpg.group(horizontal=True):
                            dpg.add_text(_("Files to search"))
                            dpg.add_input_text(tag="I2IFile", default_value=settings.search_pictures_path, callback=lambda:SavePicturePath(dpg.get_value("I2IFile"), ["I2IFile", "ISFile"], settings,"search_pictures_path"))
                            dpg.add_button(label=_("Select File"), callback=lambda:SavePicturePath(Util.OpenFile("./img"), ["I2IFile", "ISFile"], settings,"search_pictures_path"))
                        with dpg.group(horizontal=True):
                            dpg.add_text(_('Prompt'))
                            dpg.add_input_text(label="", tag="I2IPrompt")
                        dpg.add_button(label=_("Search"), callback=lambda:ImageSearch(settings, dpg.get_value("I2IFolder"), dpg.get_value("I2IPrompt"), dpg.get_value("I2IFile")))
                    #'''
                with dpg.tab(label=_("Search by image"), tag="ImgSearch"):
                    with dpg.group(horizontal=True):
                        dpg.add_text(_("Folder to search"))
                        dpg.add_input_text(tag="ISFolder", default_value=settings.pictures_path, callback=lambda:SavePicturePath(dpg.get_value("ISFolder"), ["T2IFolder","ISFolder"], settings))
                        dpg.add_button(label=_("Select Folder"), callback=lambda:SavePicturePath(Util.OpenFolder("./img"), ["T2IFolder","ISFolder"], settings))
                    with dpg.group(horizontal=True):
                        dpg.add_text(_("Files to search"))
                        dpg.add_input_text(tag="ISFile", default_value=settings.search_pictures_path, callback=lambda:SavePicturePath(dpg.get_value("ISFile"), ["ISFile"], settings,"search_pictures_path"))
                        dpg.add_button(label=_("Select File"), callback=lambda:SavePicturePath(Util.OpenFile("./img"), ["ISFile"], settings,"search_pictures_path"))
                    dpg.add_button(label=_("Search"), callback=lambda:ImageSearch(settings, dpg.get_value("ISFolder"), None, dpg.get_value("ISFile"),True))
            dpg.add_separator()
            dpg.add_text(_('Result'))
            with dpg.group(horizontal=False, tag="ResultText"):
                pass
            with dpg.group(horizontal=True, tag="ResultArea"):
                with dpg.child_window(autosize_x =False,width=560,autosize_y =True ,horizontal_scrollbar=True, tag="ResultListWindow"):
                    with dpg.group(horizontal=False, tag="Result"):
                        dpg.add_text(_("No file"))
                with dpg.child_window(autosize_x =True,autosize_y =True ,horizontal_scrollbar=True, tag="DynamicTextureWindow"):
                    with dpg.group(horizontal=False, tag="ResultPicture"):
                        dpg.add_image("DynamicTexture", tag ="DynamicImage")
                        
    dpg.set_primary_window("MainWindow", True)
    dpg.show_viewport()

    print("Diffusion Image Searcher version " + VERSION + " launch succeeded")

    prev_x, prev_y = [0, 0]

    # Loop
    while dpg.is_dearpygui_running():
        # Check window resize
        if prev_x != dpg.get_item_width("MainWindow") or prev_y != dpg.get_item_height("MainWindow"):
            prev_x = dpg.get_item_width("MainWindow")
            prev_y = dpg.get_item_height("MainWindow")
            dpg.set_item_width("ResultListWindow",int(dpg.get_item_width("MainWindow")/3))
            UpdateResultArea()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
    settings.Save()
    sys.exit()

if __name__ == '__main__':
    Main()