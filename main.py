from turtle import width
import dearpygui.dearpygui as dpg
from src.inference.inference import SetupInference
import src.util.util as Util
from src.search.search import ImageSearch, UpdateResultArea
import numpy as np


import psutil

def CreateSettingWindow():
    if dpg.does_item_exist("SettingWindow"):
        print("window already exists")
    else:
        with dpg.window(tag="SettingWindow", on_close=OnWindowClose):
            for i in range(0, 5):
                dpg.add_button(label=f"button_{i}", tag=f"button_{i}")

def CreateHelpWindow():
    if dpg.does_item_exist("HelpWindow"):
        print("window already exists")
    else:
        with dpg.window(tag="HelpWindow", on_close=OnWindowClose):
            for i in range(0, 5):
                dpg.add_button(label=f"button_{i}", tag=f"button_{i}")

def CreateAboutWindow():
    if dpg.does_item_exist("AboutWindow"):
        print("window already exists")
    else:
        with dpg.window(tag="AboutWindow", on_close=OnWindowClose):
            dpg.add_text("about")

def OnWindowClose(sender):
    # workaround
    children_dict = dpg.get_item_children(sender)
    for key in children_dict.keys():
        for child in children_dict[key]:
            dpg.delete_item(child)

    dpg.delete_item(sender)
    print("window was deleted")


def DpgSyncValue(value, tag_list):
    for tag in tag_list:
        dpg.set_value(tag, value)

def Main():
    # Setup translation
    _ = Util.GetTranslationData()

    # alart about ram
    MIN_RAM = 12
    mem_bytes = psutil.virtual_memory().total / (2**30)
    limited_mode_text = ""
    print(f'Memory total: {mem_bytes} GiB')
    if mem_bytes < MIN_RAM:
        Util.MessageboxWarn(_("Not enough RAM"), str(_("A minimum of {} GB of RAM is required to use the full functionality of this software. Some functions are limited due to lack of RAM.").format(MIN_RAM)))
        limited_mode_text = _("[Limited mode]") + " "
    # Setup DearPyGUI
    dpg.create_context()
    dpg.create_viewport(title=limited_mode_text + _('Text to Image Searching System'), width=1280, height=720)
    dpg.setup_dearpygui()

    # Setup OpenVINO for image search
    openvino_ie = SetupInference()
    search_inference_data = openvino_ie.SetupModel("./res/model/efficientnet_lite0_feature-vector_2/efficientnet_lite0_feature-vector_2")

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
            
            dpg.add_menu_item(label="Settings", callback=CreateSettingWindow)
            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="Information", callback=CreateHelpWindow)
                dpg.add_menu_item(label="About", callback=CreateAboutWindow)
        with dpg.child_window(autosize_x =True,autosize_y =True ,horizontal_scrollbar=True, label="Other Themes", tag="wb"):
            

            #dpg.add_text(_('Welcome to Text to Image Searching System'))
            with dpg.tab_bar(label="TabBar", tag="TabBar"):
                if mem_bytes >= MIN_RAM:
                    with dpg.tab(label=_("Search by text"), tag="Text2Img"):
                        with dpg.group(horizontal=True):
                            dpg.add_text(_("Folder to search"))
                            dpg.add_input_text(tag="T2IFolder", callback=lambda:DpgSyncValue(dpg.get_value("T2IFolder"), ["T2IFolder", "I2IFolder", "ISFolder"]))
                            dpg.add_button(label=_("Select Folder"), callback=lambda:DpgSyncValue(Util.OpenFolder("./img"), ["T2IFolder", "I2IFolder", "ISFolder"]))
                        with dpg.group(horizontal=True):
                            dpg.add_text(_('Prompt'))
                            dpg.add_input_text(label="", tag="T2IPrompt")
                        dpg.add_button(label=_("Search"), callback=lambda:ImageSearch(search_inference_data, "./img", dpg.get_value("T2IPrompt")))
                        dpg.add_text("")
                
                    with dpg.tab(label="Search by text and image", tag="Img2Img"):
                        with dpg.group(horizontal=True):
                            dpg.add_text(_("Folder to search"))
                            dpg.add_input_text(tag="I2IFolder", callback=lambda:DpgSyncValue(dpg.get_value("I2IFolder"), ["T2IFolder", "I2IFolder", "ISFolder"]))
                            dpg.add_button(label=_("Select Folder"), callback=lambda:DpgSyncValue(Util.OpenFolder("./img"), ["T2IFolder", "I2IFolder", "ISFolder"]))
                        with dpg.group(horizontal=True):
                            dpg.add_text(_("Files to search"))
                            dpg.add_input_text(tag="I2IFile", callback=lambda:DpgSyncValue(dpg.get_value("I2IFile"), ["I2IFile", "ISFolder"]))
                            dpg.add_button(label=_("Select File"), callback=lambda:DpgSyncValue(Util.OpenFile("./img"), ["I2IFile", "ISFolder"]))
                        with dpg.group(horizontal=True):
                            dpg.add_text(_('Prompt'))
                            dpg.add_input_text(label="", tag="I2IPrompt")
                        dpg.add_button(label=_("Search"), callback=lambda:ImageSearch(search_inference_data, "./img", dpg.get_value("I2IPrompt")))

                with dpg.tab(label=_("Search by image"), tag="ImgSearch"):
                    with dpg.group(horizontal=True):
                        dpg.add_text(_("Folder to search"))
                        dpg.add_input_text(tag="ISFolder", callback=lambda:DpgSyncValue(dpg.get_value("ISIFolder"), ["T2IFolder", "I2IFolder", "ISFolder"]))
                        dpg.add_button(label=_("Select Folder"), callback=lambda:DpgSyncValue(Util.OpenFolder("./img"), ["T2IFolder", "I2IFolder", "ISFolder"]))
                    with dpg.group(horizontal=True):
                        dpg.add_text(_("Files to search"))
                        dpg.add_input_text(tag="ISFile", callback=lambda:DpgSyncValue(dpg.get_value("ISFile"), ["I2IFile", "ISFolder"]))
                        dpg.add_button(label=_("Select File"), callback=lambda:DpgSyncValue(Util.OpenFile("./img"), ["I2IFile", "ISFolder"]))
                    dpg.add_button(label=_("Search"), callback=lambda:ImageSearch(search_inference_data, "./img", "", True))
                    dpg.add_text("")

            dpg.add_text(_('Result'))
            with dpg.group(horizontal=True, tag="ResultArea"):
                with dpg.child_window(autosize_x =False,width=560,autosize_y =True ,horizontal_scrollbar=True, tag="ResultListWindow"):
                    with dpg.group(horizontal=False, tag="Result"):
                        dpg.add_listbox([_("No file")],parent="Result",width=540,num_items=16)
                with dpg.child_window(autosize_x =True,autosize_y =True ,horizontal_scrollbar=True, tag="DynamicTextureWindow"):
                    with dpg.group(horizontal=False, tag="ResultPicture"):
                        dpg.add_image("DynamicTexture", tag ="DynamicImage")
                        
                
    dpg.set_primary_window("MainWindow", True)
    dpg.show_viewport()
    #dpg.start_dearpygui()

    prev_x, prev_y = [0, 0]
    # Loop
    while dpg.is_dearpygui_running():
        # Check window resize
        if prev_x != dpg.get_item_width("MainWindow") or prev_y != dpg.get_item_height("MainWindow"):
            prev_x = dpg.get_item_width("MainWindow")
            prev_y = dpg.get_item_height("MainWindow")
            dpg.set_item_width("ResultListWindow",int(dpg.get_item_width("MainWindow")/3))
            print(dpg.get_value("ResultPicture"))
            UpdateResultArea()
            print("resized")
            
        
        dpg.render_dearpygui_frame()
    dpg.destroy_context()

    pass



if __name__ == '__main__':
    Main()
    pass