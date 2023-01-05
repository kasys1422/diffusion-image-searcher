import gettext
import locale
import tkinter
import tkinter.filedialog as filedialog
from tkinter import messagebox
import os
import sys
import json
from pathlib import Path
import re
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS, GPSTAGS
import webbrowser
import numpy as np
import cv2


def Print(text):
    print(text)

def GetTranslationData(disable_translation=False):
    now_locale, _ = locale.getdefaultlocale()
   
    try:
        with open("res/language.json", 'r', encoding="utf-8") as f:
            load_value = json.load(f)
            now_locale = load_value['language']
    except:
        pass

    if disable_translation == True:
        now_locale = 'en-US'
        
    return gettext.translation(domain='messages',
                            localedir = './res/locale',
                            languages=[now_locale], 
                            fallback=True).gettext
# Setup translation
_ = GetTranslationData()

def OpenInExplorerCallback(sender, app_data, user_data):
    try:
        webbrowser.open(user_data)
    except:
        MessageboxError(_("Error"), _("Can not open explorer."))

def OpenFolder(initialdir, title=None):
    root = tkinter.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(initialdir=initialdir, title=title) 
    root.destroy()
    return folder_path

def OpenFile(initialdir, title=None, filetypes=None):
    root = tkinter.Tk()
    root.withdraw()
    if filetypes == None:
        folder_path = filedialog.askopenfilename(initialdir=initialdir, title=title)
    else:
        folder_path = filedialog.askopenfilename(initialdir=initialdir, title=title, filetypes=filetypes)
    root.destroy()
    return folder_path

def MessageboxWarn(title=None, message=None):
    root = tkinter.Tk()
    root.withdraw()
    messagebox.showwarning(title, message)
    root.destroy()

def MessageboxError(title=None, message=None, exit=False):
    root = tkinter.Tk()
    root.withdraw()
    messagebox.showerror(title, message)
    root.destroy()
    if exit == True:
        sys.exit()

class Log(object):
    def __init__(self, callback=None, filename=""):
        self.filename = filename
        if self.filename != "":
            self.l = open(filename, "w")
        self.callback = callback
    def write(self, mm):
        if self.filename != "":
            self.l.write(mm)
        if self.callback != None:
            self.callback(mm)
    def flush(self):
        if self.filename != "":
            self.l.flush()

def InitLogging(callback, filename=""):
    filename=Path(filename)
    if not os.path.exists(str(filename.parent)) and filename != "":
        os.mkdir(str(filename.parent))
    log = Log(callback, filename)
    sys.stdout = log
    sys.stderr = log

# Settings
class Settings():
    def __init__(self):
        self.path = "res/settings.json"
        self.search_pictures_path = ""
        if os.name == 'nt':
            self.pictures_path = os.path.expanduser('~\\Pictures')
        else:
            try:
                self.pictures_path = os.path.expanduser('~/Desktop')
            except:
                self.pictures_path = "/"
        model_data_list = self.GetModelList()
        self.check_image_similarity_models = []
        self.check_image_similarity_model_name = ""
        self.stable_diffusion_models = []
        self.stable_diffusion_model_name = ""
        for model in model_data_list:
            if model["usage"] == "check_image_similarity":
                self.check_image_similarity_models.append(model)
                self.check_image_similarity_model_name = model["name"]
            elif model["usage"] == "stable_diffusion":
                self.stable_diffusion_models.append(model)
                self.stable_diffusion_model_name = model["name"]
        if len(self.check_image_similarity_models) == 0 or len(self.stable_diffusion_models) == 0:
            MessageboxError("Error","There are missing models, please check the contents of res/model/",True)
        self.num_inference_steps = 6
        self.override_threshold = 0
        self.save_inferenced_image = False

        self.Load()

    def GetModelList(self):
        model_list = sorted([p for p in Path("./res/model").glob('**/*') if re.search('\\.modeltype$', str(p))])
        model_data_list = []
        for model in model_list:
            with open(model, 'r', newline='', encoding="utf-8") as f:
                text = re.sub(r'/\*[\s\S]*?\*/|//.*', '', f.read())
                json_text = json.loads(text)
                model_data_list.append(json_text)
        return model_data_list

    def Load(self):
        if not os.path.exists('res'):
            os.makedirs('res')
        try:
            with open( self.path, 'r', encoding="utf-8") as f:
                load_value = json.load(f)
                self.pictures_path = load_value['pictures_path']
                self.search_pictures_path = load_value['search_pictures_path']
                self.check_image_similarity_model_name = load_value['check_image_similarity_model_name']
                self.stable_diffusion_model_name = load_value['stable_diffusion_model_name']
                self.num_inference_steps = load_value['num_inference_steps']
                self.override_threshold = load_value['override_threshold']
                self.save_inferenced_image = load_value['save_inferenced_image']
        except:
            self.Save()
        pass

    def Save(self):
        if not os.path.exists('res'):
            os.makedirs('res')
        save_value = {'pictures_path' : self.pictures_path,
                      'search_pictures_path' : self.search_pictures_path,
                      'check_image_similarity_model_name' : self.check_image_similarity_model_name,
                      'stable_diffusion_model_name' : self.stable_diffusion_model_name,
                      'num_inference_steps' : self.num_inference_steps,
                      'override_threshold' : self.override_threshold,
                      'save_inferenced_image' : self.save_inferenced_image}
        with open(self.path, 'w', encoding="utf-8") as f:
            json.dump(save_value, f)
        pass

def GetExif(path):
    try:
        with Image.open(path) as im:
            exif = im.getexif()
    except:
        exif = None
    return exif

def MatchExif(exif, tag_list, none_val):
    res = []
    for tag in tag_list:
        val = none_val
        if exif is not None:
            for id, value in exif.items():
                #print(str(id) + " : " + str(value))
                try:
                    if tag == "GPSTag" and value != None and id == 34853:
                        gps = {}
                        for k, v in exif.get_ifd(34853).items():
                            gps[str(GPSTAGS.get(k, "Unknown"))] = v
                        latitude = float(gps["GPSLatitude"][0] + gps["GPSLatitude"][1] / 60 + gps["GPSLatitude"][2] / 3600)
                        if gps["GPSLatitudeRef"] != "N":
                            latitude = 0 - latitude
                        longitude = float(gps["GPSLongitude"][0] + gps["GPSLongitude"][1] / 60 + gps["GPSLongitude"][2] / 3600)
                        if gps["GPSLongitudeRef"] != "E":
                            longitude = 0 - longitude
                        val = '{:.06f}'.format(latitude) + "," + '{:.06f}'.format(longitude)
                    else:
                        if TAGS.get(id, id) == tag:
                            val = value
                except:
                    pass
        res.append(val)
    return res

def GetImageEgifTags(path, tag_list, none_val):
    return MatchExif(GetExif(path), tag_list, none_val)

def ImRead(path):
    try:
        pil_img = Image.open(path)
    except UnidentifiedImageError:
        return None
    try:
        img = np.array(pil_img)
    except:
        return None

    if img.ndim == 3:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except cv2.error:
            img = np.full((256, 256, 3), (37, 37, 37),np.uint8)
    return img
