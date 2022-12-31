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
                            localedir = 'locale',
                            languages=[now_locale], 
                            fallback=True).gettext
# Setup translation
_ = GetTranslationData()

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
        self.num_inference_steps = 8
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
                      'save_inferenced_image' : self.save_inferenced_image}
        with open(self.path, 'w', encoding="utf-8") as f:
            json.dump(save_value, f)
        pass
