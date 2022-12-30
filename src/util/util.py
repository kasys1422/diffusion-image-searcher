import gettext
import locale
import tkinter
import tkinter.filedialog as filedialog
from tkinter import messagebox

def Print(text):
    print(text)

def GetTranslationData(disable_translation=False):
    now_locale, _ = locale.getdefaultlocale()

    if disable_translation == True:
        now_locale = 'en-US'
    
    return gettext.translation(domain='messages',
                            localedir = 'locale',
                            languages=[now_locale], 
                            fallback=True).gettext

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

def MessageboxWarn(title, message):
    root = tkinter.Tk()
    root.withdraw()
    messagebox.showwarning(title, message)
    root.destroy()

class Settings:
    def __init__():
        pass
