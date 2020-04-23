"""
Replace the conditional in dogidentifier.py with this

    if dog_detector(path):
        if multiple_breeds:
            return breeds, confidence, "dog"        # returns a tuple of ([breeds],[confidence], type of species)

    elif face_detector(path):
        return breeds, confidence, "human"
"""
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk,Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import dogIdentifier as dog
import numpy as np
import random

root = Tk()
root.title("Dog Breed Identifier")
w,h = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (1000,900))
root.resizable(False,False)

#App logo
pic = ImageTk.PhotoImage(Image.open('DBI.png'))
panel = Label(root, image=pic)
panel.pack()

# Establishing frames on GUI
bFrame = Frame(root, padx=5, pady=5)
bFrame.pack(side=BOTTOM, fill=X, padx=10, pady=10)
lFrame = LabelFrame(root, width=1000, height=750)
lFrame.pack(fill=BOTH, expand=False)
rFrame1 = Frame(root, width=450, height=175)
rFrame1.pack(side=TOP, fill=BOTH, expand=True, padx=10)
rFrame2 = Frame(root, width=450, height=175)
rFrame2.pack(fill=BOTH, expand=True, padx=10)
rFrame3 = Frame(root, width=450, height=175)
rFrame3.pack(fill=BOTH, expand=True, padx=10)

for frame in [lFrame,rFrame1,rFrame2,rFrame3]:
    frame.pack_propagate(0)

# uploads photo to GUI
def open():
    global lab
    global lab2
    global imK
    
    root.filename = filedialog.askopenfilename(initialdir="/home/", title="Select a file", filetypes=(("jpeg","*.jpg"),("all","*.*")))
    image = Image.open(root.filename)
    image.thumbnail((1000,750),Image.ANTIALIAS) #maintain the original photos ratio 
    imK = ImageTk.PhotoImage(image)
    imLab = Label(lFrame, image=imK).pack(fill=BOTH, expand=True)
    breed, conf, spec = dog.make_prediction(root.filename, True)
    if spec == "human":
        lab = Label(bFrame, text="Hello Human!\nYour DOGpelganger is a: ")
        lab.pack(side=LEFT)
        lab2 = Label(bFrame, text=breed[0].replace('_', ' '), font="Verdana 14 bold", fg='white', bg='black')
        lab2.pack(side=LEFT)
    else:
        lab = Label(bFrame, text="Woof Woof!\nThis dog is most likely a: ")
        lab.pack(side=LEFT)
        lab2 = Label(bFrame, text=breed[0].replace('_', ' '), font="Verdana 14 bold", fg='white', bg='black')
        lab2.pack(side=LEFT)

    panel.pack_forget() #remove logo to display uploaded photo
    pltData(conf, breed)
    return ImageTk.PhotoImage(Image.open(root.filename))

def clean():
    for widget in lFrame.winfo_children():
        widget.destroy()
    lab.destroy()
    lab2.destroy()
    myButton["state"] = NORMAL

def click():
	plt.show()

def pltData(conf, breed):
    per = list(conf)
    labels = list(breed)
    cmap = ['#BB5DA4', '#EF954B', '#D05149', '#74B185', '#3A72A4', 
            '#8D8FD8', '#E6F69D', '#AADEA7', '#64C2A6', '#2D87BB', '#FFBD66']	
    colors = []
    for i in range(0, len(labels)):
        colors.append(cmap[i])

    explode = [0.05] * len(labels)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(aspect="equal"))
    fig1 = plt.figure(1)
    plt.close()

    wedges, texts, autotexts = ax.pie(per, colors=colors, autopct='%1.1f%%',
                                      explode=explode, startangle=90, pctdistance=0.85)
    ax.legend(wedges, labels,
              title="Breeds",
              loc="center",
              bbox_to_anchor=(0.5, 0.5))

    plt.setp(autotexts, size=8, weight="bold")

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

    cir = plt.Circle((0,0), 0.70, fc='white')
    plt.gca().add_artist(cir)

    ax.set_title('Breed Prediction Breakdown', fontsize=22, fontweight='bold', loc='center')


# sample data
dogs = []
dogs.append("lab")
dogs.append("pug")
dogs.append("golden retriever")

percen = []
percen.append(0.74)
percen.append(0.56)
percen.append(0.13)

myLabel = Label(bFrame, text="Upload a photo of a dog.",foreground="black").pack()
myButton = Button(bFrame, text = "Upload", foreground="black", command=open).pack()
nextButton = Button(bFrame, text="Clean", foreground="black", command=clean).pack()
stats = Button(bFrame, text="Statistics", foreground="black",command=click).pack(side=RIGHT)

root.mainloop()
