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
from PIL import ImageTk,Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import dogidentifier as dog


root = Tk()
root.title("Dog Breed Identifier")
w,h = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (1000,850))
root.resizable(False,False)

# Establishing frames on GUI
bFrame = Frame(root, padx=5, pady=5)
bFrame.pack(side=BOTTOM, fill=X, padx=10, pady=10)
lFrame = LabelFrame(root, width=1000, height=750)
lFrame.pack(fill=BOTH, expand=False)
rFrame1 = Frame(root,width=450, height=175)
rFrame1.pack(side=TOP, fill=BOTH, expand=True, padx=10)
rFrame2 = Frame(root,width=450, height=175)
rFrame2.pack(fill=BOTH, expand=True, padx=10)
rFrame3 = Frame(root,width=450, height=175)
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
    image = image.resize((1000,750),Image.ANTIALIAS)
    imK = ImageTk.PhotoImage(image)
    imLab = Label(lFrame, image=imK).pack(fill=BOTH, expand=True)
    breed, conf, spec = dog.make_prediction(root.filename, True)
    if spec == "human":
        lab = Label(bFrame, text="Hello Human!\nYou look most like a: ")
        lab.pack(side=LEFT)
        lab2 = Label(bFrame, text=breed[0])
        lab2.pack(side=LEFT)
    else:
        lab = Label(bFrame, text="Woof Woof!\nThis dog is most likely a: ")
        lab.pack(side=LEFT)
        lab2 = Label(bFrame, text=breed[0])
        lab2.pack(side=LEFT)
    data((1, 2, 3, 4), conf, breed)
    return ImageTk.PhotoImage(Image.open(root.filename))


def clean():
    for widget in lFrame.winfo_children():
        widget.destroy()
    lab.destroy()
    lab2.destroy()
    myButton["state"] = NORMAL


def click():
    plt.show()


def data(pos, per, type):
    positions = pos
    percentage = per
    breeds = type
    plt.title("Dog Breed Likeliness")

    plt.bar(positions, height=percentage)
    plt.xticks(positions, breeds)
    plt.text(color="black")

# sample data
dogs = []
dogs.append("lab")
dogs.append("pug")
dogs.append("golden retriever")

percen = []
percen.append(0.74)
percen.append(0.56)
percen.append(0.13)

# Applies labels and buttons
# dog1= Label(mFrame, text="Most likely: ").pack(side=LEFT, pady=150)
# dog2= Label(mFrame, text="Next most likely: ").pack(side=LEFT,pady=150)
# dog3= Label(mFrame, text="Least likely: ").pack(side=LEFT)
myLabel = Label(bFrame, text="Upload a photo of a dog.",foreground="black").pack()
myButton = Button(bFrame, text = "Upload", foreground="black", command=open).pack()
nextButton = Button(bFrame, text="Clean", foreground="black", command=clean).pack()
stats = Button(bFrame, text="Statistics", foreground="black",command=click).pack(side=RIGHT)
# root.iconbitmap('')

# puts image in gui bg
"""img = ImageTk.PhotoImage(Image.open("dog.jpg"))
lab = Label(image=img)
lab.pack()
ima = Image.open("/home/kylew/PycharmProjects/pyproj2020/pug.jpeg")
ima = ima.resize((250,250), Image.ANTIALIAS)
imTK = ImageTk.PhotoImage(ima)

img = Image.open("/home/kylew/PycharmProjects/pyproj2020/doggo.jpeg")
img = img.resize((250,250), Image.ANTIALIAS)
im = ImageTk.PhotoImage(img)

ime = Image.open("/home/kylew/PycharmProjects/pyproj2020/greyhound.jpeg")
ime = ime.resize((250,250), Image.ANTIALIAS)
imT = ImageTk.PhotoImage(ime)

# sample images to test location. Need resizing function.
a = Label(rFrame1, text=WINNING_DOG).pack(side=TOP,)
# z = Label(rFrame1, image=imTK).pack(side=BOTTOM)
b = Label(rFrame2, text=SECOND_PLACE).pack(side=TOP)
# x = Label(rFrame2, image=imT).pack(side=BOTTOM)
c = Label(rFrame3, text=THIRD_PLACE).pack(side=TOP)
# y = Label(rFrame3, image=im).pack(side=BOTTOM)

"""
root.mainloop()
