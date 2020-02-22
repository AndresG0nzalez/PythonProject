from tkinter import *
from tkinter import filedialog
from PIL import ImageTk,Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


root = Tk()
root.title("Dog Breed Identifier")
w,h = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w,h-100))
root.resizable(False,False)

# Establishing frames on GUI
bFrame = Frame(root, padx=5, pady=5)
bFrame.pack(side=BOTTOM, fill=X, padx=10, pady=10)
lFrame = LabelFrame(root, width=1000, height=750)
lFrame.pack(side=LEFT, fill=BOTH, expand=False)
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
    global imTK
    root.filename = filedialog.askopenfilename(initialdir="/home/", title="Select a file", filetypes=(("jpeg","*.jpg"),("all","*.*")))
    imTK = ImageTk.PhotoImage(Image.open(root.filename))
    imLab = Label(lFrame, image=imTK).pack(fill=BOTH, expand=True)
    return ImageTk.PhotoImage(Image.open(root.filename))


def click():
    plt.show()


def data(pos, per, type):
    positions = pos
    percentage = per
    breeds = type
    plt.title("Dog Breed Likeliness")
    plt.bar(positions, height=percentage)
    plt.xticks(positions, breeds)


# sample data
WINNING_DOG = "rat"
SECOND_PLACE = "retriever"
THIRD_PLACE = "poodle"

# Applies labels and buttons
dog1= Label(rFrame1, text="Most likely: ").pack(side=LEFT)
dog2= Label(rFrame2, text="Next most likely: ").pack(side=LEFT)
dog3= Label(rFrame3, text="Least likely: ").pack(side=LEFT)
myLabel = Label(bFrame, text="Upload a photo of a dog.").pack()
myButton = Button(bFrame, text = "Upload", command=open).pack()
stats = Button(rFrame3, text="Statistics",command=click).pack(side=BOTTOM)
# root.iconbitmap('')

# puts image in gui bg
"""img = ImageTk.PhotoImage(Image.open("dog.jpg"))
lab = Label(image=img)
lab.pack()"""

imTK = ImageTk.PhotoImage(Image.open("/home/kylew/PycharmProjects/pyproj2020/dog.jpg"))

# sample images to test location. Need resizing function.
a = Label(rFrame1, text=WINNING_DOG).pack(side=TOP)
z = Label(rFrame1, image=imTK).pack(side=BOTTOM)
b = Label(rFrame2, text=SECOND_PLACE).pack(side=TOP)
x = Label(rFrame2, image=imTK).pack(side=BOTTOM)
c = Label(rFrame3, text=THIRD_PLACE).pack(side=TOP)
y = Label(rFrame3, image=imTK).pack(side=BOTTOM)


root.mainloop()
