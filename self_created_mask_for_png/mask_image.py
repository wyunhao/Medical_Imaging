from PIL import Image
import numpy as np
import os

data_dir = "stage1/"
mask_dir = "grey/"
patients = [f for f in os.listdir(data_dir) if not f.startswith('.')]

for num,patient in enumerate(patients):
    try:
        col = Image.open(data_dir + patient + "/2d.png")
        pixels = col.load()
        mask = Image.open(mask_dir + patient + ".png")
        bw = np.asarray(mask).copy()
        for i in range(len(bw)):
            for j in range(len(bw[i])):
                if bw[i][j] == 128:
                    r,g,b,a = pixels[j,i]
                    pixels[j,i] = (128,128,128,a)
        col.save("masked/"+patient +".png")
    except:
        print ("no mask")
