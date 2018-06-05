from PIL import Image
import numpy as np
import os

data_dir = "stage1/"
files = [f for f in os.listdir(data_dir) if not f.startswith('.')]
for num,file in enumerate(files):
    col = Image.open(data_dir + file + "/2d.png")
    gray = col.convert('L')
    
# Let numpy do the heavy lifting for converting pixels to pure black or white
    bw = np.asarray(gray).copy()
    sum = 0
    cnt = 0
    for r in bw:
        for c in r:
            if c != 255:
                sum += c
                cnt += 1
    avg = sum / cnt
    print(avg)
# Pixel range is 0...255, 256/2 = 128
    bw[bw < avg] = 0    # Black
    bw[bw >= avg] = 255 # White

#convert rgb vector into supportvector array for SVM
    supportVector = np.zeros(shape = (len(bw),len(bw[0])))

    for r in range(len(bw)):
        for c in range(len(bw[r])):
            if bw[r][c] == 0:
                supportVector[r][c] = 1
    print(supportVector)
    print("\n")

# Now we put it back in Pillow/PIL land
    imfile = Image.fromarray(bw)
    imfile.save("monochrome/" + file + ".png")

