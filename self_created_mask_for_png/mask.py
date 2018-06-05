from PIL import Image
from math import floor
import numpy as np
import os

data_dir = "monochrome/"
files = [f for f in os.listdir(data_dir) if not f.startswith('.')]
for num,file in enumerate(files):
    print(file)
    col = Image.open(data_dir + file)

    bw = np.asarray(col).copy()
    sum = 0
    cnt = 0
    r,c = bw.shape
    mid_r = floor(r/2)
    mid_c = floor(c/2)

    start_r = 0
    start_c = 0
    end_r = r
    end_c = c

    for ir in range(r):
        if bw[ir][mid_c] == 0:
            start_r = ir
            break

    for ic in range(c):
        if bw[mid_r][ic] == 0:
            start_c = ic
            break

    for ir in range(r):
        if bw[r-ir-1][mid_c] == 0:
            end_r = r-ir-1
            break

    for ic in range(c):
        if bw[mid_r][c-ic-1] == 0:
            end_c = c-ic-1
            break

    # run BFS to remove all black frame

    for i in range(r):
        for j in range(c):
            if not (i >= start_r and i <= end_r and j >= start_c and j <= end_c):
                if bw[i][j] != 0:
                    bw[i][j] = 0

    
    ngbh = [(0,0)]
    while (len(ngbh) != 0):
        curr_x, curr_y = ngbh[0]
        bw[curr_x][curr_y] = 255
        ngbh = ngbh[1:]  
        if (curr_x-1 >= 0 and bw[curr_x-1][curr_y] == 0): #left
            bw[curr_x-1][curr_y] = 255
            ngbh.append((curr_x-1, curr_y))
        if (curr_y-1 >= 0 and bw[curr_x][curr_y-1] == 0): #down
            ngbh.append((curr_x, curr_y-1))
            bw[curr_x][curr_y-1] = 255
        if (curr_y+1 < c and bw[curr_x][curr_y+1] == 0): #up
            ngbh.append((curr_x, curr_y+1))
            bw[curr_x][curr_y+1] = 255
        if (curr_x+1 < r and bw[curr_x+1][curr_y] == 0): #right
            ngbh.append((curr_x+1, curr_y))
            bw[curr_x+1][curr_y] = 255


    # filter the lungs out

    for i in range(floor(r/8)):
        for j in range(floor(c/8)):
            cnt = 0
            for k in range(8):
                for l in range(8):
                    if bw[i*8+k][j*8+l] == 0:
                        cnt += 1
            if cnt <= 32:
                for k in range(8):
                    for l in range(8):
                        if bw[i*8+k][j*8+l] != 0:
                            bw[i*8+k][j*8+l] = 128
                        

    img = Image.fromarray(bw)
    img.save("./grey/" + file)
    
    
