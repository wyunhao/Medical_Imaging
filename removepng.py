import os

INPUT_FOLDER = './sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
patients.remove(".DS_Store")

def remove():
    for patient in patients:
        os.system("rm -rf " + INPUT_FOLDER + patient + "/*.png")    

if __name__ == "__main__":
    remove()
