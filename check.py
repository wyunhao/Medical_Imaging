import os

def check():
    INPUT_PATH = "./stage1/"
    dirs = os.listdir(INPUT_PATH)
    try:
        dirs.remove(".DS_Store")
    except:
        None

    all = 0
    empty = 0
    for d in dirs:
        files = os.listdir(INPUT_PATH + d)
        try:
            files.remove(".DS_Store")
        except:
            None

        if len(files) == 0:
            print(d)
            empty += 1
            os.system("rm -rf " + INPUT_PATH + d)

        all += 1

    print("all: " + str(all))
    print("empty: " + str(empty))

if __name__ == "__main__":
    check()
