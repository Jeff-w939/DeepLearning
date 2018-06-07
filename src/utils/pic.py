import glob

from PIL import Image
import numpy
import pickle
import os


def pic_to_numpy(file):
    img = Image.open(file)
    # img = img.convert("RGB")
    img = img.convert("L")
    arr = numpy.array(img)
    return arr


def total_to_numpy_pickle(root_directory):
    directorys = [dir_ for dir_ in glob.glob(root_directory + "/*")
                  if os.path.isdir(dir_)]
    directorys.sort(key=os.path.basename)
    result = []
    types = []
    for i, dir_ in enumerate(directorys):
        files = glob.glob(dir_ + "/*.jpg")
        for file in files:
            print(file)
            types.append(i)
            result.append(pic_to_numpy(file))

    out_file = os.path.join(root_directory, "result-hui.pickle")
    with open(out_file, "wb") as f:
        pickle.dump((numpy.array(result), types), f)


def all_file(directory):
    files = glob.glob(directory + "/*")
    result = []
    for file in files:
        if os.path.isdir(file):
            result += all_file(file)
        else:
            result.append(file)
    return result


if __name__ == '__main__':
    total_to_numpy_pickle("/media/computer/可移动软盘/cloth/")
