
import os
import shutil
import random

#save 60% images as train, save 20% images as test, and remaining 20% images as val

def chooseSet(dir):
    classList = os.listdir(dir)
    root = "/root/data/"
    trainFile = open(root + "train.txt", "w")
    testFile = open(root+"test.txt", "w")
    valFile = open(root+"val.txt", "w")

    for d in classList:
        
        #read image in dir
        path = root + "images/" + d
        imgList = os.listdir(path)
        total = len(imgList)
        train = int(0.8 * total)
        test = (total-train)/2
        val = total-train-test

        random.seed(101)
        random.shuffle(imgList)

        trainImgs = imgList[:train]
        testImgs = imgList[train: (train+test)]
        valImgs = imgList[(train+test):]
   
        #write files
        writeFile(trainFile, trainImgs, d)
        writeFile(testFile, testImgs, d)
        writeFile(valFile, valImgs, d)      
        
    #create train, test, val list


    trainFile.close()
    testFile.close()
    valFile.close()

    return


def writeFile(file, imgs, dir):
    for i in imgs:
        imgName = dir + "/" + i
        row = imgName + " " + str(int(dir[1:])-1) + "\n"
        file.write(row)


if __name__ == "__main__":
    dir = "/root/data/images"
    chooseSet(dir)

