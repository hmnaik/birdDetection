# The file is intended to create one data set from multiple annotations for the CUB Dataset file

import os
import pandas as pd
import cv2 as cv
import numpy as np



def convert(txtFile, bBoxFile, trainTest):
    print('Open Text File For Conversion')

    file = open(txtFile, 'r')
    data = pd.read_csv(txtFile, sep=" ", header=None)
    data.columns = ["img_id", "img_name"]


    # Create a new dataFrame storing the image size information
    imageDims = pd.DataFrame(np.zeros((len(data.index), 3)), index=list(range(len(data.index))),
                             columns=['img_width', 'img_height', 'Channel'])

    # get name of image and get the image properties
    for i in range(len(data.index)):
        # name = data.loc[:,i+1]
        img_name = data.iloc[i, 1]
        img = cv.imread(
            "/media/hemal/37960676-e41c-4a31-ab6a-e01a2521f68b/hemal/BirdDataset/CUB_200_2011/images/" + img_name)
        h, w, c = img.shape
        imageDims.iloc[i, :] = [w, h, c]

    # Read the file for the bounding box information
    bBox = open(bBoxFile, 'r')
    boxData = pd.read_csv(bBoxFile, sep=' ', header= None)
    boxData.columns = ["img_id" , "x", "y","width", "height"]

    # Read the file for train and test division information
    trainEvalFile = open(trainTest, 'r')
    trainEvalData = pd.read_csv(trainEvalFile, sep = " ", header = None)
    trainEvalData.columns = ["img_id" , "train_status"]

    # Appending the columns of the dataset, to have one big csv with all required information
    result = pd.concat([boxData , data.loc[:,"img_name"], imageDims, trainEvalData.loc[:,"train_status"]], axis = 1)
    result.to_csv('random.csv', index = None)


# The function is used to initiate the function when it is called directly from console
def main():
    print('Execution of main file')
    imageFile = ('/media/hemal/37960676-e41c-4a31-ab6a-e01a2521f68b/hemal/BirdDataset/CUB_200_2011/images.txt')
    bBoxFile = ('/media/hemal/37960676-e41c-4a31-ab6a-e01a2521f68b/hemal/BirdDataset/CUB_200_2011/bounding_boxes.txt')
    trainTest = ('/media/hemal/37960676-e41c-4a31-ab6a-e01a2521f68b/hemal/BirdDataset/CUB_200_2011/train_test_split.txt')
    convert(imageFile, bBoxFile, trainTest)


# The function is called only if the file is called on its own, this enables using files independently
if __name__ == '__main__':
    print('Called from independent console')
    main()


print('Conversion Process Ends!!')



