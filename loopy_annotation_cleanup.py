# Author - Hemal Naik
# The file is supposed to clean up the annotation provided by loopy

import pandas as pd
import glob
import os
import numpy as np
import cv2 as cv
import logging
from pandas.io.sas.sas7bdat import _column


def copyContent(annotationData,revisedAnnotation,video,trainRatio):

    revisedAnnotation.loc[:,"img_id"] = annotationData.loc[:,"index"]
    revisedAnnotation["x" ] = annotationData["x0"]
    revisedAnnotation["y"] = annotationData["y0"]
    revisedAnnotation["train_status"] = 1

    # Compute the distribution required for the training and test samples
    noOfTrainingSamples = int( trainRatio * revisedAnnotation.shape[0]) # Compute training sample for distribution
    noOfEvalSamples = revisedAnnotation.shape[0] - noOfTrainingSamples # compute the eval set
    # Default training_status is 1 for combineData.loc[:, "training_status"] so we reassign some to 0
    revisedAnnotation.loc[0:noOfEvalSamples,"train_status"] = 0
    np.random.shuffle(revisedAnnotation["train_status"]) # Shuffle the last column to get random combination of test

    # Name of video
    baseDirectory = os.path.dirname(video)
    videoName = os.path.splitext(os.path.basename(video))[0]
    imageDir = os.path.join(baseDirectory , videoName)
    if not os.path.exists(imageDir):
        os.mkdir(imageDir)

    cap = cv.VideoCapture(video)

    # We loop through all the columns and pick the relevant frames, to store the csv file relevant to that
    for i in range(revisedAnnotation.shape[0]):
        revisedAnnotation.loc[i,"width"] =  float(annotationData.loc[i,"x1"]) - float(annotationData.loc[i,"x0"])
        revisedAnnotation.loc[i, "height"] = float(annotationData.loc[i, "y2"]) - float(annotationData.loc[i, "y1"])
        frameNo = int((annotationData.loc[i, "frame_count"]))
        cap.set(cv.CAP_PROP_POS_FRAMES, frameNo)
        imageName = videoName + '_' + str(frameNo) + '.jpg'
        imagePath = os.path.join(imageDir, imageName)
        ret, frame = cap.read()

        if ret is True:
            h, w, c = frame.shape
            revisedAnnotation.loc[i, "img_width"] = w
            revisedAnnotation.loc[i, "img_height"] = h
            revisedAnnotation.loc[i, "Channel"] = c
            # Saving hierarchial names --> videoName/imageName --> Path X:\xx\xx\videoName\images_frameNo
            revisedAnnotation.loc[i, "img_name"] = videoName + '/' +imageName

            # Saving the images for the training
            if SAVE_JPEG_IMG: # Check to avoid repeated generation of files
                cv.imwrite(imagePath, frame)

            if SHOW_IMAGES: # Check if the display image option is enabled
                cv.imshow("TestImage", frame)

    # Release the video footage
    cap.release()
    cv.destroyAllWindows()

def createDataBase(annotation, video, trainRatio):
    print('Creating Database')

    # Read dataframe for the annotation file and c
    annotationFile = open(annotation, 'r')
    annotationData = pd.read_csv(annotationFile, sep=",", header=None)
    annotationData.columns = ["index","annotation_id","class","date","frame_count","frame_number","frame_timestamp","name","oid","type","video_id","x","x0","x1","x2","x3","xform","y","y0","y1","y2","y3"]
    annotationData = annotationData.loc[1:,:]
    annotationData.reset_index(inplace = True) # Reset of index is important, otherwise the first row element starts from 1, row 0 is deleted for good unless reset.

    revisedAnnotationColumnFormat = ["img_id","x","y","width","height","img_name","img_width","img_height","Channel","train_status"]
    #revisedAnnotation = pd.DataFrame( np.zeros((annotationData.shape[0],len(revisedAnnotationColumnFormat))),
                                      # columns = revisedAnnotationColumnFormat )
    revisedAnnotation = pd.DataFrame(columns = revisedAnnotationColumnFormat )
    copyContent(annotationData,revisedAnnotation,video,trainRatio)
    return revisedAnnotation


####
SAVE_JPEG_IMG = False
SHOW_IMAGES = False
DATA_FORMAT = ["img_id","x","y","width","height","img_name","img_width","img_height","Channel","train_status"]
DATASET_PATH = 'D:/BirdTrackingProject/MPI_Dataset'
VIDEO_PATH_EXT = 'videos'
ANNOTATION_PATH_EXT = 'annotations'
IMG_DIR = 'images'
ANNOTATION_DIR = 'annotation_converted'

# The function is used to initiate the function when it is called directly from console
def main(DATASET_PATH):

    # Create directory to store the converted dataset
    images_dir = os.path.join(DATASET_PATH, IMG_DIR)
    annotation_conv_dir = os.path.join(DATASET_PATH, ANNOTATION_DIR)

    if not os.path.exists(images_dir):
        logging.info("Creating directory to store images")
        os.mkdir(images_dir)

    if not os.path.exists(annotation_conv_dir):
        logging.info("Creating director to store annotations")
        os.mkdir(annotation_conv_dir)

    # Derive video and annotation path from the given directory path
    video_dir =  os.path.join( DATASET_PATH , VIDEO_PATH_EXT )
    annotation_dir = os.path.join( DATASET_PATH , ANNOTATION_PATH_EXT )

    # Save the combine dataset file in the annotation folder
    combinedDataFileName = os.path.join(annotation_dir,"combine_dataset.csv")


    # Grab name of the files given in the header folder
    videoFiles = glob.glob( os.path.join( video_dir , '*.mp4')  )
    annotationFiles = glob.glob( os.path.join(annotation_dir , '*.csv' ) )


    # Creating division between training and test set
    trainRatio = 0.6

    if len(videoFiles) != len(annotationFiles):
        raise ('Error!! Video file and Annotations files do not match')

    # Meta file for storing data
    combineData = pd.DataFrame()

    for annotation in annotationFiles:
        annotationFileName = os.path.basename(annotation)
        annotationFileNameWoExt = os.path.splitext(annotationFileName)[0]
        for video in videoFiles:
            videoFileName = os.path.basename(video)
            videoFileNameWoExt = os.path.splitext(videoFileName)[0]
            if videoFileNameWoExt in annotationFileNameWoExt:
                combinedAnnotation = createDataBase(annotation,video, trainRatio) # Send both file names
                annotationFileName = DATASET_PATH + '/' + videoFileNameWoExt + '.csv'
                combinedAnnotation.to_csv( annotationFileName, index = False )
                combineData = combineData.append(combinedAnnotation)

    # Change the first column to have proper ID assigned to images
    combineData.loc[:,"img_id"] = list(range(combineData.shape[0]))


    ## Compute the distribution required for the training and test samples --> The method has been moved up to individual
    ## level. Now the status is copied directly when the file is created
    ## ------------- Uncomment the code to do global training/eval dataset ----------- #############

    #noOfTrainingSamples = int( trainRatio * combineData.shape[0]) # Compute training sample for distribution
    #noOfEvalSamples = combineData.shape[0] - noOfTrainingSamples # compute the eval set
    ## Default training_status is 1 for combineData.loc[:, "training_status"] so we reassign some to 0
    #combineData.loc[0:noOfEvalSamples,"training_status"] = 0
    #np.random.shuffle(combineData.loc[:,"training_status"]) # Shuffle the last column to get random combination of test

    combineData.to_csv(combinedDataFileName, index=False)


# The function is called only if the file is called on its own, this enables using files independently
if __name__ == '__main__':
    print('Called from independent console')
    DATASET_PATH = 'D:/BirdTrackingProject/MPI_Dataset'
    main(DATASET_PATH)
    print('Conversion Process Ends!!')
else:
    print('Called by third party!!')
# Second read the folder for the csv information
