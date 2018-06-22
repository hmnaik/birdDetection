# Author - Hemal Naik
# The file is supposed to clean up the annotation provided by loopy

import pandas as pd
import glob
import os
import numpy as np
import cv2 as cv

def copyContent(annotationData,revisedAnnotation,video):

    revisedAnnotation.loc[:,"img_id"] = annotationData.loc[:,"index"]
    revisedAnnotation["x" ] = annotationData["x0"]
    revisedAnnotation["y"] = annotationData["y0"]

    # Name of video
    baseDirectory = os.path.dirname(video)
    videoName = os.path.splitext(os.path.basename(video))[0]
    imageDir = os.path.join(baseDirectory , videoName)
    if not os.path.exists(imageDir):
        os.mkdir(imageDir)

    cap = cv.VideoCapture(video)

    print(cap.isOpened())
    counter = 0

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
            cv.imwrite(imagePath, frame)
            cv.imshow("TestImage", frame)

    # Release the video footage
    cap.release()
    cv.destroyAllWindows()

def createDataBase(annotation,video):
    print('Creating Database')

    # Read dataframe for the annotation file and c
    annotationFile = open(annotation, 'r')
    annotationData = pd.read_csv(annotationFile, sep=",", header=None)
    annotationData.columns = ["index","annotation_id","class","date","frame_count","frame_number","frame_timestamp","name","oid","type","video_id","x","x0","x1","x2","x3","xform","y","y0","y1","y2","y3"]
    annotationData = annotationData.loc[1:,:]
    annotationData.reset_index(inplace = True)

    revisedAnnotationColumnFormat = ["img_id","x","y","width","height","img_name","img_width","img_height","Channel","train_status"]
    revisedAnnotation = pd.DataFrame( np.zeros((annotationData.shape[0],10)),
                                      columns = revisedAnnotationColumnFormat )

    copyContent(annotationData,revisedAnnotation,video)
    return revisedAnnotation


####
DATASET_PATH = 'D:/BirdTrackingProject/MPI_Dataset'
video_dir =  DATASET_PATH + '/videoDataset/'
annotation_dir = DATASET_PATH + '/Annotations/'

videoFiles = glob.glob(video_dir + '*.mp4')
annotationFiles = glob.glob(annotation_dir + '*.csv')

for files in videoFiles:
    print(files)

if len(videoFiles) != len(annotationFiles):
    raise ('Error!! Video file and Annotations files do not match')



for annotation in annotationFiles:
    annotationFileName = os.path.basename(annotation)
    annotationFileNameWoExt = os.path.splitext(annotationFileName)[0]
    for video in videoFiles:
        videoFileName = os.path.basename(video)
        videoFileNameWoExt = os.path.splitext(videoFileName)[0]
        if videoFileNameWoExt in annotationFileNameWoExt:
            combinedAnnotation = createDataBase(annotation,video) # Send both file names
            annotationFileName = DATASET_PATH + '/' + videoFileNameWoExt + '.csv'
            combinedAnnotation.to_csv( annotationFileName )


# First read the folder for the video information


# Second read the folder for the csv information
