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
    folderName = os.path.splitext(os.path.basename(video))[0]
    imageDir = os.path.join(baseDirectory , folderName)
    if not os.path.exists(imageDir):
        os.mkdir(imageDir)

    cap = cv.VideoCapture(video)

    print(cap.isOpened())
    counter = 0

    for i in range(revisedAnnotation.shape[0]):
        revisedAnnotation.loc[i,"width"] =  float(annotationData.loc[i,"x1"]) - float(annotationData.loc[i,"x0"])
        revisedAnnotation.loc[i, "height"] = float(annotationData.loc[i, "y1"]) - float(annotationData.loc[i, "y0"])
        frameNo = int((annotationData.loc[i, "y1"]))


def createDataBase(annotation,video):
    print('Creating Database')

    # Read dataframe for the annotation file and c
    annotationFile = open(annotation, 'r')
    annotationData = pd.read_csv(annotationFile, sep=",", header=None)
    annotationData.columns = ["index","annotation_id","class","date","frame_count","frame_number","frame_timestamp","name","oid","type","video_id","x","x0","x1","x2","x3","xform","y","y0","y1","y2","y3"]
    annotationData = annotationData.loc[1:,:]
    annotationData.reset_index(inplace = True)
    print(annotationData.head(3))

    revisedAnnotationColumnFormat = ["img_id","x","y","width","height","img_name","img_width","img_height","Channel","train_status"]
    revisedAnnotation = pd.DataFrame( np.zeros((annotationData.shape[0],10)),
                                      columns = revisedAnnotationColumnFormat )
    print(revisedAnnotation.head(3))

    print(annotationData.shape)
    print(revisedAnnotation.shape)

    copyContent(annotationData,revisedAnnotation,video)
    print(revisedAnnotation)


####
DATASET_PATH = 'D:/BirdTrackingProject/MPI_Dataset'
video_dir =  DATASET_PATH + '/videoDataset/'
annotation_dir = DATASET_PATH + '/Annotations/'

videoFiles = glob.glob(video_dir + '*')
annotationFiles = glob.glob(annotation_dir + '*')

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


# First read the folder for the video information


# Second read the folder for the csv information
