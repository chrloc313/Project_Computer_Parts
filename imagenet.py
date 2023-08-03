#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

def process_images(input, output, network="googlenet", topK=1):

     net = imageNet(model="models/parts/resnet18.onnx", labels="models/parts/labels.txt", 
                    input_blob="input_0", output_blob="output_0")
     input = videoSource(input, argv=sys.argv)
     output = videoOutput(output, argv=sys.argv)
     font = cudaFont()

     img = input.Capture()
     
     if img is None:
          return

    # classify the image and get the topK predictions
    # if you only want the top class, you can simply run:
    #   class_id, confidence = net.Classify(img)
    
     predictions = net.Classify(img, topK=topK)
     classLabels = []

    # draw predicted class labels
     for n, (classID, confidence) in enumerate(predictions):
        classLabel = net.GetClassLabel(classID)
        confidence *= 100.0

        if classID == 0:
             classLabel = "cpu"
        elif classID == 1:
             classLabel = "gpu"
        elif classID == 2:
             classLabel = "hdd"
        elif classID == 3:
             classLabel = "mobo"
        elif classID == 4:
             classLabel = "ram"
        elif classID == 5:
             classLabel = "ssd-m2"
        elif classID == 6:
             classLabel = "ssd-sata"
             
        print(f"imagenet:  {confidence:05.2f}% class #{classID} ({classLabel})")
             
        font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}", 
                         x=5, y=5 + n * (font.GetSize() + 5),
                         color=font.White, background=font.Gray40)
        
        classLabels.append(classLabel)
                         
    # render the image
     output.Render(img)

    # update the title bar
     output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
     net.PrintProfilerTimes()
     print(classLabels)
     return classLabels

    # exit on input/output EOS