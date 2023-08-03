import argparse
import sys
import os
import shutil

from jetson_inference import imageNet
from imagenet import process_images
from jetson_utils import videoSource, videoOutput, Log

parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.",
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="path to data folder")
parser.add_argument("output", type=str, default="", nargs='?', help="path to result folder")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--topK",type=int, default=1, help="show the topKnumber of class predictions (default:1)")

try:
        args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

categories = os.listdir(args.input)
total = 0
error = 0

def check_output(catergory, labels):
    for label in labels:
         if catergory in label:
              return True
    return False

for category in categories:
    category_folder_path = os.path.join(args.input, category)
    result_category_folder_path = os.path.join(args.output, category)

    if os.path.exists(result_category_folder_path):
      shutil.rmtree(result_category_folder_path)
    os.makedirs(result_category_folder_path, exist_ok=True)

    images = os.listdir(category_folder_path)
    total += len(images)

    for image in images:
        image_path = os.path.join(category_folder_path, image)
        output_path: str = os.path.join(result_category_folder_path, "test_{}".format(image))

        labels = process_images(image_path, output_path, args.network, args.topK)

        if not check_output(category, labels):
             error += 1
            
print("Accuracy: ", (total-error)/total)
