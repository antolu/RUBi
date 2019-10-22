#!/usr/bin/env python2

import argparse
import os
import pickle
import numpy as np
import base64
import csv
import sys

"""
This file splits a pretrained Faster R-CNN features archive .tsv into 
individual files for each COCO image id. the split files will be placed
in the same directory as the original .tsv file. 

The script has to be run with python 2, and for use with the shipped
Tensorflow/Caffe containers it is run simply as

```shell
./tools/parse_visual_features.py [path/to/tsvfile]
```
"""


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
csv.field_size_limit(sys.maxsize)

def split_tsv(infile):
    """
    Function to split the infile .tsv into individial files for each COCO
    image id, one pickled .pkl file containing the information of the image
    as a dict, and one .npz file containing the extracted features tensor
    as numpy array as well as the feature boxes.

    Parameters:
    -----------
    infile: str
        Path to the file to split
    """
    with open(infile, "r+b") as tsv_in_file:
        filepath = os.path.split(infile)[0]

        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            out = {}
            out['image_id'] = int(item['image_id'])
            out['image_h'] = int(item['image_h'])
            out['image_w'] = int(item['image_w'])
            out['num_boxes'] = int(item['num_boxes'])
            
            boxes = np.frombuffer(base64.decodestring(item["boxes"]), dtype=np.float32).reshape((out['num_boxes'], -1))
            features = np.frombuffer(base64.decodestring(item["features"]), dtype=np.float32).reshape((out['num_boxes'], -1))

            np.savez(os.path.join(filepath, str(out['image_id'])), features, boxes)

            with open(os.path.join(filepath, str(out["image_id"]) + ".pkl"), "w") as handle:
                pickle.dump(out, handle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("tsvfile", help="The path to the TSV file to split")

    args = parser.parse_args()

    split_tsv(args.tsvfile)
