import argparse
import os
import pickle
import numpy as np
import base64
import csv
import sys


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
csv.field_size_limit(sys.maxsize)

def split_tsv(infile):
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

    parser.add_argument("--tsvfile", required=True, help="The path to the TSV file to split")

    args = parser.parse_args()

    split_tsv(args.tsvfile)
