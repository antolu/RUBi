import os
import pickle
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image

class DataLoaderVQA(data.Dataset):
    def __init__(self, 
                 # args_dict,
                 set,
                 transform= transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224), 
                                                transforms.ToTensor()]),
                 dir_data='data',
                 coco_train_path="data/train2014",
                 coco_val_path="data/val2014",
                 trainval_features_path="data/trainval_36",
                 test_features_path="data/test2014_36"):
        """
        Args:
            args_dic: parameters dictionar
            set: 'train', 'val', 'test'
            transform: data transform
        """

        # self.args_dict = args_dict
        self.set = set
        self.transform = transform
        self.dir_data = dir_data
        self.coco_train_path = coco_train_path
        self.coco_val_path = coco_val_path
        self.trainval_features_path = trainval_features_path
        self.test_features_path = test_features_path


        df_annot = pd.read_json(os.path.join('data', 'vqacp_v2_train_annotations.json'))
        #top_3000_answer = set(list(df_annot['multiple_choice_answer'].value_counts(3000).index))
    
        if self.set == 'train':
            df_annot = pd.read_json(os.path.join(self.dir_data, 'vqacp_v2_train_annotations.json'))
            df_quest = pd.read_json(os.path.join(self.dir_data, 'vqacp_v2_train_questions.json'))

        elif self.set == 'test':
            df_annot = pd.read_json(os.path.join(self.dir_data, 'vqacp_v2_test_annotations.json'))
            df_quest = pd.read_json(os.path.join(self.dir_data, 'vqacp_v2_test_questions.json'))

        df = pd.merge(df_annot[['question_type', 'multiple_choice_answer',
                                'image_id', 'answer_type', 'question_id']]
                      , df_quest[['coco_split', 'question', 'question_id']], on='question_id')
        #df = df[df['answer_type'].isin(top_3000_answer)]

        self.transform = transform

        self.images_path = df.apply(lambda x: self.get_img_path(x), axis=1)
        self.questions = df['question']
        self.answers = df['answer_type']
        self.img_embeddings_path = df['image_id'].apply(lambda x: self.get_visual_features_path(x))

    def __len__(self):
        return len(self.images_path)
    
    def get_visual_features_path(self, image_id):
        return os.path.join(self.trainval_features_path, str(image_id))
        

    def get_visual_features(self, filepath):
        """
        Gets the visual features of the image of the corresponding image id

        Parameters
        ----------
        image_id : int
            The id of the image to fetch the features for

        Returns
        -------
        A dictionary containing all the features of the image.
        """

        with open(filepath + ".pkl", "rb") as handle:
            out = pickle.load(handle, encoding="ascii")

        features = np.load(filepath + ".npz")

        out["features"] = features["arr_0"]
        out["boxes"] = features["arr_1"]

        return out

    def get_img_path(self, row):
        """Returns the right path based on the question_id and coco_split"""
        img_id = str(row['image_id'])
        img_folder = row['coco_split']
        full_number = ''.join((12 - len(img_id)) * ['0']) + img_id
        return os.path.join(self.dir_data, img_folder, "COCO_" + img_folder + "_" + full_number + ".jpg")

    def __getitem__(self, index):
        """Returns data sample as a tuple [image, embedding], [question, answer]"""

        # Load image & apply transformation
        image = Image.open(self.images_path[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Image embedding
        img_embedding = self.get_visual_features(self.img_embeddings_path[index])

        # Question
        question = self.questions[index]

        # Answer
        answer = self.answers[index]

        return [image, img_embedding], [question, answer]