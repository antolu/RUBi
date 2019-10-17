import os
import pickle
import string
import json
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image

class DataLoaderVQA(data.Dataset):
    def __init__(self, 
                 args_dict,
                 dir_data='data',
                 coco_train_path="data/train2014",
                 coco_val_path="data/val2014",
                 trainval_features_path="data/trainval_36",
                 test_features_path="data/test2014_36"):
        """
        Args:
            args_dic: parameters dict
            set: 'train', 'val', 'test'
            transform: data transform
        """

        self.args_dict = args_dict
        self.dir_data = dir_data
        self.coco_train_path = coco_train_path
        self.coco_val_path = coco_val_path
        self.trainval_features_path = trainval_features_path
        self.test_features_path = test_features_path
        self.answer_type = args_dict.answer_type # list : ['yes/no', 'number', 'other']
        self.dataset = args_dict.dataset  # vqacp_v2 | vqa_v2

        # only use the top 3000 answers
        df_annot = pd.read_json(os.path.join(self.dir_data, 'vqacp_v2', 'vqacp_v2_train_annotations.json'))
        top_3000_answer = list(df_annot['multiple_choice_answer'].value_counts().index)[:3000]
        self.answer2idx = {}
        for idx, answer in enumerate(top_3000_answer):
            self.answer2idx[answer] = idx
    
        # choose train or test dataset
        if self.dataset == 'vqacp_v2':
            if args_dict.train:
                df_annot = pd.read_json(os.path.join(self.dir_data, 'vqacp_v2', 'vqacp_v2_train_annotations.json'))
                df_quest = pd.read_json(os.path.join(self.dir_data, 'vqacp_v2', 'vqacp_v2_train_questions.json'))

            elif args_dict.test:
                df_annot = pd.read_json(os.path.join(self.dir_data, 'vqacp_v2', 'vqacp_v2_test_annotations.json'))
                df_quest = pd.read_json(os.path.join(self.dir_data, 'vqacp_v2', 'vqacp_v2_test_questions.json'))
                
        elif self.dataset == 'vqa_v2':
            if args_dict.train:
                df_annot = json.load(open(os.path.join(self.dir_data, 'vqa_v2', 'v2_mscoco_train2014_annotations.json')))
                df_quest = json.load(open(os.path.join(self.dir_data, 'vqa_v2', 'v2_OpenEnded_mscoco_train2014_questions.json')))

            elif args_dict.test:
                df_annot = json.load(open(os.path.join(self.dir_data, 'vqa_v2', 'v2_mscoco_val2014_annotations.json')))
                df_quest = json.load(open(os.path.join(self.dir_data, 'vqa_v2', 'v2_OpenEnded_mscoco_val2014_questions.json')))

            elif args_dict.test_dev:
                df_annot = json.load(open(os.path.join(self.dir_data, 'vqa_v2', 'v2_mscoco_val2014_annotations.json')))
                df_quest = json.load(open(os.path.join(self.dir_data, 'vqa_v2', 'v2_OpenEnded_mscoco_test-dev2015_questions.json')))
                
            df_annot = pd.DataFrame(df_annot['annotations'])
            df_quest = pd.DataFrame(df_quest['questions'])

        df = pd.merge(df_annot[['question_type', 'multiple_choice_answer',
                                'image_id', 'answer_type', 'question_id']], 
				df_quest[['coco_split', 'question', 'question_id']], on='question_id')
        
        df = df[(df['multiple_choice_answer'].isin(top_3000_answer)) & 
                (df['answer_type'].isin(self.answer_type))]
        
	#self.images_path = df.apply(lambda x: self.get_img_path(x), axis=1)
        self.questions = df['question']
        self.answers = df['answer_type']
        self.img_embeddings_path = df['image_id'].apply(lambda x: self.get_visual_features_path(x))

    def __len__(self):
        return len(self.questions)
    
    def get_img_path(self, row):
        """
        Returns the right path based on the question_id and coco_split

        Parameters
        ----------
        row : row of DataFrame
            row with all atributes of que dataset

        Returns
        -------
        path to the corresponding image
        """
        img_id = str(row['image_id'])
        img_folder = row['coco_split']
        full_number = ''.join((12 - len(img_id)) * ['0']) + img_id
        return os.path.join(self.dir_data, img_folder, "COCO_" + img_folder + "_" + full_number + ".jpg")
    
    def get_visual_features_path(self, image_id):
        """
        Returns the right path of the features of the image based on the question_id

        Parameters
        ----------
        image_id : string
            The id of the corresponding image

        Returns
        -------
        path to the corresponding feature of the image
        """
        return os.path.join(self.trainval_features_path, str(image_id))
        

    def get_visual_features(self, filepath):
        """
        Returns the visual features of the image of the corresponding filepath

        Parameters
        ----------
        filepath : string
            The path of the respective image

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
        """
        Returns the right path based on the question_id and coco_split

        Parameters
        ----------
        row : row of DataFrame
            row with all atributes of que dataset

        Returns
        -------
        path to the corresponding image
        """
        img_id = str(row['image_id'])
        img_folder = row['coco_split']
        full_number = ''.join((12 - len(img_id)) * ['0']) + img_id
        return os.path.join(self.dir_data, img_folder, "COCO_" + img_folder + "_" + full_number + ".jpg")
    
    def preprocess_sentence(self, sentence):
        """
        returns the preprocessed question

        Parameters
        ----------
        question : string
            question to be preprocessed

        Returns
        -------
        question preprocessed
        """
        prep_quest = sentence.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        prep_quest = sentence.lower() # lower case
    
        return prep_quest

    def __getitem__(self, index):
        """Returns data sample as a dict with keys: img_embed, question, answer"""

        # Load image & apply transformation (we only use the pre-calc embbeding)
        # image = Image.open(self.images_path[index]).convert('RGB')
        # if self.transform is not None:
        #    image = self.transform(image)

        # Image embedding
        img_embedding = self.get_visual_features(self.img_embeddings_path.iloc[index])

        # Question
        question = self.preprocess_sentence(self.questions.iloc[index])

        # Answer
        answer = self.preprocess_sentence(self.answers.iloc[index])
        answer_one_hot = torch.zeros(3000)
        answer_one_hot[self.answer2idx[answer]] = 1.0
        item = {
            'img_embed': img_embedding,
            #'image': image,
            'question': question,
            'answer': answer,
            'answer_one_hot': answer_one_hot
        }

        return item



