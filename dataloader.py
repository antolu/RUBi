import os
import pickle
import re
import json
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import torchvision.transforms as transforms

from nltk.tokenize import word_tokenize
from PIL import Image


class DataLoaderVQA(data.Dataset):
    def __init__(self,
                 args_dict,
                 transform = transforms.Compose([
                                    transforms.Resize(256),  # rescale the image keeping the original aspect ratio
                                    transforms.CenterCrop(256),  # we get only the center of that rescaled
                                    transforms.RandomCrop(224),  # random crop within the center crop (data augmentation)
                                    transforms.RandomHorizontalFlip(),  # random horizontal flip (data augmentation)
                                    transforms.ToTensor()  # to pytorch tensor
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406, ], Coco need it? 
                                    #                     std=[0.229, 0.224, 0.225])
                                ]),
                 MAX_WORDS_QUESTION = 20,
                 dir_data='data',
                 coco_train_path="data/train2014",
                 coco_val_path="data/val2014",
                 trainval_features_path="data/trainval_36",
                 test_features_path="data/test2014_36"
                ):
        """
        Args:
            args_dic: parameters dict
            transform: data transformation for image
            MAX_WORDS_QUESTION: limit of words that a question can have
        """

        self.args_dict = args_dict
        self.transform = transform
        self.MAX_WORDS_QUESTION = MAX_WORDS_QUESTION
        self.dir_data = dir_data
        self.coco_train_path = coco_train_path
        self.coco_val_path = coco_val_path
        self.trainval_features_path = trainval_features_path
        self.test_features_path = test_features_path
        self.dataset = args_dict.dataset  # vqacp_v2 | vqa_v2
        self.vocab = []  # vocab related with the dataset questions
        self.question_vecs = {}
        
        if args_dict.answer_type == 'all':
            self.answer_type = ['yes/no', 'number', 'other']
        else:
            self.answer_type = [args_dict.answer_type]
        

        # only use the top 3000 answers
        df_annot = pd.read_json(os.path.join(self.dir_data, 'vqacp_v2', 'vqacp_v2_train_annotations.json'))
        top_3000_answer = list(df_annot['multiple_choice_answer'].value_counts().index)[:3000]
        self.answer2idx = {}
        for idx, answer in enumerate(top_3000_answer):
            self.answer2idx[answer] = idx

        # choose train or test dataset
        if self.dataset == 'vqa-v2-cp':
            if args_dict.train:
                df_annot = pd.read_json(os.path.join(self.dir_data, 'vqacp_v2', 'vqacp_v2_train_annotations.json'))
                df_quest = pd.read_json(os.path.join(self.dir_data, 'vqacp_v2', 'vqacp_v2_train_questions.json'))

            elif args_dict.test:
                df_annot = pd.read_json(os.path.join(self.dir_data, 'vqacp_v2', 'vqacp_v2_test_annotations.json'))
                df_quest = pd.read_json(os.path.join(self.dir_data, 'vqacp_v2', 'vqacp_v2_test_questions.json'))
                
            df = pd.merge(df_annot[['question_type', 'multiple_choice_answer',
                                    'image_id', 'answer_type', 'question_id']],
                          df_quest[['coco_split', 'question', 'question_id']], on='question_id')

        elif self.dataset == 'vqa-v2':
            if args_dict.train:
                df_annot = json.load(
                    open(os.path.join(self.dir_data, 'vqa_v2', 'v2_mscoco_train2014_annotations.json')))
                df_quest = json.load(
                    open(os.path.join(self.dir_data, 'vqa_v2', 'v2_OpenEnded_mscoco_train2014_questions.json')))

            elif args_dict.test:
                df_annot = json.load(open(os.path.join(self.dir_data, 'vqa_v2', 'v2_mscoco_val2014_annotations.json')))
                df_quest = json.load(
                    open(os.path.join(self.dir_data, 'vqa_v2', 'v2_OpenEnded_mscoco_val2014_questions.json')))

            elif args_dict.test_dev:
                df_annot = json.load(open(os.path.join(self.dir_data, 'vqa_v2', 'v2_mscoco_val2014_annotations.json')))
                df_quest = json.load(
                    open(os.path.join(self.dir_data, 'vqa_v2', 'v2_OpenEnded_mscoco_test-dev2015_questions.json')))

            df_annot = pd.DataFrame(df_annot['annotations'])
            df_quest = pd.DataFrame(df_quest['questions'])

            df = pd.merge(df_annot[['question_type', 'multiple_choice_answer',
                                    'image_id', 'answer_type', 'question_id']],
                          df_quest[['question', 'question_id']], on='question_id')

        df = df[(df['multiple_choice_answer'].isin(top_3000_answer)) &
                (df['answer_type'].isin(self.answer_type))]

        self.images_path = df.apply(lambda x: self.get_img_path(x), axis=1)
        self.questions = df['question'].apply(self.preprocess_sentence) 
        self.vocab = self.get_vocab()
        self.vocab2id = self.get_vocab2id()
        self.answers = df['multiple_choice_answer']#.apply(self.preprocess_sentence)
        self.img_embeddings_path = df['image_id'].apply(lambda x: self.get_visual_features_path(x))

    def __len__(self):
        return len(self.questions)

    def get_vocab2id(self):
        """
        Parameters
        ----------

        Returns
        -------
        a dict that links each word from vocab to a unique id
        """
        vocab2id = {}
        for pos, word in enumerate(self.vocab):
            vocab2id[word] = pos
        return vocab2id
        
    def get_vocab(self):
        """
        Parameters
        ----------

        Returns
        -------
        a set containing all the words present in the questions
        """
        vocab = set()
        def toSet(vocab, x):
            vocab = vocab.update(set(x))
        self.questions.apply(word_tokenize).apply(lambda x: toSet(vocab, x)) 
        
        return vocab

    def get_img_path(self, row):
        """
        Returns the right path based on the question_id and the dataset used

        Parameters
        ----------
        row : row of DataFrame
            row with all atributes of que dataset

        Returns
        -------
        path to the corresponding image
        files have the name on the folowing format:
        COCO_val2014_000000290981.jpg
        COCO_train2014_000000291894.jpg
        """
        
        img_id = str(row['image_id'])
        full_number = ''.join((12 - len(img_id)) * ['0']) + img_id
        
        # the dataset has a different split compared to the original coco
        if self.dataset == 'vqa-v2-cp':  
            img_folder = row['coco_split']
            
        elif self.dataset =='vqa-v2':
            if self.args_dict.train:
                img_folder = 'train2014'
            elif self.args_dict.test:
                img_folder = 'val2014'
        
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

        out["features"] = torch.tensor(features["arr_0"])
        out["boxes"] = torch.tensor(features["arr_1"])

        return out

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
        prep_quest = re.sub(r'[^\w\s]','', sentence)  # remove punctuation
        prep_quest = prep_quest.lower()  # lower case

        return prep_quest

    def __getitem__(self, index):
        """Returns data sample as a dict with keys: img_embed, question, answer"""

        # Load image & apply transformation
        if self.args_dict.baseline == 'san' or self.args_dict.test:
            image = Image.open(self.images_path.iloc[index]).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        else:
            image = torch.Tensor()
        
        # Image embedding
        img_embedding = self.get_visual_features(self.img_embeddings_path.iloc[index])['features']

        # Question
        question = self.questions.iloc[index]
        if index not in self.question_vecs or not self.args_dict.store_questions:
            question_words = word_tokenize(question)
            quest_vocab_vec = torch.zeros(self.MAX_WORDS_QUESTION)  # vector used for feeding the quest encoder
            for i, word in enumerate(question_words):
                if i >= self.MAX_WORDS_QUESTION:
                    break
                quest_vocab_vec[i] = self.vocab2id[word]

            self.question_vecs[index] = quest_vocab_vec
        else:
            quest_vocab_vec = self.question_vecs[index]

        # Answer
        answer = self.answers.iloc[index]
        answer_one_hot = torch.zeros(3000)  # vector used for the loss
        # answer_one_hot[self.answer2idx[answer]] = 1.0
        
        # item returned from the dataset
        item = {
            'img_embed': img_embedding,
            'image': image,
            # 'question': question,  # in natural language
            'quest_vocab_vec': quest_vocab_vec.type(torch.LongTensor),
            'quest_size': torch.Tensor([len(word_tokenize(question))]).type(torch.LongTensor),  # size of each question
            # 'answer': answer,  # in natural language
            #'answer_one_hot': answer_one_hot,
            'idx_answer': torch.Tensor([self.answer2idx[answer]]).type(torch.LongTensor)  # class indices
        }

        return item



