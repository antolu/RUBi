import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataLoader:
    def __init__(self,
                 args_dict,
                 data_dir="data",
                 coco_train_path="data/train2014",
                 coco_val_path="data/val2014",
                 trainval_features_path="data/trainval_36",
                 test_features_path="data/test2014_36"):

        self.args_dict = args_dict
        self.coco_train_path = coco_train_path
        self.coco_val_path = coco_val_path
        self.trainval_features_path = trainval_features_path
        self.test_features_path = test_features_path
        
    def get_dataset(self):
        """
        This function return a tf.data.Dataset in the format : [img_embadding, question, answer]
        """

        df_annot = pd.read_json(os.path.join(self.data_dir, 'vqacp_v2_train_annotations.json'))
        top_3000_answer = set(list(df_annot['multiple_choice_answer'].value_counts(3000).index))
    
        if self.args_dict == "train":
            
            df_quest = pd.read_json(os.path.join(self.args_dict.dir_data, 'vqacp_v2_train_questions.json'))
            df = pd.merge(df_annot[['question_type', 'multiple_choice_answer',
                                    'image_id', 'answer_type', 'question_id']]
                          , df_quest[['coco_split', 'question', 'question_id']], on='question_id')
            
            
            df = df[(df['coco_split'] == 'train2014') & (df['multiple_choice_answer'].isin(top_3000_answer))]
            d_question = tf.data.Dataset.from_tensor_slices(df['question'])
            d_answer = tf.data.Dataset.from_tensor_slices(df['multiple_choice_answer'])
            
        elif self.args_dict.mode == 'val':
            df_quest = pd.read_json(os.path.join(, 'vqacp_v2_train_questions.json'))
            df = pd.merge(df_annot[['question_type', 'multiple_choice_answer',
                                    'image_id', 'answer_type', 'question_id']]
                          , df_quest[['coco_split', 'question', 'question_id']], on='question_id')
            
            df = df[(df['coco_split'] == 'val2014') & (df['multiple_choice_answer'].isin(top_3000_answer))]
            d_question = tf.data.Dataset.fom_tensor_slices(df['question'])
            d_answer = tf.data.Dataset.from_tensor_slices(df['multiple_choice_answer'])
            
        elif self.args_dict.mode == 'test':  # TODO: understand the test dataset. Why there is a train2014 and val2014 ?
            df_annot = pd.read_json(os.path.join(self.args_dict.dir_data, 'vqacp_v2_test_annotations.json'))
            df_quest = pd.read_json(os.path.join(self.args_dict.dir_data, 'vqacp_v2_test_questions.json'))
            df = pd.merge(df_annot[['question_type', 'multiple_choice_answer',
                                    'image_id', 'answer_type', 'question_id']]
                          , df_quest[['coco_split', 'question', 'question_id']], on='question_id')
            
            df = df[(df['coco_split'] == 'val2014') & (df['multiple_choice_answer'].isin(top_3000_answer))]
            d_question = tf.data.Dataset.from_tensor_slices(df[df['coco_split'] == 'val2014']['question'])
            d_answer = tf.data.Dataset.from_tensor_slices(df[df['coco_split'] == 'val2014']['multiple_choice_answer'])
            
        else:
            assert "Invalid set parameter. Choose between: train, val, test"
            
            imgs_path = tf.data.Dataset.from_tensor_slices(df['image_id'].apply(lambda x: self.question_id2path(x, self.args_dict)))
            
            # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
            d_img = imgs_path.map(lambda x: self.process_path(x, self.args_dict), num_parallel_calls=AUTOTUNE)
            
            tf_dataset = tf.data.Dataset.zip((d_img, d_question, d_answer)).shuffle(500).repeat().batch(self.args_dict.batch_size)
            
        return tf_dataset

    def question_id2path(self, img_id):
        img_id = str(img_id)
        full_number = ''.join((12 - len(img_id)) * ['0']) + img_id
        if self.args_dict.mode == 'train' or 'val':
            return os.path.join(self.args_dict.dir_data, "train2014/COCO_train2014_" + full_number + ".jpg")
        else:
            return os.path.join(self.args_dict.dir_data, "val2014/COCO_val2014_" + full_number + ".jpg")
        
    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.args_dict.IMG_WIDTH, self.args_dict.IMG_HEIGHT])

    def process_path(self, file_path):
        # load the raw data from module import symbol
        the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img, self.args_dict)
        return img

    def get_visual_features(self, image_id):
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
        filepath = os.path.join(self.trainval_features_path, str(image_id))

        with open(filepath + ".pkl", "rb") as handle:
            out = pickle.load(handle, encoding="ascii")

        features = np.load(filepath + ".npz")

        out["features"] = features["arr_0"]
        out["boxes"] = features["arr_1"]

        return out
    
