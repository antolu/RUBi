import numpy as np
import pandas as pd
import tensorflow as tf
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_dataset(args_dict):
    """"
        This function return a tf.data.Dataset in the format : [img_embadding, question, answer]
    @
    """

    df_annot = pd.read_json(os.path.join(args_dict.dir_data, 'vqacp_v2_train_annotations.json'))
    top_3000_answer = set(list(df_annot['multiple_choice_answer'].value_counts(3000).index))

    if args_dict.mode == 'train':

        df_quest = pd.read_json(os.path.join(args_dict.dir_data, 'vqacp_v2_train_questions.json'))
        df = pd.merge(df_annot[['question_type', 'multiple_choice_answer',
                                     'image_id', 'answer_type', 'question_id']]
                      , df_quest[['coco_split', 'question', 'question_id']], on='question_id')


        df = df[(df['coco_split'] == 'train2014') & (df['multiple_choice_answer'].isin(top_3000_answer))]
        d_question = tf.data.Dataset.from_tensor_slices(df['question'])
        d_answer = tf.data.Dataset.from_tensor_slices(df['multiple_choice_answer'])

    elif args_dict.mode == 'val':
        df_quest = pd.read_json(os.path.join(args_dict.dir_data, 'vqacp_v2_train_questions.json'))
        df = pd.merge(df_annot[['question_type', 'multiple_choice_answer',
                                     'image_id', 'answer_type', 'question_id']]
                      , df_quest[['coco_split', 'question', 'question_id']], on='question_id')

        df = df[(df['coco_split'] == 'val2014') & (df['multiple_choice_answer'].isin(top_3000_answer))]
        d_question = tf.data.Dataset.fom_tensor_slices(df['question'])
        d_answer = tf.data.Dataset.from_tensor_slices(df['multiple_choice_answer'])

    elif args_dict.mode == 'test':  # TODO: understand the test dataset. Why there is a train2014 and val2014 ?
        df_annot = pd.read_json(os.path.join(args_dict.dir_data, 'vqacp_v2_test_annotations.json'))
        df_quest = pd.read_json(os.path.join(args_dict.dir_data, 'vqacp_v2_test_questions.json'))
        df = pd.merge(df_annot[['question_type', 'multiple_choice_answer',
                                'image_id', 'answer_type', 'question_id']]
                      , df_quest[['coco_split', 'question', 'question_id']], on='question_id')

        df = df[(df['coco_split'] == 'val2014') & (df['multiple_choice_answer'].isin(top_3000_answer))]
        d_question = tf.data.Dataset.from_tensor_slices(df[df['coco_split'] == 'val2014']['question'])
        d_answer = tf.data.Dataset.from_tensor_slices(df[df['coco_split'] == 'val2014']['multiple_choice_answer'])

    else:
        assert "Invalid set parameter. Choose between: train, val, test"

    imgs_path = tf.data.Dataset.from_tensor_slices(df['image_id'].apply(lambda x: question_id2path(x, args_dict)))

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    d_img = imgs_path.map(lambda x: process_path(x, args_dict), num_parallel_calls=AUTOTUNE)

    tf_dataset = tf.data.Dataset.zip((d_img, d_question, d_answer)).shuffle(500).repeat().batch(args_dict.batch_size)

    return tf_dataset

def question_id2path(img_id, args_dict):
    img_id = str(img_id)
    full_number = ''.join((12 - len(img_id)) * ['0']) + img_id
    if args_dict.mode == 'train' or 'val':
        return os.path.join(args_dict.dir_data, "train2014/COCO_train2014_" + full_number + ".jpg")
    else:
        return os.path.join(args_dict.dir_data, "val2014/COCO_val2014_" + full_number + ".jpg")

def decode_img(img, args_dict):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [args_dict.IMG_WIDTH, args_dict.IMG_HEIGHT])

def process_path(file_path, args_dict):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, args_dict)
    return img