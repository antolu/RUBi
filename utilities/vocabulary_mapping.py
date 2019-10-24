#!/usr/bin/env python3

import pandas as pd
import os
import json
import re
from argparse import ArgumentParser
from nltk.tokenize import word_tokenize
import csv


def get_vocab2id(vocab):
    """
    Parameters
    ----------

    Returns
    -------
    a dict that links each word from vocab to a unique id
    """
    vocab2id = {'<eos>': 0}
    for pos, word in enumerate(vocab, 1):
        vocab2id[word] = pos
    return vocab2id


def get_vocab(questions):
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

    questions.apply(word_tokenize).apply(lambda x: toSet(vocab, x))

    return vocab


def preprocess_sentence(sentence):
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
    prep_quest = re.sub(r'[^\w\s]', '', sentence)  # remove punctuation
    prep_quest = prep_quest.lower()  # lower case

    return prep_quest

def create_vocabulary_mapping(args):
    answer_type = set(['yes/no', 'number', 'other'])

    if args.dataset == 'vqa-v2-cp':

        df_annot = pd.read_json(os.path.join(args.dataset_root, 'vqacp_v2_train_annotations.json'))
        top_3000_answer = list(df_annot['multiple_choice_answer'].value_counts().index)[:3000]
        answer2idx = {}
        for idx, answer in enumerate(top_3000_answer):
            answer2idx[answer] = idx

        df_annot_train = pd.read_json(os.path.join(args.dataset_root, 'vqacp_v2_train_annotations.json'))
        df_quest_train = pd.read_json(os.path.join(args.dataset_root, 'vqacp_v2_train_questions.json'))

        df_annot_test = pd.read_json(os.path.join(args.dataset_root, 'vqacp_v2_test_annotations.json'))
        df_quest_test = pd.read_json(os.path.join(args.dataset_root, 'vqacp_v2_test_questions.json'))

        df_train = pd.merge(df_annot_train[['question_type', 'multiple_choice_answer',
                                'image_id', 'answer_type', 'question_id']],
                      df_quest_train[['coco_split', 'question', 'question_id']], on='question_id')

        df_test = pd.merge(df_annot_test[['question_type', 'multiple_choice_answer',
                                'image_id', 'answer_type', 'question_id']],
                      df_quest_test[['coco_split', 'question', 'question_id']], on='question_id')

        # dfs = [df_train, df_test]
        df = pd.concat([df_train, df_test])

    elif args.dataset == 'vqa-v2':

        df_annot = pd.DataFrame(json.load(open(os.path.join(args.dataset_root, 'v2_mscoco_train2014_annotations.json')))['annotations'])
        top_3000_answer = list(df_annot['multiple_choice_answer'].value_counts().index)[:3000]
        answer2idx = {}
        for idx, answer in enumerate(top_3000_answer):
            answer2idx[answer] = idx
            
        df_annot_train = json.load(
            open(os.path.join(args.dataset_root, 'v2_mscoco_train2014_annotations.json')))
        df_quest_train = json.load(
            open(os.path.join(args.dataset_root, 'v2_OpenEnded_mscoco_train2014_questions.json')))


        # df_train = pd.merge(df_annot_train[['question_type', 'multiple_choice_answer',
        #                         'image_id', 'answer_type', 'question_id']],
        #               df_quest_train[['question', 'question_id']], on='question_id')

        df_annot_val = json.load(open(os.path.join(args.dataset_root, 'v2_mscoco_val2014_annotations.json')))
        df_quest_val = json.load(
            open(os.path.join(args.dataset_root, 'v2_OpenEnded_mscoco_val2014_questions.json')))

        df_annot_testdev = json.load(open(os.path.join(args.dataset_root, 'v2_mscoco_val2014_annotations.json')))
        df_quest_testdev = json.load(
            open(os.path.join(args.dataset_root, 'v2_OpenEnded_mscoco_test-dev2015_questions.json')))

        df_annot_train= pd.DataFrame(df_annot_train['annotations'])
        df_quest_train = pd.DataFrame(df_quest_train['questions'])

        df_annot_val= pd.DataFrame(df_annot_val['annotations'])
        df_quest_val = pd.DataFrame(df_quest_val['questions'])

        df_annot_testdev= pd.DataFrame(df_annot_testdev['annotations'])
        df_quest_testdev = pd.DataFrame(df_quest_testdev['questions'])

        df_train = pd.merge(df_annot_train[['question_type', 'multiple_choice_answer',
                                'image_id', 'answer_type', 'question_id']],
                      df_quest_train[['question', 'question_id']], on='question_id')

        df_val = pd.merge(df_annot_val[['question_type', 'multiple_choice_answer',
                                'image_id', 'answer_type', 'question_id']],
                      df_quest_val[['question', 'question_id']], on='question_id')

        df_testdev = pd.merge(df_annot_testdev[['question_type', 'multiple_choice_answer',
                                'image_id', 'answer_type', 'question_id']],
                      df_quest_testdev[['question', 'question_id']], on='question_id')


        df = pd.concat([df_train, df_val, df_testdev])

    df = df[(df['multiple_choice_answer'].isin(top_3000_answer)) &
            (df['answer_type'].isin(answer_type))]

    question = df['question'].apply(preprocess_sentence)
    vocab = get_vocab(question)
    vocab2id = get_vocab2id(vocab)

    # THIS WHAT WE WANT
    #questions = []
    #vocab = set()
    #for item in dfs:
    #    questions.append(df['question'].apply(preprocess_sentence))
    #    vocab.update(get_vocab(questions))

    #vocab2id = get_vocab2id(vocab)

    with open(os.path.join(args.dataset_root, args.dataset + "_vocab.csv"), "w") as f:
        for word, id in vocab2id.items():
            f.write("{}, {}\n".format(word, id))

    with open(os.path.join(args.dataset_root, args.dataset + "_answers.csv"), "w") as f:
        for answer, idx in answer2idx.items():
            f.write("{}; {}\n".format(answer, idx))


def load_vocab(args):

    if args.dataset == "vqa-v2-cp":
        dataset_dir = "vqacp_v2"
    elif args.dataset == "vqa-v2":
        dataset_dir = "vqa_v2"

    vocab = set()
    vocab2id = {}
    with open(os.path.join(args.dir_data, dataset_dir, args.dataset + "_vocab.csv"), "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:
            vocab.add(row[0])
            vocab2id[row[0]] = int(row[1])

    answer2idx = {}
    idx2answer = {}
    with open(os.path.join(args.dir_data, dataset_dir, args.dataset + "_answers.csv"), "r") as f:
        csv_reader = csv.reader(f, delimiter=";")
        for row in csv_reader:
            answer2idx[row[0]] = int(row[1])
            idx2answer[int(row[1])] = row[0]

    return vocab, vocab2id, answer2idx, idx2answer


## Main
if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--dataset-root", type=str, dest="dataset_root", required=True, help="Root of the dataset")
    parser.add_argument("dataset", type=str, choices=["vqa-v2-cp", "vqa-v2"], help="Which dataset to parse")

    args = parser.parse_args()

    create_vocabulary_mapping(args)

