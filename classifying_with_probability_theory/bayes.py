"""
Created on Feb 20, 2024

@author: Ming Xu
"""

import numpy as np


def load_data_set():
    """load data set"""
    posting_list = [
        ["my", "dog", "has", "flea", "problems", "help", "please"],
        ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
        ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
        ["stop", "posting", "stupid", "worthless", "garbage"],
        ["mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"],
        ["quit", "buying", "worthless", "dog", "food", "stupid"],
    ]
    class_vector = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_list, class_vector


def create_vocabulary_list(data_set):
    """create vocabulary list"""
    vocabulary_set = set([])
    for document in data_set:
        vocabulary_set = vocabulary_set | set(document)  # union of the two sets
    return list(vocabulary_set)


def set_of_words_2_vector(vocabulary_list, input_set):
    """set of words 2 vector"""
    return_vector = [0] * len(vocabulary_list)
    for word in input_set:
        if word in vocabulary_list:
            return_vector[vocabulary_list.index(word)] = 1
        else:
            print(f"the word: {word} is not in my vocabulary!")
    return return_vector


def train_native_bayes0(train_matrix, train_category):
    """train native bayes"""
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_numerator = np.ones(num_words)
    p1_numerator = np.ones(num_words)  # change to np.ones()
    p0_denominator = 2.0
    p1_denominator = 2.0  # change to 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_numerator += train_matrix[i]
            p1_denominator += sum(train_matrix[i])
        else:
            p0_numerator += train_matrix[i]
            p0_denominator += sum(train_matrix[i])
    p0_vector = np.log(p0_numerator / p0_denominator)  # change to np.log()
    p1_vector = np.log(p1_numerator / p1_denominator)  # change to np.log()
    return p0_vector, p1_vector, p_abusive


list_o_posts, list_classes = load_data_set()
my_vocabulary_list = create_vocabulary_list(list_o_posts)
train_mat = []
for post_in_doc in list_o_posts:
    train_mat.append(set_of_words_2_vector(my_vocabulary_list, post_in_doc))
p0_vect, p1_vect, p_ab = train_native_bayes0(train_mat, list_classes)
print(p0_vect)
print(p1_vect)
print(p_ab)
