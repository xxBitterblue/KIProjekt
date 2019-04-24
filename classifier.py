#!/usr/bin/env python

import sys, os, argparse, json

"""
  "News Classifier" 
  -------------------------
  This is a small interface for document classification. Implement your own Naive Bayes classifier 
  by completing the class 'NaiveBayesDocumentClassifier' below.

  To run the code, 

  1. place the files 'train.json' and 'test.json' in the current folder.

  2. train your model on 'train.json' by calling > python classifier.py --train 

  3. apply the model to 'test.json' by calling > python classifier.py --apply

"""


class NaiveBayesDocumentClassifier:

    def __init__(self):
        """ The classifier should store all its learned information
            in this 'model' object. Pick whatever form seems appropriate
            to you. Recommendation: use 'pickle' to store/load this model! """
        self.model = None

    def train(self, features, labels, voc):
        """
        trains a document classifier and stores all relevant
        information in 'self.model'.

        @type features: dict
        @param features: Each entry in 'features' represents a document
                         by its so-called bag-of-words vector.
                         For each document in the dataset, 'features' contains
                         all terms occurring in the document and their frequency
                         in the document:
                         {
                           'doc1.html':
                              {
                                'the' : 7,   # 'the' occurs seven times
                                'world': 3,
                                ...
                              },
                           'doc2.html':
                              {
                                'community' : 2,
                                'college': 1,
                                ...
                              },
                            ...
                         }
        @type labels: dict
        @param labels: 'labels' contains the class labels for all documents
                       in dictionary form:
                       {
                           'doc1.html': 'arts',       # doc1.html belongs to class 'arts'
                           'doc2.html': 'business',
                           'doc3.html': 'sports',
                           ...
                       }
        """

        categories = {}



        # FIXME: implement training

        # FIXME: at the end of training, store self.model using pickle.

    def apply(self, features):
        """
        applies a classifier to a set of documents. Requires the classifier
        to be trained (i.e., you need to call train() before you can call apply()).

        @type features: dict
        @param features: see above (documentation of train())

        @rtype: dict
        @return: For each document in 'features', apply() returns the estimated class.
                 The return value is a dictionary of the form:
                 {
                   'doc1.html': 'arts',
                   'doc2.html': 'travel',
                   'doc3.html': 'sports',
                   ...
                 }
        """
        raise NotImplementedError()

        # FIXME: implement model application


if __name__ == "__main__":

    # parse command line arguments (no need to touch)
    parser = argparse.ArgumentParser(description='A document classifier.')
    parser.add_argument('--train', help="train the classifier", action='store_true')
    parser.add_argument('--apply', help="apply the classifier (you'll need to train or load" \
                                        "a trained model first)", action='store_true')
    parser.add_argument('--inspect', help="get some info about the learned model",
                        action='store_true')

    args = parser.parse_args()

    classifier = NaiveBayesDocumentClassifier()


    def read_json(path):
        with open(path) as f:
            all_data = json.load(f)
            data = all_data["docs"]
            voc = all_data["vocabulary"]
            features, labels = {}, {}
            for f in data:
                features[f] = data[f]['tokens']
                labels[f] = data[f]['label']
        return features, labels, voc


    if args.train:
        features, labels, voc = read_json('train_filtered.json')
        classifier.train(features, labels, voc)

    if args.apply:
        features, labels = read_json('test.json')
        result = classifier.apply(features)

        # FIXME: measure error rate on 'test.json'





