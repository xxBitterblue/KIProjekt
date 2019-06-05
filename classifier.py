#!/usr/bin/env python

import sys, os, argparse, json, pickle, math

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
        wordsInCategories = {}  #{label: {word: amount...}...}
        probability = {}  #{label: {word: probability...}.....}
        categories = {}  #{label: amount.....}
        categorieAmount = {}
        categoriesProbability = {}
        voc = [x for x in voc.keys()]  #Saves all vocabulary keys in a list for an easy iteration
        allArticle = len(features)

        "Read in amount ob categories in dic 'categories' and the amount of words for all texts of a " \
        "categorie in the dic 'wordsInCategories' in the way '{label:{word: amount...}...}' "
        for article, token in features.items():
            actLabel = labels[article]
            categories[actLabel] = 1 + categories.get(actLabel, 0)
            if actLabel not in wordsInCategories.keys():
                wordsInCategories[actLabel] = {}
            for word in voc:
                if word not in wordsInCategories[actLabel]:
                    wordsInCategories[actLabel][word] = 0
                if word in token.keys():
                    wordsInCategories[actLabel][word] = 1 + wordsInCategories[actLabel].get(word, 0)

        "Calculate the probability for each word of our vocabulary for each label"
        for label, words in wordsInCategories.items():
            for wor, amount in words.items():
                if label not in probability.keys():
                    probability[label] = {}
                probability[label][wor] = amount / categories[label]

        "Calculate probability of each label"
        sumLabel = sum(categories.values())
        for label, amount in categories.items():
            categoriesProbability[label] = amount / sumLabel

        for label, amount in categories.items():
            categorieAmount[label] = 1 / (amount + (amount / 10))


        "Saving dic into a pickle file"
        pickle.dump(probability, open("ProbabilityOfWordsInArticle.p", "wb"))
        pickle.dump(categoriesProbability, open("CategoriesInArticle.p", "wb"))
        pickle.dump(categorieAmount, open("CategoriesAmount.p", "wb"))
        #  for loading: probability = pickle.load( open( "ProbabilityOfWordsInArticle.p", "rb" ) )


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
        findLabel = {} # {articleName : {categories : probability}}
        wordProbability = pickle.load(open("ProbabilityOfWordsInArticle.p", "rb")) #wahrscheinlichkeit für ein wort pro Kategorie
        categoriesProbability = pickle.load(open("CategoriesInArticle.p", "rb")) #generelle wahrscheinlichkeit einer Kategorie
        #categorieAmount = pickle.load(open("CategoriesAmount.p", "rb")) #amount für wenn die amount 0 wäre
        standartAmount = 0.0001

        "Calculates probability for each label for every article"
        for name, words in features.items():
            findLabel[name] = {}
            for cat, probability in categoriesProbability.items():
                findLabel[name][cat] = math.log(probability,2)
                for w, amount in wordProbability[cat].items():
                    if w in words:
                        if wordProbability[cat][w] != 0:
                            findLabel[name][cat] += math.log(wordProbability[cat][w],2)
                        else:
                            findLabel[name][cat] += math.log(standartAmount,2)
                    else:
                        if wordProbability[cat][w] != 0:
                            findLabel[name][cat] += math.log(1 - wordProbability[cat][w],2)
                        else:
                            findLabel[name][cat] += math.log(1-standartAmount,2)

        "Picks for each article the label with the highest probability"
        for article, label in findLabel.items():
            aktLabel = sorted(label.items(), key=lambda x: x[1], reverse=True)
            print(article)
            print(aktLabel)
            aktLabel = aktLabel[0][0]
            print(aktLabel)
            findLabel[article] = aktLabel

        return findLabel


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
        features, labels, voc = read_json('test.json')
        result = classifier.apply(features)

        allArticle = len(features)
        wrongPicked = 0

        for name, pickedLabel in result.items():
            if pickedLabel != labels[name]:
                wrongPicked += 1

        failure = wrongPicked / allArticle
        print(failure * 100, "%")





