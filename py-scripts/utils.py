import time
import numpy
from pandas import DataFrame
import re
import os


def get_printable_tweet(tweet_text):
    '''
    As many utf8 caracters are not convertible to ascii/charmap, this function
    removes unprintable caracters for the console.
    '''
    return re.sub(u'[^\x00-\x7f]',u'', tweet_text)


def build_corpus(authors, label_lang=True, label_variety=True, 
                 label_gender=True, shuffle=True, verbosity_level=1):
    '''
    Given an Author object this function returns a corpus of tweet labelled
    Labels can be set with the gender_label and age_label
    At least one label must be set to true
    Returns all the unique labels found
    '''

    if verbosity_level:
        print("Starting Corpus Building ...")

    if not(label_lang or label_variety or label_gender):
        print("Corpus Building --- failure")
        print("No label selected.")
        return None

    # Building tweet Corpus
    t0 = time.time()
    tweets = []
    labels = []
    indexes = []

    for author in authors:

        label = ''
        if label_lang:
            label += author["lang"] + " "
        if label_variety:
            label += author["variety"] + " "
        if label_gender:
            label += author["gender"] + " "
        label = label[:-1]

        for idx, tweet in enumerate(author["tweets"]):
            tweets.append(tweet)
            indexes.append(idx)
            labels.append(label)
    
    labels_unique = list(set(labels))
    corpus = DataFrame({"tweets" : tweets, "class" : labels}, index=indexes)

    if shuffle :
        corpus = corpus.reindex(numpy.random.permutation(corpus.index))

    if verbosity_level:
        print("Labels used : " + str(len(labels_unique)))
        for l in labels_unique:
            print("   - " + l)

    if verbosity_level :
        print("Corpus Building --- success in : " + 
            "{0:.2f}".format(time.time() - t0) + " seconds" + "\n")


    # At this point, the corpus is a table with 2 columns:
    # corpus['tweets'] which contains all the tweets (textual values)
    # corpus['class'] which contains the classes associated with each tweets
    # boths columns are linked thanks to indices (but this aspect is hidden 
    # from the user)
    return corpus, labels_unique

def print_corpus(corpus):
    '''
    Prints all the tweets contained within the corpus give as parameter
    '''
    tweets = corpus['tweets'].values
    for t in tweets: 
        print(get_printable_tweet(t))

def abort_clean (error_msg, error_msg2=""):
    '''
    Stops the execution of the program.
    Displays up to 2 messages before exiting. 
    '''
    print("ERROR : " + error_msg)
    if error_msg2 :
        print("      : " + error_msg2)
    print(" -- ABORTING EXECUTION --")
    print()
    exit()

def format_dir_name(dir_path):
    '''
    Formats the name of the given directory:
        - Transforms to absolute path
        - Ensure there is a '/' at the end
    '''
    path = os.path.abspath(dir_path)
    path = os.path.join(path, '')
    return path


def print_scores(scores):
    '''
    Prints (pretty) the scores object in the console
    '''
    print("Results of the model training :")
    print("    - micro score average: " + str(scores["mean_score_micro"]))
    print("    - macro score average: " + str(scores["mean_score_macro"]))
    print("    - score of the resulting clf: "+str(scores["best_macro_score"]))
    print("    - resulting confusion matrix :")
    try :
        print_cm(scores["confusion_matrix"],scores["labels"])
    except :
        print("Confusion matrix printing failed\n")


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """
    pretty prints for confusion matrixes
    """
    columnwidth = max([len(x) for x in labels]+[10]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=' ')
    for label in labels: 
        print("%{0}s".format(columnwidth) % label, end=' ')
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=' ')
        for j in range(len(labels)): 
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=' ')
        print()


def create_dir(new_dir):
    """
    Checks if the specified direcory does exists
    Creates it if that is not the case
    """
    os.makedirs(new_dir,exist_ok=True)

def get_features_extr_name(features_extr):
    """
    Returns the features extractor name
    """
    name = "+".join([x[0] for x in features_extr])
    return name

def get_classifier_name(classifier):
    """
    Returns the classifier name
    """
    return classifier[0]