from utils import abort_clean

# Classifiers
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def get_classifier(classifier_str, verbose=1):
    '''
    Returns a classifier specified in parameter
    Available classifiers are :
        - nb  : NaiveBayes (TODO)
        - mlp : Multi-layered Perceptron (TODO)
        - rf  : Random Forest (TODO)
        - svm : Support Vector Machine

    A classifier can be specified : (TODO)
        - by its name --> a default ft_extr will be instanciated
        - by a path to a config file, --> a custom ft_extr will be instanciated
    '''


    #--------------------------------------------------------------------------
    # Check basic requirements

    if not (classifier_str == "svm"):
        abort_clean("Unknown classifier.")

    if(verbose):
        print("Starting loading classifier ... ")

    
    #--------------------------------------------------------------------------
    # Get required classifier
    clf_name = ""
    clf = None

    if classifier_str == "svm":
        clf_name, clf = get_svm()
    
    
    #--------------------------------------------------------------------------
    # Return classifier
    if(verbose):
        print("classifier loaded: '" + clf_name + "'\n")

    res = (clf_name, clf)
    return res


def get_svm(config=None):
    '''
    Returns a svm classifier.
    If specified, follows the config to setup the svm
    Else follows default svm setup.
    '''
    if not (config):
        clf = LinearSVC( #---------------------------- Default Value
                    C=1.0,
                    loss='squared_hinge',
                    penalty='l1', #------------------- l2
                    dual=False, #--------------------- True
                    tol=1e-4,
                    multi_class='crammer_singer', #--- ovr
                    fit_intercept=True,
                    intercept_scaling=1,
                    class_weight=None,
                    verbose=0,
                    random_state=None,
                    max_iter=500) #------------------- 1000
        clf_name = "svm-default"
        return clf_name, clf

    