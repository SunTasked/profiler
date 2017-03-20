from utils import abort_clean

# Classifiers
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def get_classifier(classifier_str, verbose=1):
    '''
    Returns a classifier specified in parameter
    Available classifiers are :
        - nbb : NaiveBayes (bernouilly)
        - mlp : Multi-layered Perceptron
        - rfo : Random Forest
        - svm : Support Vector Machine

    A classifier can be specified : (TODO)
        - by its name --> a default ft_extr will be instanciated
        - by a path to a config file, --> a custom ft_extr will be instanciated
    '''

    if(verbose):
        print("Starting loading classifier ... ")

    
    #--------------------------------------------------------------------------
    # Get required classifier

    clf_name = ""
    clf = None

    if classifier_str == "svm":
        clf_name, clf = get_svm()
    
    elif classifier_str == "mlp":
        clf_name, clf = get_mlp()

    elif classifier_str == "nbb":
        clf_name, clf = get_nbb()

    elif classifier_str == "rfo":
        clf_name, clf = get_rfo()

    else:
        abort_clean("Unknown classifier.")

    
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
                    max_iter=500 #-------------------- 1000
                    ) 
        clf_name = "svm-default"
        return clf_name, clf


def get_mlp(config=None):
    '''
    Returns a Multi-Layered Perceptron classifier.
    If specified, follows the config to setup the mlp classifier
    Else follows default mlp classifier setup.
    '''
    if not (config):
        clf = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate="constant",
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=200,
            shuffle=True,
            random_state=None,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        clf_name = "mlp-default"
        return clf_name, clf


def get_nbb(config=None):
    '''
    Returns a Naive Bayes classifier (bernouilly implementation).
    If specified, follows the config to setup the NB classifier
    Else follows default NB classifier setup.
    '''
    if not (config):
        clf = BernoulliNB(
            alpha=1.0,
            binarize=.0,
            fit_prior=True,
            class_prior=None
        )
        clf_name = "nbb-default"
        return clf_name, clf


def get_rfo(config=None):
    '''
    Returns a Naive Bayes classifier (bernouilly implementation).
    If specified, follows the config to setup the NB classifier
    Else follows default NB classifier setup.
    '''
    if not (config):
        clf = RandomForestClassifier(
            n_estimators=10,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            max_features="auto",
            max_leaf_nodes=None,
            min_impurity_split=1e-7,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1, #------------------------------ 1
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None
        )
        clf_name = "rfo-default"
        return clf_name, clf
