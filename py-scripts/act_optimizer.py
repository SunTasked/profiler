from time import time

from sklearn.model_selection import GridSearchCV

from classifiers import get_classifier
from features import get_features_extr
from dataset_parser import parse_tweets_from_main_dir, parse_tweets_from_dir
from persistance import load_config, save_optimisation_results
from pipeline import get_pipeline
from utils import build_corpus, abort_clean, print_scores
from utils import create_dir, get_classifier_name, get_features_extr_name


#------------------------------------------------------------------------------
#---------------------------- OPTIMIZATION MODULE -----------------------------
#------------------------------------------------------------------------------

def optimize(options):
    '''
    Optimize the given classifier or/and features extractor on a specified list
    of parameters
    Will proceed as follows :
        - loads the dataset
        - builds the corpus
        - load the parameters for tuning
        - loads the classifiers
        - loads the features extractors
        - builds the execution pipelines
        - trains and compares the different classifiers on the corpus
        - outputs the best set of parameters found
    '''

    #--------------------------------------------------------------------------
    # Check basic requirements
    if not (options["labels"]):
        abort_clean("Labels not specified", "expected 'l', 'g' or 'v'")
    
    if not (options["hyper-parameters"]):
        abort_clean("hyper parameters not specified")


    #--------------------------------------------------------------------------
    # Load the tweets
    if 'l' in options["labels"] or "language" in options["labels"]: 
        # load all tweets for language classification
        Authors = parse_tweets_from_main_dir(
            input_dir=options["input-dir"], 
            output_dir=options["processed-tweets-dir"],
            verbosity_level=options["verbosity"])
    else : 
        # load tweets in one language for variety or gender classification
        Authors = parse_tweets_from_dir(
            input_dir=options["input-dir"], 
            output_dir=options["processed-tweets-dir"],
            label=True,
            verbosity_level=options["verbosity"])

    if not (Authors):
        abort_clean("Tweets loading failed")


    #--------------------------------------------------------------------------
    # Build the corpus and label the tweets
    corpus, labels = build_corpus(
        authors=Authors, 
        labels=options["labels"],
        shuffle=False, 
        verbosity_level=options["verbosity"])

    if corpus.empty or not(labels):
        abort_clean("Corpus building failed")
    

    #--------------------------------------------------------------------------
    # Load the optimize parameters

    try:
        params = load_config(options["hyper-parameters"])
    except:
        abort_clean("Configuration couldn't be loaded","given path: " + 
            options["hyper-parameters"])
    
    #--------------------------------------------------------------------------
    # Load the classifier
    
    t0 = time()
    classifier = get_classifier(
        classifier_str=params["classifier-call"],
        config=None,
        verbose=options["verbosity"])


    #--------------------------------------------------------------------------
    # Load the features extractors

    features_extr = get_features_extr(        
        features_str=params["features-extractr-call"],
        verbose=options["verbosity"])


    #--------------------------------------------------------------------------
    # Build the execution pipeline

    pipeline = get_pipeline(
        features_extr=features_extr, 
        classifier=classifier, 
        verbose=options["verbosity"])

    
    # Set the classifier and the parameters to be tuned
    tuning_parameters = get_opt_parameters(params)
    scores = params["scores"]
    
    if options["verbosity"]:
        print("Starting the optimization process ...")

    # Launch the tuning of hyper parameters
    for score in scores:
        print("Tuning hyper-parameters for %s" % score)

        X_values = corpus["tweets"].values
        y_labels = corpus["class"].values

        clf_optimizer = GridSearchCV(
            estimator=pipeline,
            param_grid=tuning_parameters,
            scoring='%s_macro' % score,
            fit_params=None,
            n_jobs=-1,
            pre_dispatch='2*n_jobs',
            iid=True,
            cv=None,
            refit=True,
            verbose=options["verbosity"],
            error_score='raise',
            return_train_score=True
        )

        # Start optimisation
        clf_optimizer.fit(X_values, y_labels)

        if options["verbosity"]:
            print("Best parameters set found on development set:")
            best_parameters = clf_optimizer.best_params_
            for param_name in sorted(best_parameters.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
            print()

        if options["verbosity"] > 1:
            print("Grid scores on development set:")
            means = clf_optimizer.cv_results_['mean_test_score']
            stds = clf_optimizer.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds,
                clf_optimizer.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                    % (mean, std * 2, params))

        # saving results
        save_optimisation_results(
            grid=clf_optimizer, 
            output_dir=options["output-dir"], 
            score=score,
            verbose=options["verbosity"])


def get_opt_parameters(params):
    '''
    Resolves the json incompatibility parameters
    Returns the parameters properly parsed
    '''
    opti_params = params["parameters"]

    # svm 
    if params["classifier-type"] == "svm":
        # nothing to do
        pass
    
    # mlp
    elif params["classifier-type"] == "mlp":
        # list --> tuple
        if opti_params["hidden_layer_sizes"]:
            opti_params["hidden_layer_sizes"] = [tuple(h) for 
                h in opti_params["hidden_layer_sizes"]]
    
    # nbb 
    if params["classifier-type"] == "nbb":
        # nothing to do
        pass

    # rfo 
    if params["classifier-type"] == "rfo":
        # nothing to do
        pass

    return opti_params