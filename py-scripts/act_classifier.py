from time import time

import gc
from dataset_parser import parse_tweets_from_dir
from persistance import load_classifiers, save_author_file
from utils import abort_clean, get_printable_tweet, format_dir_name
from utils import get_language_dir_names, create_dir
from time import time


#------------------------------------------------------------------------------
#----------------------------- CLASSIFY MODULE ------------------------------
#------------------------------------------------------------------------------

def classify(options):
    '''
    Classifies a dataset respecting the PAN'17 specification.
    Predicts both language variety and 
    Will proceed as follows :
        - loads the dataset
        - [contextual] for each subdirectory of the dataset, loads the related
            classifiers
        - predicts the different labels for each author within the loaded 
            corpus
        - outputs the result files
        - [contextual] checks it's results
    ''' 

    #----------------------------------------------------------------------
    # Checking basic requirements
    if not(options["classification-type"]  and options["classification-type"] in ["loose", "successive"]):
        abort_clean("Classification type incorrectly specified")


    if options["verbosity"]:
        print('Classificationtype is ' + options["classification-type"])
    
    # PAN 17 specifics
    languages = get_language_dir_names()
    for lang in languages:

        if options["verbosity"]:
            print('---------------------------------------')
            print("Language up for classification: '" + lang + "'\n")
        
        processed_tweets_dir = ("" if not(options["processed-tweets-dir"]) else
            format_dir_name(options["processed-tweets-dir"]+lang))
        classifier_dir_path = format_dir_name(options["classifiers-dir"]+lang)
        output_subdir_path = format_dir_name(options["output-dir"]+lang)

        #----------------------------------------------------------------------
        # Load the tweets
        Authors = parse_tweets_from_dir(
            input_dir=format_dir_name(options["input-dir"]+lang), 
            output_dir=processed_tweets_dir,
            label=False,
            verbosity_level=options["verbosity"])

        if not (Authors):
            abort_clean("Tweets loading failed")

        #----------------------------------------------------------------------
        # Load the classifiers
        classifiers = load_classifiers(
            classifier_dir_path=classifier_dir_path,
            classification_type=options["classification-type"],
            verbose=options["verbosity"])

        #----------------------------------------------------------------------
        # Start classification
        if options["verbosity"]:
            print("Starting authors classification ...")
            t0 = time()

        classify_authors(Authors, classifiers, options["classification-type"], options["verbosity"])

        if options["verbosity"] > 1:
            for auth in Authors:
                print(auth["id"] + ":::" +
                auth["gender"] + ":::" +
                auth["variety"])
        

        if options["verbosity"]:
            print("Classification of '" + lang + 
                "' complete in %.3f seconds" %(time()-t0))
            print('---------------------------------------\n')

        create_dir(output_subdir_path)
        for auth in Authors:
            save_author_file(
                author=auth,
                output_dir=output_subdir_path,
                verbose=options["verbosity"]>1
            )

        # for memory issues, free the classifiers objects
        gc.collect()


def classify_authors(Authors, classifiers, classification_type, verbosity):
    '''
    Classifies all the tweets contained within a directory.
    Will proceed as follows :
        - predicts the different labels for each author within the corpus
        - returns the most probable labels for each author
    '''

    for auth in Authors:
        # classify gender
        gdr_classes, gdr_predictions = predict_author_proba(
            author=auth,
            model=classifiers["gender"] )
        gdr_max_idx = gdr_predictions.index(max(gdr_predictions))
        gdr_predicted = gdr_classes[gdr_max_idx]
        
        # classify variety
        var_classes, var_predictions = [], []
        if classification_type == "loose":
            var_clf = classifiers["variety"]
        elif gdr_predicted == "male": # classification_type == "successive" 
            var_clf = classifiers["variety-male"]
        else: # classification_type == "successive" and gdr == "female"
            var_clf = classifiers["variety-female"]

        var_classes, var_predictions = predict_author_proba(
        author=auth,
        model=var_clf )
        var_max_idx = var_predictions.index(max(var_predictions))
        var_predicted = var_classes[var_max_idx]

        if verbosity > 1:
            print(auth["id"] + ":::" +
                gdr_predicted + "(" +
                "{0:.2f}".format(gdr_predictions[gdr_max_idx]*100) + "%)" +                
                var_predicted + "(" +
                "{0:.2f}".format(var_predictions[var_max_idx]*100) + "%)")

        # save labels
        auth["gender"] = gdr_predicted
        auth["variety"] = var_predicted
    
    # Optional
    return Authors


def predict_author_proba(author, model):
    '''
    Classify the author object based on the tweets it contains
    Predicts the value of the "meta_label" using the model prediction method
    '''
    predicted_list = []
    classes = model.classes_.tolist()
    predictions = [0 for c in classes]

    # Handles the empty file event
    if len(author["tweets"]) == 0:
        predictions[0] = 1
        return classes, predictions

    # it is preferable to use predict_proba (in terms of statistical accuracy)
    # but this method is not always available
    if getattr(model, "predict_proba", None):
        predicted_list = model.predict_proba(author["tweets"])
        for row in predicted_list:
            predictions = [x + y for x, y in zip(predictions, row)]
    else:
        predicted_list = model.predict(author["tweets"])
        for row in predicted_list:
            predictions[classes.index(row)] += 1

    predictions = [x/sum(predictions) for x in predictions]
    return classes, predictions

