from time import time

from dataset_parser import parse_tweets_from_dir
from persistance import load_classifiers, save_author_file
from utils import abort_clean, get_printable_tweet, format_dir_name
from utils import get_language_dir_names
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
    # PAN 17 specifics
    languages = get_language_dir_names()
    Authors_processed = []

    for lang in languages:

        if options["verbosity"]:
            print('---------------------------------------')
            print("Language up for classification: '" + lang + "'\n")

        #----------------------------------------------------------------------
        # Load the tweets
        Authors = parse_tweets_from_dir(
            input_dir=format_dir_name(options["input-dir"]+lang), 
            output_dir=format_dir_name(options["processed-tweets-dir"]+lang),
            label=False,
            verbosity_level=options["verbosity"])

        if not (Authors):
            abort_clean("Tweets loading failed")

        #----------------------------------------------------------------------
        # Load the classifiers
        classifier_dir_path = format_dir_name(options["classifiers-dir"] + 
            lang)
        classifiers = load_classifiers(
            classifier_dir_path=classifier_dir_path,
            verbose=options["verbosity"] )

        #----------------------------------------------------------------------
        # Start classification
        if options["verbosity"]:
            print("Starting authors classification ...")
            t0 = time()
        classify_authors(Authors, classifiers, options["verbosity"])

        Authors_processed += Authors
        if options["verbosity"] > 1:
            for auth in Authors:
                print(auth["id"] + ":::" +
                auth["gender"] + ":::" +
                auth["variety"])
        

        if options["verbosity"]:
            print("Classification of '" + lang + 
                "' complete in %.3f seconds" %(time()-t0))
            print('---------------------------------------\n')

    for auth in Authors_processed:
        save_author_file(
            author=auth,
            output_dir=options["output-dir"],
            verbose=options["verbosity"]>1
        )


def classify_authors(Authors, classifiers, verbosity):
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
        var_classes, var_predictions = predict_author_proba(
            author=auth,
            model=classifiers["variety"] )
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

