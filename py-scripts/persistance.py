from json import loads, dumps
from os import listdir
from time import time
import xml.etree.cElementTree as ET

from numpy import array
from sklearn.externals import joblib

from utils import get_classifier_name, get_features_extr_name, abort_clean
from utils import stringify_cm


#------------------------------------------------------------------------------
#---------------------------- PERSISTANCE MODULE ------------------------------
#------------------------------------------------------------------------------


def save_model(pipeline, output_dir, filename, verbose):
    '''
    Saves a classifier (pipeline) to a file.
    Directory and filename must be specified separatly
    '''
    if verbose:
        print("Saving Model into : " + output_dir + filename + "_pipe.clf")

    # Save model
    joblib.dump(pipeline, output_dir + filename + '_pipe.clf')
    # Save model configuration
    conf_file = open(output_dir + filename + '_pipe.config', mode='w')
    for step in pipeline.steps:
        conf_file.write(str(step[0]) + "\n")
        conf_file.write(str(step[1]) + "\n\n")

    if verbose:
        print("Model Saved.\n")


def load_classifiers(classifier_dir_path, classification_type, verbose):
    '''
    Loads the required classifiers for the PAN17 task.
    '''
    classifiers = {}

    if verbose:
        print("starting loading classifiers :")
        t0 = time()
    # gender classifier
    if verbose:
        print("    - loading gender classifier")
    gdr_clf_path = [path for path in listdir(
        classifier_dir_path + "gender" ) if path.endswith('.clf')][0]
    gdr_clf_path = classifier_dir_path + "gender/" + gdr_clf_path
    gdr_pipe = load_model(gdr_clf_path)
    classifiers["gender"] = gdr_pipe

    # variety classifier(s)
    if verbose:
        print("    - loading variety classifier(s)")
    if classification_type == "loose":
        var_clf_path = [path for path in listdir(
            classifier_dir_path + "variety" ) if path.endswith('.clf')][0]
        var_clf_path = classifier_dir_path + "variety/" + var_clf_path
        var_pipe = load_model(var_clf_path)
        classifiers["variety"] = var_pipe

    else: #classification_type == "successive":
        male_var_clf_path = [path for path in listdir(
            classifier_dir_path + "variety" ) if path.endswith('.male.clf')][0]
        male_var_clf_path = classifier_dir_path + "variety/" + male_var_clf_path
        var_male_pipe = load_model(male_var_clf_path)
        classifiers["variety-male"] = var_male_pipe

        female_var_clf_path = [path for path in listdir(
            classifier_dir_path + "variety" ) if path.endswith('.female.clf')][0]
        female_var_clf_path = classifier_dir_path + "variety/" + female_var_clf_path
        var_female_pipe = load_model(female_var_clf_path)
        classifiers["variety-female"] = var_female_pipe

    if verbose:
        print("Classifiers Loading --- success in %.3f seconds\n" %(time()-t0))
    return classifiers


def load_model(filename):
    '''
    Loads a classifier (pipeline) from a file.
    '''
    # Load model
    try:
        pipe = joblib.load(filename)
    except:
        abort_clean("failed to load the classifier")
    return pipe


def save_scores(scores, output_dir, filename, verbose) :
    '''
    Exports the data contained in the scores object to files:
        - "mean_score_micro": average micro f-score 
        - "mean_score_macro": average macro f-score
        - "confusion_matrix": confusion matrix
        - "best_macro_score": score of the resulting model
    '''
    if verbose:
        print("Saving training results into : " + output_dir + 
            filename + "_***.txt")
        
    # Save scores
    outfile = open(output_dir + filename + "_scores.txt", 'w')
    outfile.write("-------------------------------\n")
    outfile.write("--- Results of the training ---\n")
    outfile.write("-------------------------------\n")
    outfile.write("\n")
    outfile.write("average micro f-score: " + str(scores["mean_score_micro"]))
    outfile.write("\n")
    outfile.write("average macro f-score: " + str(scores["mean_score_macro"]))
    outfile.write("\n")
    outfile.write("model f-score: " + str(scores["best_macro_score"]))
    outfile.write("\n")
    outfile.close()

    # Save confusion matrix
    csvfile = open(output_dir + filename + "_confusion_matrix.csv", 'w')
    labels = scores["labels"]
    confusion = scores["confusion_matrix"].tolist()
    header = ','.join([''] + labels) + '\n'
    csvfile.write(header)
    for idx, row in enumerate(confusion):
        str_row = labels[idx] + ","
        str_row += ','.join(['{0:.3f}'.format(float(x)) for x in row])  + '\n'
        csvfile.write(str_row)
    csvfile.close()

    if verbose:
        print("Training results saved.\n")


def save_comparison_table(table, extractors, classifiers, filepath):
    '''
    Exports the data contained in the table to csv format:
    by def: each row represent a different set of feature
            each column represent a different kind of classifier
    '''
    file = open(filepath, 'w')

    header = ','.join(['Features\\Classifier'] + 
        [get_classifier_name(clf) for clf in classifiers]) + '\n'
    file.write(header)

    for idx, row in enumerate(table):
        str_row = get_features_extr_name(extractors[idx]) + ","
        str_row += ','.join(['{0:.3f}'.format(float(x)) for x in row])  + '\n'
        file.write(str_row)
    file.close()


def save_optimisation_results(grid, output_dir, score, verbose):
    '''
    Exports the results of an optimisation to a csv and a text files
    '''
    if verbose:
        print("Saving optimisation results into : " + output_dir +      
            score + "_opt_***.txt/csv")

    # best hyper-parameters report
    best_clf_file = open(output_dir+score+"_opt_best.txt", 'w')
    best_clf_file.write("Best parameters set found on development set:\n")
    best_parameters = grid.best_params_
    for param_name in sorted(best_parameters.keys()):
        best_clf_file.write("\t%s: %r\n" % (param_name, 
            best_parameters[param_name]))
    best_clf_file.close()

    # full hyper-parameters report
    full_clf_file = open(output_dir+score+"_opt_full.csv", 'w')
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    keys = [key for key in grid.cv_results_['params'][0]]
    full_clf_file.write(','.join(["instance", score, "delta"] + keys)+"\n")
    for idx, (mean, std, params) in enumerate(zip(means, stds, 
            grid.cv_results_['params'])):
        line = str(idx+1)+ "," + "{0:.3f}".format(mean) + ","
        line += "{0:.3f}".format(std * 2) + ","
        line += ",".join([str(params[key]) for key in keys]) + "\n"
        full_clf_file.write(line)
    full_clf_file.close()

    if verbose:
        print("Optimisation results saved.\n")


def load_config(file_path):
    '''
    Loads a configuration file given a file_path.
    Returns the config object extracted.
    Will raise an Exception if the file can't be loaded.
    '''
    config_file = open(file_path)
    config = loads(config_file.read(), encoding="utf8")
    return config


def save_author_file(author, output_dir, verbose):
    '''
    Saves an author object to an xml file respecting the PAN'17 format
    '''

    root = ET.Element("author")
    root.attrib["id"]=author["id"]
    root.attrib["lang"]=author["lang"]
    root.attrib["variety"]=author["variety"]
    root.attrib["gender"]=author["gender"]

    tree = ET.ElementTree(root)
    if verbose :
        print("saving author " + author["id"] + " informations to file")
    tree.write(output_dir + author["id"] + ".xml")


def load_author_file(file_path, verbose):
    '''
    Loads an author object from an xml file respecting the PAN'17 format
    '''
    tree = ET.parse(file_path)
    root = tree.getroot()
    author = dict()
    author["id"] = root.attrib["id"]
    author["lang"] = root.attrib["lang"]
    author["variety"] = root.attrib["variety"]
    author["gender"] = root.attrib["gender"]

    if verbose :
        print("Loading author " + author["id"] + " complete")
    return author

def save_evaluation_results(results, input_dir, output_dir, verbose):
    '''
    Loads an author object from an xml file respecting the PAN'17 format
    '''
    if verbose:
        print("Saving evaluation results into : " + output_dir +
            "evaluation_results.txt")

    # TXT file containing all the results
    evaluation_file = open(output_dir + "evaluation_results.txt", 'w')
    evaluation_file.write(
        "Results obtained for the evaluation of the tweets contained in :\n" +
        "    " + input_dir + "\n\n" )

    for lang, res in results.items():
            evaluation_file.write("---------------------\n")
            evaluation_file.write("language : " + lang + "\n")
            evaluation_file.write("---------------------\n")
            evaluation_file.write("    - n files :                        " +
                str(res["n_files"]) + "\n")
            evaluation_file.write("    - gender successful prediction :   " +
                str(res["gdr-positive-eval"]) + " (" +
                "{0:.2f}".format(res["gdr-positive-eval"]/res["n_files"]*100) +
                "%)" + "\n")
            evaluation_file.write("    - variety successful prediction :  " +
                str(res["var-positive-eval"]) + " (" +
                "{0:.2f}".format(res["var-positive-eval"]/res["n_files"]*100) +
                "%)" + "\n")
            evaluation_file.write("    - confusion matrix (variety) :\n\n")
            evaluation_file.write(stringify_cm(
                array(res["var-confusion-matrix"]),
                res["var-labels"]))
            evaluation_file.write("\n")
            evaluation_file.write("    - confusion matrix (gender) :\n\n")
            evaluation_file.write(stringify_cm(
                array(res["gdr-confusion-matrix"]), 
                res["gdr-labels"]))
            evaluation_file.write("\n")

    if verbose:
        print("Evaluation results saved")