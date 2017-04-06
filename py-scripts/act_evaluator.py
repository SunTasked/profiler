from os import listdir
from os.path import isfile, join
from time import time

from numpy import array

from persistance import load_author_file, save_evaluation_results
from utils import get_language_dir_names, format_dir_name, abort_clean
from utils import get_variety_labels, get_gender_labels, stringify_cm

def evaluate(options):
    '''
    Evaluates the results of a classification in the context of PAN17
    The input directory must be structured according to PAN17 specifications
    Will proceed as follows :
        - loads the author files
        - loads the truth files (one per language)
        - compares the predicted labels with the truth
        - outputs the results
    '''
    # PAN 17 specifics
    language_dirs = get_language_dir_names() 
    
    #--------------------------------------------------------------------------
    # Check basic requirements
    if not (options["truth-dir"]):
        abort_clean("truth directory not specified")
    

    #--------------------------------------------------------------------------
    # Load the author files
    if options["verbosity"]:
        print("Loading authors files ...")
        t0 = time()

    Authors = []
    file_name_list = [f for f in listdir(options["input-dir"]) 
        if isfile(join(options["input-dir"], f))]
    for file_name in file_name_list:
        auth = load_author_file(
            file_path=options["input-dir"]+file_name,
            verbose=options["verbosity"]>1 )
        Authors.append(auth)

    if options["verbosity"]:
        print("Files loaded : " + str(len(Authors)))
        print("Loading author files --- success in %.3f seconds\n"  %(time()-t0))

    #--------------------------------------------------------------------------
    # Load the truth files
    if options["verbosity"]:
        print("Loading truth files ...")
        t0 = time()

    truth=dict()
    for lang in language_dirs:
        lang_dir = format_dir_name(options["truth-dir"] + lang)
        try:
            truth_file = open(lang_dir + "truth.txt")
        except:
            abort_clean("Can't open truth file",
                "Couldn't open " + lang_dir + "truth.txt")
        
        truth_lines = [x.strip().split(':::') for x in truth_file.readlines()]
        attrs = dict()
        for l in truth_lines :
            attrs[l[0]] = {
                "gender" : l[1],
                "variety" : l[2]
                }

        truth[lang] = attrs

    if options["verbosity"]:
        print("Files loaded : " + str(len(truth)))
        print("Loading truth files --- success in %.3f seconds\n"  %(time()-t0))
    
    #--------------------------------------------------------------------------
    # Compute results
    if options["verbosity"]:
        print("Computing results ...")
        t0 = time()
    
    # preparing result data-structure
    results = dict()
    for lang in language_dirs:
        var_labels = get_variety_labels(lang)
        var_confusion_matrix = [[0 for x in var_labels] for y in var_labels]
        gdr_labels = get_gender_labels()
        gdr_confusion_matrix = [[0 for x in gdr_labels] for y in gdr_labels]
        results[lang]={
            "n_files":0,
            "gdr-labels":gdr_labels,
            "gdr-confusion-matrix":gdr_confusion_matrix,
            "gdr-positive-eval":0,
            "var-labels":var_labels,
            "var-confusion-matrix":var_confusion_matrix,
            "var-positive-eval":0
            }

    # Starting computation
    for auth in Authors:
        lang_res = results[auth["lang"]]
        auth_truth = truth[auth["lang"]][auth["id"]]

        results[auth["lang"]]["n_files"] += 1

        auth_gdr_eval = auth_truth["gender"] == auth["gender"]
        auth_var_eval = auth_truth["variety"] == auth["variety"]

        var_labels = lang_res["var-labels"]
        lang_res["var-confusion-matrix"][var_labels.index(auth_truth["variety"])][var_labels.index(auth["variety"])] += 1
        gdr_labels = lang_res["gdr-labels"]
        lang_res["gdr-confusion-matrix"][gdr_labels.index(auth_truth["gender"])][gdr_labels.index(auth["gender"])] += 1

        results[auth["lang"]]["gdr-positive-eval"] += 1 if auth_gdr_eval else 0
        results[auth["lang"]]["var-positive-eval"] += 1 if auth_var_eval else 0

    if options["verbosity"]:
        print("Computing results --- success in %.3f seconds\n"  %(time()-t0))
    
    #--------------------------------------------------------------------------
    # Save results

    save_evaluation_results(
        results=results,
        input_dir=options["input-dir"],
        output_dir=options["output-dir"],
        verbose=options["verbosity"])