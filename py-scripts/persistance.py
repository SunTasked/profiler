from sklearn.externals import joblib
import numpy

def save_model(pipeline, output_dir, verbose):
    '''
    Saves a classifier (pipeline) to a file.
    Directory and filename must be specified separatly
    Returns True if the save went well
    '''
    if verbose:
        print("Saving Model into : " + output_dir + "pipe.pkl")

    # Save model
    joblib.dump(pipeline, output_dir + 'pipe.pkl')
    # Save model configuration
    conf_file = open(output_dir + 'pipe.config', mode='w')
    for step in pipeline.steps:
        conf_file.write(str(step[0]) + "\n")
        conf_file.write(str(step[1]) + "\n\n")

    if verbose:
        print("Model Saved.\n")



def save_scores(scores, output_dir, verbose) :
    '''
    Exports the data contained in the scores object to files:
        - "mean_score_micro": average micro f-score 
        - "mean_score_macro": average macro f-score
        - "confusion_matrix": confusion matrix
        - "best_macro_score": score of the resulting model
    '''
    if verbose:
        print("Saving training results into : " + output_dir + "scores.txt")
        
    # Save scores
    outfile = open(output_dir + "scores.txt", 'w')
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
    csvfile = open(output_dir + "confusion_matrix.csv", 'w')
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


def export_to_CSV(table, feature_label, classifier_label, outfile_path):
    '''
    Exports the data contained in the table to csv format:
    by def: each row represent a different set of feature
            each column represent a different kind of classifier
    '''
    outfile_path = outfile_path if outfile_path else "./result.csv"
    outfile = open(outfile_path, 'w')

    header = ','.join(['Features\\Classifier'] + classifier_label) + '\n'
    outfile.write(header)

    for idx, row in enumerate(table):
        str_row = feature_label[idx] + ","
        str_row += ','.join(['{0:.3f}'.format(float(x)) for x in row])  + '\n'
        outfile.write(str_row)
    
    outfile.close()
