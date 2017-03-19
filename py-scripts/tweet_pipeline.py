from sklearn.pipeline import Pipeline

def get_pipeline(features_extr, classifier, verbose=1):
    '''
    Builds an execution pipeline from the features extractors and the 
    classifier given as parameter.
    '''

    if(verbose):
        print("Starting building Execution Pipeline ... ")

    steps = features_extr + [classifier]
    pipe = Pipeline(steps)
    
    if(verbose):
        print("Execution Pipeline built.\n")

    return pipe