def optimizer(corpus, labels, features='', algorithm='', scores=[]):
    
    # Get transformation Pipeline
    pipe = None
    transformers = get_features_transformers(verbose=True)
    if features == 'tfidf':
        pipe = Pipeline([('Counter', transformers["Counter-2gram"]), 
                           ('Tf-idf', transformers["Tf-idf"])])
    elif features == 'lsa':
        vectorizer = TfidfVectorizer(stop_words='english', 
                                     use_idf=True, 
                                     smooth_idf=True)
        svd_model = TruncatedSVD(n_components=500, 
                         algorithm='randomized',
                         n_iter=10, random_state=42)
        pipe = Pipeline([('Tf-idf', vectorizer),
                            ('SVD', svd_model)])
    else:
        print("ERROR - no features selected")
        return

    # Transform the corpus into learnable values
    n_samples = len(corpus)
    X = pipe.fit_transform(corpus["tweets"].values)
    y = corpus["class"].values
    
    # Sample 10% of the corpus for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0)

    # Set the classifier and the parameters to be tuned
    tuned_parameters = []
    classifier = None
    if algorithm == 'svm':
        classifier = LinearSVC()
        tuned_parameters = [{'penalty': ['l1','l2'],
                            'dual' : [False],
                            'multi_class' : ['ovr', 'crammer_singer'],
                            'max_iter': [500, 1000, 1500]}]
    elif algorithm == 'mlp':
        classifier = MLPClassifier()
        tuned_parameters = [{'solver': ['sgd'],
                             'learning_rate': ['constant', 'invscaling', 'adaptive'],
                             'activation':  ['identity', 'logistic', 'tanh', 'relu'],
                             'hidden_layer_sizes': [(50,),(75,),(100,),(125,),(150,)]},
                             {'solver': ['lbfgs', 'adam'],
                             'activation':  ['identity', 'logistic', 'tanh', 'relu'],
                             'hidden_layer_sizes': [(50,),(75,),(100,),(125,),(150,)]}]
    else:
        print("ERROR - no classifier selected")
        return
    
    # Launch the tuning of hyper parameters
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(classifier, tuned_parameters, cv=10,
                        scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()