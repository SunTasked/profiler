# output

Please save your classifiers to this directory.

### Directory structure (suggested)

                classifiers_main_dir
                    |--- language_1
                    .       |--- gender
                    .       |       |--- gender_classifier.clf
                    .       |       \--- gender_classifier_configuration.config
                    .       \--- variety
                    .               |--- variety_classifier.clf
                    .               \--- variety_classifier_configuration.config
                    .
                    \--- language_n
                            |--- gender
                            |       |--- gender_classifier.clf
                            |       \--- gender_classifier_configuration.config
                            \--- variety
                                    |--- variety_classifier.clf
                                    \--- variety_classifier_configuration.config

### Notes
If more than one classifier are present in the directory, the selected classifier is to be the first one in terms of alphabetical order.