# config-files

This direct should contain all the configuration files for the classifiers and the features extractors.

### Directory structure
    config-files
        |--- classifiers
        |       |--- default
        |       |       |--- clf_1-default.json     // default config file of clf_1
        |       |       .--- ...
        |       |       \--- clf_n-default.json     // default config file of clf_n
        |       \--- custom
        |               |--- clf_1.json             // custom config file of clf_1
        |               .--- ...
        |               \--- clf_n.json             // custom config file of clf_n
        |--- feat-extractors
        |       |--- default
        |       |       |--- ext_1-default.json     // default config file of ext_1
        |       |       .--- ...
        |       |       \--- ext_n-default.json     // default config file of ext_n
        |       \--- custom
        |              |--- ext_1.json              // custom config file of ext_1
        |              .--- ...
        |              \--- ext_n.json              // custom config file of ext_n
        \--- optimize-parameters
                |--- default
                |       |--- opt-ext1-clf1-default.json     // default config file for the optimization of clf1 using ext1
                |       .--- ...
                |       \--- opt-extn-clfn-default.json     // default config file for the optimization of clfn using extn
                \--- custom
                        |--- opt-ext1-clf1.json             // custom config file for the optimization of clf1 using ext1
                        .--- ...
                        \--- opt-extn-clfn.json             // custom config file for the optimization of clfn using extn

### Notes
The **default** directories contain templates that may help you tuning the parameters of the classifiers/features-extractors to the desired values.