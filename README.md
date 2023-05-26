# Machine learning modelling for decision support in pre-grouting in hard rock

The repository containts code for preprocessing, analyzing and training machine learning models to be used for decision support in pre-grouting in hard rock.

A paper was published by Rongved, Hansen and Erharter (2023) based on the results from the code in the repo. The focus in this paper was to predict `Total grout take` in kg cement and `Grouting time` in hours.

DOI: xxx

## Setup

Install and activate environment using `Poetry` or `Conda`:

With:

```bash
poetry install
poetry shell
```

Or:

```bash
conda env create --name pregrouting --file requirements.txt
conda activate pregrouting
```


## Run

Run the steps in the machine learning project with:

### Initial steps

To retrieve data about precipitation and temperature from met.no:

```bash
python src/frost_data.py
```

To preprocess, build and combine dataset:

```bash
python src/preprocessing.py
```

To remove outliers in dataset (optional - in train_eval you choose to include dataset without outliers or not).
Choose between the methods:

- hard_values - manually chosen
- local outlier factor - automated multi-feature method
- isolation forest - automated multi-feature method

```bash
python src/outlier_removal.py
```


### ML training

**Feature selection**. To find the right features to use in training.

Run script for both labels and choose feature selection method to use.

Running this script will output a list of automatically selected features.
This list of features must then be pasted into the auto_features variable
defined in `utility.py`. In utility 6 feature-sets are defined, which you then
switch between when you train the model using `train_eval.py`. Change the variable
`TRAINING_FEATURES`. The 6 feature sets are:

- auto_features<take,time> - from automatic feature selection
- all_features - all possible features for training (not labels)
- train_features_manual_domain - manually chosen features using domain knowledge
- train_features_no_previous - train_features_manual_domain without previous grouting time, previous total grout take, and previous stop pressure
- train_features_small - a small list of features with the highest correlation and feature importance.

```bash
python src/feature_selection.py
```

To optimize hyperparameters for the model.

Optimized parameters will be saved in a yaml file in `src/config`. This file will be
loaded when running training in `train_eval.py`.
Remember to run hyperparameter optimization for both labels and for several algorithms.

```bash
python src/hyperparameter_tuning.py
```

Final run of training with optimized parameters.
Most important constants to choose: LABEL, TRAINING_FEATURES, MODEL_TYPE_PARAMS

Choose training concept between cross validation and train-test-split

Choose to make plots of results and to calculate prediction intervals.

```bash
python src/train_eval.py
```

## Contributors

Ida Rongved: development and paper writing

Georg Erharter: development and paper writing

Tom F. Hansen: development and paper writing

