# Machine learning modelling for decision support in pre-grouting in hard rock

The repository containts code for preprocessing, analyzing and training machine learning models to be used for decision support in pre-grouting in hard rock.

A paper was published by Rongved, Hansen and Erharhter (2023) based on the results from the code in the repo. The focus in this paper was to predict `Total grout take` in kg cement and `Grouting time` in hours.

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

To retrieve data about precipitation and temperature:

```bash
python src/frost_data.py
```

To preprocess, build and combine dataset:

```bash
python src/preprocessing.py
```

To remove outliers in dataset:

```bash
python src/outlier_removal.py
```


### ML training

To find the right features to use in training

```bash
python src/feature_selection.py
```

To optimize hyperparameters

```bash
python src/hyperparameter_tuning.py
```

Final run of training with optimized parameters:

```bash
python src/train_eval.py
```

## Contributors

Ida Rognved: development and paper writing
Georg Erharter: development and paper writing
Tom F. Hansen: development and paper writing

