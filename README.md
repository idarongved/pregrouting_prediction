# Machine learning modelling for decision support in pre-grouting in hard rock

The repository containts code for preprocessing, analyzing and training machine learning models to be used for decision support in pre-grouting in hard rock.

A paper was published by Rognved, Hansen and Erharhter (2023) based on the results from the code in the repo. The focus in this paper was to predict `Total grout take` in kg cement and `Grouting time` in hours.

DOI: xxx

## Setup

Install and activate environment with:

```bash
poetry install
poetry shell
```

## Run

Run the steps in the machine learning project with:

```bash
python src/preprocessing.py
python src/train_eval.py
```

## Contributors

Ida Rognved: main development and paper writing
Georg Erharter: development and paper writing
Tom F. Hansen: development and paper writing

