# STAD

Scrape Twitter And Detect


Preprocessing the dataset:
> python preProcessing.py ./../data/tweets.csv

Holdout on the dataset:
> python 1holdout_training.py

Cross-validation on the dataset:
> python crossval_training.py

Building the model:
> python STAD_training.py

Executing the Wilcoxon test:
> python wilcoxon.py

Starting the GUI:
> python gui.py
