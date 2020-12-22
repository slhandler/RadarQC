This project focuses on leveraging machine learning models to quality control (QC) WSR-88D weather radar data. Three different algorithms are examined: Logistic regression with elastic net penalty (LR), random forest (RF), and convolutional neural network (CNN). 

Radar data comes from NCEI. Verification (targets) are provided by ASOS observations. The initial dataset (SR_dataset.csv) was generated as described below (no script provided).
    1.) Map native radar data to cartesian grid
    2.) Extract radar data from 5x5 (33x33 for CNN) box centered on grid box with ASOS information.
    3.) Spatial statistics (mean, max, min, standard deviation) are computed from 5x5 box of data for RF and LR.

The directory structure for this project is as such:
 - notebooks: Jupyter notebooks consisting of some sample exploratory data analysis from the dataset along with model verification using standard evaluation metrics.
 - scripts: Python scripts that run through all amjor steps (i.e.,hyperparameter tuning, training, final evaluation) for each of the three models. CNN has its own subdirectory with various scripts/functions
 - data: Directory where data is stored/written to.
 - models: The final trained models