### ML Project - ElasticNet and Random Forest Approaches to Protein Abundance Prediction

#### Preliminary Information

##### Data Description
Organs like the pancreas consist of many types of tissues, which in turn consist of various cell types. Within the pancreas, we can distinguish cells unique to this organ, such as alpha and beta cells, as well as cells related to blood supply or the immune system.

The data in this task comes from multimodal single-cell RNA sequencing (scRNA-seq). Using scRNA-seq allows studying samples in high resolution and separating cells of different types. It is possible to compare pathological cells, taken from cancer patients, with healthy cells. In multimodal scRNA-seq technology, we obtain two types of readings for each cell:

- RNA transcript counts corresponding to gene expression (activity) in the given cell;
- The amount of surface proteins, directly related to the type of cell.

The result of an scRNA-seq experiment is matrices, where for each cell, RNA signals from thousands of genes (in our task, \( X \)) and signals from a selected number of surface proteins (for simplicity in this task, a single protein, CD361, \( y \)) are assigned.

According to the central dogma of biology, genetic information flows from RNA to proteins. Thus, a correlation between the amount of protein and the expression of the gene encoding it is expected. For technical and biological reasons, this relationship often degenerates. The problem in this task is to predict the surface protein signal based on gene expression. Predicting the protein abundance signal is crucial for most publicly available datasets, where only RNA matrices are accessible. Analyzing gene expression and protein abundance significantly facilitates identifying and naming cells in a sample.

The data is sourced from the bone marrow of human donors. Most of the collected cells are immune system cells. Correct identification of T lymphocytes based on both types of readings could be the basis for developing targeted cancer therapies (for those interested: CAR T cell therapy).

#### Data Retrieval
A link to the data folder for each laboratory group is available on the Moodle course page. Since each group works with data from different experiments, results may vary between groups. The data is compressed and saved in .csv format. Three files will be made available:

- `X_train.csv` and `X_test.csv`, containing RNA matrices. Each row corresponds to a cell, each column to a gene, and the values represent the expression level. The columns of these matrices are our explanatory variables.
- `y_train.csv`, corresponding to the amount of surface protein in the cells (those corresponding to data in `X_train.csv`). This is our dependent variable.

In the rest of this description, the data from `X_train.csv` and `y_train.csv` will be referred to as training data, while the data from `X_test.csv` will be called test data.

#### Submission of the Project
In the designated area for Task 2 submissions on the course’s Moodle page, the following files must be uploaded:

- A report in a Jupyter notebook (.ipynb), implementing the instructions described below (filename template: [AuthorID] report.ipynb, e.g., 123456 report.ipynb).
- The report should be structured in a way that naturally guides the reader through the solution to each task included in this project.
- Predictions on the test data (see Task 4) in the form of a .csv file, containing an `Id` column with observation numbers and an `Expected` column with prediction values (filename template: [AuthorID] prediction.csv, e.g., 123456 prediction.csv).
- We strongly recommend double-checking that the file names follow the required templates and that the .csv file is correctly prepared (two appropriately named columns, the correct number of rows corresponding to the test set).

#### Grading
The entire project is worth 30 points. Maximum points for each task are given in parentheses. The grading of Tasks 1 to 4 will consider:

- Fulfillment of the described instructions,
- Report quality, i.e., logical structure, visualizations, clarity of the text, description of results, and explanations of the undertaken actions,
- Code quality. Ensure that the code is clear and reproducible.

Additional details regarding grading criteria can be obtained from your lab instructor.

#### Note
Projects submitted after the deadline will not be graded and will receive 0 points.

---

### Tasks

#### 1. Exploration 
(a) Check how many observations and variables are contained in the loaded training and test data. Examine the variable types, and if deemed necessary, make appropriate conversions before further analysis. Ensure the data is complete.

(b) Investigate the empirical distribution of the dependent variable (provide several basic statistics, including a histogram or a density estimator plot).

(c) Select the 250 explanatory variables most correlated with the dependent variable. Compute the correlation for each pair of these variables. Illustrate the result using a heatmap.  
Note: The variable selection described here is only for this specific task. The analysis described in the following tasks should be conducted on the entire training dataset.

#### 2. ElasticNet 
The first model to train is ElasticNet, which includes ridge regression and lasso as special cases.

(a) In the report, present information about the ElasticNet model, explaining the parameters that are estimated, the optimized function, and the hyperparameters on which it depends. For which hyperparameter values do we get ridge regression, and for which lasso?

(b) Define a grid of hyperparameters based on at least three values for each hyperparameter. Ensure the grid includes configurations corresponding to ridge regression and lasso. Use cross-validation to select the appropriate hyperparameters (decide the number of cross-validation folds and justify your choice).

(c) Provide the training and validation errors of the model (the results should be averaged over all cross-validation folds).

#### 3. Random Forests 
In this section, a random forests model should be trained and its performance compared with the previously created ElasticNet model.

(a) Among the many hyperparameters characterizing the random forests model, select three different ones. Define a three-dimensional grid of hyperparameter combinations and use cross-validation to choose their optimal values for prediction. The cross-validation data split should be the same as in the ElasticNet case.

(b) Create a summary table of the results obtained by the methods in cross-validation for both considered models. (This comparison is why it is important to use the same splits.) Determine which model seems best to you (justify your choice). Include a basic reference model that assigns the mean of the dependent variable to any explanatory variable.

#### 4. Prediction on the Test Set 
This task is open-ended. Based on the training data, you should fit a chosen model and then use it to predict the values of the dependent variable in the test set. The method for selecting and constructing the model, along with the motivations behind this choice, should be described in the report. The generated predictions should be submitted in a separate file, whose format was described earlier. The number of points awarded will depend on the quality of the predictions, measured by the root mean squared error (RMSE).
