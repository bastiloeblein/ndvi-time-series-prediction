# Time Series Prediction with Deep Extremes Minicubes


This repository is our submission for grading in the module **12-GEO-M-DS02 Spatio-temporal Data** by Dr. Guido Kraemer in the summer semester 2024.
This project uses the data from the [deep-extremes-minicubes dataset](http://data.rsc4earth.de/deep_extremes/deepextremes-minicubes/) provided by the Remote Sensing Centre for Earth System Research.
The code produced here also serves as a basis for the final paper project (_Time series prediction of NDVI based on Sentinel-2 data of mini cubes: A comparison of SARIMA, Random Forest and LSTM_) in the course "Scientific Writing and Publishing" from Prof. Miguel Mahecha at the Institute of Earth System Data Science and Remote Sensing at Leipzig University.

## Short Project Description

### Time series prediction of NDVI based on Sentinel-2 data of mini cubes: A comparison of SARIMA, Random Forest and LSTM

In Earth System Science, Earth System Data Cubes are becoming more important as a data source. As a special type, mini cubes have a higher spatiotemporal resolution. Thus, their spatial resolution is smaller (Montero et al., 2023). This makes it possible to monitor local vegetation changes (Montero Loaiza et al., 2023).
The Normalized Difference Vegetation Index (NDVI) as a key vegetation index is used to monitor the health and vitality of vegetation. Various vegetation properties can be estimated such as biomass, chlorophyll concentration in leaves, plant productivity, fractional vegetation cover, and plant stress (Huang et al., 2020). To detect shifting trends of growing seasons or in vegetation activity, long-term analyzes of NDVI data are useful (Higgins et al., 2023). 
Several models have been implemented for NDVI prediction. Traditional statistical models, such as autoregressive integrated moving average (ARIMA), have been widely used due to their simplicity and interpretability. However, these models often struggle with capturing complex, non-linear patterns in the data (Shumway et al., 2017). Machine learning models like Random Forests have shown improved performance by handling non-linearities better and being more robust against overfitting (Breiman, 2001). Recently, deep learning models, particularly Long Short-Term Memory (LSTM) networks, have demonstrated superior performance in NDVI prediction due to their ability to model intricate temporal dependencies (Cavalli et al., 2021).

While many studies have focused on predicting NDVI using large-scale data, the use of mini cubes for localized NDVI prediction is rarely explored. Thus, a comparison of different methods for modeling NDVI predictions is still missing. 
The aim of this study is to address these gaps by comparing the performance of SARIMA, Random Forest and LSTM models for NDVI prediction using Sentinel-2 data of mini cubes. By evaluating the these models, we aim to identify the most suitable approaches for high-resolution NDVI prediction.

## Content
```
├── Notebooks                   <- Directory containing all code files for the project
├── _book                       <- Directory containing all rendered files for the quarto book
├── csvs                        <- Directory containing necessarities for data_preprocessing 
├── data                        <- Directory containing testing and training data as well as predictons and fitted data
├── src/data_processing         <- Directory containing necessarities for data_preprocessing 
├── .gitignore                  <- Gitignore file  
├── .gitlab-ci.yml              <- Gitlab CI/CD pipeline
├── Dockerfile                  <- Dockerfile for pre-built Docker container
├── LSTM.h5                     <- Safed LSTM model
├── Makefile                    <- Makefile for creating the Quarto book and virtual environment
├── README.md                   <- This file
├── _quarto.yml                 <- Quarto file to create the quarto book
├── book_ndvi_prediction.zip    <- Final Quarto book as zip
├── index.qmd                   <- Index page for the quarto book
├── references.bib              <- Bibliography of references used in the Quarto book book_ndvi_prediction.zip
├── references.qmd              <- .qmd file belongig to .bib file
└── requirements.txt            <- .txt file containing all necessary packages to run the code
```

## Getting started

Detailed instructions on how to install and set up your project. 

Clone the GitLab respository:

```git clone https://git.sc.uni-leipzig.de/ss2024-12-geo-m-ds02/ndvi-time-series-prediction.git```

Navigate to the project directory

```cd ndvi-time-series-prediction```

For computation of this project we use a virtual environment. The necessary packages can be found in requirements.txt. The environment can be built using the following commands:

Create virtual environment

```python -m venv ndvi_env```

Activate the virtual environment

```source ndvi_env/bin/activate```

Load the necessary requirements

```pip install -r requirements.txt```

If necessary, then create an Ipykernel

```pip install ipykernel```

```python -m ipykernel install --user --name=ndvi_env_kernel```

## Usage

The script is divided into different sections. 

- The first notebook is _0_Introduction_ to present you the topic and research question of our project.
- _1_Data_preprocessing_ outlines the complete workflow for processing the data.
- The next Notebooks contain the code and description of each individual model:
    - _2.1_SARIMA_ 
    - _2.2_Random_Forest_ 
    - _2.3_LSTM_
- A comparison of all three models can be found in the last Notebook _3_Evaluation_.

## References

Breiman, L. (2001). Random forests. Machine learning, 45 , 5–32`

Cavalli, S., Penzotti, G., Amoretti, M., Caselli, S., et al. (2021). A machine learning approach for ndvi forecasting based on sentinel-2 data. In Icsoft (pp. 473–480).

Higgins, S. I., Conradi, T., & Muhoko, E. (2023, February). Shifts in vegetation activity of terrestrial ecosystems attributable to climate trends. Nature Geoscience, 16 (2), 147–153. Retrieved from http://dx.doi.org/10.1038/s41561-022-01114-x doi: 10.1038/s41561-022-01114-x

Huang, S., Tang, L., Hupy, J. P., Wang, Y., & Shao, G. (2020, May). A commentary review on the use of normalized difference vegetation index (ndvi) in the era of popular remote sensing. Journal of Forestry Research, 32 (1), 1–6. Retrieved from http://dx.doi.org/10.1007/s11676 -020-01155-1 doi: 10.1007/s11676-020-01155-1

Montero, D., Aybar, C., Mahecha, M. D., Martinuzzi, F., S ̈ochting, M., & Wieneke, S. (2023, April). A standardized catalogue of spectral indices to advance the use of remote sensing in earth system research. Scientific Data, 10 (1). Retrieved from http://dx.doi.org/10.1038/ s41597-023-02096-0 doi: 10.1038/s41597-023-02096-0

Montero Loaiza, D., Kraemer, G., Anghelea, A., Aybar Camacho, C., Brandt, G., Camps-Valls, G., . . . Mahecha, M. (2023, jul). Data cubes for earth system research: Challenges ahead. Retrieved from http://dx.doi.org/10.31223/X58M2V doi: 10.31223/x58m2v

Shumway, R. H., Stoffer, D. S., Shumway, R. H., & Stoffer, D. S. (2017). Arima models. Time series analysis and its applications: with R examples, 75–163.

## Acknowledgements

We would like to thank the European Space Agency for sponsoring the JupyterLab instance we used for our analysis.

## License

NDVI Time Series Predictions with SARIMA, Random Forest and LSTM using Sentinel-2 data of Minicubes © 2024 by Finja Baumer, Lioba Braun, Charlotte Göhler, Kimberly Jahn and Sebastian Löblein is licensed under Creative Commons Attribution 4.0 International.

## Authors

- Finja Baumer, fb30qaca@studserv.uni-leipzig.de, 3759991
- Lioba Braun, l.braun@studserv.uni-leipzig.de, 3763791
- Charlotte Göhler, charlotte.goehler@studserv.uni-leipzig.de, 3748735
- Kimberly Jahn, an24onuc@studserv.uni-leipzig.de, 3763197
- Sebastian Löblein, sy82leti@studserv.uni-leipzig.de, 3780681