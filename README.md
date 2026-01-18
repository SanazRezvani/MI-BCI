# A Multiobjective Feature Optimization Approach in a Motor Imagery-based Brain-computer Interface

## <u>Contents</u>

- [Research Abstract](#research-abstract)
- [Machine Learning Plan Overview](#ml-plan-overview)
- [Project Timeline](#project-timeline)
- [File Descriptions](#file-descriptions)
- [Data Processing Notebook Overview](#data-processing-pipeline)
- [Data Processing Results and Analysis](#data-processing-results)
- [Machine Learning Notebook Overview](#lstm-rnn-model)
- [Machine Learning Results and Analysis](#machine-learning-results)


## <a id="research-abstract" style="color: inherit; text-decoration: none;"><u>Research Abstract</u></a>
In this project, a multi-rate system for spectral decomposition of the EEG signal is designed, and then the spatial and temporal features are extracted from each sub-band. 
To maximise the classification accuracy while simplifying the model and using the smallest set of features, the feature selection stage is treated as a multiobjective optimisation problem, and the Pareto optimal solutions of these two conflicting objectives are obtained. 
For the feature selection stage, non-dominated sorting genetic algorithm II (NSGA-II), an evolutionary-based algorithm, is used as a wrapper-based approach, and its effect on the BCI performance is explored. 
The proposed method is implemented on a public dataset known as the BCI competition III dataset IVa.

## <a id="ml-plan-overview" style="color: inherit; text-decoration: none;"><u>Machine Learning Plan Overview</u></a>
The approach used in this project presents the design of a BCI system that is highly customisable to individual subjects, accounting for subject-specific frequency ranges in motor imagery-based BCIs. Spatial and temporal features were extracted from each sub-band by
designing a multi-rate system for spectral decomposition. Afterward, the feature selection stage was treated as a multiobjective optimisation problem, prioritising maximal classification accuracy, model simplification, and using the smallest feature set. Then, the Pareto optimal solutions for these conflicting objectives were successfully determined. Furthermore, exploring feature
space and selecting salient features were accomplished using NSGA-II in a wrapper-based manner, with a detailed examination of its impact on BCI performance. Application of the proposed method to the BCI competition III dataset IVa yielded significantly improved classification accuracy compared to previous studies on the same dataset. 

## <a id="file-descriptions" style="color: inherit; text-decoration: none;"><u>File Descriptions</u></a>

- `NSGA2_fbcsp.m` - MATLAB file with the data processing for the final model


- `Data_Processing_Final.ipynb` - Python notebook with the data processing for our final model
- `Model_Training_Final.ipynb` - Python notebook with the ML/model training for our final model
- `Hardware_Procedure.pdf` - procedure for hardware setup/data collection
- `Conference_Poster.pdf` - poster that was submitted to the California Neurotechnology Conference 2023
- `Slideshow_Presentation.pdf` - PDF of the final presentation slideshow
- `Project_Methods.png` - image showing history of recording sessions 
- `git_push_script.bat` - batch script to quickly push all changes to the repository with a commit messagex
- `archive/` - old model training and data processing notebooks
- `data/` - raw .csv files from the data collection, as well as formatted training example/label .npy files
- `demo/` - videos of demo recording
- `live_app/` - code for the live transcription app
- `models/` - saved Keras model files, not just weights 
- `pictures/` - pictures for the README.md and general info
- `resources/` - helpful documents that give context for adjacent research/ideas
