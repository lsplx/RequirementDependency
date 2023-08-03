The respositiy is public package of the paper titled "Effective Test Case Selection via Requirements Dependency Identification in Function Regression Testing" submitted to ICSE 2024.

# Datasets:  
The datasets used in our study consists of two parts, i.e., dataset for requirements dependency identification and dataset for test case selection.  
The dataset for requirements dependency identification is given in the folder "data", while the dataset for test case selection is not public due to the confidence issue.

# Scripts:  
Run Requirements-Oriented Entity Extraction: phthon RequirementExtraction/app.py  
Run Requirements Dependency Identification: python predict.py

# Models:  
The Requirements-Oriented Entity Extraction Model:   
url: https://pan.baidu.com/s/1LhYncGaG6-fA6o6fHkI1dg;  password: p5r8  
The Mined Patterns for Requirements Dependency Identification: 
![image](https://github.com/lsplx/RequirementDependency/blob/master/IMG/DT49.png)

The following gives the required package for running environment.  
python == 3.6.0  
Jinja2 == 2.11.3  
numpy == 1.17.4  
tensorboardX == 1.6  
tqdm == 4.55.1  
transformers[sentencepiece] == 4.1.1  
scikit-learn == 0.24.0  
spacy == 2.2.4  
flask == 1.1.2  
flask-cors == 3.0.10  
gevent == 21.1.2  
pillow == 8.2.0  
pytorch-gpu == 1.7.0  
torchvision == 0.8.0  
en_core_web_sm == 2.2.5  
future == 0.18.2  
