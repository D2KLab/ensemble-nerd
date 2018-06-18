# Testing locally ensemble-nerd

## Getting Started
To be able to run and try the ensemble method on your machine, some installations steps are required.

All the application is written using Python 3.6.2.

<!-- ### Python installation

#### On Mac
Download [Python 3.6.2](https://www.python.org/ftp/python/3.6.4/python-3.6.4-macosx10.6.pkg) and install it by using the Installer application.

#### On Windows
Download [Python 3.6.2](https://www.python.org/ftp/python/3.6.4/python-3.6.4-macosx10.6.pkg) and install it by using the Windows Installer application. During this hase, pay attantion that Python is added to PATH, as in the image below
![](https://i.stack.imgur.com/CCXQG.jpg)

#### On linux
Open the Terminal and write these commands.
```
sudo apt-get update
sudo apt-get install python3.6
```

### Pip installation
After instally Python it's better to install PyPA, the recommended tool for installing Python packages. This step is not mandatory, but avoids to manually install each package required by the application. [Here](https://www.makeuseof.com/tag/install-pip-for-python/) is exaplined how to install PyPA for both Mac, Windows and Linux. -->
<!-- ```
sudo apt-get update
sudo apt-get install python3.6
``` -->

### Packages installation


Dependencies:
* Flask 0.12.2
* Cython 0.27.1
* fuzzywuzzy 0.15.1
* h5py 2.7.1
* Keras 2.0.8
* langdetect 1.0.7
* matplotlib 2.0.2
* numpy 1.14.1
* pandas 0.20.3
* scikit-learn 0.19.0
* scipy 0.19.1
* seaborn 0.8
* sklearn 0.0
* spacy 1.9.0
* igraph 0.1.11
* cysignals 1.6.8
* pyfasttext 0.4.4

Open the cloned folder and run:

```
pip3 install -r requirements.txt
pip3 install pyfasttext==0.4.4
```

> Issues in installing [pyfasttext](https://github.com/vrasneur/pyfasttext/issues/24) ?

## Download data
To be able to use the application, download [data.zip](http://enerd.eurecom.fr/data/data.zip) (14GB) and unzip it in this folder. Do not rename folders.

## Set up server
In order to locally set up the server, let's open the terminal, reach this folder and execute this command.
```
python3 server.py
```

## Create new models

### Adding a new gold standard
In order to create new ensemble models you have to train them using a gold standard.
If you have to add a new gold standard with the name <NEW_GOLD_STANDARD_NAME>, you have to follow these steps:
* create a new folder inside *data/training_data* folder called <NEW_GOLD_STANDARD_NAME>
* enter in the new folder and create two subfolders named *test* and *train*
* create inside both  *test* and *train* folders two subfolders called *csv_ground_truth* and *txt_files*
If the folders tree is correctly set up, it should appear as in the schema below:
```
data
└── training_data
    └── new_ground_truth
        ├── test
        │   ├── csv_ground_truth
        │   │   ├── document-1.csv
        │   │   ├── document-2.csv
        │   │   └── document-3.csv
        │   └── txt_files
        │       ├── document-1.txt
        │       ├── document-2.txt
        │       └── document-3.txt
        └── train
            ├── csv_ground_truth
            │   ├── document-5.csv
            │   └── document-6.csv
            └── txt_files
                ├── document-5.txt
                └── document-6.txt
```
The *txt_files* folder contains the documents used to train and test the model.
At each textual document corresponds a file in the *csv_ground_truth* folder.
Such files contain tables: each row represents a token of the related document. The table is composed by 6 columns:

1. SURFACE : the surface form related to the token
1. TYPE : the token type (in case of NoneType the cell is empty)
1. URI : the Wikidata identifier related to the token  entity (in case of the token doesn't match any entity, the cell is empty)
1. OFFSET: such column assumes 1 as value if the entity continues in the following token, otherwise 0


For example, let's assume that *document-1.txt* contains this text:
*Marvin Lee Minsky was born to an eye surgeon father, Henry, and to a Jewish mother, Fannie.*

| surface|type|uri|offset |
|:-------------:|:-------------:|:-------------:|:-------------:|
| marvin|Person|Q204815|1   |
| lee|Person|Q204815|1      |
| minsky|Person|Q204815|0   |
| was|||0                   |
| born|||0                  |
| to|||0                    |
| an|||0                    |
| eye|Role|Q774306|1        |
| surgeon|Role|Q774306|0    |
| father|Role|Q7565|0       |
| ,|||0                   |
| henry|Person||0           |
| ,|||0                   |
| and|||0                   |
| to|||0                    |
| a|||0                     |
| jewish|||0                |
| mother|Role|Q7560|0       |
| ,|||0                   |
| fannie|Person||0          |
| .|||0                   |

### Train new models
Once you correctly parsed your new gold standard, let's go in *myapp* folder and run the following command to train the model.
```
python3 train_ensemble.py <NEW_GOLD_STANDARD_NAME> --lang <NEW_GOLD_STANDARD_LANGUAGE>
```
Executing this command you'll also get the evaluation scores got by the ensemble mdoel for the new gold standard. It could also take hours depending on the number of documents presented in the ground turh.


### Evaluation
To be able to compare our method againist the state of art NED extractors, you can click on the following link to see the D2KB scores for two datasets: 
* [OKE2016](http://gerbil.aksw.org/gerbil/experiment?id=201806180000)
* [aida/CoNLL](http://gerbil.aksw.org/gerbil/experiment?id=201806180001)
