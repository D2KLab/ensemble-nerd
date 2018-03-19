# ensemble-nerd
This repository aims to show two multilingual ensemble methods that combine the responses of web services NER and NED in order to improve the quality of the predicted entities.
Both methods represent the information got by the extractor responses as real-valued vector (features engineering) and use Deep Neural Networks to produce the final output. 
I built 5 ensemble models using the training set related to these standard golds:
* aida
* oke2015
* oke2016
* neel2015
* french subtitles corpus

## Getting Started
To be able to run and try the ensemble method, some installations steps are required.

### Python installation
All the application is written using Python 3.6.2.

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
After instally Python it's better to install PyPA, the recommended tool for installing Python packages. This step is not mandatory, but avoids to manually install each package required by the application. [Here](https://www.makeuseof.com/tag/install-pip-for-python/) is exaplined how to install PyPA for both Mac, Windows and Linux.

### Packages installation

```
sudo apt-get update
sudo apt-get install python3.6
```

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
* spacy 1.9.0# 
* igraph 0.1.11
* cysignals 1.6.8
* pyfasttext 0.4.4

Open the cloned folder and run:

```
pip3 install -r requirements.txt
pip3 install pyfasttext==0.4.4
```

## Dowbload data
To be able to use the application, let's download the *data.zip* zipped folder at [this link](https://fil.email/OV1IYgGb), unzip the folder and move it inside the *myapp* folder. Do not rename folders.
