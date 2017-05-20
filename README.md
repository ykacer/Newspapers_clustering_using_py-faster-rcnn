# Newspapers_clustering_using_py-faster-rcnn
In this repository we propose a Deep Learning approach for document clustering. This is an improvment of a [previous work](https://github.com/ykacer/Newspapers_clustering) that used traditional Machine Learning approach.

## Description
We use [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) code from R.Ghirschik to learn text and image detection.

## Dataset

The used dataset is an ensemble of 102 Russian newpapers pages annotated and 
taken from [UCL Machine learning](https://archive.ics.uci.edu/ml/machine-learning-databases/00306/) site.
This is the dataset we use to tune faster-rcnn code.

We show below an example with ground truth:

#### page

<p align="center">
  <img src="1.jpg" width="300"/>
</p>

#### ground truth

<p align="center">
  <img src="1_m.png" width="300"/>
</p>

## formatting data 
To use faster-rcnn code, we need to create xml annotation files like Pascal-VOC challenge.
But first of all, clone our fork of [py-faster-rcnn](https://github.com/ykacer/py-faster-rcnn) code into your home for example.
Then open `import_data.py`, fill `main_path` variable with following path `yourhome/py-faster-rcnn/data/NewsPapers` and run this script

`python import_data.py`
