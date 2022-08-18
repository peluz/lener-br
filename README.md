# LeNER-Br: a Dataset for Named Entity Recognition in Brazilian Legal Text

This repo holds the dataset and source code described in the paper below, which was generated as a collaboration between two institutions of the University of Brasília: [NEXT (Núcleo de P&D para Excelência e Transformação do Setor Público)](http://next.unb.br/) and [CiC (Departamento de Ciência da Computação)](https://cic.unb.br/).

* [Pedro H. Luz de Araujo](http://lattes.cnpq.br/8374005378743328), [Teófilo E. de Campos](https://teodecampos.github.io/), [Renato R. R. de Oliveira](http://lattes.cnpq.br/8445622450972512), [Matheus Stauffer](http://lattes.cnpq.br/3634456971616689), [Samuel Couto](http://lattes.cnpq.br/1096145820609591) and [Paulo Bermejo](http://lattes.cnpq.br/9012704117180126)  
[LeNER-Br: a Dataset for Named Entity Recognition in Brazilian Legal Text](https://teodecampos.github.io/luz_etal_propor2018.pdf)  
[International Conference on the Computational Processing of Portuguese (PROPOR), September 24-26, Canela, Brazil, 2018](http://www.inf.ufrgs.br/propor-2018/).  
Download PDFs of the [paper](https://teodecampos.github.io/LeNER-Br/luz_etal_propor2018.pdf) and [slides](https://teodecampos.github.io/LeNER-Br/luz_etal_propor2018_slides.pdf).

```
@InProceedings{luz_etal_propor2018,
          author = {Pedro H. {Luz de Araujo} and Te\'{o}filo E. {de Campos} and
                    Renato R. R. {de Oliveira} and Matheus Stauffer and
                    Samuel Couto and Paulo Bermejo},
          title = {{LeNER-Br}: a Dataset for Named Entity Recognition in {Brazilian} Legal Text},
          booktitle = {International Conference on the Computational Processing of Portuguese
                       ({PROPOR})},
	  publisher = {Springer},
	  series = {Lecture Notes on Computer Science ({LNCS})},
	  pages = {313--323},
          year = {2018},
          month = {September 24-26},
          address = {Canela, RS, Brazil},	  
	  doi = {10.1007/978-3-319-99722-3_32},
	  url = {https://teodecampos.github.io/LeNER-Br/},
}	  
```

We also provide the LSTM-CRF model described in the paper, which achieved an average f1-score of 92.53% (token) and 86.61% (entity) on the test set. 

The sections below describe the requirements and the dataset and model files.

We kindly request that users cite our paper in any publication that is generated as a result of the use of our source code, our dataset or our pre-trained models.

**Note**: although this GitHub repository was created in May 2020 to increase the visibility of this project, the dataset and source code has been available from the [site of the authors](https://teodecampos.github.io/LeNER-Br/) since September 2018.

## Requirements
1. [Python 3.6](https://www.python.org/downloads/)	
3. [pip](https://pip.pypa.io/en/stable/installing/)

## LeNER-Br Dataset

The directory structure is as follows:
* the train, test and dev folders hold space separated text files where the first column are the words and the second column are the correspondent named entity tags. Sentences are separeted by empty lines. In addition, each folder has a file that is the concatenation of all the other conll files of the same folder (train.conll, dev.conll and test.conll).
* metadata holds json files with additional information from each annotated document.
* raw_text holds the source txt files that originated the conll files.
* scripts hold an abbreviation list used for sentence segmentation and the script that generated the conll files. To use the script:
```
python textToConll.py path/to/txtfile
```


## Model

The model code is adapted from [this repo](https://github.com/guillaumegenthial/sequence_tagging) and implements a NER model using Tensorflow (LSTM + CRF + chars embeddings). All code files modified are marked as such at the beginning.
The section below summarizes the use of the model. For more in depth explanations of how to use the model and change its configurations refer to the README of the original implementation.

### Evaluation

* To install the required python packages, run from the model folder:
```
pip install -r requirements.txt
```

* To obtain the f1 scores (per token) for each class on each part of the dataset:
```
python classScores.py train
python classScores.py dev
python classScores.py test
```

* To obtain the f1 scores (per entity) for each class on each part of the dataset:
```
python evaluate.py train
python evaluate.py dev
python evaluate.py test
```

* To tag a raw text file:
```
python evaluateText path/to/txtfile
```

* To tag sentences in a interactive way:
```
python evaluate.py
```
or
```
python evaluateSentence.py
```

* To retrain the model from scratch:
```
python train.py
```
