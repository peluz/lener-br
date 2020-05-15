'''
Code entirely written by the authors of leNER-Br paper
'''

# This file was developed as part of the project reported in the paper below.
# We kindly request that users cite our paper in any publication that is 
# generated as a result of the use of our source code or our dataset.
# 
# Pedro H. Luz de Araujo, Teófilo E. de Campos, Renato R. R. de Oliveira, Matheus Stauffer, Samuel Couto and Paulo Bermejo.
# LeNER-Br: a Dataset for Named Entity Recognition in Brazilian Legal Text.
# International Conference on the Computational Processing of Portuguese (PROPOR),
# September 24-26, Canela, Brazil, 2018. 
#
#    @InProceedings{luz_etal_propor2018,
#          author = {Pedro H. {Luz de Araujo} and Te\'{o}filo E. {de Campos} and
#          Renato R. R. {de Oliveira} and Matheus Stauffer and
#          Samuel Couto and Paulo Bermejo},
#          title = {LeNER-Br: a Dataset for Named Entity Recognition in Brazilian Legal Text},
#          booktitle = {International Conference on the Computational Processing of Portuguese
#          ({PROPOR})},
#          year = {2018},
#          month = {September 24-26},
#          address = {Canela, RS, Brazil},
#          note = {Available from \url{https://cic.unb.br/~teodecampos/LeNER-Br/}}
#      }      





from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
from model.data_utils import minibatches, pad_sequences, get_chunks, create_tag_dict
from sklearn.metrics import classification_report, f1_score
import numpy as np
import sys


def classScores(model, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["f1_<label>"] = 98.4, ...

        """
        preds = []
        labels = []
        for words, labs in minibatches(test, model.config.batch_size):
            labels_pred, sequence_lengths = model.predict_batch(words)

            for lab, lab_pred, length in zip(labs, labels_pred,
                                             sequence_lengths):
                lab_pred = lab_pred[:length]
                lab = lab[:length]
                preds.append(lab_pred)
                labels.append(lab)

        return preds, labels


def main(dataset, config):

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # index to tag dic
    indxToTag = create_tag_dict("./data/tags.txt")

    preds, labels = classScores(model, dataset)
    preds = [indxToTag[item] for items in preds for item in items]
    labels = [indxToTag[item] for items in labels for item in items]
    model.logger.info("Results on {} set".format(dataset.filename))
    model.logger.info(classification_report(labels, preds,
          labels=['JURISPRUDENCIA', 'LOCAL', 'TEMPO',  'PESSOA', 'LEGISLACAO', 'ORGANIZACAO'], digits=4))
    model.logger.info(f1_score(labels, preds, average='micro', labels=['JURISPRUDENCIA', 'LOCAL', 'TEMPO',  'PESSOA', 'LEGISLACAO', 'ORGANIZACAO']))

if len(sys.argv) != 2 or sys.argv[1] not in ["train", "test", "dev"]:
    print("Usage: python classScores.py <train or test or dev>")
    sys.exit(0)

# create instance of config
config = Config()

dataset = sys.argv[1]
if dataset == "train":
    dataset = CoNLLDataset(config.filename_train, config.processing_word,
                           config.processing_tag, config.max_iter)
elif dataset == "dev":
    dataset = CoNLLDataset(config.filename_dev, config.processing_word,
                           config.processing_tag, config.max_iter)
else:
    dataset = CoNLLDataset(config.filename_test, config.processing_word,
                           config.processing_tag, config.max_iter)


if __name__ == "__main__":
    main(dataset, config)
