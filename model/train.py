# This file was used as part of the project reported in the paper below.
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


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    score = 0
    # model.restore_session(config.dir_model) # optional, restore weights
    # model.reinitialize_weights("proj")
    # make sure to make score equals models best score

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev, score)

if __name__ == "__main__":
    main()
