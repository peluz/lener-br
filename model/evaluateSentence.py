'''
Code entirely written by the leNER-Br paper authors
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




from model.ner_model import NERModel
from model.config import Config
from nltk import word_tokenize

# Pessoa is blue, tempo is green, Local is yellow and organizacao is red
bcolors = {
    "PESSOA": '\033[94m',
    "TEMPO": '\033[92m',
    "LOCAL": '\033[93m',
    "ORGANIZACAO": '\033[91m',
    "JURISPRUDENCIA": '\033[35m',
    "LEGISLACAO": '\033[36m',
    "ENDC": '\033[0m',
    "O": ""
}

# create instance of config
config = Config()

# build model
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)

while(True):
    words = input("Escreva frase a ser analisada: ")
    words = word_tokenize(words, language='portuguese')
    preds = model.predict(words)
    for index, word in enumerate(words):
        if preds[index][0:2] in ['B-', 'I-', 'E-', 'S-']:
            preds[index] = preds[index][2:]
        print(bcolors[preds[index]] +
              word + bcolors["ENDC"], end=' ')
    print('\n')
