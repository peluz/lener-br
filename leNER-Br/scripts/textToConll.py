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




from nltk import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import sys
import pickle

assert sys.version_info >= (3, 0)

if len(sys.argv) != 2:
    sys.exit("Usage: textToConll.py <path/to/file>")
else:
    fileName = sys.argv[1]

with open(fileName, 'r', encoding='UTF-8') as f:
    text = f.read()

punkt_param = PunktParameters()
with open("./abbrev_list.pkl", "rb") as fp:
    abbrev_list = pickle.load(fp)
punkt_param.abbrev_types = set(abbrev_list)
tokenizer = PunktSentenceTokenizer(punkt_param)
tokenizer.train(text)
print(tokenizer._params.abbrev_types)

all_sentences = tokenizer.tokenize(text)

seen = set()
sentences = []
for sentence in all_sentences:
    if sentence not in seen:
        seen.add(sentence)
        sentences.append(sentence)


output = fileName.rstrip('.txt') + '_temp.conll'

with open(output, 'w', encoding='UTF-8') as f:
    for sentence in sentences:
        words = word_tokenize(sentence, language='portuguese')
        for word in words:
            f.write("{} O\n".format(word))
        f.write("\n")
