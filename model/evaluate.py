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


def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned



def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)

    # evaluate and interact
    model.evaluate(test)
    interactive_shell(model)


if __name__ == "__main__":
    main()
