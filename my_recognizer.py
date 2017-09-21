import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # Implement the recognizer
    Xlengths = test_set.get_all_Xlengths()
    for seq in test_set.get_all_sequences():

        best_logL = float("-inf")
        best_guess = None
        probability_dict = {}
        X, lengths = Xlengths[seq]

        for word, model in models.items():
            try:
                logL = model.score(X, lengths)
            except:
                logL = float("-inf")

            probability_dict[word] = logL
            if logL > best_logL:
                best_logL = logL
                best_guess = word


        probabilities.append(probability_dict)
        guesses.append(best_guess)

    # return probabilities, guesses
    return probabilities, guesses
