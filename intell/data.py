

class Word:
    """ Store the number of occurrences and correct responses to a word, grouped by utterance length
    """
    def __init__(self, word, phone):
        self.word = word
        self.phone = phone
        self.utterance = [0, 0, 0, 0, 0, 0, 0]
        self.correct = [0, 0, 0, 0, 0, 0, 0]

    def update(self, utterance_length, correct):
        try:
            self.utterance[utterance_length - 1] += 1
            if correct:
                self.correct[utterance_length - 1] += 1
        except IndexError:
            while len(self.utterance) < utterance_length:
                self.utterance.append(0)
                self.correct.append(0)
            self.update(utterance_length, correct)  # try again

