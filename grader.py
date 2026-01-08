def grade(prediction, sample):
    expected = sample["first_sentence"]
    return expected in prediction