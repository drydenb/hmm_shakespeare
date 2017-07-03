# -*- coding: UTF-8 -*-


import hmm.process


def test_flatten():
    assert hmm.process.flatten([]) == []
    unflattened = [[[1], [2], [3]], [[4], [5], [6]]]
    flattened = [[1], [2], [3], [4], [5], [6]]
    assert hmm.process.flatten(unflattened) == flattened


def test_get_data():
    try:
        hmm.process.get_data()
    except Exception:
        assert False


def test_filter_data():
    data = [
        # Poem 1
        [["This", "is", "a", "poem"],
         ["containing", "gibberish", "words"],
         ["like", "oooofeiiii", "and", "NAhhhhooieqqpp"]],
        # Poem 2
        [["Another", "poemooooooohhhh"],
         ["that", "has", "random", "junk", "like", "yaiiiiooooooop"]],
    ]
    filtered = hmm.process.filter_data(data)
    filtered_poem_1 = {
        "This",
        "is",
        "a",
        "poem",
        "containing",
        "gibberish",
        "words",
        "like",
        "and",
    }
    filtered_poem_2 = {
        "Another",
        "that",
        "has",
        "random",
        "junk",
        "like",
    }
    assert set(hmm.process.flatten(filtered[0])) == filtered_poem_1
    assert set(hmm.process.flatten(filtered[1])) == filtered_poem_2


def test_get_hashes():
    data = [
        # Poem
        [["This", "poem", "has"],
         ["repeating", "words", "words", "words"],
         ["broken", "record", "broken", "record"]],
    ]
    filtered = hmm.process.flatten(hmm.process.filter_data(data))
    num_unique, id_to_token, token_to_id = hmm.process.get_hashes(filtered)
    # There are 7 unique words in the poem above, by inspection
    assert num_unique == 7
    # Test the maps explicitly
    assert id_to_token[token_to_id["This"]] == "This"
    assert id_to_token[token_to_id["poem"]] == "poem"
    assert id_to_token[token_to_id["has"]] == "has"
    assert id_to_token[token_to_id["repeating"]] == "repeating"
    assert id_to_token[token_to_id["words"]] == "words"
    assert id_to_token[token_to_id["broken"]] == "broken"
    assert id_to_token[token_to_id["record"]] == "record"


def test_create_training():
    data = [
        # Poem
        [["This", "is", "a"],
         ["nice", "poem", "with", "interesting"],
         ["content"]],
    ]
    filtered = hmm.process.flatten(hmm.process.filter_data(data))
    num_unique, id_to_token, token_to_id = hmm.process.get_hashes(filtered)
    train_data = hmm.process.create_training(filtered, token_to_id)

    # Retrieve the indices for some words in the poem
    this_idx = token_to_id["This"]
    interesting_idx = token_to_id["interesting"]

    # Indices for training data should be equivalent to their mapped indices
    assert train_data[0][0] == this_idx
    assert train_data[1][3] == interesting_idx

