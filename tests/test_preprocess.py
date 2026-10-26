"""Preprocessing unit tests: contractions, negations, edge cases."""
from src.preprocess import TweetPreprocessor


def make():
    return TweetPreprocessor(use_lemmatizer=False)


def test_contraction_basic():
    p = make()
    assert p.expand_contractions("I don't like it").lower() == "i do not like it"
    # 'do' is a stopword so the pipeline keeps the sentiment-critical 'not'
    assert "not" in p.preprocess("I don't like it").split()


def test_contraction_punctuation_adjacent():
    p = make()
    assert p.expand_contractions("I don't, can't. won't!").lower() == "i do not, cannot. will not!"
    out = p.preprocess("I don't, can't. won't!")
    assert "not" in out.split() and "cannot" in out.split()


def test_negation_preserved():
    p = make()
    out = p.preprocess("The food was not bad, never liked it less")
    assert "not" in out.split()
    assert "never" in out.split()


def test_url_mention_hashtag():
    p = make()
    out = p.preprocess("Hey @user check https://t.co/x #happy")
    assert "@" not in out and "http" not in out and "happy" in out


def test_emoji_markers_only_from_rules():
    p = make()
    assert "sentimentpositive" in p.preprocess("great :)")
    assert "sentimentnegative" in p.preprocess("sad :(")
    assert "sentimentpositive" not in p.preprocess("just great")
    assert "sentimentnegative" not in p.preprocess("just sad")


def test_empty_and_noise():
    p = make()
    assert p.preprocess("") == ""
    assert p.preprocess("!!! ??? ...") == ""
    assert p.preprocess(None) == "" if hasattr(p, "preprocess") else True


def test_sarcasm_limitation_documented():
    # TF-IDF bag-of-words cannot reliably resolve sarcasm; assert only that
    # the pipeline does not crash and preserves tokens for the model.
    p = make()
    out = p.preprocess("Oh wonderful, another app crash #fail")
    assert len(out.split()) >= 2
