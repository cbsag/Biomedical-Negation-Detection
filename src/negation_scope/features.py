"""Word-level feature extraction for the CRF baseline."""

from __future__ import annotations


def word_shape(token: str) -> str:
    """Collapse a token into a simple shape signature."""
    shape_chars = []
    for char in token:
        if char.isupper():
            shape_chars.append("X")
        elif char.islower():
            shape_chars.append("x")
        elif char.isdigit():
            shape_chars.append("d")
        else:
            shape_chars.append(char)
    return "".join(shape_chars)


def token_features(tokens: list[str], index: int) -> dict[str, object]:
    """Build a CRF feature dictionary for one token."""
    token = tokens[index]
    features: dict[str, object] = {
        "bias": 1.0,
        "token": token,
        "token.lower": token.lower(),
        "token.isupper": token.isupper(),
        "token.istitle": token.istitle(),
        "token.isdigit": token.isdigit(),
        "token.has_hyphen": "-" in token,
        "token.prefix2": token[:2],
        "token.prefix3": token[:3],
        "token.suffix2": token[-2:],
        "token.suffix3": token[-3:],
        "token.shape": word_shape(token),
        "token.length": len(token),
    }

    if index == 0:
        features["BOS"] = True
    else:
        previous = tokens[index - 1]
        features.update(
            {
                "-1:token.lower": previous.lower(),
                "-1:token.istitle": previous.istitle(),
                "-1:token.isupper": previous.isupper(),
                "-1:token.shape": word_shape(previous),
            }
        )

    if index == len(tokens) - 1:
        features["EOS"] = True
    else:
        following = tokens[index + 1]
        features.update(
            {
                "+1:token.lower": following.lower(),
                "+1:token.istitle": following.istitle(),
                "+1:token.isupper": following.isupper(),
                "+1:token.shape": word_shape(following),
            }
        )

    return features


def sentence_features(tokens: list[str]) -> list[dict[str, object]]:
    """Extract CRF features for an entire sentence."""
    return [token_features(tokens, index) for index in range(len(tokens))]

