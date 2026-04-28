import numpy as np

from privacy_filter_coreml.bioes import decode_bioes_predictions
from privacy_filter_coreml.viterbi import constrained_viterbi


ID2LABEL = {
    0: "O",
    1: "B-private_email",
    2: "I-private_email",
    3: "E-private_email",
    4: "S-secret",
}


def test_decode_bioes_with_offsets():
    text = "email alice@example.com now"
    spans = decode_bioes_predictions(
        [0, 1, 2, 3, 0],
        ID2LABEL,
        offsets=[(0, 5), (6, 11), (11, 19), (19, 23), (24, 27)],
        text=text,
    )

    assert len(spans) == 1
    assert spans[0].label == "private_email"
    assert spans[0].start == 6
    assert spans[0].end == 23
    assert spans[0].text == "alice@example.com"


def test_constrained_viterbi_repairs_invalid_argmax():
    logits = np.full((3, len(ID2LABEL)), -10.0, dtype=np.float32)
    logits[0, 2] = 10.0
    logits[1, 2] = 10.0
    logits[2, 3] = 10.0

    path = constrained_viterbi(logits, ID2LABEL)

    assert path == [1, 2, 3]
