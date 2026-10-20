import torch

from asthunt.models.define_realbogus_cnn import define_realbogus_cnn


def test_output_has_one_score_per_stamp():
    model = define_realbogus_cnn()
    output = model(torch.zeros(4, 3, 31, 31))
    assert output.shape == (4,)
