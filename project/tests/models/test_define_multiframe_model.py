import torch

from asthunt.models.define_multiframe_model import define_multiframe_model


def test_any_number_of_frames_gives_one_score_per_tracklet():
    model = define_multiframe_model(4, feature_mean=[0, 0, 0, 0], feature_std=[1, 1, 1, 1])
    for frames in (3, 5, 55):
        output = model(torch.zeros(6, frames, 3, 21, 21), torch.zeros(6, 4))
        assert output.shape == (6,)


def test_feature_scaling_travels_with_the_weights():
    model = define_multiframe_model(2, feature_mean=[10.0, 20.0], feature_std=[2.0, 4.0])
    reloaded = define_multiframe_model(2, feature_mean=[0.0, 0.0], feature_std=[1.0, 1.0])
    reloaded.load_state_dict(model.state_dict())
    assert torch.equal(reloaded.feature_mean, torch.tensor([10.0, 20.0]))
