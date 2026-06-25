"""Tests for CNN-assisted extraction utilities."""

import numpy as np

from watermark.cnn_extraction import prepare_cnn_input_from_image, predict_payload_bits


class DummyModel:
    def predict(self, inputs, verbose=0):
        assert inputs.shape == (1, 128, 128, 2)
        return np.tile(np.array([[0.2, 0.8, 0.49, 0.51]], dtype=float), (1, 32))


def test_cnn_input_shape_from_512_rgb():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    tensor = prepare_cnn_input_from_image(image)
    assert tensor.shape == (128, 128, 2)
    assert tensor.dtype == np.float32


def test_predict_payload_bits_thresholds_sigmoid_outputs():
    cnn_input = np.zeros((128, 128, 2), dtype=np.float32)
    bits = predict_payload_bits(DummyModel(), cnn_input)
    assert bits.shape == (128,)
    assert np.array_equal(bits[:4], np.array([0, 1, 0, 1], dtype=np.uint8))

