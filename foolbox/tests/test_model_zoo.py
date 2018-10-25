from foolbox import zoo
import numpy as np
import foolbox
import sys
import pytest
from foolbox.zoo.model_loader import ModelLoader


@pytest.fixture(autouse=True)
def unload_foolbox_model_module():
    # reload foolbox_model from scratch for every run
    # to ensure atomic tests without side effects
    module_names = ['foolbox_model', 'model']
    for module_name in module_names:
        if module_name in sys.modules:
            del sys.modules[module_name]


test_data = [
    ('git@github.com:bethgelab/analysis-by-synthesis-model.git', (1, 28, 28)),
    ('git@github.com:bethgelab/convex_adversarial.git', (1, 28, 28)),
    ('git@github.com:bethgelab/mnist_challenge.git', 784)
]


@pytest.mark.parametrize("url, dim", test_data)
def test_loading_model(url, dim):
    # download model
    model = zoo.get_model(url)

    # create a dummy image
    x = np.zeros(dim, dtype=np.float32)
    x[:] = np.random.randn(*x.shape)

    # run the model
    logits = model.predictions(x)
    probabilities = foolbox.utils.softmax(logits)
    predicted_class = np.argmax(logits)

    # sanity check
    assert predicted_class >= 0
    assert np.sum(probabilities) >= 0.9999


def test_non_default_module_throws_error():
    with pytest.raises(RuntimeError):
        ModelLoader.get(key='other')