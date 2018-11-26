import numpy as np
import pytest

from foolbox.attacks import PrecomputedImagesAttack as Attack


def test_attack(bn_adversarial_batch):
    adv = bn_adversarial_batch

    input_images = adv.original_image
    output_images = np.zeros_like(input_images)

    attack = Attack(input_images, output_images)

    attack(adv)

    assert adv.image is not None
    assert adv.distance.value < np.inf


def test_unknown_image(bn_adversarial_batch):
    adv = bn_adversarial_batch

    images = adv.original_image
    input_images = np.zeros_like(images)
    output_images = np.zeros_like(input_images)

    attack = Attack(input_images, output_images)

    with pytest.raises(ValueError):
        attack(adv)

    assert adv.image is None
    assert adv.distance.value == np.inf
