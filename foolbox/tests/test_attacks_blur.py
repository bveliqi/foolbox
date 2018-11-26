import numpy as np

from foolbox.attacks import GaussianBlurAttack as Attack


def test_attack(bn_adversarial_batch):
    adv = bn_adversarial_batch
    attack = Attack()
    attack(adv)
    assert adv.image is None
    assert adv.distance.value == np.inf
    # BlurAttack will fail for brightness model


def test_attack_gl(gl_bn_adversarial_batch):
    adv = gl_bn_adversarial_batch
    attack = Attack()
    attack(adv)
    assert adv.image is None
    assert adv.distance.value == np.inf
    # BlurAttack will fail for brightness model


def test_attack_trivial(bn_trivial_batch):
    adv = bn_trivial_batch
    attack = Attack()
    attack(adv)
    assert adv.image is not None
    assert adv.distance.value < np.inf
