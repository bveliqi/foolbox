import numpy as np

from foolbox.attacks import ContrastReductionAttack as Attack


def test_attack(bn_adversarial_batch):
    adv = bn_adversarial_batch
    attack = Attack()
    attack(adv)
    assert adv.image is not None
    assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_adversarial_batch):
    adv = gl_bn_adversarial_batch
    attack = Attack()
    attack(adv)
    assert adv.image is not None
    assert adv.distance.value < np.inf
