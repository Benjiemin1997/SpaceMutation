import unittest

import torch

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.remove_activation import remove_activations


class TestShuffleNetV2(unittest.TestCase):

    def setUp(self):
        self.model = ShuffleNetV2()
        self.activations_removed_model = remove_activations(self.model)

    def test_model_structure_after_activation_removal(self):
        expected_model_structure = '''
        (pre): Sequential(
            (0): Conv2d(3, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (stage2): Sequential(
            (0): ShuffleUnit(
                (residual): Sequential(
                    (0): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(24, 58, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24)
                    (4): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1))
                    (6): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
                (shortcut): Sequential()
            )
            (1): ShuffleUnit(
                (shortcut): Sequential()
                (residual): Sequential(
                    (0): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(58, 58, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=58)
                    (4): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1))
                    (6): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
            )
            (2): ShuffleUnit(
                (shortcut): Sequential()
                (residual): Sequential(
                    (0): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(58, 58, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=58)
                    (4): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1))
                    (6): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
            )
            (3): ShuffleUnit(
                (shortcut): Sequential()
                (residual): Sequential(
                    (0): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(58, 58, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=58)
                    (4): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1))
                    (6): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
            )
        )
        (stage3): Sequential(
            (0): ShuffleUnit(
                (residual): Sequential(
                    (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116)
                    (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                    (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
                (shortcut): Sequential()
            )
            (1): ShuffleUnit(
                (shortcut): Sequential()
                (residual): Sequential(
                    (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116)
                    (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                    (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
            )
            (2): ShuffleUnit(
                (shortcut): Sequential()
                (residual): Sequential(
                    (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116)
                    (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                    (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
            )
            (3): ShuffleUnit(
                (shortcut): Sequential()
                (residual): Sequential(
                    (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116)
                    (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                    (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
            )
            (4): ShuffleUnit(
                (shortcut): Sequential()
                (residual): Sequential(
                    (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116)
                    (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                    (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
            )
            (5): ShuffleUnit(
                (shortcut): Sequential()
                (residual): Sequential(
                    (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116)
                    (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                    (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
            )
            (6): ShuffleUnit(
                (shortcut): Sequential()
                (residual): Sequential(
                    (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116)
                    (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Sequential(
                        (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                        (1): SELU()
                    )
                    (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
            )
            (7): ShuffleUnit(
                (shortcut): Sequential()
                (residual): Sequential(
                    (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116)
                    (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Sequential(
                        (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1))
                        (1): Identity()
                    )
                    (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
            )
        )
        (stage4): Sequential(
            (0): ShuffleUnit(
                (residual): Sequential(
                    (0): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(232, 232, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=232)
                    (4): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1))
                    (6): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
                (shortcut): Sequential()
            )
            (1): ShuffleUnit(
                (shortcut): Sequential()
                (residual): Sequential(
                    (0): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(232, 232, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=232)
                    (4): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1))
                    (6): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
            )
            (2): ShuffleUnit(
                (shortcut): Sequential()
                (residual): Sequential(
                    (0): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(232, 232, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=232)
                    (4): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Sequential(
                        (0): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1))
                        (1): Identity()
                    )
                    (6): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
            )
        )
        (conv5): Sequential(
            (0): Conv2d(464, 1024, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Identity()
        )
        (fc): Sequential(
            (0): Linear(in_features=1024, out_features=100, bias=True)
            (1): Identity()
        )
        '''
        self.assertEqual(str(self.activations_removed_model), expected_model_structure)

    def test_model_equivalence(self):
        # Check if the original and modified models produce the same output on some input data
        input_data = torch.randn(1, 3, 224, 224)
        output_original = self.model(input_data)
        output_modified = self.activations_removed_model(input_data)
        self.assertTrue(torch.allclose(output_original, output_modified, atol=1e-4))

if __name__ == '__main__':
    unittest.main()