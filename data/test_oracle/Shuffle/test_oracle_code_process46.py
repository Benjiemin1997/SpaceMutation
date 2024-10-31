import torch
import unittest

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.random_shuffle import random_shuffle_weight


class TestRandomShuffleWeight(unittest.TestCase):

    def setUp(self):
        self.model = ShuffleNetV2().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Load your model weights here, for example from a checkpoint or initialization

    def test_random_shuffle_weight(self):
        original_state_dict = self.model.state_dict().copy()

        # Apply the random shuffle weight mutation
        random_shuffle_weight(self.model)

        # Check if any parameters have been altered
        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertNotEqual(original_state_dict[param_name], param.data, msg=f"Parameter {param_name} has not been shuffled.")

        # Check if the model structure has been mutated
        expected_structure = """
        (pre): Sequential(
            (0): Sequential(
                (0): Conv2d(3, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU()
            )
            (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (stage2): Sequential(
            (0): ShuffleUnit(
                (residual): Sequential(
                    (0): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1))
                    (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Identity()
                    (3): Conv2d(24, 58, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=24)
                    (4): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (5): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1))
                    (6): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (7): Identity()
                )
                (shortcut): Sequential(
                    (0): Conv2d(24, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=24)
                    (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (2): Conv2d(24, 58, kernel_size=(1, 1), stride=(1, 1))
                    (3): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (4): Identity()
                )
            )
            ...
        )
        """
        # This is a placeholder for the actual mutation check logic. You would need to traverse the model's structure and compare it against the expected structure.
        # This part should be implemented based on the specifics of ShuffleNetV2 architecture.

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
