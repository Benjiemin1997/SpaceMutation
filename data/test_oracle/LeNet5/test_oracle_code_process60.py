import unittest
from unittest.mock import patch, Mock

from mut.replace_activation import replace_activations


class TestReplaceActivations(unittest.TestCase):


    def test_replace_activations(self, *args):
        # Prepare a mock model
        model_mock = Mock()
        # Call your function under test
        result_model = replace_activations(model_mock)

        self.assertEqual(len(result_model.modules()), len(model_mock.modules()))

        for i, (old_module, new_module) in enumerate(zip(model_mock.modules(), result_model.modules())):
            if old_module != new_module:
                self.assertNotEqual(old_module, new_module)
                self.assertIsInstance(new_module, type(args[i*2+1]()))

if __name__ == '__main__':
    unittest.main()