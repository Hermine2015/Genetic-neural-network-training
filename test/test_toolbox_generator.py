from unittest import TestCase
from unittest.mock import MagicMock
from app.toolbox_configuration import ToolboxConfiguration
from app.toolbox_generator import ToolboxGenerator
import random


class TestToolboxGenerator(TestCase):
    def test_given_a_configuration_when_register_from_configuration_the_attribute_should_be_registered_on_the_toolbox(self):
        configuration = ToolboxConfiguration(
            'total_hidden_layers',
            random.randint,
            1,
            5
        )

        toolbox_mock = MagicMock()

        ToolboxGenerator().register_from_configuration(toolbox_mock, configuration)

        print('########################')
        toolbox_mock.register.assert_called_with('total_hidden_layers', random.randint, 1, 5)