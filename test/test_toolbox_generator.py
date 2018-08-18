from unittest import TestCase
from unittest.mock import MagicMock
from app.toolbox_configuration import ToolboxConfiguration
from app.toolbox_generator import ToolboxGenerator
import random
from unittest.mock import call
from unittest.mock import patch
from unittest.mock import ANY

class TestToolboxGenerator(TestCase):
    def test_given_a_list_of_configurations_when_register_all_from_configurations_the_attributes_should_all_be_registered_on_the_toolbox(self):
        configurations = [
            ToolboxConfiguration('total_hidden_layers', random.randint, 1, 5),
            ToolboxConfiguration('beta_1', random.uniform, 0.5, 0.8),
            ToolboxConfiguration('epsilon', random.uniform, 0.1, 0.9)
        ]

        toolbox_mock = MagicMock()

        ToolboxGenerator().register_all_from_configurations(toolbox_mock, configurations)

        calls = [
            call.register('total_hidden_layers', random.randint, 1, 5),
            call.register('beta_1', random.uniform, 0.5, 0.8),
            call.register('epsilon', random.uniform, 0.1, 0.9)
        ]

        toolbox_mock.assert_has_calls(calls)

    def test_given_a_configuration_when_register_from_configuration_the_attribute_should_be_registered_on_the_toolbox(self):
        configuration = ToolboxConfiguration(
            'total_hidden_layers',
            random.randint,
            1,
            5
        )

        toolbox_mock = MagicMock()

        ToolboxGenerator().register_from_configuration(toolbox_mock, configuration)

        toolbox_mock.register.assert_called_with('total_hidden_layers', random.randint, 1, 5)

    @patch('deap.tools.initRepeat')
    @patch('deap.creator.create')
    @patch('deap.tools.initCycle')
    @patch('deap.creator')
    def test_given_attributes_to_evolve_when_initialise_individuals_the_attributes_for_individuals_should_be_registered_on_the_toolbox(self,
                individual_mock, initialization_mock, create_mock, repeat_mock):

        attributes_to_evolve = MagicMock()
        toolbox_mock = MagicMock()
        mock_fitness = MagicMock()
        individual_size = 5

        individual_mock.return_value = MagicMock(Individual='fake')

        ToolboxGenerator().initialise_individuals(toolbox_mock, attributes_to_evolve, individual_size, mock_fitness)

        calls = [
            call.register('individual', initialization_mock, ANY, attributes_to_evolve, n=5),
            call.register('population', repeat_mock, list, toolbox_mock.individual)
        ]

        toolbox_mock.assert_has_calls(calls)

    def test_given_desired_scores_when_add_fitness_to_creator_then_the_fitness_with_given_weights_should_be_registered(self):
        desired_scores = (-1.0, -1.0, -1.0, -1.0, 1.0)
        creator_mock = MagicMock()

        ToolboxGenerator().add_fitness_to_creator(creator_mock, desired_scores)

        creator_mock.create.assert_called_with("FitnessMulti", ANY, weights=desired_scores)