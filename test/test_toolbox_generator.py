from unittest import TestCase
from unittest.mock import MagicMock
from app.toolbox_configuration import ToolboxConfiguration
from app.toolbox_generator import ToolboxGenerator
import random
from unittest.mock import call
from unittest.mock import patch
from unittest.mock import ANY
from app.evolution_configuration import EvolutionConfiguration

class TestToolboxGenerator(TestCase):

    def test_given_a_list_of_configurations_when_get_toolbox_a_toolbox_should_be_registered_and_returned(self):
        evolution_configuration = self._get_evolution_configuration()

        def evaluate():
            return None

        result = ToolboxGenerator().get_toolbox(evolution_configuration, evaluate)

        self.assertIsNotNone(result.total_hidden_layers)
        self.assertIsNotNone(result.beta_1)
        self.assertIsNotNone(result.epsilon)
        self.assertIsNotNone(result.individual)
        self.assertIsNotNone(result.population)
        self.assertIsNotNone(result.mutate)
        self.assertIsNotNone(result.mate)
        self.assertIsNotNone(result.select)
        self.assertIsNotNone(result.evaluate)

    def test_given_a_list_of_configurations_when_register_all_from_configurations_the_attributes_should_all_be_registered_on_the_toolbox(self):
        configurations = [
            ToolboxConfiguration('total_hidden_layers', random.randint, 1, 5),
            ToolboxConfiguration('beta_1', random.uniform, 0.5, 0.8),
            ToolboxConfiguration('epsilon', random.uniform, 0.1, 0.9)
        ]

        toolbox_mock = MagicMock()

        ToolboxGenerator()._register_all_from_configurations(toolbox_mock, configurations)

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

        ToolboxGenerator()._register_from_configuration(toolbox_mock, configuration)

        toolbox_mock.register.assert_called_with('total_hidden_layers', random.randint, 1, 5)

    @patch('app.toolbox_generator.ToolboxGenerator._get_attributes_to_evolve_from_configurations')
    @patch('deap.tools.initRepeat')
    @patch('deap.creator.create')
    @patch('deap.tools.initCycle')
    @patch('deap.creator')
    def test_given_attributes_to_evolve_when_initialise_individuals_the_attributes_for_individuals_should_be_registered_on_the_toolbox(self,
                individual_mock, initialization_mock, create_mock, repeat_mock, get_attributes_mock):

        configurations = [
            ToolboxConfiguration('total_hidden_layers', random.randint, 1, 5),
            ToolboxConfiguration('beta_1', random.uniform, 0.5, 0.8),
            ToolboxConfiguration('epsilon', random.uniform, 0.1, 0.9)
        ]

        get_attributes_mock.return_value = ( 'fake 1', 'fake 2', 'fake 3' )

        toolbox_mock = MagicMock()
        mock_fitness = MagicMock()

        individual_mock.return_value = MagicMock(Individual='fake')

        ToolboxGenerator()._initialise_individuals(toolbox_mock, configurations, mock_fitness)

        calls = [
            call.register('individual', initialization_mock, ANY, ( 'fake 1', 'fake 2', 'fake 3' ), n=3),
            call.register('population', repeat_mock, list, toolbox_mock.individual)
        ]

        toolbox_mock.assert_has_calls(calls)

    def test_given_desired_scores_when_add_fitness_to_creator_then_the_fitness_with_given_weights_should_be_registered(self):
        desired_scores = (-1.0, -1.0, -1.0, -1.0, 1.0)
        creator_mock = MagicMock()

        ToolboxGenerator()._add_fitness_to_creator(creator_mock, desired_scores)

        creator_mock.create.assert_called_with("FitnessMulti", ANY, weights=desired_scores)

    def test_given_configurations_when_get_attributes_to_evolve_from_configurations_then_the_attributes_should_be_returned_as_a_tuple(self):
        configurations = [
            ToolboxConfiguration('total_hidden_layers', random.randint, 1, 5),
            ToolboxConfiguration('beta_1', random.uniform, 0.5, 0.8),
            ToolboxConfiguration('epsilon', random.uniform, 0.1, 0.9)
        ]

        toolbox_mock = MagicMock()
        toolbox_mock.total_hidden_layers = 'fake layers'
        toolbox_mock.beta_1 = 'fake beta'
        toolbox_mock.epsilon = 'fake epsilon'

        result = ToolboxGenerator()._get_attributes_to_evolve_from_configurations(toolbox_mock, configurations)

        self.assertEqual(('fake layers', 'fake beta', 'fake epsilon'), result)

    @patch("deap.tools.mutGaussian")
    def test_given_an_evolution_configuration_with_gaussian_mutation_when_register_mutation_then_the_gaussian_mutation_should_be_registered(
            self, mutation_mock):
        evolutionary_configuration = self._get_evolution_configuration()

        toolbox_mock = MagicMock()

        ToolboxGenerator()._register_mutation(toolbox_mock, evolutionary_configuration)

        toolbox_mock.register.assert_called_with("mutate", mutation_mock, mu=0.0, sigma=0.2, indpb=0.2)

    @patch("deap.tools.mutFlipBit")
    def test_given_an_evolution_configuration_with_flip_bit_mutation_when_register_mutation_then_the_flip_bit_mutation_should_be_registered(
            self, mutation_mock):
        evolutionary_configuration = self._get_evolution_configuration()
        evolutionary_configuration.mutation["name"] = "Flip-bit"
        evolutionary_configuration.mutation["indpb"] = 0.3

        toolbox_mock = MagicMock()

        ToolboxGenerator()._register_mutation(toolbox_mock, evolutionary_configuration)

        toolbox_mock.register.assert_called_with("mutate", mutation_mock, indpb=0.3)

    @patch("deap.tools.mutFlipBit")
    def test_given_an_evolution_configuration_with_unsupported_mutation_when_register_mutation_then_an_exception_should_be_thrown(
            self, mutation_mock):
        evolutionary_configuration = self._get_evolution_configuration()
        evolutionary_configuration.mutation["name"] = "Radiation"

        toolbox_mock = MagicMock()

        with self.assertRaises(Exception) as context:
            ToolboxGenerator()._register_mutation(toolbox_mock, evolutionary_configuration)
            self.assertTrue('Unsupported mutation type: Radiation' in context.exception)

    @patch("deap.tools.cxTwoPoint")
    def test_given_an_evolution_configuration_with_two_point_crossover_when_register_crossover_then_the_two_point_crossover_should_be_registered(
            self, crossover_mock):
        evolutionary_configuration = self._get_evolution_configuration()

        toolbox_mock = MagicMock()

        ToolboxGenerator()._register_crossover(toolbox_mock, evolutionary_configuration)

        toolbox_mock.register.assert_called_with("mate", crossover_mock)

    @patch("deap.tools.cxTwoPoint")
    def test_given_an_evolution_configuration_with_unknown_crossover_when_register_crossover_then_an_exception_should_be_thrown(
            self, crossover_mock):
        evolutionary_configuration = self._get_evolution_configuration()
        evolutionary_configuration.crossover["name"] = "Spring"

        toolbox_mock = MagicMock()

        with self.assertRaises(Exception) as context:
            ToolboxGenerator()._register_crossover(toolbox_mock, evolutionary_configuration)
            self.assertTrue('Unsupported crossover type: Spring' in context.exception)

    @patch("deap.tools.selTournament")
    def test_given_an_evolution_configuration_with_tournament_selection_when_register_selection_then_the_tournament_selection_should_be_registered(
            self, selection_mock):
        evolutionary_configuration = self._get_evolution_configuration()

        toolbox_mock = MagicMock()

        ToolboxGenerator()._register_selection(toolbox_mock, evolutionary_configuration)

        toolbox_mock.register.assert_called_with("select", selection_mock, tournsize=3)

    @patch("deap.tools.selTournament")
    def test_given_an_evolution_configuration_with_unknown_selection_when_register_selection_then_an_exception_should_be_thrown(
            self, selection_mock):
        evolutionary_configuration = self._get_evolution_configuration()
        evolutionary_configuration.selection["name"] = "Privileged"

        toolbox_mock = MagicMock()

        with self.assertRaises(Exception) as context:
            ToolboxGenerator()._register_selection(toolbox_mock, evolutionary_configuration)
            self.assertTrue('Unsupported selection type: Privileged' in context.exception)

    def test_given_an_evaluation_function_when_register_evaluation_then_the_evaluation_should_be_registered(
            self):
        toolbox_mock = MagicMock()
        def evaluate(): return None

        ToolboxGenerator()._register_evaluation(toolbox_mock, evaluate)

        toolbox_mock.register.assert_called_with("evaluate", evaluate)

    def _get_evolution_configuration(self):
        configurations = [
            ToolboxConfiguration('total_hidden_layers', random.randint, 1, 5),
            ToolboxConfiguration('beta_1', random.uniform, 0.5, 0.8),
            ToolboxConfiguration('epsilon', random.uniform, 0.1, 0.9)
        ]

        return EvolutionConfiguration(
            configurations,
            [{"name": "accuracy", "minimize": False}],
            {
                "name": "Gaussian",
                "mu": 0.0,
                "sigma": 0.2,
                "indpb": 0.2,
                "probability": 0.3
            },
            {
                "name": "Two-point",
                "probability": 0.5
            },
            {
                "name": "Tournament",
                "tournament-size": 3
            }
        )