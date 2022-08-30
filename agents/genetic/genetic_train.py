import os

from utils.csv_handler import csv_handler
from utils.enums import StatsType

if __name__ == '__main__':
    os.chdir(os.path.join("..", ".."))
import torch
from pygad import torchga
import pygad.kerasga
import pygad

import utils.general_utils
from utils.general_utils import action_mapping, dump_arguments
from assets.assets_images import FLOOR_IMG
from utils.lot_generator import *
from utils.reward_analyzer import *
from utils.feature_extractor import *
from simulation.simulator import Simulator, DrawingMethod
from agents.genetic.genetic_model import GeneticModel
from utils.plot_maker import plot_all_generation_from_lines

AGENT_TYPE = "Genetic"

torch_ga, model, observation_space_size, env = None, None, None, None
folder, train_plot = None, None
generation = 1
time_difference = None
writer = None


def fitness_func(solution, sol_idx):
    global torch_ga, model, observation_space_size, env, time_difference, generation

    model_weights_dict1 = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict1)

    # play game
    observation = env.reset()
    sum_reward = 0
    done = False
    c = 0
    reward = 0
    results = None
    while not done:
        state = torch.tensor([observation], dtype=torch.float).to(model.device)
        q_values = model(state)
        action = torch.argmax(q_values).item()
        observation_next, reward, done, results = env.do_step(action_mapping[action][0], action_mapping[
            action][1],
                                                              time_difference)
        if env.draw_screen:
            pygame.event.pump()
            env.update_screen({"Solution Number": sol_idx})
        observation = observation_next
        sum_reward += reward
        c += 1

    writer.write_row({
        StatsType.GENERATION: generation,
        StatsType.LAST_REWARD: reward,
        StatsType.TOTAL_REWARD: sum_reward,
        StatsType.DISTANCE_TO_TARGET: results[Results.DISTANCE_TO_TARGET],
        StatsType.PERCENTAGE_IN_TARGET: results[Results.PERCENTAGE_IN_TARGET],
        StatsType.ANGLE_TO_TARGET: results[Results.ANGLE_TO_TARGET],
        StatsType.SUCCESS: results[Results.SUCCESS],
        StatsType.COLLISION: results[Results.COLLISION]
    })

    return sum_reward


def callback_generation(ga_instance):
    global model, generation, writer
    generation = ga_instance.generations_completed + 1
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    model_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)
    model.save_checkpoint()
    lines = writer.get_current_data()
    plot_all_generation_from_lines(lines, folder, train_plot)
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


@dump_arguments(agent_type=AGENT_TYPE)
def main(lot_generator=generate_lot,
         reward_analyzer=AnalyzerAccumulating4FrontBack,
         feature_extractor=Extractor9,
         load_model=False,
         load_folder=None,
         time_difference_secs=0.1,
         max_iteration_time=800,
         draw_screen=True,
         resize_screen=False,
         num_generations=100,
         num_parents_mating=5,
         genes_mutation_percent=10,
         num_parents_to_keep=-1,
         plot_in_training=False,
         save_folder=None):
    global torch_ga, model, observation_space_size, env, time_difference, writer, folder, train_plot
    assert (not load_model) or (load_model and isinstance(load_folder, str))
    time_difference = time_difference_secs

    if save_folder is None:
        save_folder = utils.general_utils.get_agent_output_folder(AGENT_TYPE)
    folder = save_folder
    train_plot = plot_in_training

    writer = csv_handler(save_folder, [StatsType.LAST_REWARD,
                                       StatsType.TOTAL_REWARD,
                                       StatsType.DISTANCE_TO_TARGET,
                                       StatsType.PERCENTAGE_IN_TARGET,
                                       StatsType.ANGLE_TO_TARGET,
                                       StatsType.SUCCESS,
                                       StatsType.COLLISION,
                                       StatsType.GENERATION])

    env = Simulator(lot_generator, reward_analyzer, feature_extractor,
                    max_iteration_time_sec=max_iteration_time,
                    draw_screen=draw_screen,
                    resize_screen=resize_screen,
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT)

    observation_space_size = env.feature_extractor.input_num
    action_space_size = len(utils.general_utils.action_mapping)
    model = GeneticModel(observation_space_size, action_space_size, save_folder=save_folder)
    if load_model:
        model.change_checkpoint_dir(load_folder)
        model.load_checkpoint()
        model.change_checkpoint_dir(save_folder)

    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=15)

    # Prepare the PyGAD parameters.
    initial_population = torch_ga.population_weights  # Initial population of network weights
    parent_selection_type = "sss"  # Type of parent selection.
    crossover_type = "single_point"  # Type of the crossover operator.
    mutation_type = "random"  # Type of the mutation operator.

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=genes_mutation_percent,
                           keep_parents=num_parents_to_keep,
                           on_generation=callback_generation,
                           save_solutions=True)

    ga_instance.run()

    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_result(title="PyGAD & Pytorch - Iteration vs. Fitness", linewidth=4)
    ga_instance.plot_new_solution_rate(title="New Solution Rate")
    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    model_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)
    model.save_checkpoint()


if __name__ == '__main__':
    main(lot_generator=generate_lot,
         reward_analyzer=AnalyzerAccumulating4FrontBack,
         feature_extractor=Extractor8,
         load_model=False,
         load_folder=None,
         time_difference_secs=0.1,
         max_iteration_time=800,
         draw_screen=True,
         resize_screen=False,
         num_generations=10,
         num_parents_mating=5,
         genes_mutation_percent=10,
         num_parents_to_keep=-1)
