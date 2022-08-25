import os

import torch
from pygad import torchga
import pygad.kerasga
import pygad

import utils
from assets_images import FLOOR_IMG
from car import Movement, Steering
from lot_generator import *
from reward_analyzer import *
from feature_extractor import *
from simulator import Simulator, DrawingMethod
from genetic_model import GeneticModel

TIME_SECS = 0.1

action_mapping = {
    0: (Movement.NEUTRAL, Steering.NEUTRAL),
    1: (Movement.NEUTRAL, Steering.LEFT),
    2: (Movement.NEUTRAL, Steering.RIGHT),
    3: (Movement.FORWARD, Steering.NEUTRAL),
    4: (Movement.FORWARD, Steering.LEFT),
    5: (Movement.FORWARD, Steering.RIGHT),
    6: (Movement.BACKWARD, Steering.NEUTRAL),
    7: (Movement.BACKWARD, Steering.LEFT),
    8: (Movement.BACKWARD, Steering.RIGHT),
    9: (Movement.BRAKE, Steering.NEUTRAL),
    10: (Movement.BRAKE, Steering.LEFT),
    11: (Movement.BRAKE, Steering.RIGHT)
}


def fitness_func(solution, sol_idx):
    global torch_ga, model, observation_space_size, env

    model_weights_dict1 = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict1)

    # play game
    observation = env.reset()
    sum_reward = 0
    done = False
    c = 0
    while (not done) and c < 1000:
        state = torch.tensor([observation], dtype=torch.float).to(model.device)
        q_values = model(state)
        action = torch.argmax(q_values).item()
        observation_next, reward, done, _ = env.do_step(action_mapping[action][0], action_mapping[action][1],
                                                     TIME_SECS)
        if env.draw_screen:
            pygame.event.pump()
            env.update_screen({"Solution Number": sol_idx})
        observation = observation_next
        sum_reward += reward
        c += 1

    return sum_reward


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


def get_agent_output_folder():
    folder = os.path.join("model", f'Genetic_{utils.get_time()}')
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


if __name__ == '__main__':
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LOAD MODEL HERE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    load_model = False
    model_folder = os.path.join("model", "PPO_20-08-2022__15-03-47")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CHANGE HYPER-PARAMETERS HERE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    lot_generator = example2
    reward_analyzer = AnalyzerAccumulating4
    feature_extractor = Extractor4
    time_difference_secs = 0.1
    max_iteration_time = 60
    draw_screen = True
    draw_rate = 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    env = Simulator(lot_generator, reward_analyzer, feature_extractor,
                    max_iteration_time_sec=max_iteration_time,
                    draw_screen=draw_screen,
                    resize_screen=False,
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT,
                    background_image=FLOOR_IMG)

    observation_space_size = env.feature_extractor.input_num
    action_space_size = 12
    folder = get_agent_output_folder()
    model = GeneticModel(observation_space_size, action_space_size, chkpt_dir=folder)
    if load_model:
        model.change_checkpoint_dir(model_folder)
        model.load_checkpoint()
        model.change_checkpoint_dir(folder)

    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=15)

    # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
    num_generations = 100  # Number of generations.
    num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.
    initial_population = torch_ga.population_weights  # Initial population of network weights
    parent_selection_type = "sss"  # Type of parent selection.
    crossover_type = "single_point"  # Type of the crossover operator.
    mutation_type = "random"  # Type of the mutation operator.
    mutation_percent_genes = 10  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
    keep_parents = -1  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           keep_parents=keep_parents,
                           on_generation=callback_generation)

    ga_instance.run()

    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_result(title="PyGAD & Pytorch - Iteration vs. Fitness", linewidth=4)

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    model_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)
    model.save_checkpoint()
