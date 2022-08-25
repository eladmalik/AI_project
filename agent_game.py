import os
import sys

from utils.lot_generator import *
from utils.feature_extractor import *
from utils.reward_analyzer import *
from simulation.simulator import Simulator, DrawingMethod
from agents.ppo_lstm.ppo_lstm_model import PPO_Agent
from utils.general_utils import action_mapping

FPS = 30
DEBUG = False

lot_generator = generate_lot
reward_analyzer = AnalyzerAccumulating4FrontBack
feature_extractor = Extractor8
# time_difference_secs = 0.03333333
time_difference_secs = 0.1
max_iteration_time = 200
draw_screen = True
draw_rate = 1

if __name__ == '__main__':
    # initializing the parking lot
    sim = Simulator(lot_generator, reward_analyzer,
                    feature_extractor,
                    draw_screen=True,
                    resize_screen=False,
                    max_iteration_time_sec=60,
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT)
    clock = pygame.time.Clock()
    total_reward = 0
    successful_parks = 0
    in_parking = 0
    simulations = 0

    # agent loading
    # PPO
    load_folder = os.path.join("model", "PPO_LSTM2_25-08-2022__01-30-21")
    agent = PPO_Agent(feature_extractor.input_num, 12, load_folder)
    agent.load()
    hidden = agent.get_init_hidden()

    # DQN
    # model = DQNAgent3(feature_extractor.input_num, 12, 128, 128, 128, 128, 128)
    # model.load_state_dict(torch.load(
    #     os.path.join("model", "DQNAgent3_17-08-2022__17-37-56.pth",
    #                  "DQNAgent3_17-08-2022__17-37-56.pth")))
    while True:
        # The main loop of the simulator. every iteration of this loop equals to one frame in the simulator.
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        # analyzing the agent's input
        action, hidden = agent.get_action(sim.get_state(), hidden)
        # state = sim.get_state()
        # state0 = torch.tensor(state, dtype=torch.float)
        # with torch.no_grad():
        #     prediction = model(state0)
        # action = torch.argmax(prediction).item()

        movement, steering = action_mapping[action]

        # performing the input in the simulator
        _, reward, done, results = sim.do_step(movement, steering, 1 / FPS)
        total_reward += reward
        if done:
            if results[Results.PERCENTAGE_IN_TARGET] >= 0.93 and sim.agent.velocity.magnitude() == 0:
                successful_parks += 1
            if results[Results.PERCENTAGE_IN_TARGET] >= 1:
                in_parking += 1
            print(f"simulation {simulations}, reward: {total_reward}, success_rate: "
                  f"{(successful_parks / (simulations + 1)):.3f}")

            simulations += 1
            sim.reset()
            total_reward = 0

        # printing the results:
        if DEBUG:
            # print(sim.agent.velocity)
            print(f"reward: {total_reward:.9f}, done: {done}")
            # print(sim.agent.velocity)
        # updating the screen
        text = {
            "Simulation": f"{simulations + 1}",
            "Velocity": f"{sim.agent.velocity.x:.1f}",
            "Reward": f"{reward:.8f}",
            "Total Reward": f"{total_reward:.8f}",
            "Angle to target": f"{calculations.get_angle_to_target(sim.agent, sim.parking_lot.target_park):.1f}",
            "Success Rate": f"{(successful_parks / (simulations + 1)):.3f}",
            "In Parking Rate": f"{(in_parking / (simulations + 1)):.3f}"
        }
        sim.update_screen(text)
