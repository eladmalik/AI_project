import sys

from assets.assets_images import FLOOR_IMG
from utils import calculations
from utils.feature_extractor import Extractor7
from utils.lot_generator import *
from utils.reward_analyzer import AnalyzerAccumulating4FrontBack
from simulation.simulator import Simulator, DrawingMethod
from utils.enums import Movement, Steering

FPS = 60
DEBUG = True
if __name__ == '__main__':
    # initializing the parking lot
    sim = Simulator(generate_lot, AnalyzerAccumulating4FrontBack,
                    Extractor7,
                    draw_screen=True,
                    resize_screen=True,
                    max_iteration_time_sec=2000,
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT,
                    background_image=FLOOR_IMG)
    clock = pygame.time.Clock()
    total_reward = 0
    while True:
        # The main loop of the simulator. every iteration of this loop equals to one frame in the simulator.
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        # analyzing the user's input
        keys_pressed = pygame.key.get_pressed()
        movement = Movement.NEUTRAL
        steering = Steering.NEUTRAL
        if keys_pressed[pygame.K_w]:
            movement = Movement.FORWARD
        elif keys_pressed[pygame.K_s]:
            movement = Movement.BACKWARD
        elif keys_pressed[pygame.K_SPACE]:
            movement = Movement.BRAKE
        if keys_pressed[pygame.K_a]:
            steering = Steering.LEFT
        if keys_pressed[pygame.K_d]:
            steering = Steering.RIGHT

        # performing the input in the simulator
        _, reward, done, _ = sim.do_step(movement, steering, 1 / FPS)
        total_reward += reward
        # if done:
        #     print(f"reward: {reward}")
        #     sim.reset()
        #     total_reward = 0

        # printing the results:
        if DEBUG:
            # print(sim.agent.velocity)
            print(f"reward: {total_reward:.9f}, done: {done}")
            # print(sim.agent.velocity)
        # updating the screen
        text = {
            "Velocity": f"{sim.agent.velocity.x:.1f}",
            "Reward": f"{reward:.8f}",
            "Total Reward": f"{total_reward:.8f}",
            "Angle to target": f""
                               f""
                               f"{calculations.get_angle_to_target(sim.agent, sim.parking_lot.target_park):.1f}"
        }
        # sim.update_screen(text)
        sim.update_screen()
