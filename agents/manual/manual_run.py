import sys

from assets.assets_images import FLOOR_IMG
from utils.calculations import get_angle_to_target
from utils.feature_extractor import *
from utils.lot_generator import *
from utils.reward_analyzer import AnalyzerAccumulating4FrontBack
from simulation.simulator import Simulator, DrawingMethod
from utils.enums import Movement, Steering, Results


def main(
        lot_generator: LotGenerator = generate_lot,
        max_iteration_time: int = 2000,
        FPS: int = 60,
        debug: bool = False):
    # initializing the parking lot
    sim = Simulator(lot_generator, AnalyzerAccumulating4FrontBack,
                    Extractor9,
                    draw_screen=True,
                    resize_screen=True,
                    max_iteration_time_sec=max_iteration_time,
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
        if keys_pressed[pygame.K_w] or keys_pressed[pygame.K_UP]:
            movement = Movement.FORWARD
        elif keys_pressed[pygame.K_s] or keys_pressed[pygame.K_DOWN]:
            movement = Movement.BACKWARD
        elif keys_pressed[pygame.K_SPACE]:
            movement = Movement.BRAKE
        if keys_pressed[pygame.K_a] or keys_pressed[pygame.K_LEFT]:
            steering = Steering.LEFT
        if keys_pressed[pygame.K_d] or keys_pressed[pygame.K_RIGHT]:
            steering = Steering.RIGHT

        # performing the input in the simulator
        _, reward, done, results = sim.do_step(movement, steering, 1 / FPS)
        total_reward += reward
        # printing the results:
        if debug:
            print(f"reward: {total_reward:.9f}, done: {done}")
        if done:
            print(f"total reward: {total_reward}")
            sim.reset()
            total_reward = 0
        if debug:
            text = {
                "Velocity": f"{sim.agent.velocity.x:.1f}",
                "Reward": f"{reward:.8f}",
                "Total Reward": f"{total_reward:.8f}",
                "Angle to target": f""
                                   f"{get_angle_to_target(sim.agent, sim.parking_lot.target_park):.1f}",
                "Percentage in target": f"{results[Results.PERCENTAGE_IN_TARGET]:.4f}"
            }
            sim.update_screen(text)
        else:
            sim.update_screen()
