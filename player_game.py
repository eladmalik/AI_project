import sys

import pygame

import lot_generator
from feature_extractor import Extractor, Extractor2, Extractor3, Extractor4
from reward_analyzer import Analyzer, AnalyzerStopOnTarget, AnalyzerDistanceCritical, \
    AnalyzerCollisionReduceNearTarget, AnalyzerNoCollision, AnalyzerAccumulating, AnalyzerAccumulating3, \
    AnalyzerAccumulating4, AnalyzerAccumulating5
from simulator import Simulator, DrawingMethod
from car import Car, Movement, Steering

FPS = 60
DEBUG = True

if __name__ == '__main__':
    # initializing the parking lot
    sim = Simulator(lot_generator.generate_lot, AnalyzerAccumulating5,
                    Extractor4,
                    draw_screen=True,
                    resize_screen=True,
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT)
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
        _, reward, done = sim.do_step(movement, steering, 1 / FPS)
        total_reward += reward
        if done:
            pygame.quit()
            print(f"total reward: {total_reward}")
            exit()
            # sim.reset()
            total_reward = 0

        # printing the results:
        if DEBUG:
            print(f"reward: {total_reward:.3f}, done: {done}")

        # updating the screen
        sim.update_screen()
