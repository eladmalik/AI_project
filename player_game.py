import math
import sys

import pygame

import training_utils.lot_generator as lot_generator
from training_utils.feature_extractor import Extractor, Extractor2, Extractor3, Extractor4, ExtractorNew
from training_utils.reward_analyzer import Analyzer, AnalyzerStopOnTarget, AnalyzerDistanceCritical, \
    AnalyzerCollisionReduceNearTarget, AnalyzerNoCollision, AnalyzerAccumulating, AnalyzerAccumulating3, \
    AnalyzerAccumulating4, AnalyzerAccumulating5, AnalyzerAccumulating6, AnalyzerNew
from sim.simulator import Simulator, DrawingMethod
from sim.car import Car, Movement, Steering

FPS = 60
DEBUG = True

if __name__ == '__main__':
    # initializing the parking lot
    sim = Simulator(lot_generator.generate_only_target, AnalyzerNew,
                    ExtractorNew,
                    draw_screen=True,
                    resize_screen=True,
                    max_iteration_time_sec=2000,
                    drawing_method=DrawingMethod.FULL)
    clock = pygame.time.Clock()
    total_reward = 0
    extractor = ExtractorNew()
    i = 0
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
        _, reward, done = sim.do_step(movement, steering, 0.02)#1 / FPS)
        total_reward += reward
        # if done:
        #     print(f"reward: {reward}")
        #     input()
        #     sim.reset()
        #     total_reward = 0
        #     pass

        # printing the results:
        if DEBUG and i % 60 == 0:
            print(f"reward: {reward:.3f}, done: {done}")
            # distance_to_target, relative_rotation, \
            #     angle_left, angle_right, velocity, str_left, str_right, velocity, steering, *sensors = extractor.get_state(sim.parking_lot)
            # print(sensors[8:])

        # updating the screen
        sim.update_screen()
        i += 1
