import sys

import pygame

import lot_generator
from feature_extractor import Extractor, Extractor2
from reward_analyzer import Analyzer, AnalyzerStopOnTarget, AnalyzerDistanceCritical, \
    AnalyzerCollisionReduceNearTarget, AnalyzerNoCollision
from simulator import Simulator, Results, DrawingMethod
from parking_lot import ParkingLot
from parking_cell import ParkingCell
from obstacles import Sidewalk
from car import Car, Movement, Steering
from assets_images import AGENT_IMG, PARKING_IMG, PARKING_SIDEWALK_IMG, CAR_IMG, \
    ICON_IMG, FLOOR_IMG, PARKING_SIDEWALK_TARGET_IMG

FPS = 60
DEBUG = True

if __name__ == '__main__':
    # initializing the parking lot
    # car = Car(300, 150, 100, 50, 180, PATH_CAR_IMG)
    # cell1 = ParkingCell(300, 150, 300, 150, 0, PATH_PARKING_IMG, car)
    # cell2 = ParkingCell(300, 300, 300, 150, 0, PATH_PARKING_IMG)
    # cell3 = ParkingCell(500, 500, 300, 150, 30, PATH_PARKING_IMG, topleft=True)
    # agent_car = Car(600, 500, 100, 50, 0, PATH_AGENT_IMG)
    # lot = ParkingLot(1000, 1000, agent_car, [cell1, cell2, cell3])
    lot = lot_generator.generate_lot()
    # lot = get_example_lot()

    # initializing the simulator
    # sim = Simulator(lot, Analyzer1(), Extractor1(), drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT,
    #                 background_image=PATH_FLOOR_IMG)
    sim = Simulator(lot_generator.generate_lot, AnalyzerCollisionReduceNearTarget(),
                    Extractor2(),
                    draw_screen=True,
                    resize_screen=True,
                    drawing_method=DrawingMethod.BACKGROUND_SNAPSHOT,
                    background_image=FLOOR_IMG)
    clock = pygame.time.Clock()
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
        reward, done = sim.do_step(movement, steering, 1 / FPS)
        if done:
            sim.reset()

        # printing the results:
        if DEBUG:
            # print(
            # f"current loc: {sim.parking_lot.car_agent.location}, "
            # f" vel magnitude: "
            # f"{sim.parking_lot.car_agent.velocity.magnitude():.2f},"
            # f" acceleration: {sim.parking_lot.car_agent.acceleration:.2f}, "
            # f"collision: {results[Results.COLLISION]}, % in free cell: "
            # f"{0.0 if len(results[Results.UNOCCUPIED_PERCENTAGE]) == 0 else max(results[Results.UNOCCUPIED_PERCENTAGE].values()):.2f}")
            print(f"reward: {reward:.3f}, done: {done}")

        # updating the screen
        sim.update_screen()
