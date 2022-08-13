import sys

import pygame

import lot_generator
from feature_extractor import Extractor1
from reward_analyzer import Analyzer1
from simulator import Simulator, Results, DrawingMethod
from parking_lot import ParkingLot
from parking_cell import ParkingCell
from obstacles import Sidewalk
from car import Car, Movement, Steering
from assets_paths import PATH_AGENT_IMG, PATH_PARKING_IMG, PATH_PARKING_SIDEWALK_IMG, PATH_CAR_IMG, \
    PATH_ICON_IMG, PATH_FLOOR_IMG

FPS = 60
DEBUG = True


def get_example_lot():
    sidewalk_left = Sidewalk(900, 0, 100, 1000, 0, topleft=True)
    sidewalk_right = Sidewalk(0, 0, 100, 1000, 0, topleft=True)
    agent = Car(500, 500, 100, 50, 0, PATH_AGENT_IMG)
    parking_cells = [
        ParkingCell(100, 0, 130, 65, 90, PATH_PARKING_SIDEWALK_IMG, topleft=True),
        ParkingCell(100, 130, 130, 65, 90, PATH_PARKING_SIDEWALK_IMG, topleft=True),
        ParkingCell(100, 260, 130, 65, 90, PATH_PARKING_SIDEWALK_IMG, topleft=True),
        ParkingCell(100, 390, 130, 65, 90, PATH_PARKING_SIDEWALK_IMG, topleft=True).place_car(100, 50,
                                                                                              PATH_CAR_IMG),
        ParkingCell(100, 520, 130, 65, 90, PATH_PARKING_SIDEWALK_IMG, topleft=True),
        ParkingCell(100, 650, 130, 65, 90, PATH_PARKING_SIDEWALK_IMG, topleft=True),

        ParkingCell(835, 0, 130, 65, 270, PATH_PARKING_SIDEWALK_IMG, topleft=True),
        ParkingCell(835, 130, 130, 65, 270, PATH_PARKING_SIDEWALK_IMG, topleft=True),
        ParkingCell(835, 260, 130, 65, 270, PATH_PARKING_SIDEWALK_IMG, topleft=True).place_car(100, 50,
                                                                                               PATH_CAR_IMG,
                                                                                               rotation=50),
        ParkingCell(835, 390, 130, 65, 270, PATH_PARKING_SIDEWALK_IMG, topleft=True),
        ParkingCell(835, 520, 130, 65, 270, PATH_PARKING_SIDEWALK_IMG, topleft=True).place_car(100, 50,
                                                                                               PATH_CAR_IMG,
                                                                                               rotation=70),
        ParkingCell(835, 650, 130, 65, 270, PATH_PARKING_SIDEWALK_IMG, topleft=True),
    ]
    ParkingCell(800, 0, 130, 65, 270, PATH_PARKING_SIDEWALK_IMG, topleft=True)
    lot = ParkingLot(1000, 1000, agent, parking_cells, [sidewalk_left, sidewalk_right])
    return lot


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
    sim = Simulator(lot, Analyzer1(), Extractor1(), drawing_method=DrawingMethod.FULL)
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
        reward, collision = sim.do_step(movement, steering, 1 / FPS)
        state = sim.get_state()

        # printing the results:
        if DEBUG:
            # print(
            # f"current loc: {sim.parking_lot.car_agent.location}, "
            # f" vel magnitude: "
            # f"{sim.parking_lot.car_agent.velocity.magnitude():.2f},"
            # f" acceleration: {sim.parking_lot.car_agent.acceleration:.2f}, "
            # f"collision: {results[Results.COLLISION]}, % in free cell: "
            # f"{0.0 if len(results[Results.UNOCCUPIED_PERCENTAGE]) == 0 else max(results[Results.UNOCCUPIED_PERCENTAGE].values()):.2f}")
            print(f"reward: {reward:.3f}, collision: {collision}")

        # updating the screen
        sim.update_screen()
