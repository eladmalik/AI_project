import sys

import pygame

from simulator import Simulator, PATH_PARKING_IMG, PATH_AGENT_IMG, PATH_CAR_IMG, Results
from parking_lot import ParkingLot
from parking_cell import ParkingCell
from car import Car, Movement, Steering

FPS = 60

if __name__ == '__main__':
    car = Car(300, 150, 100, 50, 180, PATH_CAR_IMG)
    cell1 = ParkingCell(300, 150, 300, 150, 0, PATH_PARKING_IMG, car)
    cell2 = ParkingCell(300, 300, 300, 150, 0, PATH_PARKING_IMG)
    cell3 = ParkingCell(500, 500, 300, 150, 30, PATH_PARKING_IMG, topleft=True)
    agent_car = Car(600, 500, 100, 50, 0, PATH_AGENT_IMG)
    lot = ParkingLot(1000, 1000, agent_car, [cell1, cell2, cell3])
    sim = Simulator(lot)
    clock = pygame.time.Clock()
    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
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
        results = sim.move_agent(movement, steering, 1 / FPS)
        print(
            f"current loc: {sim.parking_lot.car_agent.location}, "
            f" vel magnitude: "
            f"{sim.parking_lot.car_agent.velocity.magnitude():.2f},"
            f" acceleration: {sim.parking_lot.car_agent.acceleration:.2f}, "
            f"collision: {results[Results.COLLISION]}, % in free cell: "
            f"{0.0 if len(results[Results.UNOCCUPIED_PERCENTAGE]) == 0 else max(results[Results.UNOCCUPIED_PERCENTAGE].values()):.2f}")
        sim.draw_screen()
        pygame.display.update()
