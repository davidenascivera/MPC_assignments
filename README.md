# Space Debris Removal Simulation with Astrobee Robots

## Project Overview

This project focuses on simulating a solution to a current problem in space robotics: **removing space debris from Earth's orbit**. With the ever-increasing number of satellites and other space-bound objects, Earth’s orbit is becoming cluttered with “space garbage,” which poses significant risks to critical assets like the **International Space Station (ISS)** and new launches.

A proposed solution to mitigate this issue is to develop **cleaning satellites** capable of performing a "rendezvous maneuver" with space debris, where the satellite carefully approaches the debris and matches its momentum before impelling it towards Earth's atmosphere for destruction. 

For more information on the current state of space debris, satellites, and rocket bodies, visit: [AstriaGraph](http://astria.tacc.utexas.edu/AstriaGraph/).

### Project Simulation

To simulate this solution, we use **Astrobee robots**, which are free-flying robots developed by NASA. The project will simulate a debris cleaning maneuver using two Astrobee robots:
- **Honey**: Represents a space debris object that follows a pre-identified trajectory.
- **Bumble**: Represents the cleaning satellite that performs the rendezvous maneuver and the debris removal.

Before executing the simulation in space, we will first practice this maneuver on a **granite table**, which allows for nearly frictionless movement, mimicking space-like conditions.

## Key Components

1. **Debris Simulation**: Honey will move freely along a predefined trajectory to simulate the debris floating in orbit.
2. **Rendezvous Maneuver**: Bumble will attempt to approach Honey, match its momentum, and simulate the act of impelling the debris into a safe disposal trajectory.
3. **Granite Table**: Used for testing the feasibility of the approach in a controlled, low-friction environment.

## Repository Structure

- `/src`: Contains all the code required to control the robots and simulate the maneuver.
- `/docs`: Contains additional documentation and resources related to the project.
- `/tests`: Includes test scripts for verifying the performance of the simulation.
- `/assets`: Includes any images or media used in the project (e.g., diagrams, simulation results).

## Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/space-debris-simulation.git
