# Model Predictive Control (MPC) Coursework

## Project Overview

This repository contains assignments and projects centered around **Model Predictive Control (MPC)**, with applications in space robotics. The primary objective is to design and implement advanced control strategies for **Astrobee robots**, which are free-flying robots operating aboard the **International Space Station (ISS)**. Key tasks include developing control solutions like **finite-time optimal control**, **Linear Quadratic Regulators (LQR)**, and **nonlinear MPC**. The final project applies these methods to an Astrobee rendezvous maneuver, simulating a controlled approach in space-like conditions.

### Assignments Breakdown

This coursework is divided into several assignments, each focusing on a different aspect of control system design:

- **Assignment 1: Discrete-Time Linear Systems**  
   *Description:* An introduction to discrete-time state-feedback control design for the Astrobee robot. The objective is to model Astrobee's movement and design a feedback controller to manage its motion effectively. This task includes simulating the robot's behavior under the control system and analyzing stability and performance.

- **Assignment 2: Finite-Time Optimal Control**  
   *Description:* This assignment explores finite-time optimal control strategies aimed at completing tasks within strict time limits. The task involves designing an optimal control strategy to manage a robotic satellite’s approach to a target object, simulating how it would efficiently conduct a rendezvous maneuver.

- **Assignment 3: Linear Quadratic Regulator (LQR)**  
   *Description:* Here, we implement a Linear Quadratic Regulator (LQR) to minimize a quadratic cost function for optimal control in linear systems. This assignment involves designing and evaluating the LQR to achieve efficient control over the Astrobee’s motion.

- **Assignment 4: Model Predictive Control (MPC)**  
   *Description:* This assignment covers the basics of MPC, implementing it for systems where control actions are optimized over a moving time horizon, considering system constraints and disturbances.

### Project: Nonlinear MPC for Astrobee Rendezvous

The final project applies nonlinear MPC to develop a robust control system for the **Astrobee robot's rendezvous maneuver**. The goal is to create a controller capable of handling potential system uncertainties, such as thruster malfunctions, during approach and docking operations. This simulation models conditions on the ISS, supporting future space applications of autonomous systems.

## Repository Structure

- `/src`: Contains all Python scripts and code files for implementing and simulating the control systems.
- `/docs`: Includes additional documentation, references, and resources relevant to the assignments.
- `/tests`: Contains scripts for verifying the performance of each control approach.
- `/assets`: Holds any media or graphical representations used in the project (e.g., diagrams, simulation results).

## Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/mpc-space-robotics.git
