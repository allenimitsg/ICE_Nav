# ICE_Nav
Polar Ship Ice-Theta* Path Planning System
An intelligent path planning software system for polar ships navigating in broken ice areas, based on high-resolution optical imagery and improved Ice-Theta* algorithm.
Overview
This project is an intelligent path planning software system for polar ships navigating in localized broken ice areas, developed to support the corresponding academic research.To address the problems of low environmental perception accuracy, unsmooth paths from traditional grid-based algorithms, and neglect of ice resistance, this system implements an integrated path planning method combining high-resolution optical remote sensing image segmentation and an improved Ice-Theta* algorithm.
It constructs a multi-level ice navigation environment and generates smooth, low-resistance, low-collision-risk paths, significantly improving the safety and efficiency of polar navigation.
Key Features
High-Fidelity Ice Environment ModelingBased on high-resolution optical remote sensing images, an intensity-based segmentation algorithm is used to automatically construct a high-fidelity simulation environment with multi-level navigable attributes including open water, broken ice, and thick ice.
Improved Ice-Theta Path Planning*Breaks through the direction limitations of traditional grid-based methods and supports any-angle smooth path planning. It adopts an improved line-of-sight check mechanism and soft cost field integration.
Integration of Ice Resistance and Safety Potential FieldA multi-objective cost function is constructed to comprehensively optimize path length, ice navigation resistance, and collision risk, prioritizing low-resistance and safe channels.
Ice Field Simulation and Performance EvaluationSupports testing in random ice fields of varying concentrations, and automatically calculates path length, travel time, collision energy, comprehensive energy consumption, and other indicators.
Advantages
Average path length reduced by approximately 8.3% compared with the traditional A* algorithm
Collision energy significantly decreased by about 30.8%
Navigation time shortened by up to 12.8% under low-density ice conditions
Smooth paths without redundant turns, conforming to real ship maneuvering habits
Balances navigation safety and energy consumption optimization, suitable for complex polar broken ice environments
Related Paper
Title: Polar Ship Ice-Theta* Path Planning Method Considering Ice Resistance and Safety Potential Field
Problems Solved:
Low accuracy of environmental perception for polar ships in broken ice areas
Zigzag paths and redundant distance in traditional grid-based path planning
Ignorance of ice resistance and navigation safety in path planning
Quick Start
Clone this repository locally
Configure the runtime environment (Python and dependencies listed in requirements)
Import high-resolution optical ice images or load random ice fields
Run the main program and set start point, end point, and ship parameters
Automatically generate smooth paths and view visualization results and performance metrics
Applications
Path planning for polar research vessels and transport ships in broken ice areas
Ice navigation environment simulation and safety assessment
Development of intelligent navigation systems for polar ships
Research and teaching demonstrations on maritime safety in ice-covered regions
License
This project is open-sourced for academic research only. Please cite this repository and the corresponding paper when using the code or results.
