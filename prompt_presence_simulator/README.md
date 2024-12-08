# prompt-mac-simulator
Prompt presence notification, random access MAC strategies simulator.

Python 3 simulation framework, to verify reseach-related theses & hypotheses. Framework simulates, calculates[^1] and visualizes communication probability success of `Bernoulli-Trials (BT)` & `tSlots` MAC[^2] random access strategies, developed in scope of the research of uncontrolled, distributed, no-ack MAC strategy in a slotted channel. Main point of the research was to derive all-nodes communication success probability formulas and compare performance for 2 fundamental approaches of no-ack communication, to provide a background for a time-constrained MAC protocol.

# Simulator
This repository contains a source code of the simulator, useful for hypotheses verification, statistical tests and communication success probability presentation. Simulator is an adapted Distributed Communication Simulator framework, developed also by author.

# Author of the source code
Patryk Stopyra

Wroclaw University of Technology

[^1]: With use of https://github.com/V-I-S/prompt-mac-calculator
[^2]: Medium Access Control
