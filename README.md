# HQ-PIDT
This repository is the code for the ungoing HQ-PIDT work.

The first work of HQ-PIDT is in the paper "HQ-PIDT: A Hybrid Quantum Physics-Informed Decision Transformer for Resilient Distribution System Restoration Under Uncertainty". The link is [here](https://doi.org/10.31224/8230). Then I attended the QCUP Hackathon program as a core team member, and with the help of IBM mentors, 

This code contains different code development phases. All these code are performed with the help of vibe coding.
- [x] **First phase**: the initial version of HQ-PIDT. This version uses Pennylane for QML operations, and uses Pennylane simulators as simulation backends.
- [x] **Second phase**: the product of QCUP Hackathon. This version uses Qiskit for QML operations, and include a 4-stage process:
  - **Stage 1**: HQ-PIDT model trained from scratch using the Qiskit Aer Simulator.
  - **Stage 2**: Trained HQ-PIDT model inference with ideal Qiskit Aer Simulation. From this stage, all QML operations are finite-shot based.
  - **Stage 3**: Trained HQ-PIDT model inference with noisy Qiskit Aer Simulation.
  - **Stage 4**: Trained HQ-PIDT model inference with real IBM QPUs.
- [ ] **Third phase**: ongoing HQ-PIDT optimization based on the product of second phase.

The inference test case is the Distribution System Restoration problem for power system, and testbeds are several standard power simulation systems. Data used for offline training are generated using normal deep reinforcement learning experts. This repositary will not include the generated datasets and the testbeds used.
