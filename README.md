# Challenge for the QEC Hackathon 2023 at ETH Zurich

## Summary
We can use quantum computers to help us with machine learning. By utilizing variational quantum circuits, we can use quantum computers to learn patterns in data.

In order for this to work well, we need to come up with a way to encode the patterns we want to learn into the quantum computer. Especially on near-term quantum hardware, this requires quite some resources. One way to reduce these resource requirements and learn more efficiently is to take advantage of symmetries in the data we are trying to learn.

One approach is taken in [this paper](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.010328) (CC-BY 4.0, Johannes Jakob Meyer, Marian Mularski, Elies Gil-Fuster, Antonio Anna Mele, Francesco Arzani, Alissa Wilms, and Jens Eisert) and we will use it as a guideline for this challenge. A huge thank you to the authors for providing this idea and explaing it so well :-)

## What's inside
This repository contains a Jupyter Notebook that walks you through the basic idea of constructing a circuit that utilizes the problem symmetry and applies it to Tic-Tac-Toe.
