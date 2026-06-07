# Optimization Algorithms

---

## Overview

A comprehensive repository dedicated to the exploration, implementation, and visualization of various optimization and sampling algorithms. This project serves as a practical sandbox to understand the inner workings of mathematical optimization, experimental design, and search heuristics.

## Implemented Algorithms

### Genetic Algorithm (GA)

An evolutionary algorithm inspired by the process of natural selection. It is commonly used to generate high-quality solutions to optimization and search problems by relying on biologically inspired operators such as mutation, crossover, and selection.

- Includes auxiliary notebooks and pre-trained regressor models used as fitness functions or surrogate models.

#### Latin Hypercube Sampling (LHS)

A statistical method for generating a random sample of parameter values from a multidimensional distribution. It is often used to construct computer experiments efficiently, guaranteeing that each sample is comprehensively distributed across the parameter space.

## Project Structure

```text
optimization/
├── genetic_algorithm
│   ├── ga.ipynb                    # Genetic algorithm main file
│   └── sample_solver               # Sample solver
│       ├── GaussianRegressor.pkl
│       └── LinearRegressor.pkl
├── pyproject.toml
├── README.md
├── sampling_algorithm              # Sampling algorithm
│   └── lhs.ipynb
└── uv.lock
```

## Prerequisites

UV as python package manager

## Usage

```bash
uv sync
```
