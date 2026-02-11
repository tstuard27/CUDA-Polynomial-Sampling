# CUDA-Polynomial-Sampling
Personal project to learn CUDA which implements a parallelized GPU pipeline to generate, analyze, and visualize cubic polynamial functions at scale.

## Overview
Given a polynomial of the form: 
$$y = w_0x^3 + w_1x^2 + w_2x + b$$

The program:
* Generates 10,000+ uniformly spaced samples over the domain [-10, 10]
* Evaluates the polynomial in parallel using CUDA
* Computes first and second-order finite differences
* Computes local min/maximums and inflection points
* Transfers results back to the CPU
* Renders a 21×11 ASCII graph over the range [-5, 5]

## CUDA Concepts Demonstrated
* Grid and block configuration using dim3
* Global thread indexing with blockIdx and threadIdx
* Ceiling division for block count computation
* Device memory allocation and transfers
* Synchronization using cudaDeviceSynchronize
* Warp-aware block sizing
* Atomic operations for safe parallel counters

