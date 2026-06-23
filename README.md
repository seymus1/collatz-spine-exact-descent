# collatz-spine-exact-descent

This repository contains the Python code and output summaries for the exact-integer verification and AAS-based metamorphic post-evaluation reported in the paper:

**Exact and Delayed Descent in Accelerated Odd Collatz Spines with AAS-Based Metamorphic Separation**

## Purpose

The code verifies finite-range first-descent behavior for spine-level cases of the accelerated odd Collatz map

\[
T(n)=\frac{3n+1}{2^{v_2(3n+1)}}.
\]

The tested spine elements have the form

\[
n=2^m q-1.
\]

The repository also contains code for the AAS-based diagnostic post-evaluation comparing certified exact-descent cases with non-certified control groups.

## Scope

The computations are finite-range exact-integer verifications up to

\[
N=10^7.
\]

They do not prove the Collatz conjecture and do not prove universal descent for all non-certified spine elements.

## Main script

The main script is located at:

```text
code/exact_verification_aas.py
