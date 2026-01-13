# Code accompanying "Exact analytical solutions of the Bloch equation for the hyperbolic-secant and chirp pulses"

Smith, Ryan HB, Donald Garwood, and Michael Garwood. "Exact analytical solutions of the Bloch equation for the hyperbolic‐secant and chirp pulses." *Magnetic resonance in medicine* (2025).

The full manuscript is open access and available at: [https://doi.org/10.1002/mrm.30603Digital Object Identifier (DOI)](https://doi.org/10.1002/mrm.30603)

## Abstract
### Purpose
To improve the accuracy and generality of analytical solutions of the Bloch equation for the hyperbolic-secant (HS1) and chirp pulses in order to facilitate application to truncated and composite pulses and use in quantitative methods.

### Theory and Methods
Previous analytical solutions of the Bloch equation during an HS1 pulse driving function are refined and extended in this exact solution for arbitrary initial magnetization and pulse parameters including asymmetrical truncation. An unapproximated general solution during the chirp pulse is derived in a non-spinor formulation for the first time. The solution on the extended complex plane for the square pulse is included for completeness.

### Results
The exact solutions for the HS1, chirp, and square pulses demonstrate high consistency with Runge-Kutta simulations for all included pulse and isochromat parameters. The HS1 solution is strictly more accurate than the most complete prior general solution. The analytical solution of the BIR-4 composite pulse constructed using asymmetrically truncated HS1 component pulses likewise agrees with simulation results.

### Conclusion
The derived analytical solutions for the Bloch equation during an HS1 or chirp pulse are exact regardless of pulse parameters and initial magnetization and precisely conform with simulations enabling their use in quantitative MRI applications and setting a foundation for the analytical consideration of relaxation and pulses in multiply rotating frames.

## Description

The above manuscript describes and validates an exact analytical solution to the evolution of the magnetic moment subject to HS1, chirp, and square pulses. This repository contains a computational implementation of the code used in the preparation of that manuscript including a Runge-Kutta fourth order Bloch simulator and analytical calculations using the mpmath library. This is a snapshot of a code base under active development, so it is provided "as is" without formal packaging.

The functions to prepare and run a Bloch simulation are in [simulate.py](./mrpy/simulate.py), [w1_gen.py](./mrpy/w1_gen.py), and [bloch.py](./mrpy/bloch.py). The analytical calculations are performed in [calculate.py](./mrpy/calculate.py). Functions written specifically to generate the data used in the paper (but useful as examples) are included in [hs1chirp_paper.py](./mrpy/hs1chirp_paper.py); these functions are called in [hspaper_2.0.ipynb](./hspaper_2.0.ipynb) with the figures saved in [HS1chirp_figs_2.0](./HS1chirp_figs_2.0/).

As an example, below is a pre-annotation version of Figure 4 created by [hspaper_2.0.ipynb](./hspaper_2.0.ipynb) demonstrating the performance of calculations when applied to the BIR-4 composite pulse.
![Figure 4](./HS1chirp_figs_2.0/f4_bir4_accuracy.svg)

## Getting Started

### Dependencies

For the simulation and calculation code:
* mpmath
* numpy

For the figure preparation code:
* matplotlib
* pandas
* seaborn

### Installing

* Download the repository and add to your environment
* For a local-only import, the first cell of hspaper_2.0.ipynb can be edited with the path of the repository permitting importation.

## Help

Please reach out with any questions regarding the code or bugs. Updated code will be released in a separate repository while preserving this publication-associated snapshot.

## Authors

Ryan H.B. Smith
[smi03101@umn.edu](mailto:smi03101@umn.edu)
Donald Garwood
Michael Garwood

## Version History

* 2.0
    * Paper codebase snahpshot release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details