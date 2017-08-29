# SubPID - Subatomic Partial Information Decomposition

[![DOI](https://zenodo.org/badge/61385385.svg)](https://zenodo.org/badge/latestdoi/61385385)

This package is composed by two parts:

- A MATLAB implementation of the partial information decomposition
  (PID) for distributions over three random variables using the
  definitions from Bertschinger, N., Rauh, J., Olbrich, E., Jost, J.,
  and Ay, N. *Quantifying unique information.*. Entropy 2014,
  16(4):2161–2183.
- A MATLAB numerical implementation of the subatomic partial
  information decomposition proposed in Pica, G., Piasini, E.,
  Chicharro, D., and Panzeri, S. *Invariant Components of Synergy,
  Redundancy, and Unique Information Among Three Variables.*. Entropy
  2017, 19, 451,
  doi:[10.3390/e19090451](https://dx.doi.org/10.3390/e19090451).

## Requirements

`SubPID` requires [glpkmex](https://github.com/blegat/glpkmex) to be
installed on your system.

## Usage

The PID and the subatomic PID are implemented by `src/matlab/pid.m`
and `src/matlab/subatomic_pid.m`, respectively. See the function
descriptions for details.

## License
	
This program is licensed under the GNU General Public License, version
3, or any later version. See LICENSE for details.
