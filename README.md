# SubPID - Subatomic Partial Information Decomposition

This package is composed by two parts:

- A MATLAB implementation of the partial information decomposition
  (PID) for distributions over three random variables using the
  definitions from Bertschinger, N., Rauh, J., Olbrich, E., Jost, J.,
  and Ay, N. *Quantifying unique information.* Entropy,
  16(4):2161–2183, 2014.
- A MATLAB numerical implementation of the subatomic partial
  information decomposition proposed in Pica et al, *Invariant
  components of synergy, redundancy, and unique information among
  three variables.*, to appear in Entropy,
  [arXiv:1706.08921 \[cs.IT\]](https://arxiv.org/abs/1706.08921),
  2017.

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
