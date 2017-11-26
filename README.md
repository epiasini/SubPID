# Intersection information and SubPID (Subatomic Partial Information Decomposition)

[![DOI](https://zenodo.org/badge/61385385.svg)](https://zenodo.org/badge/latestdoi/61385385)

This package is composed by four parts:

- A MATLAB implementation of the partial information decomposition
  (PID) for distributions over three random variables using the
  definitions from Bertschinger, N., Rauh, J., Olbrich, E., Jost, J.,
  and Ay, N. *Quantifying unique information.*. Entropy 2014,
  16(4):2161–2183.
- Two MATLAB implementations of the intersection information I_{II}(S;R;C)
  that, in perceptual discrimination experiments, quantifies the sensory 
  information in the recorded neural response R that is relevant to behavior.
  This measure is defined and described in Pica, G., Piasini, E., Safaai, H., 
  Runyan, C.A., Diamond, M.E., Fellin, T., Kayser, C., Harvey, C.D., Panzeri, S.,
  *Quantifying how much sensory information in a neural code is 
  relevant for behavior*, Advances in neural information processing 2017, 3687-3697.
  The first implementation, "src/matlab/intersection information.m", evaluates 
  I_{II}(S;R;C) starting from the empirical joint probability distribution p(s,r,c).
  The second implementation, "src/matlab/intersection information_from_binned_response.m", 
  evaluates I_{II}(S;R;C) starting from vectors containing the stimulus s, the response r, 
  and the choice c, corresponding to each trial. Here, the response is discretized into equipopulated
  bins for a conservative estimate of I_{II}(S;R;C).
- A MATLAB numerical implementation of the subatomic partial
  information decomposition proposed in Pica, G., Piasini, E.,
  Chicharro, D., and Panzeri, S. *Invariant Components of Synergy,
  Redundancy, and Unique Information Among Three Variables.*. Entropy
  2017, 19, 451,
  doi:[10.3390/e19090451](https://dx.doi.org/10.3390/e19090451).

## Requirements

`Intersection information and SubPID` requires [glpkmex](https://github.com/blegat/glpkmex) to be
installed on your system.

## Usage

The intersection information can be estimated from raw response data
with "src/matlab/intersection information_from_binned_response.m", 
while the more general "src/matlab/intersection information.m" should 
be used if the user has already estimated p(s,r,c) from the experimental dataset.
The PID and the subatomic PID are implemented by `src/matlab/pid.m`
and `src/matlab/subatomic_pid.m`, respectively. See the function
descriptions for details.

## License
	
This program is licensed under the GNU General Public License, version
3, or any later version. See LICENSE for details.
