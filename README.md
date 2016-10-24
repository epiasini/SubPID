PIDpy/IIpy (working title)
=========================

This package is composed by two parts:

- A proof-of-principle toy implementation of the positive information
  decomposition (PID) for distributions over three binary variables
  using the definitions from Bertschinger, N., Rauh, J., Olbrich, E.,
  Jost, J., and Ay, N. /Quantifying unique information./ Entropy,
  16(4):2161–2183, 2014.
- A numerical implementation of a proposal for the "Intersection
  Information" measure (II; Panzeri et al., under review) which is
  based upon the PID.

Usage
-----

For the PID, the function you'll be interested in is
`information_decomposition.decomposition`. For the II, it is
`intersection_information.intersection_information`. See the
docstrings for details.

License
-------

Upon publication, this program will be licensed under version 3 of the
GPL or any later version. See COPYING for details.
