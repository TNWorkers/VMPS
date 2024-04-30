# Variational Matrix Product State library

This is a library for dealing with many-body Hamiltonians using algorithms based on variational matrix-product states (VMPS or MPS).

## Features

- ground states for one-dimensional chains (open and periodic boundary conditions)
- ground states for general geometries: ladders, molecules, 2D and 3D clusters
- infinite systems using the VUMPS (Variational Uniform Matrix-Product State) formalism
- Heisenberg, Hubbard and Kondo type models
- U(1), SU(2) [spin and charge], Z(N) symmetries
- time propagation using an adaptive TDVP (time-dependent variational principle) algorithm
- spectral functions using the Chebyshev polynomial expansion or time propagation
- finite-temperature properties (static and dynamic)

## Dependencies

- C++, Eigen, Boost

## Contributing

A project by Roman Rausch, Matthias Peschke, Cassian Plorin

## Showcases

<p>
• Spin-spin correlations of the Heisenberg model on the C60 fullerene geometry [https://scipost.org/SciPostPhys.10.4.087]:
<div style="text-align: center"><img src="http://sindanoorie.net/vmps/C60correlations.png" width=50% height=50%></div>
</p>

<p>
• Incommensurate spin-spin correlations in the FM-AFM sawtooth chain [https://scipost.org/SciPostPhys.14.3.052]:
<img src="http://sindanoorie.net/vmps/Sawtooth.png" width=60% height=60%>
</p>

<p>
• Even-odd mass differences in Pb isotopes [https://onlinelibrary.wiley.com/doi/full/10.1002/andp.202300436]
<img src="http://sindanoorie.net/vmps/Sawtooth.png" width=70% height=70%>
</p>

<p>
• Four-electron bound state ("quadruplon") in the two-hole spectral function of the Hubbard model [https://iopscience.iop.org/article/10.1088/1367-2630/18/2/023033/meta]:
<img src="http://sindanoorie.net/vmps/Multiplon.png" width=70% height=70%>
</p>