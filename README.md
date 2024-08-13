This project provides a Python routine resolving the Plumb's Quasi-Biennial Oscillation unidimensional model.

The kernel of the model is found in "qbo1d_utils.py". The numerical scheme makes use of implicit diffusion, a 3rd order Adams-Bashforth timestepping,
and a numerical criterion tampering wave momentum deposition above critical levels.

Two script examples with different forcing configurations are given: one with monochromatic wave forcing "QBO1Dstart_1wave.py", and the other with a wave forcing
composed of two pairs of waves "QBO1Dstart_2waves.py".
