# Third-party software

The `src/fasterwam` implementation includes components adapted from
FasterWAM, distributed under the Apache
License, Version 2.0. Its license is retained in `LICENSE`. The Wan video model,
VAE, continuous flow scheduler, and supporting tensor utilities are retained
within their original module namespace. This release changes packaging, removes
machine-specific paths, and adds portable examples and task interfaces.

Wan VAE weights are not included. Obtain them separately under their upstream
terms. MuJoCo, PyTorch, NumPy, SciPy, OpenCV and other dependencies are installed
separately; their respective licenses continue to apply. No third-party dataset,
pretrained weight, or private recording is redistributed here.

Public upstream attribution identifies dependencies, not the anonymous authors.

The two informational web-address lines are omitted from the supplied license
text for this link-free distribution. License terms and attribution are retained.
