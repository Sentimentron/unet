# unet

Work in progress!

## TODO
* Figure out whether we want to mix OpenCL/CPU networks and at what level of granularity (e.g. multiplication on GPU, erorr function on CPU?)
* How to test something so complicated.
* What the end user API will look like for adding custom loss, activation, error functions.
* * What to do if end users want to target OpenCL/regular users?
* Should CUDA be considered too?
* * Need to dig into the literature

## Goals
* Basic backprop
* SGD
* Dropout

## Aims
* Testable
* Embeddable
