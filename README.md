# MultiHMMBase.jl
This is a module to do HMM when there are more than one kind of observations, thus when the distribution matrix for data is multidimensional array.

## Credit to HMMBase
This module is modified based on HMMBase, see https://github.com/maxmouchet/HMMBase.jl

## Functions
Currently, the viterbi algorithm, generating y, and liklihood calculation works, but the fit_mle function doesn't work yet.

## Exmpales
```{julia}
push!(LOAD_PATH, pwd())
using Multi_HMMBase

hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1) Normal(2,1); Normal(2,1) Normal(0,1)])
y = rand(hmm, 1000) # get random observatoin
likelihoods(hmm,y) # calculate liklihood of observation given hmm object
zv = viterbi(hmm, y) #  calculate hidden state trace 
```