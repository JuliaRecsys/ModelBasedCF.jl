module ModelBasedCF

# package code goes here
using Persa
using ProgressMeter
using Statistics
using LinearAlgebra: norm
using Random: shuffle

abstract type MatrixFactorization <: Persa.Model
end

include("irsvd.jl")
include("rsvd.jl")
include("train.jl")

include("baseline.jl")

end # module
