module ModelBasedCF

# package code goes here
using Persa
using ProgressMeter
using Statistics
using LinearAlgebra: norm
using Random: shuffle

abstract type MatrixFactorization{T} <: Persa.Model{T}
end

include("irsvd.jl")
include("rsvd.jl")
include("train.jl")

include("baseline.jl")
include("random.jl")

end # module
