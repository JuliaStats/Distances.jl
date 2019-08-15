using Distances

using Test
using LinearAlgebra
using Random
using Statistics

@test isempty(detect_ambiguities(Distances))

include("F64.jl")
include("test_dists.jl")
