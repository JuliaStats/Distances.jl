using Distances

using Test
using LinearAlgebra
using OffsetArrays
using Random
using Statistics
using Unitful.DefaultSymbols

@test isempty(detect_ambiguities(Distances))

include("F64.jl")
include("test_dists.jl")
