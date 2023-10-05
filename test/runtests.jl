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

# Test ChainRules definitions on Julia versions that support weak dependencies
# Support for extensions was added in
# https://github.com/JuliaLang/julia/commit/93587d7c1015efcd4c5184e9c42684382f1f9ab2
# https://github.com/JuliaLang/julia/pull/47695 
if VERSION >= v"1.9.0-alpha1.18"
    include("chainrules.jl")
end