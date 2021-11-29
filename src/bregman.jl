# Bregman divergence

"""
Implements the Bregman divergence, a friendly introduction to which can be found
[here](http://mark.reid.name/blog/meet-the-bregman-divergences.html).
Bregman divergences are a minimal implementation of the "mean-minimizer" property.

It is assumed that the (convex differentiable) function F maps vectors (of any type or size) to real numbers.
The inner product used is `Base.dot`, but one can be passed in either by defining `inner` or by
passing in a keyword argument. If an analytic gradient isn't available, Julia offers a suite
of good automatic differentiation packages.

function evaluate(dist::Bregman, p::AbstractVector, q::AbstractVector)
"""
struct Bregman{T1 <: Function, T2 <: Function, T3 <: Function} <: PreMetric
    F::T1
    ∇::T2
    inner::T3
end

# Default costructor.
Bregman(F, ∇) =  Bregman(F, ∇, LinearAlgebra.dot)

# Evaluation fuction
function (dist::Bregman)(p, q)
    # Create cache vals.
    FP_val = dist.F(p)
    FQ_val = dist.F(q)
    DQ_val = dist.∇(q)
    p_size = length(p)
    # Check F codomain.
    if !(isa(FP_val, Real) && isa(FQ_val, Real))
        throw(ArgumentError("F Codomain Error: F doesn't map the vectors to real numbers"))
    end
    # Check vector size.
    if p_size != length(q)
        throw(DimensionMismatch("The vector p ($(size(p))) and q ($(size(q))) are different sizes."))
    end
    # Check gradient size.
    if length(DQ_val) != p_size
        throw(DimensionMismatch("The gradient result is not the same size as p and q"))
    end
    # Return the Bregman divergence.
    return FP_val - FQ_val - dist.inner(DQ_val, p .- q)
end

# Convenience function.
bregman(F, ∇, x, y; inner = LinearAlgebra.dot) = Bregman(F, ∇, inner)(x, y)
