# Bregman divergence 

"""
    Implements the Bregman divergence, a friendly introduction to which can be found [here](http://mark.reid.name/blog/meet-the-bregman-divergences.html). Bregman divergences are a minimal generalization of the "mean-minimizer" property. 
    
    It is assumed that the function F maps vectors (of any type or size) to real numbers. The inner product used is `Base.dot`, but one can be passed in either by defining `inner` or by passing in a keyword argument. If an analytic gradient isn't available, Julia offers a suite of good automatic differentiation packages. 

    function evaluate(dist::Bregman{T <: Real}, F::Function, p::AbstractVector, q::AbstractVector, ∇::Function; inner=Base.dot)
"""
struct Bregman <: PreMetric
    F::Function 
    ∇::Function
    inner::Function 
end

# Default costructor. 
Bregman(F, ∇) =  Bregman(F, ∇, Base.dot)

# Evaluation fuction 
function evaluate(dist::Bregman, p::AbstractVector, q::AbstractVector)
    # Check inputs.
    F = dist.F;
    ∇ = dist.∇;
    inner = dist.inner; 
    FP_val = F(p);
    FQ_val = F(q); 
    DQ_val = ∇(q);
    # Check F codomain. 
    if isa(FP_val, Real) && isa(FQ_val, Real) 
    else 
        throw(ArgumentError("F Codomain Error: F doesn't map the vectors to real numbers"))
    end 
    # Check vector size. 
    if size(p) == size(q) 
    else
        throw(DimensionMismatch("The vector p ($(size(p))) and q ($(size(q))) are different sizes."))
    end
    # Check gradient size. 
    if size(DQ_val) == size(p)
    else 
        throw(DimensionMismatch("The gradient result is not the same size as p and q"))
    end 
    # Return the Bregman divergence. 
    return FP_val - FQ_val - inner(DQ_val, p-q);
end 

# Conveniece function. 
bregman(F, ∇, x, y; inner = Base.dot) = evaluate(Bregman(F, ∇, inner), x, y)
