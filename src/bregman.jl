# Bregman divergence 

"""
    Implements the Bregman divergence, a friendly introduction to which can be found [here](http://mark.reid.name/blog/meet-the-bregman-divergences.html). Bregman divergences are a minimal generalization of the "mean-minimizer" property. 
    
    It is assumed that the function F maps vectors (of any type or size) to real numbers. The inner product used is `Base.dot`, but one can be passed in either by defining `inner` or by passing in a keyword argument. If an analytic gradient isn't available, using a package like `ForwardDiff` is possible. 

    function evaluate(dist::Bregman{T <: Real}, F::Function, p::AbstractVector, q::AbstractVector, ∇::Function; inner=Base.dot)
"""
struct Bregman <: PreMetric
end

function evaluate(dist::Bregman, F::Function, p::AbstractVector, q::AbstractVector, ∇::Function; inner=Base.dot)
    # Check inputs. 
    @assert isa(F(p), Real) &&isa(F(q), Real) # Function codomain
    if (size(∇(p)) == size(∇(q)) == size(p) == size(q)) # Sizes of vectors 
    else
        throw(DimensionMismatch("Either the arrays don't vectors, or the gradients don't match the vectors."))
    end
    # Return the Bregman divergence. 
    return F(p) - F(q) - inner(∇(q), p-q);
end 

bregman(args...; kwargs...) = evaluate(Bregman(), args...; kwargs...)

function colwise(dist::Bregman, F::Function, p::AbstractMatrix, q::AbstractMatrix, ∇::Function; inner=Base.dot)
    # Check inputs. 
    @assert isa(F(p[:, 1]), Real) && isa(F(q[:, 1]), Real) # Function codomain 
    if size(p) == size(q) # Matrix sizes.
    else
        throw(DimensionMismatch("Matrices are of different sizes."))
    end
    if size(∇(p)) == size(p) && size(∇(q)) == size(q)
    else
        throw(DimensionMismatch("The gradients don't conform to the vectors"))
    end
    # Allocate return.   
    results = collect(1:1:size(p)[1])
    # Compute and return. 
    map!(results, results) do colindex
        return evaluate(dist, F, p[:, colindex], q[:, colindex], ∇; inner=inner)
    end
    return results 
end

function pairwise(dist::Bregman, F::Function, p::AbstractMatrix, q::AbstractMatrix, ∇::Function; inner=Base.dot)
    # Check inputs. 
    @assert isa(F(p[:, 1]), Real) && isa(F(q[:, 1]), Real) # Function codomain 
    if size(p) == size(q) # Matrix sizes.
    else
        throw(DimensionMismatch("Matrices are of different sizes."))
    end
    if size(∇(p)) == size(p) && size(∇(q)) == size(q)
    else
        throw(DimensionMismatch("The gradients don't conform to the vectors"))
    end
    # Define output.
    rows = size(p)[1];
    cols = size(p)[2];
    results = Matrix(cols, cols)
    # Compute 
    for i in 1:cols
        for j in 1:cols
            results[i, j] = evaluate(dist, F, p[:, i], q[:, j], ∇; inner=inner);
        end 
    end
    return results 
end

function colwise!(results::AbstractArray, dist::Bregman, F::Function, p::AbstractMatrix, q::AbstractMatrix, ∇::Function; inner=Base.dot)
    # Check inputs. 
    @assert isa(F(p[:, 1]), Real) && isa(F(q[:, 1]), Real) # Function codomain 
    if size(p) == size(q) # Matrix sizes.
    else
        throw(DimensionMismatch("Matrices are of different sizes."))
    end
    if size(∇(p)) == size(p) && size(∇(q)) == size(q)
    else
        throw(DimensionMismatch("The gradients don't conform to the vectors"))
    end
    cols = size(p)[2]
    @assert length(results) == cols
    # Compute and return. 
    for i in 1:cols 
        results[i] = evaluate(dist, F, p, q, ∇; inner=inner)
    end 
    return results 
end

function pairwise!(results::AbstractArray, dist::Bregman, F::Function, p::AbstractMatrix, q::AbstractMatrix, ∇::Function; inner=Base.dot)
    # Check inputs. 
    @assert isa(F(p[:, 1]), Real) && isa(F(q[:, 1]), Real) # Function codomain 
    if size(p) == size(q) # Matrix sizes.
    else
        throw(DimensionMismatch("Matrices are of different sizes."))
    end
    if size(∇(p)) == size(p) && size(∇(q)) == size(q)
    else
        throw(DimensionMismatch("The gradients don't conform to the vectors"))
    end
    rows = size(p)[1];
    cols = size(p)[2];
    @assert size(results) = (cols, cols)
    # Compute 
    for i in 1:cols
        for j in 1:cols
            results[i, j] = evaluate(dist, F, p[:, i], q[:, j], ∇; inner=inner);
        end 
    end
    return results 
end
