struct PeriodicEuclidean{W <: RealAbstractArray} <: Metric
    periods::W
end

"""
    PeriodicEuclidean(L)
Create a Euclidean metric on a rectangular periodic domain (i.e., a torus or
a cylinder). Periods per dimension are contained in the vector `L`.
For dimensions without periodicity put `Inf` in the respective component.

# Example
```
julia> x, y, L = [0.0, 0.0], [0.7, 0.0], [0.5, Inf]
([0.0, 0.0], [0.7, 0.0], [0.5, Inf])

julia> Distances.evaluate(PeriodicEuclidean(L),x,y)
```
"""
# PeriodicEuclidean without periods is considered as classic Euclidean
PeriodicEuclidean() = Euclidean()
Base.eltype(x::PeriodicEuclidean) = eltype(x.periods)

# Specialized for Arrays and avoids a branch on the size
@inline Base.@propagate_inbounds function evaluate(d::PeriodicEuclidean, a::Union{Array, ArraySlice}, b::Union{Array, ArraySlice})
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    @boundscheck if length(a) != length(d.periods)
        throw(DimensionMismatch("arrays have length $(length(a)) but periods have length $(length(d.periods))."))
    end
    if length(a) == 0
        return zero(result_type(d, a, b))
    end
    @inbounds begin
        s = eval_start(d, a, b)
        @simd for I in eachindex(a, b, d.periods)
            ai = a[I]
            bi = b[I]
            li = d.periods[I]
            s = eval_reduce(d, s, eval_op(d, ai, bi, li))
        end
        return eval_end(d, s)
    end
end

@inline function evaluate(d::PeriodicEuclidean, a::AbstractArray, b::AbstractArray)
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    @boundscheck if length(a) != length(d.periods)
        throw(DimensionMismatch("arrays have length $(length(a)) but periods have length $(length(d.periods))."))
    end
    if length(a) == 0
        return zero(result_type(d, a, b))
    end
    @inbounds begin
        s = eval_start(d, a, b)
        if size(a) == size(b)
            @simd for I in eachindex(a, b, d.periods)
                ai = a[I]
                bi = b[I]
                li = d.periods[I]
                s = eval_reduce(d, s, eval_op(d, ai, bi, li))
            end
        else
            for (Ia, Ib, Ip) in zip(eachindex(a), eachindex(b), eachindex(d.periods))
                ai = a[Ia]
                bi = b[Ib]
                li = d.periods[Ip]
                s = eval_reduce(d, s, eval_op(d, ai, bi, li))
            end
        end
    end
    return eval_end(d, s)
end

function evaluate(dist::PeriodicEuclidean, a::T, b::T) where {T <: Number}
    eval_end(dist, eval_op(dist, a, b, one(eltype(dist))))
end
function result_type(dist::PeriodicEuclidean, ::AbstractArray{T1}, ::AbstractArray{T2}) where {T1, T2}
    typeof(evaluate(dist, one(T1), one(T2)))
end
@inline function eval_start(d::PeriodicEuclidean, a::AbstractArray, b::AbstractArray)
    zero(result_type(d, a, b))
end
@inline function eval_op(::PeriodicEuclidean, ai, bi, li)
    d = mod(abs(ai - bi), li)
    d = min(d, li - d)
    abs2(d)
end
@inline eval_reduce(::PeriodicEuclidean, s1, s2) = s1 + s2
@inline eval_end(::PeriodicEuclidean, s) = sqrt(s)

peuclidean(a::AbstractArray, b::AbstractArray, p::AbstractArray) = evaluate(PeriodicEuclidean(p), a, b)
peuclidean(a::AbstractArray, b::AbstractArray) = euclidean(a, b)
# this avoids unnecessary squaring and taking square root
function peuclidean(a::Number, b::Number, p::Number)
    d = mod(abs(a - b), p)
    d = min(d, p - d)
end
