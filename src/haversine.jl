"""
    Haversine(radius)

The haversine distance between two locations on a sphere of given `radius`.

Locations are described with longitude and latitude in degrees.
The computed distance has the same units as that of the radius.
"""
struct Haversine{T<:Real} <: Metric
    radius::T
end

const VecOrLengthTwoTuple{T} = Union{AbstractVector{T}, NTuple{2, T}}

function evaluate(dist::Haversine, x::VecOrLengthTwoTuple, y::VecOrLengthTwoTuple) 
    length(x) == length(y) == 2 || haversine_error()

    @inbounds begin
        # longitudes
        Δλ = deg2rad(y[1] - x[1])

        # latitudes
        φ₁ = deg2rad(x[2])
        φ₂ = deg2rad(y[2])
    end

    Δφ = φ₂ - φ₁

    # haversine formula
    a = sin(Δφ/2)^2 + cos(φ₁)*cos(φ₂)*sin(Δλ/2)^2
    c = 2atan2(√a, √(1-a))

    # distance on the sphere
    c*dist.radius
end

haversine(x::VecOrLengthTwoTuple, y::VecOrLengthTwoTuple, radius::Real) = evaluate(Haversine(radius), x, y)

@noinline haversine_error() = throw(ArgumentError("expected both inputs to have length 2 in Haversine distance"))