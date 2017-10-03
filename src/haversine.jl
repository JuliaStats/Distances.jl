"""
    Haversine(radius)

The haversine distance between two locations on a sphere of given `radius`.

Locations are described with longitude and latitude in degrees.
The computed distance has the same units as that of the radius.
"""
struct Haversine{T<:Real} <: Metric
    radius::T
end

function evaluate(dist::Haversine{T}, x::AbstractVector, y::AbstractVector) where {T<:Real}
    # longitudes
    Δλ = deg2rad(y[1] - x[1])

    # latitudes
    φ₁ = deg2rad(x[2])
    φ₂ = deg2rad(y[2])
    Δφ = φ₂ - φ₁

    # haversine formula
    a = sin(Δφ/2)^2 + cos(φ₁)*cos(φ₂)*sin(Δλ/2)^2
    c = 2atan2(√a, √(1-a))

    # distance on the sphere
    c*dist.radius
end

haversine(x::AbstractVector, y::AbstractVector, radius::T) where {T<:Real} = evaluate(Haversine(radius), x, y)
