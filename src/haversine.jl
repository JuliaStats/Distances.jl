"""
    Haversine(radius)

The haversine distance between two locations on a sphere of given `radius`.

Locations are described with longitude and latitude in degrees and
the radius of the Earth is used by default (≈ 6371km). The computed
distance has the same units as that of the radius.

### Notes

The haversine formula is widely used to approximate the geodesic distance
between two points at the surface of the Earth. The error from approximating
the Earth as a sphere is typically negligible for most applications. It is
no more than 0.3%.
"""
struct Haversine{T<:Real} <: Metric
    radius::T
end

# use Earth radius ≈ 6371km by default
Haversine() = Haversine(6371.)

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
haversine(x::AbstractVector, y::AbstractVector) = evaluate(Haversine(), x, y)
