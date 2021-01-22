"""
    Haversine(radius=6_371_000)

The haversine distance between two locations on a sphere of given `radius`, whose
default value is 6,371,000, i.e., the Earth's (volumetric) mean radius in meters; cf.
[NASA's Earth Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html).

Locations are described with longitude and latitude in degrees.
The computed distance has the unit of the radius.
"""
struct Haversine{T} <: Metric
    radius::T
end
Haversine() = Haversine(Float32(6371000))

function (dist::Haversine)(x, y)
    length(x) == length(y) == 2 || haversine_error(dist)

    @inbounds x1, x2 = x
    @inbounds y1, y2 = y
    # longitudes
    Δλ = _deg2rad(y1 - x1)

    # latitudes
    φ₁ = _deg2rad(x2)
    φ₂ = _deg2rad(y2)
    Δφ = φ₂ - φ₁

    # haversine formula
    a = sin(Δφ/2)^2 + cos(φ₁)*cos(φ₂)*sin(Δλ/2)^2

    # distance on the sphere
    2 * dist.radius * asin( min(√a, one(a)) ) # take care of floating point errors
end

haversine(x, y, radius=Float32(6371000)) = Haversine(radius)(x, y)

@noinline haversine_error(dist) = throw(ArgumentError("expected both inputs to have length 2 in $dist distance"))

_deg2rad(x::Real) = deg2rad(x)
_deg2rad(x) = x # targets unitful numbers

"""
    SphericalAngle()

The spherical angle distance between two locations on a sphere.

Locations are described with two angles, longitude and latitude, in radians.
The distance is computed with the haversine formula and also has units of radians.
"""
struct SphericalAngle <: Metric end

function (dist::SphericalAngle)(x, y)
    length(x) == length(y) == 2 || haversine_error(dist)

    @inbounds λ₁, φ₁ = x
    @inbounds λ₂, φ₂ = y

    Δλ = λ₂ - λ₁  # longitudes
    Δφ = φ₂ - φ₁  # latitudes

    # haversine formula
    a = sin(Δφ/2)^2 + cos(φ₁)*cos(φ₂)*sin(Δλ/2)^2

    # distance on the sphere
    2 * asin( min(√a, one(a)) ) # take care of floating point errors
end

spherical_angle(x, y) = SphericalAngle()(x, y)
