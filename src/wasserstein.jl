# Wasserstein distance

struct Wasserstein <: Metric
    p::Float64

    function Wasserstein(p::Float64)
        @assert p >= 1
        new(p)
    end
end

Wasserstein() = Wasserstein(1.0)

function (dist::Wasserstein)(a::AbstractVector, b::AbstractVector)
    throw("implement me")
end
