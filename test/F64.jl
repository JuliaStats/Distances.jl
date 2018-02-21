# dummy type wrapping a Float64 used in tests
struct F64 <: AbstractFloat
    x::Float64
end
F64(x::F64) = x

# operations
for op in (:+, :-, :sin, :cos, :asin, :acos)
    @eval Base.$op(a::F64) = F64($op(a.x))
end
for op in (:+, :-, :*, :/, :atan2)
    @eval Base.$op(a::F64, b::F64) = F64($op(a.x, b.x))
end
for op in (:zero, :one,)
    @eval Base.$op(::Type{F64}) = F64($op(Float64))
end

if VERSION.minor >= 7
    Random.rand(rng::AbstractRNG, ::Random.SamplerTrivial{Random.CloseOpen01{F64}}) = F64(rand(rng))
else
    Base.rand(rng::AbstractRNG, ::Type{F64}) = F64(rand())
end
Base.sqrt(a::F64) = F64(sqrt(a.x))
Base.:^(a::F64, b::Number) = F64(a.x^b)
Base.:^(a::F64, b::Int) = F64(a.x^b)
Base.:^(a::F64, b::F64) = F64(a.x^b.x)
Base.:^(a::Number, b::F64) = a^b.x
Base.log(a::F64) = F64(log(a.x))
Base.isfinite(a::F64) = isfinite(a.x)
Base.float(a::F64) = a.x
Base.rtoldefault(a::Type{F64}, b::Type{F64}) = Base.rtoldefault(Float64, Float64)
# comparison
Base.isapprox(a::F64, b::F64) = isapprox(a.x, b.x)
Base.:<(a::F64, b::F64) = a.x < b.x
Base.:<=(a::F64, b::F64) = a.x <= b.x
Base.eps(::Type{F64}) = eps(Float64)

# promotion
Base.promote_type(::Type{Float32}, ::Type{F64}) = Float64 # for eig
Base.promote_type(::Type{Float64}, ::Type{F64}) = Float64 # for vecnorm
Base.promote(a::F64, b::T) where {T <: Number} = a, F64(float(b))
Base.promote(a::T, b::F64) where {T <: Number} = F64(float(a)), b

Base.convert(::Type{F64}, a::F64) = a
Base.convert(::Type{Float64}, a::F64) = a.x
Base.convert(::Type{F64}, a::T) where {T <: Number} = F64(a)

# conversion
Base.Int64(a::F64) = Int64(a.x)
Base.Int32(a::F64) = Int32(a.x)
