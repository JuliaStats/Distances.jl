# Copied from https://github.com/rdeits/NNLS.jl/blob/7017cbbedfb40a4dac52c448545d217f685e4150/src/NNLS.jl#L211
# Copyright (c) 2017: Robin Deits.

"""
Views in Julia still allocate some memory (since they need to keep
a reference to the original array). This type allocates no memory
and does no bounds checking. Use it with caution.
"""
immutable UnsafeVectorView{T} <: AbstractVector{T}
    offset::Int
    len::Int
    ptr::Ptr{T}
end

@inline UnsafeVectorView{T}(parent::DenseArray{T}, start_ind::Integer, len::Integer) = UnsafeVectorView{T}(start_ind - 1, len, pointer(parent))
@inline Base.size(v::UnsafeVectorView) = (v.len,)
@inline Base.getindex(v::UnsafeVectorView, idx) = unsafe_load(v.ptr, idx + v.offset)
@inline Base.setindex!(v::UnsafeVectorView, value, idx) = unsafe_store!(v.ptr, value, idx + v.offset)
@inline Base.length(v::UnsafeVectorView) = v.len
@inline Base.IndexStyle{V <: UnsafeVectorView}(::Type{V}) = Base.IndexLinear()

"""
UnsafeVectorView only works for isbits types. For other types, we're already
allocating lots of memory elsewhere, so creating a new View is fine.
This function looks type-unstable, but the isbits(T) test can be evaluated
by the compiler, so the result is actually type-stable.
"""
@inline function fastview{T}(parent::Matrix{T}, ::Colon, col::Int)
    if isbits(T)
        UnsafeVectorView(parent, size(parent, 1) * (col - 1) + 1, size(parent, 1))
    else
        @view(parent[start_ind:(start_ind + len - 1)])
    end
end

"""
Fallback for non-contiguous arrays, for which UnsafeVectorView does not make
sense.
"""
fastview(parent::AbstractArray, start_ind::Integer, len::Integer) = @view(parent[start_ind:(start_ind + len - 1)])
