module DistancesSparseArraysExt

using Distances
import Distances: _evaluate
using Distances: UnionMetrics, result_type, eval_start, eval_op, eval_reduce, eval_end
using SparseArrays: SparseVectorUnion, nonzeroinds, nonzeros, nnz

eval_op_a(d, ai, b) = eval_op(d, ai, zero(eltype(b)))
eval_op_b(d, bi, a) = eval_op(d, zero(eltype(a)), bi)

# It is assumed that eval_reduce(d, s, eval_op(d, zero(eltype(a)), zero(eltype(b)))) == s
# This justifies ignoring all terms where both inputs are zero.
Base.@propagate_inbounds function _evaluate(d::UnionMetrics, a::SparseVectorUnion, b::SparseVectorUnion, ::Nothing)
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    if length(a) == 0
        return zero(result_type(d, a, b))
    end
    anzind = nonzeroinds(a)
    bnzind = nonzeroinds(b)
    anzval = nonzeros(a)
    bnzval = nonzeros(b)
    ma = nnz(a)
    mb = nnz(b)
    ia = 1; ib = 1
    s = eval_start(d, a, b)
    @inbounds while ia <= ma && ib <= mb
        ja = anzind[ia]
        jb = bnzind[ib]
        if ja == jb
            v = eval_op(d, anzval[ia], bnzval[ib])
            ia += 1; ib += 1
        elseif ja < jb
            v = eval_op_a(d, anzval[ia], b)
            ia += 1
        else
            v = eval_op_b(d, bnzval[ib], a)
            ib += 1
        end
        s = eval_reduce(d, s, v)
    end
    @inbounds while ia <= ma
        v = eval_op_a(d, anzval[ia], b)
        s = eval_reduce(d, s, v)
        ia += 1
    end
    @inbounds while ib <= mb
        v = eval_op_b(d, bnzval[ib], a)
        s = eval_reduce(d, s, v)
        ib += 1
    end
    return eval_end(d, s)
end

@inline function _bhattacharyya_coeff(a::SparseVectorUnion, b::SparseVectorUnion)
    anzind = nonzeroinds(a)
    bnzind = nonzeroinds(b)
    anzval = nonzeros(a)
    bnzval = nonzeros(b)
    ma = nnz(a)
    mb = nnz(b)

    ia = 1; ib = 1
    s = zero(typeof(sqrt(oneunit(eltype(a))*oneunit(eltype(b)))))
    @inbounds while ia <= ma && ib <= mb
        ja = anzind[ia]
        jb = bnzind[ib]
        if ja == jb
            s += sqrt(anzval[ia] * bnzval[ib])
            ia += 1; ib += 1
        elseif ja < jb
            ia += 1
        else
            ib += 1
        end
    end
    # efficient method for sum for SparseVectorView is missing
    return s, sum(anzval), sum(bnzval)
end

end # module DistancesSparseArraysExt
