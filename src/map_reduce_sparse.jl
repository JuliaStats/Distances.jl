# Assumptions:
# f(x, 0) == 0, for all x
# f(0, y) == 0, for all y
# op(v, 0) == v
function _binary_map_reduce1(f::Function, op::Function, mx::Int, my::Int,
                            xnzind, xnzval::AbstractVector{Tx},
                            ynzind, ynzval::AbstractVector{Ty}) where {Tx,Ty}
    ix = 1; iy = 1
    s = zero(f(zero(Tx), zero(Ty)))
    @inbounds while ix <= mx && iy <= my
        jx = xnzind[ix]
        jy = ynzind[iy]
        if jx == jy
            v = f(xnzval[ix], ynzval[iy])
            s = op(v, s)
            ix += 1; iy += 1
        elseif jx < jy
            ix += 1
        else
            iy += 1
        end
    end
    return s
end

eval_op_a(d, ai, b) = eval_op(d, ai, zero(eltype(b)))
eval_op_b(d, bi, a) = eval_op(d, zero(eltype(a)), bi)

Base.@propagate_inbounds function _evaluate(d::UnionMetrics, a::SparseVector, b::SparseVector, ::Nothing)
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
    while ia <= ma && ib <= mb
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
    while ia <= ma
        v = eval_op_a(d, anzval[ia], b)
        s = eval_reduce(d, s, v)
        ia += 1
    end
    while ib <= mb
        v = eval_op_b(d, bnzval[ib], a)
        s = eval_reduce(d, s, v)
        ib += 1
    end
    return eval_end(d, s)
end
