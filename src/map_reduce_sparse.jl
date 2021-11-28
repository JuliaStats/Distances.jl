# Assumptions:
# f(0, 0) == 0
# op(v, 0) == v
#
# one_nonz_count is returned so that one can compute mapreduce for
# f(0, 0) != 0 for some op's such as +
function _binary_map_reduce1(f::Function, fx::Function, fy::Function,
                            op::Function, mx::Int, my::Int,
                            xnzind, xnzval::AbstractVector{Tx},
                            ynzind, ynzval::AbstractVector{Ty}) where {Tx,Ty}

    ix = 1; iy = 1
    s = zero(f(zero(Tx), zero(Ty)))
    one_nonz_count = 0 # number of indices at which at least one vector is non-zero
    @inbounds while ix <= mx && iy <= my
        jx = xnzind[ix]
        jy = ynzind[iy]
        if jx == jy
            v = f(xnzval[ix], ynzval[iy])
            s = op(v, s)
            ix += 1; iy += 1
        elseif jx < jy
            v = fx(xnzval[ix])
            s = op(v, s)
            ix += 1
        else
            v = fy(ynzval[iy])
            s = op(s, v)
            iy += 1
        end
        one_nonz_count += 1
    end
    @inbounds while ix <= mx
        v = fx(xnzval[ix])
        s = op(s, v)
        ix += 1
        one_nonz_count += 1
    end
    @inbounds while iy <= my
        v = fy(ynzval[iy])
        s = op(s, v)
        iy += 1
        one_nonz_count += 1
    end
    return s, one_nonz_count
end
