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
