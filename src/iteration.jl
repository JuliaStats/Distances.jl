

function colwise_noalloc!(r::AbstractVector, metric::UnionMetrics, A::AbstractMatrix, B::AbstractMatrix)
    @boundscheck begin
        if !(indices(A,2) == indices(B,2) == indices(r,1))#length(a) != length(b)
            throw(DimensionMismatch("r, A, B have incompatible indices"))
        end
        if !(indices(A,1) == indices(B,1))
             throw(DimensionMismatch("A, B have incompatible indices"))
         end
    end
    for j in indices(r,1)
        @inbounds( r[j] = evaluate(metric,A, j, B,j))
    end
    r
end

function pairwise_noalloc!(r::AbstractMatrix, metric::UnionMetrics, A::AbstractMatrix, B::AbstractMatrix)
    @boundscheck begin
        if !(indices(A,1) == indices(B,1))
             throw(DimensionMismatch("A, B have incompatible indices"))
         end
         if (indices(A,2) != indices(r,1)) |  (indices(B,2) != indices(r,2))#length(a) != length(b)
             throw(DimensionMismatch("r, A, B have incompatible indices"))
         end
    end
    for jA in indices(A,2)
        for jB in indices(B,2)
            @inbounds( r[jA,jB] = evaluate(metric,A, jA, B,jB))
        end
    end
    r
end

@enum hilb_state _TLTR _TLBL _BRTR _BRBL

function pairwise_advanced!(r::Matrix, metric::UnionMetrics, A::Matrix, B::Matrix, multithread::Bool = false, hilb::Bool = false, debug::Bool = false)
    #can be multithreaded. Can use hilbert order.
    #if not hilbert-order, then we waste some memory-bandwith / cache hits.
    #however, the added complexity does not seem to be worth it.
    #only works for one-based indexing.
    sz = sizeof(eltype(A))* size(A,1)
    #we optimize for 20kb of cache use. If we use more then hyperthreading will kill us dead
    cache_thresh::Int = max(4, convert(Int, round(Int, 10_000 / sz)))
    debug && println("cache_thresh: $(cache_thresh)")
    @boundscheck begin
        if !(indices(A,1) == indices(B,1))
             throw(DimensionMismatch("A, B have incompatible indices"))
         end
         if (indices(A,2) != indices(r,1)) |  (indices(B,2) != indices(r,2))#length(a) != length(b)
             throw(DimensionMismatch("r, A, B have incompatible indices"))
         end
    end
    nA = size(A,2)
    nB = size(B,2)

    if !(multithread && Threads.nthreads() > 1)
        if hilb
            debug && println("single threaded, using hilbert order:")
            pairwise_co_internal!(r, metric, A, 1,nA, B, 1,nB,cache_thresh,Val{_TLTR});
        else
            debug && println("single threaded, using naive order:")
            pairwise_co_internal!(r, metric, A, 1,nA, B, 1,nB,cache_thresh);
        end
    else
        #let's cut the matrix up
        #FIXME: This is terrible because I fail at multithreaded API

        sA = nA
        sB = nB
        nt = 1
        ntmax = Threads.nthreads()
        while ( (nt < ntmax) & ((sA>10) | (sB >10)) )
            nt = nt*2
            if sA > sB
                sA = div(sA,2)# (sA>>1)
            else
                sB = div(sB,2) # (sB >>1)
            end
        end
        todo = Vector{Tuple{Int,Int,Int,Int}}(nt)
        resize!(todo, 0)
        jA = 1
        while jA < nA
            jB = 1
            while jB < nB
                push!(todo, (jA, min(jA+sA, nA), jB, min(jB+sB, nB)))
                jB += sB
            end
            jA += sA
        end
      debug && ( println("using hilbert: $(hilb), split into $(nt) packages for $(ntmax) threads:");  dump(todo);)
        Threads.@threads for (jAs,jAe,jBs,jBe) in todo
            if hilb
                pairwise_co_internal!(r, metric, A, jAs,jAe, B, jBs,jBe,cache_thresh,Val{_TLTR});
            else
                pairwise_co_internal!(r, metric, A, jAs,jAe, B, jBs,jBe,cache_thresh);
            end
        end
    end
    r
end


function pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T}, A_start, A_end, B::Matrix{T}, B_start,B_end,cache_thresh::Int) where {T}

    #check whether we split.
    nA = A_end - A_start
    nB = B_end - B_start
   # println("called from $(A_start):$(A_end) / $(B_start):$(B_end), thresh $(cache_thresh)")
    if ( (nA > cache_thresh) & (nB > cache_thresh) )
        #full split.
        pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
            A_start, A_start + (nA>>1), B::Matrix{T},
            B_start, B_start + (nB>>1),cache_thresh)
        pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
            A_start, A_start + (nA>>1),B::Matrix{T},
            B_start + (nB>>1)+1, B_end,cache_thresh)
        pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
            A_start + (nA>>1)+1, A_end, B::Matrix{T},
            B_start + (nB>>1)+1, B_end,cache_thresh)
        pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
            A_start + (nA>>1)+1, A_end, B::Matrix{T},
            B_start, B_start + (nB>>1)+1,cache_thresh)
    elseif (nA > cache_thresh)
        pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
            A_start, A_start + (nA>>1), B::Matrix{T},
            B_start, B_end,cache_thresh)
        pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
            A_start + (nA>>1)+1, A_end, B::Matrix{T},
            B_start, B_end,cache_thresh)
    elseif (nB > cache_thresh)
        pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
            A_start,A_end, B::Matrix{T},
            B_start, B_start + (nB>>1), cache_thresh)
        pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
            A_start,A_end, B::Matrix{T},
            B_start + (nB>>1)+1, B_end,cache_thresh)
    else
        for jA = A_start:A_end
            for jB = B_start:B_end
                @inbounds( r[jA,jB] = evaluate(metric,A, jA, B,jB);)
            end
        end
    end
    ;
end


function pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T}, A_start, A_end, B::Matrix{T}, B_start,B_end,cache_thresh::Int, ::hstate) where {T, hstate}
    #uses hilbert curve as iteration order
    #check whether we split.
    nA = A_end - A_start
    nB = B_end - B_start
   # println("called from $(A_start):$(A_end) / $(B_start):$(B_end), thresh $(cache_thresh)")
    if ( (nA > cache_thresh) & (nB > cache_thresh) )
        #full split.
        if hstate == _TLTR
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start, A_start + (nA>>1), B::Matrix{T},
                B_start, B_start + (nB>>1),cache_thresh, Val{_TLBL})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start, A_start + (nA>>1),B::Matrix{T},
                B_start + (nB>>1)+1, B_end,cache_thresh, Val{_TLTR})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start + (nA>>1)+1, A_end, B::Matrix{T},
                B_start + (nB>>1)+1, B_end,cache_thresh, Val{_TLTR})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start + (nA>>1)+1, A_end, B::Matrix{T},
                B_start, B_start + (nB>>1)+1,cache_thresh, Val{_BRTR})
        elseif hstate == _TLBL
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start, A_start + (nA>>1), B::Matrix{T},
                B_start, B_start + (nB>>1),cache_thresh, Val{_TLTR})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start + (nA>>1)+1, A_end, B::Matrix{T},
                B_start, B_start + (nB>>1)+1,cache_thresh, Val{_TLBL})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start + (nA>>1)+1, A_end, B::Matrix{T},
                B_start + (nB>>1)+1, B_end,cache_thresh, Val{_TLBL})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start, A_start + (nA>>1),B::Matrix{T},
                B_start + (nB>>1)+1, B_end,cache_thresh, Val{_BRBL})
        elseif hstate == _BRTR
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start + (nA>>1)+1, A_end, B::Matrix{T},
                B_start + (nB>>1)+1, B_end,cache_thresh, Val{_BRBL})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start, A_start + (nA>>1),B::Matrix{T},
                B_start + (nB>>1)+1, B_end,cache_thresh, Val{_BRTR})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start, A_start + (nA>>1), B::Matrix{T},
                B_start, B_start + (nB>>1),cache_thresh, Val{_BRTR})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start + (nA>>1)+1, A_end, B::Matrix{T},
                B_start, B_start + (nB>>1)+1,cache_thresh, Val{_TLTR})
        else #hstate == _BRBL
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start + (nA>>1)+1, A_end, B::Matrix{T},
                B_start + (nB>>1)+1, B_end,cache_thresh,Val{_BRTR})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start + (nA>>1)+1, A_end, B::Matrix{T},
                B_start, B_start + (nB>>1)+1,cache_thresh, Val{_BRBL})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start, A_start + (nA>>1), B::Matrix{T},
                B_start, B_start + (nB>>1),cache_thresh, Val{_BRBL})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start, A_start + (nA>>1),B::Matrix{T},
                B_start + (nB>>1)+1, B_end,cache_thresh,Val{_TLBL})
        end #end hilbert_switch


    elseif (nA > cache_thresh)
        if hstate == _TLBR
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start, A_start + (nA>>1), B::Matrix{T},
                B_start, B_end,cache_thresh,Val{hstate})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start + (nA>>1)+1, A_end, B::Matrix{T},
                B_start, B_end,cache_thresh,Val{hstate})
        else
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start + (nA>>1)+1, A_end, B::Matrix{T},
                B_start, B_end,cache_thresh, Val{hstate})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start, A_start + (nA>>1), B::Matrix{T},
                B_start, B_end,cache_thresh,Val{hstate})
        end
    elseif (nB > cache_thresh)
        if hstate == _TLBL
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start,A_end, B::Matrix{T},
                B_start, B_start + (nB>>1), cache_thresh,Val{hstate})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start,A_end, B::Matrix{T},
                B_start + (nB>>1)+1, B_end,cache_thresh,Val{hstate})
        else
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start,A_end, B::Matrix{T},
                B_start + (nB>>1)+1, B_end,cache_thresh,Val{hstate})
            pairwise_co_internal!(r::Matrix{T}, metric::UnionMetrics, A::Matrix{T},
                A_start,A_end, B::Matrix{T},
                B_start, B_start + (nB>>1), cache_thresh,Val{hstate})
        end

    else
        for jA = A_start:A_end
            for jB = B_start:B_end
                @inbounds( r[jA,jB] = evaluate(metric,A, jA, B,jB);)
            end
        end
    end
    ;
end
