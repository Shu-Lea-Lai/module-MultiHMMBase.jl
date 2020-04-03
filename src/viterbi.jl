# Original implementations by @nantonel
# https://github.com/maxmouchet/HMMBase.jl/pull/6

# *log! methods use the samples log-likelihood instead of the likelihood.
function viterbi!(T1::AbstractArray, T2::AbstractArray,
    z::AbstractVector, a::AbstractVector, A::AbstractMatrix, L::AbstractArray)
    # @argcheck size(T1, 1) == size(T2, 1) == size(L, 1) == size(z, 1)
    # @argcheck size(T1, 2) == size(T2, 2) == size(L, 2)
    # @argcheck size(T1, 3) == size(T2, 3) == size(L, 3) == size(a, 1) == size(A, 1) == size(A, 2)
    @argcheck size(T1, 1) == size(T2, 1) == size(L, 1) == size(z, 1)
    @argcheck size(T1, 2) == size(T2, 2) == size(L, 3) == size(a, 1) == size(A, 1) == size(A, 2)

    T, M, K = size(L)
    (T == 0) && return

    fill!(T1, 0.0)
    fill!(T2, 0)

    c = 0.0

    for i in OneTo(K)
        T1[1,i] = a[i] * L[1,1,i]
        for m in 2:M
            T1[1,i] = T1[1,i] * L[1,m,i]
        end
        c += T1[1,i]
    end

    for i in OneTo(K)
        T1[1,i] /= c
    end

    @inbounds for t in 2:T
        c = 0.0

        for j in OneTo(K)
            # TODO: If there is NaNs in T1 this may
            # stay to 0 (NaN > -Inf == false).
            # Hence it will crash when computing z[t].
            # Maybe we should check for NaNs beforehand ?
            amax = 0
            vmax = -Inf

            for i in OneTo(K)
                v = T1[t-1,i] * A[i,j]
                if v > vmax
                    amax = i
                    vmax = v
                end
            end

            vCalc = L[t,1,j]
            for m in 2:M
                vCalc = vCalc*L[t,m,j]
            end
            T1[t,j] = vmax + vCalc
            T2[t,j] = amax
            c += T1[t,j]
        end

        for i in OneTo(K)
            T1[t,i] /= c
        end
    end

    z[T] = argmax(T1[T,:])
    for t in T-1:-1:1
        z[t] = T2[t+1,z[t+1]]
    end
end

function viterbilog!(T1::AbstractArray, T2::AbstractArray, z::AbstractVector, a::AbstractVector, A::AbstractMatrix, LL::AbstractArray)
    T, M, K = size(LL)
    (T == 0) && return

    fill!(T1, 0.0)
    fill!(T2, 0)

    al = log.(a)
    Al = log.(A)

    for i in OneTo(K)
        T1[1,i] = a[i] + LL[1,1,i]
        for m in 2:M
            T1[1,i] = T1[1,i] + LL[1,m,i]
        end
    end

    @inbounds for t in 2:T
        for j in OneTo(K)
            amax = 0
            vmax = -Inf

            for i in OneTo(K)
                v = T1[t-1,i] + Al[i,j]
                if v > vmax
                    amax = i
                    vmax = v
                end
            end

            vCalc = L[t,1,j]
            for m in 2:M
                vCalc = vCalc + LL[t,m,j]
            end
            T1[t,j] = vmax + vCalc
            T2[t,j] = amax
        end
    end

    z[T] = argmax(T1[T,:])
    for t in T-1:-1:1
        z[t] = T2[t+1,z[t+1]]
    end
end
