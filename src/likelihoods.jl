"""
T: number of obvervations
M: number of joint prob
K: number of hidden states
"""
function likelihoods!(L::AbstractArray, hmm::AbstractHMM{Univariate}, observations)
    T, M, K = size(observations, 2), size(observations, 1), size(hmm, 1)
    @argcheck size(L) == (T, M, K)
    @inbounds for k in OneTo(K), t in OneTo(T), m in OneTo(M)
        L[t,m,k] = pdf(hmm.B[m, k], observations[m, t])
    end
end

function loglikelihoods!(LL::AbstractArray, hmm::AbstractHMM{Univariate}, observations)
    T, M, K = size(observations, 2), size(observations, 1), size(hmm, 1)
    @argcheck size(LL) == (T, M, K)
    @inbounds for k in OneTo(K), t in OneTo(T), m in OneTo(M)
        LL[t,m,k] = logpdf(hmm.B[m, k], observations[m, t])
    end
end

function likelihoods!(L::AbstractArray, hmm::AbstractHMM{Multivariate}, observations)
    T, M, K = size(observations, 2), size(observations, 1), size(hmm, 1)
    @argcheck size(L) == (T, M, K)
    @inbounds for k in OneTo(K), t in OneTo(T), m in OneTo(M)
        L[t,m,k] = pdf(hmm.B[m, k], view(observations, m, t))
    end
end

function loglikelihoods!(LL::AbstractArray, hmm::AbstractHMM{Multivariate}, observations)
    T, M, K = size(observations, 2), size(observations, 1), size(hmm, 1)
    @argcheck size(LL) == (T, M, K)
    @inbounds for k in OneTo(K), t in OneTo(T), m in OneTo(M)
        LL[t,m,k] = logpdf(hmm.B[m, k], view(observations, m, t))
    end
end


# function likelihoods(hmm::AbstractHMM, observations; logl = false, robust = false)
#     T, M, K = size(observations, 2), size(observations, 1), size(hmm, 1)
#     L = Array{Float64}(undef,T,M,K)#Matrix{Float64}(undef, T, M, K)
#
#     if logl
#         loglikelihoods!(L, hmm, observations)
#         robust && replace!(L, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
#     else
#         likelihoods!(L, hmm, observations)
#         robust && replace!(L, -Inf => nextfloat(-Inf), Inf => prevfloat(Inf))
#     end
#
#     L
# end
# hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1) Normal(10,1); Normal(10,1) Normal(0,1)])
# y = rand(hmm, 1000)
# likelihoods(hmm, y)
# hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1) Normal(10,1); Normal(10,1) Normal(0,1)])
# obs = [rand(hmm,10);rand(hmm,10)];
# obs = reshape(obs,2,10);
# size(obs);
# likelihoods(hmm, obs)





#
# T, M, K = size(obs, 2), size(obs, 1), size(hmm, 1)
# observations = obs
# L = Array{Float64}(undef,T,M,K)
# observations[1, 2]
# hmm.B[1,2]
#
# L[1,1,1] = pdf(hmm.B[1, 1], observations[1, 1])
#
# for k in OneTo(K), t in OneTo(T), m in OneTo(M)
#     L[t,m,k] = pdf(hmm.B[m, k], observations[m, t])
# end
