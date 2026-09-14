# Wrapper that turns `operator` into the deflated operator `P * operator` with
# `P = I - L * L'` the projector complementary to the span of the locked basis `L`. Since
# every vector it is applied to already lies in the orthogonal complement of `L`, this is
# equivalent to `P * operator * P`, which is again Hermitian; the Lanczos three-term
# recurrence therefore remains valid. The locked basis is referenced, not copied, so that
# locking additional vectors is immediately reflected here.
struct DeflatedOperator{F, T, O <: Orthogonalizer}
    operator::F
    locked::OrthonormalBasis{T}
    orth::O
end
function (op::DeflatedOperator)(x)
    w = apply(op.operator, x)
    isempty(op.locked.basis) && return w
    w, = orthogonalize!!(w, op.locked, op.orth)
    return w
end

function eigsolve(
        A, x₀, howmany::Int, which::Selector, alg::LockedLanczos;
        alg_rrule = Arnoldi(;
            tol = alg.tol,
            krylovdim = alg.krylovdim,
            maxiter = alg.maxiter,
            eager = alg.eager,
            orth = alg.orth
        )
    )
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    if krylovdim < 2
        error("krylov dimension $(krylovdim) too small, it should be at least 2")
    end

    ## FIRST ITERATION: setting up
    # Initialize Lanczos factorization; the locked basis is still empty at this point, so
    # the operator does not need to be deflated yet
    fact = initialize(LanczosIterator(A, x₀, alg.orth); verbosity = alg.verbosity)
    numops = 1
    numiter = 1
    sizehint!(fact, krylovdim)
    β = normres(fact)
    tol::typeof(β) = alg.tol

    # Storage for the locked (hard-deflated) eigenpairs. All pairs locked during a single
    # restart share the same residual direction `r / β`, so only one vector per locking
    # event needs to be kept in order to be able to report residuals at the end.
    S = typeof(β)
    Tv = eltype(basis(fact))
    locked = OrthonormalBasis{Tv}()
    lockedvalues = Vector{S}()
    lockednormres = Vector{S}()
    lockedresdirs = Vector{Tv}()
    lockedresevent = Vector{Int}()
    lockedrescoeff = Vector{S}()

    # From here on the Krylov subspace is built for the deflated operator
    iter = LanczosIterator(DeflatedOperator(A, locked, alg.orth), x₀, alg.orth)

    # allocate storage
    HH = fill(zero(eltype(fact)), krylovdim + 1, krylovdim)
    UU = fill(zero(eltype(fact)), krylovdim, krylovdim)

    converged = 0
    local D, U, f
    while true
        β = normres(fact)
        K = length(fact)
        nlocked = length(locked)

        # diagonalize Krylov factorization
        if β <= tol && nlocked + K < howmany
            if alg.verbosity >= WARN_LEVEL
                msg = "Invariant subspace of dimension $(nlocked + K) (up to requested tolerance `tol = $tol`), "
                msg *= "which is smaller than the number of requested eigenvalues (i.e. `howmany == $howmany`)."
                @warn msg
            end
        end
        if K == krylovdim || β <= tol || (alg.eager && nlocked + K >= howmany)
            U = copyto!(view(UU, 1:K, 1:K), I)
            f = view(HH, K + 1, 1:K)
            T = rayleighquotient(fact) # symtridiagonal

            # compute eigenvalues
            if K == 1
                D = [T[1, 1]]
                f[1] = β
                converged = Int(β <= tol)
            else
                if K < krylovdim
                    T = deepcopy(T)
                end
                D, U = tridiageigh!(T, U)
                by, rev = eigsort(which)
                p = sortperm(D; by = by, rev = rev)
                D, U = permuteeig!(D, U, p)
                mul!(f, view(U, K, :), β)
                converged = 0
                while converged < K && abs(f[converged + 1]) <= tol
                    converged += 1
                end
            end

            if nlocked + converged >= howmany || β <= tol
                break
            elseif alg.verbosity >= EACHITERATION_LEVEL
                nshow = min(howmany - nlocked, K)
                @info "LockedLanczos eigsolve in iteration $numiter, step = $K: $nlocked values locked, $(nlocked + converged) values converged, normres = $(normres2string(abs.(f[1:nshow])))"
            end
        end

        if K < krylovdim # expand Krylov factorization
            fact = expand!(iter, fact; verbosity = alg.verbosity)
            numops += 1
        else ## lock, shrink and restart
            if numiter == maxiter
                break
            end

            # Determine how many converged Ritz vectors to lock. The locking tolerance is
            # clamped from below by the attainable residual floor, since demanding more
            # accuracy than floating point allows would stall the iteration, and from above
            # by the convergence tolerance. The last requested eigenpair is never locked,
            # so that the active subspace always contributes at least one Ritz pair to the
            # final result.
            tol_lock = min(
                max(alg.tol_lock, (nlocked + K) * maximum(abs, D) * eps(one(β))),
                tol
            )
            nl = 0
            while nl < converged && nlocked + nl < howmany - 1 &&
                    abs(f[nl + 1]) <= tol_lock
                nl += 1
            end

            # Determine how many of the remaining Ritz vectors to keep active; strictly
            # smaller than `krylovdim - nl`, so that every restart makes progress
            keep = div(3 * krylovdim + 2 * (converged - nl), 5)
            keep = min(keep, krylovdim - nl)
            iszero(nl) && (keep = min(keep, krylovdim - 1))
            keep = max(keep, 1)

            # Restore Lanczos form in the `keep` columns following the locked ones
            H = fill!(view(HH, 1:(keep + 1), 1:keep), zero(eltype(HH)))
            @inbounds for j in 1:keep
                H[j, j] = D[nl + j]
                H[keep + 1, j] = f[nl + j]
            end
            Ukeep = view(U, :, (nl + 1):(nl + keep))
            @inbounds for j in keep:-1:1
                h, ν = householder(H, j + 1, 1:j, j)
                H[j + 1, j] = ν
                H[j + 1, 1:(j - 1)] .= zero(eltype(H))
                lmul!(h, H)
                rmul!(view(H, 1:j, :), h')
                rmul!(Ukeep, h')
            end
            @inbounds for j in 1:keep
                fact.αs[j] = H[j, j]
                fact.βs[j] = H[j + 1, j]
            end

            # Rotate the basis to the Ritz vectors: the first `nl` of these are locked
            # away, the next `keep` are retained as the new active basis
            B = basis(fact)
            B = basistransform!(B, view(U, :, 1:(nl + keep)))
            if nl > 0
                push!(lockedresdirs, scale(residual(fact), 1 / β))
                @inbounds for i in 1:nl
                    push!(locked, B[i])
                    push!(lockedvalues, D[i])
                    push!(lockednormres, abs(f[i]))
                    push!(lockedresevent, length(lockedresdirs))
                    push!(lockedrescoeff, f[i])
                end
                @inbounds for j in 1:keep
                    B[j] = B[nl + j]
                end
            end
            r = residual(fact)
            B[keep + 1] = scale!!(r, 1 / β)

            # Shrink Lanczos factorization
            fact = shrink!(fact, keep; verbosity = alg.verbosity)
            numiter += 1
        end
    end

    nlocked = length(locked)
    howmany′ = howmany
    if nlocked + converged > howmany
        howmany′ = nlocked + converged
    elseif nlocked + length(D) < howmany
        howmany′ = nlocked + length(D)
    end
    nactive = howmany′ - nlocked
    values = append!(copy(lockedvalues), view(D, 1:nactive))

    # Compute eigenvectors
    V = view(U, :, 1:nactive)

    # Compute convergence information. Locked pairs report the residual they had at the
    # moment they were locked, measured with respect to the deflated operator.
    vectors = let B = basis(fact)
        vcat(copy(locked.basis), [B * v for v in cols(V)])
    end
    residuals = let r = residual(fact)
        vcat(
            [scale(lockedresdirs[lockedresevent[i]], lockedrescoeff[i]) for i in 1:nlocked],
            [scale(r, last(v)) for v in cols(V)]
        )
    end
    normresiduals = let f = f
        vcat(lockednormres, [abs(f[i]) for i in 1:nactive])
    end
    converged = nlocked + converged

    if (converged < howmany) && alg.verbosity >= WARN_LEVEL
        @warn """LockedLanczos eigsolve stopped without convergence after $numiter iterations:
        * $converged eigenvalues converged ($nlocked of which locked)
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info """LockedLanczos eigsolve finished after $numiter iterations:
        * $converged eigenvalues converged ($nlocked of which locked)
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    end

    return values, vectors,
        ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end
