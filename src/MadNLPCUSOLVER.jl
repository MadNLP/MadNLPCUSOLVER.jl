module MadNLPCUSOLVER

import SparseArrays: SparseMatrixCSC

import MadNLP

import CUDA
import CUDA: CUSPARSE, CUSOLVER
import CUSOLVERRF

@kwdef struct RFSolverOptions <: MadNLP.AbstractOptions
    symbolic_analysis::Symbol = :klu
    fast_mode::Bool = true
    factorization_algo::CUSOLVER.cusolverRfFactorization_t = CUSOLVER.CUSOLVERRF_FACTORIZATION_ALG0
    triangular_solve_algo::CUSOLVER.cusolverRfTriangularSolve_t = CUSOLVER.CUSOLVERRF_TRIANGULAR_SOLVE_ALG1
end

struct RFSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::CUSOLVERRF.RFLowLevel

    tril::SparseMatrixCSC{T}
    full::SparseMatrixCSC{T}
    tril_to_full_view::MadNLP.SubVector{T}
    K::CUSPARSE.CuSparseMatrixCSR{T}

    buffer::CUDA.CuVector{T}
    opt::RFSolverOptions
    logger::MadNLP.MadNLPLogger
end

function RFSolver(
    csc::SparseMatrixCSC{Float64};
    opt=RFSolverOptions(),
    logger=MadNLP.MadNLPLogger(),
)
    n, m = size(csc)
    @assert n == m

    full,tril_to_full_view = MadNLP.get_tril_to_full(csc)
    full.nzval .= tril_to_full_view

    sym_lu = CUSOLVERRF.klu_symbolic_analysis(full)
    inner = CUSOLVERRF.RFLowLevel(
        sym_lu;
        fast_mode=opt.fast_mode,
        factorization_algo=opt.factorization_algo,
        triangular_algo=opt.triangular_solve_algo,
    )
    K = CUSPARSE.CuSparseMatrixCSR(full)
    buffer = CUDA.zeros(Float64, n)

    return RFSolver{Float64}(
        inner, csc, full, tril_to_full_view, K, buffer,
        opt, logger,
    )
end

function MadNLP.factorize!(M::RFSolver)
    M.full.nzval .= M.tril_to_full_view
    copyto!(M.K.nzVal, M.full.nzval)
    CUSOLVERRF.rf_refactor!(M.inner, M.K)
    return M
end

function MadNLP.solve!(M::RFSolver{Float64}, x::Vector{Float64})
    copyto!(M.buffer, x)
    CUSOLVERRF.rf_solve!(M.inner, M.buffer)
    copyto!(x, M.buffer)
    return x
end

MadNLP.input_type(::Type{RFSolver}) = :csc
MadNLP.default_options(::Type{RFSolver}) = RFSolverOptions()
MadNLP.is_inertia(M::RFSolver) = false
MadNLP.improve!(M::RFSolver) = false
MadNLP.is_supported(::Type{RFSolver},::Type{Float32}) = false
MadNLP.is_supported(::Type{RFSolver},::Type{Float64}) = true

MadNLP.introduce(M::RFSolver) = "cuSolverRF"

end # module MadNLPGLU
