using Test
using MadNLP
using MadNLPTests
using MadNLPGLU

@testset "RFSolver Test" begin

    @testset "Interface" begin
        MadNLPTests.test_linear_solver(MadNLPGLU.RFSolver, Float64)
    end

    @testset "HS15 Test" begin
        nlp = MadNLPTests.HS15Model()
        solver = MadNLP.MadNLPSolver(nlp; linear_solver=MadNLPGLU.RFSolver, print_level=MadNLP.ERROR)
        MadNLP.solve!(solver)
        @test solver.status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "MadNLP Test" begin
        constructor = () -> MadNLP.Optimizer(
            linear_solver=MadNLPGLU.RFSolver,
            print_level=MadNLP.ERROR,
        )
        test_madnlp("cusolverRF", constructor, ["eigmina"])
    end

end

