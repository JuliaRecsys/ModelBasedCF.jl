using Persa
using Test
using DatasetsCF

using ModelBasedCF

dataset = DatasetsCF.MovieLens()

@testset "Random Model Tests" begin
    model = ModelBasedCF.RandomModel(dataset)
    Persa.train!(model, dataset)

    @test !isnan(model[1,1])
end

@testset "Factorization Matrix Models Tests" begin
    @testset "Baseline Tests" begin
        model = ModelBasedCF.Baseline(dataset)
        Persa.train!(model, dataset, max_epochs = 1)

        @test !isnan(model[1,1])
    end

    @testset "RSVD Tests" begin
        model = ModelBasedCF.RSVD(dataset, 1)
        Persa.train!(model, dataset, max_epochs = 1)

        @test !isnan(model[1,1])
    end

    @testset "IRSVD Tests" begin
        model = ModelBasedCF.IRSVD(dataset, 1)
        Persa.train!(model, dataset, max_epochs = 1)

        @test !isnan(model[1,1])
    end
end