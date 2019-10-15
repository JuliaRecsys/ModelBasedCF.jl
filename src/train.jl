function Persa.train!(model::MatrixFactorization,
                dataset::Persa.Dataset;
                γ = 0.001,
                λ = 0.02,
                max_epochs = 1000)

    err = Inf
    p = Progress(max_epochs)
    for epoch = 1:max_epochs
        update!(model, dataset, γ, λ)
        e = objective(model, dataset, 0.1)

        if e > err
            break
        end

        err = e

        ProgressMeter.next!(p; showvalues = [(:epoch, epoch), (:error, err)])
    end

    return nothing
end
