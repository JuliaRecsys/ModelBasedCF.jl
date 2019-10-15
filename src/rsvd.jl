mutable struct RegularizedSVD{T} <: MatrixFactorization{T}
    P::Array
    Q::Array
    preference::Persa.Preference{T}
    users::Int
    items::Int
end

const RSVD = RegularizedSVD

function RegularizedSVD(dataset::Persa.Dataset, features::Int)
    (users, items) = size(dataset)

    P = rand(users, features)
    Q = rand(items, features)

    return RegularizedSVD(P, Q, dataset.preference, Persa.users(dataset), Persa.items(dataset))
end

Persa.predict(model::RegularizedSVD, user::Int, item::Int) = model.P[user, :]' * model.Q[item, :]

function objective(model::RegularizedSVD, dataset::Persa.Dataset, λ::Float64)
    total = 0

    for (u, v, r) in dataset
        total += (r - model[u, v])^2
        total += λ * (norm(model.P[u,:])^2 + norm(model.Q[v,:])^2)
    end

    return total
end

function update!(model::RegularizedSVD, dataset::Persa.Dataset, γ::Float64, λ::Float64)

    idx = shuffle(1:length(dataset))

    for i = 1:length(dataset)
        (u, v, r) = dataset[idx[i]]

        e = r - Persa.predict(model, u, v)

        P = model.P[u,:]
        Q = model.Q[v,:]

        model.P[u,:] += γ * (e .* Q .- λ .* P)
        model.Q[v,:] += γ * (e .* P .- λ .* Q)
    end
end
