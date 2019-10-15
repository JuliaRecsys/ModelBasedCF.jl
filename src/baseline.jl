mutable struct Baseline{T} <: MatrixFactorization{T}
    μ::Float64
    bias_user::Array
    bias_item::Array
    preference::Persa.Preference{T}
    users::Int
    items::Int
end

function Baseline(dataset::Persa.Dataset)
    (users, items) = size(dataset)

    μ = mean(dataset)
    bu = zeros(users)
    bv = zeros(items)

    return Baseline(μ, bu, bv, dataset.preference, Persa.users(dataset), Persa.items(dataset))
end

Persa.predict(model::Baseline, user::Int, item::Int) = model.μ + model.bias_user[user] + model.bias_item[item]

function objective(model::Baseline, dataset::Persa.Dataset, λ::Float64)
    total = 0

    for (u, v, r) in dataset
        total += (r - model[u, v])^2
        total += λ * (model.bias_user[u]^2 + model.bias_item[v]^2)
    end

    return total
end

function update!(model::Baseline, dataset::Persa.Dataset, γ::Float64, λ::Float64)

    idx = shuffle(1:length(dataset))

    for i = 1:length(dataset)
        (u, v, r) = dataset[idx[i]]

        e = r - Persa.predict(model, u, v)

        model.bias_user[u,:] += γ * (e .- λ * model.bias_user[u,:]);
        model.bias_item[v,:] += γ * (e .- λ * model.bias_item[v,:]);
    end
end
