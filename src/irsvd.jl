mutable struct ImprovedRegularizedSVD <: MatrixFactorization
    μ::Float64
    bias_user::Array
    bias_item::Array
    P::Array
    Q::Array
    preference::Persa.Preference
    users::Int
    items::Int
end

const IRSVD = ImprovedRegularizedSVD

function ImprovedRegularizedSVD(dataset::Persa.Dataset, features::Int)
    (users, items) = size(dataset)

    μ = mean(dataset)
    bu = zeros(users)
    bv = zeros(items)

    P = rand(users, features)
    Q = rand(items, features)

    return ImprovedRegularizedSVD(μ, bu, bv, P, Q, dataset.preference, Persa.users(dataset), Persa.items(dataset))
end

Persa.predict(model::ImprovedRegularizedSVD, user::Int, item::Int) = model.μ + model.bias_user[user] + model.bias_item[item] + model.P[user, :]' * model.Q[item, :]

function objective(model::ImprovedRegularizedSVD, dataset::Persa.Dataset, λ::Float64)
    total = 0

    for (u, v, r) in dataset
        total += (r - model[u, v]) ^ 2
        total += λ * (model.bias_user[u] ^ 2 + model.bias_item[v] ^ 2)
        total += λ * (norm(model.P[u,:]) ^ 2 + norm(model.Q[v,:]) ^ 2)
    end

    return total
end
