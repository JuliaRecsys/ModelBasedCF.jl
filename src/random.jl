using Random

mutable struct RandomModel <: Persa.Model
    preference::Persa.Preference
    users::Int
    items::Int
end

RandomModel(dataset::Persa.Dataset) = RandomModel(dataset.preference, Persa.users(dataset), Persa.items(dataset))

Persa.predict(model::RandomModel, user::Int, item::Int) = rand(model.preference.possibles)
