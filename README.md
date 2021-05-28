# ModelBasedCF.jl

*Model based algorithms for Collaborative Filtering in Julia*

[![][ci-img]][ci-url]
[![][codecov-img]][codecov-url]

**Installation**: at the Julia REPL, `Pkg.add("ModelBasedCF")`

**Reporting Issues and Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## Example

```
julia> using DatasetsCF

julia> dataset = DatasetsCF.MovieLens();

julia> using ModelBasedCF

julia> model = ModelBasedCF.IRSVD(dataset, 10)

julia> Persa.train!(model, dataset, max_epochs = 10)

julia> model[1,1]
```

## Models

List of package models:

Models      | Title
-------------|------------------------------------------------------------------------
Baseline  | Koren, Y. (2009). Collaborative filtering with temporal dynamics. Knowledge Discovery and Data Mining {KDD}, 447–456.
Regularized SVD    | Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30–37.
Improved Regularized SVD   | Koren, Y. (2009). Collaborative filtering with temporal dynamics. Knowledge Discovery and Data Mining {KDD}, 447–456.


[ci-img]: https://img.shields.io/github/checks-status/JuliaRecsys/ModelBasedCF.jl/master?style=flat-square
[ci-url]: https://github.com/JuliaRecsys/ModelBasedCF.jl/actions

[codecov-img]: https://img.shields.io/codecov/c/github/JuliaRecsys/ModelBasedCF.jl?style=flat-square
[codecov-url]: https://codecov.io/gh/JuliaRecsys/ModelBasedCF.jl

[issues-url]: https://github.com/JuliaRecsys/ModelBasedCF.jl/issues
