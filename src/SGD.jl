module SGD
	using Stats
	using DataFrames
	using Distributions

    export SGDModel, LinearModel, LogisticModel
    export fit_sgd, fit_sgd!, cost
    export predict!, residuals!, gradient!, update!

    include("abstract.jl")
    include("linear.jl")
    include("logistic.jl")
end
