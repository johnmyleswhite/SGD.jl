using SGD
using Stats
using DataFrames
using Distributions
using Vega
using VGPlot
using GLM

β = [11.7, 9.5, 7.3, 5.1]
μ = [0.0, 0.0, 0.0, 0.0]

for ε in [0.1, 1.0, 10.0]
    for σ in [0.1:0.1:0.9, 0.91:0.01:0.99]
        Σ = [1 σ σ σ;
             σ 1 σ σ;
             σ σ 1 σ;
             σ σ σ 1]
        path = "covariance.csv"
        d = MultivariateNormal(μ, Σ)
        N = 10_000
        X = rand(d, N)'
        y = 13.9 + X * β + ε * randn(N)
        df = DataFrame(X)
        df["y"] = y
        writetable("covariance.csv", df)
        ds = openstream("covariance.csv", nrows = N)
        m = fit_sgd(LinearModel,
                    :(y ~ x1 + x2 + x3 + x4),
                    ds, epochs = 25, alpha = 0.1)
        m2 = lm(:(y ~ x1 + x2 + x3 + x4), df)
        @printf " * %.3f %.3f %.3f %.3f\n" ε σ norm(m.gr) SGD.cost(m)
        @printf " -- %.4f %.4f %.4f %.4f %.4f\n" m.w[1] m.w[2] m.w[3] m.w[4] m.w[5]
        @printf " -- %.4f %.4f %.4f %.4f %.4f\n\n" coef(m2)[1] coef(m2)[2] coef(m2)[3] coef(m2)[4] coef(m2)[5]
    end
end

rm("covariance.csv")
