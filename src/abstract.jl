abstract SGDModel

function fit_sgd!(m::SGDModel,
                  ex::Expr,
                  ds::DataFrames.AbstractDataStream;
                  epochs::Int64 = 5,
                  alpha::Float64 = 0.1,
                  tol::Float64 = 1e-4)
    c0, c1 = Inf, cost(m)
    for epoch = 1:epochs
        m.epoch += 1
        for df in ds
            mf = ModelFrame(ex, df)
            X = ModelMatrix(mf).m'
            y = vector(mf.df[1])
            SGD.update!(m, X, y)
        end
        c0, c1 = c1, cost(m)
        if abs(c0 - c1) < tol
            break
        end
    end
    return m
end
