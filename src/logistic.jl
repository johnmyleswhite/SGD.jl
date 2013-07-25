type LogisticModel <: SGDModel
    w::Vector{Float64} # Weights
    gr::Vector{Float64} # Gradient
    pr::Vector{Float64} # Predictions
    r::Vector{Float64} # Residuals
    n::Int # Number of observations processed
    epoch::Int # Number of passes through entire data
    alpha::Float64 # Learning rate
end

function LogisticModel(p::Integer, n::Integer)
    LogisticModel(zeros(p),
                  Array(Float64, p),
                  Array(Float64, n),
                  Array(Float64, n),
                  0,
                  0,
                  0.1)
end

function predict!(m::LogisticModel, X::Matrix{Float64})
    p, n = size(X)
    for i = 1:n
        m.pr[i] = invlogit(dot(X[:, i], m.w))
    end
    return
end

function residuals!(m::LogisticModel, X::Matrix{Float64}, y::Vector{Float64})
    p, n = size(X)
    for i = 1:n
        m.r[i] = y[i] - m.pr[i]
    end
    return
end

function gradient!(m::LogisticModel, X::Matrix{Float64}, y::Vector{Float64})
    p, n = size(X)
    fill!(m.gr, 0.0)
    for i in 1:n
        residual = m.r[i]
        for o in 1:p
            m.gr[o] += residual * -X[o, i]
        end
    end
    for o in 1:p
        m.gr[o] /= n
    end
    return
end

# TODO: Dynamically resize vectors when input has improper length
function update!(m::LogisticModel,
                 X::Matrix{Float64},
                 y::Vector{Float64})
    p, n = size(X)

    # Increment the number of examples we've seen
    m.n += n

    # Evaluate the gradient on these examples
    predict!(m, X)
    residuals!(m, X, y)
    gradient!(m, X, y)

    # Update weights using gradient
    for i in 1:length(m.w)
        m.w[i] = m.w[i] - m.alpha * m.gr[i]
        if isnan(m.w[i])
            error("Weight updates produced NaN's")
        end
    end

    return
end

function cost(m::LogisticModel)
    s = 0.0
    for i in 1:length(m.r)
        p = m.pr[i]
        y = m.r[i] + p
        s += log(p) * y + log(1 - p) * (1 - y)
        # s += m.r[i]^2
    end
    return s
end

function fit_sgd(::Type{LogisticModel},
                 ex::Expr,
                 ds::DataFrames.AbstractDataStream;
                 epochs::Int64 = 5,
                 alpha::Float64 = 0.1)
    df = start(ds)
    if done(ds, df)
        error("DataStream is empty")
    end
    df, df = next(ds, df)
    mf = ModelFrame(ex, df)
    X = ModelMatrix(mf).m'
    y = vector(mf.df[1])
    p, n = size(X)
    m = LogisticModel(p, n)
    m.alpha = alpha
    m.epoch = 1
    SGD.update!(m, X, y)
    while !done(ds, df)
        df, df = next(ds, df)
        mf = ModelFrame(ex, df)
        X = ModelMatrix(mf).m'
        y = vector(mf.df[1])
        SGD.update!(m, X, y)
    end
    fit_sgd!(m, ex, ds, epochs = epochs - 1)
    return m
end
