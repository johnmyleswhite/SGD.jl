using SGD
using Stats
using DataFrames
using Distributions
using Vega
using VGPlot

n = 10_000
df = DataFrame()
df["Class"] = zeros(n)
df["X"] = randn(n)
df["Y"] = randn(n)
for i in 1:n
	s = -0.516 + 0.3425 * df[i, "X"] - 0.7198 * df[i, "Y"]
	df[i, "Class"] = rand(Bernoulli(invlogit(s)))
end

vgplot(df, x = "X", y = "Y", group = "Class") + geom_point()

writetable("logistic.csv", df)

for r in [1000, 2000, 3000, 4000, 5000]
	ds = openstream("logistic.csv", nrows = r)
	for i in 1:50
		m = fit_sgd(LogisticModel,
			        :(Class ~ X + Y),
			        ds,
			        epochs = i,
			        alpha = 0.1)
		@printf " * %5.d %3.d %3.d %.4f" r i m.epoch norm(m.gr)
		@printf " %.3f %.4f %.4f %.4f\n" SGD.cost(m) m.w[1] m.w[2] m.w[3]
	end
end

rm("logistic.csv")
