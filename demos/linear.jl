using SGD
using Stats
using DataFrames
using Distributions
using Vega
using VGPlot

n = 10_000
df = DataFrame()
df["Z"] = zeros(n)
df["X"] = randn(n)
df["Y"] = randn(n)
for i in 1:n
	s = 11.0 + 0.3425 * df[i, "X"] - 0.7198 * df[i, "Y"]
	df[i, "Z"] = s + rand(Normal())
end

vgplot(df, x = "X", y = "Z") + geom_point()
vgplot(df, x = "Y", y = "Z") + geom_point()

writetable("linear.csv", df)

for r in [1000, 2000, 3000, 4000, 5000]
	ds = openstream("linear.csv", nrows = r)
	for i in 1:50
		m = fit_sgd(LinearModel,
			        :(Z ~ X + Y),
			        ds,
			        epochs = i,
			        alpha = 0.1)
		@printf " * %5.d %3.d %3.d %.4f" r i m.epoch norm(m.gr)
		@printf " %.3f %.4f %.4f %.4f\n" SGD.cost(m) m.w[1] m.w[2] m.w[3]
	end
end

rm("linear.csv")
