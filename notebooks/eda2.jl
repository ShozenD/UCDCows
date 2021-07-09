### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 0fc79ee8-c951-11eb-0090-c34e070479ea
begin
	using Pkg
	Pkg.activate("../")
end

# ╔═╡ 32efb3a2-934a-4ddb-bc25-3b0f5c7e7955
using 
	CSV, 
	DataFrames, 
	DataFramesMeta,
	CategoricalArrays,
	Dates,
	Statistics,
	GLM, 
	Gadfly,
	Glob

# ╔═╡ 4033ba99-3145-453e-8f3b-c57ba9550e3a
fpaths = glob("../data/QMPS*");

# ╔═╡ 8c92abe2-c243-4653-926b-2c91fe4bf7bb
df = CSV.read(fpaths[1], DataFrame)

# ╔═╡ 35bceea6-5d5e-44e6-ad34-980304441487
md"### MDi"

# ╔═╡ bbb373b1-c5fb-4279-aefa-1ac8a60a18fe
plot(df, x="MDi", Geom.histogram)

# ╔═╡ 5097f2c3-69cc-4527-a3f0-0c9c4f0f5dea
plot(df, x="Total Conductivity", y = "MDi", Geom.point)

# ╔═╡ 5ff3faf1-19e0-47df-a57a-e2194c8d5cbe
transform()

# ╔═╡ 02c19ca7-d602-4f6f-b498-b25888bcb957
select(df, ["Conductivity RR", "Conductivity RF", "Conductivity LR", "Conductivity LF"])

# ╔═╡ 919e92cb-9ec1-4149-87c5-e6809b53644a
function zero_to_missing(x)	
	if x === 0.0
		return missing
	else
		return x
	end
end

# ╔═╡ ca9e77ca-0958-4ae2-b388-40b5831f4027
begin
	select!(df, :, ["Conductivity RR"] .=> ByRow(zero_to_missing) => :condRR)
	select!(df, :, ["Conductivity RF"] .=> ByRow(zero_to_missing) => :condRF)
	select!(df, :, ["Conductivity LR"] .=> ByRow(zero_to_missing) => :condLR)
	select!(df, :, ["Conductivity LF"] .=> ByRow(zero_to_missing) => :condLF)
end;

# ╔═╡ 5e1048c5-d69b-41c4-879c-07b53e8726e4
select(df, :, [:condRR, :condRF, :condLR, :condLF] => (a,b,c,d) -> mean(skipmissing([a,b,c,d])) => :avgcond)

# ╔═╡ a540fbf9-3b1a-42c3-b199-04aee0bd1fa2


# ╔═╡ 81322876-c1f7-4c5f-ab0a-eba07a15ed3e


# ╔═╡ c4fedf1b-12af-4a3d-a36b-835c6f392777


# ╔═╡ Cell order:
# ╠═0fc79ee8-c951-11eb-0090-c34e070479ea
# ╠═32efb3a2-934a-4ddb-bc25-3b0f5c7e7955
# ╠═4033ba99-3145-453e-8f3b-c57ba9550e3a
# ╠═8c92abe2-c243-4653-926b-2c91fe4bf7bb
# ╟─35bceea6-5d5e-44e6-ad34-980304441487
# ╠═bbb373b1-c5fb-4279-aefa-1ac8a60a18fe
# ╠═5097f2c3-69cc-4527-a3f0-0c9c4f0f5dea
# ╠═5ff3faf1-19e0-47df-a57a-e2194c8d5cbe
# ╠═02c19ca7-d602-4f6f-b498-b25888bcb957
# ╠═919e92cb-9ec1-4149-87c5-e6809b53644a
# ╠═ca9e77ca-0958-4ae2-b388-40b5831f4027
# ╠═5e1048c5-d69b-41c4-879c-07b53e8726e4
# ╠═a540fbf9-3b1a-42c3-b199-04aee0bd1fa2
# ╠═81322876-c1f7-4c5f-ab0a-eba07a15ed3e
# ╠═c4fedf1b-12af-4a3d-a36b-835c6f392777
