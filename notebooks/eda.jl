### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ e912b9a2-a7ef-11eb-2117-9373bc3ef946
begin
	using Pkg
	Pkg.activate("../")
end

# ╔═╡ 0f96552f-12e1-46a1-b576-a78a8d7df9f6
using 
	CSV, 
	DataFrames, 
	DataFramesMeta,
	Statistics,
	Impute,
	GLM, 
	Gadfly,
	PlutoUI

# ╔═╡ 4d93dd33-38b6-4d6f-b836-f096c29236dc
begin
	include("../src/visualization.jl")
	include("../src/utilities.jl")
end

# ╔═╡ 75359ed1-dc72-41f9-9d73-6d26d2d2cd64
md"# Exploratory Data Analysis"

# ╔═╡ 7d3634f5-9bb9-45e9-a2c8-122a155ffb50
df = CSV.read("../data/analytical/cows-analytic.csv", DataFrame);

# ╔═╡ a9bd609b-8478-4c14-ae42-5822ff9341f1
md"## Cleaning"

# ╔═╡ 01beb164-64e2-48da-8945-a758b61b9b36
md"## Basic Descriptive Stats"

# ╔═╡ 18188ad5-3f8b-4ef8-a4e9-e2901f539d9f
describe(df)

# ╔═╡ 49de5332-a396-4494-bd61-c43a3ac18318
md"The number of observation for each lactation number"

# ╔═╡ 1d343f02-eac9-4d97-b6b6-83f9054d20ad
@by(df, :lactnum, obs = length(:id)) |> x -> sort(x, :lactnum)

# ╔═╡ cb95ba52-b752-4d59-8d0d-3eb84bbf9f10
md"The average number of data points per cow"

# ╔═╡ 35230364-60bc-44cd-b1f3-1ac71ea836e9
@by(df, :id, obs = length(:id)) |> describe

# ╔═╡ a2216dc3-bab6-475f-a4b5-8c11af6a3509
begin
	# Subset only those cows on their first or second lactation cycle
	df2 = @where(df, :lactnum .<= 2);
	# A vector of cow IDs
	cowid = unique(df2.id);
end

# ╔═╡ 31c4e712-8a37-4399-a65f-2c4083c146a9
md"## Conductivity and Yield"

# ╔═╡ 0bc4324e-5528-4185-a28e-21a65aa84dbe
md"### Individual Time Series"

# ╔═╡ b170c090-e177-4717-bba1-0f4565b816b1
md"**Cow Number**"

# ╔═╡ ee10c4d7-07b7-4644-8ada-eccdf35f30ba
@bind cid NumberField(1:length(cowid), default=1)

# ╔═╡ b4abb700-32f1-434e-afdf-2ba169fbe822
md"**MDi threshold**"

# ╔═╡ 0fcb820f-6777-4057-b950-7f1b58898bcc
@bind thmdi NumberField(1.0:0.1:maximum(unique(skipmissing(df2.mdi))), default = 1.0)

# ╔═╡ 85e0d4f3-0c34-49a8-b8de-c461255adda0
md"**Smoothness** (Rolling Average)"

# ╔═╡ a578fd8d-9107-4990-90d9-3d3072461ed8
@bind sm Slider(1:8, default = 1, show_value=true)

# ╔═╡ 5fc3552f-b61d-4840-a3ba-83c46e16088e
md"Cow ID: $(cowid[cid])"

# ╔═╡ 8c285f45-3c39-4cf5-8d36-fd7c6731f133
begin
	set_default_plot_size(6inch, 7inch)
	p_cond = plot_conductivity(df2, cowid[cid], convert(Float64, thmdi), sm)
	p_yield = plot_yield(df2, cowid[cid], convert(Float64, thmdi), sm)
	vstack(p_cond, p_yield)
end

# ╔═╡ 691dda5d-1060-4830-a4f4-7372711db25e
md"## Yield, Conductivity, and MDi"

# ╔═╡ 6a16eb40-cb58-4718-801c-7aa564068485
md"### MDi and Yield"

# ╔═╡ e9af2979-b21a-4dca-b5f2-6f6bc7f10c73
begin
	function mctyield(D::DataFrame)
	  # Filter out missing values
	  D = @where(D, :condrr .!= !ismissing(:condrr))
	  D = @where(D, :condrf .!= !ismissing(:condrf))
	  D = @where(D, :condlr .!= !ismissing(:condlr))
	  D = @where(D, :condlf .!= !ismissing(:condlf))

	  Nr = nrow(D)

	  D.mct = [minmax_teat(D, i; min=false) for i in 1:Nr]

	  y = zeros(Nr)
	  z = zeros(Nr)
	  negteats = Dict([
		:rr => [2,3,4],
		:rf => [1,3,4],
		:lr => [1,2,4],
		:lf => [1,2,3],
		:NA => [1,2,3,4]
	  ])
	  posteats = Dict([:rr => 1, :rf => 2, :lr => 3, :lf => 4])
	  for i in 1:Nr
		yield = [D.yieldrr[i], D.yieldrf[i], D.yieldlr[i], D.yieldlf[i]]
		y[i] += yield[posteats[D.mct[i]]] 
		z[i] += mean(yield[negteats[D.mct[i]]])
	  end

	  D.mctyield = y
	  D.nonmctyield = z

	  return D
	end
	
	function mctcond(D::DataFrame)
	  # Filter out missing values
	  D = @where(D, :condrr .!= !ismissing(:condrr))
	  D = @where(D, :condrf .!= !ismissing(:condrf))
	  D = @where(D, :condlr .!= !ismissing(:condlr))
	  D = @where(D, :condlf .!= !ismissing(:condlf))

	  Nr = nrow(D)

	  D.mct = [minmax_teat(D, i; min=false) for i in 1:Nr]

	  y = zeros(Nr)
	  z = zeros(Nr)
	  negteats = Dict([
		:rr => [2,3,4],
		:rf => [1,3,4],
		:lr => [1,2,4],
		:lf => [1,2,3],
		:NA => [1,2,3,4]
	  ])
	  posteats = Dict([:rr => 1, :rf => 2, :lr => 3, :lf => 4])
	  for i in 1:Nr
		cond = [D.condrr[i], D.condrf[i], D.condlr[i], D.condlf[i]]
		y[i] += cond[posteats[D.mct[i]]] 
		z[i] += mean(cond[negteats[D.mct[i]]])
	  end

	  D.mctcond = y
	  D.nonmctcond = z

	  return D
	end

	df3 = mctyield(df2)
	df3 = mctcond(df3)
	df3 = @where(df3, :mdi .!= ismissing(:mdi))
	
	dfmctcond = DataFrame([
	  :type => [repeat(["max cond teat"], nrow(df3)); 
				repeat(["non max cond teats"], nrow(df3))],
	  :cond => [df3.mctcond; df3.nonmctcond],
	  :mdi => repeat(df3.mdi, 2)
	])

	dfmctyield = DataFrame([
	  :type => [repeat(["max cond teat"], nrow(df3)); 
				repeat(["non max cond teats"], nrow(df3))],
	  :yield => [df3.mctyield; df3.nonmctyield],
	  :mdi => repeat(df3.mdi, 2)
	])
end;

# ╔═╡ 3f21086e-2d47-40c4-bf3e-581d63f92584
begin
	set_default_plot_size(7inch, 4inch)
	plot(dfmctcond, 
		x = :mdi, y =:cond, 
		color =:type, 
		Geom.boxplot,
		Scale.color_discrete_manual(colorant"#de425b",colorant"#488f31"),
		Theme(
			key_position = :bottom,
			background_color = "white"
		)
	)
end

# ╔═╡ 36ce1bf5-3c4f-4c7b-a65c-65116de80fe5
begin
	set_default_plot_size(7inch, 4inch)
	plot(dfmctyield, 
		x = :mdi, y =:yield, 
		color =:type, 
		Geom.boxplot,
		Scale.color_discrete_manual(colorant"#de425b",colorant"#488f31"),
		Theme(
			key_position = :bottom,
			background_color = "white"
		)
	)
end

# ╔═╡ 0b4e16e0-a3f6-4976-9e66-c3b768898082
md"### Yield During Warning"

# ╔═╡ 7ee46d6a-a5d6-466d-a5c1-dfe1560054e2
md"**MDi Threshold**"

# ╔═╡ 34590bff-17b2-497d-a9ea-d6fabc0767d7
@bind thmdi2 NumberField(1.1:0.1:maximum(unique(skipmissing(df2.mdi))), default = 1.1)

# ╔═╡ 33ae8738-3222-4dfa-af2a-a0e1099bc4f8
begin
	abnrmcow = @where(df, :mdi .> thmdi2)
	abnrmcowid = unique(abnrmcow.id)
end;

# ╔═╡ a04b89bf-1285-4e24-a17a-a4b90ae36aa8
begin
	Yield = Array{Float64,2}(undef, length(abnrmcowid), 4)
	MCTeat = Vector{Symbol}(undef, length(abnrmcowid))

	for i in 1:length(abnrmcowid)
		cow = @where(abnrmcow, :id .== abnrmcowid[i])
  		cow.mdi = linear_interpolation(cow.mdi)

  		h, t = find_headtail(cow.mdi .> thmdi2)
  		h₁, t₁ = h[1], t[1]

		try
			maxcondteat = minmax_teat(cow, h₁; min = false)

			yieldrr = mean(cow.yieldrr[h₁:t₁])
			yieldrf = mean(cow.yieldrf[h₁:t₁])
			yieldlr = mean(cow.yieldlr[h₁:t₁])
			yieldlf = mean(cow.yieldlf[h₁:t₁])

			Yield[i,:] = [yieldrr, yieldrf, yieldlr, yieldlf]
			MCTeat[i] = maxcondteat
	  	catch
			Yield[i,:] = [-999,-999,-999,-999]
			MCTeat[i] = :NA
		end
	end
	
	AvgLowCondYield = Vector{Float64}(undef, length(abnrmcowid))
	HighCondYield = Vector{Float64}(undef, length(abnrmcowid))

	for i in 1:length(abnrmcowid)
	  negteats = Dict([
		:rr => [2,3,4],
		:rf => [1,3,4],
		:lr => [1,2,4],
		:lf => [1,2,3],
		:NA => [1,2,3,4]
	  ])
	  posteats = Dict([:rr => 1, :rf => 2, :lr => 3, :lf => 4, :NA => 4])

	  AvgLowCondYield[i] = mean(Yield[i,negteats[MCTeat[i]]])
	  HighCondYield[i] = Yield[i,posteats[MCTeat[i]]]
	end

	dfy = DataFrame([
	  :AvgLowCondYield => AvgLowCondYield,
	  :HighCondYield => HighCondYield
	])

	dfy = @where(dfy, :AvgLowCondYield .!= -999)
end;

# ╔═╡ ec78353f-ecec-4a1e-957e-55e56ca2b39f
md"**Number of Cows in Raw Sample:** $(length(abnrmcowid))"

# ╔═╡ b34d3df7-1627-4af1-af54-7deaa768db5d
md"**Yield of Teat with High Conductivity:** $(mean(dfy.AvgLowCondYield))

**Average Yield of the three other teats:** $(mean(dfy.HighCondYield))
"

# ╔═╡ ea0519ed-36ab-431a-8ca8-0970976389b4
begin
	set_default_plot_size(6inch, 4inch)
	plot(dfy,
	  layer(x = :HighCondYield, Geom.density, color=[colorant"#de425b"]),
	  layer(x = :AvgLowCondYield, Geom.density, color=[colorant"#488f31"]),
	  Guide.xlabel("Yield"),
	  Guide.ylabel("Density"),
	  Theme(background_color="white")
	)
end

# ╔═╡ 80856753-2cf7-4e04-b734-2c300568d27e
md"**Parametric Difference in Mean Test**"

# ╔═╡ 399a93d5-6a1c-481f-a7b6-0b877ad321f8
# Parametric
UnequalVarianceTTest(dfy.AvgLowCondYield, dfy.HighCondYield)

# ╔═╡ d786c783-a64a-4be8-bdbc-7bd3529911ee
md"**Non-Parametric Difference in Mean Test**"

# ╔═╡ d414fdd9-c405-47aa-ba48-e6027c5fb171
# Non-parametric
MannWhitneyUTest(dfy.AvgLowCondYield, dfy.HighCondYield)

# ╔═╡ 1ac947d1-f53c-407e-9513-90f28111720d
md"### Yield Prior Warning"

# ╔═╡ 35e13feb-b24f-4e0c-a980-18aeec3c1dc2
md"**MDi Threshold**"

# ╔═╡ e387bea8-3a70-4d51-8c34-945111bd130b
@bind thmdi3 NumberField(1.1:0.1:maximum(unique(skipmissing(df2.mdi))), default = 1.1)

# ╔═╡ 660578a0-8a24-4d58-b956-3c08fe300a61
md"**Cow Number**"

# ╔═╡ 134f03c4-1f6f-4556-b189-95092a5316b3
md"**N Days Prior Warning**"

# ╔═╡ df6d9808-436b-43bb-b11f-02074f2352a4
@bind dprior Slider(2:10, default = 5, show_value=true)

# ╔═╡ d48412e7-d3c2-42fa-acb8-8f8842a92251
begin
	temp = @where(df2, :mdi .> thmdi2)
	abnrm_cowid = unique(temp.id) # Cows with MDi greater than threshold
end;

# ╔═╡ 04c1e8cf-7bba-432d-89f9-19ed47cc55dc
@bind cid2 NumberField(1:length(abnrm_cowid), default=1)

# ╔═╡ 7d22512b-a16d-48a6-bd6c-eb11dabb8ba5
abnrm_cow = @where(df2, :id .== abnrm_cowid[cid2]);

# ╔═╡ 04a799ea-aea7-4bf6-9c53-481af67e483a
begin
	set_default_plot_size(6inch, 4inch)
	plot_yieldpriorwarn(abnrm_cow, thmdi3, dprior)
end

# ╔═╡ Cell order:
# ╟─75359ed1-dc72-41f9-9d73-6d26d2d2cd64
# ╠═e912b9a2-a7ef-11eb-2117-9373bc3ef946
# ╠═0f96552f-12e1-46a1-b576-a78a8d7df9f6
# ╠═4d93dd33-38b6-4d6f-b836-f096c29236dc
# ╠═7d3634f5-9bb9-45e9-a2c8-122a155ffb50
# ╟─a9bd609b-8478-4c14-ae42-5822ff9341f1
# ╟─01beb164-64e2-48da-8945-a758b61b9b36
# ╟─18188ad5-3f8b-4ef8-a4e9-e2901f539d9f
# ╟─49de5332-a396-4494-bd61-c43a3ac18318
# ╟─1d343f02-eac9-4d97-b6b6-83f9054d20ad
# ╟─cb95ba52-b752-4d59-8d0d-3eb84bbf9f10
# ╠═35230364-60bc-44cd-b1f3-1ac71ea836e9
# ╟─a2216dc3-bab6-475f-a4b5-8c11af6a3509
# ╟─31c4e712-8a37-4399-a65f-2c4083c146a9
# ╟─0bc4324e-5528-4185-a28e-21a65aa84dbe
# ╟─b170c090-e177-4717-bba1-0f4565b816b1
# ╟─ee10c4d7-07b7-4644-8ada-eccdf35f30ba
# ╟─b4abb700-32f1-434e-afdf-2ba169fbe822
# ╟─0fcb820f-6777-4057-b950-7f1b58898bcc
# ╟─85e0d4f3-0c34-49a8-b8de-c461255adda0
# ╠═a578fd8d-9107-4990-90d9-3d3072461ed8
# ╟─5fc3552f-b61d-4840-a3ba-83c46e16088e
# ╠═8c285f45-3c39-4cf5-8d36-fd7c6731f133
# ╟─691dda5d-1060-4830-a4f4-7372711db25e
# ╟─6a16eb40-cb58-4718-801c-7aa564068485
# ╟─e9af2979-b21a-4dca-b5f2-6f6bc7f10c73
# ╟─3f21086e-2d47-40c4-bf3e-581d63f92584
# ╟─36ce1bf5-3c4f-4c7b-a65c-65116de80fe5
# ╟─0b4e16e0-a3f6-4976-9e66-c3b768898082
# ╟─7ee46d6a-a5d6-466d-a5c1-dfe1560054e2
# ╟─34590bff-17b2-497d-a9ea-d6fabc0767d7
# ╟─33ae8738-3222-4dfa-af2a-a0e1099bc4f8
# ╟─a04b89bf-1285-4e24-a17a-a4b90ae36aa8
# ╟─ec78353f-ecec-4a1e-957e-55e56ca2b39f
# ╟─b34d3df7-1627-4af1-af54-7deaa768db5d
# ╟─ea0519ed-36ab-431a-8ca8-0970976389b4
# ╟─80856753-2cf7-4e04-b734-2c300568d27e
# ╟─399a93d5-6a1c-481f-a7b6-0b877ad321f8
# ╟─d786c783-a64a-4be8-bdbc-7bd3529911ee
# ╟─d414fdd9-c405-47aa-ba48-e6027c5fb171
# ╟─1ac947d1-f53c-407e-9513-90f28111720d
# ╟─35e13feb-b24f-4e0c-a980-18aeec3c1dc2
# ╟─e387bea8-3a70-4d51-8c34-945111bd130b
# ╟─660578a0-8a24-4d58-b956-3c08fe300a61
# ╟─04c1e8cf-7bba-432d-89f9-19ed47cc55dc
# ╟─134f03c4-1f6f-4556-b189-95092a5316b3
# ╟─df6d9808-436b-43bb-b11f-02074f2352a4
# ╠═d48412e7-d3c2-42fa-acb8-8f8842a92251
# ╠═7d22512b-a16d-48a6-bd6c-eb11dabb8ba5
# ╟─04a799ea-aea7-4bf6-9c53-481af67e483a
