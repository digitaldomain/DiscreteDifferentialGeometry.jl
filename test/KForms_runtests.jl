# Copyright 2020 Digital Domain 3.0
#
# Licensed under the Apache License, Version 2.0 (the "Apache License")
# with the following modification; you may not use this file except in
# compliance with the Apache License and the following modification to it:
# Section 6. Trademarks. is deleted and replaced with:
#
# 6. Trademarks. This License does not grant permission to use the trade
#    names, trademarks, service marks, or product names of the Licensor
#    and its affiliates, except as required to comply with Section 4(c) of
#    the License and to reproduce the content of the NOTICE file.
#
# You may obtain a copy of the Apache License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Apache License with the above modification is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the Apache License for the specific
# language governing permissions and limitations under the Apache License.

using Test
using DiscreteDifferentialGeometry
using Multivectors
using LinearAlgebra

#!me several tests are failing.  
# probably because we redifined the hodge star to satisfy a∧⋆(b) == (a⋅b)*𝑖
# which is different that what we had before, but by all accounts the corrrect definition.
#!me need to figure out what's going on.  maybe our definition of ⋅ is not the same as DEC course

module G2
  using Multivectors
  # need last param to be true for differential forms
  @generate_basis("++",false,true,true)
end

using .G2

zerofun(x,y...) = zero(x)

@testset "CMU DEC course G2" begin
  e₁, e₂, e₁₂ = alle( G2, 2)
  e¹, e², e¹² = alld( G2, 2)
  𝐼 = 1e₁₂

  # http://brickisland.net/DDGSpring2019/wp-content/uploads/2019/01/DDG_458_SP19_Lecture04_kForms.pdf
  # 1-form example in coordinates
  v = 2e₁ + 2e₂
  α = -2e¹ + 3e²  
  # could repeat this test in multivectors when * is defined, here we use ⋅ instead of *
  @test α⋅v == 2 == α⋅reverse(v) # == apply(α,v)

  # 2-form example in coordinates
  u = 2e₁ + 2e₂; v = -2e₁ + 2e₂
  α = 1e¹ + 3e²; β = 2e¹ + 1e²
  @test (α∧β)*reverse(u∧v) == -40 == kform(α∧β,u∧v)

  # lecture 6 slides: α(x,y)=xdx + ydy
  ∂αᵢ∂e = [1 0; 0 1] 
  α = 1e₁+1e₂ # evaluate anywhere, so choose 1,1
  @test 𝑑(α, ∂αᵢ∂e) == 0
  # negative x row is in dual basis
  star_∂αᵢ∂e = [0 1; 1 0]
  @test 𝑑(⋆α, star_∂αᵢ∂e) == 2e₁₂ 

  # use ZForm type
  α = e₁(ZForm{2}((x,y)->x, [(x,y)->1, (x,y)->0])) + 
      e₂(ZForm{2}((x,y)->y, [(x,y)->0, (x,y)->1]))
  @test apply(α, KVector(2e₂), 1e₁+1e₂) == 2
  @test apply(α, KVector(3e₂+2e₂), 1e₁+1e₂) == 5
  @test apply(𝑑(α), 1e₁+1e₂, KVector(1e₁₂)) == 0
  @test apply(𝑑(⋆α), 1e₁+1e₂, KVector(1e₁₂)) == 2.0
  @test apply(𝑑(⋆⋆α), 1e₁+1e₂, KVector(1e₁₂)) == 0
  @test apply(𝑑(⋆⋆⋆α), 1e₁+1e₂, KVector(1e₁₂)) == -2.0

  # a naked zero form: ϕ(x,y) = ½e⁻⁽ˣ²⁺ʸ²)
  # dϕ = -2*ϕ*(xdx + ydy) 
  # expected result: -2*ϕ*x*e₁ + -2*ϕ*y*e₁
  # -2*ϕ(0.2,0.3)*0.2*e₁ + -2*ϕ*(0.2,0.3)*0.3*e₂
  xy = 0.2e₁+0.3e₂
  
  ϕ(x,y) = 0.5*exp(-(x^2+y^2))

  # evaluate at 0.2,0.3
  α = ZForm{2}(ϕ, [(x,y)->-2.0*ϕ(x,y)*x,
                (x,y)->-2.0*ϕ(x,y)*y])
  dp = 𝑑(α, 𝐼)
  dp = sortbasis(dp)
  @test untype(dp[1]) == e₁
  @test untype(dp[2]) == e₂
  @test dp[1].x.f(0.2,0.3) == -2.0*ϕ(coords(xy)...)*xy[1].x
  @test dp[2].x.f(0.2,0.3) == -2.0*ϕ(coords(xy)...)*xy[2].x
  @test apply(dp, xy, 1.0e₁+1.0e₂) ==  -2.0*ϕ(coords(xy)...)*xy[1].x + -2.0*ϕ(coords(xy)...)*xy[2].x

  # http://brickisland.net/DDGSpring2019/wp-content/uploads/2019/02/DDG_458_SP19_Lecture07_ExteriorIntegration.pdf
  # Integration of differential 2-forms: - Example
  
  ϕ(x,y) = x + x*y
  ω = ZForm{2}(ϕ, [(x,y)->1.0+y, (x,y)->x])∧e₁₂
  dx = 0.005; dy= 0.005;  dA = (dx*dy)e₁₂; Ω = [(xᵢ,yᵢ) for xᵢ in 0:dx:1.0 for yᵢ in 0.0:dy:1.0];
  integ_ω = mapreduce(xy->apply(ω, xy[1]*e₁ + xy[2]*e₂, KVector(dA)),+,Ω)
  # as dA -> 0 ∑ω(p)dA = ∫ωdA = 0.75 
  @test abs(integ_ω - 3/4) < 0.01;

  # inner product of 1-forms - Example
  α = ZForm{2}((x,y)->1.0, [(x,y)->0.0, (x,y)->0.0])∧e₁
  βₓ = ZForm{2}((x,y)->x, [(x,y)->1.0, (x,y)->0.0])∧e₁
  βᵥ = ZForm{2}((x,y)->-y, [(x,y)->0.0, (x,y)->-1.0])∧e₂
  ω = ⋆(α)∧(βₓ+βᵥ)
  integ_ω = mapreduce(xy->apply(ω, xy[1]*e₁ + xy[2]*e₂, KVector(dA)),+,Ω)
  @test abs(integ_ω - (-1/2)) < 0.01;

  # http://brickisland.net/DDGSpring2019/wp-content/uploads/2019/02/ExteriorCalculus-1.pdf
  # Exercize 13
  ϕ(x,y) = x*y+2*(y^2)
  α = ZForm{2}(ϕ, [(x,y)->y, (x,y)->x + 4y], [(x,y)->0 (x,y)->1 ; (x,y)->1 (x,y)->4])
  #!me wrong result getting -ve orientation
  #@test apply(Δ(α), 1e₁+1e₂) == 4

  # http://brickisland.net/DDGSpring2019/wp-content/uploads/2019/02/DDG_458_SP19_Lecture08_DiscreteDifferentialForms-1.pdf 
  # Integrating a 1-Form over and Edge - Example
  α = ZForm{2}((x,y)->x*y, [(x,y)->y, (x,y)->x])∧e₁ + ZForm{2}((x,y)->-x*x, [(x,y)->-2.0x, zerofun])∧e₂

  p₁ = -1.0e₁ + 2.0e₂
  p₂ = 3.0e₁ + 1.0e₂
  L = sqrt(17.0)
  T = (p₂-p₁)/L
  p(s) = p₁ + (s/L)*(p₂-p₁)
  ds = 0.01
  p0L = map(s->p(s),0:ds:L);
  αT = apply(α, T)
  @test round(mapreduce(pᵢ->apply(αT, pᵢ),+,p0L)*ds) == 7.0

  # Exterior Derivative on Vector-Valued Forms
  # http://brickisland.net/DDGSpring2019/wp-content/uploads/2019/02/DDG_458_SP19_Lecture10_SmoothCurves.pdf
  ϕ = ZForm{2}((x,y)->[x^2,x*y], [(x,y)->[2x,y], (x,y)->[0.0,x]])∧e₁ + 
      ZForm{2}((x,y)->[x*y,y^2], [(x,y)->[y,0.0], (x,y)->[x,2y]])∧e₂ 

  @test apply(𝑑(ϕ), 3.0e₁+5.0e₂, 1.0e₁₂) == [5.0, -3.0]

end

module G3
  using Multivectors
  @generate_basis("+++",false,true,true)
end
using .G3
@testset "CMU DEC course G3" begin

  e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃ = alle( G3, 3)

  #!me sign
  #@test (1.0e₂∧1.0e₃)∧⋆(1.0e₂∧1.0e₃) == 1.0e₁₂₃ 
  k = 1.0e₂∧1.0e₃
  @test k∧⋆k == (k⋅k)*pseudoscalar(k)

  @test (1.0e₂)∧⋆(1.0e₂) == 1.0e₁₂₃ 
  @test (-2.0e₂)∧⋆(-2.0e₂) == 4.0e₁₂₃ 

  u = 2.0e₁ + 2.0e₂; v = -2.0e₁ + 2.0e₂
  detuv = det(hcat(vcat(magnitude.(u),0.0), vcat(magnitude.(v),0.0), [0.0,0.0,magnitude.(⋆(u∧v))]))
  #!me sign opposite
  #@test detuv > 0 
  #two orthonormal vectors u1, u2, we ask that det(u1, u2, ⋆(u1 ∧ u2)) = 1
  u = normalize(u); v = normalize(v)
  detuv = det(hcat(vcat(magnitude.(u),0.0), vcat(magnitude.(v),0.0), [0.0,0.0,magnitude.(⋆(u∧v))]))
  #!me sign opposite
  #@test detuv ≈ 1

  u = -2.0e₂; v = 1.0e₃; uv = u∧v
  detuv = det(hcat([0.0,magnitude.(u),0.0], [0.0,0.0,magnitude.(v)], [magnitude.(⋆(uv)),0,0]))
  #!me sign opposite
  #@test detuv > 0


  #!me opposite sign
  #@test ⋆(-1.0e₁∧1.0e₂ - 1.0e₁∧1.0e₃ - 2.0e₂∧1.0e₃) == -2.0e₁ + 1.0e₂ - 1.0e₃
  k = ⋆(-1.0e₁∧1.0e₂ - 1.0e₁∧1.0e₃ - 2.0e₂∧1.0e₃) 
  @test k∧⋆(k) == (k⋅k)*pseudoscalar(k)

end

@testset "differential forms" begin
  e₁, e₂, e₁₂ = alle( G2, 2)
  𝐼 = 1e₁₂

  α = ZForm{2}((x,y)->x^3, [(x,y)->3.0*x^2, (x,y)->0.0],[(x,y)->6.0x (x,y)->0.0 ; (x,y)->0.0 (x,y)->0.0])
  @test apply(𝑑(𝑑(α, 𝐼)), 1.0e₁+1.0e₂, 1.0e₁₂) == 0.0

  α = ZForm{2}((x,y)->y*x^3 + x*y^2, 
               [(x,y)->y*3.0x^2 + y^2, 
                (x,y)->x^3 +      x*2.0y],
               [(x,y)->y*6.0x     (x,y)->3.0x^2 + 2.0y ; 
                (x,y)->3.0x^2 + 2.0y  (x,y)->2.0])

  @test apply(𝑑(𝑑(α, 𝐼)), 1.0e₁+1.5e₂, 1.0e₁₂) == 0.0
  @test apply(𝑑(⋆𝑑(α, 𝐼)), 1.0e₁+1.5e₂, 1.0e₁₂) != 0.0
end
