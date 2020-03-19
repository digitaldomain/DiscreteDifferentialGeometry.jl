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

"""
k-forms can be viewed as a linear map from k vectors or a k-vector to a scalar.
Or as the "dual" input to some implicit (k k) tensor.   !me check that this is correct, I kinda just coined this "implicit tensor word" on the spot...
A (k k) tensor is a multi-linear map from k forms + k vectors to a scalar.

"""

export 
apply,
𝑑

using Multivectors
using Multivectors: isazero
import Multivectors.kform
import Multivectors: ⋆

"""
    apply(α, v)

Apply k-form α to v.
"""
apply( α::K, v::V ) where { TF<:ZForm,T,N, K<:KVector{TF,N}, V<:KVector{T,N} } = 
  mapreduce(*,+,scalar.(α),scalar.(v)) #mapreduce(α⋅reverse(u)

"""
    𝑑(k, jacobian)

Exterior Derivative of KVector of k-forms with current jacobian
"""
𝑑( k::K, ∂ϕ∂x ) where {K<:KVector} = mapreduce(b->𝑑(b, ∂ϕ∂x), +, k) 

function 𝑑( b::B, ∂ϕ∂x ) where {T, N, B<:Blade{T,N}} 
  bᵢ = factor(b)
  β = mapreduce(one, ∧, bᵢ[3:end])
  k = N-1
  α = bᵢ[1]*bᵢ[2]
  𝑑α = 𝑑(one(bᵢ[2]), ∂ϕ∂x)
  𝑑α∧β + (-1)^k*α∧𝑑(β, ∂ϕ∂x)
end

function 𝑑( b::B, ∂ϕ∂x ) where {T, B<:Blade{T,1}} 
  s,eⱼ = factor(b)
  be = basis_1blades(b)
  j = subspace(eⱼ)[1]
  # expand 1-form on scalar out to ds*e₁ + ds*e₂...
  s*sum(Iterators.filter( !isazero, map( i->(∂ϕ∂x[j,i]*be[i])∧one(eⱼ), 1:length(be) )))
end

#== with 0-forms ==#
𝑑( s::R ) where {R<:Real} = zero(s)

"""
    𝑑(k)

Exterior derivative operator on KVector wedged with ZForms.
Differential of k-form.
"""
𝑑( k::K ) where {D, Z<:ZForm{D}, K<:KVector{Z}} = mapreduce(b->𝑑(b), +, k)

"differential of simple k-form"
function 𝑑( b::B ) where {T, N, B<:Blade{T,N}}
  bᵢ = factor(b)
  # 𝑑α∧β + (-1)ᵏα∧𝑑β
  β = mapreduce(e->e(one(T)), ∧, bᵢ[3:end])
  k = N-1
  α = bᵢ[1]
  sum(filter(!isazero, [𝑑(α∧bᵢ[2])∧β, (-1)^k*α∧𝑑(β)]))
end

𝑑( b::B ) where { T<:Real, B<:Blade{T,1} } = zero(T)

"differential of simple 1-form"
function 𝑑( b::B ) where {D, T<:ZForm{D}, B<:Blade{T,1}}
  j = subspace(b)
  be = basis_1blades(b)
  dxⱼ = be[j]
  k = b.x
  sum(Iterators.filter( !isazero, map( i->be[i](ZForm{D}(k.gradient[i], k.hessian[i,:]))∧one(dxⱼ), 1:D )))
end

"""
  𝑑( k, 𝐼 )  

differential of 0-form. 𝐼 is the psuedovector of the algebra
"""
function 𝑑( k::Z, 𝐼::B ) where {D, Z<:ZForm{D}, B<:Blade}
  be = basis_1blades(𝐼)
  sum(Iterators.filter(!isazero, map( i->be[i](ZForm{D}(k.gradient[i], k.hessian[i,:])), 1:D)))
end

"""
  𝑑( k )  

differential of 0-form
"""
function 𝑑( k::Z ) where {Z<:ZForm}
  @warn "resolving Blade types in top-level module via dual(1)"
  #𝐼 = 1pseudoscalar(Z) #dual(1)
  𝐼 = dual(1)
  𝑑(k, 𝐼)
end

const differential = 𝑑

Base.iszero(b::B) where {D,K,N,B<:KVector{ZForm{D},K,N}} = mapreduce(isazero, max, b)

𝑑(𝑓::F) where F<:Function = α->𝑑(𝑓(α))
⋆(𝑓::F) where F<:Function = α->⋆(𝑓(α))

δ(α) = ⋆(𝑑(⋆α))
Δ(α) = sum(filter(!isazero, [δ(𝑑(α)), 𝑑(δ(α))]))

#Base.:*(α::KF,u::K) where{F<:ZForm,KF<:KVector{F},K<:KVector} = sum(map(kfᵢ->kfᵢ.x.f(scalar.(u)...),α))

# very literal implementation of applying a k-form
"""
    apply(α, u, k)

Apply differential k-form α at point u to k-vector k.
Point u is a 1-vector.
"""
function apply(α::KF, u::P, k::K) where{F<:ZForm, KF<:KVector{F}, 
                                       U, P<:KVector{U,1}, 
                                       T, K<:KVector{T}} 
  rα = raise.(α)
  αk = [ (eᵢ*reverse(one(eʲ)), j) for (i,eᵢ) in zip(1:length(k),k) 
                                           for (j,eʲ) in zip(1:length(rα),untype.(rα)) ]
  mapreduce( ((s,j),)->s*apply(α[j].x, u), +, filter(!isazero∘first, αk))
end

apply(α::B, u, k::KF) where {F<:ZForm, KF<:KVector{F}, B<:Blade} = apply(KVector(α), u, k)

const CN = Union{KVector, Blade}
apply(α::CN, u::CN, k::CN) = apply(KVector(α), KVector(u), KVector(k))  

"""
    apply(α, u)

Apply differential 0-form α at point u.
Point u is a 1-vector.
"""
apply(α::F, u::P) where {F<:ZForm, U, P<:KVector{U,1}} = α.f(coords(u)...)

"Constant 0-form.  Same value at any point"
apply(s::T, u::P) where {T<:Real, U, P<:KVector{U,1}} = s

Base.:*(s::T, b::B) where {T<:ZForm, B<:KVector} = B(s*b.k)
Base.:*(b::B, s::T) where {T<:ZForm, B<:KVector} = B(b.k*s)

