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

module DiscreteDifferentialGeometry

#!me check that the signs of the cell differentials satisfy 𝑑_cellⁿ⁻ᵏ == (-1)ᵏ(𝑑_primalᵏ⁻¹)ᵀ
#!me might not have the sign right

"""
Ωᵢ : k-form (k == i) domain of the mesh.  i.e. Ω₀ is on verts Ω₁ is on edges, Ω₂ is on faces
Ω⋆ᵢ : k-form (k == i) dual domain of the mesh.  i.e. Ω⋆₀ is on dual verts Ω₁ is on dual edges, Ω₂ is on dual faces
⋆ᵢ : hodge star operator
𝑑ᵢ : differential operator

   Ω₀---𝑑₀-→Ω₁----𝑑₁-→Ω₂ 
   ↑        ↑         ↑   
 ⋆₀|⋆₀⁻   ⋆₁|⋆₁⁻    ⋆₂|⋆₂⁻
   ↓        ↓         ↓     
   Ω⋆₂←-𝑑₁--Ω⋆₁←-𝑑₀---Ω⋆₀ 

"""

using Multivectors
using LinearAlgebra, SparseArrays
using HalfEdges
using HalfEdges: incidence, area, nvertices, faces, isboundary, head, edge, next, opposite
import Multivectors: ⋆

include("KForms.jl")

export 
DiscreteDifferentialOperator,
PrimalKForm,
CellKForm,
array,
δ, codifferential,
Δ, laplace_beltrami,
Δcell, laplace_beltrami_cell,
𝑑, differential,
⋆, star,
poisson,
constrain_system,
constrain_rhs,
interpolate,
CircumcentricDual,
BarycentricDual

struct Curve
  indices::Vector{Int}
  P
end

partial = (f::Function,y...)->(z...)->f(y...,z...)
∂(f::Function,y...) = partial(f,y...)

abstract type DifferentialOperator end
abstract type DiscreteDifferentialOperator{N, A} <: DifferentialOperator end

struct DiscreteDifferential{N, A} <: DiscreteDifferentialOperator{N, A}
  D::A
end

struct DiscreteCellDifferential{N, A} <: DiscreteDifferentialOperator{N, A}
  D::A
end

struct DiscreteHodgeStar{N,A} <: DiscreteDifferentialOperator{N, A}
  D::A 
end

struct DiscreteHodgeStarInv{N,A} <: DiscreteDifferentialOperator{N, A}
  D::A
end

struct ZeroOperator{N,A} <: DiscreteDifferentialOperator{N,A}
  D::A
end

"""
    DiscreteKForm{N,A}
    
A discrete k-form of grade N, represented by an indexible container A ( usually a Vector )
A 0-form (N==0) is a k-form with a value per vertex
A 1-form (N==1) is a k-form with a value per edge
A 2-form (N==2) is a k-form with a value per face
"""
struct PrimalKForm{N,A} <: DiscreteDifferentialOperator{N,A}
  D::A
end

PrimalKForm(K::A, N) where A = PrimalKForm{N,A}(K)
PrimalKForm{N}(K::A) where {A<:Diagonal, N} = PrimalKForm{N,A}(K)
PrimalKForm{N}(K::AbstractVector) where {N} = PrimalKForm{N}(Diagonal(K))
PrimalKForm(K::V, N) where V<:AbstractVector = PrimalKForm{N}(Diagonal(K))

"""
    DiscreteCellKForm{N,A}
    
A discrete k-form of grade N, represented by an indexible container A ( usually a Vector )
A 0-form (N==0) is a k-form with a value per dual face ( point at circumcentre, on cell boundary ) 
A 1-form (N==1) is a k-form with a value per dual edge ( boundary of one ring cell )
A 2-form (N==2) is a k-form with a value per dual vertex ( one ring cell around vertex )

"""
struct CellKForm{N,A} <: DiscreteDifferentialOperator{N,A}
  D::A
end

CellKForm(K::A, k) where A = CellKForm{k,A}(K)
CellKForm{N}(K::A) where {A<:Diagonal, N} = CellKForm{N,A}(K)
CellKForm{N}(K::AbstractVector) where {N} = CellKForm{N}(Diagonal(K))
CellKForm(K::V, k) where V<:AbstractVector = CellKForm{k}(Diagonal(K))


array(D::DiscreteDifferentialOperator) = D.D
array(D::PrimalKForm) = D.D
array(D::CellKForm) = D.D
array(P::T) where {V<:AbstractArray,T<:AbstractArray{V}} = reduce(hcat, P)'

isdual( ::DiscreteHodgeStar ) = true
isdual( ::DiscreteHodgeStarInv ) = false
isdual( ::DiscreteCellDifferential ) = true
isdual( ::DiscreteDifferential ) = false
isdual( ::CellKForm ) = true
isdual( ::PrimalKForm ) = false

Base.:*( 𝑑̂n::D, A::T ) where {D<:DiscreteDifferentialOperator, T<:AbstractArray} = 𝑑̂n.D*A
Base.:*( A::T, 𝑑̂n::D) where {D<:DiscreteDifferentialOperator, T<:AbstractArray} = A*𝑑̂n.D
Base.iszero( 𝑑̂n::D ) where {D<:DiscreteDifferentialOperator, T<:AbstractArray} = iszero(𝑑̂n.D)
Base.:*( 𝑑̂n::D, s::T ) where {D<:DiscreteDifferentialOperator, T<:Real} = D(𝑑̂n.D*s)
Base.:*( s::T, 𝑑̂n::D ) where {D<:DiscreteDifferentialOperator, T<:Real} = D(s*𝑑̂n.D)
Base.:/( 𝑑̂n::D, s::T ) where {D<:DiscreteDifferentialOperator, T<:Real} = D(𝑑̂n.D/s)


Base.:+( n::Union{D, ZeroOperator}, m::Union{D, ZeroOperator} ) where {D<:DiscreteDifferentialOperator} = D(n.D+m.D)
Base.:-( n::Union{D, ZeroOperator}, m::Union{D, ZeroOperator} ) where {D<:DiscreteDifferentialOperator} = D(n.D-m.D)

struct MultiDifferential <: DifferentialOperator
  MD::Vector{DifferentialOperator}
end

array(D::MultiDifferential) = sum(D.MD)

Base.:+(a::DifferentialOperator, b::DifferentialOperator) = MultiDifferential( [a,b] )
Base.:-(a::DifferentialOperator, b::DifferentialOperator) = MultiDifferential( [a,-b] )
Base.:-(a::MultiDifferential) = MultiDifferential(-1.0.*(a.MD)) 
Base.:+(a::MultiDifferential, b::DifferentialOperator) = MultiDifferential( vcat(a.MD, b) )
Base.:+(a::DifferentialOperator, b::MultiDifferential) = MultiDifferential( vcat(b.MD, b) )
Base.:+(a::MultiDifferential, b::MultiDifferential) = MultiDifferential( vcat(a.MD, b.MD) )

Base.:(==)( a::DifferentialOperator, b::DifferentialOperator ) = array(a) == array(b)

"""the cotangent of the angle between two vectors"""
cotan( v1::T, v2::T ) where {T<:AbstractArray} = (v1⋅v2)/norm(v1×v2)

"""
  across( topo, P, heh )

the triangle vertex across from the halfedge.
"""
function across( topo::Topology, P, heh::HalfEdgeHandle ) where {PT}
  if isboundary(topo, heh)
    nothing
  else
    @inbounds P[head(topo, next(topo, heh))]
  end
end

"""
vector cotangent of angle at Pₐ, ccw from P₁ to P₂
"""
function cotanweight( P₁::T, P₂::T, Pₐ::T ) where T
  cotan( P₁-Pₐ, P₂-Pₐ )
end

cotanweight( P₁::T, P₂::T, P₃::Nothing ) where T =  zero(eltype(T))

"""
  cotanweight( topo, P, h )

calculate cotangent weight across an edge.
0.5*(cot(α) + cot(Β)), where α,Β are the angles of edges opposite our halfedge
"""
function cotanweight( topo::Topology, P, he::HalfEdgeHandle )
  αᵢ, αⱼ = edge(topo, he)
  Pαᵢ = P[αᵢ]
  Pαⱼ = P[αⱼ]
  cotα = cotanweight(Pαᵢ, Pαⱼ, across(topo, P, he))
  cotΒ = cotanweight(Pαⱼ, Pαᵢ, across(topo, P, opposite(topo, he)))
  0.5*(cotα + cotΒ)
end

"""
Barycentric dual vertex area. One third of the summed areas of triangles incident to given vertex
"""
function vertexarea( mesh, P, i::VertexHandle )
  ring = OneRing(mesh,i)
  iring = Iterators.filter(heh->!isboundary(mesh,heh),ring)
  mapreduce(heh->area(mesh,P,heh)/3.0,+,iring)
end

##===== Operators implementations =======##


#======= MESHES =======#

struct CircumcentricDual end
struct BarycentricDual end

⋆(topo::Topology, P, D::DO, ::DT) where {DO <: DifferentialOperator, DT<:Union{CircumcentricDual, BarycentricDual}} = ⋆(topo, P, D)

# From Primal 0

""" ⋆(topo, P)  Ω₀->⋆₀->Ω⋆₂.  A map to dual 2-forms (dual vertices) """
function ⋆(topo::Topology, P, ::CircumcentricDual )
  aa = map(VertexHandle.(1:nvertices(topo))) do vh
    reduce(Iterators.filter(!∂(isboundary, topo), (OneRing(topo, vh))); init=0.0) do acc, h
      i,j = edge(topo, h)
      k = head(topo, next(topo, h))
      Pi, Pj, Pk = P[[i, j, k]]
      Pki = Pi-Pk; Pij = P[j]-P[i] 
      acc+0.125*((Pki⋅Pki)*cotanweight( Pk, Pi, Pj) + 
           (Pij⋅Pij)*cotanweight( Pi, Pj, Pk))
    end
  end |> Diagonal
  DiscreteHodgeStar{0,typeof(aa)}(aa)
end

""" ⋆(topo, P)  Ω₀->⋆₀->Ω⋆₂.  A map to dual 2-forms (dual vertices) """
function ⋆(topo::Topology, P, ::BarycentricDual )
  aa = map(VertexHandle.(1:nvertices(topo))) do vh
    vertexarea( topo, P, vh )
  end |> Diagonal
  DiscreteHodgeStar{0,typeof(aa)}(aa)
end

""" ⋆(topo, P)   Ω₀->⋆₀->Ω⋆₂.  A map to dual 2-forms (dual vertices) """
⋆(topo::Topology, P) = ⋆(topo, P, CircumcentricDual())

const Omega0{A} = Union{DiscreteHodgeStarInv{0,A}, PrimalKForm{0,A}}

""" Ω₀->𝑑₀->Ω₁ """
function 𝑑( topo::Topology )
  𝑑0 = incidence( topo, VertexHandle, EdgeHandle, true )
  DiscreteDifferential{0, typeof(𝑑0)}(𝑑0)
end

""" ⋆₀⁻¹->𝑑₀->Ω₁ """
function 𝑑( topo::Topology, D::Omega0 )
  𝑑0star = incidence( topo, VertexHandle, EdgeHandle, true )*array(D)
  DiscreteDifferential{0, typeof(𝑑0star)}(𝑑0star)
end

""" 
        Ω₀->⋆₀->Ω⋆₂
  ⋆₀⁻¹->Ω₀->⋆₀->Ω⋆₂

A map to 2-forms (vertex cell).
"""
function ⋆(topo::Topology, P, D::Omega0{A}) where A
  aa = (⋆(topo, P)).D*D.D
  DiscreteHodgeStar{0,typeof(aa)}(aa) 
end

# From Primal 1

const Omega1{A} = Union{DiscreteDifferential{0,A}, DiscreteHodgeStarInv{1,A}, PrimalKForm{1,A}}

""" 
    𝑑( topo, Ω₁ )

     Ω₁->𝑑₁->Ω₂
 𝑑₀->Ω₁->𝑑₁->Ω₂
⋆⁻¹->Ω₁->𝑑₁->Ω₂

A map to 2-forms (faces) 
"""
function 𝑑( topo::Topology, D::Omega1{A} ) where A
  𝑑1 = incidence( topo, EdgeHandle, FaceHandle, true )*D.D
  DiscreteDifferential{1, typeof(𝑑1)}(𝑑1)
end

""" 
    ⋆( topo, Ω₁ )

     Ω₁->⋆₁->Ω⋆₁
 𝑑₀->Ω₁->⋆₁->Ω⋆₁
⋆⁻₁->Ω₁->⋆₁->Ω⋆₁

A map to 1-forms (cell edges)
"""
function ⋆( topo::Topology, P, D::Omega1{A} ) where {A}
  stard0 = (map(∂(cotanweight, topo, P), UniqueHalfEdges(topo)) |> Diagonal)*D.D
  DiscreteHodgeStar{1, typeof(stard0)}(stard0)
end

# From Primal 2

const Omega2{A} = Union{DiscreteDifferential{1,A}, DiscreteHodgeStarInv{2,A}, PrimalKForm{2,A}}

""" 
    ⋆( topo, Ω₂ )

     Ω₂->⋆₂->Ω⋆₀
 𝑑₁->Ω₂->⋆₂->Ω⋆₀
⋆⁻₂->Ω₂->⋆₂->Ω⋆₀

A map to 0-forms (cell centred vertex)
"""
function ⋆( topo::Topology, P, D::Omega2{A} ) where {A}
  stard1 = ((inv∘∂(area, topo, P)).((((x->Polygon(topo, x)).(faces(topo))))) |> Diagonal)*array(D)
  DiscreteHodgeStar{2, typeof(stard1)}(stard1)
end

# From Dual 2

const OmegaStar2{A} = Union{DiscreteHodgeStar{0,A}, DiscreteCellDifferential{1,A}, CellKForm{2,A}}

""" ->⋆₀->⋆₀⁻¹->Ω₀ """
function ⋆(topo::Topology, P, D::OmegaStar2{A}) where A
  aa = inv(⋆(topo, P).D)*D.D
  DiscreteHodgeStarInv{0,typeof(aa)}(aa) 
end

#==
""" ->𝑑₁ᵀ->⋆₀⁻¹->Ω₀ """
function ⋆( topo::Topology, P, 𝑑̂1t::D1, dual_area_type = CircumcentricDual()) where {A, D1<:DiscreteDifferentialT{1, A}}
  stard1t = (⋆(topo, P, dual_area_type) |> inv∘array)*𝑑̂1t.D
  DiscreteHodgeStarInv{0, typeof(stard1t)}(stard1t)
end
==#

# From Dual 1

const OmegaStar1{A} = Union{DiscreteCellDifferential{0,A}, DiscreteHodgeStar{1,A}, CellKForm{1,A}}

""" 
    𝑑( topo, Ω⋆₁ )

    Ω⋆₁->𝑑₁->Ω⋆₂
𝑑₀->Ω⋆₁->𝑑₁->Ω⋆₂
⋆₁->Ω⋆₁->𝑑₁->Ω⋆₂
"""
function 𝑑( topo::Topology, D::OmegaStar1{A} ) where A
  𝑑1t = adjoint(incidence( topo, VertexHandle, EdgeHandle, true ))*D.D
  DiscreteCellDifferential{1, typeof(𝑑1t)}(𝑑1t)
end

""" 
    ⋆( topo, Ω⋆₁ )

    Ω⋆₁->⋆₁⁻¹->Ω₁
𝑑₀->Ω⋆₁->⋆₁⁻¹->Ω₁
⋆₁->Ω⋆₁->⋆₁⁻¹->Ω₁
"""
function ⋆( topo, P, D::OmegaStar1 )
  starstar = (map(∂(cotanweight, topo, P), UniqueHalfEdges(topo)) |> inv∘Diagonal)*D.D
  DiscreteHodgeStarInv{1, typeof(starstar)}(starstar)
end

# From Dual 0
const OmegaStar0{A} = Union{DiscreteHodgeStar{2,A}, CellKForm{0,A}}

""" ->⋆₂->⋆₂⁻¹->Ω₂ """
function ⋆( topo::Topology, P, D::OmegaStar0 )
  starstar = ((∂(area, topo, P)).((((x->Polygon(topo, x)).(faces(topo))))) |> Diagonal)*array(D)
  DiscreteHodgeStarInv{2, typeof(starstar)}(starstar)
end

""" 
    𝑑( topo, s )  

Ω₂⋆->𝑑₀->Ω⋆₁.  A map to dual edges 
"""
function 𝑑( topo::Topology, D::OmegaStar0{A}) where {A}
  dstar = adjoint(incidence( topo, EdgeHandle, FaceHandle, true ))*array(D)
  DiscreteCellDifferential{0, typeof(dstar)}(dstar)
end
#==
""" 
    𝑑( topo, s )  

Ω₂⋆->𝑑₀->Ω⋆₁.  A map to dual edges 
"""
function 𝑑( topo::Topology, s::H2 ) where {A, H2<:DiscreteHodgeStar{2, A}}
  d0tstar = adjoint(incidence( topo, EdgeHandle, FaceHandle, true ))*array(s)
  DiscreteCellDifferential{0, typeof(d0tstar)}(d0tstar)
end
==#

#==
""" ->⋆₀⁻¹->⋆₀->Ω₀ """
function ⋆(topo::Topology, P, D::DiscreteHodgeStarInv{0,A}) where A
  aa = (⋆(topo, P)).D*D.D
  DiscreteHodgeStar{0,typeof(aa)}(aa) 
end
==#

zero_operator(::Type{A}) where A = (𝑑z = zero(eltype(A))*I; ZeroOperator{0, typeof(𝑑z)}(𝑑z))

""" 𝑑(topo, Z) differential of zero is zero """
𝑑( topo::Topology, Z::ZeroOperator{N,A} ) where {N,A} = zero_operator(A)

""" 𝑑(topo, 𝑑₀ᵗ) differential of 𝑑₀ᵗ which is a map to dual vertices is zero, since dual simplex of degree 3 doesn't exist on 𝑅² manifold """
𝑑( topo::Topology, D::DiscreteCellDifferential{1,A} ) where A = zero_operator(A)

""" 𝑑(topo, 𝑑₁) differential of 𝑑₁ which is a map to faces is zero, since simplex of degree 3 doesn't exist on 𝑅² manifold """
𝑑( topo::Topology, D::DiscreteDifferential{1,A} ) where A = zero_operator(A)

""" 𝑑(topo, ⋆₀) differential of ⋆₀ which is a map to dual vertices is zero, since dual simplex of degree 3 doesn't exist on 𝑅² manifold """
𝑑( topo::Topology, H::DiscreteHodgeStar{0,A} ) where A = zero_operator(A)

""" 𝑑(topo, ⋆₂⁻¹) differential of ⋆₂⁻¹ which is a map to faces is zero, since simplex of degree 3 doesn't exist on 𝑅² manifold """
𝑑( topo::Topology, H::DiscreteHodgeStarInv{2,A} ) where A = zero_operator(A)

⋆( topo::Topology, P, Z::ZeroOperator ) = Z


"""
    δ(topo, P) 
   
Discrete coderivative    
"""
δ(topo::Topology, P) = ⋆(topo, P, 𝑑(topo, ⋆(topo, P)))

δ(topo::Topology, P, D::DO) where DO<:DifferentialOperator = ⋆(topo, P, 𝑑(topo, ⋆(topo, P, D)))

δ(topo::Topology, P, bcd::BarycentricDual) = 
  ⋆(topo, P, 𝑑(topo, ⋆(topo, P, bcd)), bcd)

δ(topo::Topology, P, D::DO, bcd::BarycentricDual) where DO<:DifferentialOperator = 
  ⋆(topo, P, 𝑑(topo, ⋆(topo, P, D, bcd)), bcd)

"""
    Δ(topo, P)

Discrete Laplace-Beltrami operator    
"""
Δ(topo::Topology, P) = δ(topo, P, 𝑑(topo)) + 𝑑(topo, δ(topo, P))
Δ(topo::Topology, P, bcd::BarycentricDual) = δ(topo, P, 𝑑(topo), bcd) + 𝑑(topo, δ(topo, P, bcd))

"""
    Δcell(topo, P)

The symmetric Laplace-Beltrami operator.  ⋆Δcell = Δ.  
The cell Δ differs from the (strong) Δ in that cell version maps to the dual simplex.

i.e. Δcell(topo, P) -> Ω⋆₀ and Δ(topo, P) -> Ω₀

The rhs must also be on dual. You must apply ⋆ to the rhs.  
Solve: Δ(topo, P) = b  -----> Solve: Δcell(topo, P) = ⋆(topo,P)*b

Useful for constructing symmetric operators.
"""
Δcell(topo, P) = 𝑑(topo, ⋆(topo, P, 𝑑(topo))) 
# Note: we have dropped the term corresponding to ⋆𝑑δ(topo, P) as it is 0 when starting from vertices

# Optimized Laplace-Beltrami operator given face and vertex lists
"""
local Laplace matrix for a triangle
"""
function Δ( i::PT, j::PT, k::PT ) where { T, PT<:AbstractVector{T} }
  ij = j-i
  jk = k-j
  ki = i-k
  a = cotan(ij,-ki)*T(0.5)
  b = cotan(jk,-ij)*T(0.5)
  c = cotan(ki,-jk)*T(0.5)
  Matrix3( [b+c -c -b; -c c+a -a;-b -a a+b] ) 
end

"""
    Δcell(F, P)

Laplace-Beltrami operator built from list of triangle indices and points.  Optimized for performance.

"""
function Δcell( F::Vector{T}, P::Vector{PT} ) where {T<:Union{AbstractArray,NTuple{3}}, FT, PT<:AbstractVector{FT}}
  n = length(F)*9
  indexI = Vector{Int}(undef,n)
  indexJ = Vector{Int}(undef,n)
  element = Vector{FT}(undef,n)
  s = 1
  @inbounds for ijk in F
    i,j,k = ijk
    Pᵢ = P[i]
    Pⱼ = P[j]
    Pk = P[k]
    ij = Pⱼ-Pᵢ
    jk = Pk-Pⱼ
    ki = Pᵢ-Pk
    a = cotan(ij,-ki)*FT(0.5)
    b = cotan(jk,-ij)*FT(0.5)
    c = cotan(ki,-jk)*FT(0.5)
    #L[i,i] += b+c
    indexI[s] = i; indexJ[s] = i; element[s] = b+c; s+=1
    #L[i,j] -= c
    indexI[s] = i; indexJ[s] = j; element[s] = -c; s+=1
    #L[i,k] -= b
    indexI[s] = i; indexJ[s] = k; element[s] = -b; s+=1
    #L[j,i] -= c
    indexI[s] = j; indexJ[s] = i; element[s] = -c; s+=1
    #L[j,j] += c+a
    indexI[s] = j; indexJ[s] = j; element[s] = c+a; s+=1
    #L[j,k] -= a
    indexI[s] = j; indexJ[s] = k; element[s] = -a; s+=1
    #L[k,i] -= b
    indexI[s] = k; indexJ[s] = i; element[s] = -b; s+=1
    #L[k,j] -= a
    indexI[s] = k; indexJ[s] = j; element[s] = -a; s+=1
    #L[k,k] += a+b
    indexI[s] = k; indexJ[s] = k; element[s] = a+b; s+=1
  end
  sparse(indexI, indexJ, element)
end

Δcell( F::Vector{IT}, P ) where {IT<:Integer} = Δ(Iterators.partition( F, 3 ) |> collect, P)

#======= SHARP, FLAT =====#

#♯₀(topo, P) = ⋆(topo, P, ⋆(topo, P))
#

#======= ALIASES =========#

const codifferential = δ
const laplace_beltrami = Δ
const laplace_beltrami_cell = Δcell
const differential = 𝑑
const star = ⋆

#======= CURVES =======#

"""  
    𝑑(curve)

Ω₀->𝑑₀->Ω₁ 
map from vertex to edges
"""
function 𝑑( c::Curve )
  V = length(c.P)
  E = length(c.indices)-1
  T = eltype(c.P)
  𝑑0 = sparse( reduce(vcat, [[i,i] for i in 1:E]), 
               reduce( vcat, [ [ij...] for ij in zip(c.indices[1:end-1], c.indices[2:end])]),
               reduce(vcat, [[T(1),-T(1)] for i in 1:E]),
               E, V) 
  DiscreteDifferential{0, typeof(𝑑0)}(𝑑0)
end

#======= VECTOR FIELDS ======#
# some useful details here: https://www.math.arizona.edu/~agillette/research/decNotes.pdf
# Discrete Vector Fields are valued at either vertices or dual triangles.  i.e. 0-simplices 
# The dual 0-simplex is the circumcentre of a primal triangle ( 2-simplex ). i.e. a dual triangle
# This is not really explained in the CMU course at the point I've read up to, but is used in their code
# Also geodesic heat method uses it for computing the gradient. 
# So heat values are stored on primal vertices and then interpolated to dual 0-simplex... hmmm
# could you instead compute heat values on dual 0-simplex and then apply differential?
# actually circumetric location is used because the tangent plane is well defined on a face.
# ϕᵢ𝑑ϕⱼ - ϕⱼ𝑑ϕᵢ

baryc(p::K, (b, c)::KK, A⁻¹::KB) where {K<:KVector, KK<:Tuple{KVector,KVector}, KB<:Union{KVector, Blade}} = 
  grade(((c-b)∧(p-b))*A⁻¹, 0)

#==
ϕ₁(p::K, a::K,b::K,c::K, A⁻¹::K) where K<:KVector = ϕᵢ(p, (b,c), A⁻¹)
ϕ₂(p::K, a::K,b::K,c::K, A⁻¹::K) where K<:KVector = ϕᵢ(p, (c,a), A⁻¹)
ϕ₃(p::K, a::K,b::K,c::K, A⁻¹::K) where K<:KVector = ϕᵢ(p, (a,b), A⁻¹)
==#

function barycentric_coords( p::K ) where {K<:KVector}
  A = (b-a)∧(c-a)
  ((c-b)∧(p-b)/A, (a-c)∧(p-c)/A, (b-a)∧(p-a)/A)  # a tuple of coords
end

function ϕᵢ(edge_bc::KK, A⁻¹::KB) where {K<:KVector, KK<:Tuple{KVector,KVector}, KB<:Union{KVector, Blade}}
  A₁₂, A₁₃, A₂₃ = coords(A⁻¹)
  b,c = edge_bc
  e₁, e₂, e₃ = basis_1blades(b[1])
  cbx, cby, cbz = coords(c-b)
  ∂ϕᵢ∂x =  cby*(A₁₂) + cbz*(A₁₃)
  ∂ϕᵢ∂y =  cbz*(A₂₃) - cbx*(A₁₂)
  ∂ϕᵢ∂z = -cbx*(A₁₃) - cby*(A₂₃)

  ZForm{3}( (x, y, z)->baryc(e₁(x)+e₂(y)+e₃(z), edge_bc, A⁻¹),
           [(x, y, z)->∂ϕᵢ∂x,
            (x, y, z)->∂ϕᵢ∂y,
            (x, y, z)->∂ϕᵢ∂z])
end

function ϕᵢ(a::KVector, b::KVector, c::KVector)
  A = (b-a)∧(c-a)
  A⁻¹ = sortbasis(inv(A)) # ordered e₁₂, e₁₃, e₂₃
  (ϕᵢ((b, c), A⁻¹), ϕᵢ((c, a), A⁻¹), ϕᵢ((a, b), A⁻¹))
end

𝑑ϕ( ϕ, p ) = map(∂ϕ->∂ϕ(coords(p)...), ϕ.gradient) .* basis_1blades(p)

"""
    interpolate( (a,b,c), edgevals, p)

Interpolate the discrete values on the edges of a triangle with points a,b,c to a vector at location p.

edgevals: values on edges should be ordered so the value at index i corresponds to the edge across from vertex i.
"""
function interpolate( (a,b,c)::Tri, edgeval::V, p::X ) where 
  {Tri<:Union{Tuple{KVector,KVector,KVector}, AbstractVector{<:KVector}}, V<:Union{AbstractArray, Tuple}, X<:KVector} 

  ϕ₁, ϕ₂, ϕ₃ = ϕᵢ(a, b, c)
  xyz = coords(p)

  wab = ϕ₁.f(xyz...)*𝑑ϕ(ϕ₂, p) - ϕ₂.f(xyz...)*𝑑ϕ(ϕ₁, p)
  wbc = ϕ₂.f(xyz...)*𝑑ϕ(ϕ₃, p) - ϕ₃.f(xyz...)*𝑑ϕ(ϕ₂, p)
  wca = ϕ₃.f(xyz...)*𝑑ϕ(ϕ₁, p) - ϕ₁.f(xyz...)*𝑑ϕ(ϕ₃, p)

  v = edgeval[3]*wab + edgeval[1]*wbc + edgeval[2]*wca
  KVector(v)
end

interpolate(a,b,c,eva,evb,evc,p) = interpolate((a,b,c),(eva,evb,evc),p)

#======= BOUNDARY ======#

"""
    constrain( L, pin )

Add Diriclete boundary conditions to Linear Operator L. 
return tuple with reduced L and columns of L representing constrained dofs 
"""
function constrain_system( L_in::S, pin ) where {S<:AbstractArray}
  L = copy(L_in)
  I_ = sparse(one(eltype(S))*I,size(L)...)
  L_constrained = L[:,pin]
  L[:,pin] = I_[:,pin]
  L[pin,:] = I_[pin,:]

  (L,L_constrained)
end


"""
  add boundary conditions to rhs of linear system
"""
function constrain_rhs(rhs, u, pin, L_constrained)
  b = rhs - L_constrained*u[pin,:]

  for ic in pin
    b[ic,:] = u[ic,:]
  end
  b
end

"""
    poisson(L, b, u_boundary, boundary_index)

Solve the Poisson problem Lu = b, where L is a Laplace-beltrami operator with Dirichlet boundary specified by u_boundary at locations boundary_index.
"""
function poisson(L, b, u_boundary, boundary_index)
  (Lₒ, Lpin) = constrain_system(L, boundary_index)

  rhs = constrain_rhs(b, u_boundary, boundary_index, Lpin)

  Lₒ\rhs
end

"""
    apply(𝑑̂ₐ, α, P, k)

Apply discrete differential operator 𝑑̂ₐ to k-form α at points P with k-vector k.
"""
function apply(𝑑̂ₐ::D, α::KF, P::A, k::K) where{D<:DiscreteDifferentialOperator,
                                        F<:ZForm, KF<:KVector{F}, 
                                        T, V<:AbstractVector{T}, A<:AbstractVector{V},
                                        K<:KVector{T}}
  ϕₐ = map(P) do pᵢ
    apply(α, pᵢ, k)
  end
  apply(𝑑̂ₐ, ϕₐ)
end

"""
    apply(𝑑̂ₐ, ϕₐ)

Apply discrete differential operator against ϕₐ samples taken at k-simplicies.
"""
apply(𝑑̂ₐ::D, ϕₐ::T) where {D<:DiscreteDifferentialOperator, T<:AbstractVector} = 𝑑̂ₐ.D*ϕₐ


# . It is helpful, as a matter of notation first, to consider differentiation as an abstract operation that accepts a function and returns another function

#==
𝑑̂(D::A) where A<:AbstractArray = (ϕ::Fu)->DiscreteDifferentialOperator()
⋆(DiscreteDifferentialOperator

abstract type Simplex end
struct Vertex <: Simplex end
struct Edge <: Simplex end
struct Face <: Simplex end

simplex_trait(::Type{VertexHandle}) = Vertex()
simplex_trait(::Type{HalfEdgeHandle}) = Edge()
simplex_trait(::Type{EdgeHandle}) = Edge()
simplex_trait(::Type{FaceHandle}) = Face()


𝑑( topo::Topology, ::Type{VertexHandle} ) = 𝑑̂( incidence(topo, VertexHandle, EdgeHandle, true) ) 
𝑑( topo::Topology, ::Type{EdgeHandle} ) = 𝑑̂( incidence(topo, EdgeHandle, FaceHandle, true) ) 
𝑑( topo::Topology, ::Type{FaceHandle} ) = 𝑑̂( [0.0] ) 

apply(𝑑̂₀(ϕ), P_flat_centred)

𝑑(::Vertex, mesh::T) where T =  
𝑑(::Edge, mesh::T) where T = (u, v, w)->edge(mesh, u) 

function ⋆̂(::Edge, mesh::T, h::H) where {T}

end

⋆̂(mesh::T, h::H) where {T,H} = ⋆̂(simplex_trait(H), mesh, h)
==#


end # module
