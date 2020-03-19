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
using HalfEdges
using LinearAlgebra
using StaticArrays
using Base.Iterators

using DiscreteDifferentialGeometry: cotan

zerofun(x,y...) = zero(x)

include("KForms_runtests.jl")

module G2
  using Multivectors
  # need last param to be true for differential forms
  @generate_basis("++",false,true,true)
end
using .G2

module G3
  using Multivectors
  @generate_basis("+++",false,true,true)
end

using .G3

function cube()
  P = [0 0 0
       1 0 0
       0 0 1
       1 0 1
       1 1 1
       0 1 1
       1 1 0
       0 1 0
      ]
  P = SVector{3}.(eachrow(Float64.(P)))
  (Topology([1,2,3, 2,4,3, 3,4,6, 4,5,6, 2,5,4, 2,7,5, 7,8,5, 5,8,6, 1,3,6, 1,6,8, 1,8,2, 2,8,7]), P)
end

# not actually a D8, two tetrahedron glued together 
function D8()
  w = sqrt(2.0); peak = 0.8164965809277261; base = -0.4082482904638631;
  P = [ -w/2 base 0.0
       w/2 base 0.0
       0.0 peak 0.0
       0.0 0.0 peak-base
       0.0 0.0 base-peak ]

  P = SVector{3}.(eachrow(Float64.(P)))
  (Topology([1,2,4, 3,4,2, 1,4,3, 1,5,2, 1,3,5, 5,3,2]), P)
end

@testset begin "interpolation 3D"
  e₁, e₂, e₃  = alle( G3, 3)[1:3]

  # an equlateral triangle
  l = -0.5e₁ + 0.0e₂; r = 0.5e₁+0.0e₂; t = 0.0e₁ + e₂(sqrt(3.0)/2.0);

  # integrate whitney interpolating form over the edge opposite the bottom left corner
  # should end up with the same value we set on the edge ( 1.0 )
  ddg_ = DiscreteDifferentialGeometry
  dx = 0.005;  rtval = 42.0;
  edge_rt = sum(map(x->ddg_.interpolate((l,r,t), [rtval,0.0,0.0], r + (t-r)*x)⋅(t-r)*dx, 0.0:dx:(1.0-dx)))
  @test edge_rt ≈ rtval 
  @test norm(ddg_.interpolate((l,r,t), [1.0,1.0,1.0], (r+l+t)/3.0)) ≈ 0.0
  # interpolate gradient of function with value of 1.0 at l and 0.0 at opposite edge
  gradl = ddg_.interpolate((l,r,t), [0.0,1.0,1.0], (r+l+t)/3.0)
  @test gradl⋅(1.0e₁+1.0e₂) < -0.1
  @test norm(gradl⋅⋆(t-r)) ≈ 0.0

end

function interpolate_ga(pa::KVector,pb,pc,va,vb,vc)
  tri𝐼 = (pb-pa)∧(pc-pa)
  tri𝐼 = tri𝐼/Multivectors.norm_sqr(tri𝐼)
  vb*((pa-pc)⋅tri𝐼) + va*((pc-pb)⋅tri𝐼) + vc*((pb-pa)⋅tri𝐼)
end

interpolate_ga(pa::SVector{Float64},pb,pc,ea,eb,ec) = interpolate_ga(KVector(pa), KVector(pb), KVector(pc),ea,eb,ec) 

function interpolate_vec(pa,pb,pc,va,vb,vc)
  N = (pb-pa)×(pc-pa)
  twoA = norm(N)
  N = N/twoA
  1.0/twoA*(N×(va*(pc-pb)+vb*(pa-pc)+vc*(pb-pa)))
end

@testset begin "operators"
  e₁, e₂, e₃  = alle( G3, 3)[1:3]
  e¹, e², e³ = alld( G3, 3)[1:3]
  𝐼 = pseudoscalar(e₁)

  #== hex grid
        

  1.732    1---2---3 
          / \ / \ / \
  0.866  4---5---6---7
        / \ / \ / \ / \
  0    8---9---A---B---C
        \ / \ / \ / \ /
         D---E---F---16
          \ / \ / \ /
  -1.732   17--18--19
  ==#

  A = 10; B = 11; C = 12; D = 13; E = 14; F = 15;
  h = 1.732

  topo_hex = Topology([1,4,5, 2,1,5, 2,5,6, 3,2,6, 3,6,7,  
                       4,8,9, 5,4,9, 5,9,A, 6,5,A, 6,A,B, 7,6,B, 7,B,C,
                       9,8,D, 9,D,E, A,9,E, A,E,F, B,A,F, B,F,16, C,B,16,
                       E,D,17, E,17,18, F,E,18, F,18,19, 16,F,19])
                       
  P_hex = 
    SVector{3, Float64}.([[-1,h,0],[0,h,0],[1,h,0],
                          [-1.5,.5h,0],[-.5,.5h,0],[.5,.5h,0],[1.5,.5h,0],
                          [-2,0,0],[-1,0,0],[0,0,0],[1,0,0],[2,0,0],
                          [-1.5,-.5h,0],[-.5,-.5h,0],[.5,-.5h,0],[1.5,-.5h,0],
                          [-1,-h,0],[0,-h,0],[1,-h,0]])
  P = P_hex; topo = topo_hex  
  V = nvertices(topo); E = nedges(topo); F = nfaces(topo);

  @test iszero(𝑑(topo, 𝑑(topo))) == true
  @test size(array(𝑑(topo))) == (E, V)
  @test size(array(𝑑(topo,𝑑(topo)))) == (F, V)
  @test ⋆(topo,P,⋆(topo,P)).D |> tr ≈ nvertices(topo)
  @test ⋆(topo,P,⋆(topo,P)).D |> det ≈ 1.0

  @test size(array(⋆(topo, P))) == (V, V)
  @test size(array(⋆(topo, P, ⋆(topo, P)))) == (V, V)

  @test size(array(⋆(topo, P, 𝑑(topo)))) == (E, V)
  @test size(array(⋆(topo, P, ⋆(topo, P, 𝑑(topo))))) == (E, V)

  𝑑mesh = op->𝑑(topo, op)
  smesh = op->⋆(topo, P, op)

  # navigate this structure :
  #== 
   Ω₀---𝑑₀-→Ω₁----𝑑₁-→Ω₂ 
   ↑        ↑         ↑   
 ⋆₀|⋆₀⁻   ⋆₁|⋆₁⁻    ⋆₂|⋆₂⁻
   ↓        ↓         ↓     
   Ω⋆₂←-𝑑₁--Ω⋆₁←-𝑑₀---Ω⋆₀ 
  ==#
  # around the outside
  @test 𝑑(topo) |> 𝑑mesh |> smesh |> 𝑑mesh |> 𝑑mesh |> smesh |> size∘array == (V,V)
  # accross top dipping down twice
  @test 𝑑(topo) |> smesh |> smesh |> 𝑑mesh |> smesh |> smesh |> size∘array == (F,V)
  # left loop
  @test 𝑑(topo) |> smesh |> 𝑑mesh |> size∘array == (V,V)
  # right loop
  @test 𝑑(topo) |> 𝑑mesh |> smesh |> 𝑑mesh |> smesh |> size∘array == (E,V)

  # solve a simple heat diffusion step
  dt = (1.0/24.0)
  L = array(Δ(topo, P))*dt + I
  b = zeros(nvertices(topo)); b[1] = 100.0; b[19] = -75.0;
  x = L\b
  @test x[10] > 0.0 # middle. more heat than cold
  @test x[5] > 0.0  # close to hotspot
  @test x[15] < 0.0 # close to coldspot
  @test x[8] > 0.0 # closer to hotspot
  @test poisson(L, b, zeros(length(b)), []) ≈ x
  @test poisson(L, b, ones(length(b)), [5])[5] == 1.0
  @test poisson(L, b, ones(length(b)), [5])[6] != 1.0

  ## geodesic distance.
  ## 3 steps
  ## 1) integrate heat flow ( heat diffusion ).  
  
  #    Going to use symmetric laplacian, could use L from before if we wanted to
  Lc = array(Δcell(topo, P))

  # can get vertex area matrix M from the hodge star taking values of 1.0 on a vertex to the vertex cell
  M = ⋆(topo, P, PrimalKForm{0}(ones(nvertices(topo)))) |> array

  # or the hodge star method that implies 
  @test ⋆(topo,P) == ⋆(topo, P, PrimalKForm{0}(ones(nvertices(topo))))

  L = M + dt*Lc
  @test issymmetric(Lc) == true
  @test issymmetric(L) == true

  source = 5;
  b = zeros(nvertices(topo)); b[source] = 1.0
  # can leave this ⋆ out if desired, since entries of b are integrated dirac delta functions
  # leaving it in to show general way of using Δcell rather than Δ
  b = ⋆(topo,P)*b

  u = L\b

  # u holds the temperatures at each vertex

  # with vector algebra
  vec∇face = map( HalfEdges.polygons(topo) ) do abc
    a,b,c = abc
    pa,pb,pc = P[abc]
    N = (pb-pa)×(pc-pa)
    twoA = norm(N) 
    N = N/twoA
    1.0/twoA*(N×(u[a]*(pc-pb)+u[b]*(pa-pc)+u[c]*(pb-pa)))
  end

  alledge = HalfEdges.edges(topo);
  allface = HalfEdges.polygons(topo);
  
  # a 1-form taking values from vertices to edges.
  dϕ = 𝑑(topo, PrimalKForm{0}(ones(nvertices(topo))))
  # apply to u (values on vertices) and get gradient of u on edges
  ∇u = dϕ*u

  # interpolate with whitney 1-forms to triangle barycentres
  ∇face = map(enumerate(IncidentEdges(topo, FaceHandle))) do (iface, edgeindices)
    abc = (p->KVector(p, 1𝐼)).(P[allface[iface]])
    # triangles vertices and edges are ordered so first edge points at first vertex
    # so need to loop almost all the way around 
    nedge = length(edgeindices)
    oei = collect(take(drop(cycle(edgeindices), nedge-1), nedge))
    # order so value on edge is the one opposite each vertex
    edgeval = ∇u[oei] 
    faceabc = allface[iface]
    orient = map(ie->HalfEdges.orientation(faceabc, alledge[ie]), oei)
    edgeval = orient .* edgeval
    centrep = sum(abc)/3.0
    interpolate( abc, edgeval, centrep )
  end

  # similar to closed-form vector algebra but using Geometric Algebra
  ga∇face = map( HalfEdges.polygons(topo) ) do abc
    a,b,c = abc
    pa,pb,pc = map(p->KVector(p, 1.0𝐼), P[abc])
    tri𝐼 = (pb-pa)∧(pc-pa); tri𝐼 *= 1.0/Multivectors.norm_sqr(tri𝐼);
    (u[b]*((pa-pc)⋅tri𝐼) + u[a]*((pc-pb)⋅tri𝐼) + u[c]*((pb-pa)⋅tri𝐼))
  end

   
  @test abs(sum(sum(vec∇face - coords.(∇face)))) < eps(1.0)*nvertices(topo)
  @test abs(sum(sum(vec∇face - coords.(ga∇face)))) < eps(1.0)*nvertices(topo)

  ## 3) find divergence of our vector field at vertices 

  # go from dual triangle ( vertex in centre ) -> dual edge -> primal edge
  he_ = HalfEdges
  
  X = -map(normalize_safe, ∇face)

  vecX = map(coords, X)
  # vector algebra calculation
  vecdivX = map( map(h->OneRing(h, topo), he_.vertices(topo)) ) do vring
    vringⱼ = Iterators.filter(h->!he_.isboundary(topo,h), vring)
    
    cote(h) = ( fh = he_.face(topo, h);
                     Xⱼ = vecX[fh];
                     ea = P[he_.tail(topo, h)] - P[he_.head(topo, h)]; 
                     h = he_.next(topo, h);
                     eb = P[he_.head(topo, h)] - P[he_.tail(topo, h)];
                     ei = eb-ea;
                     cotθa = cotan(-eb, -ei);
                     cotθb = cotan(ei, -ea);
                     cotθa*(ea⋅Xⱼ) + cotθb*(eb⋅Xⱼ))

    0.5*mapreduce(cote, + ,vringⱼ)
  end

  # the distance function from our source point
  #vecϕ = Lc\vecdivX
  vecϕ = poisson(Lc, vecdivX, zeros(length(P)), [source])

  # with differential operators
  #
  # ∬𝑑ω = ⨚ω 
  # each time you encounter an (evaluation of) exterior derivative of a form, 
  # replace any evaluation over a simplex σ by a direct evaluation of the form itself over the boundary of σ
  # div F = ∇⋅F = ⋆𝑑⋆(F♭)
  
  #!me not quit working
  #==
  measuringform = ⋆(topo, P, 𝑑(topo))*array(P)
  valueform = 𝑑(topo, CellKForm{0}(ones(nfaces(topo))))*array(vecX)
  hecirculate = zeros(nhalfedges(topo));
  for (i,hh) in enumerate(UniqueHalfEdges(topo))
    hecirculate[hh] = measuringform[i,:]⋅valueform[i,:]
    hecirculate[he_.opposite(topo,hh)] = -hecirculate[hh]
  end
  decdivX = map(map(vh->OneRing(vh, topo), he_.vertices(topo))) do oner
    mapreduce(hh->hecirculate[hh], +, oner)
  end

  decϕ = poisson(Lc, decdivX, zeros(length(P)), [source])
  ==#
 
  #==
  sdf = ⋆(topo, P, 𝑑(topo, CellKForm{0}(ones(nfaces(topo)))))
  sdfX = array(sdf)*array(vecX)
  XX = SVector{3}[]
  resize!(XX, nhalfedges(topo))
  for (i,hh) in enumerate(UniqueHalfEdges(topo))
    XX[hh] = SVector{3}(sdfX[i,:])
    XX[he_.opposite(topo, hh)] = -XX[hh]
  end

  decX = map(map(vh->OneRing(vh, topo), he_.vertices(topo))) do oner
    mapreduce(hh->XX[hh], +, oner)
  end
  decdivX = sum.(decX)
  #decϕ = Lc\decdivX
  decϕ = poisson(Lc, decdivX, 10.0*ones(length(P)), [source])
  ==#
   
  # start with primal 1-form.
  # since it's a 1-form we will ultimately need ωᵢ⋅Xᵢ
  # this could be accomplished  

  #== WIP
  gc = array(vecX)

  divX = ⋆(topo, P, 𝑑(topo, CellKForm{0}(gc[:,1])))
  divY = ⋆(topo, P, 𝑑(topo, CellKForm{0}(gc[:,2])))
  divZ = ⋆(topo, P, 𝑑(topo, CellKForm{0}(gc[:,3])))
  
  divG = array(divX+divY+divZ)*ones(nfaces(topo))
 
  divVert = zeros(nvertices(topo))
  for (iedge, uhe) in enumerate(UniqueHalfEdges(topo))
    b = head(topo, uhe)
    a = tail(topo, uhe)

    if a < b 
      divVert[a] += divG[iedge]
      divVert[b] -= divG[iedge]
    else
      divVert[b] += divG[iedge]
      divVert[a] -= divG[iedge]
    end

  end
  
  b = -divVert
  ϕ = Lc\b
  ==#

    #!me need barycentric dual simplex for this to work ( otherwise ⋆₁⁻¹ has divide by zero )  
  #== a 3,2,3,2,3 grid
  
  1   1---2---3 
      |\ / \ /|
      | 4---5 |
      |/|\ /|\|
  0   6 | 7 | 8
      |\|/ \|/|
      | 9---A |
      |/ \ / \|
  -1  B---C---D
  y    
  ↑→x -1  0   1  
  ==#
  #==
  topo = Topology( [4,2,1,4,5,2,5,3,2,  6,4,1,7,5,4,5,8,3, 6,9,4, 9,7,4,7,10,5,10,8,5, 
             11,9,6,9,10,7,10,13,8,   11,12,9,9,12,10,12,13,10] )

  @test iszero(𝑑(topo, 𝑑(topo))) == true

  P_flat_centred = 
    SVector{3, Float64}.([[-1,1,0],[0,1,0],[1,1,0],
                         [-0.5,0.5,0],[0.5,0.5,0],
                         [-1,0,0],[0,0,0],[1,0,0],
                         [-0.5,-0.5,0],[0.5,-0.5,0],
                         [-1,-1,0],[0,-1,0],[1,-1,0]])
  P_flat_centred_2D = 
    SVector{2, Float64}.([[-1,1],[0,1],[1,1],
                         [-0.5,0.5],[0.5,0.5],
                         [-1,0],[0,0],[1,0],
                         [-0.5,-0.5],[0.5,-0.5],
                         [-1,-1],[0,-1],[1,-1]])

  P = P_flat_centred_2D
  ==#

end


#!me cool example. need to work out some issues still
# kinda works but there appears to be some discretization artifacts visible on sphere.
function Δbarycentre(topo, P) 
  he_ = HalfEdges
  Δrows = map(1:nvertices(topo)) do i
    oringᵢ = OneRing(topo, VertexHandle(i))
    map( oringᵢ ) do hⱼ 
      vᵢ = P[he_.tail(topo, hⱼ)]
      a = P[he_.head(topo, hⱼ)]
      b = P[he_.head(topo, he_.next(topo, hⱼ))]

      vab = (vᵢ+a+b)/3.0

      ohⱼ = he_.opposite(topo, hⱼ)
      if !he_.isboundary(topo, ohⱼ)
        c = P[he_.head(topo, he_.next(topo, ohⱼ))]
        vca = (vᵢ+c+a)/3.0 
      else
        vca = (vᵢ+a)/2.0
      end

      # a rotor taking objects in the tangent space of vᵢ from barycentre of ∠vab to barycentre of ∠vca
      #!me normalizing or any scaling needed here?
      rotor = (vab-vᵢ)/(vca-vᵢ)
      (he_.head(topo, hⱼ), rotor)
    end
  end

end

vector(k::KA) where {K<:KVector, KA<:AbstractVector{K}} = SVector{3}.(coords.(k))
function curvature_rotor(topo, Pga)
  Lbi = Δbarycentre(topo, Pga)
  Lbidot(i,P) = mapreduce( ((vj, rot),)-> rot*(Pga[vj]-Pga[i]) - (Pga[vj]-Pga[i]), +, Lbi[i])
  M = ⋆(topo, vector(Pga), PrimalKForm{0}(ones(nvertices(topo)))) |> array
  bicurve = inv(M)*map(i->Lbidot(i, Pga), 1:length(Pga))
  su = norm.(bicurve)
end

#==


  vector(k::KA) where {K<:KVector, KA<:AbstractVector{K}} = SVector{3}.(coords.(k))
  @generate_basis("+++")
  𝐼 = pseudoscalar(e₁)
  Pga = map(p->KVector(p, 1𝐼), (P))

  Lbi = Δbarycentre(topo, Pga)
  Lbidot(i,P) = mapreduce( ((vj, rot),)-> normalize(rot)*(Pga[i]-Pga[vj]), +, Lbi[i])
  M = ⋆(topo, vector(Pga), PrimalKForm{0}(ones(nvertices(topo)))) |> array
  bicurve = inv(M)*map(i->Lbidot(i, Pga), 1:length(Pga))
  su = norm.(bicurve)
  drawvertexx(topo, vector(Pga), su, [])

==#
#==
## Example: Exploring Discrete Laplace-Beltrami Operators

To illustrate the utillity of Multivectors as a tool for prototyping and exploring Geometric Algebra in CGI applications, we develop a Laplace-Beltrami operator using Multivector Types.

The core idea is that we can use the action of rotors on gradient vectors to compute their divergence in the neighbourhood of a vertex.  Mapping between the smooth setting and the discrete, each rotor on a face is conceptually operating in the tangent space generated by the differential of scalar function at the vertex.  
From this argument we can see that, in the limit, all the rotors are acting in the same plane on gradient vectors in that same plane.  The important observation is that we don't need half angles or the sandwich operator to apply these tangent rotors.  The half-angle and sandwich operator are so the vector elements perpendicular to the plane of rotation are correctly handled.  We simply apply the rotors using the geometric product.  This makes the application rotors linear and simple to compute.

To compute the rotors for each incident edge ( triangle pair ) we just need the vertex, a start point and an end point.  The ratio of the two offset points is the desired rotor.  Note that the cotan formula used in the standard formulation can be interpreted as the aspect _ratio_ of integrated area contribution.
 
Now it is somewhat trivial to explore different discretizations.  All we need to do is compute the actual geometric points involved.  Circumcentre, orthocentre, barycentre, are all just points to our algorithm.

    function ΔGA(topo, P, cellvert) 
      he_ = HalfEdges
      Δrows = map(1:nvertices(topo)) do i
        oringᵢ = OneRing(topo, VertexHandle(i))
        map( oringᵢ ) do hⱼ
          # points for triangle 1
          vᵢ = P[he_.tail(topo, hⱼ)]
          a  = P[he_.head(topo, hⱼ)]
          b_left = P[he_.head(topo, he_.next(topo, hⱼ))]

          # points for triangle 2
          ohⱼ = he_.opposite(topo, hⱼ)
          b_right = P[he_.head(topo, he_.next(topo, ohⱼ))]

          # point on triangle 1 (left of edge)
          v1 = cellvert(vᵢ, a, b_left)

          # point on triangle 2 (right of edge)
          v2 = cellvert(vᵢ, a, b_right)

          # a rotor taking objects in the tangent space of vᵢ from right triangle to left
          rotor = (v1-vᵢ)/(v2-vᵢ)

          # store the index of the vertex at end of edge and the rotor across the edge
          (he_.head(topo, hⱼ), rotor)
        end
      end
    end

    barycentre(v, a, b) = (v+a+b)/3.0
==#
