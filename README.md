[![Build Status](https://travis-ci.com/mewertd2/DiscreteDifferentialGeometry.jl.svg?branch=master)](https://travis-ci.com/digitaldomain/DiscreteDifferentialGeometry.jl)
[![codecov.io](https://codecov.io/github/digitaldomain/DiscreteDifferentialGeometry.jl/coverage.svg?branch=master)](https://codecov.io/github/mewertd2/DiscreteDifferentialGeometry.jl?branch=master)

# DiscreteDifferentialGeometry

The `DiscreteDifferentialGeometry` [Julia](http://julialang.org) package defines Types and methods to implement Discrete Differential Geometry.

## Discrete Exterior Calculus

A concise description of the implemented operators follows. 
For learning Discrete Exterior Calculus we recommend the course available on-line (http://geometry.cs.cmu.edu/ddg).

The Discrete Exterior Calculus methods and Types supported by this package closely follow CMU's DDG course.

The intention is to extend this package to Discrete Geometric Algebra, any differences from DEC as currently taught in Computer Science will be due to this.

### Operators

The `𝑑` and `⋆` operators interact with each other and elements of a simplical complex ( i.e. a mesh ) in a way illustrated by the following graph.  

      Ωᵢ : k-form domain of the mesh.  i.e. Ω₀ represents verts, Ω₁ edges, Ω₂ faces
      Ω⋆ᵢ : k-form dual domain of the mesh.  i.e. Ω⋆₂ represents dual verts, Ω₁ dual edges, Ω₀ dual faces
      ⋆ᵢ : hodge star operator
      𝑑ᵢ : differential operator

      Ω₀---𝑑₀-→Ω₁----𝑑₁-→Ω₂ 
      ↑        ↑         ↑   
    ⋆₀|⋆₀⁻   ⋆₁|⋆₁⁻    ⋆₂|⋆₂⁻
      ↓        ↓         ↓     
      Ω⋆₂←-𝑑₁--Ω⋆₁←-𝑑₀---Ω⋆₀ 

The dual side of this graph are the simplex components of a one ring cell around each vertex of the primal mesh.
This cell connects the centres of each face (dual faces) via dual edges.  To disambiguate, we use the word cell rather than dual from now on.
The Ω⋆₂ cell is the circumcentric area around the vertex Ω₀.
The Ω⋆₁ edges are perpendicular to the primal Ω₁ edges.
The Ω⋆₀ cell vertices are the centres of a primal Ω₂ face.  
The circumcentre is conceptually the correct Ω⋆₀ centre, although in practice the barycentre may be used instead.

#### Discrete Differential

The method `𝑑` implements the discrete differential operator on discrete k-forms or discrete differential forms.

Also called the exterior derivative.

You can enter this into julia with the character sequence "\itd[Tab]"

In the discrete setting, k-forms and differential forms are typically represented as matrices and vectors.  Many of which you will recoginize as incidence matrices or simple column/row vectors corresponding to values at vertices/edges/faces.

#### Discrete Hodge Star

The method `⋆` implements the discrete hodge star on discrete k-forms or discrete differential forms.

You can enter this into julia with the character sequence "\star[Tab]"

### Discrete Codifferential

The method `δ` implements the discrete codifferential. 

    δ = ⋆𝑑⋆

You can enter this into julia with the character sequence "\delta[Tab]"

### Discrete Laplace Operator

The method `Δ` implements the discrete Laplace-Beltrami operator when applied to a triangle mesh.

You can enter this into julia with the character sequence "\Delta[Tab]"

By far the most important operator in DiscreteDifferentialGeometry.

    Δ = δ𝑑 + 𝑑δ

### Examples

Solve a simple poisson problem that doesn't involve boundary conditions.
Simple heat diffusion.

    julia> using DiscreteDifferentialGeometry, LinearAlgebra

    # For triangle mesh topology
    julia> using HalfEdges; using HalfEdges: loadmesh

    # Load a mesh
    julia> topo, P = loadmesh("resource/mesh/bunny.obj")

    # Discrete Laplace-Beltrami operator
    julia> L = Δ(topo, P);

    # Time step
    julia> dt = 1.0/24.0;

    # System matrix for implicit integration step applied to heat diffusion.
    julia> A = array(L*dt) + I;

    # Set up right hand side with some hot and cold spots 
    julia> b = zeros(nvertices(topo)); b[1000:1000:14000] = repeat([100.0,-100.0],7);

    # Solve the system for new temperatures.
    julia> b′ = A\b;

    # Step it one more time to get a bit more diffusion
    julia> b′ = A\b′;

    # Display the results
    julia> using Makie; mesh(P, reduce(hcat, HalfEdges.facelist(topo))', color = b′)

## Project Information

### Contributing

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

### Authors

* **Michael Alexander Ewert** - Developer - [Digital Domain](https://digitaldomain.com)

### License

This project is licensed under a modified Apache 2.0 license - see the [LICENSE](./LICENSE) file for details
