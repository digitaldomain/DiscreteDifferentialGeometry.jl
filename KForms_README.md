# KForms

The `KForms` [Julia](http://julialang.org) package defines differential operators `𝑑, δ, Δ` and `apply` methods for continous [k-forms](https://en.wikipedia.org/wiki/Exterior_derivative).  

Applying k-forms at a point in a subspace over a specified volume ( area ) is a way of adding up little differential quantities defined for some arbitrary function.

A k-form is represented as a `KVector` with a `ZForm` field.  The `ZForm` is a 0-form, i.e. a scalar function.  Applying the differential operator to a 0-form will result in a 1-form, or equivalently a 1-vector with a 0-form holding the derivative of the original 0-form as it's field.
To get a value you need to `apply` the k-form at a point in space to a p-vector.  This "measures" the p-vector with the `ZForm` function at the point via a contraction with the k-form.

## Exterior Calculus

Support for Exterior Calculus is provided.

k-forms are supported, although notation will be different.  For greatest overlap with Geometric Algebra, the basis vectors for 1-forms are the same as the primary basis 1-blades.

The main operator is `𝑑` which implements the [Exterior Derivative](https://en.wikipedia.org/wiki/Exterior_derivative) 
Applying `𝑑` to a k-form results in a (k+1)-form.  

On a scalar function, `𝑑` produces a form with the action of a directional derivative.

Together with a couple more axioms this defines `𝑑` [in terms of axioms](https://en.wikipedia.org/wiki/Exterior_derivative#In_terms_of_axioms).

Utilizing the most trivial bijection between 1-blades and coordinate differentials, KVectors can be used to apply the Exterior Derivative to any k-form [in terms of local coordinates](https://en.wikipedia.org/wiki/Exterior_derivative#In_terms_of_local_coordinates)

## KVectors
Most of the heavy lifting is done by the [Multivectors](https://github.com/mewertd2/Multivectors.jl) package.  This is a Geometric Algebra flavoured implementation of k-forms.

## Examples
Easier to demonstrate with some code than mathematical word salad.

    julia> using Multivectors, KVectors, KForms

    julia> @generate_basis("++",false,true,true)

Simple 1-form `α = xe₁ + ye₂`

    # scalar function for each 0-form ( there are two ) and first partial derivatives in ZForm{2}  
    # the 2 in ZForm{2} indicates the manifold (surface) is 2 dimensional
    # the 0-forms each take 2 coordinate which are called during apply
    julia> α = e₁(ZForm{2}((x,y)->x, [(x,y)->1, (x,y)->0])) + 
               e₂(ZForm{2}((x,y)->y, [(x,y)->0, (x,y)->1]))

    # apply the 1-form α to the 1-vector 2e₂ at the point 1e₁+1e₂
    julia> apply(α, KVector(2e₂), 1e₁+1e₂)
    2  

Zero form: `ϕ(x,y) = ½e⁻⁽ˣ²⁺ʸ²)`

    # dϕ = -2*ϕ*(xdx + ydy) 
    # expected result: -2*ϕ*x*e₁ + -2*ϕ*y*e₁
    # -2*ϕ(0.2,0.3)*0.2*e₁ + -2*ϕ*(0.2,0.3)*0.3*e₂
    julia> xy = 0.2e₁+0.3e₂
    
    julia> ϕ(x,y) = 0.5*exp(-(x^2+y^2))

    julia> α = ZForm{2}(ϕ, [(x,y)->-2.0*ϕ(x,y)*x,
                            (x,y)->-2.0*ϕ(x,y)*y])

Differential operator takes it from a 0-form to a 1-form

    julia> dα = 𝑑(α)
    julia> dα = sortbasis(dp)
    julia> apply(dp, xy, 1.0e₁+1.0e₂) ==  -2.0*ϕ(coords(xy)...)*xy[1].x + -2.0*ϕ(coords(xy)...)*xy[2].x

See the [tests](./test/KForms_runtests.jl) for more examples.
