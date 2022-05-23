r""":mod:`mirgecom.navierstokes` methods and utils for compressible Navier-Stokes.

Compressible Navier-Stokes equations:

.. math::

    \partial_t \mathbf{Q} + \nabla\cdot\mathbf{F}_{I} = \nabla\cdot\mathbf{F}_{V}

where:

-  fluid state $\mathbf{Q} = [\rho, \rho{E}, \rho\mathbf{v}, \rho{Y}_\alpha]$
-  with fluid density $\rho$, flow energy $E$, velocity $\mathbf{v}$, and vector
   of species mass fractions ${Y}_\alpha$, where $1\le\alpha\le\mathtt{nspecies}$.
-  inviscid flux $\mathbf{F}_{I} = [\rho\mathbf{v},(\rho{E} + p)\mathbf{v}
   ,(\rho(\mathbf{v}\otimes\mathbf{v})+p\mathbf{I}), \rho{Y}_\alpha\mathbf{v}]$
-  viscous flux $\mathbf{F}_V = [0,((\tau\cdot\mathbf{v})-\mathbf{q}),\tau_{:i}
   ,J_{\alpha}]$
-  viscous stress tensor $\mathbf{\tau} = \mu(\nabla\mathbf{v}+(\nabla\mathbf{v})^T)
   + (\mu_B - \frac{2}{3}\mu)(\nabla\cdot\mathbf{v})$
-  diffusive flux for each species $J_\alpha = \rho{D}_{\alpha}\nabla{Y}_{\alpha}$
-  total heat flux $\mathbf{q}=\mathbf{q}_c+\mathbf{q}_d$, is the sum of:
    -  conductive heat flux $\mathbf{q}_c = -\kappa\nabla{T}$
    -  diffusive heat flux $\mathbf{q}_d = \sum{h_{\alpha} J_{\alpha}}$
-  fluid pressure $p$, temperature $T$, and species specific enthalpies $h_\alpha$
-  fluid viscosity $\mu$, bulk viscosity $\mu_{B}$, fluid heat conductivity $\kappa$,
   and species diffusivities $D_{\alpha}$.

RHS Evaluation
^^^^^^^^^^^^^^

.. autofunction:: grad_cv_operator
.. autofunction:: grad_t_operator
.. autofunction:: ns_operator
"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from arraycontext import map_array_container
from functools import partial

from grudge.trace_pair import (
    TracePair,
    interior_trace_pairs,
    tracepair_with_discr_tag
)
from grudge.dof_desc import DOFDesc, as_dofdesc, DISCR_TAG_BASE
from grudge.projection import volume_quadrature_project
from grudge.flux_differencing import volume_flux_differencing

import grudge.op as op

from mirgecom.inviscid import (
    inviscid_flux,
    entropy_conserving_flux_chandrashekar,
    entropy_stable_inviscid_flux_rusanov,
    inviscid_facial_flux_rusanov,
    inviscid_flux_on_element_boundary
)
from mirgecom.viscous import (
    viscous_flux,
    viscous_facial_flux_central,
    viscous_flux_on_element_boundary
)
from mirgecom.flux import num_flux_central

from mirgecom.operators import (
    div_operator, grad_operator
)
from mirgecom.gas_model import (
    project_fluid_state,
    make_fluid_state_trace_pairs,
    make_entropy_projected_fluid_state,
    conservative_to_entropy_vars,
    entropy_to_conservative_vars
)
from mirgecom.gas_model import make_operator_fluid_states

from meshmode.dof_array import DOFArray
from arraycontext import thaw


class _NSGradCVTag:
    pass


class _NSGradTemperatureTag:
    pass


def _gradient_flux_interior(discr, numerical_flux_func, tpair):
    """Compute interior face flux for gradient operator."""
    from arraycontext import outer
    actx = tpair.int.array_context
    dd = tpair.dd
    normal = thaw(discr.normal(dd), actx)
    flux = outer(numerical_flux_func(tpair.int, tpair.ext), normal)
    return op.project(discr, dd, dd.with_dtag("all_faces"), flux)


def grad_cv_operator(
        discr, gas_model, boundaries, state, *, time=0.0,
        numerical_flux_func=num_flux_central,
        quadrature_tag=DISCR_TAG_BASE,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        operator_states_quad=None):
    r"""Compute the gradient of the fluid conserved variables.

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state, and dependent
        quantities.

    boundaries
        Dictionary of boundary functions keyed by btags

    time
        Time

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    quadrature_tag
        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    Returns
    -------
    :class:`numpy.ndarray`

        Array of :class:`~mirgecom.fluid.ConservedVars` representing the
        gradient of the fluid conserved variables.
    """
    dd_vol_quad = DOFDesc("vol", quadrature_tag)
    dd_faces_quad = DOFDesc("all_faces", quadrature_tag)

    if operator_states_quad is None:
        operator_states_quad = make_operator_fluid_states(
            discr, state, gas_model, boundaries, quadrature_tag)

    vol_state_quad, inter_elem_bnd_states_quad, domain_bnd_states_quad = \
        operator_states_quad

    get_interior_flux = partial(
        _gradient_flux_interior, discr, numerical_flux_func)

    cv_interior_pairs = [TracePair(state_pair.dd,
                                   interior=state_pair.int.cv,
                                   exterior=state_pair.ext.cv)
                         for state_pair in inter_elem_bnd_states_quad]

    cv_flux_bnd = (

        # Domain boundaries
        sum(bdry.cv_gradient_flux(
            discr,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            gas_model=gas_model,
            state_minus=domain_bnd_states_quad[btag],
            time=time,
            numerical_flux_func=numerical_flux_func)
            for btag, bdry in boundaries.items())

        # Interior boundaries
        + sum(get_interior_flux(tpair) for tpair in cv_interior_pairs)
    )

    # [Bassi_1997]_ eqn 15 (s = grad_q)
    return grad_operator(
        discr, dd_vol_quad, dd_faces_quad, vol_state_quad.cv, cv_flux_bnd)


def grad_t_operator(
        discr, gas_model, boundaries, state, *, time=0.0,
        numerical_flux_func=num_flux_central,
        quadrature_tag=DISCR_TAG_BASE,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        operator_states_quad=None):
    r"""Compute the gradient of the fluid temperature.

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state, and dependent
        quantities.

    boundaries
        Dictionary of boundary functions keyed by btags

    time
        Time

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    quadrature_tag
        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    Returns
    -------
    :class:`numpy.ndarray`

        Array of :class:`~meshmode.dof_array.DOFArray` representing the gradient of
        the fluid temperature.
    """
    dd_vol_quad = DOFDesc("vol", quadrature_tag)
    dd_faces_quad = DOFDesc("all_faces", quadrature_tag)

    if operator_states_quad is None:
        operator_states_quad = make_operator_fluid_states(
            discr, state, gas_model, boundaries, quadrature_tag)

    vol_state_quad, inter_elem_bnd_states_quad, domain_bnd_states_quad = \
        operator_states_quad

    get_interior_flux = partial(
        _gradient_flux_interior, discr, numerical_flux_func)

    # Temperature gradient for conductive heat flux: [Ihme_2014]_ eqn (3b)
    # Capture the temperature for the interior faces for grad(T) calc
    # Note this is *all interior faces*, including partition boundaries
    # due to the use of *interior_state_pairs*.
    t_interior_pairs = [TracePair(state_pair.dd,
                                  interior=state_pair.int.temperature,
                                  exterior=state_pair.ext.temperature)
                        for state_pair in inter_elem_bnd_states_quad]

    t_flux_bnd = (

        # Domain boundaries
        sum(bdry.temperature_gradient_flux(
            discr,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            gas_model=gas_model,
            state_minus=domain_bnd_states_quad[btag],
            time=time,
            numerical_flux_func=numerical_flux_func)
            for btag, bdry in boundaries.items())

        # Interior boundaries
        + sum(get_interior_flux(tpair) for tpair in t_interior_pairs)
    )

    # Fluxes in-hand, compute the gradient of temperature
    return grad_operator(
        discr, dd_vol_quad, dd_faces_quad, vol_state_quad.temperature, t_flux_bnd)


def ns_operator(discr, gas_model, state, boundaries, *, time=0.0,
                inviscid_numerical_flux_func=inviscid_facial_flux_rusanov,
                gradient_numerical_flux_func=num_flux_central,
                viscous_numerical_flux_func=viscous_facial_flux_central,
                quadrature_tag=DISCR_TAG_BASE, return_gradients=False,
                # Added to avoid repeated computation
                # FIXME: See if there's a better way to do this
                operator_states_quad=None,
                grad_cv=None, grad_t=None):
    r"""Compute RHS of the Navier-Stokes equations.

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state, and dependent
        quantities.

    boundaries
        Dictionary of boundary functions keyed by btags

    time
        Time

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    quadrature_tag
        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    Returns
    -------
    :class:`mirgecom.fluid.ConservedVars`

        The right-hand-side of the Navier-Stokes equations:

        .. math::

            \partial_t \mathbf{Q} = \nabla\cdot(\mathbf{F}_V - \mathbf{F}_I)
    """
    if not state.is_viscous:
        raise ValueError("Navier-Stokes operator expects viscous gas model.")

    dd_base = as_dofdesc("vol")
    dd_vol_quad = DOFDesc("vol", quadrature_tag)
    dd_faces_quad = DOFDesc("all_faces", quadrature_tag)

    # Make model-consistent fluid state data (i.e. CV *and* DV) for:
    # - Volume: vol_state_quad
    # - Element-element boundary face trace pairs: inter_elem_bnd_states_quad
    # - Interior states (Q_minus) on the domain boundary: domain_bnd_states_quad
    #
    # Note: these states will live on the quadrature domain if one is given,
    # otherwise they stay on the interpolatory/base domain.
    if operator_states_quad is None:
        operator_states_quad = make_operator_fluid_states(
            discr, state, gas_model, boundaries, quadrature_tag)

    vol_state_quad, inter_elem_bnd_states_quad, domain_bnd_states_quad = \
        operator_states_quad

    # {{{ Local utilities

    # transfer trace pairs to quad grid, update pair dd
    interp_to_surf_quad = partial(tracepair_with_discr_tag, discr, quadrature_tag)

    # }}}

    # {{{ === Compute grad(CV) ===

    if grad_cv is None:
        grad_cv = grad_cv_operator(
            discr, gas_model, boundaries, state, time=time,
            numerical_flux_func=gradient_numerical_flux_func,
            quadrature_tag=quadrature_tag,
            operator_states_quad=operator_states_quad)

    # Communicate grad(CV) and put it on the quadrature domain
    grad_cv_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair=tpair)
        for tpair in interior_trace_pairs(discr, grad_cv, tag=_NSGradCVTag)
    ]

    # }}} Compute grad(CV)

    # {{{ === Compute grad(temperature) ===

    if grad_t is None:
        grad_t = grad_t_operator(
            discr, gas_model, boundaries, state, time=time,
            numerical_flux_func=gradient_numerical_flux_func,
            quadrature_tag=quadrature_tag,
            operator_states_quad=operator_states_quad)

    # Create the interior face trace pairs, perform MPI exchange, interp to quad
    grad_t_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair=tpair)
        for tpair in interior_trace_pairs(discr, grad_t, tag=_NSGradTemperatureTag)
    ]

    # }}} compute grad(temperature)

    # {{{ === Navier-Stokes RHS ===

    # Compute the volume term for the divergence operator
    vol_term = (

        # Compute the volume contribution of the viscous flux terms
        # using field values on the quadrature grid
        viscous_flux(state=vol_state_quad,
                     # Interpolate gradients to the quadrature grid
                     grad_cv=op.project(discr, dd_base, dd_vol_quad, grad_cv),
                     grad_t=op.project(discr, dd_base, dd_vol_quad, grad_t))

        # Compute the volume contribution of the inviscid flux terms
        # using field values on the quadrature grid
        - inviscid_flux(state=vol_state_quad)
    )

    # Compute the boundary terms for the divergence operator
    bnd_term = (

        # All surface contributions from the viscous fluxes
        viscous_flux_on_element_boundary(
            discr, gas_model, boundaries, inter_elem_bnd_states_quad,
            domain_bnd_states_quad, grad_cv, grad_cv_interior_pairs,
            grad_t, grad_t_interior_pairs, quadrature_tag=quadrature_tag,
            numerical_flux_func=viscous_numerical_flux_func, time=time)

        # All surface contributions from the inviscid fluxes
        - inviscid_flux_on_element_boundary(
            discr, gas_model, boundaries, inter_elem_bnd_states_quad,
            domain_bnd_states_quad, quadrature_tag=quadrature_tag,
            numerical_flux_func=inviscid_numerical_flux_func, time=time)

    )
    ns_rhs = div_operator(discr, dd_vol_quad, dd_faces_quad, vol_term, bnd_term)
    if return_gradients:
        return ns_rhs, grad_cv, grad_t
    return ns_rhs


def entropy_stable_ns_operator(
        discr, state, gas_model, boundaries, time=0.0,
        inviscid_numerical_flux_func=entropy_stable_inviscid_flux_rusanov,
        gradient_numerical_flux_func=num_flux_central,
        viscous_numerical_flux_func=viscous_facial_flux_central,
        quadrature_tag=None):
    r"""Compute RHS of the Navier-Stokes equations using flux-differencing.

    Returns
    -------
    numpy.ndarray
        The right-hand-side of the Navier-Stokes equations:

        .. math::

            \partial_t \mathbf{Q} = \nabla\cdot(\mathbf{F}_V - \mathbf{F}_I)

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state, and dependent
        quantities.

    boundaries
        Dictionary of boundary functions, one for each valid btag

    time
        Time

    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.
        Implementing the transport properties including heat conductivity,
        and species diffusivities type(mirgecom.transport.TransportModel).

    quadrature_tag
        An optional identifier denoting a particular quadrature
        discretization to use during operator evaluations.
        The default value is *None*.

    Returns
    -------
    :class:`mirgecom.fluid.ConservedVars`

        Agglomerated object array of DOF arrays representing the RHS of the
        Navier-Stokes equations.
    """
    if not state.is_viscous:
        raise ValueError("Navier-Stokes operator expects viscous gas model.")

    actx = state.array_context
    dd_base = as_dofdesc("vol")
    dd_vol = DOFDesc("vol", quadrature_tag)
    dd_faces = DOFDesc("all_faces", quadrature_tag)
    # NOTE: For single-gas this is just a fixed scalar.
    # However, for mixtures, gamma is a DOFArray. For now,
    # we are re-using gamma from here and *not* recomputing
    # after applying entropy projections. It is unclear at this
    # time whether it's strictly necessary or if this is good enough
    gamma = gas_model.eos.gamma(state.cv, state.temperature)

    # Interpolate state to vol quad grid
    quadrature_state = \
        project_fluid_state(discr, dd_base, dd_vol, state, gas_model)

    # Compute the projected (nodal) entropy variables
    entropy_vars = volume_quadrature_project(
        discr, dd_vol,
        # Map to entropy variables
        conservative_to_entropy_vars(gamma, quadrature_state))

    modified_conserved_fluid_state = \
        make_entropy_projected_fluid_state(discr, dd_vol, dd_faces,
                                           state, entropy_vars, gamma, gas_model)

    def _reshape(shape, ary):
        if not isinstance(ary, DOFArray):
            return map_array_container(partial(_reshape, shape), ary)

        return DOFArray(ary.array_context, data=tuple(
            subary.reshape(grp.nelements, *shape)
            # Just need group for determining the number of elements
            for grp, subary in zip(discr.discr_from_dd("vol").groups, ary)))

    flux_matrices = entropy_conserving_flux_chandrashekar(
        gas_model,
        _reshape((1, -1), modified_conserved_fluid_state),
        _reshape((-1, 1), modified_conserved_fluid_state))

    # Compute volume derivatives using flux differencing
    inviscid_flux_vol = \
        -volume_flux_differencing(discr, dd_vol, dd_faces, flux_matrices)

    def interp_to_surf_quad(utpair):
        local_dd = utpair.dd
        local_dd_quad = local_dd.with_discr_tag(quadrature_tag)
        return TracePair(
            local_dd_quad,
            interior=op.project(discr, local_dd, local_dd_quad, utpair.int),
            exterior=op.project(discr, local_dd, local_dd_quad, utpair.ext)
        )

    tseed_interior_pairs = None
    if state.is_mixture:
        # If this is a mixture, we need to exchange the temperature field because
        # mixture pressure (used in the inviscid flux calculations) depends on
        # temperature and we need to seed the temperature calculation for the
        # (+) part of the partition boundary with the remote temperature data.
        tseed_interior_pairs = [
            # Get the interior trace pairs onto the surface quadrature
            # discretization (if any)
            interp_to_surf_quad(tpair)
            for tpair in interior_trace_pairs(discr, state.temperature)
        ]

    def interp_to_surf_modified_conservedvars(gamma, utpair):
        """Takes a trace pair containing the projected entropy variables
        and converts them into conserved variables on the quadrature grid.
        """
        local_dd = utpair.dd
        local_dd_quad = local_dd.with_discr_tag(quadrature_tag)
        # Interpolate entropy variables to the surface quadrature grid
        vtilde_tpair = op.project(discr, local_dd, local_dd_quad, utpair)
        if isinstance(gamma, DOFArray):
            gamma = op.project(discr, dd_base, local_dd_quad, gamma)
        return TracePair(
            local_dd_quad,
            # Convert interior and exterior states to conserved variables
            interior=entropy_to_conservative_vars(gamma, vtilde_tpair.int),
            exterior=entropy_to_conservative_vars(gamma, vtilde_tpair.ext)
        )

    cv_interior_pairs = [
        # Compute interior trace pairs using modified conservative
        # variables on the quadrature grid
        # (obtaining state from projected entropy variables)
        interp_to_surf_modified_conservedvars(gamma, tpair)
        for tpair in interior_trace_pairs(discr, entropy_vars)
    ]

    domain_bnd_states_quad = {
        # TODO: Use modified conserved vars as the input state?
        # Would need to make an "entropy-projection" variant
        # of *project_fluid_state*
        btag: project_fluid_state(
            discr, dd_base,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            state, gas_model) for btag in boundaries
    }

    # Interior interface state pairs consisting of modified conservative
    # variables and the corresponding temperature seeds
    inter_elem_bnd_states_quad = make_fluid_state_trace_pairs(cv_interior_pairs,
                                                              gas_model,
                                                              tseed_interior_pairs)

    # Surface contributions
    inviscid_flux_bnd = (

        # Domain boundaries
        sum(boundaries[btag].inviscid_divergence_flux(
            discr,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            gas_model,
            state_minus=boundary_states[btag],
            time=time,
            numerical_flux_func=inviscid_numerical_flux_func)
            for btag in boundaries)

        # Interior boundaries (using entropy stable numerical flux)
        + sum(inviscid_facial_flux(discr, gas_model=gas_model, state_pair=state_pair,
                                   numerical_flux_func=inviscid_numerical_flux_func)
              for state_pair in interior_states)
    )

    inviscid_term = op.inverse_mass(
        discr,
        inviscid_flux_vol - op.face_mass(discr, dd_faces, inviscid_flux_bnd)
    )

    def gradient_flux_interior(tpair):
        dd = tpair.dd
        normal = thaw(discr.normal(dd), actx)
        from arraycontext import outer
        flux = outer(gradient_numerical_flux_func(tpair), normal)
        return op.project(discr, dd, dd.with_dtag("all_faces"), flux)

    cv_flux_bnd = (

        # Domain boundaries
        sum(boundaries[btag].cv_gradient_flux(
            discr,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            gas_model=gas_model,
            state_minus=domain_bnd_states_quad[btag],
            time=time,
            numerical_flux_func=gradient_numerical_flux_func)
            for btag in domain_bnd_states_quad)

        # Interior boundaries
        + sum(gradient_flux_interior(tpair) for tpair in cv_interior_pairs)
    )

    # [Bassi_1997]_ eqn 15 (s = grad_q)
    grad_cv = grad_operator(discr, dd_vol, dd_faces,
                            quadrature_state.cv, cv_flux_bnd)

    grad_cv_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair)
        for tpair in interior_trace_pairs(discr, grad_cv)
    ]

    # Temperature gradient for conductive heat flux: [Ihme_2014]_ eqn (3b)
    # Capture the temperature for the interior faces for grad(T) calc
    # Note this is *all interior faces*, including partition boundaries
    # due to the use of *interior_state_pairs*.
    t_interior_pairs = [TracePair(state_pair.dd,
                                  interior=state_pair.int.temperature,
                                  exterior=state_pair.ext.temperature)
                        for state_pair in inter_elem_bnd_states_quad]

    t_flux_bnd = (

        # Domain boundaries
        sum(boundaries[btag].temperature_gradient_flux(
            discr,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            gas_model=gas_model,
            state_minus=boundary_states[btag],
            time=time)
            for btag in boundary_states)

        # Interior boundaries
        + sum(gradient_flux_interior(tpair) for tpair in t_interior_pairs)
    )

    # Fluxes in-hand, compute the gradient of temperature and mpi exchange it
    grad_t = grad_operator(discr, dd_vol, dd_faces,
                           quadrature_state.temperature, t_flux_bnd)

    grad_t_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair)
        for tpair in interior_trace_pairs(discr, grad_t)
    ]

    # viscous fluxes across interior faces (including partition and periodic bnd)
    def fvisc_divergence_flux_interior(state_pair, grad_cv_pair, grad_t_pair):
        return viscous_facial_flux(discr=discr, gas_model=gas_model,
                                   state_pair=state_pair, grad_cv_pair=grad_cv_pair,
                                   grad_t_pair=grad_t_pair,
                                   numerical_flux_func=viscous_numerical_flux_func)

    # viscous part of bcs applied here
    def fvisc_divergence_flux_boundary(btag, boundary_state):
        # Make sure we fields on the quadrature grid
        # restricted to the tag *btag*
        dd_btag = as_dofdesc(btag).with_discr_tag(quadrature_tag)
        return boundaries[btag].viscous_divergence_flux(
            discr=discr,
            btag=dd_btag,
            gas_model=gas_model,
            state_minus=boundary_state,
            grad_cv_minus=op.project(discr, dd_base, dd_btag, grad_cv),
            grad_t_minus=op.project(discr, dd_base, dd_btag, grad_t),
            time=time,
            numerical_flux_func=viscous_numerical_flux_func
        )

    visc_vol_term = viscous_flux(
        state=quadrature_state,
        # Interpolate gradients to the quadrature grid
        grad_cv=op.project(discr, dd_base, dd_vol, grad_cv),
        grad_t=op.project(discr, dd_base, dd_vol, grad_t))

    visc_bnd_term = (
        # All surface contributions from the viscous fluxes
        (
            # Domain boundary contributions for the viscous terms
            sum(fvisc_divergence_flux_boundary(btag, boundary_states[btag])
                for btag in boundary_states)

            # Interior interface contributions for the viscous terms
            + sum(fvisc_divergence_flux_interior(state_pair,
                                                 grad_cv_pair,
                                                 grad_t_pair)
                  for state_pair, grad_cv_pair, grad_t_pair in zip(
                      interior_states, grad_cv_interior_pairs,
                      grad_t_interior_pairs))
        )
    )
    viscous_term = div_operator(discr, dd_vol, dd_faces,
                                visc_vol_term, visc_bnd_term)

    # NS RHS
    return viscous_term + inviscid_term
    # }}} NS RHS
