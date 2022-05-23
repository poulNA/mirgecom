r""":mod:`mirgecom.euler` helps solve Euler's equations of gas dynamics.

Euler's equations of gas dynamics:

.. math::

    \partial_t \mathbf{Q} = -\nabla\cdot{\mathbf{F}} +
    (\mathbf{F}\cdot\hat{n})_{\partial\Omega}

where:

-  state $\mathbf{Q} = [\rho, \rho{E}, \rho\vec{V}, \rho{Y}_\alpha]$
-  flux $\mathbf{F} = [\rho\vec{V},(\rho{E} + p)\vec{V},
   (\rho(\vec{V}\otimes\vec{V}) + p*\mathbf{I}), \rho{Y}_\alpha\vec{V}]$,
-  unit normal $\hat{n}$ to the domain boundary $\partial\Omega$,
-  vector of species mass fractions ${Y}_\alpha$,
   with $1\le\alpha\le\mathtt{nspecies}$.

RHS Evaluation
^^^^^^^^^^^^^^

.. autofunction:: euler_operator

Logging Helpers
^^^^^^^^^^^^^^^

.. autofunction:: units_for_logging
.. autofunction:: extract_vars_for_logging
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

from grudge.projection import volume_quadrature_project
from grudge.flux_differencing import volume_flux_differencing
from arraycontext import map_array_container
import numpy as np  # noqa
from meshmode.dof_array import DOFArray
import grudge.op as op
from grudge.dof_desc import DOFDesc, as_dofdesc
from grudge.trace_pair import (
    interior_trace_pairs, TracePair
)
from mirgecom.gas_model import (
    make_operator_fluid_states,
    conservative_to_entropy_vars,
    entropy_to_conservative_vars,
    project_fluid_state,
    make_fluid_state_trace_pairs
)
from mirgecom.inviscid import (
    inviscid_flux,
    inviscid_facial_flux_rusanov,
    inviscid_flux_on_element_boundary,
    entropy_stable_inviscid_flux_rusanov,
    entropy_conserving_flux_chandrashekar,
    make_entropy_projected_fluid_state
)

from mirgecom.operators import div_operator


def euler_operator(discr, state, gas_model, boundaries, time=0.0,
                   inviscid_numerical_flux_func=inviscid_facial_flux_rusanov,
                   quadrature_tag=None, operator_states_quad=None):
    r"""Compute RHS of the Euler flow equations.

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars`

        The right-hand-side of the Euler flow equations:

        .. math::

            \dot{\mathbf{q}} = - \nabla\cdot\mathbf{F} +
                (\mathbf{F}\cdot\hat{n})_{\partial\Omega}

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state, and dependent
        quantities.

    boundaries

        Dictionary of boundary functions, one for each valid btag

    time

        Time

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    quadrature_tag

        An optional identifier denoting a particular quadrature
        discretization to use during operator evaluations.
        The default value is *None*.
    """
    dd_quad_vol = DOFDesc("vol", quadrature_tag)
    dd_quad_faces = DOFDesc("all_faces", quadrature_tag)

    if operator_states_quad is None:
        operator_states_quad = make_operator_fluid_states(discr, state, gas_model,
                                                          boundaries, quadrature_tag)

    volume_state_quad, interior_state_pairs_quad, domain_boundary_states_quad = \
        operator_states_quad

    # Compute volume contributions
    inviscid_flux_vol = inviscid_flux(volume_state_quad)

    # Compute interface contributions
    inviscid_flux_bnd = inviscid_flux_on_element_boundary(
        discr, gas_model, boundaries, interior_state_pairs_quad,
        domain_boundary_states_quad, quadrature_tag=quadrature_tag,
        numerical_flux_func=inviscid_numerical_flux_func, time=time)

    return -div_operator(discr, dd_quad_vol, dd_quad_faces,
                         inviscid_flux_vol, inviscid_flux_bnd)


def entropy_stable_euler_operator(
        discr, state, gas_model, boundaries, time=0.0,
        inviscid_numerical_flux_func=entropy_stable_inviscid_flux_rusanov,
        quadrature_tag=None):
    """Compute RHS of the Euler flow equations using flux-differencing.

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state, and dependent
        quantities.

    boundaries

        Dictionary of boundary functions, one for each valid btag

    time

        Time

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    quadrature_tag
        An optional identifier denoting a particular quadrature
        discretization to use during operator evaluations.
        The default value is *None*.

    Returns
    -------
    :class:`mirgecom.fluid.ConservedVars`

        Agglomerated object array of DOF arrays representing the RHS of the Euler
        flow equations.
    """
    dd_base_vol = as_dofdesc("vol")
    dd_quad_vol = DOFDesc("vol", quadrature_tag)
    dd_quad_faces = DOFDesc("all_faces", quadrature_tag)

    # NOTE: For single-gas this is just a fixed scalar.
    # However, for mixtures, gamma is a DOFArray. For now,
    # we are re-using gamma from here and *not* recomputing
    # after applying entropy projections. It is unclear at this
    # time whether it's strictly necessary or if this is good enough
    gamma = gas_model.eos.gamma(state.cv, state.temperature)

    # Interpolate state to vol quad grid
    # state_quad = project_fluid_state(discr, "vol", dd_vol, state, gas_model)

    # Compute the projected (nodal) entropy variables
    entropy_vars = volume_quadrature_project(
        discr, dd_base_vol,
        # Map to entropy variables
        conservative_to_entropy_vars(gamma, state))

    modified_conserved_fluid_state = \
        make_entropy_projected_fluid_state(discr, dd_quad_vol, dd_quad_faces,
                                           state, entropy_vars, gamma,
                                           gas_model)

    from functools import partial

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
        -volume_flux_differencing(discr, dd_base_vol, dd_quad_faces, flux_matrices)

    def interp_to_surf_quad(utpair):
        local_dd = utpair.dd
        local_dd_quad = local_dd.with_discr_tag(quadrature_tag)
        return TracePair(
            local_dd_quad,
            interior=op.project(discr, local_dd, local_dd_quad, utpair.int),
            exterior=op.project(discr, local_dd, local_dd_quad, utpair.ext)
        )

    tseed_interior_pairs_quad = None
    if state.is_mixture:
        # If this is a mixture, we need to exchange the temperature field because
        # mixture pressure (used in the inviscid flux calculations) depends on
        # temperature and we need to seed the temperature calculation for the
        # (+) part of the partition boundary with the remote temperature data.
        tseed_interior_pairs_quad = [
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
            gamma = op.project(discr, dd_base_vol, local_dd_quad, gamma)
        return TracePair(
            local_dd_quad,
            # Convert interior and exterior states to conserved variables
            interior=entropy_to_conservative_vars(gamma, vtilde_tpair.int),
            exterior=entropy_to_conservative_vars(gamma, vtilde_tpair.ext)
        )

    cv_interior_pairs_quad = [
        # Compute interior trace pairs using modified conservative
        # variables on the quadrature grid
        # (obtaining state from projected entropy variables)
        interp_to_surf_modified_conservedvars(gamma, tpair)
        for tpair in interior_trace_pairs(discr, entropy_vars)
    ]

    domain_boundary_states_quad = {
        # TODO: Use modified conserved vars as the input state?
        # Would need to make an "entropy-projection" variant
        # of *project_fluid_state*
        btag: project_fluid_state(
            discr, dd_base_vol,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            state, gas_model) for btag in boundaries
    }

    # Interior interface state pairs consisting of modified conservative
    # variables and the corresponding temperature seeds
    interior_state_pairs_quad = \
        make_fluid_state_trace_pairs(cv_interior_pairs_quad, gas_model,
                                     tseed_interior_pairs_quad)

    # Surface contributions
    inviscid_flux_bnd = inviscid_flux_on_element_boundary(
        discr, gas_model, boundaries, interior_state_pairs_quad,
        domain_boundary_states_quad, quadrature_tag=quadrature_tag,
        numerical_flux_func=inviscid_numerical_flux_func, time=time)

    return op.inverse_mass(
        discr,
        inviscid_flux_vol - op.face_mass(discr, dd_quad_faces, inviscid_flux_bnd)
    )


# By default, run unitless
NAME_TO_UNITS = {
    "mass": "",
    "energy": "",
    "momentum": "",
    "temperature": "",
    "pressure": ""
}


def units_for_logging(quantity: str) -> str:
    """Return unit for quantity."""
    return NAME_TO_UNITS[quantity]


def extract_vars_for_logging(dim: int, state, eos) -> dict:
    """Extract state vars."""
    dv = eos.dependent_vars(state)

    from mirgecom.utils import asdict_shallow
    name_to_field = asdict_shallow(state)
    name_to_field.update(asdict_shallow(dv))
    return name_to_field
