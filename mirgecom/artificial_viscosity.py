r""":mod:`mirgecom.artificial_viscosity` Artificial viscosity for hyperbolic systems.

Consider the following system of conservation laws of the form:

.. math::

    \partial_t \mathbf{Q} + \nabla\cdot\mathbf{F} = 0,

where $\mathbf{Q}$ is the state vector and $\mathbf{F}$ is the vector of
fluxes. This module applies an artificial viscosity term by augmenting
the governing equations in the following way:

.. math::

    \partial_t \mathbf{Q} + \nabla\cdot\mathbf{F} =
    \nabla\cdot{\varepsilon\nabla\mathbf{Q}},

where $\varepsilon$ is the artificial viscosity coefficient.
To evalutate the second order derivative numerically, the problem
is recast as a set of first order problems:

.. math::

    \partial_t{\mathbf{Q}} + \nabla\cdot\mathbf{F} &= \nabla\cdot\mathbf{R} \\
    \mathbf{R} &= \varepsilon\nabla\mathbf{Q}

where $\mathbf{R}$ is an auxiliary variable, and the artificial viscosity
coefficient, $\varepsilon$, is spatially dependent and calculated using the
smoothness indicator of [Persson_2012]_:

.. math::

    S_e = \frac{\langle u_{N_p} - u_{N_{p-1}}, u_{N_p} -
        u_{N_{p-1}}\rangle_e}{\langle u_{N_p}, u_{N_p} \rangle_e}

where:

- $S_e$ is the smoothness indicator
- $u_{N_p}$ is the solution in modal basis at the current polynomial order
- $u_{N_{p-1}}$ is the truncated modal represention to the polynomial order $p-1$
- The $L_2$ inner product on an element is denoted $\langle \cdot,\cdot \rangle_e$

The elementwise viscosity is then calculated:

.. math::

    \varepsilon_e =
        \begin{cases}
            0, & s_e < s_0 - \kappa \\
            \frac{1}{2}\left( 1 + \sin \frac{\pi(s_e - s_0)}{2 \kappa} \right ),
            & s_0-\kappa \le s_e \le s_0+\kappa \\
            1, & s_e > s_0+\kappa
        \end{cases}

where:

- $\varepsilon_e$ is the element viscosity
- $s_e = \log_{10}{S_e} \sim 1/p^4$ is the smoothness indicator
- $s_0$ is a reference smoothness value
- $\kappa$ controls the width of the transition between 0 to 1


Boundary Conditions
^^^^^^^^^^^^^^^^^^^

The artificial viscosity operator as currently implemented re-uses the fluid
solution gradient $\nabla{\mathbf{Q}}$ for the auxiliary equation:

.. math::

    \mathbf{R} = \varepsilon\nabla\mathbf{Q}_\text{fluid}

As such, the fluid-system imposes the appropriate boundary solution $\mathbf{Q}^+$
for the comptuation of $\nabla{\mathbf{Q}}$.  This approach leaves the boundary
condition on $\mathbf{R}$ to be imposed by boundary treatment for the operator when
computing the divergence for the RHS, $\nabla \cdot \mathbf{R}$.

Similar to the fluid boundary treatments; when no boundary conditions are imposed
on $\mathbf{R}$, the interior solution is simply extrapolated to the boundary,
(i.e., $\mathbf{R}^+ = \mathbf{R}^-$).  If such a boundary condition is imposed,
usually for selected components of $\mathbf{R}$, then such boundary conditions
are used directly:  $\mathbf{R}^+ = \mathbf{R}_\text{bc}$.

A central numerical flux is then employed to transmit the boundary condition to
the domain for the divergence operator:

.. math::

    \mathbf{R} \cdot \hat{mathbf{n}} = \frac{1}{2}\left(\mathbf{R}^-
    + \mathbf{R}^+\right) \cdot \hat{\mathbf{n}}


Smoothness Indicator Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: smoothness_indicator

AV RHS Evaluation
^^^^^^^^^^^^^^^^^

.. autofunction:: av_laplacian_operator
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
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

import numpy as np

from pytools import memoize_in, keyed_memoize_in
from functools import partial
from meshmode.dof_array import DOFArray

from mirgecom.flux import num_flux_central
from mirgecom.operators import div_operator

from grudge.trace_pair import (
    interior_trace_pairs,
    tracepair_with_discr_tag
)

from grudge.dof_desc import (
    DOFDesc,
    as_dofdesc,
    DD_VOLUME_MODAL,
    DD_VOLUME
)

import grudge.op as op


class _AVCVTag:
    pass


class _AVRTag:
    pass


def av_laplacian_operator(discr, boundaries, fluid_state, alpha, gas_model=None,
                          kappa=1., s0=-6., time=0, operator_states_quad=None,
                          grad_cv=None, quadrature_tag=None, boundary_kwargs=None,
                          indicator=None, divergence_numerical_flux=num_flux_central,
                          **kwargs):
    r"""Compute the artificial viscosity right-hand-side.

    Computes the the right-hand-side term for artificial viscosity.

    .. math::

        \mbox{RHS}_{\mbox{av}} = \nabla\cdot{\varepsilon\nabla\mathbf{Q}}

    Parameters
    ----------
    fluid_state: :class:`mirgecom.gas_model.FluidState`
        Fluid state object with the conserved and thermal state.

    boundaries: dict
        Dictionary of boundary functions, one for each valid boundary tag

    alpha: float
        The maximum artificial viscosity coefficient to be applied

    indicator: :class:`~meshmode.dof_array.DOFArray`
        The indicator field used for locating where AV should be applied. If not
        supplied by the user, then
        :func:`~mirgecom.artificial_viscosity.smoothness_indicator` will be used
        with fluid mass density as the indicator field.

    kappa
        An optional argument that controls the width of the transition from 0 to 1,
        $\kappa$. This parameter defaults to $\kappa=1$.

    s0
        An optional argument that sets the smoothness level to limit
        on, $s_0$. Values in the range $(-\infty,0]$ are allowed, where $-\infty$
        results in all cells being tagged and 0 results in none.  This parameter
        defaults to $s_0=-6$.

    quadrature_tag
        An optional identifier denoting a particular quadrature
        discretization to use during operator evaluations.
        The default value is *None*.

    boundary_kwargs: :class:`dict`
        dictionary of extra arguments to pass through to the boundary conditions

    Returns
    -------
    :class:`mirgecom.fluid.ConservedVars`
        The artificial viscosity operator applied to *q*.
    """
    cv = fluid_state.cv
    actx = cv.array_context
    dd_vol = DOFDesc("vol", quadrature_tag)
    dd_faces = DOFDesc("all_faces", quadrature_tag)

    from warnings import warn

    if boundary_kwargs is not None:
        warn("The AV boundary_kwargs interface is deprecated, please pass gas_model"
             " and time directly.")
        if gas_model is None:
            gas_model = boundary_kwargs["gas_model"]
            if "time" in boundary_kwargs:
                time = boundary_kwargs["time"]

    interp_to_surf_quad = partial(tracepair_with_discr_tag, discr, quadrature_tag)

    def interp_to_vol_quad(u):
        return op.project(discr, "vol", dd_vol, u)

    if operator_states_quad is None:
        from mirgecom.gas_model import make_operator_fluid_states
        operator_states_quad = make_operator_fluid_states(
            discr, fluid_state, gas_model, boundaries, quadrature_tag)

    vol_state_quad, inter_elem_bnd_states_quad, domain_bnd_states_quad = \
        operator_states_quad

    # Get smoothness indicator based on mass component
    if indicator is None:
        indicator = smoothness_indicator(discr, fluid_state.mass_density,
                                         kappa=kappa, s0=s0)

    if grad_cv is None:
        from mirgecom.navierstokes import grad_cv_operator
        grad_cv = grad_cv_operator(discr, gas_model, boundaries, fluid_state,
                                   time=time, quadrature_tag=quadrature_tag,
                                   operator_states_quad=operator_states_quad)

    # Compute R = alpha*grad(Q)
    r = -alpha * indicator * grad_cv

    def central_flux_div(utpair):
        dd = utpair.dd
        normal = actx.thaw(discr.normal(dd))
        return op.project(discr, dd, dd.with_dtag("all_faces"),
                          # This uses a central vector flux along nhat:
                          # flux = 1/2 * (grad(Q)- + grad(Q)+) .dot. nhat
                          divergence_numerical_flux(utpair.int, utpair.ext)@normal)

    # Total flux of grad(Q) across element boundaries
    r_bnd = (
        # Rank-local and cross-rank (across parallel partitions) contributions
        + sum(central_flux_div(interp_to_surf_quad(tpair=tpair))
              for tpair in interior_trace_pairs(discr, r, tag=_AVRTag))

        # Contributions from boundary fluxes
        + sum(boundaries[btag].av_flux(
            discr,
            btag=as_dofdesc(btag).with_discr_tag(quadrature_tag),
            diffusion=r) for btag in boundaries)
    )

    # Return the AV RHS term
    return -div_operator(discr, dd_vol, dd_faces, interp_to_vol_quad(r), r_bnd)


def smoothness_indicator(discr, u, kappa=1.0, s0=-6.0):
    r"""Calculate the smoothness indicator.

    Parameters
    ----------
    u: meshmode.dof_array.DOFArray
        The field that is used to calculate the smoothness indicator.

    kappa
        An optional argument that controls the width of the transition from 0 to 1.

    s0
        An optional argument that sets the smoothness level to limit
        on. Values in the range $(-\infty,0]$ are allowed, where $-\infty$ results in
        all cells being tagged and 0 results in none.

    Returns
    -------
    meshmode.dof_array.DOFArray
        The elementwise constant values between 0 and 1 which indicate the smoothness
        of a given element.
    """
    if not isinstance(u, DOFArray):
        raise ValueError("u argument must be a DOFArray.")

    actx = u.array_context

    @memoize_in(actx, (smoothness_indicator, "smooth_comp_knl"))
    def indicator_prg():
        """Compute the smoothness indicator for all elements."""
        from arraycontext import make_loopy_program
        from meshmode.transform_metadata import (ConcurrentElementInameTag,
                                                 ConcurrentDOFInameTag)
        import loopy as lp
        t_unit = make_loopy_program([
            "{[iel]: 0 <= iel < nelements}",
            "{[idof]: 0 <= idof < ndiscr_nodes_in}",
            "{[jdof]: 0 <= jdof < ndiscr_nodes_in}",
            "{[kdof]: 0 <= kdof < ndiscr_nodes_in}"
            ],
            """
                result[iel,idof] = sum(kdof, vec[iel, kdof]               \
                                             * vec[iel, kdof]             \
                                             * modes_active_flag[kdof]) / \
                                   sum(jdof, vec[iel, jdof]               \
                                             * vec[iel, jdof]             \
                                             + 1.0e-12 / ndiscr_nodes_in)
            """,
            name="smooth_comp",
        )
        return lp.tag_inames(t_unit, {"iel": ConcurrentElementInameTag(),
                                      "idof": ConcurrentDOFInameTag()})

    @keyed_memoize_in(actx, (smoothness_indicator,
                             "highest_mode"),
                      lambda grp: grp.discretization_key())
    def highest_mode(grp):
        return actx.from_numpy(
            np.asarray([1 if sum(mode_id) == grp.order
                        else 0
                        for mode_id in grp.mode_ids()])
        )

    # Convert to modal solution representation
    modal_map = discr.connection_from_dds(DD_VOLUME, DD_VOLUME_MODAL)
    uhat = modal_map(u)

    # Compute smoothness indicator value
    if actx.supports_nonscalar_broadcasting:
        from meshmode.transform_metadata import DiscretizationDOFAxisTag
        indicator = DOFArray(
            actx,
            data=tuple(
                actx.tag_axis(
                    1,
                    DiscretizationDOFAxisTag(),
                    actx.np.broadcast_to(
                        ((actx.einsum("ek,k->e",
                                      uhat[grp.index]**2,
                                      highest_mode(grp))
                          / (actx.einsum("ej->e",
                                         (uhat[grp.index]**2+(1e-12/grp.nunit_dofs))
                                         )))
                         .reshape(-1, 1)),
                        uhat[grp.index].shape))
                for grp in discr.discr_from_dd("vol").groups
            )
        )
    else:
        indicator = DOFArray(
            actx,
            data=tuple(
                actx.call_loopy(
                    indicator_prg(),
                    vec=uhat[grp.index],
                    modes_active_flag=highest_mode(grp)
                )["result"]
                for grp in discr.discr_from_dd("vol").groups
            )
        )

    indicator = actx.np.log10(indicator + 1.0e-12)

    # Compute artificial viscosity percentage based on indicator and set parameters
    yesnol = actx.np.greater(indicator, (s0 - kappa))
    yesnou = actx.np.greater(indicator, (s0 + kappa))
    saintly_value = 1.0
    sin_indicator = actx.np.where(
        yesnol,
        0.5 * (1.0 + actx.np.sin(np.pi * (indicator - s0) / (2.0 * kappa))),
        0.0 * indicator,
    )
    indicator = actx.np.where(yesnou, saintly_value, sin_indicator)

    return indicator
