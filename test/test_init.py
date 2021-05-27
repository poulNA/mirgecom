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

import logging
import numpy as np
import numpy.linalg as la  # noqa
from pytools.obj_array import make_obj_array
import pytest

from arraycontext import (  # noqa
    thaw,
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests
)

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from mirgecom.initializers import Vortex2D
from mirgecom.initializers import Lump
from mirgecom.initializers import MulticomponentLump

from mirgecom.fluid import get_num_species

from mirgecom.initializers import SodShock1D
from mirgecom.eos import IdealSingleGas

from grudge.discretization import DiscretizationCollection
import grudge.op as op

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("nspecies", [0, 10])
def test_uniform_init(actx_factory, dim, nspecies):
    """Test the uniform flow initializer.

    Simple test to check that uniform initializer
    creates the expected solution field.
    """
    actx = actx_factory()
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=[(0.0,), (-5.0,)], b=[(10.0,), (5.0,)], nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    logger.info(f"Number of elements: {mesh.nelements}")

    dcoll = DiscretizationCollection(actx, mesh, order=order)
    nodes = thaw(dcoll.nodes(), actx)

    velocity = np.ones(shape=(dim,))
    from mirgecom.initializers import Uniform
    mass_fracs = np.array([float(ispec+1) for ispec in range(nspecies)])

    initializer = Uniform(dim=dim, mass_fracs=mass_fracs, velocity=velocity)
    cv = initializer(nodes)

    def inf_norm(data):
        if len(data) > 0:
            return op.norm(dcoll, data, np.inf)
        else:
            return 0.0

    p = 0.4 * (cv.energy - 0.5 * np.dot(cv.momentum, cv.momentum) / cv.mass)
    exp_p = 1.0
    perrmax = inf_norm(p - exp_p)

    exp_mass = 1.0
    merrmax = inf_norm(cv.mass - exp_mass)

    exp_energy = 2.5 + .5 * dim
    eerrmax = inf_norm(cv.energy - exp_energy)

    exp_species_mass = exp_mass * mass_fracs
    mferrmax = inf_norm(cv.species_mass - exp_species_mass)

    assert perrmax < 1e-15
    assert merrmax < 1e-15
    assert eerrmax < 1e-15
    assert mferrmax < 1e-15


def test_lump_init(actx_factory):
    """
    Simple test to check that Lump initializer
    creates the expected solution field.
    """
    actx = actx_factory()
    dim = 2
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=[(0.0,), (-5.0,)], b=[(10.0,), (5.0,)], nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    logger.info(f"Number of elements: {mesh.nelements}")

    dcoll = DiscretizationCollection(actx, mesh, order=order)
    nodes = thaw(dcoll.nodes(), actx)

    # Init soln with Vortex
    center = np.zeros(shape=(dim,))
    velocity = np.zeros(shape=(dim,))
    center[0] = 5
    velocity[0] = 1
    lump = Lump(dim=dim, center=center, velocity=velocity)
    cv = lump(nodes)

    p = 0.4 * (cv.energy - 0.5 * np.dot(cv.momentum, cv.momentum) / cv.mass)
    exp_p = 1.0
    errmax = op.norm(dcoll, p - exp_p, np.inf)

    logger.info(f"lump_soln = {cv}")
    logger.info(f"pressure = {p}")

    assert errmax < 1e-15


def test_vortex_init(actx_factory):
    """
    Simple test to check that Vortex2D initializer
    creates the expected solution field.
    """
    actx = actx_factory()
    dim = 2
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=[(0.0,), (-5.0,)], b=[(10.0,), (5.0,)], nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    logger.info(f"Number of elements: {mesh.nelements}")

    dcoll = DiscretizationCollection(actx, mesh, order=order)
    nodes = thaw(dcoll.nodes(), actx)

    # Init soln with Vortex
    vortex = Vortex2D()
    cv = vortex(nodes)
    gamma = 1.4
    p = 0.4 * (cv.energy - 0.5 * np.dot(cv.momentum, cv.momentum) / cv.mass)
    exp_p = cv.mass ** gamma
    errmax = op.norm(dcoll, p - exp_p, np.inf)

    logger.info(f"vortex_soln = {cv}")
    logger.info(f"pressure = {p}")

    assert errmax < 1e-15


def test_shock_init(actx_factory):
    """
    Simple test to check that Shock1D initializer
    creates the expected solution field.
    """
    actx = actx_factory()

    nel_1d = 10
    dim = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=[(0.0,), (1.0,)], b=[(-0.5,), (0.5,)], nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    print(f"Number of elements: {mesh.nelements}")

    dcoll = DiscretizationCollection(actx, mesh, order=order)
    nodes = thaw(dcoll.nodes(), actx)

    initr = SodShock1D()
    cv = initr(t=0.0, x_vec=nodes)
    print("Sod Soln:", cv)
    xpl = 1.0
    xpr = 0.1
    tol = 1e-15
    nodes_x = nodes[0]
    eos = IdealSingleGas()
    p = eos.pressure(cv)

    assert op.norm(dcoll, actx.np.where(nodes_x < 0.5, p-xpl, p-xpr), np.inf) < tol


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_uniform(actx_factory, dim):
    """
    Simple test to check that Uniform initializer
    creates the expected solution field.
    """
    actx = actx_factory()

    nel_1d = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 1
    print(f"Number of elements: {mesh.nelements}")

    dcoll = DiscretizationCollection(actx, mesh, order=order)
    nodes = thaw(dcoll.nodes(), actx)
    print(f"DIM = {dim}, {len(nodes)}")
    print(f"Nodes={nodes}")

    from mirgecom.initializers import Uniform
    initr = Uniform(dim=dim)
    cv = initr(t=0.0, x_vec=nodes)
    tol = 1e-15

    assert op.norm(dcoll, cv.mass - 1.0, np.inf) < tol
    assert op.norm(dcoll, cv.energy - 2.5, np.inf) < tol

    print(f"Uniform Soln:{cv}")
    eos = IdealSingleGas()
    p = eos.pressure(cv)
    print(f"Press:{p}")

    assert op.norm(dcoll, p - 1.0, np.inf) < tol


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_pulse(actx_factory, dim):
    """
    Test of Gaussian pulse generator.
    If it looks, walks, and quacks like a Gaussian, then ...
    """
    actx = actx_factory()

    nel_1d = 10

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 1
    print(f"Number of elements: {mesh.nelements}")

    dcoll = DiscretizationCollection(actx, mesh, order=order)
    nodes = thaw(dcoll.nodes(), actx)
    print(f"DIM = {dim}, {len(nodes)}")
    print(f"Nodes={nodes}")

    tol = 1e-15
    from mirgecom.initializers import make_pulse
    amp = 1.0
    w = .1
    rms2 = w * w
    r0 = np.zeros(dim)
    r2 = np.dot(nodes, nodes) / rms2
    pulse = make_pulse(amp=amp, r0=r0, w=w, r=nodes)
    print(f"Pulse = {pulse}")

    # does it return the expected exponential?
    pulse_check = actx.np.exp(-.5 * r2)
    print(f"exact: {pulse_check}")
    pulse_resid = pulse - pulse_check
    print(f"pulse residual: {pulse_resid}")
    assert(op.norm(dcoll, pulse_resid, np.inf) < tol)

    # proper scaling with amplitude?
    amp = 2.0
    pulse = 0
    pulse = make_pulse(amp=amp, r0=r0, w=w, r=nodes)
    pulse_resid = pulse - (pulse_check + pulse_check)
    assert(op.norm(dcoll, pulse_resid, np.inf) < tol)

    # proper scaling with r?
    amp = 1.0
    rcheck = np.sqrt(2.0) * nodes
    pulse = make_pulse(amp=amp, r0=r0, w=w, r=rcheck)
    assert(op.norm(dcoll, pulse - (pulse_check * pulse_check), np.inf) < tol)

    # proper scaling with w?
    w = w / np.sqrt(2.0)
    pulse = make_pulse(amp=amp, r0=r0, w=w, r=nodes)
    assert(op.norm(dcoll, pulse - (pulse_check * pulse_check), np.inf) < tol)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_multilump(actx_factory, dim):
    """Test the multi-component lump initializer."""
    actx = actx_factory()

    nel_1d = 4
    nspecies = 10

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-1.0,) * dim, b=(1.0,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    logger.info(f"Number of elements: {mesh.nelements}")

    dcoll = DiscretizationCollection(actx, mesh, order=order)
    nodes = thaw(dcoll.nodes(), actx)

    rho0 = 1.5
    centers = make_obj_array([np.zeros(shape=(dim,)) for i in range(nspecies)])
    spec_y0s = np.ones(shape=(nspecies,))
    spec_amplitudes = np.ones(shape=(nspecies,))

    for i in range(nspecies):
        centers[i][0] = (.1 * i)
        spec_y0s[i] += (.1 * i)
        spec_amplitudes[i] += (.1 * i)

    velocity = np.zeros(shape=(dim,))
    velocity[0] = 1

    lump = MulticomponentLump(dim=dim, nspecies=nspecies, rho0=rho0,
                              spec_centers=centers, velocity=velocity,
                              spec_y0s=spec_y0s, spec_amplitudes=spec_amplitudes)

    cv = lump(nodes)
    numcvspec = get_num_species(dim, cv.join())
    print(f"get_num_species = {numcvspec}")

    assert get_num_species(dim, cv.join()) == nspecies
    assert op.norm(dcoll, cv.mass - rho0) == 0.0

    p = 0.4 * (cv.energy - 0.5 * np.dot(cv.momentum, cv.momentum) / cv.mass)
    exp_p = 1.0
    errmax = op.norm(dcoll, p - exp_p, np.inf)
    assert len(cv.species_mass) == nspecies
    species_mass = cv.species_mass

    spec_r = make_obj_array([nodes - centers[i] for i in range(nspecies)])
    r2 = make_obj_array([np.dot(spec_r[i], spec_r[i]) for i in range(nspecies)])
    expfactor = make_obj_array([spec_amplitudes[i] * actx.np.exp(- r2[i])
                                for i in range(nspecies)])
    exp_mass = make_obj_array([rho0 * (spec_y0s[i] + expfactor[i])
                               for i in range(nspecies)])
    mass_resid = species_mass - exp_mass

    print(f"exp_mass = {exp_mass}")
    print(f"mass_resid = {mass_resid}")

    assert op.norm(dcoll, mass_resid, np.inf) == 0.0

    logger.info(f"lump_soln = {cv}")
    logger.info(f"pressure = {p}")

    assert errmax < 1e-15
