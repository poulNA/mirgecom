"""Demonstrate wave MPI example."""

__copyright__ = "Copyright (C) 2020 University of Illinois Board of Trustees"

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
import pyopencl as cl

from pytools.obj_array import flat_obj_array

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.shortcuts import make_visualizer
import grudge.op as op

from mirgecom.discretization import create_discretization_collection
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import rk4_step
from mirgecom.wave import wave_operator
from mirgecom.utils import force_evaluation

import pyopencl.tools as cl_tools

from logpyle import IntervalTimer, set_dt

from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_cl_device_info,
                                         logmgr_add_device_memory_usage)


def bump(actx, nodes, t=0):
    """Create a bump."""
    dim = len(nodes)
    source_center = np.array([0.2, 0.35, 0.1])[:dim]
    source_width = 0.05
    source_omega = 3

    center_dist = flat_obj_array([
        nodes[i] - source_center[i]
        for i in range(dim)
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


@mpi_entry_point
def main(actx_class, snapshot_pattern="wave-mpi-{step:04d}-{rank:04d}.pkl",
         restart_step=None, use_profiling=False, use_logmgr=False, lazy=False):
    """Drive the example."""
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr,
        filename="wave-mpi.sqlite", mode="wu", mpi_comm=comm)
    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    if restart_step is None:

        from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
        mesh_dist = MPIMeshDistributor(comm)

        dim = 2
        nel_1d = 16

        if mesh_dist.is_mananger_rank():
            from meshmode.mesh.generation import generate_regular_rect_mesh
            mesh = generate_regular_rect_mesh(
                a=(-0.5,)*dim, b=(0.5,)*dim,
                nelements_per_axis=(nel_1d,)*dim)

            print("%d elements" % mesh.nelements)
            part_per_element = get_partition_by_pymetis(mesh, num_parts)
            local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

            del mesh

        else:
            local_mesh = mesh_dist.receive_mesh_part()

        fields = None

    else:
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(
            actx, snapshot_pattern.format(step=restart_step, rank=rank)
        )
        local_mesh = restart_data["local_mesh"]
        nel_1d = restart_data["nel_1d"]
        assert comm.Get_size() == restart_data["num_parts"]

    order = 3

    discr = create_discretization_collection(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = actx.thaw(discr.nodes())

    current_cfl = 0.485
    wave_speed = 1.0
    from grudge.dt_utils import characteristic_lengthscales
    nodal_dt = characteristic_lengthscales(actx, discr) / wave_speed

    dt = actx.to_numpy(current_cfl * op.nodal_min(discr, "vol", nodal_dt))[()]

    t_final = 1

    if restart_step is None:
        t = 0
        istep = 0

        fields = flat_obj_array(
            bump(actx, nodes),
            [discr.zeros(actx) for i in range(dim)]
            )

    else:
        t = restart_data["t"]
        istep = restart_step
        assert istep == restart_step
        restart_fields = restart_data["fields"]
        old_order = restart_data["order"]
        if old_order != order:
            old_discr = create_discretization_collection(
                actx, local_mesh, order=old_order, mpi_communicator=comm
            )
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(actx, discr.discr_from_dd("vol"),
                                                   old_discr.discr_from_dd("vol"))
            fields = connection(restart_fields)
        else:
            fields = restart_fields

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches(["step.max", "t_step.max", "t_log.max"])

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    vis = make_visualizer(discr)

    def rhs(t, w):
        return wave_operator(discr, c=wave_speed, w=w)
    fields = force_evaluation(actx, fields)
    compiled_rhs = actx.compile(rhs)

    while t < t_final:
        if logmgr:
            logmgr.tick_before()

        # restart must happen at beginning of step
        if istep % 100 == 0 and (
                # Do not overwrite the restart file that we just read.
                istep != restart_step):
            from mirgecom.restart import write_restart_file
            write_restart_file(
                actx, restart_data={
                    "local_mesh": local_mesh,
                    "order": order,
                    "fields": fields,
                    "t": t,
                    "step": istep,
                    "nel_1d": nel_1d,
                    "num_parts": num_parts},
                filename=snapshot_pattern.format(step=istep, rank=rank),
                comm=comm
            )

        if istep % 10 == 0:
            print(istep, t, actx.to_numpy(op.norm(discr, fields[0], 2)))
            vis.write_parallel_vtk_file(
                comm,
                "fld-wave-mpi-%03d-%04d.vtu" % (rank, istep),
                [
                    ("u", fields[0]),
                    ("v", fields[1:]),
                ], overwrite=True
            )

        fields = rk4_step(fields, t, dt, compiled_rhs)
        fields = force_evaluation(actx, fields)

        t += dt
        istep += 1

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()

    final_soln = actx.to_numpy(op.norm(discr, fields[0], 2))
    assert np.abs(final_soln - 0.04409852463947439) < 1e-14


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    # Turn off profiling to not overwhelm CI
    use_profiling = False
    use_logging = True

    import argparse
    parser = argparse.ArgumentParser(description="Wave (MPI version)")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    args = parser.parse_args()
    lazy = args.lazy

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    main(actx_class, use_profiling=use_profiling, use_logmgr=use_logging, lazy=lazy)

# vim: foldmethod=marker
