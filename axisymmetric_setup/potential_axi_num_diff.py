"""Test axisymmetry."""

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
import yaml
import logging
import sys
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial

from arraycontext import thaw, freeze

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.dof_desc import DTAG_BOUNDARY
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    generate_and_distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    global_reduce,
    force_evaluation
)
from mirgecom.restart import (
    write_restart_file
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
# from mirgecom.checkstate import compare_states
from mirgecom.integrators import (
    rk4_step,
    #rk2_step,
    euler_step
)
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    OutflowBoundary,
    SymmetryBoundary
)
from mirgecom.fluid import make_conserved#, species_mass_fraction_gradient
from mirgecom.transport import SimpleTransport
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import GasModel, make_fluid_state

from mirgecom.viscous import (
    get_viscous_timestep,
    get_viscous_cfl,
    #get_healthy_viscous_timestep
)

from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info, logmgr_set_time, LogUserQuantity,
    set_sim_state
)

#from mirgecom.limiter import (
#    limiter_liu_osher,
#    neighbor_list,
#    positivity_preserving_limiter
#)

from pytools.obj_array import make_obj_array

####################################################################

class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


h1 = logging.StreamHandler(sys.stdout)
f1 = SingleLevelFilter(logging.INFO, False)
h1.addFilter(f1)
root_logger = logging.getLogger()
root_logger.addHandler(h1)
h2 = logging.StreamHandler(sys.stderr)
f2 = SingleLevelFilter(logging.INFO, True)
h2.addFilter(f2)
root_logger.addHandler(h2)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass
    
    
class PotentialFlow:

    def __init__(
            self, *, dim=3, temperature, pressure=None, mass=None, amplitude):
                   
        self._dim = dim
        self._temp = temperature
        self._pres = pressure
        self._mass = mass
        self._amp = amplitude

    def __call__(self, x_vec, eos, *, time=0.0):

        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        x = x_vec[0]
        actx = x.array_context

        gamma = eos.gamma()
        R = eos.gas_const()

#        u_x = +1.0*self._amp*x_vec[0]
#        u_y = -2.0*self._amp*x_vec[1]
#        momentum = make_obj_array([u_x, u_y])

#        temperature = self._temp - ((gamma - 1.0)/(2.0*gamma*R))*(u_x**2 + u_y**2)
#        
#        const = ((self._pres**(gamma-1.0))/(self._temp**gamma))
#        pressure = ( (temperature**gamma)*const )**(1.0/(gamma-1.0))

#        mass = pressure/(temperature*R)

        ru_x = +1.0*self._amp*x_vec[0]
        ru_y = -2.0*self._amp*x_vec[1]
        momentum = make_obj_array([ru_x, ru_y])     

        const = (self._mass**(gamma-1.0))/self._temp
        alfa = (const)**(-2.0/(gamma-1.0))
        beta = (gamma-1.0)/(2.0*gamma*R)*(np.dot(momentum, momentum))
        
        temperature = 0.0*x + self._temp
        for i in range(0,5):         
          function = temperature + alfa*beta*temperature**(-2.0/(gamma-1.0)) - self._temp
          derivative = 1.0 + alfa*beta*(-2.0/(gamma-1.0)*temperature**(-(gamma+1.0)/(gamma-1.0)))         
          temperature = temperature - function/derivative
          
        if self._mass is None:
            const = ((self._pres**(gamma-1.0))/(self._temp**gamma))
            pressure = ( (temperature**gamma)*const )**(1.0/(gamma-1.0))
            mass = pressure/(temperature*R)
        if self._pres is None:
            const = self._temp/self._mass**(gamma-1.0)
            mass = (temperature/const)**(1.0/(gamma-1.0))
            pressure = mass*R*temperature

        velocity = momentum/mass

        internal_energy = pressure/(gamma - 1.0)
        kinetic_energy = 0.5 * np.dot(velocity, velocity) * mass
        energy = (internal_energy + kinetic_energy)

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                                  momentum=momentum) 


def get_mesh(dim, read_mesh=True):
    """Get the mesh."""
    from meshmode.mesh.io import read_gmsh
    mesh_filename = "mesh_half.msh"
    #mesh = read_gmsh(mesh_filename, force_ambient_dim=dim)
    mesh = partial(read_gmsh, filename=mesh_filename, force_ambient_dim=dim)
    #mesh = read_gmsh(mesh_filename)

    return mesh


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         rst_filename=None):
         
    """Drive the 1D Flame example."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)

    cl_ctx = ctx_factory()

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    # ~~~~~~~~~~~~~~~~~~
    restart_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    snapshot_pattern = restart_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 5
    nrestart = 5000
    nhealth = 100
    nstatus = 100
    do_checkpoint = True

    # default timestepping control
    integrator = "euler"
    current_dt = 1.25e-6
    t_final = 1.25e-5

    niter = int(np.rint(t_final/current_dt))

    # discretization and model control
    order = 2

    use_limiter = False
    use_overintegration = False

    storeCFL = True
    constant_cfl = False
    current_cfl = 0.25              

    ##################################################

    # param sanity check
    allowed_integrators = ["rk2", "rk4", "euler", "ssprk43"]
    if(integrator not in allowed_integrators):
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    # if integrator == "rk2":
    #    timestepper = rk2_step
    if integrator == "rk4":
        timestepper = rk4_step
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "ssprk43":
        timestepper = ssprk43_step

    ##################################################  
    dim = 2
    current_t = 0
    current_step = 0

    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        if (constant_cfl is False):
            print(f"\tcurrent_dt = {current_dt}")
            print(f"\tt_final = {t_final}")
        else:
            print(f"\tconstant_cfl = {constant_cfl}")
            print(f"\tcurrent_cfl = {current_cfl}")
        print(f"\tniter = {niter}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")
        print(f"\tuse_limiter = {use_limiter}")
        print(f"\tuse_overintegration = {use_overintegration}")      

    ##################################################

    kappa = 0.0  # Pr = mu*cp/kappa = 0.75
    mu = 0.0
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa)
    eos = IdealSingleGas()

    gas_model = GasModel(eos=eos, transport=transport_model)                 

    def get_fluid_state(cv):
        return make_fluid_state(cv=cv, gas_model=gas_model)

    construct_fluid_state = actx.compile(get_fluid_state)

    ##################################################

    amplitude = 91.23
    flow_init = PotentialFlow(dim=dim, temperature=300.0,
                                    mass=1.0,
                                    amplitude=amplitude)
    ref_state = PotentialFlow(dim=dim, temperature=300.0,
                                    mass=1.0,
                                    amplitude=amplitude)

    ##################################################

    restart_step = None
    if restart_file is None:
    
        char_len = 0.025
        box_ll = (0.0, 0.0)
        box_ur = (+0.5, 0.5)
        num_elements = (int((box_ur[0]-box_ll[0])/char_len)+1,
                            int((box_ur[1]-box_ll[1])/char_len)+1)

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh,
                                a=box_ll,
                                b=box_ur,
                                n=num_elements,
                                mesh_type="X",
                                boundary_tag_to_face={
                                    "wall": ["-y"],
                                    "flow": ["+x", "+y"],
                                    "symmetry": ["-x"]})

        local_mesh, global_nelements = (
            generate_and_distribute_mesh(comm, generate_mesh))
        local_nelements = local_mesh.nelements

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert comm.Get_size() == restart_data["num_parts"]

############################################################

    if rank == 0:
        print("Making discretization")
        logging.info("Making discretization")

    from grudge.dof_desc import DISCR_TAG_QUAD
    from mirgecom.discretization import create_discretization_collection
    discr = create_discretization_collection(actx, local_mesh, order, comm)
    nodes = thaw(discr.nodes(), actx)

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    ##################################################

    current_cv = flow_init(x_vec=nodes, eos=eos, time=0.)
    current_state = construct_fluid_state(cv=current_cv)
    current_state = force_evaluation(actx, current_state)

    ref_cv = ref_state(x_vec=nodes, eos=eos, time=0.)
    flow_btag = DTAG_BOUNDARY("flow")
    flow_bnd_discr = discr.discr_from_dd(flow_btag)
    flow_nodes = thaw(flow_bnd_discr.nodes(), actx)
    flow_cv = ref_state(x_vec=flow_nodes, eos=eos, time=0.)
    flow_state = construct_fluid_state(cv=flow_cv)
    flow_state = force_evaluation(actx, flow_state)

    def _flow_boundary_state_func(**kwargs):
        return flow_state

    # def _boundary_state_func(discr, btag, gas_model, state_minus, init_func,
    #                         **kwargs):
    #    actx = state_minus.array_context
    #    bnd_discr = discr.discr_from_dd(btag)
    #    nodes = thaw(bnd_discr.nodes(), actx)
    #    return make_fluid_state(init_func(nodes=nodes, eos=gas_model.eos,
    #                                      cv=state_minus.cv, **kwargs),
    #                            gas_model=gas_model)

    # def _inflow_boundary_state(discr, btag, gas_model, state_minus, **kwargs):
    #    return _boundary_state_func(discr, btag, gas_model, state_minus,
    #                                _inflow_func, **kwargs)
   
    wall_symmetry = SymmetryBoundary()
    flow_boundary = \
        PrescribedFluidBoundary(boundary_state_func=_flow_boundary_state_func)
    # outflow_boundary = \
    #    PrescribedFluidBoundary(boundary_state_func=_inflow_boundary_state)

    # DTAG_BOUNDARY("inflow"): inflow_boundary,
    # DTAG_BOUNDARY("outflow"): outflow_boundary,
    boundaries = {DTAG_BOUNDARY("symmetry"): wall_symmetry,
                  DTAG_BOUNDARY("flow"): flow_boundary,
                  DTAG_BOUNDARY("wall"): wall_symmetry}

    
    ##################################################

    vis_timer = None
    log_cfl = LogUserQuantity(name="cfl", value=current_cfl)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)
        # logmgr_add_package_versions(logmgr)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "------- step walltime: {value:6g} s\n")])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    visualizer = make_visualizer(discr)

    initname = "potential"
    eosname = gas_model.eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

#########################################################################

    def get_dt(state):
        return make_obj_array([get_viscous_timestep(discr, state=state)])
    compute_dt = actx.compile(get_dt)

    def get_cfl(state, dt):
        return make_obj_array([get_viscous_cfl(discr, dt, state=state)])
    compute_cfl = actx.compile(get_cfl)

    def vol_min_loc(x):
        from grudge.op import nodal_min_loc
        return actx.to_numpy(nodal_min_loc(discr, "vol", x))[()]

    def vol_max_loc(x):
        from grudge.op import nodal_max_loc
        return actx.to_numpy(nodal_max_loc(discr, "vol", x))[()]

    def vol_min(x):
        from grudge.op import nodal_min
        return actx.to_numpy(nodal_min(discr, "vol", x))[()]

    def vol_max(x):
        from grudge.op import nodal_max
        return actx.to_numpy(nodal_max(discr, "vol", x))[()]

#########################################################################
    from grudge.dof_desc import DOFDesc, as_dofdesc, DISCR_TAG_BASE            
    from mirgecom.flux import num_flux_central
    from arraycontext import outer
    from grudge.trace_pair import interior_trace_pairs
    from grudge.trace_pair import interior_trace_pair
    from mirgecom.boundary import DummyBoundary
    from mirgecom.operators import grad_operator
    from grudge.trace_pair import TracePair #XXX
    from arraycontext import outer

    def _elbnd_flux(discr, compute_interior_flux, compute_boundary_flux,
                int_tpairs, boundaries):
        return (compute_interior_flux(int_tpairs)
            + sum(compute_boundary_flux(btag) for btag in boundaries))

    def _elbnd_flux2(discr, compute_interior_flux, compute_boundary_flux,
                int_tpairs, boundaries):
        return (sum(compute_interior_flux(int_tpair)
                    for int_tpair in int_tpairs)
            + sum(compute_boundary_flux(btag) for btag in boundaries))

    def central_flux_interior(actx, discr, int_tpair):
        """Compute a central flux for interior faces."""       
        normal = thaw(discr.normal(int_tpair.dd), actx)
        flux_weak = outer(num_flux_central(int_tpair.int, int_tpair.ext), normal)
        dd_all_faces = int_tpair.dd.with_dtag("all_faces")
        return discr.project(int_tpair.dd, dd_all_faces, flux_weak)

    def central_flux_boundary(actx, discr, field, btag):
        """Compute a central flux for boundary faces."""
        dd_base_vol = DOFDesc("vol")
        # btag = as_dofdesc(btag)
        bnd_solution_quad = discr.project(dd_base_vol,      
              as_dofdesc(btag).with_discr_tag(quadrature_tag), field)       
        bnd_nhat = thaw(discr.normal(btag), actx)
        bnd_tpair = TracePair(btag, interior=bnd_solution_quad, # XXX
                              exterior=bnd_solution_quad) # XXX
        flux_weak = outer(num_flux_central(bnd_tpair.int, bnd_tpair.ext), bnd_nhat)
        # flux_weak = outer(bnd_solution_quad, bnd_nhat)
        dd_all_faces = bnd_tpair.dd.with_dtag("all_faces")
        return discr.project(bnd_tpair.dd, dd_all_faces, flux_weak)

    field_bounds = {DTAG_BOUNDARY("symmetry"): DummyBoundary(),
                    DTAG_BOUNDARY("flow"): DummyBoundary(),
                    DTAG_BOUNDARY("wall"): DummyBoundary()}

    def second_derivative(actx, discr, field):

        int_flux = partial(central_flux_interior, actx, discr)
        bnd_flux = partial(central_flux_boundary, actx, discr, field)

        int_tpairs = interior_trace_pair(discr, field)
        flux_bnd = _elbnd_flux(discr, int_flux, bnd_flux, int_tpairs, field_bounds)

        dd_vol = as_dofdesc("vol")
        dd_faces = as_dofdesc("all_faces")

        return grad_operator(discr, dd_vol, dd_faces, field, flux_bnd)

#########################################################################
    off_axis_x = 1e-7
    nodes_are_off_axis = actx.np.greater(nodes[0], off_axis_x)  # noqa

    def axisymmetry_source_terms(actx, discr, cv, dv, mu, beta, grad_cv):

        nodes = thaw(discr.nodes(), actx)

        u = cv.velocity[0]
        v = cv.velocity[1]

        qr = u*0.0 # FIXME

        grad_v = velocity_gradient(cv, grad_cv)
        dudr = grad_v[0][0]
        dudy = grad_v[0][1]
        dvdr = grad_v[1][0]
        dvdy = grad_v[1][1]

        dbetadr = 0.0*u  # FIXME
        dbetady = 0.0*u  # FIXME

        tau_rr = 2.0*mu*dudr + beta*(dudr + dvdy)
        # tau_ry = tau_rr   # this works!
        tau_ry = mu*(dvdr + dudy)   # this is what we want (doesn't work)
        tau_yy = 2.0*mu*dvdy + beta*(dudr + dvdy)  # noqa
        tau_tt = (beta*(dudr + dvdy)
                  + 2.0*mu*actx.np.where(nodes_are_off_axis,
                                         u/nodes[0], 0.0*u))

        source_mass_dom = -(cv.momentum[0])

        source_rhoU_dom = (-(cv.momentum[0]*u - tau_rr + tau_tt)  # noqa
                            + u*dbetadr + beta*dudr
                            + actx.np.where(nodes_are_off_axis,
                                            -beta*u/nodes[0], 0.0*u))

        source_rhoV_dom = -(cv.momentum[0]*v - tau_ry) + u*dbetady + beta*dudy  # noqa

        source_rhoE_dom = (-((cv.energy+dv.pressure)*u - u*tau_rr - v*tau_ry + qr) # noqa
                           + u**2*dbetadr + beta*2.0*u*dudr
                           + u*v*dbetady + u*beta*dvdy + v*beta*dudy)

        drhoudr = (grad_cv.momentum[0])[0]

        d2udr2 = second_derivative(actx, discr, dudr)[0]
        d2udrdy = second_derivative(actx, discr, dudy)[0]
        dtaurydr = second_derivative(actx, discr, tau_ry)[0]

        # d2udr2 = 0*u
        # d2udrdy = 0*u
        # dtaurydr = 0*u

        source_mass_sng = - drhoudr
        source_rhoU_sng = + mu*d2udr2 + 0.5*beta*d2udr2
        source_rhoV_sng = -v*drhoudr + dudr*dbetady + beta*d2udrdy + dtaurydr
        #source_rhoV_sng = -v*drhoudr + dtaurydr + dudr*dbetady + beta*d2udrdy
        source_rhoE_sng = (-(cv.energy + dv.pressure)*dudr + tau_rr*dudr
                           + tau_ry*dvdr + v*dtaurydr
                           + 2.0*beta*dudr**2 + v*dudr*dbetady + beta*dudr*dvdy
                           + v*beta*d2udrdy
                           )

        source_mass = actx.np.where(nodes_are_off_axis,
                                source_mass_dom/nodes[0], source_mass_sng)
        source_rhoU = actx.np.where(nodes_are_off_axis,
                                source_rhoU_dom/nodes[0], source_rhoU_sng)
        source_rhoV = actx.np.where(nodes_are_off_axis,
                                source_rhoV_dom/nodes[0], source_rhoV_sng )
        source_rhoE = actx.np.where(nodes_are_off_axis,
                                source_rhoE_dom/nodes[0], source_rhoE_sng )
        source_spec = None
        if cv.nspecies > 0:
            source_spec_sng = cv.species_mass  # FIXME
            source_spec = 0*source_spec_sng
            # source_spec = actx.np.where(nodes_are_off_axis,
            #                            source_spec_dom/nodes[0], source_spec_sng )

        source_rhoE = source_rhoE
        source_mom = make_obj_array([source_rhoU, source_rhoV])
        # source_mass = 0*cv.mass

        return make_conserved(dim=2, mass=source_mass, energy=source_rhoE,
                              momentum=source_mom,
                              species_mass=source_spec)

#########################################################################

    def my_write_status(step, t, dt, dv, state):
        if constant_cfl:
            cfl = current_cfl
        else:
            from mirgecom.viscous import get_viscous_cfl
            cfl_field = get_viscous_cfl(discr, dt, state=state)
            from grudge.op import nodal_max
            cfl = actx.to_numpy(nodal_max(discr, "vol", cfl_field))
        status_msg = f"Step: {step}, T: {t}, DT: {dt}, CFL: {cfl}"

        if rank == 0:
            logger.info(status_msg)
            
    def my_write_viz(step, t, cv, dv,
                     ts_field=None,
                     rhs=None, grad_cv=None, grad_t=None, grad_v=None,
                     grad_y=None,flux=None,ref_cv=None,sponge_sigma=None,
                     sources=None):
                     
        viz_fields = [("CV", cv),
                      ("DV_U", cv.momentum/cv.mass),
                      ("DV_P", dv.pressure),
                      ("DV_T", dv.temperature),
                      ]

        if rhs is not None:

            grad_v = velocity_gradient(cv,grad_cv)
            grad_P = gas_model.eos.gas_const()*(
                       dv.temperature*grad_cv.mass + cv.mass*grad_t )

            diff = ref_cv - cv 
            ref_cv_state = make_fluid_state(cv=ref_cv, gas_model=gas_model)

            dudr = grad_v[0][0]
            dudy = grad_v[0][1]
            dvdr = grad_v[1][0]
            dvdy = grad_v[1][1]
            
            #            d2rhodr2 = second_derivative(actx,discr,grad_cv.mass[0])[0]
            #            d2udr2   = second_derivative(actx,discr,dudr)[0]
            #            d2udrdy  = second_derivative(actx,discr,dudy)[0]
            #            d2vdr2   = second_derivative(actx,discr,dvdr)[0]

            viz_ext = [("ref_U", ref_cv.momentum/ref_cv.mass),
                       ("ref_mass", ref_cv_state.mass_density),
                       ("ref_P", ref_cv_state.pressure),
                       ("ref_T", ref_cv_state.temperature),
                       ("diff_U", cv.momentum/cv.mass - ref_cv.momentum/ref_cv.mass),
                       ("diff_mass", cv.mass - ref_cv_state.mass_density),
                       ("diff_P", dv.pressure - ref_cv_state.pressure),
                       ("diff_T", dv.temperature - ref_cv_state.temperature),
                       ("rhs", rhs),
                       # ("source", sources),
                       ("grad_mom_Ux", grad_cv.momentum[0][0]),
                       ("grad_mom_Uy", grad_cv.momentum[0][1]),
                       ("grad_mom_Vx", grad_cv.momentum[1][0]),
                       ("grad_mom_Vy", grad_cv.momentum[1][1]),
                       # ("grad_d2rhodr2", d2rhodr2),
                       # ("grad_d2udr2", d2udr2),
                       # ("grad_d2udrdy", d2udrdy),
                       # ("grad_d2vdr2", d2vdr2),
                       ("grad_mass", grad_cv.mass),                      
                       ("grad_P", grad_P),
                       ("grad_T", grad_t),
                       # ("dt" if constant_cfl else "cfl", ts_field)
                      ]
            viz_fields.extend(viz_ext)
                      
        from mirgecom.simutil import write_visfile
        write_visfile(discr, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)                      

    def my_write_restart(step, t, cv):
        rst_fname = snapshot_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != restart_file:
            rst_data = {
                "local_mesh": local_mesh,
                "state": cv,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, rst_data, rst_fname, comm)

#########################################################################

    def my_health_check(cv, dv):
        health_error = False
        pressure = dv.pressure
        temperature = dv.temperature

        if global_reduce(check_naninf_local(discr, "vol", pressure), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_naninf_local(discr, "vol", temperature), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        return health_error

#########################################################################

    def my_get_timestep(t, dt, state):
        t_remaining = max(0, t_final - t)
        if constant_cfl:
            cfl = current_cfl
            ts_field = cfl * compute_dt(state)[0]
            dt = global_reduce(vol_min_loc(ts_field), op="min", comm=comm)
        else:
            ts_field = compute_cfl(state, current_dt)[0]
            cfl = global_reduce(vol_max_loc(ts_field), op="max", comm=comm)

        return ts_field, cfl, min(t_remaining, dt)

#########################################################################

    from mirgecom.fluid import velocity_gradient
    # from mirgecom.simutil import compare_fluid_solutions
    # import os

    def my_pre_step(step, t, dt, state):
        fluid_state = None

        try:

            if logmgr:
                logmgr.tick_before()

            if do_checkpoint:

                do_viz = check_step(step=step, interval=nviz)
                do_restart = check_step(step=step, interval=nrestart)
                do_health = check_step(step=step, interval=nhealth)
                do_status = check_step(step=step, interval=nstatus)

                # If we plan on doing anything with the state, then
                # we need to make sure it is evaluated first.
                if any([do_viz, do_restart, do_health, do_status, constant_cfl]):
                    fluid_state = construct_fluid_state(state)
                    fluid_state = force_evaluation(actx, fluid_state)

                dt = get_sim_timestep(discr, fluid_state, t=t, dt=dt,
                                      cfl=current_cfl, t_final=t_final,
                                      constant_cfl=constant_cfl)

                if do_health:
                    dv = fluid_state.dv
                    cv = fluid_state.cv
                    health_errors = global_reduce(my_health_check(cv, dv), op="lor")
                    if health_errors:
                        if rank == 0:
                            logger.info("Fluid solution failed health check.")
                        raise MyRuntimeError("Failed simulation health check.")

                # if do_status:
                #    my_write_status(dt=dt, cfl=current_cfl, dv=dv)

                if do_restart:
                    my_write_restart(step=step, t=t, cv=cv)

                if do_viz:
                    dv = fluid_state.dv
                    cv = fluid_state.cv
                    ts_field, cfl, dt_viz = my_get_timestep(t, dt, fluid_state)
                    # log_cfl.set_quantity(cfl)

                    ns_rhs, grad_cv, grad_t = \
                        ns_operator(discr, state=fluid_state, time=t,
                                    boundaries=boundaries, gas_model=gas_model,
                                    return_gradients=True,
                                    quadrature_tag=quadrature_tag)

                    # beta = fluid_state.tv.volume_viscosity         
                    # sources = axisymmetry_source_terms(
                    #    actx, discr, fluid_state.cv, fluid_state.dv, mu, beta, grad_cv)
                    #
                    # ns_rhs = ns_rhs + sources
                    sources = None
                    my_write_viz(step=step, t=t, cv=cv, dv=dv,
                                 ts_field=ts_field, ref_cv=ref_cv, rhs=ns_rhs,
                                 grad_cv=grad_cv, grad_t=grad_t,
                                 sources=sources)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):

        fluid_state = make_fluid_state(cv=state, gas_model=gas_model)

        cv_rhs, grad_cv, grad_t = (
            ns_operator(discr, state=fluid_state, time=t,
                        boundaries=boundaries, gas_model=gas_model,
                        return_gradients=True, quadrature_tag=quadrature_tag)
            )
        
        beta = fluid_state.tv.volume_viscosity
        sources = axisymmetry_source_terms(
            actx, discr, fluid_state.cv, fluid_state.dv, mu, beta, grad_cv)

        return cv_rhs + sources

    current_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, current_cv) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      state=current_state.cv,
                      dt=current_dt, t_final=t_final, t=current_t,
                      istep=current_step)

    current_state = make_fluid_state(cv=current_cv, gas_model=gas_model)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = current_state.dv
    ts_field, cfl, dt = my_get_timestep(current_t, current_dt, current_state)
    my_write_viz(step=current_step, t=current_t,
                 cv=current_state.cv, dv=final_dv)
    my_write_restart(step=current_step, t=current_t, cv=current_state.cv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    import sys
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="MIRGE-Com 1D Flame Driver")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
    parser.add_argument("-i", "--input_file",  type=ascii,
                        dest="input_file", nargs="?", action="store",
                        help="simulation config file")
    parser.add_argument("-c", "--casename",  type=ascii,
                        dest="casename", nargs="?", action="store",
                        help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()

    # for writing output
    casename = "potential"
    if(args.casename):
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_file}")

    input_file = None
    if(args.input_file):
        input_file = (args.input_file).replace("'", "")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy, distributed=True)

    main(actx_class, use_logmgr=args.log, 
         use_profiling=args.profile,
         lazy=args.lazy, casename=casename, rst_filename=restart_file)

