"""Provide some wrappers for centralizing grudge discretization creation.

Discretization creation
-----------------------

.. autofunction:: create_discretization_collection
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
import logging

logger = logging.getLogger(__name__)


# Centralize the discretization collection creation routine so that
# we can replace it more easily when we refactor the drivers and
# examples to use discretization collections, and change it centrally
# when we want to change it.
def create_discretization_collection(actx, mesh, order, *, mpi_communicator=None,
                                     quadrature_order=-1):
    """Create and return a grudge DG discretization collection."""
    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from grudge.discretization import DiscretizationCollection
    from meshmode.discretization.poly_element import (
        QuadratureSimplexGroupFactory,
        PolynomialRecursiveNodesGroupFactory
    )

    if quadrature_order < 0:
        quadrature_order = 2*order+1

    return DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: PolynomialRecursiveNodesGroupFactory(order=order,
                                                                 family="lgl"),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(quadrature_order),
        },
        mpi_communicator=mpi_communicator
    )
