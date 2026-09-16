#include <pybind11/pybind11.h>

#include "pyrxmesh/plugin_api.h"

namespace py = pybind11;
using namespace rxmesh;

void compute_normals(py::object mesh_obj,
                          py::object coords_obj,
                          py::object normals_obj)
{
    auto coords = pyrxmesh::vertex_attribute<float>(coords_obj);
    auto normals    = pyrxmesh::vertex_attribute<float>(normals_obj);

    pyrxmesh::for_each<Op::FV, 256>(
        mesh_obj,
        [coords, normals] __device__(const FaceHandle& eh,
                                 const VertexIterator& fv) mutable {
            const Eigen::Vector3f c0 = coords.to_eigen<3>(fv[0]);
            const Eigen::Vector3f c1 = coords.to_eigen<3>(fv[1]);
            const Eigen::Vector3f c2 = coords.to_eigen<3>(fv[2]);
            
            Eigen::Vector3f n = (c1 - c0).cross(c2 - c0);

            Eigen::Vector3f l((c0 - c1).norm(), (c1 - c2).norm(), (c2 - c0).norm());

            for(uint32_t v = 0; v < 3; ++v) {
                for(uint32_t i = 0; i < 3; ++i) {
                    atomicAdd(
                        &normals(fv[v], i), 
                        n[i] / (l[v] + l[(v+2) % 3])
                    );
                }
            }
        });
}

PYBIND11_MODULE(_vertex_normal, m)
{
    pyrxmesh::require_compatible_runtime(m);
    m.def("compute_normals",
          &compute_normals,
          py::arg("mesh"),
          py::arg("coords"),
          py::arg("normals"));
}
