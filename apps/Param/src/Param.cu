#include <pybind11/pybind11.h>

#include "pyrxmesh/plugin_api.h"

using T = float;

namespace py = pybind11;
using namespace rxmesh;

void compute_rest_shape(py::object mesh_obj,
                          py::object coords_obj,
                          py::object rest_coords_obj)
{
    auto coordinates = pyrxmesh::vertex_attribute<float>(coords_obj);
    auto rest_shape    = pyrxmesh::face_attribute<Eigen::Matrix<T, 2, 2>>(rest_coords_obj);

    pyrxmesh::for_each<Op::FV, 256>(
        mesh_obj,
        [coordinates, rest_shape] __device__(const FaceHandle& fh,
                                 const VertexIterator& iter) mutable {
            const VertexHandle v0 = iter[0];
            const VertexHandle v1 = iter[1];
            const VertexHandle v2 = iter[2];
            
            assert(v0.is_valid() && v1.is_valid() && v2.is_valid());

            // 3d position
            Eigen::Vector3<T> ar_3d = coordinates.to_eigen<3>(v0);
            Eigen::Vector3<T> br_3d = coordinates.to_eigen<3>(v1);
            Eigen::Vector3<T> cr_3d = coordinates.to_eigen<3>(v2);

            // Local 2D coordinate system
            Eigen::Vector3<T> n  = (br_3d - ar_3d).cross(cr_3d - ar_3d);
            Eigen::Vector3<T> b1 = (br_3d - ar_3d).normalized();
            Eigen::Vector3<T> b2 = n.cross(b1).normalized();

            // Express a, b, c in local 2D coordinates system
            Eigen::Vector2<T> ar_2d(T(0.0), T(0.0));
            Eigen::Vector2<T> br_2d((br_3d - ar_3d).dot(b1), T(0.0));
            Eigen::Vector2<T> cr_2d((cr_3d - ar_3d).dot(b1),
                                    (cr_3d - ar_3d).dot(b2));

            // Save 2-by-2 matrix with edge vectors as columns
            Eigen::Matrix<T, 2, 2> fout = col_mat(br_2d - ar_2d, cr_2d - ar_2d);

            rest_shape(fh) = fout;
        });
}

PYBIND11_MODULE(_Param, m)
{
    pyrxmesh::require_compatible_runtime(m);
    m.def("compute_normals",
          &compute_rest_shape,
          py::arg("mesh"),
          py::arg("coords"),
          py::arg("rest_coords"));
}
