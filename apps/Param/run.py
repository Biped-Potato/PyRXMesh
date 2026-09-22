from pathlib import Path

import argparse
import pyrxmesh as rx
import Param
import igl

import rxmesh_diff_energy

#general outline
#generate uvs if none are passed in
#build rest shape
#use gradient descent and energy function to find good UV positions
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--obj-file-name",   default=Path(__file__).parent.parent.parent / "meshes" / "bunnyhead.obj", type=Path)
    #parser.add_argument("--output-folder",   default=Path(__file__), type=Path)
    parser.add_argument("--uv-file-name",    default="", type=str)
    parser.add_argument("--solver",          default="cudss_choi", type=str)
    parser.add_argument("--device-id",       default=0, type=int)
    parser.add_argument("--cg-abs-tol",      default=1e-6, type=Path)
    parser.add_argument("--cg-rel-tol",      default=0.0, type=Path)
    parser.add_argument("--cg-max-iter",     default=10, type=Path)
    parser.add_argument("--newton-max-iter", default=100, type=Path)

    args = parser.parse_args()

    rx.init(args.device_id)
    mesh = rx.RXMeshStatic(str(args.obj_file_name))

    if mesh.is_closed():
        raise ValueError("The input mesh is closed. THe input mesh should have boundaries.")
        return

    coordinates = mesh.input_vertex_coordinates()
    
    #use a column major vector of dimension 4 instead of a 2x2 matrix
    rest_shape = mesh.add_face_attribute("fRestShape", dtype="float32", dim=4)

    #define energy term
    energy = rxmesh_diff_energy.make_energy(mesh)

    #initialize uv coordinates attribute in mesh
    uv_attr = mesh.add_vertex_attribute("uv", dtype="float32", dim=2)
    if not args.uv_file_name:
        V = mesh.vertices()
        F = mesh.faces()
        b = igl.boundary_loop(F)
        bc = igl.map_vertices_to_circle(V, b)
        #pass this as the optimization variable
        uv = igl.harmonic(V, F, b, bc, 1)
        uv_attr.from_numpy_copy(uv, target="all")
    else:
        #uv here would be UV coordinates with z = anything and F are indices
        (uv, fv) = igl.read_triangle_mesh(args.uv_file_name)
        if V.shape[0] != mesh.num_vertices:
            raise ValueError(
                f"Number of vertices in the input UV file {V.shape[0]} does not match \
                the number of vertices in the mesh {mesh.num_vertices}."
            )
        uv_attr.from_numpy_copy(uv[:,:2], target="all")


    Param.compute_rest_shape(mesh,coordinates,rest_shape)
    values = rest_shape.to_numpy_copy(source="device")

    for i in range(len(values)):
        print(values[i])

if __name__ == "__main__":
    main()