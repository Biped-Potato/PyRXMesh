from pathlib import Path

import pyrxmesh as rx
import vertex_normal

mesh_path = Path(__file__).parent.parent / "meshes" / "sphere3.obj"

rx.init(0)
mesh = rx.RXMeshStatic(str(mesh_path))
coords = mesh.input_vertex_coordinates()
normals = mesh.add_vertex_attribute("normals", dtype="float32", dim=3)

vertex_normal.compute_normals(mesh,coords,normals)
values = normals.to_numpy_copy(source="device")

for i in range(len(values)):
    print(values[i])