# Param

Custom CUDA kernels for PyRXMesh.

Edit `src/Param.cu`, then build in the environment containing PyRXMesh:

```bash
python -m pip install -v --no-build-isolation .
```

```python
import pyrxmesh as rx
import Param

mesh = rx.RXMeshStatic("mesh.obj")
coords = mesh.input_vertex_coordinates()
lengths = mesh.add_edge_attribute("edge_lengths", dtype="float32", dim=1)
Param.compute_edge_lengths(mesh, coords, lengths)
```

See PyRXMesh's `CUSTOM_CUDA_PLUGINS.md` for the complete authoring guide.
