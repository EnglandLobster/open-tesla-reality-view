# 3D Ego-Vehicle PBR Assets

This directory holds local (gitignored) `.glb` vehicle meshes used by `EgoVehicleCompositor` for photorealistic Stage 0 and Reality View ego-car compositing.

## Default Asset: `ego_car_concept.glb`
- **Source:** Khronos Group glTF 2.0 `CarConcept` sample asset (via `studio-public-demos/car-concept-3d-dashboard`)
- **Original Author:** Unity Fan (Sketchfab / Khronos Group)
- **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**)
- **Download Command:**
  ```bash
  curl -sSL "https://raw.githubusercontent.com/studio-public-demos/car-concept-3d-dashboard/main/CarConcept.glb" \
    -o assets/ego_car_concept.glb
  ```

You can drop any standard glTF 2.0 binary (`.glb`) vehicle model here and pass `--ego-glb path/to/model.glb` to the visualizer or exporter.
