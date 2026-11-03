# Primary source index

Research checked for this wave; source documentation is a contract reference, not proof that a particular exporter executed. Native versions actually run are in manifests. Online documentation may describe newer releases than those installed.

## ROS
- Quaternion fundamentals (official ROS 2 documentation source): https://github.com/ros2/ros2_documentation/blob/rolling/source/ROS-Framework/interfaces/Working-with-interfaces/Quaternion-Fundamentals.rst
- tf2 APIs and source: https://github.com/ros2/geometry2
- REP-103 coordinate conventions: https://www.ros.org/reps/rep-0103.html

## Cesium
- HeadingPitchRoll (axis signs and radians): https://cesium.com/learn/cesiumjs/ref-doc/HeadingPitchRoll.html
- Quaternion: https://cesium.com/learn/cesiumjs/ref-doc/Quaternion.html
- Transforms (ENU, HPR-to-fixed): https://cesium.com/learn/cesiumjs/ref-doc/Transforms.html

## glTF and consumers
- glTF 2.0 specification, including normalized cubic rotation interpolation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Khronos validator: https://github.com/KhronosGroup/glTF-Validator
- Three.js GLTFLoader: https://threejs.org/docs/pages/GLTFLoader.html
- VTK glTF importer: https://vtk.org/doc/nightly/html/classvtkGLTFImporter.html
- Trimesh glTF exchange: https://trimesh.org/trimesh.exchange.gltf.html

The installed VTK 9.6.2 importer was measured to support animation and sampled it successfully. Do not infer actual installed capabilities from a stale overview sentence in online documentation; retain the runtime API observations and logs.

## OpenUSD
- Gf.Quatd: https://openusd.org/release/api/class_gf_quatd.html
- Gf.Matrix3d: https://openusd.org/release/api/class_gf_matrix3d.html
- Gf.Matrix4d: https://openusd.org/release/api/class_gf_matrix4d.html
- UsdGeom.Xformable: https://openusd.org/release/api/class_usd_geom_xformable.html
- Candidate Python distribution: https://pypi.org/project/usd-core/

## Mathematical references
- OpenCV Rodrigues and Jacobian documentation: https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html
- PROJ topocentric conversion: https://proj.org/en/stable/operations/conversions/topocentric.html
- pyproj Transformer: https://pyproj4.github.io/pyproj/stable/api/transformer.html
- CSPICE quaternion multiplication and convention: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/qxq_c.html
- MuJoCo native functions: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html

Separate reference code uses SciPy 1.17.0 Rotation and Slerp, explicit glTF Hermite evaluation, and analytic WGS84/ENU equations. Those are labelled derived_reference, not rewritten native outputs.
