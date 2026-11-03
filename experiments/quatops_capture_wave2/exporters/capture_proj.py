"""Capture PROJ ENU/ECEF transforms as independent support for the Cesium frame profile."""
from __future__ import annotations

import json
from pathlib import Path

import pyproj
from pyproj import CRS, Transformer, network
from pyproj.enums import TransformDirection

from shared.capture_io import ROOT, Capture, Observation, contract, numeric, parser


def main() -> None:
    cli=parser(__doc__ or "PROJ capture")
    cli.add_argument("--geographic",type=Path,default=ROOT/"corpus/cesium_geographic_inputs.json")
    args=cli.parse_args()
    network.set_network_enabled(False)
    data=json.loads(args.geographic.read_text())
    module_dir=Path(pyproj.__file__).parent
    binary=next(iter(module_dir.glob("_transformer*.so")),None)
    cap=Capture("proj_geographic","PROJ via pyproj",pyproj.proj_version_str,args.geographic,args.output,Path(__file__),
        {"input":"binary64 geographic degrees/metres", "arithmetic":"native PROJ double", "output":"binary64", "serialized":"JSON plus IEEE bits"},
        oracle_family="proj",native_binary=binary,dependencies=[{"name":"pyproj","version":pyproj.__version__}],
        limitations=["This is PROJ execution, NOT Cesium execution.",
            "Local Cartesian point transforms only; no quaternion/HPR function is inferred from these outputs.",
            "At geographic poles longitude fixes a convention; Cesium may choose a different exact-pole fallback.",
            "Network disabled; WGS84 ellipsoid only, no external datum grids."])
    cap.artifacts.append((args.geographic.resolve(),"geographic_input_corpus"))
    geocentric=Transformer.from_crs(4979,4978,always_xy=True)
    ellipsoid=CRS.from_epsg(4979).ellipsoid
    for item in data["locations"]:
        lon,lat,h=item["longitude_degrees"],item["latitude_degrees"],item["height_m"]
        origin=geocentric.transform(lon,lat,h,errcheck=True)
        pipeline=f"+proj=topocentric +ellps=WGS84 +lon_0={lon:.17g} +lat_0={lat:.17g} +h_0={h:.17g}"
        local=Transformer.from_pipeline(pipeline)
        probes=[[0,0,0],[1,0,0],[0,1,0],[0,0,1],data["local_point_xyz_m"]]
        earth=[local.transform(*v,direction=TransformDirection.INVERSE,errcheck=True) for v in probes]
        back=[local.transform(*v,errcheck=True) for v in earth]
        cap.add(Observation(item["case_id"],"geographic-enu-ecef", "pyproj.Transformer.from_crs(4979,4978); from_pipeline(topocentric); transform",
            contract(units="geographic degrees; Cartesian metres",frame="forward ECEF -> ENU; inverse ENU -> ECEF"),
            {"longitude_degrees":numeric(lon),"latitude_degrees":numeric(lat),"height_m":numeric(h),
             "pipeline":pipeline,"always_xy":True,"ellipsoid_semi_major_m":numeric(ellipsoid.semi_major_metre),
             "ellipsoid_inverse_flattening":numeric(ellipsoid.inverse_flattening),"local_probe_points":numeric(probes)},
            {"origin_ecef_m":numeric(list(origin)),"ecef_probe_points":numeric([list(v) for v in earth]),
             "local_readback_points":numeric([list(v) for v in back])}))
    print(cap.finish(["FRAME"]))


if __name__=="__main__":
    main()
