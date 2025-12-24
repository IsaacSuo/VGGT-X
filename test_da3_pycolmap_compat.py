#!/usr/bin/env python3
"""
Test DA3's pycolmap API usage for compatibility with pycolmap 3.13.
This tests the specific API patterns used in DA3's COLMAP export.
"""

import pycolmap
import numpy as np
import tempfile
import os

def test_da3_colmap_export_pattern():
    """
    Test the exact API pattern used in DA3's export_to_colmap function.
    Reference: Depth-Anything-3/src/depth_anything_3/utils/export/colmap.py
    """
    print("=" * 60)
    print("Testing DA3 COLMAP Export Pattern with pycolmap", pycolmap.__version__)
    print("=" * 60)

    # Simulate DA3 prediction data
    num_frames = 3
    h, w = 64, 64

    # Create sample data
    points = np.random.rand(100, 3).astype(np.float64)
    colors = (np.random.rand(100, 3) * 255).astype(np.uint8)

    # Sample intrinsics and extrinsics
    intrinsics = np.array([
        [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
    ], dtype=np.float64)

    # w2c extrinsics with identity rotation and increasing translation
    extrinsics = np.zeros((num_frames, 3, 4), dtype=np.float64)
    for i in range(num_frames):
        extrinsics[i, :3, :3] = np.eye(3)
        extrinsics[i, :3, 3] = [i * 0.5, 0, 0]  # Translation along x

    orig_w, orig_h = 640, 480

    # Following DA3's export_to_colmap logic
    reconstruction = pycolmap.Reconstruction()

    # 1. Add 3D points (same as DA3)
    print("\n[1] Adding 3D points...")
    point3d_ids = []
    for vidx in range(len(points)):
        point3d_id = reconstruction.add_point3D(points[vidx], pycolmap.Track(), colors[vidx])
        point3d_ids.append(point3d_id)
    print(f"    Added {len(point3d_ids)} points. OK")

    # 2. Process each frame (DA3's loop)
    print("\n[2] Processing frames with DA3's pattern...")

    for fidx in range(num_frames):
        print(f"\n    Frame {fidx}:")

        intrinsic = intrinsics[fidx]
        pycolmap_intri = np.array([
            intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        ])

        extrinsic = extrinsics[fidx]

        # DA3 line 77: Create Rigid3d from rotation matrix and translation
        print(f"      Creating Rigid3d from rotation matrix and translation...")
        try:
            cam_from_world = pycolmap.Rigid3d(
                pycolmap.Rotation3d(extrinsic[:3, :3]),
                extrinsic[:3, 3]
            )
            print(f"      Rigid3d created. OK")
        except Exception as e:
            print(f"      ERROR: {e}")
            print(f"      This may need alternative API!")
            return False

        # DA3 line 80-86: Create and add camera
        print(f"      Creating camera...")
        camera = pycolmap.Camera()
        camera.camera_id = fidx + 1
        camera.model = pycolmap.CameraModelId.PINHOLE
        camera.width = orig_w
        camera.height = orig_h
        camera.params = pycolmap_intri
        reconstruction.add_camera(camera)
        print(f"      Camera {camera.camera_id} added. sensor_id={camera.sensor_id}")

        # DA3 line 88-92: Create rig and add sensor
        print(f"      Creating rig...")
        rig = pycolmap.Rig()
        rig.rig_id = camera.camera_id

        # DA3 uses add_ref_sensor - check if this exists
        print(f"      Checking rig.add_ref_sensor...")
        if hasattr(rig, 'add_ref_sensor'):
            print(f"      rig.add_ref_sensor exists")
            try:
                rig.add_ref_sensor(camera.sensor_id)
                print(f"      rig.add_ref_sensor(sensor_id={camera.sensor_id}) OK")
            except Exception as e:
                print(f"      ERROR: {e}")
                # Try alternative
                print(f"      Trying rig.add_sensor with identity transform...")
                try:
                    rig.add_sensor(camera.sensor_id, pycolmap.Rigid3d())
                    print(f"      rig.add_sensor with identity OK")
                except Exception as e2:
                    print(f"      ERROR: {e2}")
                    return False
        else:
            print(f"      rig.add_ref_sensor NOT FOUND!")
            print(f"      Trying rig.add_sensor with identity transform...")
            try:
                rig.add_sensor(camera.sensor_id, pycolmap.Rigid3d())
                print(f"      rig.add_sensor OK")
            except Exception as e:
                print(f"      ERROR: {e}")
                return False

        # Add rig to reconstruction
        print(f"      Adding rig to reconstruction...")
        reconstruction.add_rig(rig)
        print(f"      Rig {rig.rig_id} added. OK")

        # DA3 line 95-105: Create image and frame
        print(f"      Creating image...")
        image = pycolmap.Image()
        image.image_id = fidx + 1
        image.camera_id = camera.camera_id

        print(f"      Creating frame...")
        frame = pycolmap.Frame()
        frame.frame_id = image.image_id
        frame.rig_id = camera.camera_id

        # DA3 line 103: frame.add_data_id(image.data_id)
        print(f"      Adding image.data_id to frame...")
        try:
            frame.add_data_id(image.data_id)
            print(f"      frame.add_data_id OK. data_id={image.data_id}")
        except Exception as e:
            print(f"      ERROR: {e}")
            return False

        # DA3 line 104
        frame.rig_from_world = cam_from_world

        print(f"      Adding frame to reconstruction...")
        try:
            reconstruction.add_frame(frame)
            print(f"      Frame {frame.frame_id} added. OK")
        except Exception as e:
            print(f"      ERROR: {e}")
            return False

        # DA3 line 107-118: Add point2D
        print(f"      Creating Point2D list...")
        point2d_list = []
        # Just add a few test points
        for vidx in range(min(5, len(point3d_ids))):
            point2d = np.array([100.0 + vidx, 200.0 + vidx])
            point3d_id = point3d_ids[vidx]
            try:
                p2d = pycolmap.Point2D(point2d, point3d_id)
                point2d_list.append(p2d)
            except Exception as e:
                print(f"      ERROR creating Point2D: {e}")
                return False
        print(f"      Created {len(point2d_list)} Point2D objects. OK")

        # DA3 line 120-124: Finalize and add image
        image.frame_id = image.image_id
        image.name = f"image_{fidx:04d}.jpg"

        print(f"      Creating Point2DList...")
        try:
            image.points2D = pycolmap.Point2DList(point2d_list)
            print(f"      Point2DList assigned to image. OK")
        except Exception as e:
            print(f"      ERROR: {e}")
            return False

        print(f"      Adding image to reconstruction...")
        try:
            reconstruction.add_image(image)
            print(f"      Image {image.image_id} added. OK")
        except Exception as e:
            print(f"      ERROR: {e}")
            return False

    # 3. Test writing reconstruction
    print("\n[3] Writing reconstruction to disk...")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            reconstruction.write(tmpdir)
            print(f"    Written to {tmpdir}")

            # List files
            files = os.listdir(tmpdir)
            print(f"    Files created: {files}")

            # Try reading back
            print(f"    Reading back reconstruction...")
            recon2 = pycolmap.Reconstruction()
            recon2.read(tmpdir)
            print(f"    Read back: {len(recon2.cameras)} cameras, {len(recon2.images)} images, {len(recon2.points3D)} points")
            print(f"    Write/Read test OK")
        except Exception as e:
            print(f"    ERROR: {e}")
            return False

    print("\n" + "=" * 60)
    print("All DA3 pycolmap patterns verified successfully!")
    print("=" * 60)
    return True


def check_alternative_apis():
    """Check alternative APIs that might be needed."""
    print("\n" + "=" * 60)
    print("Checking Alternative APIs")
    print("=" * 60)

    # Check what methods are available on Rig
    rig = pycolmap.Rig()
    print("\nRig methods containing 'sensor':")
    for attr in dir(rig):
        if 'sensor' in attr.lower():
            print(f"  - {attr}")

    # Check Rotation3d construction
    print("\nRotation3d construction options:")
    try:
        r1 = pycolmap.Rotation3d()
        print(f"  Rotation3d() [no args]: OK")
    except Exception as e:
        print(f"  Rotation3d() [no args]: ERROR - {e}")

    try:
        mat = np.eye(3)
        r2 = pycolmap.Rotation3d(mat)
        print(f"  Rotation3d(3x3 matrix): OK")
    except Exception as e:
        print(f"  Rotation3d(3x3 matrix): ERROR - {e}")

    # Check Rigid3d construction
    print("\nRigid3d construction options:")
    try:
        t1 = pycolmap.Rigid3d()
        print(f"  Rigid3d() [no args]: OK (identity)")
    except Exception as e:
        print(f"  Rigid3d() [no args]: ERROR - {e}")

    try:
        t2 = pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([0, 0, 0]))
        print(f"  Rigid3d(Rotation3d, translation): OK")
    except Exception as e:
        print(f"  Rigid3d(Rotation3d, translation): ERROR - {e}")


if __name__ == "__main__":
    check_alternative_apis()
    print()
    success = test_da3_colmap_export_pattern()
    exit(0 if success else 1)
