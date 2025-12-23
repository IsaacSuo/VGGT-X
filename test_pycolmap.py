"""
Comprehensive pycolmap 3.13 API test script.
Tests ALL code paths used in np_to_pycolmap.py, including:
1. shared_camera=True scenario (multiple frames, one camera)
2. shared_camera=False scenario (multiple frames, multiple cameras)
3. All pycolmap API calls used in the codebase
"""
import pycolmap
import numpy as np

print(f"pycolmap version: {pycolmap.__version__}")
print("=" * 70)

# ============================================================
# PART 1: Basic API Exploration
# ============================================================
print("\n" + "=" * 70)
print("PART 1: Basic API Exploration")
print("=" * 70)

# --- Rotation3d ---
print("\n--- Rotation3d methods ---")
for m in sorted(dir(pycolmap.Rotation3d)):
    if not m.startswith('_'):
        print(f"  {m}")

print("\nRotation3d creation:")
try:
    r1 = pycolmap.Rotation3d()
    print(f"  Rotation3d() -> OK, matrix:\n{r1.matrix()}")
except Exception as e:
    print(f"  Rotation3d() -> FAILED: {e}")

try:
    r2 = pycolmap.Rotation3d(np.eye(3))
    print(f"  Rotation3d(np.eye(3)) -> OK")
except Exception as e:
    print(f"  Rotation3d(np.eye(3)) -> FAILED: {e}")

# --- Rigid3d ---
print("\n--- Rigid3d methods ---")
for m in sorted(dir(pycolmap.Rigid3d)):
    if not m.startswith('_'):
        print(f"  {m}")

print("\nRigid3d creation:")
try:
    t1 = pycolmap.Rigid3d()
    print(f"  Rigid3d() -> OK, matrix:\n{t1.matrix()}")
except Exception as e:
    print(f"  Rigid3d() -> FAILED: {e}")

try:
    rot = pycolmap.Rotation3d(np.eye(3))
    t2 = pycolmap.Rigid3d(rot, [0, 0, 0])
    print(f"  Rigid3d(Rotation3d, [0,0,0]) -> OK")
except Exception as e:
    print(f"  Rigid3d(Rotation3d, [0,0,0]) -> FAILED: {e}")

try:
    rot = pycolmap.Rotation3d(np.eye(3))
    t3 = pycolmap.Rigid3d(rot, np.zeros(3))
    print(f"  Rigid3d(Rotation3d, np.zeros(3)) -> OK")
except Exception as e:
    print(f"  Rigid3d(Rotation3d, np.zeros(3)) -> FAILED: {e}")

# --- Camera ---
print("\n--- Camera methods ---")
for m in sorted(dir(pycolmap.Camera)):
    if not m.startswith('_'):
        print(f"  {m}")

# --- Image ---
print("\n--- Image methods ---")
for m in sorted(dir(pycolmap.Image)):
    if not m.startswith('_'):
        print(f"  {m}")

# --- Frame ---
print("\n--- Frame methods ---")
for m in sorted(dir(pycolmap.Frame)):
    if not m.startswith('_'):
        print(f"  {m}")

# --- Rig ---
print("\n--- Rig methods ---")
for m in sorted(dir(pycolmap.Rig)):
    if not m.startswith('_'):
        print(f"  {m}")

# --- Reconstruction ---
print("\n--- Reconstruction methods ---")
for m in sorted(dir(pycolmap.Reconstruction)):
    if not m.startswith('_'):
        print(f"  {m}")

# --- Track ---
print("\n--- Track methods ---")
for m in sorted(dir(pycolmap.Track)):
    if not m.startswith('_'):
        print(f"  {m}")

# --- Point2D ---
print("\n--- Point2D methods ---")
for m in sorted(dir(pycolmap.Point2D)):
    if not m.startswith('_'):
        print(f"  {m}")

# Check Point2DList
print("\n--- Point2DList check ---")
if hasattr(pycolmap, 'Point2DList'):
    print("  pycolmap.Point2DList exists")
else:
    print("  pycolmap.Point2DList NOT FOUND")
    # Try alternatives
    for name in dir(pycolmap):
        if 'point2d' in name.lower() or 'list' in name.lower():
            print(f"    Alternative: {name}")

# ============================================================
# PART 2: Test shared_camera=False (Multiple Cameras)
# ============================================================
print("\n" + "=" * 70)
print("PART 2: Test shared_camera=False (N=3 frames, 3 cameras)")
print("=" * 70)

N_FRAMES = 3

try:
    rec = pycolmap.Reconstruction()
    print("Created Reconstruction")

    # Create Rig
    rig = pycolmap.Rig(rig_id=1)
    print(f"Created Rig: rig_id={rig.rig_id}")

    # Add Point3D
    xyz = np.array([1.0, 2.0, 3.0])
    rgb = np.array([255, 128, 64])
    rec.add_point3D(xyz, pycolmap.Track(), rgb)
    print("Added Point3D")

    rig_added = False

    for fidx in range(N_FRAMES):
        print(f"\n--- Frame {fidx} ---")

        # Create camera for each frame
        camera = pycolmap.Camera(
            model='PINHOLE',
            width=100, height=100,
            params=[50, 50, 50, 50],
            camera_id=fidx + 1
        )
        print(f"  Created Camera: camera_id={camera.camera_id}, sensor_id={camera.sensor_id}")
        rec.add_camera(camera)
        print(f"  Added Camera to reconstruction")

        # Add sensor to rig
        if fidx == 0:
            rig.add_ref_sensor(camera.sensor_id)
            print(f"  Added ref_sensor: {camera.sensor_id}")
        else:
            identity_pose = pycolmap.Rigid3d()
            rig.add_sensor(camera.sensor_id, identity_pose)
            print(f"  Added sensor: {camera.sensor_id}")

        # Add rig once
        if not rig_added:
            rec.add_rig(rig)
            rig_added = True
            print(f"  Added Rig to reconstruction")

        # Create pose
        rot = pycolmap.Rotation3d(np.eye(3))
        trans = np.array([float(fidx), 0.0, 0.0])
        pose = pycolmap.Rigid3d(rot, trans)

        frame_id = fidx + 1

        # Create Image
        image = pycolmap.Image(
            image_id=fidx + 1,
            name=f'image_{fidx + 1}.jpg',
            camera_id=camera.camera_id,
            frame_id=frame_id
        )
        print(f"  Created Image: image_id={image.image_id}, data_id={image.data_id}")

        # Create Frame
        frame = pycolmap.Frame(
            frame_id=frame_id,
            rig_id=1,
            rig_from_world=pose
        )
        print(f"  Created Frame: frame_id={frame.frame_id}")

        # Add data_id to frame
        frame.add_data_id(image.data_id)
        print(f"  Added data_id to frame")

        # Add frame to reconstruction
        rec.add_frame(frame)
        print(f"  Added Frame to reconstruction")

        # Add Point2D
        p2d = pycolmap.Point2D(np.array([10.0, 20.0]), 1)
        p2d_list = pycolmap.Point2DList([p2d])
        image.points2D = p2d_list
        print(f"  Set image.points2D")

        # Add image to reconstruction
        rec.add_image(image)
        print(f"  Added Image to reconstruction")

    print(f"\nFinal reconstruction: num_cameras={rec.num_cameras()}, num_images={rec.num_images()}, num_frames={rec.num_frames()}")
    print("PART 2: SUCCESS")

except Exception as e:
    import traceback
    print(f"PART 2 FAILED: {e}")
    traceback.print_exc()

# ============================================================
# PART 3: Test shared_camera=True (Single Camera, Multiple Frames)
# ============================================================
print("\n" + "=" * 70)
print("PART 3: Test shared_camera=True (N=3 frames, 1 shared camera)")
print("=" * 70)

N_FRAMES = 3

try:
    rec = pycolmap.Reconstruction()
    print("Created Reconstruction")

    # Create Rig
    rig = pycolmap.Rig(rig_id=1)
    print(f"Created Rig: rig_id={rig.rig_id}")

    # Add Point3D
    xyz = np.array([1.0, 2.0, 3.0])
    rgb = np.array([255, 128, 64])
    rec.add_point3D(xyz, pycolmap.Track(), rgb)
    print("Added Point3D")

    rig_added = False
    camera = None

    for fidx in range(N_FRAMES):
        print(f"\n--- Frame {fidx} ---")

        # Only create camera on first iteration (shared_camera=True)
        if camera is None:
            camera = pycolmap.Camera(
                model='PINHOLE',
                width=100, height=100,
                params=[50, 50, 50, 50],
                camera_id=1  # Always 1 for shared camera
            )
            print(f"  Created Camera: camera_id={camera.camera_id}, sensor_id={camera.sensor_id}")
            rec.add_camera(camera)
            print(f"  Added Camera to reconstruction")

            # Add sensor to rig (only once)
            rig.add_ref_sensor(camera.sensor_id)
            print(f"  Added ref_sensor: {camera.sensor_id}")
        else:
            print(f"  Reusing camera: camera_id={camera.camera_id}, sensor_id={camera.sensor_id}")

        # Add rig once
        if not rig_added:
            rec.add_rig(rig)
            rig_added = True
            print(f"  Added Rig to reconstruction")

        # Create pose
        rot = pycolmap.Rotation3d(np.eye(3))
        trans = np.array([float(fidx), 0.0, 0.0])
        pose = pycolmap.Rigid3d(rot, trans)

        frame_id = fidx + 1

        # Create Image - always use camera.camera_id (which is 1)
        image = pycolmap.Image(
            image_id=fidx + 1,
            name=f'image_{fidx + 1}.jpg',
            camera_id=camera.camera_id,
            frame_id=frame_id
        )
        print(f"  Created Image: image_id={image.image_id}, camera_id={image.camera_id}, data_id={image.data_id}")

        # Check if data_id.sensor_id matches what's in the rig
        print(f"  Checking rig.has_sensor for image's sensor...")
        # We need to extract sensor_id from data_id
        # data_id is data_t(sensor_id=sensor_t(...), id=...)

        # Create Frame
        frame = pycolmap.Frame(
            frame_id=frame_id,
            rig_id=1,
            rig_from_world=pose
        )
        print(f"  Created Frame: frame_id={frame.frame_id}")

        # Add data_id to frame
        frame.add_data_id(image.data_id)
        print(f"  Added data_id to frame: {image.data_id}")

        # Check rig sensors before add_frame
        print(f"  Rig sensor_ids: {rig.sensor_ids}")
        print(f"  Rig num_sensors: {rig.num_sensors}")

        # Add frame to reconstruction
        print(f"  Calling rec.add_frame(frame)...")
        rec.add_frame(frame)
        print(f"  Added Frame to reconstruction - SUCCESS")

        # Add Point2D
        p2d = pycolmap.Point2D(np.array([10.0, 20.0]), 1)
        p2d_list = pycolmap.Point2DList([p2d])
        image.points2D = p2d_list

        # Add image to reconstruction
        rec.add_image(image)
        print(f"  Added Image to reconstruction")

    print(f"\nFinal reconstruction: num_cameras={rec.num_cameras()}, num_images={rec.num_images()}, num_frames={rec.num_frames()}")
    print("PART 3: SUCCESS")

except Exception as e:
    import traceback
    print(f"PART 3 FAILED: {e}")
    traceback.print_exc()

# ============================================================
# PART 4: Deep dive into data_id and sensor_id relationship
# ============================================================
print("\n" + "=" * 70)
print("PART 4: Deep dive into data_id and sensor_id")
print("=" * 70)

try:
    rec = pycolmap.Reconstruction()

    # Create ONE camera
    camera = pycolmap.Camera(
        model='PINHOLE',
        width=100, height=100,
        params=[50, 50, 50, 50],
        camera_id=1
    )
    rec.add_camera(camera)
    print(f"Camera: camera_id={camera.camera_id}")
    print(f"Camera: sensor_id={camera.sensor_id}")
    print(f"Camera sensor_id type: {type(camera.sensor_id)}")

    # Create multiple images with same camera
    for i in range(3):
        image = pycolmap.Image(
            image_id=i + 1,
            name=f'img_{i+1}.jpg',
            camera_id=1,
            frame_id=i + 1
        )
        print(f"\nImage {i+1}:")
        print(f"  image_id={image.image_id}")
        print(f"  camera_id={image.camera_id}")
        print(f"  data_id={image.data_id}")
        print(f"  data_id type: {type(image.data_id)}")

        # Try to access sensor_id from data_id
        if hasattr(image.data_id, 'sensor_id'):
            print(f"  data_id.sensor_id={image.data_id.sensor_id}")

        # Check if data_id has any useful attributes
        print(f"  data_id attributes: {[m for m in dir(image.data_id) if not m.startswith('_')]}")

except Exception as e:
    import traceback
    print(f"PART 4 FAILED: {e}")
    traceback.print_exc()

# ============================================================
# PART 5: Test reading cam_from_world after reconstruction
# ============================================================
print("\n" + "=" * 70)
print("PART 5: Test reading cam_from_world from Image")
print("=" * 70)

try:
    rec = pycolmap.Reconstruction()

    # Setup camera
    camera = pycolmap.Camera(
        model='PINHOLE',
        width=100, height=100,
        params=[50, 50, 50, 50],
        camera_id=1
    )
    rec.add_camera(camera)

    # Setup rig
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(camera.sensor_id)
    rec.add_rig(rig)

    # Create image and frame with non-identity pose
    rot_matrix = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    rot = pycolmap.Rotation3d(rot_matrix)
    trans = np.array([1.0, 2.0, 3.0])
    pose = pycolmap.Rigid3d(rot, trans)

    image = pycolmap.Image(
        image_id=1,
        name='test.jpg',
        camera_id=1,
        frame_id=1
    )

    frame = pycolmap.Frame(
        frame_id=1,
        rig_id=1,
        rig_from_world=pose
    )
    frame.add_data_id(image.data_id)
    rec.add_frame(frame)
    rec.add_image(image)

    # Now read back cam_from_world
    print("Reading back from reconstruction:")
    img = rec.images[1]
    print(f"  image.image_id = {img.image_id}")
    print(f"  image.has_pose = {img.has_pose}")

    if hasattr(img, 'cam_from_world'):
        cfw = img.cam_from_world
        print(f"  image.cam_from_world exists")
        if hasattr(cfw, 'matrix'):
            mat = cfw.matrix()
            print(f"  cam_from_world.matrix():\n{mat}")
        if hasattr(cfw, 'rotation'):
            print(f"  cam_from_world.rotation.matrix():\n{cfw.rotation.matrix()}")
        if hasattr(cfw, 'translation'):
            print(f"  cam_from_world.translation: {cfw.translation}")
    else:
        print("  image.cam_from_world NOT AVAILABLE")
        print(f"  Available attributes: {[m for m in dir(img) if not m.startswith('_')]}")

    print("PART 5: SUCCESS")

except Exception as e:
    import traceback
    print(f"PART 5 FAILED: {e}")
    traceback.print_exc()

# ============================================================
# PART 6: Test pycolmap_to_batch_np_matrix reading pattern
# ============================================================
print("\n" + "=" * 70)
print("PART 6: Test reading pattern (pycolmap_to_batch_np_matrix)")
print("=" * 70)

try:
    rec = pycolmap.Reconstruction()

    # Create reconstruction with 2 images
    camera = pycolmap.Camera(
        model='SIMPLE_PINHOLE',
        width=100, height=100,
        params=[50, 50, 50],
        camera_id=1
    )
    rec.add_camera(camera)

    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(camera.sensor_id)
    rec.add_rig(rig)

    for i in range(2):
        rot = pycolmap.Rotation3d(np.eye(3))
        trans = np.array([float(i), 0.0, 0.0])
        pose = pycolmap.Rigid3d(rot, trans)

        image = pycolmap.Image(
            image_id=i + 1,
            name=f'img_{i+1}.jpg',
            camera_id=1,
            frame_id=i + 1
        )

        frame = pycolmap.Frame(
            frame_id=i + 1,
            rig_id=1,
            rig_from_world=pose
        )
        frame.add_data_id(image.data_id)
        rec.add_frame(frame)
        rec.add_image(image)

        # Add a point
        xyz = np.array([float(i), 1.0, 2.0])
        rgb = np.array([255, 0, 0])
        rec.add_point3D(xyz, pycolmap.Track(), rgb)

    print(f"Created reconstruction with {rec.num_images()} images, {rec.num_points3D()} points")

    # Test reading pattern from pycolmap_to_batch_np_matrix
    print("\nTesting reading pattern:")

    num_images = len(rec.images)
    print(f"  len(rec.images) = {num_images}")

    # Check if point3D_ids() is a method or property
    print(f"\n  rec.point3D_ids type: {type(rec.point3D_ids)}")
    if callable(rec.point3D_ids):
        point_ids = rec.point3D_ids()
        print(f"  rec.point3D_ids() = {list(point_ids)}")
    else:
        print(f"  rec.point3D_ids = {list(rec.point3D_ids)}")

    # Read images
    for i in range(num_images):
        print(f"\n  Reading image {i+1}:")
        pyimg = rec.images[i + 1]
        pycam = rec.cameras[pyimg.camera_id]

        print(f"    image.image_id = {pyimg.image_id}")
        print(f"    image.camera_id = {pyimg.camera_id}")

        # Get pose matrix
        matrix = pyimg.cam_from_world.matrix()
        print(f"    cam_from_world.matrix() shape: {matrix.shape}")
        print(f"    cam_from_world.matrix():\n{matrix}")

        # Get calibration
        calib = pycam.calibration_matrix()
        print(f"    camera.calibration_matrix():\n{calib}")
        print(f"    camera.params: {pycam.params}")

    print("\nPART 6: SUCCESS")

except Exception as e:
    import traceback
    print(f"PART 6 FAILED: {e}")
    traceback.print_exc()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("""
Key findings to apply to np_to_pycolmap.py:

1. Identity transforms:
   - Use pycolmap.Rigid3d() for identity transform
   - Use pycolmap.Rotation3d() or pycolmap.Rotation3d(np.eye(3)) for identity rotation

2. Frame/Rig architecture (pycolmap 3.12+):
   - Create Rig first
   - Add sensors to Rig BEFORE adding Rig to Reconstruction
   - For ref sensor: rig.add_ref_sensor(camera.sensor_id)
   - For other sensors: rig.add_sensor(camera.sensor_id, pycolmap.Rigid3d())
   - Add Rig to Reconstruction: rec.add_rig(rig)
   - For each frame:
     a. Create Image (to get data_id)
     b. Create Frame with pose
     c. Add data_id to Frame: frame.add_data_id(image.data_id)
     d. Add Frame: rec.add_frame(frame)
     e. Add Image: rec.add_image(image)

3. shared_camera=True:
   - Only ONE camera is created
   - Only ONE sensor is added to rig (via add_ref_sensor)
   - All images use the same camera_id
   - Each image has SAME sensor_id but different data_id.id
   - The rig only needs to have the one sensor

4. Reading from reconstruction:
   - image.cam_from_world.matrix() returns 3x4 extrinsic matrix
   - camera.calibration_matrix() returns 3x3 intrinsic matrix
""")
