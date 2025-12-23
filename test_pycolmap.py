"""
Comprehensive pycolmap 3.13 API test script.
Tests all functions and attributes used in np_to_pycolmap.py
"""
import pycolmap
import numpy as np

print(f"pycolmap version: {pycolmap.__version__}")
print("=" * 60)

# ============================================================
# Section 1: Basic Types
# ============================================================
print("\n=== Section 1: Basic Types ===")

# Test Rotation3d creation methods
print("\n--- Rotation3d ---")
print("Available Rotation3d methods/attributes:")
for m in sorted(dir(pycolmap.Rotation3d)):
    if not m.startswith('_'):
        print(f"  {m}")

# Try different ways to create identity rotation
print("\nTrying to create identity rotation:")

# Method 1: identity() class method
try:
    rot_identity = pycolmap.Rotation3d.identity()
    print(f"  Method 1 - Rotation3d.identity(): SUCCESS")
except Exception as e:
    print(f"  Method 1 - Rotation3d.identity(): FAILED - {e}")

# Method 2: from identity matrix
try:
    rot_from_eye = pycolmap.Rotation3d(np.eye(3))
    print(f"  Method 2 - Rotation3d(np.eye(3)): SUCCESS")
except Exception as e:
    print(f"  Method 2 - Rotation3d(np.eye(3)): FAILED - {e}")

# Method 3: no arguments
try:
    rot_default = pycolmap.Rotation3d()
    print(f"  Method 3 - Rotation3d(): SUCCESS")
except Exception as e:
    print(f"  Method 3 - Rotation3d(): FAILED - {e}")

# Method 4: from quaternion [w, x, y, z] - identity is [1, 0, 0, 0]
try:
    rot_quat = pycolmap.Rotation3d(np.array([1.0, 0.0, 0.0, 0.0]))
    print(f"  Method 4 - Rotation3d([1,0,0,0]) quaternion: SUCCESS")
except Exception as e:
    print(f"  Method 4 - Rotation3d([1,0,0,0]) quaternion: FAILED - {e}")

# Test Rigid3d creation
print("\n--- Rigid3d ---")
print("Available Rigid3d methods/attributes:")
for m in sorted(dir(pycolmap.Rigid3d)):
    if not m.startswith('_'):
        print(f"  {m}")

# Try to create identity Rigid3d
print("\nTrying to create identity Rigid3d:")
try:
    rigid_identity = pycolmap.Rigid3d()
    print(f"  Method 1 - Rigid3d(): SUCCESS")
except Exception as e:
    print(f"  Method 1 - Rigid3d(): FAILED - {e}")

try:
    rot = pycolmap.Rotation3d(np.eye(3))
    rigid_from_rot = pycolmap.Rigid3d(rot, np.zeros(3))
    print(f"  Method 2 - Rigid3d(Rotation3d(eye), zeros): SUCCESS")
except Exception as e:
    print(f"  Method 2 - Rigid3d(Rotation3d(eye), zeros): FAILED - {e}")

try:
    rigid_from_rot2 = pycolmap.Rigid3d(rot, [0, 0, 0])
    print(f"  Method 3 - Rigid3d(Rotation3d(eye), [0,0,0]): SUCCESS")
except Exception as e:
    print(f"  Method 3 - Rigid3d(Rotation3d(eye), [0,0,0]): FAILED - {e}")

# ============================================================
# Section 2: Reconstruction and Camera
# ============================================================
print("\n=== Section 2: Reconstruction and Camera ===")

rec = pycolmap.Reconstruction()
print(f"Created Reconstruction")

# Camera
print("\n--- Camera ---")
camera = pycolmap.Camera(
    model='PINHOLE',
    width=100, height=100,
    params=[50, 50, 50, 50],
    camera_id=1
)
print(f"Camera created: camera_id={camera.camera_id}")

# Check if sensor_id exists
if hasattr(camera, 'sensor_id'):
    print(f"  sensor_id={camera.sensor_id}")
else:
    print("  sensor_id: NOT AVAILABLE")

rec.add_camera(camera)
print("Camera added to reconstruction")

# Check camera methods
print("\nCamera methods/attributes:")
for m in sorted(dir(camera)):
    if not m.startswith('_'):
        try:
            attr = getattr(camera, m)
            if callable(attr):
                print(f"  {m}()")
            else:
                print(f"  {m} = {attr}")
        except:
            print(f"  {m}")

# ============================================================
# Section 3: Rig
# ============================================================
print("\n=== Section 3: Rig ===")

print("Available Rig methods/attributes:")
for m in sorted(dir(pycolmap.Rig)):
    if not m.startswith('_'):
        print(f"  {m}")

rig = pycolmap.Rig(rig_id=1)
print(f"Rig created: rig_id={rig.rig_id}")

# Test add_ref_sensor
print("\nTrying add_ref_sensor:")
try:
    rig.add_ref_sensor(camera.sensor_id)
    print(f"  add_ref_sensor({camera.sensor_id}): SUCCESS")
except Exception as e:
    print(f"  add_ref_sensor({camera.sensor_id}): FAILED - {e}")

# Create a second camera to test add_sensor
camera2 = pycolmap.Camera(
    model='PINHOLE',
    width=100, height=100,
    params=[50, 50, 50, 50],
    camera_id=2
)
rec.add_camera(camera2)

# Test add_sensor with different identity pose methods
print("\nTrying add_sensor with identity pose:")

# Find working identity pose method
identity_pose = None
try:
    identity_pose = pycolmap.Rigid3d()
    print("  Using Rigid3d() for identity pose")
except:
    try:
        rot = pycolmap.Rotation3d(np.eye(3))
        identity_pose = pycolmap.Rigid3d(rot, [0, 0, 0])
        print("  Using Rigid3d(Rotation3d(eye), [0,0,0]) for identity pose")
    except Exception as e:
        print(f"  Could not create identity pose: {e}")

if identity_pose is not None:
    try:
        rig.add_sensor(camera2.sensor_id, identity_pose)
        print(f"  add_sensor({camera2.sensor_id}, identity_pose): SUCCESS")
    except Exception as e:
        print(f"  add_sensor({camera2.sensor_id}, identity_pose): FAILED - {e}")

# Add rig to reconstruction
print("\nAdding rig to reconstruction:")
try:
    rec.add_rig(rig)
    print("  add_rig: SUCCESS")
except Exception as e:
    print(f"  add_rig: FAILED - {e}")

# ============================================================
# Section 4: Image and Frame
# ============================================================
print("\n=== Section 4: Image and Frame ===")

print("Available Image constructor parameters:")
# Try to inspect Image signature
import inspect
try:
    sig = inspect.signature(pycolmap.Image.__init__)
    print(f"  {sig}")
except:
    print("  Could not get signature")

print("\nImage methods/attributes (from class):")
for m in sorted(dir(pycolmap.Image)):
    if not m.startswith('_'):
        print(f"  {m}")

# Create Image for pycolmap 3.12+
print("\nCreating Image (pycolmap 3.12+ style):")
try:
    image = pycolmap.Image(
        image_id=1,
        name='test.jpg',
        camera_id=camera.camera_id,
        frame_id=1
    )
    print(f"  Image created: image_id={image.image_id}")
    if hasattr(image, 'data_id'):
        print(f"  data_id={image.data_id}")
    else:
        print("  data_id: NOT AVAILABLE")
except Exception as e:
    print(f"  FAILED: {e}")

# Frame
print("\n--- Frame ---")
print("Available Frame methods/attributes:")
for m in sorted(dir(pycolmap.Frame)):
    if not m.startswith('_'):
        print(f"  {m}")

# Create pose
rot = pycolmap.Rotation3d(np.eye(3))
trans = np.array([0.0, 0.0, 1.0])
pose = pycolmap.Rigid3d(rot, trans)
print(f"\nCreated pose: Rigid3d")

# Create Frame
print("\nCreating Frame:")
try:
    frame = pycolmap.Frame(
        frame_id=1,
        rig_id=1,
        rig_from_world=pose
    )
    print(f"  Frame created: frame_id={frame.frame_id}")
except Exception as e:
    print(f"  FAILED: {e}")

# Add data_id to frame
print("\nAdding data_id to frame:")
try:
    frame.add_data_id(image.data_id)
    print(f"  add_data_id({image.data_id}): SUCCESS")
except Exception as e:
    print(f"  add_data_id: FAILED - {e}")

# Add frame to reconstruction
print("\nAdding frame to reconstruction:")
try:
    rec.add_frame(frame)
    print("  add_frame: SUCCESS")
except Exception as e:
    print(f"  add_frame: FAILED - {e}")

# Add image to reconstruction
print("\nAdding image to reconstruction:")
try:
    rec.add_image(image)
    print("  add_image: SUCCESS")
except Exception as e:
    print(f"  add_image: FAILED - {e}")

# ============================================================
# Section 5: Point2D and Point3D
# ============================================================
print("\n=== Section 5: Point2D and Point3D ===")

# Track
print("--- Track ---")
track = pycolmap.Track()
print(f"Track created")
print("Track methods:")
for m in sorted(dir(track)):
    if not m.startswith('_'):
        print(f"  {m}")

# Add Point3D
print("\n--- Point3D ---")
xyz = np.array([1.0, 2.0, 3.0])
rgb = np.array([255, 128, 64])
try:
    rec.add_point3D(xyz, pycolmap.Track(), rgb)
    print(f"add_point3D: SUCCESS")
except Exception as e:
    print(f"add_point3D: FAILED - {e}")

# Point2D
print("\n--- Point2D ---")
print("Point2D methods:")
for m in sorted(dir(pycolmap.Point2D)):
    if not m.startswith('_'):
        print(f"  {m}")

try:
    p2d = pycolmap.Point2D(np.array([10.0, 20.0]), 1)
    print(f"Point2D created: xy={p2d.xy}, point3D_id={p2d.point3D_id}")
except Exception as e:
    print(f"Point2D creation FAILED: {e}")

# Point2DList
print("\n--- Point2DList ---")
try:
    p2d_list = pycolmap.Point2DList([p2d])
    print(f"Point2DList created with {len(p2d_list)} points")
except Exception as e:
    print(f"Point2DList creation FAILED: {e}")

# Assign to image
print("\nAssigning points2D to image:")
try:
    image.points2D = p2d_list
    print("  image.points2D assignment: SUCCESS")
except Exception as e:
    print(f"  image.points2D assignment: FAILED - {e}")

# ============================================================
# Section 6: Track operations
# ============================================================
print("\n=== Section 6: Track operations ===")

try:
    point3d = rec.points3D[1]
    print(f"Got point3D[1]: xyz={point3d.xyz}")
    track = point3d.track
    print(f"Got track from point3D")
    track.add_element(1, 0)  # image_id=1, point2D_idx=0
    print("track.add_element(1, 0): SUCCESS")
except Exception as e:
    print(f"Track operations FAILED: {e}")

# ============================================================
# Section 7: Reading back from reconstruction
# ============================================================
print("\n=== Section 7: Reading back ===")

print(f"num_cameras: {rec.num_cameras}")
print(f"num_images: {rec.num_images}")
print(f"num_rigs: {rec.num_rigs}")
print(f"num_frames: {rec.num_frames}")
print(f"num_points3D: {rec.num_points3D}")

# Read image cam_from_world
print("\nReading image pose:")
try:
    img = rec.images[1]
    print(f"  image.image_id = {img.image_id}")
    print(f"  image.camera_id = {img.camera_id}")
    if hasattr(img, 'cam_from_world'):
        cfw = img.cam_from_world
        print(f"  image.cam_from_world exists")
        if hasattr(cfw, 'matrix'):
            mat = cfw.matrix()
            print(f"  cam_from_world.matrix() shape: {mat.shape}")
    else:
        print("  image.cam_from_world: NOT AVAILABLE")
except Exception as e:
    print(f"  Reading image: FAILED - {e}")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 60)
print("=== SUMMARY ===")
print("=" * 60)

# Determine working identity pose method
print("\nWorking identity pose creation:")
identity_methods = []
try:
    pycolmap.Rigid3d()
    identity_methods.append("pycolmap.Rigid3d()")
except:
    pass
try:
    pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), [0, 0, 0])
    identity_methods.append("pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), [0, 0, 0])")
except:
    pass
try:
    pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.zeros(3))
    identity_methods.append("pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.zeros(3))")
except:
    pass

for m in identity_methods:
    print(f"  {m}")

print("\n=== TEST COMPLETE ===")
