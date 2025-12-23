import pycolmap
import numpy as np

print(f"pycolmap version: {pycolmap.__version__}")

# 创建 reconstruction
rec = pycolmap.Reconstruction()

# 1. 创建并添加 Camera
print("\n=== Step 1: Add Camera ===")
camera = pycolmap.Camera(
    model='PINHOLE',
    width=100, height=100,
    params=[50, 50, 50, 50],
    camera_id=1
)
rec.add_camera(camera)
print(f"Camera added: camera_id={camera.camera_id}, sensor_id={camera.sensor_id}")

# 2. 创建 Rig
print("\n=== Step 2: Create Rig ===")
rig = pycolmap.Rig(rig_id=1)
print(f"Rig created: rig_id={rig.rig_id}")

# 3. 添加 sensor 到 rig
print("\n=== Step 3: Add sensor to Rig ===")
try:
    rig.add_ref_sensor(camera.sensor_id)
    print(f"ref_sensor added: {camera.sensor_id}")
except Exception as e:
    print(f"add_ref_sensor FAILED: {e}")

# 也尝试 add_sensor
print("\nTrying rig methods:")
for m in dir(rig):
    if 'sensor' in m.lower() or 'add' in m.lower():
        print(f"  {m}")

# 4. 添加 Rig 到 reconstruction
print("\n=== Step 4: Add Rig to reconstruction ===")
rec.add_rig(rig)
print("Rig added to reconstruction")

# 5. 创建 Image
print("\n=== Step 5: Create Image ===")
image = pycolmap.Image(
    image_id=1,
    name='test.jpg',
    camera_id=camera.camera_id,
    frame_id=1
)
print(f"Image created: image_id={image.image_id}, data_id={image.data_id}")

# 6. 创建 Frame
print("\n=== Step 6: Create Frame ===")
rot = pycolmap.Rotation3d(np.eye(3))
trans = np.array([0.0, 0.0, 0.0])
pose = pycolmap.Rigid3d(rot, trans)
frame = pycolmap.Frame(
    frame_id=1,
    rig_id=1,
    rig_from_world=pose
)
print(f"Frame created: frame_id={frame.frame_id}")

# 7. 添加 data_id 到 Frame
print("\n=== Step 7: Add data_id to Frame ===")
try:
    frame.add_data_id(image.data_id)
    print(f"data_id added to frame")
except Exception as e:
    print(f"add_data_id FAILED: {e}")

# 8. 添加 Frame 到 reconstruction
print("\n=== Step 8: Add Frame to reconstruction ===")
try:
    rec.add_frame(frame)
    print("Frame added to reconstruction")
except Exception as e:
    print(f"add_frame FAILED: {e}")

# 9. 添加 Image 到 reconstruction
print("\n=== Step 9: Add Image to reconstruction ===")
try:
    rec.add_image(image)
    print("Image added to reconstruction")
except Exception as e:
    print(f"add_image FAILED: {e}")

# 10. 验证
print("\n=== Verification ===")
print(f"num_cameras: {rec.num_cameras}")
print(f"num_images: {rec.num_images}")
print(f"num_rigs: {rec.num_rigs}")
print(f"num_frames: {rec.num_frames}")

print("\n=== SUCCESS ===")
