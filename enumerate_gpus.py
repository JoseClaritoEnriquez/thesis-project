import openvino as ov
core = ov.Core()
devices = core.available_devices
print(f"Available Devices: {devices}")

for d in devices:
    try:
        print(f"--- Properties for {d} ---")
        print(f"  Name: {core.get_property(d, 'FULL_DEVICE_NAME')}")
    except:
        pass

# Try to enumerate GPU instances directly
for i in range(10):
    try:
        id = f"GPU.{i}"
        name = core.get_property(id, "FULL_DEVICE_NAME")
        print(f"Found {id}: {name}")
    except:
        pass
