import os
import urllib.request
import zipfile
import subprocess
import time

packages = [
    ("Google.Protobuf", "3.21.12", "lib/net45/Google.Protobuf.dll"),
    ("Grpc.Core", "2.46.6", "lib/net45/Grpc.Core.dll"),
    ("Grpc.Core.Api", "2.46.6", "lib/net45/Grpc.Core.Api.dll"),
    ("System.Memory", "4.5.3", "lib/netstandard2.0/System.Memory.dll"),
    ("System.Runtime.CompilerServices.Unsafe", "4.5.2", "lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"),
    ("System.Buffers", "4.4.0", "lib/netstandard2.0/System.Buffers.dll"),
    ("System.Interactive.Async", "3.2.0", "lib/net45/System.Interactive.Async.dll")
]

nt8_custom_dir = os.path.expanduser(r"~\Documents\NinjaTrader 8\bin\Custom")
os.makedirs(nt8_custom_dir, exist_ok=True)

print("Đang tải thư viện C# (DLLs) từ NuGet...")
for pkg_name, version, dll_path in packages:
    url = f"https://www.nuget.org/api/v2/package/{pkg_name}/{version}"
    zip_path = f"{pkg_name}.zip"
    
    print(f"Downloading {pkg_name} v{version}...")
    urllib.request.urlretrieve(url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        dll_filename = f"{pkg_name}.dll"
        actual_path = None
        for name in z.namelist():
            if name.endswith(dll_filename) and ("net45" in name or "netstandard2.0" in name):
                actual_path = name
                break
        
        if not actual_path:
            for name in z.namelist():
                if name.endswith(dll_filename):
                    actual_path = name
                    break

        if actual_path:
            with z.open(actual_path) as source:
                target_path = os.path.join(nt8_custom_dir, dll_filename)
                with open(target_path, "wb") as f:
                    f.write(source.read())
                print(f" -> Đã giải nén: {target_path}")
        else:
            print(f" -> KHÔNG TÌM THẤY {dll_filename} TRONG {pkg_name}.zip")

# Download grpc_csharp_ext.x64.dll cho Grpc.Core
print("Downloading Grpc Native Extension...")
urllib.request.urlretrieve("https://www.nuget.org/api/v2/package/Grpc.Core/2.46.6", "grpc_core.zip")
with zipfile.ZipFile("grpc_core.zip", 'r') as z:
    with z.open("runtimes/win-x64/native/grpc_csharp_ext.x64.dll") as source:
        target_path = os.path.join(nt8_custom_dir, "grpc_csharp_ext.x64.dll")
        with open(target_path, "wb") as f:
            f.write(source.read())
        print(f" -> Đã giải nén native: {target_path}")

# Generate C# protobuf
print("Downloading Grpc.Tools for protoc plugin...")
tools_zip = "grpc_tools.zip"
urllib.request.urlretrieve("https://www.nuget.org/api/v2/package/Grpc.Tools/2.46.6", tools_zip)

proto_out_dir = os.path.join(nt8_custom_dir, "Nt8Bridge")
os.makedirs(proto_out_dir, exist_ok=True)

with zipfile.ZipFile(tools_zip, 'r') as z:
    z.extract("tools/windows_x64/protoc.exe", ".")
    z.extract("tools/windows_x64/grpc_csharp_plugin.exe", ".")

protoc_exe = os.path.abspath("tools/windows_x64/protoc.exe")
plugin_exe = os.path.abspath("tools/windows_x64/grpc_csharp_plugin.exe")

print("Generating C# Protobuf files...")
cmd = [
    protoc_exe,
    "-I", "protos",
    f"--csharp_out={proto_out_dir}",
    f"--grpc_out={proto_out_dir}",
    f"--plugin=protoc-gen-grpc={plugin_exe}",
    "protos/nt8_bridge.proto"
]
subprocess.run(cmd, check=True)

print("Thành công! Toàn bộ DLL và C# Protobuf đã vào NinjaTrader.")
