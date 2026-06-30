import os
from collections import defaultdict

BASE_DIR = r"D:\gs\angiogenesis data"

def scan_directory(base_dir):
    data_map = defaultdict(lambda: defaultdict(list))
    
    # We walk the directory
    for root, dirs, files in os.walk(base_dir):
        # We check if there are TIF or JPG images in this directory
        image_files = [f for f in files if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.bmp'))]
        if not image_files:
            continue
            
        # Parse path relative to base_dir
        rel_path = os.path.relpath(root, base_dir)
        parts = rel_path.split(os.sep)
        
        # Structure is generally: [SN_Series]\[Hour]\[Concentration]...
        if len(parts) >= 3:
            series = parts[0]
            hour = parts[1]
            concentration = parts[2]
            
            # Normalize concentration name (e.g. "1 ug" -> "1ug", "10 ug" -> "10ug")
            norm_conc = concentration.replace(" ", "").lower()
            
            # Normalize hour (e.g. "0" -> "0h", "2" -> "2h")
            norm_hour = hour.lower()
            if not norm_hour.endswith('h') and norm_hour.isdigit():
                norm_hour = norm_hour + "h"
                
            key = (norm_conc, norm_hour)
            data_map[norm_conc][norm_hour].extend([os.path.join(root, f) for f in image_files])

    print("\n=== Angiogenesis Raw Data Structure ===")
    for conc, hours in sorted(data_map.items()):
        print(f"\nConcentration: {conc}")
        for hr, paths in sorted(hours.items(), key=lambda x: int(x[0].replace('h', '')) if x[0].replace('h', '').isdigit() else 999):
            print(f"  Time Point: {hr} -> {len(paths)} images")

if __name__ == "__main__":
    scan_directory(BASE_DIR)
