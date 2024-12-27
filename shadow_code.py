import open3d as o3d
import numpy as np
import os
import glob
from tqdm import tqdm
from time import time
import subprocess
import pandas as pd

COLOR_MAP = {
    (255, 0, 0): 'beam',
    (0, 255, 0): 'column',
    (0, 0, 255): 'door',
    (255, 255, 0): 'floor',
    (0, 255, 255): 'roof',
    (64, 64, 64): 'wall',
    (30, 190, 192): 'wallagg',
    (255, 0, 255): 'window',
    (0, 0, 0): 'stairs',
    (50, 2000, 150): 'clutter',
    (255, 128, 0): 'busbar',
    (153, 0, 0): 'cable',
    (127, 0, 255): 'duct',
    (255, 153, 255): 'pipe',
    (192, 192, 192): 'curvedagg'
}

def sub_sample_cloud(input_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    input_file_name = os.path.basename(input_file).split('.')[0]

    # Define the output filenames with modulo values
    modulo_values = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]

    # Process the file
    with open(input_file, "r") as infile:
        lines = infile.readlines()

    # Write lines based on modulo conditions
    for mod_value in tqdm(modulo_values):
        output_file = os.path.join(output_dir, f"{input_file_name}_mod_40{mod_value}.txt")
        with open(output_file, "w") as outfile:
            # Filter lines where the line number modulo 40 equals mod_value
            outfile.writelines(line for idx, line in enumerate(lines, start=1) if idx % 100 == mod_value)


def create_shadow_data(area_name: str, sub_sampled_dir: str, output_dir: str, interval: int = 15, radius: int = 10000):
    def check_dimensions(pcd):
        size = pcd.get_max_bound() - pcd.get_min_bound()
        if size[0] // interval < 2.0 or size[1] // interval < 2.0 or size[2] // interval < 2.0:
            raise ValueError(
                "Building Dimension: {}. But dimension in any axis must be greater than twice the interval value: {}".format(
                    size, interval))
        else:
            return True

    def load_xyzrgb(filename):
        # Load data from a text file
        df = pd.read_csv(filename, sep=' ', header=None, names=['x', 'y', 'z', 'r', 'g', 'b', 'label', 'element'])
        df.drop(columns=['label', 'element'], inplace=True)
        data = df.to_numpy(dtype=np.float32)
        # data = np.loadtxt(filename)  # Assuming X Y Z R G B format
        points = data[:, :3]  # Extract XYZ coordinates
        colors = data[:, 3:6] / 255.0  # Extract RGB and normalize to [0, 1]

        # Create an Open3D PointCloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        return point_cloud

    def save_xyzrgb(point_cloud, filename):
        # Combine points and colors
        try:
            points = np.asarray(point_cloud.points)
            colors = np.asarray(point_cloud.colors) * 255.0  # Denormalize to [0, 255]
            data = np.hstack((points, colors))

            # Create a pandas DataFrame
            columns = ['x', 'y', 'z', 'r', 'g', 'b']
            df = pd.DataFrame(data, columns=columns)

            # Add 'label' column using apply
            label_function = lambda row: COLOR_MAP.get((int(row['r']), int(row['g']), int(row['b'])), 'unknown')
            df['label'] = df.apply(label_function, axis=1)

            # Add 'element' column
            df['element'] = df['label'] + '_element'

            # Display the DataFrame
            # print(df)

            # Save as a text file
            # np.savetxt(filename, data, fmt="%.6f %.6f %.6f %.0f %.0f %.0f")
            df.to_csv(filename, sep=' ', header=False, index=False)
        except Exception as e:
            print(e)

    pcds = []
    size_ok = False
    for sub_sampled_file in os.listdir(sub_sampled_dir):
        if not size_ok:
            size_ok = check_dimensions(
                o3d.io.read_point_cloud(os.path.join(sub_sampled_dir, sub_sampled_file), format='xyz',
                                        print_progress=True))
        if size_ok:
            pcds.append(load_xyzrgb(os.path.join(sub_sampled_dir, sub_sampled_file)))
            print("Loaded", sub_sampled_file)
    mod_val = len(os.listdir(sub_sampled_dir))
    if len(pcds) > 0:
        max = pcds[0].get_max_bound()
        min = pcds[0].get_min_bound()
        building_size = max - min
        print("Building Size [x, y, z]: {}".format(building_size))
        diameter = np.linalg.norm(np.asarray(pcds[0].get_max_bound()) - np.asarray(pcds[0].get_min_bound()))
        radius = 100 * diameter
        z_pts = np.array([5, 15])  # manual
        # z_pts = interval * np.arange(building_size[2] // interval)[1:]
        # y_pts = interval * np.arange(building_size[1] // interval)[1:]
        y_pts = np.array([30, 60, 90, 120, 150, 180, 210])
        # x_pts = interval * np.arange(building_size[0] // interval)[1:]
        x_pts = np.array([20])
        print("x intervals at ", x_pts)
        print("y intervals at ", y_pts)
        print("z intervals at ", z_pts)

        i = 0
        for cam_x in x_pts:
            for cam_y in y_pts:
                for cam_z in z_pts:
                    print("i=", i)
                    camera = [min[0] + cam_x, min[1] + cam_y, min[2] + cam_z]
                    print("camera at ", camera)
                    index = np.random.randint(mod_val)
                    print("index: ", index)
                    _, pt_map = pcds[index].hidden_point_removal(camera, radius)
                    pcd_new = pcds[index].select_by_index(pt_map)
                    os.makedirs(os.path.join(output_dir, area_name), exist_ok=True)
                    file_name = os.path.join(output_dir, area_name,
                                             "{}_{}_{}_{}.txt".format(area_name, cam_x, cam_y, cam_z))
                    print("Writing", file_name)
                    # o3d.io.write_point_cloud(file_name, pcd_new)
                    save_xyzrgb(pcd_new, file_name)
                    i = i + 1


def get_file_name(file_path: str) -> str:
    return os.path.basename(file_path).split('.')[0]


def get_max_min_dim(file_path: str, dim_size=0.01):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, skiprows=1, encoding="gbk", engine='python', sep=' ', delimiter=None,
                         index_col=False, header=None, skipinitialspace=True)

        xMin = df[0].min()
        xMax = df[0].max()
        xDim = xMax - xMin
        xMinLimit = xMin - (xDim * dim_size)
        xMaxLimit = xMax + (xDim * dim_size)

        yMin = df[1].min()
        yMax = df[1].max()
        yDim = yMax - yMin
        yMinLimit = yMin - (yDim * dim_size)
        yMaxLimit = yMax + (yDim * dim_size)

        zMin = df[2].min()
        zMax = df[2].max()
        zDim = zMax - zMin
        zMinLimit = zMin - (zDim * dim_size)
        zMaxLimit = zMax + (zDim * dim_size)

        # print(xMinLimit, xMaxLimit, yMinLimit, yMaxLimit, zMinLimit, zMaxLimit)
        limits = []
        limits.append(xMinLimit)
        limits.append(xMaxLimit)
        limits.append(yMinLimit)
        limits.append(yMaxLimit)
        limits.append(zMinLimit)
        limits.append(zMaxLimit)
        suffix = file_path + '.txt'
        with open(suffix, 'w') as file:
            for limit in limits:
                file.write(str(limit))
                file.write(' ')


def extract_pcd_from_mesh(area_name: str, fbx_dir: str, fbx_classes: list, temp_folder: str, output_dir: str):
    # Define paths
    cloud_compare_path = r"c:\Program Files\CloudCompare\CloudCompare.exe"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.makedirs(temp_folder, exist_ok=True)
    for fbx_class in fbx_classes:
        print("Extracting from ", fbx_class)
        # Loop through all .fbx files in the input folder
        fbx_files = glob.glob(os.path.join(fbx_dir, fbx_class, "*.fbx"))
        if len(fbx_files) > 0:
            print("Found {} fbx files for class {}".format(len(fbx_files), fbx_class))
            for fbx_file in tqdm(fbx_files):
                file_name = get_file_name(fbx_file)
                # Generate intermediate filenames
                merged_bin = os.path.join(temp_folder, file_name + "_merged.bin")
                csv_file = os.path.join(temp_folder, file_name + ".csv")

                # Step 1: Sample mesh and generate merged binary
                subprocess.run([
                    cloud_compare_path, "-silent", "-auto_save", "off", "-o", fbx_file,
                    "-sample_mesh", "density", "500", "-merge_clouds", "-save_clouds",
                    "file", merged_bin, "-clear", "-o", merged_bin,
                    "-c_export_fmt", "asc", "-add_header", "-ext", "csv", "-save_clouds", "file", csv_file
                ])

                get_max_min_dim(csv_file)

                # Read the modified CSV file and process tokens
                txt_file = f"{csv_file}.txt"
                if os.path.exists(txt_file):
                    with open(txt_file, "r") as file:
                        for line in file:
                            tokens = line.split()  # Split line into tokens
                            if len(tokens) >= 6:
                                a, b, c, d, e, f = tokens[:6]

                                # Step 3: Crop and save cropped binary
                                cropped_bin = os.path.join(temp_folder, file_name + "_cropped.bin")
                                consolidated_bin = area_name + "_consolidated.bin"
                                subprocess.run([
                                    cloud_compare_path, "-silent", "-auto_save", "off",
                                    "-o", consolidated_bin, "-crop",
                                    f"{a}:{c}:{e}:{b}:{d}:{f}", "-save_clouds", "file", cropped_bin
                                ])

                                # Step 4: Export cropped binary to .pts format
                                output_path = os.path.join(output_dir, area_name + ".pts")
                                subprocess.run([
                                    cloud_compare_path, "-silent", "-auto_save", "off",
                                    "-o", cropped_bin, "-c_export_fmt", "asc",
                                    "-add_header", "-ext", "pts", "-save_clouds", "file", output_path
                                ])


MASTER_ROOT = "Master"
sub_sampled_dir = "sub_sampled"
shadow_output_dir = "Shadow"
# final_output_dir = "Shadow"
# fbx_dir = "Fbx_Floor"
# fbx_classes = ['floor', 'beam', 'roof', 'wall', 'column', 'door', 'window', 'busbar', 'cable', 'duct', 'pipe']

AREAS = os.listdir(MASTER_ROOT)
master_files = []
for area in AREAS:
    files = os.listdir(os.path.join(MASTER_ROOT, area))
    for master_file in files:
        if master_file == area + '.txt':
            print(master_file)
            # subprocess.run([r"C:\Program Files\CloudCompare\CloudCompare.exe", "-VERBOSITY", "1", "-auto_save", "off",
            #                 "-o", os.path.join(MASTER_ROOT, area, master_file), "-save_clouds", "file",
            #                 f"{area}_consolidated.bin"])
            master_files.append(os.path.join(MASTER_ROOT, area, master_file))
print(f"Found {len(master_files)} files from {len(AREAS)} areas in {MASTER_ROOT}")

failures = []
for master_file in master_files:
    start_time = time()
    area_name = get_file_name(master_file)
    print("Subsampling", master_file)
    sub_sample_cloud(master_file, str(os.path.join(sub_sampled_dir, area_name)))
    print("Creating shadow pcd of", master_file)
    try:
        create_shadow_data(area_name, str(os.path.join(sub_sampled_dir, area_name)), shadow_output_dir)
        end_time = time()
        elapsed_time = (end_time - start_time) / 60  # in minutes
        print("Shadow data of {} created in {} minutes".format(master_file, elapsed_time))
    except Exception as e:
        failures.append(master_file)
        continue
    # print("Creating final Shadow data of {} with RGB fields".format(master_file))
    # extract_pcd_from_mesh(area_name, fbx_dir, fbx_classes, "temp_data", final_output_dir)
if len(failures) > 0:
    print("Failed to create shadow data for")
    for failure in failures:
        print(failure)
