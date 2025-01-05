import argparse
import open3d as o3d
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
import random
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
    (50, 200, 150): 'clutter',
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


def merge_shadow_clouds_randomly(input_dir: str, area_name: str, output_dir: str, num_files_to_merge: int = 5):
    shadow_files = glob(input_dir + os.sep + area_name + os.sep + "*.txt")
    selected_files = random.sample(shadow_files, num_files_to_merge)
    merge_dfs = []
    for selected_file in selected_files:
        df = pd.read_csv(selected_file, sep=" ", header=None, names=['x', 'y', 'z', 'r', 'g', 'b', 'label', 'element'])
        merge_dfs.append(df)
    final_df = pd.concat(merge_dfs, ignore_index=True)
    output_path = os.path.join(output_dir, area_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    final_df.to_csv(os.path.join(output_path, f"{area_name}.txt"), header=False, index=False, sep=' ')


def create_shadow_data(area_name: str, sub_sampled_dir: str, output_dir: str, interval: int = 40, radius: int = 10000):
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
            colors = colors.astype(np.uint8)
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
    size_ok = True
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
        z_pts = np.array([5, 15, 25, 35])  # manual
        # z_pts = interval * np.arange(building_size[2] // interval)[1:]
        y_pts = interval * np.arange(building_size[1] // interval)[1:]
        # y_pts = np.array([30, 60, 90, 120, 150, 180, 210])
        x_pts = interval * np.arange(building_size[0] // interval)[1:]
        # x_pts = np.array([20])
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

def split_master_into_labels(master_dir: str):
    print("Creating label-wise files")
    areas = os.listdir(master_dir)
    for area in areas:
        master_file = os.path.join(master_dir, area, area+".txt")
        print("Processing", master_file)
        df = pd.read_csv(master_file, sep=" ", header=None, names=['x', 'y', 'z', 'r', 'g', 'b', 'label', 'element'])
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            output_dir = os.path.join(master_dir, area)
            label_df.to_csv(os.path.join(output_dir, area+f"_{label}.txt"), header=False, index=False, sep=' ')


if __name__ == '__main__':
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process point cloud data and generate shadow data.")
    parser.add_argument("-r", "--MASTER_ROOT", type=str, default="Master",
                        help="Root directory containing master data.")
    parser.add_argument("-o", "--shadow_output_dir", type=str, default="Shadow_Master",
                        help="Output directory for shadow data.")
    parser.add_argument("-a", "--areas", type=str, nargs='+', help="List of area names to process.")
    args = parser.parse_args()

    MASTER_ROOT = args.MASTER_ROOT
    shadow_output_dir = args.shadow_output_dir
    sub_sampled_dir = "sub_sampled"
    shadow_files_dir = "Shadow"

    areas = args.areas
    master_files = []
    for area in areas:
        try:
            files = os.listdir(os.path.join(MASTER_ROOT, area))
            for master_file in files:
                if master_file == area + '.txt':
                    master_files.append(os.path.join(MASTER_ROOT, area, master_file))
        except FileNotFoundError as e:
            print(e)
            continue
    print(f"Found {len(master_files)} files from {len(areas)} areas in {MASTER_ROOT}")

    failures = []
    for master_file in master_files:
        area_name = get_file_name(master_file)
        print("Subsampling", master_file)
        start = time()
        sub_sample_cloud(master_file, str(os.path.join(sub_sampled_dir, area_name)))
        sampling_time = (time() - start)/60
        print(f"Subsampling completed in {sampling_time:.2f} minutes")
        print("Creating shadow pcd of", master_file)
        try:
            start = time()
            create_shadow_data(area_name, str(os.path.join(sub_sampled_dir, area_name)), shadow_files_dir, interval=40)
            end_time = time()
            shadow_elapsed_time = (end_time - start) / 60  # in minutes
            print("Shadow data of {} created in {:.2f} minutes".format(master_file, shadow_elapsed_time))
            print("Merging point clouds randomly")
            start = time()
            merge_shadow_clouds_randomly(shadow_files_dir, area_name, shadow_output_dir)
            merge_elapsed_time = (time() - start)/60
            print("Merging completed in {:.2f} minutes".format(merge_elapsed_time))
        except Exception as e:
            print(e)
            failures.append(master_file)
            continue
    if len(failures) > 0:
        print("Failed to create shadow data for")
        for failure in failures:
            print(failure)
    else:
        split_master_into_labels(shadow_output_dir)
    print("Process completed in {:.2f} minutes".format(sampling_time + shadow_elapsed_time + merge_elapsed_time))