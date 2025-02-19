import argparse
import open3d as o3d
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
import random
import pandas as pd
from multiprocessing import Pool

COLOR_MAP = {
    (255, 0, 0): 'beam',
    (0, 255, 0): 'column',
    (0, 0, 255): 'door',
    (255, 255, 0): 'floor',
    (0, 255, 255): 'roof',
    (64, 64, 64): 'wall',
    (150, 150, 150): 'wallagg',
    (255, 0, 255): 'window',
    (0, 0, 0): 'stairs',
    (50, 200, 150): 'clutter',
    (255, 128, 0): 'busbar',
    (153, 0, 0): 'cable',
    (127, 0, 255): 'duct',
    (255, 153, 255): 'pipe',
    (192, 192, 192): 'curvedagg',
    (30, 150, 200): 'elecequip',
    (215, 175, 30): 'fan',
    (200, 75, 30): 'fire',
    (35, 60, 125): 'fireequip',
    (80, 255, 130): 'lighting',
    (60, 35, 45): 'mechequip',
    (230, 255, 0): 'sprinklers',
    (255, 200, 0): 'streetlight',
    (140, 100, 170): 'switch',
    (150, 70, 50): 'intercom',
}
x_pts = None
y_pts = None
z_pts = None

def process_mod_value(args):
    input_file, output_dir, input_file_name, mod_value = args
    output_file = os.path.join(output_dir, f"{input_file_name}_mod_40{mod_value}.txt")
    if not os.path.exists(output_file):
        print("Generating", output_file)
        # Process the file
        with open(input_file, "r") as infile:
            lines = infile.readlines()
        with open(output_file, "w") as outfile:
            # Filter lines where the line number modulo 40 equals mod_value
            outfile.writelines(line for idx, line in enumerate(lines, start=1) if idx % 100 == mod_value)
    else:
        print(f"File {output_file} already exists. Skipping...")

def sub_sample_cloud(input_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    input_file_name = os.path.basename(input_file).split('.')[0]

    # Define the output filenames with modulo values
    modulo_values = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]

    # num_workers = min(8, len(modulo_values))  # Set the number of workers, limit to 4 or the length of modulo_values
    # with Pool(processes=num_workers) as pool:
    #     list(pool.map(process_mod_value, [(input_file, output_dir, input_file_name, mod_value) for mod_value in modulo_values]))
    for mod_value in modulo_values:
       process_mod_value((input_file, output_dir, input_file_name, mod_value))

def sample_file(files: list[str], num_files_to_merge: int, interval: int) -> list[str]:
    if x_pts is None or y_pts is None or z_pts is None:
        raise ValueError("Please run create_shadow_data() first to set the interval values")
    if num_files_to_merge > len(files):
        raise ValueError("Number of files to merge is greater than the number of files available")
    if num_files_to_merge < len(z_pts):
        print(f"WARNING..! Number of files to merge is less than the number of z intervals {z_pts}. May not produce good quality data")
        return random.sample(files, num_files_to_merge)
    else:
        selected_coords = []
        attempts = 0
        max_attempts = 10000  # To avoid infinite loop in case of impossible selection

        while len(selected_coords) < num_files_to_merge and attempts < max_attempts:
            attempts += 1
            x = random.choice(x_pts)
            y = random.choice(y_pts)
            z = random.choice(z_pts)
            coord = (x, y, z)

            # Check if the coordinate is adjacent to any already selected coordinate
            is_adjacent = False
            for selected in selected_coords:
                if abs(selected[0] - x) <= float(interval) and abs(selected[1] - y) <= float(interval) and abs(selected[2] - z) <= float(interval):
                    is_adjacent = True
                    break

            if not is_adjacent and coord not in selected_coords:
                selected_coords.append(coord)

        if len(selected_coords) < num_files_to_merge:
            raise ValueError("Unable to select the required number of non-adjacent coordinates")
        
        selected_files = []
        for file in files:
            for coord in selected_coords:
                if f"{coord[0]}_{coord[1]}_{coord[2]}.txt" in file:
                    print(f"Selected {file} for merging")
                    selected_files.append(file)

        return selected_files


def merge_shadow_clouds_randomly(input_dir: str, area_name: str, output_dir: str, interval:int, random_select_prop: float = 0.3, num_clouds_to_generate: int = 1):
    shadow_files = glob(input_dir + os.sep + area_name + os.sep + "*.txt")
    for i in range(num_clouds_to_generate):
        print(f"Generating {i+1}th Master point cloud")
        # selected_files = sample_file(shadow_files, num_files_to_merge, interval)
        num_files_to_merge = int(random_select_prop * len(shadow_files))
        selected_files = random.sample(shadow_files, num_files_to_merge)
        merge_dfs = []
        for selected_file in selected_files:
            df = pd.read_csv(selected_file, sep=" ", header=None, names=['x', 'y', 'z', 'r', 'g', 'b', 'label', 'element'])
            merge_dfs.append(df)
        final_df = pd.concat(merge_dfs, ignore_index=True)
        if i==0:
            output_path = os.path.join(output_dir, area_name)
        else:    
            output_path = os.path.join(output_dir, area_name+"_{}".format(i+1))
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

    global x_pts, y_pts, z_pts
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
        try:
            num_shadow_files = os.listdir(os.path.join(output_dir, area_name))
        except FileNotFoundError as e:
            num_shadow_files = None
        if num_shadow_files is None or len(num_shadow_files) < len(x_pts) * len(y_pts) * len(z_pts):
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
        else:
            print("Shadow data already exists for", area_name)

def get_file_name(file_path: str) -> str:
    return os.path.basename(file_path).split('.')[0]

def split_master_into_labels(master_dir: str):
    print("Creating label-wise files")
    areas = os.listdir(master_dir)
    for area in areas:
        master_file = os.path.join(master_dir, area, os.listdir(os.path.join(master_dir, area))[0])
        print("Processing", master_file)
        df = pd.read_csv(master_file, sep=" ", header=None, names=['x', 'y', 'z', 'r', 'g', 'b', 'label', 'element'])
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            output_dir = os.path.join(master_dir, area)
            label_df.to_csv(os.path.join(output_dir, area+f"_{label}.txt"), header=False, index=False, sep=' ')


if __name__ == '__main__':
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process point cloud data and generate shadow data.")
    parser.add_argument("-m", "--MASTER_ROOT", type=str, default="Master",
                        help="Root directory containing master data.")
    parser.add_argument("-o", "--shadow_output_dir", type=str, default="Shadow_Master",
                        help="Output directory for shadow data.")
    parser.add_argument("-a", "--areas", type=str, nargs='+', help="List of area names to process.")
    parser.add_argument("-r", "--randomness", type=float, default=0.3, help="Proportion of sub sampled files to select randomly to merge into final shadow master. Default is 0.3")
    parser.add_argument("-n", "--num_pcds", type=int, default=1, choices=range(1, 11), help="Number of Master point clouds to generate from the Shadow data of each area")
    parser.add_argument("-i", "--interval", type=int, default=40, choices=range(20, 201), help="Interval between camera points")
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
            create_shadow_data(area_name, str(os.path.join(sub_sampled_dir, area_name)), shadow_files_dir, interval=args.interval)
            end_time = time()
            shadow_elapsed_time = (end_time - start) / 60  # in minutes
            print("Shadow data of {} created in {:.2f} minutes".format(master_file, shadow_elapsed_time))
            print("Merging point clouds randomly")
            start = time()
            merge_shadow_clouds_randomly(shadow_files_dir, area_name, shadow_output_dir, args.interval, args.randomness, args.num_pcds)
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