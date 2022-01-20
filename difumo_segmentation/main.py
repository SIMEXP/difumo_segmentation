#!/usr/bin/python3

import os
import argparse
import nibabel as nib
import nilearn
import nilearn.datasets
import nilearn.regions
import nilearn.plotting
import utils.utils as utils


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="", epilog="""
    Documentation at https://github.com/SIMEXP/difumo_segmentation
    """)

    parser.add_argument(
        "-i", "--input-path", required=False, help="Input difumo path (default: \"./data/raw\")",
    )

    parser.add_argument(
        "-o", "--output-path", required=False, help="Output segmented difumo path (default: \"./data/processed\")",
    )

    parser.add_argument(
        "-d", "--dim", required=False, default=-1, help=""
        "Number of dimensions in the dictionary. Valid resolutions available are {64, 128, 256, 512, 1024}, -1 for all (default: -1)",
    )

    parser.add_argument(
        "-r", "--res", required=False, default=-1, help=""
        "The resolution in mm of the atlas to fetch. Valid options available are {2, 3}, -1 for all (default: -1)",
    )

    parser.add_argument(
        "--version", action="version", version=utils.get_version()
    )

    return parser


def write_labels(input_labels_path, output_labels_path, regions_idx):
    # load label file
    with open(input_labels_path, 'r') as f:
        input_labels = f.readlines()
    # write new labels using extracted region indexes
    output_labels = [input_labels[0].rstrip()]
    for ii, region_idx in enumerate(regions_idx):
        # re-compute matter ratios
        # https://github.com/Parietal-INRIA/DiFuMo/blob/master/region_labeling/brain_masks_overlaps.py
        curr_region_metadata = str(ii + 1) + input_labels[region_idx + 1][1:]
        curr_region_metadata = ",".join(curr_region_metadata.split(",")[:-3])
        output_labels += [curr_region_metadata]
    with open(output_labels_path, 'w') as f:
        f.write("\n".join(output_labels))


def main():
    difumo_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    output_path = os.path.join(os.path.dirname(
        __file__), "..", "data", "processed")
    dimensions = [64, 128, 256, 512, 1024]
    resolutions = [2, 3]
    args = get_parser().parse_args()
    if args.input_path is not None:
        difumo_path = args.input_path
    if args.output_path is not None:
        output_path = args.output_path
    if not (args.dim == -1):
        dimensions = [int(args.dim)]
    if not (args.res == -1):
        resolutions = [int(args.res)]
    print(args)

    for dimension in dimensions:
        for resolution in resolutions:
            # download difumo atlas using nilearn
            atlas = nilearn.datasets.fetch_atlas_difumo(
                dimension=dimension, resolution_mm=resolution, data_dir=difumo_path)
            # extract independent regions from difumo
            extractor = nilearn.regions.RegionExtractor(maps_img=atlas['maps'])
            extractor.fit()
            extracted_regions_img = extractor.regions_img_
            regions_idx = extractor.index_
            num_extracted_regions = len(regions_idx)
            # saving atlas using original atlaas affine and header
            output_res_path = os.path.join(
                output_path, "segmented_difumo_atlases", f"{num_extracted_regions}", f"{resolution}")
            os.makedirs(output_res_path, exist_ok=True)
            curr_atlas_path = os.path.join(output_res_path, "maps.nii.gz")
            nib.save(extracted_regions_img, curr_atlas_path)
            # computing dictionnary labels
            input_labels_path = os.path.join(
                difumo_path, "difumo_atlases", f"{dimension}", f"labels_{dimension}_dictionary.csv")
            curr_labels_path = os.path.join(output_path, "segmented_difumo_atlases",
                                            f"{num_extracted_regions}", f"labels_{num_extracted_regions}_dictionary.csv")
            write_labels(input_labels_path, curr_labels_path, regions_idx)


if __name__ == '__main__':
    main()
