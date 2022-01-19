#!/usr/bin/python3

import os
import argparse
import nilearn
import nilearn.datasets
import nilearn.regions
import nilearn.plotting
import utils.utils as utils


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="", epilog="""
    Documentation at https://github.com/ccna-biomarkers/ccna_qc_summary
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
            img_extracted_regions = extractor.regions_img_
            index_regions = extractor.index_
            # nilearn.save(img_extracted_regions)
            # TODO: save index using original region name, and index_regions

if __name__ == '__main__':
    main()
