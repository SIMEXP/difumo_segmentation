#!/usr/bin/python3

import os
import argparse
import nibabel as nib
import nilearn
import nilearn.datasets
import nilearn.regions
import nilearn.plotting
from scipy import sparse, ndimage
from sklearn.utils.extmath import safe_sparse_dot
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


def compute_template_sparse(map_img, template_dir):
    # https://github.com/Parietal-INRIA/DiFuMo/blob/361d2f0515c12fda36dcf9d3ba56c9c626e7e01b/region_labeling/brain_masks_overlaps.py#L25-L33
    icbm = nilearn.datasets.fetch_icbm152_2009(data_dir=template_dir)
    masks = nilearn.image.load_img([icbm.gm, icbm.wm, icbm.csf])
    masks = nilearn.image.resample_to_img(masks, map_img)
    m = masks.get_data()
    m = m.reshape([-1, 3])
    m = sparse.csr_matrix(m)

    return m


def compute_ratios(extracted_regions_img, template_mask_sparse):
    # https://github.com/Parietal-INRIA/DiFuMo/blob/master/region_labeling/brain_masks_overlaps.py
    dd = extracted_regions_img.get_data()
    data = dd.reshape([-1, dd.shape[-1]]).T
    data = sparse.csr_matrix(data)
    overlaps = safe_sparse_dot(data, template_mask_sparse, dense_output=True)

    return overlaps


def write_labels(input_labels_path, output_labels_path, regions_idx, gm_wm_csf_ratios):
    # load label file
    with open(input_labels_path, 'r') as f:
        input_labels = f.readlines()
    # write new labels using extracted region indexes
    output_labels = [input_labels[0].rstrip()]
    for ii, region_idx in enumerate(regions_idx):
        # re-compute matter ratios
        # https://github.com/Parietal-INRIA/DiFuMo/blob/master/region_labeling/brain_masks_overlaps.py
        curr_region_metadatas = input_labels[region_idx + 1].split(",")
        curr_region_metadatas[0] = str(ii + 1)
        for jj in range(3):
            curr_region_metadatas[-(3 - jj)] = str(gm_wm_csf_ratios[ii, jj])
        curr_region_metadata = ",".join(curr_region_metadatas)
        output_labels += [curr_region_metadata]
    with open(output_labels_path, 'w') as f:
        f.write("\n".join(output_labels))


def main():
    # TODO compute pixel matter ratio for labels
    # TODO visualization
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

    template_mask_sparse = None
    for dimension in dimensions:
        for resolution in resolutions:
            # download difumo atlas using nilearn
            atlas = nilearn.datasets.fetch_atlas_difumo(
                dimension=dimension, resolution_mm=resolution, data_dir=difumo_path)
            # extract independent regions from difumo
            extractor = nilearn.regions.RegionExtractor(maps_img=atlas.maps)
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
            # computing matter ratios
            if template_mask_sparse is None:
                template_mask_sparse = compute_template_sparse(atlas.maps, difumo_path)
            gm_wm_csf_ratios = compute_ratios(extracted_regions_img, template_mask_sparse)
            # writing dictionnary labels
            input_labels_path = os.path.join(
                difumo_path, "difumo_atlases", f"{dimension}", f"labels_{dimension}_dictionary.csv")
            curr_labels_path = os.path.join(output_path, "segmented_difumo_atlases",
                                            f"{num_extracted_regions}", f"labels_{num_extracted_regions}_dictionary.csv")
            write_labels(input_labels_path, curr_labels_path,
                         regions_idx, gm_wm_csf_ratios)


if __name__ == '__main__':
    main()
