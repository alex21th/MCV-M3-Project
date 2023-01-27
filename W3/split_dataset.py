from src.utils import generate_image_patches_db


if __name__ == "__main__":
    # user defined variables
    PATCH_SIZE = 64
    DATASET_DIR = '../data/MIT_split'
    PATCHES_DIR = '../data/MIT_split_patches'

    generate_image_patches_db(DATASET_DIR, PATCHES_DIR, patch_size=PATCH_SIZE)
