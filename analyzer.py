
from pathlib import Path
from typing import List, Dict, Tuple

from anndata import AnnData

from datasets import rawData, VisiumData, VisiumHDData
from profiles import Profile, VisiumProfile, VisiumHDProfile
from models import iStar, ImSpiRE
from run_in_conda import run_command_in_conda_env

class Pipeline:

    def run(self):
        visium_path = Path('/home/yiriso/Research/Super-resolvedST/data/DLPFC/sample_151673')
        iStar_visium = iStar()
        # iStar_visium = ImSpiRE()
        visium_profile = VisiumProfile(slide_serial=1)

        iStar_visium.load(
            path=visium_path,
            profile=visium_profile,
            source_image_path=visium_path/"151673_full_image.tif"
        )

        # iStar_visium.image = iStar_visium.vision_spots(in_tissue=True)
        iStar_visium.tissue_mask(mask_image_path=visium_path/"mask.png")
        iStar_visium.select_HVG(n_top_genes=2000)

        super_pixel_size = 8
        HD_profile = VisiumHDProfile(bin_size=super_pixel_size)

        visiumHD_demo = iStar_visium.Visium2HD(HDprofile=HD_profile, quiet=True)
        iStar_visium.set_target_VisiumHD(visiumHD_demo)

        istarHD_dir = Path(f'test/istarHD_DLPFC_{super_pixel_size:03}')
        # istarHD_dir = Path(f'test/ImSpiREHD_DLPFC_{super_pixel_size:03}')
        iStar_visium.save_input(istarHD_dir)

        run_command_in_conda_env(
            'iStar',
            f'./Run-iStar.sh {istarHD_dir.resolve()}/',
            f'{istarHD_dir.resolve()}/istar.log'
        )

        # run_command_in_conda_env(
        #     'imspire',
        #     f'./Run-ImSpiRE.sh {istarHD_dir.resolve()}/',
        #     f'{istarHD_dir.resolve()}/ImSpiRE.log'
        # )

        iStar_visium.load_output(istarHD_dir)
        iStar_visium.to_VisiumHD(istarHD_dir/"VisiumHD_result")


