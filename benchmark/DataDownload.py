import pandas as pd
from pathlib import Path

RAWsuffixs = [
    "_binned_outputs.tar.gz",
    "_spatial.tar.gz",
    "_tissue_image.tif"
]

def main(prefix, metafile):
    prefix = Path(prefix)
    metaDF = pd.read_csv(metafile, sep="\t")
    for i in range(len(metaDF)):
        id = metaDF.loc[i,"Sample ID"]
        url = metaDF.loc[i,"Url"]
        ver = metaDF.loc[i,"Space Ranger Version"]
        home = prefix / id
        file_content = [f"# The data source is {url}\n"]
        file_content.extend(
            [
                f"curl -O https://cf.10xgenomics.com/samples/spatial-exp/{ver}/{id}/{id}{suffix}\n"
                for suffix in RAWsuffixs
            ]
        )
        home.mkdir(parents=True, exist_ok=True)
        with open(home/"download.sh", "w") as f:
            f.writelines(file_content)

if __name__ == "__main__":
    prefix = "." # path to download
    metafile = "./metadata.tsv"
    main(prefix, metafile)