import os
import requests
import tarfile

# Paths
downloaded_path = "downloaded/"
extracted_path = "data/"

# Dataset parts to download
protocols = {
    "P1E": ["S1", "S2", "S3", "S4"],
    "P1L": ["S1", "S2", "S3", "S4"],
    "P2E": ["S1", "S2", "S3", "S4", "S5"],
    "P2L": ["S1", "S2", "S3", "S4", "S5"],
}


def download_tar_xz(url: str, dest_file: str):
    """
    Download a tar.xz file.

    Args:
        url (str): URL to download.
        dest_file (str): Destination file.
    """
    # Check if already downloaded
    if os.path.exists(dest_file):
        print(f"The file '{dest_file}' already exists. Skipping download.")
        return

    # Download file
    print(f"Downloading {url}...")
    r = requests.get(url, allow_redirects=True)
    open(dest_file, 'wb').write(r.content)


def download_dataset():
    """
    Download the dataset from the GitHub repository and extract it.
    """
    # Download dataset
    os.makedirs(downloaded_path, exist_ok=True)
    print("#"*5 + " Downloading dataset " + "#"*5)
    for key in protocols.keys():
        # Download face photos
        url = f"https://github.com/davegabe/suspect-identification-cctv/releases/download/dataset/{key}.tar.xz"
        download_tar_xz(url, os.path.join(downloaded_path, f"{key}.tar.xz"))

        # Download video
        for scenario in protocols[key]:
            file = f"{key}_{scenario}"
            url = f"https://github.com/davegabe/suspect-identification-cctv/releases/download/dataset/{file}.tar.xz"
            download_tar_xz(url, os.path.join(downloaded_path, f"{file}.tar.xz"))


def extract_faces(archive: str, dest_dir: str):
    """
    Extract faces from a tar.xz archive.

    Args:
        archive (str): Path to the tar.xz archive.
        dest_dir (str): Destination directory.
    """
    # Check if already extracted
    if os.path.exists(dest_dir):
        print(f"The directory '{dest_dir}' already exists. Skipping extraction.")
        return

    # Extract faces
    print(f"Extracting faces from {archive}...")
    tar = tarfile.open(archive, "r:xz")
    tar.extractall(dest_dir)
    tar.close()


def extract_video(archive: str, dest_dir: str):
    """
    Extract every tar.xz archives inside the main tar.xz archive.

    Args:
        archive (str): Path to the tar.xz archive.
        dest_dir (str): Destination directory.
    """
    # Extract video
    print(f"Extracting video from {archive}...")
    tar = tarfile.open(archive, "r:xz")
    for member in tar.getmembers():
        # Extract every tar.xz archives
        if member.name.endswith(".tar.xz"):
            # Check if already extracted
            dest_sub_dir = os.path.join(dest_dir, member.name.split(".")[0])
            if os.path.exists(dest_sub_dir):
                print(f"The directory '{dest_sub_dir}' already exists. Skipping extraction.")
                continue

            # Extract
            f = tar.extractfile(member)
            sub_tar = tarfile.open(fileobj=f)
            sub_tar.extractall(dest_dir)
            sub_tar.close()
    tar.close()


def extract_dataset():
    """
    Extract downloaded files.
    """
    # Extract dataset
    os.makedirs(extracted_path, exist_ok=True)
    print("#"*5 + " Extracting dataset " + "#"*5)
    for key in protocols.keys():
        # Extract face photos
        file = os.path.join(downloaded_path, f"{key}.tar.xz")
        extract_faces(file, os.path.join(extracted_path, key+"_faces"))

        # Extract video
        for scenario in protocols[key]:
            file = os.path.join(downloaded_path, f"{key}_{scenario}.tar.xz")
            extract_video(file, os.path.join(extracted_path, key))


if __name__ == "__main__":
    download_dataset()
    extract_dataset()
