"""Utility functions used by multiple trees
"""

def get_ids(study_folder: str) -> list:
    """Get subfolders in directory, excluding registry and hidden folders

    Args:
        study_folder(str): Filepath to the folder containing desired
            subdirectories. Should be an absolute filepath

    Returns:
        List of subdirectories of the study_folder.
    """
    list_of_dirs = []
    for subdir in os.listdir(study_folder):
        is_folder = os.path.isdir(os.path.join(study_folder, subdir))
        if (not subdir.startswith(".")) and subdir != "registry" and is_folder:
            list_of_dirs.append(subdir)
    return list_of_dirs
