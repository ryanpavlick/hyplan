import requests
import logging
import os

logger = logging.getLogger(__name__)

def download_file(filepath: str, url: str, chunk_size: int = int(1E6), timeout: int = 30, replace: bool = False) -> None:
    """
    Download a file from the specified URL if it does not exist or if `replace` is True.

    Args:
        filepath (str): The path where the downloaded file will be saved.
        url (str): The URL of the file to download.
        chunk_size (int, optional): The size of each chunk in bytes. Default is 1 MB.
        timeout (int, optional): The timeout in seconds for the request. Default is 30 seconds.
        replace (bool, optional): Whether to replace the file if it already exists. Default is False.
    """
    directory = os.path.dirname(filepath) or "."  # Default to current directory if no directory path is given
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

    # Check if the file already exists
    if os.path.exists(filepath) and not replace:
        logger.info(f"The file at {filepath} already exists. Skipping download.")
        return

    # Download to a temp file first, then rename atomically.
    # This prevents partial/corrupt files from persisting if the
    # download is interrupted.
    tmp_path = filepath + ".tmp"
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    file.write(chunk)
        os.replace(tmp_path, filepath)
        logger.info(f"Data downloaded successfully to {filepath}.")
    except requests.RequestException as e:
        # Clean up partial temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        logger.error(f"Error downloading data from {url}: {e}")
        raise
