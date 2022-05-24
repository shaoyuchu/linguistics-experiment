import logging
from os import getenv
from pathlib import Path
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from azure.storage.blob import ContainerClient, ContentSettings

conn_str = getenv("connection_string")
logger = logging.getLogger(__name__)


def verify_data(csv_data: str):
    return csv_data.startswith(
        '"success","trial_type","trial_index","time_elapsed","internal_node_id","view_history","rt","stimulus","response","task","correct_response","No","correct"'  # noqa: E501
    )


def export(csv_data: str):
    if not verify_data(csv_data):
        logger.error(f"Not a valid result. ({csv_data})")
        raise ValueError()

    container_client = ContainerClient.from_connection_string(conn_str, "data")
    current_time = datetime.now(tz=timezone(timedelta(hours=8)))
    file_name = (
        f"{current_time.strftime('%Y%m%d_%H%M_')}_{str(uuid4())[:4]}.csv"
    )
    url = upload_blob(csv_data, container_client, Path(file_name))
    logger.info(f"Successfully upload to {url} ({csv_data}).")


def upload_blob(
    content: str, container_client: ContainerClient, dst_path: Path
):
    blob = container_client.upload_blob(
        str(dst_path),
        content,
        content_settings=ContentSettings(content_type="text/csv"),
    )
    return blob.url
