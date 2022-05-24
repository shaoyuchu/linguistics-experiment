import logging

import azure.functions as func
from utils.export import export


def main(req: func.HttpRequest) -> func.HttpResponse:
    logger = logging.getLogger(__name__)

    try:
        req_body = req.get_json()
        data = req_body["data"]
        logger.info("data: %s", str(data))
    except ValueError:
        return func.HttpResponse(status_code=500)

    export(data)
    return func.HttpResponse(status_code=200)
