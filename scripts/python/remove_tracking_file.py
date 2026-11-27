"""Remove tracking file."""

import sys

from qgsw.cli import parse_output_folder, parse_run_id
from qgsw.logging.core import getLogger
from qgsw.logging.utils import box
from qgsw.scripts.output import get_tracking_filename, retrieve_output_folder

logger = getLogger(__name__)

if __name__ == "__main__":
    output_folder = parse_output_folder()
    run_id = parse_run_id()

    try:
        output = retrieve_output_folder(
            output_folder,
            run_id,
        )
    except FileNotFoundError:
        msg = "ERROR: Not output file available."
        logger.critical(msg)
        sys.exit(1)
    if run_id is not None:
        tracking_file = get_tracking_filename(run_id)
        tracking_file.unlink(missing_ok=True)
        msg = "Tracking file removed."
        logger.info(box(msg))
