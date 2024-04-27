import pandas as pd
import os
import sys
from src.logger import logging
from scipy.stats import ks_2samp

from src.utility_file import Utility
from src.utility_yaml_file import read_yaml_file, write_yaml_file

params = Utility().read_params()


class DataValidation:

    def __init__(self) -> None:
        pass

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise e

    def validate(self):
        try:
            status = True
            report = {}
            threshold = params["basic"]["THRESHOLD"]
            # print(threshold)
            logging.info("Loading schema file")
            SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

            schema_config = read_yaml_file(SCHEMA_FILE_PATH)

            number_of_columns = len(schema_config["columns"])

            # reading paths
            artifact_dir = params["DATA_LOCATION"]["ARTIFACT_DIR"]
            model_dir = params["DATA_LOCATION"]["main_model_folder"]
            raw_data_path = params["DATA_LOCATION"]["RAW_FILE_NAME"]
            train_data_path = params["DATA_LOCATION"]["TRAIN_FILE_NAME"]
            test_data_path = params["DATA_LOCATION"]["TEST_FILE_NAME"]
            # reading data artifacts

            raw_data = DataValidation.read_data(
                os.path.join(artifact_dir, str(raw_data_path))
            )
            train_data = DataValidation.read_data(
                os.path.join(artifact_dir, str(train_data_path))
            )
            test_data = DataValidation.read_data(
                os.path.join(artifact_dir, str(test_data_path))
            )

            logging.info(f"Schema files number of columns: {number_of_columns}")
            logging.info(f"Data frame columns for raw file: {len(raw_data.columns)}")
            logging.info(
                f"Data frame columns for train file: {len(train_data.columns)}"
            )
            logging.info(f"Data frame columns for test file: {len(test_data.columns)}")

            if (
                len(raw_data.columns) & len(train_data.columns) & len(test_data.columns)
                == number_of_columns
            ):
                is_found = True
                train_path = train_data.columns
                test_path = test_data.columns
                train_test_set = ks_2samp(train_path, test_path)

                logging.info(f"{train_test_set}")
                # print(train_test_set.pvalue)
                if threshold <= train_test_set.pvalue:
                    report.update(
                        {
                            "p-value": float(train_test_set.pvalue),
                            "drift_status": is_found,
                        }
                    )
                    # print(report)
                    logging.info("Output drift report for the train and test data")
                    report_name = params["DATA_LOCATION"][
                        "DATA_VALIDATION_DRIFT_REPORT_FILE_NAME"
                    ]
                    # print(report_name)
            
                    Utility().create_folder(model_dir)
                    drift_report_file_path = os.path.join(model_dir, str(report_name))
                    # print(drift_report_file_path)

                    write_yaml_file(drift_report_file_path,report)
                    logging.info("Output artifacts-> report.yaml.")

                    return status

                else:
                    status = False
                    logging.error("Error has occurred")

        except Exception as e:
            logging.error(e)
            raise e
