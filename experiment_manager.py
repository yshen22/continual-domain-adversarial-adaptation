import json
from datetime import datetime
import collections
import os
from typing import Any, Dict
import tempfile
import pandas as pd
import tensorflow as tf
import tree
import numpy as np
import shutil


class ScalarMetricsManager():
  """Utility class for saving/loading scalar experiment metrics.

  The metrics are backed by CSVs stored on the file system.
  """

  def __init__(self,
               root_metrics_dir: str = '/tmp',
               prefix: str = 'experiment',
               use_bz2: bool = False):
    """Returns an initialized `ScalarMetricsManager`.

    This class will maintain metrics in a CSV file in the filesystem. The path
    of the file is {`root_metrics_dir`}/{`prefix`}.metrics.csv (if use_bz2 is
    set to False) or {`root_metrics_dir`}/{`prefix`}.metrics.csv.bz2 (if
    use_bz2 is set to True). To use this class upon restart of an experiment at
    an earlier round number, you can initialize and then call the
    clear_rounds_after() method to remove all rows for round numbers later than
    the restart round number. This ensures that no duplicate rows of data exist
    in the CSV.

    Args:
      root_metrics_dir: A path on the filesystem to store CSVs.
      prefix: A string to use as the prefix of filename. Usually the name of a
        specific run in a larger grid of experiments sharing a common
        `root_metrics_dir`.
      use_bz2: A boolean indicating whether to zip the result metrics csv using
        bz2.

    Raises:
      ValueError: If `root_metrics_dir` is empty string.
      ValueError: If `prefix` is empty string.
      ValueError: If the specified metrics csv file already exists but does not
        contain a `round_num` column.
    """
    super().__init__()
    if not root_metrics_dir:
      raise ValueError('Empty string passed for root_metrics_dir argument.')
    if not prefix:
      raise ValueError('Empty string passed for prefix argument.')

    if use_bz2:
      # Using .bz2 rather than .zip due to
      # https://github.com/pandas-dev/pandas/issues/26023
      self._metrics_filename = os.path.join(root_metrics_dir,
                                            f'{prefix}.metrics.csv.bz2')
    else:
      self._metrics_filename = os.path.join(root_metrics_dir,
                                            f'{prefix}.metrics.csv')
    if not tf.io.gfile.exists(self._metrics_filename):
      atomic_write_to_csv(pd.DataFrame(), self._metrics_filename)

    self._metrics = atomic_read_from_csv(self._metrics_filename)
    if not self._metrics.empty and 'round_num' not in self._metrics.columns:
      raise ValueError(
          f'The specified csv file ({self._metrics_filename}) already exists '
          'but was not created by ScalarMetricsManager (it does not contain a '
          '`round_num` column.')

    self._latest_round_num = (None if self._metrics.empty else
                              self._metrics.round_num.max(axis=0))

  def update_metrics(self, round_num,
                     metrics_to_append: Dict[str, Any]) -> Dict[str, float]:
    """Updates the stored metrics data with metrics for a specific round.

    The specified `round_num` must be later than the latest round number for
    which metrics exist in the stored metrics data. This method will atomically
    update the stored CSV file. Also, if stored metrics already exist and
    `metrics_to_append` contains a new, previously unseen metric name, a new
    column in the dataframe will be added for that metric, and all previous rows
    will fill in with NaN values for the metric.

    Args:
      round_num: Communication round at which `metrics_to_append` was collected.
      metrics_to_append: A dictionary of metrics collected during `round_num`.
        These metrics can be in a nested structure, but the nesting will be
        flattened for storage in the CSV (with the new keys equal to the paths
        in the nested structure).

    Returns:
      A `collections.OrderedDict` of the data just added in a new row to the
        pandas.DataFrame. Compared with the input `metrics_to_append`, this data
        is flattened, with the key names equal to the path in the nested
        structure. Also, `round_num` has been added as an additional key.

    Raises:
      ValueError: If the provided round number is negative.
      ValueError: If the provided round number is less than or equal to the
        latest round number in the stored metrics data.
    """
    if round_num < 0:
      raise ValueError(f'Attempting to append metrics for round {round_num}, '
                       'which is negative.')
    if self._latest_round_num and round_num <= self._latest_round_num:
      raise ValueError(f'Attempting to append metrics for round {round_num}, '
                       'but metrics already exist through round '
                       f'{self._latest_round_num}.')

    # Add the round number to the metrics before storing to csv file. This will
    # be used if a restart occurs, to identify which metrics to trim in the
    # _clear_invalid_rounds() method.
    metrics_to_append['round_num'] = round_num

    flat_metrics = tree.flatten_with_path(metrics_to_append)
    flat_metrics = [
        ('/'.join(map(str, path)), item) for path, item in flat_metrics
    ]
    flat_metrics = collections.OrderedDict(flat_metrics)
    self._metrics = self._metrics.append(flat_metrics, ignore_index=True)
    atomic_write_to_csv(self._metrics, self._metrics_filename)
    self._latest_round_num = round_num

    return flat_metrics

  def get_metrics(self) -> pd.DataFrame:
    """Retrieve the stored experiment metrics data for all rounds.

    Returns:
      A `pandas.DataFrame` containing experiment metrics data for all rounds.
        This DataFrame is in `wide` format: a row for each round and a column
        for each metric. The data has been flattened, with the column names
        equal to the path in the original nested metric structure. There is a
        column (`round_num`) to indicate the round number.
    """
    return self._metrics

  def clear_all_rounds(self) -> None:
    """Existing metrics for all rounds are cleared out.

    This method will atomically update the stored CSV file.
    """
    self._metrics = pd.DataFrame()
    atomic_write_to_csv(self._metrics, self._metrics_filename)
    self._latest_round_num = None

  def clear_rounds_after(self, last_valid_round_num: int) -> None:
    """Metrics for rounds greater than `last_valid_round_num` are cleared out.

    By using this method, this class can be used upon restart of an experiment
    at `last_valid_round_num` to ensure that no duplicate rows of data exist in
    the CSV file. This method will atomically update the stored CSV file.

    Args:
      last_valid_round_num: All metrics for rounds later than this are expunged.

    Raises:
      RuntimeError: If metrics do not exist (none loaded during construction '
        nor recorded via `update_metrics()` and `last_valid_round_num` is not
        zero.
      ValueError: If `last_valid_round_num` is negative.
    """
    if last_valid_round_num < 0:
      raise ValueError('Attempting to clear metrics after round '
                       f'{last_valid_round_num}, which is negative.')
    if self._latest_round_num is None:
      if last_valid_round_num == 0:
        return
      raise RuntimeError('Metrics do not exist yet.')
    self._metrics = self._metrics.drop(
        self._metrics[self._metrics.round_num > last_valid_round_num].index)
    atomic_write_to_csv(self._metrics, self._metrics_filename)
    self._latest_round_num = last_valid_round_num

  @property
  def metrics_filename(self) -> str:
    return self._metrics_filename

def _setup_outputs(root_output_dir,
                   hparam_dict):
    now =datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    
    if (not os.path.exists(root_output_dir)):
        os.makedirs(root_output_dir)
    with open(os.path.join(root_output_dir, dt_string +'.txt'), 'w') as outfile:
        json.dump(hparam_dict, outfile)
    metrics_mngr = ScalarMetricsManager(
      root_output_dir, dt_string, use_bz2=False)
    return metrics_mngr

def atomic_write_to_csv(dataframe: pd.DataFrame,
                        output_file: str,
                        overwrite: bool = True) -> None:
  """Writes `dataframe` to `output_file` as a (possibly zipped) CSV file.

  Args:
    dataframe: A `pandas.Dataframe`.
    output_file: The final output file to write. The output will be compressed
      depending on the filename, see documentation for
      pandas.DateFrame.to_csv(compression='infer').
    overwrite: Whether to overwrite output_file if it exists.
  """
  # Exporting via to_hdf() is an appealing option, because we could perhaps
  # maintain more type information, and also write both hyperparameters and
  # results to the same HDF5 file. However, to_hdf() call uses pickle under the
  # hood, and there seems to be no way to tell it to use pickle protocol=2, it
  # defaults to 4. This means the results cannot be read from Python 2. We
  # currently still want Python 2 support, so sticking with CSVs for now.

  # At least when writing a zip, .to_csv() is not happy taking a gfile,
  # so we need a temp file on the local filesystem.
  tmp_dir = tempfile.mkdtemp(prefix='atomic_write_to_csv_tmp')
  # We put the output_file name last so we preserve the extension to allow
  # inference of the desired compression format. Note that files with .zip
  # extension (but not .bz2, .gzip, or .xv) have unexpected internal filenames
  # due to https://github.com/pandas-dev/pandas/issues/26023, not
  # because of something we are doing here.
  tmp_name = os.path.join(tmp_dir, os.path.basename(output_file))
  assert not tf.io.gfile.exists(tmp_name), 'file [{!s}] exists'.format(tmp_name)
  dataframe.to_csv(tmp_name, header=True)

  # Now, copy to a temp gfile next to the final target, allowing for
  # an atomic move.
  tmp_gfile_name = os.path.join(
      os.path.dirname(output_file), '{}.tmp{}'.format(
          os.path.basename(output_file),
          np.random.randint(0, 2**63, dtype=np.int64)))
  tf.io.gfile.copy(src=tmp_name, dst=tmp_gfile_name, overwrite=overwrite)

  # Finally, do an atomic rename and clean up:
  tf.io.gfile.rename(tmp_gfile_name, output_file, overwrite=overwrite)
  shutil.rmtree(tmp_dir)

def atomic_read_from_csv(csv_file):
  """Reads a `pandas.DataFrame` from the (possibly zipped) `csv_file`.

  Format note: The CSV is expected to have an index column.

  Args:
    csv_file: A (possibly zipped) CSV file.

  Returns:
    A `pandas.Dataframe`.
  """
  # return pd.read_csv(
  #     tf.io.gfile.GFile(csv_file, mode='rb'),
  #     compression='bz2' if csv_file.endswith('.bz2') else None,
  #     engine='c',
  #     index_col=0)
  return pd.read_csv(
      tf.io.gfile.GFile(csv_file),
      compression= None,
      engine='c',
      index_col=0)

