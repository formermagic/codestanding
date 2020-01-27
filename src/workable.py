import faulthandler
import logging
import typing
from abc import ABC, abstractmethod
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor

from tqdm import tqdm

faulthandler.enable()


class Workable(ABC):
    ResultType = typing.TypeVar("T")

    @abstractmethod
    def run(self) -> "Workable.ResultType":
        pass


class WorkableRunner:
    def __init__(
        self, success_fn: typing.Callable[[Workable.ResultType], None]
    ) -> None:
        self.success_fn = success_fn

    def run_workable(self, workable: Workable) -> Workable.ResultType:
        return workable.run()

    # pylint: disable=broad-except
    def execute(
        self, workables: typing.List[Workable], max_workers: int = 16
    ) -> None:
        with ProcessPoolExecutor(max_workers) as executor:
            kwargs = {
                "total": len(workables),
                "unit": "it",
                "unit_scale": True,
                "leave": True,
            }

            futures = [
                executor.submit(self.run_workable, workable)
                for workable in workables
            ]

            for future in tqdm(as_completed(futures), **kwargs):
                try:
                    result = future.result()
                    self.success_fn(result)
                except BaseException as error:
                    logging.error(
                        "Unable to complete a future due to error: %s",
                        str(error),
                    )
