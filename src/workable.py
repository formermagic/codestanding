import typing
from abc import ABC, abstractmethod
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor

from tqdm import tqdm


class Workable(ABC):
    @abstractmethod
    def run(self) -> None:
        pass


class WorkableRunner:
    def run_workable(self, workable: Workable) -> None:
        workable.run()

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

            for _ in tqdm(as_completed(futures), **kwargs):
                continue
