from typing import List


def abstract():
    raise RuntimeError('abstract')


class IDetectionMetadata:
    def tlbr_r(self) -> (float, float, float, float):
        abstract()
    
    def tlbr_a(self) -> (int, int, int, int):
        abstract()

    def confidence(self) -> float:
        abstract()

    def class_(self) -> str:
        abstract()


class TrackMetadata:
    def __init__(self, id_, trace, class_, inner):
        self.id_ = id_
        self.trace = trace
        self.class_ = class_
        self.inner = inner


class Tracker:
    def feed_im(self, im, det: List[IDetectionMetadata]) -> List[TrackMetadata]:
        abstract()
