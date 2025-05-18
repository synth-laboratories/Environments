from src.environment.shared_engine import Engine


class StatefulEngineSnapshot:
    pass


class StatefulEngine(Engine):
    async def serialize(self):
        pass

    @classmethod
    async def deserialize(self, engine_snapshot: StatefulEngineSnapshot):
        pass

    async def _step_engine(self):
        pass
