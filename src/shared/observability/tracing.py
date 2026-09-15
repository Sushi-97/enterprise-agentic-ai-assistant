import time
import uuid


class RequestTrace:
    def __init__(self, query: str):
        self.trace_id = str(uuid.uuid4())
        self.query = query
        self.start_time = time.perf_counter()
        self.events = []

    def add_event(
        self,
        component: str,
        **details,
    ):
        self.events.append(
            {
                "component": component,
                **details,
            }
        )

    def finish(self):
        total_latency = time.perf_counter() - self.start_time

        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "total_latency_seconds": total_latency,
            "events": self.events,
        }