from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from typing import List, Dict
import json
import asyncio
from datetime import datetime


class LiveManager:
    def __init__(self):
        self.active: List[WebSocket] = []
        self.trace_updates: List[Dict] = []  # Recent updates for new connections
        self.max_history = 100

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

        # Send recent updates to new connection
        if self.trace_updates:
            await ws.send_json(
                {
                    "event": "history",
                    "updates": self.trace_updates[-10:],  # Last 10 updates
                }
            )

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        """Broadcast update to all connected clients."""
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()

        # Store in history
        self.trace_updates.append(data)
        if len(self.trace_updates) > self.max_history:
            self.trace_updates.pop(0)

        # Broadcast to all active connections
        disconnected = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except (WebSocketDisconnect, Exception):
                disconnected.append(ws)

        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws)

    async def broadcast_new_trace(self, trace_id: str, run_id: str, env_name: str):
        """Convenience method to broadcast new trace event."""
        await self.broadcast(
            {
                "event": "new_trace",
                "trace_id": trace_id,
                "run_id": run_id,
                "env_name": env_name,
            }
        )

    async def broadcast_trace_update(self, trace_id: str, update_type: str, data: Dict):
        """Broadcast trace update event."""
        await self.broadcast(
            {
                "event": "trace_update",
                "trace_id": trace_id,
                "update_type": update_type,
                "data": data,
            }
        )


# Global instance
live = LiveManager()
