"""Repository for computation session management."""

from datetime import datetime, timedelta
from typing import Any

from ..connection import DatabaseManager
from ..models import ComputationSession, ProtocolType, SessionStatus
from .base import BaseRepository


class SessionRepository(BaseRepository[ComputationSession]):
    """Repository for managing computation sessions."""

    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, ComputationSession)

    def _get_table_name(self) -> str:
        return "computation_sessions"

    def _model_to_dict(self, model: ComputationSession) -> dict[str, Any]:
        """Convert ComputationSession to dictionary."""
        return model.to_dict()

    def _dict_to_model(self, data: dict[str, Any]) -> ComputationSession:
        """Convert dictionary to ComputationSession."""
        return ComputationSession.from_dict(data)

    async def get_by_status(self, status: SessionStatus, limit: int | None = None) -> list[ComputationSession]:
        """Get sessions by status."""
        if self.db_manager.database_type.value == "postgresql":
            query = "SELECT * FROM computation_sessions WHERE status = $1 ORDER BY created_at DESC"
            if limit:
                query += f" LIMIT {limit}"

            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query, status.value)
                return [
                    self._dict_to_model(self._deserialize_json_fields(dict(row)))
                    for row in results
                ]

        elif self.db_manager.database_type.value == "sqlite":
            query = "SELECT * FROM computation_sessions WHERE status = ? ORDER BY created_at DESC"
            if limit:
                query += f" LIMIT {limit}"

            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, (status.value,)) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]

                    return [
                        self._dict_to_model(
                            self._deserialize_json_fields(dict(zip(columns, row, strict=False)))
                        )
                        for row in rows
                    ]

        elif self.db_manager.database_type.value == "mongodb":
            async with self.db_manager.get_connection() as db:
                collection = db[self.table_name]
                cursor = collection.find({"status": status.value}).sort("created_at", -1)

                if limit:
                    cursor = cursor.limit(limit)

                results = await cursor.to_list(length=None)
                return [
                    self._dict_to_model({k: v for k, v in doc.items() if k != '_id'})
                    for doc in results
                ]

        return []

    async def get_by_protocol(self, protocol_type: ProtocolType, limit: int | None = None) -> list[ComputationSession]:
        """Get sessions by protocol type."""
        if self.db_manager.database_type.value == "postgresql":
            query = "SELECT * FROM computation_sessions WHERE protocol_type = $1 ORDER BY created_at DESC"
            if limit:
                query += f" LIMIT {limit}"

            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query, protocol_type.value)
                return [
                    self._dict_to_model(self._deserialize_json_fields(dict(row)))
                    for row in results
                ]

        elif self.db_manager.database_type.value == "sqlite":
            query = "SELECT * FROM computation_sessions WHERE protocol_type = ? ORDER BY created_at DESC"
            if limit:
                query += f" LIMIT {limit}"

            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, (protocol_type.value,)) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]

                    return [
                        self._dict_to_model(
                            self._deserialize_json_fields(dict(zip(columns, row, strict=False)))
                        )
                        for row in rows
                    ]

        elif self.db_manager.database_type.value == "mongodb":
            async with self.db_manager.get_connection() as db:
                collection = db[self.table_name]
                cursor = collection.find({"protocol_type": protocol_type.value}).sort("created_at", -1)

                if limit:
                    cursor = cursor.limit(limit)

                results = await cursor.to_list(length=None)
                return [
                    self._dict_to_model({k: v for k, v in doc.items() if k != '_id'})
                    for doc in results
                ]

        return []

    async def get_recent_sessions(self, hours: int = 24, limit: int | None = None) -> list[ComputationSession]:
        """Get sessions from the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        if self.db_manager.database_type.value == "postgresql":
            query = "SELECT * FROM computation_sessions WHERE created_at >= $1 ORDER BY created_at DESC"
            if limit:
                query += f" LIMIT {limit}"

            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query, cutoff_time)
                return [
                    self._dict_to_model(self._deserialize_json_fields(dict(row)))
                    for row in results
                ]

        elif self.db_manager.database_type.value == "sqlite":
            query = "SELECT * FROM computation_sessions WHERE created_at >= ? ORDER BY created_at DESC"
            if limit:
                query += f" LIMIT {limit}"

            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, (cutoff_time.isoformat(),)) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]

                    return [
                        self._dict_to_model(
                            self._deserialize_json_fields(dict(zip(columns, row, strict=False)))
                        )
                        for row in rows
                    ]

        elif self.db_manager.database_type.value == "mongodb":
            async with self.db_manager.get_connection() as db:
                collection = db[self.table_name]
                cursor = collection.find({"created_at": {"$gte": cutoff_time}}).sort("created_at", -1)

                if limit:
                    cursor = cursor.limit(limit)

                results = await cursor.to_list(length=None)
                return [
                    self._dict_to_model({k: v for k, v in doc.items() if k != '_id'})
                    for doc in results
                ]

        return []

    async def get_performance_stats(self, hours: int = 24) -> dict[str, Any]:
        """Get performance statistics for sessions."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        if self.db_manager.database_type.value == "postgresql":
            query = """
                SELECT 
                    COUNT(*) as total_sessions,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_sessions,
                    AVG(latency_ms) as avg_latency_ms,
                    MIN(latency_ms) as min_latency_ms,
                    MAX(latency_ms) as max_latency_ms,
                    AVG(memory_usage_mb) as avg_memory_usage_mb,
                    AVG(gpu_utilization) as avg_gpu_utilization
                FROM computation_sessions 
                WHERE created_at >= $1 AND status = 'completed'
            """

            async with self.db_manager.get_connection() as conn:
                result = await conn.fetchrow(query, cutoff_time)
                return dict(result) if result else {}

        elif self.db_manager.database_type.value == "sqlite":
            query = """
                SELECT 
                    COUNT(*) as total_sessions,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_sessions,
                    AVG(latency_ms) as avg_latency_ms,
                    MIN(latency_ms) as min_latency_ms,
                    MAX(latency_ms) as max_latency_ms,
                    AVG(memory_usage_mb) as avg_memory_usage_mb,
                    AVG(gpu_utilization) as avg_gpu_utilization
                FROM computation_sessions 
                WHERE created_at >= ? AND status = 'completed'
            """

            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, (cutoff_time.isoformat(),)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        columns = [description[0] for description in cursor.description]
                        return dict(zip(columns, row, strict=False))

        return {}

    async def cleanup_old_sessions(self, days: int = 30) -> int:
        """Delete sessions older than specified days."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        if self.db_manager.database_type.value == "postgresql":
            query = "DELETE FROM computation_sessions WHERE created_at < $1"

            async with self.db_manager.get_connection() as conn:
                result = await conn.execute(query, cutoff_time)
                return int(result.split()[-1]) if result else 0

        elif self.db_manager.database_type.value == "sqlite":
            query = "DELETE FROM computation_sessions WHERE created_at < ?"

            async with self.db_manager.get_connection() as conn:
                await conn.execute(query, (cutoff_time.isoformat(),))
                await conn.commit()
                return conn.total_changes

        elif self.db_manager.database_type.value == "mongodb":
            async with self.db_manager.get_connection() as db:
                collection = db[self.table_name]
                result = await collection.delete_many({"created_at": {"$lt": cutoff_time}})
                return result.deleted_count

        return 0

    async def get_active_sessions(self) -> list[ComputationSession]:
        """Get all currently running sessions."""
        return await self.get_by_status(SessionStatus.RUNNING)

    async def get_failed_sessions(self, limit: int | None = 50) -> list[ComputationSession]:
        """Get failed sessions for debugging."""
        return await self.get_by_status(SessionStatus.FAILED, limit)

    async def update_session_status(self, session_id: str, status: SessionStatus,
                                   error_message: str | None = None) -> bool:
        """Update session status and optionally error message."""
        session = await self.get_by_id(session_id)
        if not session:
            return False

        session.status = status
        if error_message:
            session.error_message = error_message

        if status == SessionStatus.RUNNING and not session.started_at:
            session.start()
        elif status in [SessionStatus.COMPLETED, SessionStatus.FAILED]:
            if not session.completed_at:
                session.completed_at = datetime.utcnow()

        await self.update(session)
        return True

    async def get_session_statistics(self) -> dict[str, Any]:
        """Get overall session statistics."""
        if self.db_manager.database_type.value == "postgresql":
            query = """
                SELECT 
                    status,
                    protocol_type,
                    COUNT(*) as count,
                    AVG(latency_ms) as avg_latency,
                    AVG(memory_usage_mb) as avg_memory
                FROM computation_sessions 
                GROUP BY status, protocol_type
                ORDER BY status, protocol_type
            """

            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query)
                return {
                    "by_status_protocol": [dict(row) for row in results],
                    "total_sessions": await self.count()
                }

        # Simplified stats for other databases
        return {
            "total_sessions": await self.count(),
            "active_sessions": len(await self.get_active_sessions()),
            "failed_sessions": len(await self.get_failed_sessions())
        }
