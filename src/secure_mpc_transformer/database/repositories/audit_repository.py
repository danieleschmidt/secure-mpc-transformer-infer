"""Repository for audit log management."""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .base import BaseRepository
from ..models import AuditLog, AuditEventType
from ..connection import DatabaseManager


class AuditRepository(BaseRepository[AuditLog]):
    """Repository for managing audit logs."""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, AuditLog)
    
    def _get_table_name(self) -> str:
        return "audit_logs"
    
    def _model_to_dict(self, model: AuditLog) -> Dict[str, Any]:
        """Convert AuditLog to dictionary."""
        return model.to_dict()
    
    def _dict_to_model(self, data: Dict[str, Any]) -> AuditLog:
        """Convert dictionary to AuditLog."""
        return AuditLog.from_dict(data)
    
    async def get_by_event_type(self, event_type: AuditEventType, 
                               limit: Optional[int] = None) -> List[AuditLog]:
        """Get audit logs by event type."""
        if self.db_manager.database_type.value == "postgresql":
            query = "SELECT * FROM audit_logs WHERE event_type = $1 ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query, event_type.value)
                return [
                    self._dict_to_model(self._deserialize_json_fields(dict(row)))
                    for row in results
                ]
        
        elif self.db_manager.database_type.value == "sqlite":
            query = "SELECT * FROM audit_logs WHERE event_type = ? ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, (event_type.value,)) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    return [
                        self._dict_to_model(
                            self._deserialize_json_fields(dict(zip(columns, row)))
                        )
                        for row in rows
                    ]
        
        elif self.db_manager.database_type.value == "mongodb":
            async with self.db_manager.get_connection() as db:
                collection = db[self.table_name]
                cursor = collection.find({"event_type": event_type.value}).sort("timestamp", -1)
                
                if limit:
                    cursor = cursor.limit(limit)
                
                results = await cursor.to_list(length=None)
                return [
                    self._dict_to_model({k: v for k, v in doc.items() if k != '_id'})
                    for doc in results
                ]
        
        return []
    
    async def get_by_risk_level(self, risk_level: str, 
                               limit: Optional[int] = None) -> List[AuditLog]:
        """Get audit logs by risk level."""
        if self.db_manager.database_type.value == "postgresql":
            query = "SELECT * FROM audit_logs WHERE risk_level = $1 ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query, risk_level)
                return [
                    self._dict_to_model(self._deserialize_json_fields(dict(row)))
                    for row in results
                ]
        
        elif self.db_manager.database_type.value == "sqlite":
            query = "SELECT * FROM audit_logs WHERE risk_level = ? ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, (risk_level,)) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    return [
                        self._dict_to_model(
                            self._deserialize_json_fields(dict(zip(columns, row)))
                        )
                        for row in rows
                    ]
        
        elif self.db_manager.database_type.value == "mongodb":
            async with self.db_manager.get_connection() as db:
                collection = db[self.table_name]
                cursor = collection.find({"risk_level": risk_level}).sort("timestamp", -1)
                
                if limit:
                    cursor = cursor.limit(limit)
                
                results = await cursor.to_list(length=None)
                return [
                    self._dict_to_model({k: v for k, v in doc.items() if k != '_id'})
                    for doc in results
                ]
        
        return []
    
    async def get_by_session_id(self, session_id: str) -> List[AuditLog]:
        """Get all audit logs for a specific session."""
        if self.db_manager.database_type.value == "postgresql":
            query = "SELECT * FROM audit_logs WHERE session_id = $1 ORDER BY timestamp ASC"
            
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query, session_id)
                return [
                    self._dict_to_model(self._deserialize_json_fields(dict(row)))
                    for row in results
                ]
        
        elif self.db_manager.database_type.value == "sqlite":
            query = "SELECT * FROM audit_logs WHERE session_id = ? ORDER BY timestamp ASC"
            
            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, (session_id,)) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    return [
                        self._dict_to_model(
                            self._deserialize_json_fields(dict(zip(columns, row)))
                        )
                        for row in rows
                    ]
        
        elif self.db_manager.database_type.value == "mongodb":
            async with self.db_manager.get_connection() as db:
                collection = db[self.table_name]
                cursor = collection.find({"session_id": session_id}).sort("timestamp", 1)
                
                results = await cursor.to_list(length=None)
                return [
                    self._dict_to_model({k: v for k, v in doc.items() if k != '_id'})
                    for doc in results
                ]
        
        return []
    
    async def get_by_party_id(self, party_id: int, hours: int = 24) -> List[AuditLog]:
        """Get audit logs for a specific party."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        if self.db_manager.database_type.value == "postgresql":
            query = """
                SELECT * FROM audit_logs 
                WHERE party_id = $1 AND timestamp >= $2 
                ORDER BY timestamp DESC
            """
            
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query, party_id, cutoff_time)
                return [
                    self._dict_to_model(self._deserialize_json_fields(dict(row)))
                    for row in results
                ]
        
        elif self.db_manager.database_type.value == "sqlite":
            query = """
                SELECT * FROM audit_logs 
                WHERE party_id = ? AND timestamp >= ? 
                ORDER BY timestamp DESC
            """
            
            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, (party_id, cutoff_time.isoformat())) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    return [
                        self._dict_to_model(
                            self._deserialize_json_fields(dict(zip(columns, row)))
                        )
                        for row in rows
                    ]
        
        elif self.db_manager.database_type.value == "mongodb":
            async with self.db_manager.get_connection() as db:
                collection = db[self.table_name]
                cursor = collection.find({
                    "party_id": party_id,
                    "timestamp": {"$gte": cutoff_time}
                }).sort("timestamp", -1)
                
                results = await cursor.to_list(length=None)
                return [
                    self._dict_to_model({k: v for k, v in doc.items() if k != '_id'})
                    for doc in results
                ]
        
        return []
    
    async def get_security_events(self, hours: int = 24) -> List[AuditLog]:
        """Get security-related audit events."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        security_events = [
            AuditEventType.SECURITY_VIOLATION.value,
            AuditEventType.AUTHENTICATION_FAILED.value,
            AuditEventType.MAC_VERIFICATION_FAILED.value
        ]
        
        if self.db_manager.database_type.value == "postgresql":
            placeholders = ",".join([f"${i+2}" for i in range(len(security_events))])
            query = f"""
                SELECT * FROM audit_logs 
                WHERE timestamp >= $1 AND event_type IN ({placeholders})
                ORDER BY timestamp DESC
            """
            
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query, cutoff_time, *security_events)
                return [
                    self._dict_to_model(self._deserialize_json_fields(dict(row)))
                    for row in results
                ]
        
        elif self.db_manager.database_type.value == "sqlite":
            placeholders = ",".join(["?" for _ in security_events])
            query = f"""
                SELECT * FROM audit_logs 
                WHERE timestamp >= ? AND event_type IN ({placeholders})
                ORDER BY timestamp DESC
            """
            
            async with self.db_manager.get_connection() as conn:
                params = [cutoff_time.isoformat()] + security_events
                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    return [
                        self._dict_to_model(
                            self._deserialize_json_fields(dict(zip(columns, row)))
                        )
                        for row in rows
                    ]
        
        elif self.db_manager.database_type.value == "mongodb":
            async with self.db_manager.get_connection() as db:
                collection = db[self.table_name]
                cursor = collection.find({
                    "timestamp": {"$gte": cutoff_time},
                    "event_type": {"$in": security_events}
                }).sort("timestamp", -1)
                
                results = await cursor.to_list(length=None)
                return [
                    self._dict_to_model({k: v for k, v in doc.items() if k != '_id'})
                    for doc in results
                ]
        
        return []
    
    async def get_events_requiring_investigation(self) -> List[AuditLog]:
        """Get events that require investigation."""
        if self.db_manager.database_type.value == "postgresql":
            query = """
                SELECT * FROM audit_logs 
                WHERE requires_investigation = true 
                ORDER BY timestamp DESC
            """
            
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query)
                return [
                    self._dict_to_model(self._deserialize_json_fields(dict(row)))
                    for row in results
                ]
        
        elif self.db_manager.database_type.value == "sqlite":
            query = """
                SELECT * FROM audit_logs 
                WHERE requires_investigation = 1 
                ORDER BY timestamp DESC
            """
            
            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    return [
                        self._dict_to_model(
                            self._deserialize_json_fields(dict(zip(columns, row)))
                        )
                        for row in rows
                    ]
        
        elif self.db_manager.database_type.value == "mongodb":
            async with self.db_manager.get_connection() as db:
                collection = db[self.table_name]
                cursor = collection.find({"requires_investigation": True}).sort("timestamp", -1)
                
                results = await cursor.to_list(length=None)
                return [
                    self._dict_to_model({k: v for k, v in doc.items() if k != '_id'})
                    for doc in results
                ]
        
        return []
    
    async def get_audit_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit log statistics."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        if self.db_manager.database_type.value == "postgresql":
            query = """
                SELECT 
                    event_type,
                    risk_level,
                    COUNT(*) as count
                FROM audit_logs 
                WHERE timestamp >= $1
                GROUP BY event_type, risk_level
                ORDER BY count DESC
            """
            
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query, cutoff_time)
                
                # Also get totals
                total_query = """
                    SELECT 
                        COUNT(*) as total_events,
                        COUNT(CASE WHEN requires_investigation THEN 1 END) as requires_investigation,
                        COUNT(CASE WHEN risk_level = 'critical' THEN 1 END) as critical_events,
                        COUNT(CASE WHEN risk_level = 'high' THEN 1 END) as high_risk_events
                    FROM audit_logs 
                    WHERE timestamp >= $1
                """
                
                totals = await conn.fetchrow(total_query, cutoff_time)
                
                return {
                    "by_event_risk": [dict(row) for row in results],
                    "totals": dict(totals) if totals else {},
                    "time_period_hours": hours
                }
        
        elif self.db_manager.database_type.value == "sqlite":
            query = """
                SELECT 
                    event_type,
                    risk_level,
                    COUNT(*) as count
                FROM audit_logs 
                WHERE timestamp >= ?
                GROUP BY event_type, risk_level
                ORDER BY count DESC
            """
            
            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, (cutoff_time.isoformat(),)) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    by_event_risk = [dict(zip(columns, row)) for row in rows]
                
                # Get totals
                total_query = """
                    SELECT 
                        COUNT(*) as total_events,
                        COUNT(CASE WHEN requires_investigation = 1 THEN 1 END) as requires_investigation,
                        COUNT(CASE WHEN risk_level = 'critical' THEN 1 END) as critical_events,
                        COUNT(CASE WHEN risk_level = 'high' THEN 1 END) as high_risk_events
                    FROM audit_logs 
                    WHERE timestamp >= ?
                """
                
                async with conn.execute(total_query, (cutoff_time.isoformat(),)) as cursor:
                    total_row = await cursor.fetchone()
                    total_columns = [description[0] for description in cursor.description]
                    totals = dict(zip(total_columns, total_row)) if total_row else {}
                
                return {
                    "by_event_risk": by_event_risk,
                    "totals": totals,
                    "time_period_hours": hours
                }
        
        return {"time_period_hours": hours}
    
    async def cleanup_old_logs(self, days: int = 365) -> int:
        """Delete audit logs older than specified days."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        if self.db_manager.database_type.value == "postgresql":
            query = "DELETE FROM audit_logs WHERE timestamp < $1"
            
            async with self.db_manager.get_connection() as conn:
                result = await conn.execute(query, cutoff_time)
                return int(result.split()[-1]) if result else 0
        
        elif self.db_manager.database_type.value == "sqlite":
            query = "DELETE FROM audit_logs WHERE timestamp < ?"
            
            async with self.db_manager.get_connection() as conn:
                await conn.execute(query, (cutoff_time.isoformat(),))
                await conn.commit()
                return conn.total_changes
        
        elif self.db_manager.database_type.value == "mongodb":
            async with self.db_manager.get_connection() as db:
                collection = db[self.table_name]
                result = await collection.delete_many({"timestamp": {"$lt": cutoff_time}})
                return result.deleted_count
        
        return 0
    
    async def search_logs(self, search_terms: List[str], hours: int = 24) -> List[AuditLog]:
        """Search audit logs by description or event data."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        if self.db_manager.database_type.value == "postgresql":
            # Build search conditions
            search_conditions = []
            params = [cutoff_time]
            
            for i, term in enumerate(search_terms):
                param_idx = i + 2
                search_conditions.append(f"(event_description ILIKE ${param_idx} OR event_data::text ILIKE ${param_idx})")
                params.append(f"%{term}%")
            
            query = f"""
                SELECT * FROM audit_logs 
                WHERE timestamp >= $1 AND ({' OR '.join(search_conditions)})
                ORDER BY timestamp DESC
            """
            
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query, *params)
                return [
                    self._dict_to_model(self._deserialize_json_fields(dict(row)))
                    for row in results
                ]
        
        # Simplified search for other databases
        return await self.get_recent_logs(hours)
    
    async def get_recent_logs(self, hours: int = 24, limit: Optional[int] = 100) -> List[AuditLog]:
        """Get recent audit logs."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        if self.db_manager.database_type.value == "postgresql":
            query = "SELECT * FROM audit_logs WHERE timestamp >= $1 ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query, cutoff_time)
                return [
                    self._dict_to_model(self._deserialize_json_fields(dict(row)))
                    for row in results
                ]
        
        elif self.db_manager.database_type.value == "sqlite":
            query = "SELECT * FROM audit_logs WHERE timestamp >= ? ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, (cutoff_time.isoformat(),)) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    return [
                        self._dict_to_model(
                            self._deserialize_json_fields(dict(zip(columns, row)))
                        )
                        for row in rows
                    ]
        
        elif self.db_manager.database_type.value == "mongodb":
            async with self.db_manager.get_connection() as db:
                collection = db[self.table_name]
                cursor = collection.find({"timestamp": {"$gte": cutoff_time}}).sort("timestamp", -1)
                
                if limit:
                    cursor = cursor.limit(limit)
                
                results = await cursor.to_list(length=None)
                return [
                    self._dict_to_model({k: v for k, v in doc.items() if k != '_id'})
                    for doc in results
                ]
        
        return []