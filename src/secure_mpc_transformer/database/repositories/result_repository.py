"""Repository for inference result management."""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .base import BaseRepository
from ..models import InferenceResult
from ..connection import DatabaseManager


class ResultRepository(BaseRepository[InferenceResult]):
    """Repository for managing inference results."""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, InferenceResult)
    
    def _get_table_name(self) -> str:
        return "inference_results"
    
    def _model_to_dict(self, model: InferenceResult) -> Dict[str, Any]:
        """Convert InferenceResult to dictionary."""
        return model.to_dict()
    
    def _dict_to_model(self, data: Dict[str, Any]) -> InferenceResult:
        """Convert dictionary to InferenceResult."""
        return InferenceResult.from_dict(data)
    
    async def get_by_session_id(self, session_id: str) -> List[InferenceResult]:
        """Get all results for a specific session."""
        if self.db_manager.database_type.value == "postgresql":
            query = "SELECT * FROM inference_results WHERE session_id = $1 ORDER BY created_at DESC"
            
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query, session_id)
                return [
                    self._dict_to_model(self._deserialize_json_fields(dict(row)))
                    for row in results
                ]
        
        elif self.db_manager.database_type.value == "sqlite":
            query = "SELECT * FROM inference_results WHERE session_id = ? ORDER BY created_at DESC"
            
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
                cursor = collection.find({"session_id": session_id}).sort("created_at", -1)
                
                results = await cursor.to_list(length=None)
                return [
                    self._dict_to_model({k: v for k, v in doc.items() if k != '_id'})
                    for doc in results
                ]
        
        return []
    
    async def get_performance_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics for results."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        if self.db_manager.database_type.value == "postgresql":
            query = """
                SELECT 
                    COUNT(*) as total_results,
                    AVG(computation_time_ms) as avg_computation_time_ms,
                    AVG(communication_time_ms) as avg_communication_time_ms,
                    AVG(preprocessing_time_ms) as avg_preprocessing_time_ms,
                    SUM(total_operations) as total_operations,
                    SUM(arithmetic_operations) as arithmetic_operations,
                    SUM(boolean_operations) as boolean_operations,
                    SUM(conversions) as conversions,
                    AVG(mac_verifications) as avg_mac_verifications,
                    SUM(security_violations) as total_security_violations
                FROM inference_results 
                WHERE created_at >= $1
            """
            
            async with self.db_manager.get_connection() as conn:
                result = await conn.fetchrow(query, cutoff_time)
                return dict(result) if result else {}
        
        elif self.db_manager.database_type.value == "sqlite":
            query = """
                SELECT 
                    COUNT(*) as total_results,
                    AVG(computation_time_ms) as avg_computation_time_ms,
                    AVG(communication_time_ms) as avg_communication_time_ms,
                    AVG(preprocessing_time_ms) as avg_preprocessing_time_ms,
                    SUM(total_operations) as total_operations,
                    SUM(arithmetic_operations) as arithmetic_operations,
                    SUM(boolean_operations) as boolean_operations,
                    SUM(conversions) as conversions,
                    AVG(mac_verifications) as avg_mac_verifications,
                    SUM(security_violations) as total_security_violations
                FROM inference_results 
                WHERE created_at >= ?
            """
            
            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, (cutoff_time.isoformat(),)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        columns = [description[0] for description in cursor.description]
                        return dict(zip(columns, row))
        
        return {}
    
    async def get_privacy_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get privacy-related metrics."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        if self.db_manager.database_type.value == "postgresql":
            query = """
                SELECT 
                    COUNT(*) as total_results,
                    AVG(privacy_epsilon) as avg_epsilon,
                    AVG(privacy_delta) as avg_delta,
                    SUM(privacy_spent) as total_privacy_spent,
                    MIN(privacy_epsilon) as min_epsilon,
                    MAX(privacy_epsilon) as max_epsilon
                FROM inference_results 
                WHERE created_at >= $1 AND privacy_epsilon IS NOT NULL
            """
            
            async with self.db_manager.get_connection() as conn:
                result = await conn.fetchrow(query, cutoff_time)
                return dict(result) if result else {}
        
        elif self.db_manager.database_type.value == "sqlite":
            query = """
                SELECT 
                    COUNT(*) as total_results,
                    AVG(privacy_epsilon) as avg_epsilon,
                    AVG(privacy_delta) as avg_delta,
                    SUM(privacy_spent) as total_privacy_spent,
                    MIN(privacy_epsilon) as min_epsilon,
                    MAX(privacy_epsilon) as max_epsilon
                FROM inference_results 
                WHERE created_at >= ? AND privacy_epsilon IS NOT NULL
            """
            
            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, (cutoff_time.isoformat(),)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        columns = [description[0] for description in cursor.description]
                        return dict(zip(columns, row))
        
        return {}
    
    async def get_security_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get security-related metrics."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        if self.db_manager.database_type.value == "postgresql":
            query = """
                SELECT 
                    COUNT(*) as total_results,
                    SUM(mac_verifications) as total_mac_verifications,
                    SUM(proof_generations) as total_proof_generations,
                    SUM(proof_verifications) as total_proof_verifications,
                    SUM(security_violations) as total_security_violations,
                    AVG(mac_verifications) as avg_mac_verifications,
                    AVG(proof_generations) as avg_proof_generations,
                    AVG(proof_verifications) as avg_proof_verifications
                FROM inference_results 
                WHERE created_at >= $1
            """
            
            async with self.db_manager.get_connection() as conn:
                result = await conn.fetchrow(query, cutoff_time)
                return dict(result) if result else {}
        
        elif self.db_manager.database_type.value == "sqlite":
            query = """
                SELECT 
                    COUNT(*) as total_results,
                    SUM(mac_verifications) as total_mac_verifications,
                    SUM(proof_generations) as total_proof_generations,
                    SUM(proof_verifications) as total_proof_verifications,
                    SUM(security_violations) as total_security_violations,
                    AVG(mac_verifications) as avg_mac_verifications,
                    AVG(proof_generations) as avg_proof_generations,
                    AVG(proof_verifications) as avg_proof_verifications
                FROM inference_results 
                WHERE created_at >= ?
            """
            
            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, (cutoff_time.isoformat(),)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        columns = [description[0] for description in cursor.description]
                        return dict(zip(columns, row))
        
        return {}
    
    async def get_operation_breakdown(self, session_id: Optional[str] = None, 
                                    hours: Optional[int] = 24) -> Dict[str, Any]:
        """Get breakdown of operations performed."""
        conditions = []
        params = []
        
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        
        if hours:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            conditions.append("created_at >= ?")
            params.append(cutoff_time.isoformat())
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        if self.db_manager.database_type.value == "sqlite":
            query = f"""
                SELECT 
                    SUM(total_operations) as total_operations,
                    SUM(arithmetic_operations) as arithmetic_operations,
                    SUM(boolean_operations) as boolean_operations,
                    SUM(conversions) as conversions,
                    ROUND(100.0 * SUM(arithmetic_operations) / SUM(total_operations), 2) as arithmetic_percentage,
                    ROUND(100.0 * SUM(boolean_operations) / SUM(total_operations), 2) as boolean_percentage,
                    ROUND(100.0 * SUM(conversions) / SUM(total_operations), 2) as conversion_percentage
                FROM inference_results 
                {where_clause}
            """
            
            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, params) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        columns = [description[0] for description in cursor.description]
                        return dict(zip(columns, row))
        
        return {}
    
    async def get_timing_breakdown(self, session_id: Optional[str] = None,
                                 hours: Optional[int] = 24) -> Dict[str, Any]:
        """Get breakdown of timing metrics."""
        conditions = []
        params = []
        
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        
        if hours:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            conditions.append("created_at >= ?")
            params.append(cutoff_time.isoformat())
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        if self.db_manager.database_type.value == "sqlite":
            query = f"""
                SELECT 
                    AVG(computation_time_ms) as avg_computation_time_ms,
                    AVG(communication_time_ms) as avg_communication_time_ms,
                    AVG(preprocessing_time_ms) as avg_preprocessing_time_ms,
                    AVG(computation_time_ms + communication_time_ms + preprocessing_time_ms) as avg_total_time_ms,
                    MIN(computation_time_ms) as min_computation_time_ms,
                    MAX(computation_time_ms) as max_computation_time_ms
                FROM inference_results 
                {where_clause}
            """
            
            async with self.db_manager.get_connection() as conn:
                async with conn.execute(query, params) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        columns = [description[0] for description in cursor.description]
                        return dict(zip(columns, row))
        
        return {}
    
    async def cleanup_old_results(self, days: int = 90) -> int:
        """Delete results older than specified days."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        if self.db_manager.database_type.value == "postgresql":
            query = "DELETE FROM inference_results WHERE created_at < $1"
            
            async with self.db_manager.get_connection() as conn:
                result = await conn.execute(query, cutoff_time)
                return int(result.split()[-1]) if result else 0
        
        elif self.db_manager.database_type.value == "sqlite":
            query = "DELETE FROM inference_results WHERE created_at < ?"
            
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
    
    async def get_top_performing_results(self, limit: int = 10, 
                                       metric: str = "computation_time_ms") -> List[InferenceResult]:
        """Get top performing results by specified metric."""
        order = "ASC" if metric.endswith("_time_ms") else "DESC"  # Lower time is better
        
        if self.db_manager.database_type.value == "postgresql":
            query = f"""
                SELECT * FROM inference_results 
                WHERE {metric} IS NOT NULL 
                ORDER BY {metric} {order}
                LIMIT {limit}
            """
            
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query)
                return [
                    self._dict_to_model(self._deserialize_json_fields(dict(row)))
                    for row in results
                ]
        
        elif self.db_manager.database_type.value == "sqlite":
            query = f"""
                SELECT * FROM inference_results 
                WHERE {metric} IS NOT NULL 
                ORDER BY {metric} {order}
                LIMIT {limit}
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
        
        return []
    
    async def get_results_with_security_violations(self) -> List[InferenceResult]:
        """Get all results that had security violations."""
        if self.db_manager.database_type.value == "postgresql":
            query = "SELECT * FROM inference_results WHERE security_violations > 0 ORDER BY created_at DESC"
            
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query)
                return [
                    self._dict_to_model(self._deserialize_json_fields(dict(row)))
                    for row in results
                ]
        
        elif self.db_manager.database_type.value == "sqlite":
            query = "SELECT * FROM inference_results WHERE security_violations > 0 ORDER BY created_at DESC"
            
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
                cursor = collection.find({"security_violations": {"$gt": 0}}).sort("created_at", -1)
                
                results = await cursor.to_list(length=None)
                return [
                    self._dict_to_model({k: v for k, v in doc.items() if k != '_id'})
                    for doc in results
                ]
        
        return []