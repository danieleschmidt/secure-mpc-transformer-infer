"""Base repository class with common CRUD operations."""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar

from ..connection import DatabaseManager, DatabaseType
from ..models import AuditLog, ComputationSession, InferenceResult

# Generic type for model classes
T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """Base repository with common database operations."""

    def __init__(self, db_manager: DatabaseManager, model_class: type[T]):
        self.db_manager = db_manager
        self.model_class = model_class
        self.table_name = self._get_table_name()

    @abstractmethod
    def _get_table_name(self) -> str:
        """Get the database table name for this repository."""
        pass

    @abstractmethod
    def _model_to_dict(self, model: T) -> dict[str, Any]:
        """Convert model instance to dictionary for database storage."""
        pass

    @abstractmethod
    def _dict_to_model(self, data: dict[str, Any]) -> T:
        """Convert dictionary from database to model instance."""
        pass

    async def create(self, model: T) -> T:
        """Create a new record in the database."""
        data = self._model_to_dict(model)

        if self.db_manager.database_type == DatabaseType.POSTGRESQL:
            await self._create_postgresql(data)
        elif self.db_manager.database_type == DatabaseType.MONGODB:
            await self._create_mongodb(data)
        elif self.db_manager.database_type == DatabaseType.SQLITE:
            await self._create_sqlite(data)
        elif self.db_manager.database_type == DatabaseType.REDIS:
            await self._create_redis(data)

        return model

    async def get_by_id(self, record_id: str) -> T | None:
        """Get a record by its ID."""
        if self.db_manager.database_type == DatabaseType.POSTGRESQL:
            return await self._get_by_id_postgresql(record_id)
        elif self.db_manager.database_type == DatabaseType.MONGODB:
            return await self._get_by_id_mongodb(record_id)
        elif self.db_manager.database_type == DatabaseType.SQLITE:
            return await self._get_by_id_sqlite(record_id)
        elif self.db_manager.database_type == DatabaseType.REDIS:
            return await self._get_by_id_redis(record_id)

        return None

    async def update(self, model: T) -> T:
        """Update an existing record."""
        data = self._model_to_dict(model)

        if self.db_manager.database_type == DatabaseType.POSTGRESQL:
            await self._update_postgresql(data)
        elif self.db_manager.database_type == DatabaseType.MONGODB:
            await self._update_mongodb(data)
        elif self.db_manager.database_type == DatabaseType.SQLITE:
            await self._update_sqlite(data)
        elif self.db_manager.database_type == DatabaseType.REDIS:
            await self._update_redis(data)

        return model

    async def delete(self, record_id: str) -> bool:
        """Delete a record by its ID."""
        if self.db_manager.database_type == DatabaseType.POSTGRESQL:
            return await self._delete_postgresql(record_id)
        elif self.db_manager.database_type == DatabaseType.MONGODB:
            return await self._delete_mongodb(record_id)
        elif self.db_manager.database_type == DatabaseType.SQLITE:
            return await self._delete_sqlite(record_id)
        elif self.db_manager.database_type == DatabaseType.REDIS:
            return await self._delete_redis(record_id)

        return False

    async def list_all(self, limit: int | None = None, offset: int = 0) -> list[T]:
        """List all records with optional pagination."""
        if self.db_manager.database_type == DatabaseType.POSTGRESQL:
            return await self._list_all_postgresql(limit, offset)
        elif self.db_manager.database_type == DatabaseType.MONGODB:
            return await self._list_all_mongodb(limit, offset)
        elif self.db_manager.database_type == DatabaseType.SQLITE:
            return await self._list_all_sqlite(limit, offset)
        elif self.db_manager.database_type == DatabaseType.REDIS:
            return await self._list_all_redis(limit, offset)

        return []

    async def count(self) -> int:
        """Count total number of records."""
        if self.db_manager.database_type == DatabaseType.POSTGRESQL:
            return await self._count_postgresql()
        elif self.db_manager.database_type == DatabaseType.MONGODB:
            return await self._count_mongodb()
        elif self.db_manager.database_type == DatabaseType.SQLITE:
            return await self._count_sqlite()
        elif self.db_manager.database_type == DatabaseType.REDIS:
            return await self._count_redis()

        return 0

    # PostgreSQL implementations
    async def _create_postgresql(self, data: dict[str, Any]) -> None:
        """Create record in PostgreSQL."""
        columns = list(data.keys())
        placeholders = [f'${i+1}' for i in range(len(columns))]
        values = [self._serialize_json_field(data[col]) for col in columns]

        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """

        async with self.db_manager.get_connection() as conn:
            await conn.execute(query, *values)

    async def _get_by_id_postgresql(self, record_id: str) -> T | None:
        """Get record by ID from PostgreSQL."""
        id_column = self._get_id_column()
        query = f"SELECT * FROM {self.table_name} WHERE {id_column} = $1"

        async with self.db_manager.get_connection() as conn:
            result = await conn.fetchrow(query, record_id)
            if result:
                data = dict(result)
                data = self._deserialize_json_fields(data)
                return self._dict_to_model(data)
        return None

    async def _update_postgresql(self, data: dict[str, Any]) -> None:
        """Update record in PostgreSQL."""
        id_column = self._get_id_column()
        record_id = data.pop(id_column)

        if not data:  # No fields to update
            return

        set_clauses = [f"{col} = ${i+2}" for i, col in enumerate(data.keys())]
        values = [self._serialize_json_field(data[col]) for col in data]

        query = f"""
            UPDATE {self.table_name}
            SET {', '.join(set_clauses)}
            WHERE {id_column} = $1
        """

        async with self.db_manager.get_connection() as conn:
            await conn.execute(query, record_id, *values)

    async def _delete_postgresql(self, record_id: str) -> bool:
        """Delete record from PostgreSQL."""
        id_column = self._get_id_column()
        query = f"DELETE FROM {self.table_name} WHERE {id_column} = $1"

        async with self.db_manager.get_connection() as conn:
            result = await conn.execute(query, record_id)
            return 'DELETE 1' in result

    async def _list_all_postgresql(self, limit: int | None, offset: int) -> list[T]:
        """List all records from PostgreSQL."""
        query = f"SELECT * FROM {self.table_name} ORDER BY created_at DESC"

        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"

        async with self.db_manager.get_connection() as conn:
            results = await conn.fetch(query)
            return [
                self._dict_to_model(self._deserialize_json_fields(dict(row)))
                for row in results
            ]

    async def _count_postgresql(self) -> int:
        """Count records in PostgreSQL."""
        query = f"SELECT COUNT(*) FROM {self.table_name}"

        async with self.db_manager.get_connection() as conn:
            result = await conn.fetchval(query)
            return result or 0

    # MongoDB implementations
    async def _create_mongodb(self, data: dict[str, Any]) -> None:
        """Create record in MongoDB."""
        async with self.db_manager.get_connection() as db:
            collection = db[self.table_name]
            await collection.insert_one(data)

    async def _get_by_id_mongodb(self, record_id: str) -> T | None:
        """Get record by ID from MongoDB."""
        id_field = self._get_id_column()

        async with self.db_manager.get_connection() as db:
            collection = db[self.table_name]
            result = await collection.find_one({id_field: record_id})
            if result:
                result.pop('_id', None)  # Remove MongoDB's _id field
                return self._dict_to_model(result)
        return None

    async def _update_mongodb(self, data: dict[str, Any]) -> None:
        """Update record in MongoDB."""
        id_field = self._get_id_column()
        record_id = data.pop(id_field)

        async with self.db_manager.get_connection() as db:
            collection = db[self.table_name]
            await collection.update_one(
                {id_field: record_id},
                {'$set': data}
            )

    async def _delete_mongodb(self, record_id: str) -> bool:
        """Delete record from MongoDB."""
        id_field = self._get_id_column()

        async with self.db_manager.get_connection() as db:
            collection = db[self.table_name]
            result = await collection.delete_one({id_field: record_id})
            return result.deleted_count > 0

    async def _list_all_mongodb(self, limit: int | None, offset: int) -> list[T]:
        """List all records from MongoDB."""
        async with self.db_manager.get_connection() as db:
            collection = db[self.table_name]

            cursor = collection.find().sort('created_at', -1)
            if offset > 0:
                cursor = cursor.skip(offset)
            if limit:
                cursor = cursor.limit(limit)

            results = await cursor.to_list(length=None)
            return [
                self._dict_to_model({k: v for k, v in doc.items() if k != '_id'})
                for doc in results
            ]

    async def _count_mongodb(self) -> int:
        """Count records in MongoDB."""
        async with self.db_manager.get_connection() as db:
            collection = db[self.table_name]
            return await collection.count_documents({})

    # SQLite implementations
    async def _create_sqlite(self, data: dict[str, Any]) -> None:
        """Create record in SQLite."""
        columns = list(data.keys())
        placeholders = ['?' for _ in columns]
        values = [self._serialize_json_field(data[col]) for col in columns]

        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """

        async with self.db_manager.get_connection() as conn:
            await conn.execute(query, values)
            await conn.commit()

    async def _get_by_id_sqlite(self, record_id: str) -> T | None:
        """Get record by ID from SQLite."""
        id_column = self._get_id_column()
        query = f"SELECT * FROM {self.table_name} WHERE {id_column} = ?"

        async with self.db_manager.get_connection() as conn:
            async with conn.execute(query, (record_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    # Convert row to dict
                    columns = [description[0] for description in cursor.description]
                    data = dict(zip(columns, row, strict=False))
                    data = self._deserialize_json_fields(data)
                    return self._dict_to_model(data)
        return None

    async def _update_sqlite(self, data: dict[str, Any]) -> None:
        """Update record in SQLite."""
        id_column = self._get_id_column()
        record_id = data.pop(id_column)

        if not data:
            return

        set_clauses = [f"{col} = ?" for col in data]
        values = [self._serialize_json_field(data[col]) for col in data]
        values.append(record_id)

        query = f"""
            UPDATE {self.table_name}
            SET {', '.join(set_clauses)}
            WHERE {id_column} = ?
        """

        async with self.db_manager.get_connection() as conn:
            await conn.execute(query, values)
            await conn.commit()

    async def _delete_sqlite(self, record_id: str) -> bool:
        """Delete record from SQLite."""
        id_column = self._get_id_column()
        query = f"DELETE FROM {self.table_name} WHERE {id_column} = ?"

        async with self.db_manager.get_connection() as conn:
            await conn.execute(query, (record_id,))
            await conn.commit()
            return conn.total_changes > 0

    async def _list_all_sqlite(self, limit: int | None, offset: int) -> list[T]:
        """List all records from SQLite."""
        query = f"SELECT * FROM {self.table_name} ORDER BY created_at DESC"

        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"

        async with self.db_manager.get_connection() as conn:
            async with conn.execute(query) as cursor:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]

                return [
                    self._dict_to_model(
                        self._deserialize_json_fields(dict(zip(columns, row, strict=False)))
                    )
                    for row in rows
                ]

    async def _count_sqlite(self) -> int:
        """Count records in SQLite."""
        query = f"SELECT COUNT(*) FROM {self.table_name}"

        async with self.db_manager.get_connection() as conn:
            async with conn.execute(query) as cursor:
                result = await cursor.fetchone()
                return result[0] if result else 0

    # Redis implementations (key-value storage)
    async def _create_redis(self, data: dict[str, Any]) -> None:
        """Create record in Redis."""
        id_field = self._get_id_column()
        record_id = data[id_field]
        key = f"{self.table_name}:{record_id}"

        async with self.db_manager.get_connection() as redis_client:
            await redis_client.hset(key, mapping={
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in data.items()
            })

            # Add to index set
            await redis_client.sadd(f"{self.table_name}:index", record_id)

    async def _get_by_id_redis(self, record_id: str) -> T | None:
        """Get record by ID from Redis."""
        key = f"{self.table_name}:{record_id}"

        async with self.db_manager.get_connection() as redis_client:
            data = await redis_client.hgetall(key)
            if data:
                # Deserialize JSON fields
                for k, v in data.items():
                    try:
                        data[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        pass  # Keep as string

                return self._dict_to_model(data)
        return None

    async def _update_redis(self, data: dict[str, Any]) -> None:
        """Update record in Redis."""
        id_field = self._get_id_column()
        record_id = data[id_field]
        key = f"{self.table_name}:{record_id}"

        async with self.db_manager.get_connection() as redis_client:
            await redis_client.hset(key, mapping={
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in data.items()
            })

    async def _delete_redis(self, record_id: str) -> bool:
        """Delete record from Redis."""
        key = f"{self.table_name}:{record_id}"

        async with self.db_manager.get_connection() as redis_client:
            result = await redis_client.delete(key)
            await redis_client.srem(f"{self.table_name}:index", record_id)
            return result > 0

    async def _list_all_redis(self, limit: int | None, offset: int) -> list[T]:
        """List all records from Redis."""
        async with self.db_manager.get_connection() as redis_client:
            # Get all record IDs from index
            record_ids = await redis_client.smembers(f"{self.table_name}:index")

            # Apply pagination
            record_ids = sorted(record_ids)
            if offset > 0:
                record_ids = record_ids[offset:]
            if limit:
                record_ids = record_ids[:limit]

            # Fetch records
            records = []
            for record_id in record_ids:
                record = await self._get_by_id_redis(record_id)
                if record:
                    records.append(record)

            return records

    async def _count_redis(self) -> int:
        """Count records in Redis."""
        async with self.db_manager.get_connection() as redis_client:
            return await redis_client.scard(f"{self.table_name}:index")

    # Helper methods
    def _get_id_column(self) -> str:
        """Get the ID column name for the model."""
        if self.model_class == ComputationSession:
            return 'session_id'
        elif self.model_class == InferenceResult:
            return 'result_id'
        elif self.model_class == AuditLog:
            return 'log_id'
        else:
            return 'id'

    def _serialize_json_field(self, value: Any) -> Any:
        """Serialize JSON fields for database storage."""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        return value

    def _deserialize_json_fields(self, data: dict[str, Any]) -> dict[str, Any]:
        """Deserialize JSON fields from database."""
        json_fields = self._get_json_fields()

        for field in json_fields:
            if field in data and isinstance(data[field], str):
                try:
                    data[field] = json.loads(data[field])
                except (json.JSONDecodeError, TypeError):
                    pass  # Keep as string if not valid JSON

        return data

    def _get_json_fields(self) -> list[str]:
        """Get list of fields that should be treated as JSON."""
        # Override in subclasses to specify JSON fields
        return ['metadata', 'security_config', 'performance_config', 'event_data',
                'party_ids', 'input_tokens', 'logits', 'predicted_tokens', 'confidence_scores']
