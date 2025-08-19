"""Database connection management."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

# Database drivers
try:
    import asyncpg  # PostgreSQL
except ImportError:
    asyncpg = None

try:
    import motor.motor_asyncio  # MongoDB
except ImportError:
    motor = None

try:
    import redis.asyncio as redis  # Redis
except ImportError:
    redis = None

try:
    import aiosqlite  # SQLite
except ImportError:
    aiosqlite = None


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    REDIS = "redis"
    SQLITE = "sqlite"


class DatabaseManager:
    """Manages database connections and transactions."""

    def __init__(self, database_type: DatabaseType, connection_config: dict[str, Any]):
        self.database_type = database_type
        self.connection_config = connection_config
        self.pool = None
        self.redis_client = None
        self.mongo_client = None
        self.sqlite_path = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize database connections."""
        try:
            if self.database_type == DatabaseType.POSTGRESQL:
                await self._init_postgresql()
            elif self.database_type == DatabaseType.MONGODB:
                await self._init_mongodb()
            elif self.database_type == DatabaseType.REDIS:
                await self._init_redis()
            elif self.database_type == DatabaseType.SQLITE:
                await self._init_sqlite()
            else:
                raise ValueError(f"Unsupported database type: {self.database_type}")

            self.logger.info(f"Database connection initialized: {self.database_type.value}")

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise

    async def _init_postgresql(self) -> None:
        """Initialize PostgreSQL connection pool."""
        if asyncpg is None:
            raise ImportError("asyncpg not installed. Install with: pip install asyncpg")

        dsn = (
            f"postgresql://{self.connection_config['user']}:"
            f"{self.connection_config['password']}@"
            f"{self.connection_config['host']}:"
            f"{self.connection_config['port']}/"
            f"{self.connection_config['database']}"
        )

        self.pool = await asyncpg.create_pool(
            dsn,
            min_size=5,
            max_size=20,
            command_timeout=60,
            server_settings={
                'jit': 'off',  # Disable JIT for better cold start performance
                'application_name': 'secure-mpc-transformer'
            }
        )

    async def _init_mongodb(self) -> None:
        """Initialize MongoDB connection."""
        if motor is None:
            raise ImportError("motor not installed. Install with: pip install motor")

        uri = self.connection_config.get('uri') or (
            f"mongodb://{self.connection_config['host']}:"
            f"{self.connection_config['port']}"
        )

        self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
            uri,
            maxPoolSize=20,
            minPoolSize=5,
            serverSelectionTimeoutMS=5000
        )

        # Test connection
        await self.mongo_client.admin.command('ping')

    async def _init_redis(self) -> None:
        """Initialize Redis connection."""
        if redis is None:
            raise ImportError("redis not installed. Install with: pip install redis[hiredis]")

        self.redis_client = redis.Redis(
            host=self.connection_config['host'],
            port=self.connection_config['port'],
            password=self.connection_config.get('password'),
            db=self.connection_config.get('db', 0),
            decode_responses=True,
            max_connections=20
        )

        # Test connection
        await self.redis_client.ping()

    async def _init_sqlite(self) -> None:
        """Initialize SQLite connection."""
        if aiosqlite is None:
            raise ImportError("aiosqlite not installed. Install with: pip install aiosqlite")

        self.sqlite_path = self.connection_config.get('path', 'secure_mpc_transformer.db')

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.sqlite_path) or '.', exist_ok=True)

        # Test connection
        async with aiosqlite.connect(self.sqlite_path) as db:
            await db.execute('SELECT 1')

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection context manager."""
        if self.database_type == DatabaseType.POSTGRESQL:
            async with self.pool.acquire() as connection:
                yield connection
        elif self.database_type == DatabaseType.MONGODB:
            yield self.mongo_client[self.connection_config['database']]
        elif self.database_type == DatabaseType.REDIS:
            yield self.redis_client
        elif self.database_type == DatabaseType.SQLITE:
            async with aiosqlite.connect(self.sqlite_path) as connection:
                yield connection
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")

    @asynccontextmanager
    async def transaction(self):
        """Database transaction context manager."""
        if self.database_type == DatabaseType.POSTGRESQL:
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    yield connection
        elif self.database_type == DatabaseType.MONGODB:
            # MongoDB transactions require replica set
            async with await self.mongo_client.start_session() as session:
                async with session.start_transaction():
                    yield self.mongo_client[self.connection_config['database']]
        elif self.database_type == DatabaseType.SQLITE:
            async with aiosqlite.connect(self.sqlite_path) as connection:
                try:
                    await connection.execute('BEGIN')
                    yield connection
                    await connection.commit()
                except Exception:
                    await connection.rollback()
                    raise
        else:
            # Redis doesn't support transactions in the same way
            async with self.get_connection() as connection:
                yield connection

    async def execute_query(self, query: str, params: tuple | None = None) -> Any:
        """Execute a database query."""
        async with self.get_connection() as connection:
            if self.database_type == DatabaseType.POSTGRESQL:
                if params:
                    return await connection.fetch(query, *params)
                else:
                    return await connection.fetch(query)
            elif self.database_type == DatabaseType.SQLITE:
                if params:
                    async with connection.execute(query, params) as cursor:
                        return await cursor.fetchall()
                else:
                    async with connection.execute(query) as cursor:
                        return await cursor.fetchall()
            else:
                raise NotImplementedError(f"Query execution not implemented for {self.database_type}")

    async def execute_update(self, query: str, params: tuple | None = None) -> int:
        """Execute an update/insert/delete query."""
        async with self.get_connection() as connection:
            if self.database_type == DatabaseType.POSTGRESQL:
                if params:
                    result = await connection.execute(query, *params)
                else:
                    result = await connection.execute(query)
                # Extract number of affected rows from result
                return int(result.split()[-1]) if result else 0
            elif self.database_type == DatabaseType.SQLITE:
                if params:
                    await connection.execute(query, params)
                else:
                    await connection.execute(query)
                return connection.total_changes
            else:
                raise NotImplementedError(f"Update execution not implemented for {self.database_type}")

    async def close(self) -> None:
        """Close database connections."""
        try:
            if self.pool:
                await self.pool.close()
            if self.redis_client:
                await self.redis_client.close()
            if self.mongo_client:
                self.mongo_client.close()

            self.logger.info("Database connections closed")

        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")

    async def health_check(self) -> dict[str, Any]:
        """Perform database health check."""
        try:
            start_time = asyncio.get_event_loop().time()

            if self.database_type == DatabaseType.POSTGRESQL:
                async with self.get_connection() as conn:
                    await conn.fetch('SELECT 1')
            elif self.database_type == DatabaseType.MONGODB:
                await self.mongo_client.admin.command('ping')
            elif self.database_type == DatabaseType.REDIS:
                await self.redis_client.ping()
            elif self.database_type == DatabaseType.SQLITE:
                async with self.get_connection() as conn:
                    await conn.execute('SELECT 1')

            response_time = (asyncio.get_event_loop().time() - start_time) * 1000

            return {
                "status": "healthy",
                "database_type": self.database_type.value,
                "response_time_ms": round(response_time, 2),
                "timestamp": asyncio.get_event_loop().time()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "database_type": self.database_type.value,
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }

    @classmethod
    def from_env(cls) -> "DatabaseManager":
        """Create DatabaseManager from environment variables."""
        db_type = DatabaseType(os.getenv('DATABASE_TYPE', 'postgresql'))

        if db_type == DatabaseType.POSTGRESQL:
            config = {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': int(os.getenv('POSTGRES_PORT', 5432)),
                'database': os.getenv('POSTGRES_DB', 'secure_mpc_transformer'),
                'user': os.getenv('POSTGRES_USER', 'mpc_user'),
                'password': os.getenv('POSTGRES_PASSWORD', 'password')
            }
        elif db_type == DatabaseType.MONGODB:
            config = {
                'uri': os.getenv('MONGO_URI'),
                'host': os.getenv('MONGO_HOST', 'localhost'),
                'port': int(os.getenv('MONGO_PORT', 27017)),
                'database': os.getenv('MONGO_DB', 'secure_mpc_transformer')
            }
        elif db_type == DatabaseType.REDIS:
            config = {
                'host': os.getenv('REDIS_HOST', 'localhost'),
                'port': int(os.getenv('REDIS_PORT', 6379)),
                'password': os.getenv('REDIS_PASSWORD'),
                'db': int(os.getenv('REDIS_DB', 0))
            }
        elif db_type == DatabaseType.SQLITE:
            config = {
                'path': os.getenv('SQLITE_PATH', 'data/secure_mpc_transformer.db')
            }
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        return cls(db_type, config)

    def get_stats(self) -> dict[str, Any]:
        """Get database connection statistics."""
        stats = {
            "database_type": self.database_type.value,
            "connection_config": {k: v for k, v in self.connection_config.items() if 'password' not in k.lower()}
        }

        if self.database_type == DatabaseType.POSTGRESQL and self.pool:
            stats.update({
                "pool_size": self.pool.get_size(),
                "pool_free_size": self.pool.get_idle_size(),
                "pool_max_size": self.pool.get_max_size(),
                "pool_min_size": self.pool.get_min_size()
            })

        return stats
