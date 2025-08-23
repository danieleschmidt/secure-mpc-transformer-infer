#!/bin/bash
# Database Backup Script

set -e

# Configuration
DB_NAME="secure_mpc"
DB_USER="app_user"
BACKUP_DIR="/app/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/backup_${DB_NAME}_${DATE}.sql"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Perform backup
echo "Starting database backup..."
pg_dump -h postgres -U "$DB_USER" "$DB_NAME" > "$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_FILE"
BACKUP_FILE="${BACKUP_FILE}.gz"

echo "Backup completed: $BACKUP_FILE"

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "backup_${DB_NAME}_*.sql.gz" -mtime +7 -delete

# Upload to cloud storage (optional)
if [ -n "$S3_BUCKET" ]; then
    aws s3 cp "$BACKUP_FILE" "s3://$S3_BUCKET/backups/"
    echo "Backup uploaded to S3"
fi

echo "Backup process completed successfully"
