#!/bin/bash
# Generate self-signed SSL certificate for development/testing
# For production, use certificates from a trusted CA

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout ssl/key.pem \
    -out ssl/cert.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"

chmod 600 ssl/key.pem
chmod 644 ssl/cert.pem

echo "✅ SSL certificates generated in ssl/ directory"
echo "⚠️  Note: These are self-signed certificates for development only"
echo "🔒 For production, obtain certificates from a trusted CA"
