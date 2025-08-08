# Incident Response Runbook - MPC Transformer Production

## Overview

This runbook provides step-by-step procedures for responding to production incidents in the MPC Transformer global deployment. It covers incident classification, response procedures, escalation paths, and post-incident activities.

## Incident Classification

### Severity Levels

#### P0 - Critical (Page Immediately)
- **Definition**: Complete service outage or critical security breach
- **Response Time**: Immediate (0-5 minutes)
- **Examples**:
  - Global service unavailable
  - Multiple regions down
  - Security breach detected
  - Data loss or corruption
  - Customer-facing SLA breach >99.99%

#### P1 - High (Page During Business Hours)
- **Definition**: Significant service degradation
- **Response Time**: 15 minutes
- **Examples**:
  - Single region outage
  - High error rate (>5%)
  - Performance degradation
  - GPU resource exhaustion
  - MPC protocol failures

#### P2 - Medium
- **Definition**: Minor service impact
- **Response Time**: 1 hour
- **Examples**:
  - Non-critical feature degradation
  - Monitoring alerts
  - Resource utilization warnings
  - SSL certificate expiration warnings

#### P3 - Low
- **Definition**: No immediate service impact
- **Response Time**: Next business day
- **Examples**:
  - Documentation updates
  - Minor configuration issues
  - Capacity planning alerts

## Incident Response Process

### 1. Detection and Alerting

#### Automated Detection
- **Prometheus Alerts**: Configured for all severity levels
- **Health Check Failures**: External monitoring via Pingdom/DataDog
- **Customer Reports**: Zendesk integration
- **Security Monitoring**: SIEM alerts

#### Manual Detection
- **Dashboard Anomalies**: Grafana dashboard monitoring
- **Customer Complaints**: Support ticket escalation
- **Team Observations**: Slack notifications

### 2. Initial Response (0-5 minutes)

#### For P0 Incidents:

1. **Acknowledge the Alert**
   ```bash
   # Acknowledge in PagerDuty
   curl -X PUT "https://api.pagerduty.com/incidents/{incident_id}" \
     -H "Authorization: Token token={API_TOKEN}" \
     -H "Content-Type: application/json" \
     -d '{"incident": {"type": "incident_reference", "status": "acknowledged"}}'
   ```

2. **Join Incident Response Channel**
   ```
   Slack: #incident-response
   Bridge: +1-555-INCIDENT
   Zoom: https://zoom.us/j/incident-room
   ```

3. **Declare Incident**
   ```
   /incident declare "P0: Global service outage - investigating"
   ```

4. **Quick Triage**
   - Check global dashboard: https://grafana.mpc-transformer.com/d/mpc-global-overview
   - Verify alert authenticity
   - Identify affected regions/services
   - Estimate impact scope

#### For P1 Incidents:

1. **Acknowledge Alert** (within 15 minutes)
2. **Create Incident Ticket**
3. **Initial Assessment**
4. **Notify Team Lead**

### 3. Investigation and Diagnosis

#### Data Gathering Commands

```bash
# Check service status globally
kubectl get pods -A -l app=mpc-transformer

# Check recent deployments
kubectl get deployments -A -l app=mpc-transformer -o wide

# Check resource utilization
kubectl top nodes
kubectl top pods -A -l app=mpc-transformer

# Check logs from all regions
kubectl logs -l app=mpc-transformer -n mpc-transformer-americas --tail=100
kubectl logs -l app=mpc-transformer -n mpc-transformer-europe --tail=100
kubectl logs -l app=mpc-transformer -n mpc-transformer-apac --tail=100

# Check external connectivity
for region in us-east us-west eu-west ap-northeast; do
  echo "Testing $region..."
  curl -f https://${region}.mpc-transformer.com/health || echo "FAILED: $region"
done

# Check database status
kubectl exec -it postgres-0 -n data -- psql -c "SELECT version();"

# Check GPU status
kubectl exec -it $(kubectl get pods -l app=mpc-transformer,node-type=gpu -o name | head -1) -- nvidia-smi
```

#### Common Investigation Areas

1. **Infrastructure Issues**
   - Kubernetes cluster health
   - Node availability and resources
   - Network connectivity
   - DNS resolution

2. **Application Issues**
   - Service logs and errors
   - Database connectivity
   - MPC protocol failures
   - GPU availability

3. **External Dependencies**
   - Cloud provider status
   - External API availability
   - Network peering status

### 4. Mitigation and Resolution

#### Common Mitigation Strategies

1. **Traffic Routing**
   ```bash
   # Drain traffic from problematic region
   kubectl annotate service mpc-lb -n mpc-transformer-americas \
     traffic.mpc-transformer.com/weight=0
   
   # Increase traffic to healthy regions
   kubectl annotate service mpc-lb -n mpc-transformer-europe \
     traffic.mpc-transformer.com/weight=50
   ```

2. **Scaling Operations**
   ```bash
   # Scale up replicas
   kubectl scale deployment mpc-transformer \
     -n mpc-transformer-europe --replicas=10
   
   # Scale up GPU nodes
   kubectl patch deployment mpc-transformer \
     -n mpc-transformer-europe -p '{"spec":{"template":{"spec":{"nodeSelector":{"node-type":"gpu"}}}}}'
   ```

3. **Rollback Procedures**
   ```bash
   # Rollback to previous version
   kubectl rollout undo deployment/mpc-transformer -n mpc-transformer-americas
   
   # Check rollback status
   kubectl rollout status deployment/mpc-transformer -n mpc-transformer-americas
   ```

4. **Emergency Maintenance Mode**
   ```bash
   # Enable maintenance mode
   kubectl patch configmap app-config -n mpc-transformer-americas \
     -p '{"data":{"maintenance_mode":"true"}}'
   
   # Update all pods
   kubectl rollout restart deployment/mpc-transformer -n mpc-transformer-americas
   ```

### 5. Communication

#### Internal Communication

1. **Incident Commander**: Coordinates response
2. **Technical Lead**: Drives technical resolution
3. **Communication Lead**: Manages stakeholder updates

#### External Communication

1. **Status Page Updates**
   ```bash
   # Update status page
   curl -X POST "https://api.statuspage.io/v1/pages/{PAGE_ID}/incidents" \
     -H "Authorization: OAuth {TOKEN}" \
     -d "incident[name]=Service Disruption in US Region" \
     -d "incident[status]=investigating"
   ```

2. **Customer Notifications**
   - Email alerts for enterprise customers
   - In-app notifications
   - Social media updates (@MPCTransformer)

#### Communication Templates

**Initial Update (within 10 minutes)**:
```
🚨 INCIDENT ALERT - P0
We are investigating reports of service disruption affecting [REGION/COMPONENT].
Our team is actively working on a resolution.
Status: https://status.mpc-transformer.com
Updates: Every 15 minutes
ETA: Under investigation
```

**Progress Update (every 15-30 minutes)**:
```
📋 INCIDENT UPDATE - [TIME]
Issue: [BRIEF DESCRIPTION]
Impact: [AFFECTED SERVICES/REGIONS]
Root Cause: [IF KNOWN]
Current Actions: [WHAT WE'RE DOING]
Next Update: [TIME]
```

**Resolution Update**:
```
✅ INCIDENT RESOLVED - [TIME]
Issue: [DESCRIPTION]
Resolution: [WHAT WE FIXED]
Duration: [TOTAL TIME]
Next Steps: Post-incident review scheduled for [DATE]
```

## Escalation Procedures

### Escalation Matrix

| Severity | Initial Response | 30 Minutes | 1 Hour | 2 Hours |
|----------|------------------|------------|---------|---------|
| P0 | On-call Engineer | Team Lead + VP Engineering | CTO + CEO | All Hands |
| P1 | On-call Engineer | Team Lead | VP Engineering | CTO |
| P2 | On-call Engineer | Team Lead | - | - |
| P3 | Assignee | - | - | - |

### Contact Information

```yaml
# Escalation Contacts
primary_oncall: "+1-555-ON-CALL"
team_lead: "+1-555-TEAM-LEAD" 
vp_engineering: "+1-555-VP-ENG"
cto: "+1-555-CTO"

# Emergency Services
aws_support: "+1-555-AWS-SUPPORT" (Enterprise)
gcp_support: "+1-555-GCP-SUPPORT" (Premium)
azure_support: "+1-555-AZURE-SUPPORT" (Professional)

# Vendor Contacts
cloudflare: "+1-855-435-7827"
datadog: "+1-866-329-4466"
pagerduty: "+1-844-732-3349"
```

## Specific Incident Playbooks

### Global Service Outage

1. **Immediate Actions** (0-2 minutes)
   ```bash
   # Check global health endpoints
   ./scripts/check-global-health.sh
   
   # Check DNS resolution
   dig api.mpc-transformer.com
   dig eu.mpc-transformer.com  
   dig asia.mpc-transformer.com
   ```

2. **Traffic Routing** (2-5 minutes)
   ```bash
   # Route to healthy regions only
   ./scripts/emergency-traffic-routing.sh --exclude-down-regions
   ```

3. **Scale Up** (5-10 minutes)
   ```bash
   # Emergency scale up in all healthy regions
   ./scripts/emergency-scale-up.sh --regions=healthy --factor=2
   ```

### Single Region Outage

1. **Verify Outage Scope**
   ```bash
   # Check region-specific health
   curl -f https://us-east.mpc-transformer.com/health
   kubectl get pods -n mpc-transformer-americas
   ```

2. **Redirect Traffic**
   ```bash
   # Update DNS to remove failed region
   ./scripts/dns-failover.sh --remove-region=us-east
   ```

3. **Investigate Infrastructure**
   ```bash
   # Check cluster status
   kubectl cluster-info --context=us-east-cluster
   
   # Check node status  
   kubectl get nodes --context=us-east-cluster
   ```

### High Error Rate

1. **Identify Error Sources**
   ```bash
   # Check error breakdown by region
   curl -G 'http://prometheus:9090/api/v1/query' \
     --data-urlencode 'query=sum by (region) (rate(mpc_transformer_errors_total[5m]))'
   ```

2. **Check Recent Changes**
   ```bash
   # Recent deployments
   kubectl rollout history deployment/mpc-transformer -n mpc-transformer-americas
   
   # Recent config changes
   git log --oneline --since="1 hour ago" config/
   ```

3. **Circuit Breaker**
   ```bash
   # Enable circuit breaker for failing endpoints
   kubectl patch configmap circuit-breaker-config \
     -p '{"data":{"mpc_protocol_timeout":"30s","max_failures":"5"}}'
   ```

### GPU Resource Exhaustion

1. **Check GPU Status**
   ```bash
   # GPU utilization across regions
   kubectl exec -it $(kubectl get pods -l node-type=gpu -o name | head -1) -- nvidia-smi
   ```

2. **Scale GPU Nodes**
   ```bash
   # Scale up GPU node pool
   kubectl scale deployment mpc-transformer-gpu \
     --replicas=8 -n mpc-transformer-americas
   ```

3. **Optimize Resource Allocation**
   ```bash
   # Reduce GPU memory allocation temporarily
   kubectl patch deployment mpc-transformer -p '{"spec":{"template":{"spec":{"containers":[{"name":"mpc-transformer","env":[{"name":"GPU_MEMORY_FRACTION","value":"0.7"}]}]}}}}'
   ```

### Security Incident

1. **Immediate Containment** (0-5 minutes)
   ```bash
   # Isolate affected components
   kubectl patch networkpolicy default-deny -p '{"spec":{"ingress":[],"egress":[]}}'
   ```

2. **Evidence Collection** (5-15 minutes)
   ```bash
   # Collect logs
   kubectl logs -l app=mpc-transformer --since=1h > incident-logs.txt
   
   # Snapshot system state
   kubectl get all -A -o yaml > cluster-state.yaml
   ```

3. **Notify Security Team**
   ```bash
   # Escalate to security team immediately
   curl -X POST https://hooks.slack.com/services/T00/B00/XXX \
     -d '{"text":"🚨 Security incident detected - immediate attention required"}'
   ```

## Post-Incident Activities

### 1. Service Restoration Verification

```bash
# Verify all regions are healthy
./scripts/verify-global-health.sh

# Check SLA metrics
curl -G 'http://prometheus:9090/api/v1/query' \
  --data-urlencode 'query=mpc_transformer:global_sla_1h'

# Validate end-to-end functionality
./scripts/e2e-test.sh --all-regions
```

### 2. Post-Incident Review (PIR)

#### Timeline Template
```
## Incident Timeline

| Time (UTC) | Event | Action Taken | Person |
|------------|--------|--------------|--------|
| 14:32 | Alert fired | Acknowledged | @oncall |
| 14:35 | Investigation started | Checked logs | @oncall |
| 14:40 | Root cause identified | Applied fix | @team-lead |
| 14:45 | Service restored | Verified health | @oncall |

## Impact Assessment
- **Duration**: 13 minutes
- **Users Affected**: ~50,000 (estimated)
- **Revenue Impact**: $X,XXX
- **SLA Breach**: 99.97% (below 99.99% target)

## Root Cause Analysis
[Detailed technical explanation]

## Action Items
1. [ ] Fix underlying issue (@engineer, Due: Date)
2. [ ] Improve monitoring (@sre, Due: Date)
3. [ ] Update runbook (@team-lead, Due: Date)
```

### 3. Follow-up Actions

1. **Update Monitoring**
   - Add new alerts based on incident
   - Adjust alert thresholds
   - Enhance dashboards

2. **Update Documentation**
   - Update runbooks
   - Document new procedures
   - Update architecture diagrams

3. **Process Improvements**
   - Review response time
   - Improve automation
   - Update escalation procedures

## Tools and Resources

### Monitoring and Observability
- **Grafana**: https://grafana.mpc-transformer.com
- **Prometheus**: https://prometheus.mpc-transformer.com
- **AlertManager**: https://alertmanager.mpc-transformer.com
- **Jaeger**: https://jaeger.mpc-transformer.com

### Communication
- **Slack**: #incident-response, #ops-alerts
- **PagerDuty**: https://mpc-transformer.pagerduty.com
- **Status Page**: https://status.mpc-transformer.com
- **Zoom Room**: https://zoom.us/j/incident-room

### Infrastructure
- **Kubernetes**: kubectl contexts for all regions
- **Terraform**: Infrastructure state and modifications
- **Ansible**: Configuration management
- **Scripts**: `/scripts/` directory in this repository

### External Services
- **AWS Console**: https://console.aws.amazon.com
- **GCP Console**: https://console.cloud.google.com  
- **Azure Portal**: https://portal.azure.com
- **Cloudflare**: https://dash.cloudflare.com

## Emergency Contacts

### Internal Team
| Role | Name | Phone | Email | Slack |
|------|------|-------|-------|-------|
| On-call Engineer | Rotating | +1-555-ON-CALL | oncall@mpc-transformer.com | @oncall |
| Team Lead | Alice Johnson | +1-555-TEAM-LEAD | alice@mpc-transformer.com | @alice |
| VP Engineering | Bob Smith | +1-555-VP-ENG | bob@mpc-transformer.com | @bob |
| CTO | Carol Davis | +1-555-CTO | carol@mpc-transformer.com | @carol |

### External Vendors
| Service | Support Number | Account ID | Notes |
|---------|----------------|------------|--------|
| AWS | +1-800-AWS-SUPPORT | 123456789 | Enterprise Support |
| GCP | +1-855-GCP-SUPPORT | gcp-project-123 | Premium Support |
| Azure | +1-800-AZURE-SUPPORT | sub-123-456 | Professional Support |
| Cloudflare | +1-855-CF-SUPPORT | cf-account-123 | Enterprise Plan |

---

**Document Version**: 2.0
**Last Updated**: 2024-08-08
**Next Review**: 2024-09-08
**Owner**: Platform Team