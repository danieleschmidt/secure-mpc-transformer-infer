# Multi-cloud provider configuration for global deployment

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
}

# Variables
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "mpc-transformer"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "regions" {
  description = "Map of regions to deploy to across cloud providers"
  type = object({
    aws = list(object({
      name     = string
      primary  = bool
      azs      = list(string)
    }))
    gcp = list(object({
      name     = string
      primary  = bool
      zones    = list(string)
    }))
    azure = list(object({
      name     = string
      primary  = bool
      zones    = list(string)
    }))
  })
  default = {
    aws = [
      {
        name     = "us-east-1"
        primary  = true
        azs      = ["us-east-1a", "us-east-1b", "us-east-1c"]
      },
      {
        name     = "us-west-2"
        primary  = false
        azs      = ["us-west-2a", "us-west-2b", "us-west-2c"]
      },
      {
        name     = "eu-west-1"
        primary  = false
        azs      = ["eu-west-1a", "eu-west-1b", "eu-west-1c"]
      }
    ]
    gcp = [
      {
        name     = "us-central1"
        primary  = false
        zones    = ["us-central1-a", "us-central1-b", "us-central1-c"]
      },
      {
        name     = "europe-west1"
        primary  = true
        zones    = ["europe-west1-b", "europe-west1-c", "europe-west1-d"]
      }
    ]
    azure = [
      {
        name     = "East US"
        primary  = false
        zones    = ["1", "2", "3"]
      },
      {
        name     = "Japan East"
        primary  = true
        zones    = ["1", "2", "3"]
      }
    ]
  }
}

variable "kubernetes_version" {
  description = "Kubernetes version to use"
  type        = string
  default     = "1.28"
}

variable "node_instance_types" {
  description = "Instance types for Kubernetes nodes across cloud providers"
  type = object({
    aws = object({
      cpu_nodes = string
      gpu_nodes = string
    })
    gcp = object({
      cpu_nodes = string
      gpu_nodes = string
    })
    azure = object({
      cpu_nodes = string
      gpu_nodes = string
    })
  })
  default = {
    aws = {
      cpu_nodes = "m6i.2xlarge"
      gpu_nodes = "g5.4xlarge"
    }
    gcp = {
      cpu_nodes = "n2-standard-8"
      gpu_nodes = "n1-standard-4"
    }
    azure = {
      cpu_nodes = "Standard_D8s_v4"
      gpu_nodes = "Standard_NC6s_v3"
    }
  }
}

# Local values
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Purpose     = "mpc-transformer-global"
  }
  
  # Generate unique cluster names
  aws_clusters = {
    for region in var.regions.aws : region.name => {
      name    = "${var.project_name}-${var.environment}-aws-${region.name}"
      region  = region.name
      primary = region.primary
      azs     = region.azs
    }
  }
  
  gcp_clusters = {
    for region in var.regions.gcp : region.name => {
      name    = "${var.project_name}-${var.environment}-gcp-${region.name}"
      region  = region.name
      primary = region.primary
      zones   = region.zones
    }
  }
  
  azure_clusters = {
    for region in var.regions.azure : replace(lower(region.name), " ", "-") => {
      name    = "${var.project_name}-${var.environment}-azure-${replace(lower(region.name), " ", "-")}"
      region  = region.name
      primary = region.primary
      zones   = region.zones
    }
  }
}

# AWS Provider Configuration
provider "aws" {
  alias = "primary"
  region = var.regions.aws[0].name
  
  default_tags {
    tags = local.common_tags
  }
}

# Additional AWS providers for multi-region
provider "aws" {
  for_each = { for idx, region in var.regions.aws : region.name => region if idx > 0 }
  alias    = each.key
  region   = each.value.name
  
  default_tags {
    tags = local.common_tags
  }
}

# Google Cloud Provider Configuration
provider "google" {
  project = var.project_name
  region  = var.regions.gcp[0].name
}

# Azure Provider Configuration
provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# AWS EKS Clusters
module "aws_eks_clusters" {
  source = "./modules/aws-eks"
  
  for_each = local.aws_clusters
  
  providers = {
    aws = aws.primary
  }
  
  cluster_name     = each.value.name
  cluster_version  = var.kubernetes_version
  region          = each.value.region
  availability_zones = each.value.azs
  
  # Node groups
  node_groups = {
    cpu_nodes = {
      instance_types = [var.node_instance_types.aws.cpu_nodes]
      capacity_type  = "ON_DEMAND"
      scaling_config = {
        desired_size = 3
        max_size     = 10
        min_size     = 1
      }
      
      labels = {
        "node-type" = "cpu"
        "workload"  = "mpc-transformer"
      }
      
      taints = []
    }
    
    gpu_nodes = {
      instance_types = [var.node_instance_types.aws.gpu_nodes]
      capacity_type  = "ON_DEMAND"
      scaling_config = {
        desired_size = 2
        max_size     = 8
        min_size     = 0
      }
      
      labels = {
        "node-type" = "gpu"
        "workload"  = "mpc-transformer"
        "nvidia.com/gpu.present" = "true"
      }
      
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "present"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
  
  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  tags = merge(local.common_tags, {
    "kubernetes.io/cluster/${each.value.name}" = "owned"
    "CloudProvider" = "aws"
    "Region"        = each.value.region
    "IsPrimary"     = tostring(each.value.primary)
  })
}

# Google GKE Clusters
module "gcp_gke_clusters" {
  source = "./modules/gcp-gke"
  
  for_each = local.gcp_clusters
  
  cluster_name        = each.value.name
  location           = each.value.region
  kubernetes_version = var.kubernetes_version
  zones             = each.value.zones
  
  # Node pools
  node_pools = {
    cpu_pool = {
      machine_type   = var.node_instance_types.gcp.cpu_nodes
      disk_size_gb   = 100
      disk_type      = "pd-ssd"
      node_count     = 3
      min_node_count = 1
      max_node_count = 10
      
      node_labels = {
        "node-type" = "cpu"
        "workload"  = "mpc-transformer"
      }
      
      node_taints = []
    }
    
    gpu_pool = {
      machine_type     = var.node_instance_types.gcp.gpu_nodes
      accelerator_type = "nvidia-tesla-t4"
      accelerator_count = 1
      disk_size_gb     = 100
      disk_type        = "pd-ssd"
      node_count       = 1
      min_node_count   = 0
      max_node_count   = 5
      
      node_labels = {
        "node-type" = "gpu"
        "workload"  = "mpc-transformer"
        "nvidia.com/gpu.present" = "true"
      }
      
      node_taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "present"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
  
  # Cluster features
  enable_private_nodes     = true
  enable_autopilot        = false
  enable_workload_identity = true
  
  labels = merge(local.common_tags, {
    "cloud-provider" = "gcp"
    "region"         = each.value.region
    "is-primary"     = tostring(each.value.primary)
  })
}

# Azure AKS Clusters
module "azure_aks_clusters" {
  source = "./modules/azure-aks"
  
  for_each = local.azure_clusters
  
  cluster_name           = each.value.name
  location              = each.value.region
  resource_group_name   = "${each.value.name}-rg"
  kubernetes_version    = var.kubernetes_version
  availability_zones    = each.value.zones
  
  # Node pools
  node_pools = {
    system = {
      vm_size             = var.node_instance_types.azure.cpu_nodes
      node_count          = 3
      min_count          = 1
      max_count          = 5
      enable_auto_scaling = true
      
      node_labels = {
        "kubernetes.io/role" = "system"
        "node-type"          = "system"
      }
      
      node_taints = [
        "CriticalAddonsOnly=true:NoSchedule"
      ]
    }
    
    cpu_nodes = {
      vm_size             = var.node_instance_types.azure.cpu_nodes
      node_count          = 3
      min_count          = 1
      max_count          = 10
      enable_auto_scaling = true
      
      node_labels = {
        "node-type" = "cpu"
        "workload"  = "mpc-transformer"
      }
      
      node_taints = []
    }
    
    gpu_nodes = {
      vm_size             = var.node_instance_types.azure.gpu_nodes
      node_count          = 1
      min_count          = 0
      max_count          = 5
      enable_auto_scaling = true
      
      node_labels = {
        "node-type" = "gpu"
        "workload"  = "mpc-transformer"
        "nvidia.com/gpu.present" = "true"
      }
      
      node_taints = [
        "nvidia.com/gpu=present:NoSchedule"
      ]
    }
  }
  
  # AKS features
  enable_rbac                    = true
  enable_azure_policy           = true
  enable_auto_scaling           = true
  enable_host_encryption        = true
  enable_workload_identity      = true
  
  tags = merge(local.common_tags, {
    "CloudProvider" = "azure"
    "Region"        = each.value.region
    "IsPrimary"     = tostring(each.value.primary)
  })
}

# Global DNS Configuration
resource "aws_route53_zone" "global" {
  provider = aws.primary
  name     = "${var.project_name}.com"
  
  tags = merge(local.common_tags, {
    "Type" = "global-dns"
  })
}

# Global load balancer health checks
resource "aws_route53_health_check" "cluster_health" {
  provider = aws.primary
  
  for_each = merge(
    { for name, cluster in local.aws_clusters : "aws-${name}" => cluster },
    { for name, cluster in local.gcp_clusters : "gcp-${name}" => cluster },
    { for name, cluster in local.azure_clusters : "azure-${name}" => cluster }
  )
  
  fqdn                            = "${each.key}.${var.project_name}.com"
  port                           = 443
  type                          = "HTTPS"
  resource_path                 = "/health"
  failure_threshold             = 3
  request_interval              = 30
  measure_latency               = true
  
  tags = merge(local.common_tags, {
    "Type"    = "health-check"
    "Cluster" = each.key
  })
}

# Cross-cloud networking (VPC peering, VPN, etc.)
module "cross_cloud_networking" {
  source = "./modules/cross-cloud-networking"
  
  aws_vpcs = {
    for name, cluster in module.aws_eks_clusters : name => {
      vpc_id     = cluster.vpc_id
      cidr_block = cluster.vpc_cidr_block
      region     = cluster.region
    }
  }
  
  gcp_networks = {
    for name, cluster in module.gcp_gke_clusters : name => {
      network_name = cluster.network_name
      subnet_cidr  = cluster.subnet_cidr
      region       = cluster.region
    }
  }
  
  azure_vnets = {
    for name, cluster in module.azure_aks_clusters : name => {
      vnet_id     = cluster.vnet_id
      address_space = cluster.vnet_address_space
      region      = cluster.region
    }
  }
  
  tags = local.common_tags
}

# Global secrets management
module "global_secrets" {
  source = "./modules/global-secrets"
  
  clusters = merge(
    { for name, cluster in module.aws_eks_clusters : "aws-${name}" => {
      provider = "aws"
      cluster_name = cluster.cluster_name
      region = cluster.region
    }},
    { for name, cluster in module.gcp_gke_clusters : "gcp-${name}" => {
      provider = "gcp"
      cluster_name = cluster.cluster_name
      region = cluster.region
    }},
    { for name, cluster in module.azure_aks_clusters : "azure-${name}" => {
      provider = "azure"
      cluster_name = cluster.cluster_name
      region = cluster.region
    }}
  )
  
  secrets = {
    "mpc-tls-certs" = {
      type = "tls"
      data = {
        "tls.crt" = file("${path.module}/certs/tls.crt")
        "tls.key" = file("${path.module}/certs/tls.key")
        "ca.crt"  = file("${path.module}/certs/ca.crt")
      }
    }
    
    "mpc-api-keys" = {
      type = "generic"
      data = {
        "jwt-secret"     = random_password.jwt_secret.result
        "encryption-key" = random_password.encryption_key.result
      }
    }
  }
  
  tags = local.common_tags
}

# Random passwords for secrets
resource "random_password" "jwt_secret" {
  length  = 64
  special = true
}

resource "random_password" "encryption_key" {
  length  = 32
  special = false
}

# Outputs
output "aws_clusters" {
  description = "AWS EKS cluster information"
  value = {
    for name, cluster in module.aws_eks_clusters : name => {
      cluster_name     = cluster.cluster_name
      cluster_endpoint = cluster.cluster_endpoint
      region          = cluster.region
      vpc_id          = cluster.vpc_id
    }
  }
}

output "gcp_clusters" {
  description = "GCP GKE cluster information"
  value = {
    for name, cluster in module.gcp_gke_clusters : name => {
      cluster_name     = cluster.cluster_name
      cluster_endpoint = cluster.cluster_endpoint
      region          = cluster.region
      network_name    = cluster.network_name
    }
  }
}

output "azure_clusters" {
  description = "Azure AKS cluster information"
  value = {
    for name, cluster in module.azure_aks_clusters : name => {
      cluster_name     = cluster.cluster_name
      cluster_endpoint = cluster.cluster_endpoint
      region          = cluster.region
      resource_group  = cluster.resource_group_name
    }
  }
}

output "global_dns_zone" {
  description = "Global DNS zone information"
  value = {
    zone_id     = aws_route53_zone.global.zone_id
    name_servers = aws_route53_zone.global.name_servers
    domain      = aws_route53_zone.global.name
  }
}

output "health_checks" {
  description = "Health check information"
  value = {
    for name, check in aws_route53_health_check.cluster_health : name => {
      health_check_id = check.id
      fqdn           = check.fqdn
    }
  }
}