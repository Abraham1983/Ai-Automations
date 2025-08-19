# Installation Guide

## Prerequisites

Before installing the AI Automation Platform, ensure you have the following:

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 2GB free space
- **OS**: Linux, macOS, or Windows

### Required Services
- **Database**: PostgreSQL 12+ (recommended) or MySQL 8+
- **Cache**: Redis 6+ (for session management and caching)
- **AI Services**: OpenAI API access (required for AI features)

### Optional Services
- **Docker**: For containerized deployment
- **Kubernetes**: For scalable production deployments
- **Cloud Storage**: AWS S3, Azure Blob, or Google Cloud Storage

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Abraham1983/Ai-Automations.git
cd Ai-Automations
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy configuration template
cp config/settings.yaml.example config/settings.yaml

# Edit configuration file
nano config/settings.yaml
```

### 5. Set Environment Variables

Create a `.env` file in the root directory:

```bash
# AI Services
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_automation
DB_USER=postgres
DB_PASSWORD=your_db_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Security
SECRET_KEY=your_secret_key_here

# Optional: Email
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_email_password
```

### 6. Initialize Database

```bash
# Run database migrations
python -m alembic upgrade head

# Create initial data (optional)
python scripts/create_sample_data.py
```

### 7. Start the Application

```bash
# Development mode
python examples/quick_start.py

# Production mode
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Detailed Installation

### Database Setup

#### PostgreSQL (Recommended)

1. **Install PostgreSQL**:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install postgresql postgresql-contrib
   
   # macOS (using Homebrew)
   brew install postgresql
   brew services start postgresql
   
   # Windows
   # Download installer from https://www.postgresql.org/download/windows/
   ```

2. **Create Database**:
   ```bash
   sudo -u postgres psql
   CREATE DATABASE ai_automation;
   CREATE USER ai_user WITH ENCRYPTED PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE ai_automation TO ai_user;
   \q
   ```

#### Redis Setup

1. **Install Redis**:
   ```bash
   # Ubuntu/Debian
   sudo apt install redis-server
   
   # macOS (using Homebrew)
   brew install redis
   brew services start redis
   
   # Windows
   # Download from https://github.com/microsoftarchive/redis/releases
   ```

2. **Configure Redis** (optional):
   ```bash
   sudo nano /etc/redis/redis.conf
   # Uncomment and set: requirepass your_redis_password
   sudo systemctl restart redis
   ```

### AI Service Setup

#### OpenAI API

1. **Get API Key**:
   - Visit [OpenAI Platform](https://platform.openai.com/)
   - Create account or sign in
   - Navigate to API Keys section
   - Create new API key

2. **Set Usage Limits** (recommended):
   - Set monthly usage limits in OpenAI dashboard
   - Monitor usage to avoid unexpected charges

#### Anthropic API (Optional)

1. **Get API Key**:
   - Visit [Anthropic Console](https://console.anthropic.com/)
   - Create account and get API access
   - Generate API key

### Python Environment Setup

#### Using pyenv (Recommended)

```bash
# Install pyenv
curl https://pyenv.run | bash

# Install Python 3.9
pyenv install 3.9.16
pyenv global 3.9.16

# Create virtual environment
python -m venv ai-automation-env
source ai-automation-env/bin/activate
```

#### Using conda

```bash
# Create conda environment
conda create -n ai-automation python=3.9
conda activate ai-automation

# Install pip packages
pip install -r requirements.txt
```

## Docker Installation

### Quick Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Docker Build

```bash
# Build image
docker build -t ai-automation .

# Run container
docker run -d \
  --name ai-automation \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e DB_HOST=your_db_host \
  ai-automation
```

### Docker Compose Configuration

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DB_HOST=postgres
      - REDIS_HOST=redis
    depends_on:
      - postgres
      - redis
    volumes:
      - ./config:/app/config
      - ./data:/app/data

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=ai_automation
      - POSTGRES_USER=ai_user
      - POSTGRES_PASSWORD=your_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Kubernetes Deployment

### Basic Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/

# Check deployment status
kubectl get pods -l app=ai-automation

# Get service URL
kubectl get services
```

### Helm Chart (Advanced)

```bash
# Add Helm repository (if available)
helm repo add ai-automation https://charts.ai-automation.com
helm repo update

# Install with Helm
helm install ai-automation ai-automation/ai-automation \
  --set config.openai.apiKey=your_key \
  --set config.database.host=your_db_host
```

## Cloud Deployment

### AWS Deployment

1. **Using AWS ECS**:
   ```bash
   # Build and push to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
   docker build -t ai-automation .
   docker tag ai-automation:latest your-account.dkr.ecr.us-east-1.amazonaws.com/ai-automation:latest
   docker push your-account.dkr.ecr.us-east-1.amazonaws.com/ai-automation:latest
   ```

2. **Using AWS Lambda** (for specific functions):
   ```bash
   # Install serverless framework
   npm install -g serverless
   
   # Deploy Lambda functions
   cd deployment/aws/lambda
   serverless deploy
   ```

### Azure Deployment

```bash
# Login to Azure
az login

# Create resource group
az group create --name ai-automation-rg --location eastus

# Deploy container instance
az container create \
  --resource-group ai-automation-rg \
  --name ai-automation \
  --image your-registry/ai-automation:latest \
  --ports 8000 \
  --environment-variables OPENAI_API_KEY=your_key
```

### Google Cloud Deployment

```bash
# Setup gcloud
gcloud auth login
gcloud config set project your-project-id

# Deploy to Cloud Run
gcloud run deploy ai-automation \
  --image gcr.io/your-project-id/ai-automation \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Development Installation

### Additional Dev Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with hot reload
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### IDE Setup

#### VS Code

Install recommended extensions:
- Python
- Pylance
- Black Formatter
- autoDocstring

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.testing.pytestEnabled": true
}
```

#### PyCharm

1. Open project in PyCharm
2. Configure Python interpreter: Settings > Project > Python Interpreter
3. Set up code style: Settings > Editor > Code Style > Python > Black
4. Configure pytest: Settings > Tools > Python Integrated Tools

## Verification

### Health Check

```bash
# Check application health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "services": {
    "database": "connected",
    "redis": "connected",
    "ai_models": "available"
  }
}
```

### Run Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_customer_experience.py
pytest tests/test_sales_automation.py
pytest tests/test_research_automation.py

# Run with coverage
pytest --cov=src tests/
```

### Quick Functional Test

```bash
# Run the quick start demo
python examples/quick_start.py

# Test individual modules
python examples/customer_demo.py
python examples/sales_demo.py
python examples/research_demo.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate
   
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

2. **Database Connection Issues**:
   ```bash
   # Check database status
   pg_isready -h localhost -p 5432
   
   # Test connection
   psql -h localhost -U ai_user -d ai_automation
   ```

3. **Redis Connection Issues**:
   ```bash
   # Check Redis status
   redis-cli ping
   
   # Expected response: PONG
   ```

4. **API Key Issues**:
   ```bash
   # Verify environment variables
   echo $OPENAI_API_KEY
   
   # Test API connection
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models
   ```

### Performance Optimization

1. **Database Optimization**:
   ```sql
   -- Create indexes for better performance
   CREATE INDEX idx_leads_score ON leads(score);
   CREATE INDEX idx_conversations_customer_id ON conversations(customer_id);
   CREATE INDEX idx_reports_created_at ON reports(created_at);
   ```

2. **Redis Configuration**:
   ```bash
   # Optimize Redis memory usage
   redis-cli CONFIG SET maxmemory 256mb
   redis-cli CONFIG SET maxmemory-policy allkeys-lru
   ```

3. **Application Tuning**:
   ```yaml
   # In config/settings.yaml
   performance:
     cache:
       ttl_default: 3600
     batch_processing:
       chunk_size: 1000
       max_concurrent_tasks: 5
   ```

## Next Steps

After successful installation:

1. **Configure Business Rules**: Edit `config/settings.yaml` for your specific use case
2. **Customize Models**: Train AI models with your historical data
3. **Set Up Integrations**: Connect to your CRM, email systems, and data sources
4. **User Training**: Train your team on the new AI automation features
5. **Monitoring**: Set up logging and monitoring for production use

For detailed configuration options, see [Configuration Guide](configuration.md).

For best practices and optimization tips, see [Best Practices](best_practices.md).