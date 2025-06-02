# README.md

# Tax Scenario Data Extraction Pipeline

Comprehensive data extraction pipeline for generating realistic tax preparation scenarios.

## Features

- **Census Demographics**: State-level race, family structures, employment, education
- **OEWS Employment Data**: Occupation codes, employment counts, wage distributions
- **Education Mappings**: Education requirements by occupation, probability matrices
- **Production Ready**: Error handling, validation, logging, GitHub Actions

## Quick Start

1. **Setup Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Extraction**
   ```bash
   # Full extraction (recommended)
   python comprehensive_data_extractor.py
   
   # Census data only
   python comprehensive_data_extractor.py --census-only
   
   # OEWS data only  
   python comprehensive_data_extractor.py --oews-only
   
   # Validate existing data
   python comprehensive_data_extractor.py --validate
   ```

## Database Schema

The script creates a comprehensive schema across multiple schemas:

- `census.*` - Demographics, family structures, employment
- `oews.*` - Occupation details, employment, wages  
- `education.*` - Education levels, requirements, probabilities
- `tax_scenarios.*` - Generation metadata

## GitHub Actions

The included workflow runs monthly to keep data current:

- Scheduled: 1st of each month at 2 AM UTC
- Manual trigger: Via GitHub Actions UI
- Artifact upload: Logs preserved for 30 days
- Validation: Automatic data quality checks

## Configuration

Key environment variables:

- `NEON_CONNECTION_STRING` (required) - Database connection
- `CENSUS_API_KEY` (optional) - Increases API rate limits
- `EXTRACTION_LOG_LEVEL` (optional) - Logging verbosity

## Data Sources

- **US Census Bureau**: American Community Survey (ACS) 2022
- **Bureau of Labor Statistics**: Occupational Employment and Wage Statistics (OEWS) 2023
- **Education Mappings**: SOC occupation education requirements

## Validation

Built-in validation checks:

- ✅ State coverage completeness
- ✅ Data consistency across sources  
- ✅ Reasonable value ranges
- ✅ Education-occupation mapping completeness
- ✅ Overall data quality scoring

## Output

After successful extraction, your database will contain:

- **51 states** with demographic profiles
- **800+ occupations** with wage data
- **Education probabilities** for realistic job assignment
- **Validation metadata** for quality assurance