#!/usr/bin/env python3
"""
Comprehensive Tax Scenario Data Extractor

A unified data extraction pipeline that gathers all necessary data for realistic
tax preparation scenario generation:

1. US Census demographic data (race, family structures, employment)
2. BLS OEWS occupation and wage data by state
3. Educational attainment patterns by demographics
4. Education-to-occupation probability mappings
5. Industry employment patterns

Designed for production use with GitHub Actions and cloud databases.

Requirements:
- psycopg2-binary, pandas, requests, python-dotenv, openpyxl
- NEON_CONNECTION_STRING environment variable
- CENSUS_API_KEY environment variable (optional but recommended)

Usage:
    python comprehensive_data_extractor.py                    # Full extraction
    python comprehensive_data_extractor.py --census-only      # Census data only
    python comprehensive_data_extractor.py --oews-only        # OEWS data only
    python comprehensive_data_extractor.py --validate         # Validate existing data
"""

import os
import sys
import json
import logging
import requests
import pandas as pd
import psycopg2
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from psycopg2.extras import execute_values
import time
import random

# Configuration
CONFIG = {
    'CENSUS_YEAR': '2022',
    'OEWS_YEARS': [2023],  # Can be expanded: [2022, 2023]
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 5,  # seconds
    'BATCH_SIZE': 1000,
    'REQUEST_TIMEOUT': 60,
    'RATE_LIMIT_DELAY': 1,  # seconds between API calls
}

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup logging
def setup_logging():
    """Configure logging for production use"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f'logs/data_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from requests library
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

class DatabaseManager:
    """Manages database connections and schema creation"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn = None
        self.cursor = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            self.conn.autocommit = False
            self.cursor = self.conn.cursor()
            self.logger.info("Database connection established")
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise
    
    def create_comprehensive_schema(self):
        """Create complete database schema for all tax scenario data"""
        self.logger.info("Creating comprehensive database schema...")
        
        try:
            # Create schemas
            schemas = ['oews', 'census', 'education', 'tax_scenarios']
            for schema in schemas:
                self.cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            
            # Core lookup tables
            self._create_lookup_tables()
            
            # Census demographic tables
            self._create_census_tables()
            
            # OEWS employment and wage tables
            self._create_oews_tables()
            
            # Education and occupation mapping tables
            self._create_education_tables()
            
            # Tax scenario generation tables
            self._create_tax_scenario_tables()
            
            # Create indexes for performance
            self._create_indexes()
            
            self.conn.commit()
            self.logger.info("✓ Database schema created successfully")
            
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Schema creation failed: {e}")
            raise
    
    def _create_lookup_tables(self):
        """Create fundamental lookup tables"""
        
        # US Regions
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS regions (
                id SERIAL PRIMARY KEY,
                region_name VARCHAR(50) UNIQUE NOT NULL,
                region_code VARCHAR(10),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert US regions
        regions = [
            ('Northeast', 'NE'), ('Midwest', 'MW'), 
            ('South', 'SO'), ('West', 'WE')
        ]
        for region_name, region_code in regions:
            self.cursor.execute("""
                INSERT INTO regions (region_name, region_code) 
                VALUES (%s, %s) ON CONFLICT (region_name) DO NOTHING
            """, (region_name, region_code))
        
        # Race/Ethnicity categories
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS race_ethnicity (
                id SERIAL PRIMARY KEY,
                race_key VARCHAR(50) UNIQUE NOT NULL,
                race_name VARCHAR(100) NOT NULL,
                census_category VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert race categories
        race_categories = [
            ('WHITE_NON_HISPANIC', 'White Non-Hispanic', 'White alone, not Hispanic or Latino'),
            ('BLACK', 'Black or African American', 'Black or African American alone'),
            ('HISPANIC', 'Hispanic or Latino', 'Hispanic or Latino (of any race)'),
            ('ASIAN', 'Asian', 'Asian alone'),
            ('NATIVE', 'American Indian and Alaska Native', 'American Indian and Alaska Native alone'),
            ('PACIFIC_ISLANDER', 'Native Hawaiian and Other Pacific Islander', 'Native Hawaiian and Other Pacific Islander alone'),
            ('TWO_OR_MORE', 'Two or More Races', 'Two or more races'),
            ('OTHER', 'Some Other Race', 'Some other race alone')
        ]
        
        for race_key, race_name, census_category in race_categories:
            self.cursor.execute("""
                INSERT INTO race_ethnicity (race_key, race_name, census_category)
                VALUES (%s, %s, %s) ON CONFLICT (race_key) DO NOTHING
            """, (race_key, race_name, census_category))
        
        # Education levels
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS education_levels (
                id SERIAL PRIMARY KEY,
                level_key VARCHAR(50) UNIQUE NOT NULL,
                level_name VARCHAR(100) NOT NULL,
                sort_order INTEGER,
                typical_years INTEGER,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        education_levels = [
            ('less_than_hs', 'Less than High School', 1, 11, 'No high school diploma'),
            ('high_school', 'High School Graduate', 2, 12, 'High school diploma or equivalent'),
            ('some_college', 'Some College or Associate Degree', 3, 14, 'Some college, no degree or associate degree'),
            ('bachelors', 'Bachelor\'s Degree', 4, 16, 'Bachelor\'s degree'),
            ('graduate', 'Graduate or Professional Degree', 5, 18, 'Master\'s, professional, or doctoral degree')
        ]
        
        for level_key, level_name, sort_order, typical_years, description in education_levels:
            self.cursor.execute("""
                INSERT INTO education_levels (level_key, level_name, sort_order, typical_years, description)
                VALUES (%s, %s, %s, %s, %s) ON CONFLICT (level_key) DO NOTHING
            """, (level_key, level_name, sort_order, typical_years, description))
    
    def _create_census_tables(self):
        """Create Census demographic data tables"""
        
        # State demographics (primary state data)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS census.state_demographics (
                id SERIAL PRIMARY KEY,
                state_code VARCHAR(2) UNIQUE NOT NULL,
                state_name VARCHAR(100) NOT NULL,
                region_id INTEGER REFERENCES regions(id),
                total_population BIGINT,
                total_households BIGINT,
                population_weight DECIMAL(10,6),
                data_year INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Race/ethnicity distribution by state
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS census.state_race_ethnicity (
                id SERIAL PRIMARY KEY,
                state_id INTEGER REFERENCES census.state_demographics(id) ON DELETE CASCADE,
                race_id INTEGER REFERENCES race_ethnicity(id),
                population_count BIGINT,
                population_percent DECIMAL(8,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(state_id, race_id)
            )
        """)
        
        # Family structure distribution by state
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS census.family_structures (
                id SERIAL PRIMARY KEY,
                structure_key VARCHAR(50) UNIQUE NOT NULL,
                structure_name VARCHAR(100) NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert family structure types
        family_structures = [
            ('SINGLE_PERSON', 'Single Person Household', 'One person living alone'),
            ('MARRIED_COUPLE', 'Married Couple Family', 'Married couple with or without children'),
            ('MARRIED_COUPLE_CHILDREN', 'Married Couple with Children', 'Married couple with children under 18'),
            ('SINGLE_PARENT_MALE', 'Single Father Household', 'Male householder, no spouse, with children'),
            ('SINGLE_PARENT_FEMALE', 'Single Mother Household', 'Female householder, no spouse, with children'),
            ('MULTIGENERATIONAL', 'Multigenerational Household', 'Three or more generations'),
            ('OTHER_FAMILY', 'Other Family Household', 'Other family arrangements')
        ]
        
        for structure_key, structure_name, description in family_structures:
            self.cursor.execute("""
                INSERT INTO census.family_structures (structure_key, structure_name, description)
                VALUES (%s, %s, %s) ON CONFLICT (structure_key) DO NOTHING
            """, (structure_key, structure_name, description))
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS census.state_family_structures (
                id SERIAL PRIMARY KEY,
                state_id INTEGER REFERENCES census.state_demographics(id) ON DELETE CASCADE,
                structure_id INTEGER REFERENCES census.family_structures(id),
                household_count BIGINT,
                probability_percent DECIMAL(8,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(state_id, structure_id)
            )
        """)
        
        # Employment statistics by state
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS census.state_employment_stats (
                id SERIAL PRIMARY KEY,
                state_id INTEGER REFERENCES census.state_demographics(id) ON DELETE CASCADE,
                total_civilian_labor_force BIGINT,
                employed BIGINT,
                unemployed BIGINT,
                not_in_labor_force BIGINT,
                employment_rate DECIMAL(8,4),
                unemployment_rate DECIMAL(8,4),
                labor_force_participation_rate DECIMAL(8,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(state_id)
            )
        """)
    
    def _create_oews_tables(self):
        """Create OEWS employment and wage tables"""
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS oews.employment_wages (
                id SERIAL PRIMARY KEY,
                year INTEGER NOT NULL,
                area_title VARCHAR(255),
                area_type VARCHAR(50),
                state_code VARCHAR(2),
                occ_code VARCHAR(20),
                occ_title VARCHAR(500),
                tot_emp BIGINT,
                emp_prse DECIMAL(8,4),
                h_mean DECIMAL(12,2),
                a_mean DECIMAL(18,2),
                h_median DECIMAL(12,2),
                a_median DECIMAL(18,2),
                h_pct10 DECIMAL(12,2),
                h_pct25 DECIMAL(12,2),
                h_pct75 DECIMAL(12,2),
                h_pct90 DECIMAL(12,2),
                a_pct10 DECIMAL(18,2),
                a_pct25 DECIMAL(18,2),
                a_pct75 DECIMAL(18,2),
                a_pct90 DECIMAL(18,2),
                source_file VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Occupation categories and details
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS oews.occupation_details (
                id SERIAL PRIMARY KEY,
                occ_code VARCHAR(20) UNIQUE NOT NULL,
                occ_title VARCHAR(500),
                major_group VARCHAR(100),
                minor_group VARCHAR(100),
                broad_occupation VARCHAR(200),
                detailed_occupation VARCHAR(500),
                occ_group VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_education_tables(self):
        """Create education and occupation mapping tables"""
        
        # Educational attainment by demographics
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS education.attainment_demographics (
                id SERIAL PRIMARY KEY,
                state_code VARCHAR(2) NOT NULL,
                race_key VARCHAR(50),
                age_group VARCHAR(20),
                gender VARCHAR(10),
                education_level_id INTEGER REFERENCES education_levels(id),
                population_count INTEGER,
                percentage DECIMAL(8,4),
                margin_of_error DECIMAL(8,4),
                data_year INTEGER DEFAULT 2022,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(state_code, race_key, age_group, gender, education_level_id)
            )
        """)
        
        # Occupation education requirements
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS education.occupation_requirements (
                id SERIAL PRIMARY KEY,
                occ_code VARCHAR(20) UNIQUE NOT NULL,
                occ_title VARCHAR(500),
                typical_education_level_id INTEGER REFERENCES education_levels(id),
                work_experience VARCHAR(100),
                on_job_training VARCHAR(100),
                additional_requirements TEXT,
                source VARCHAR(50),
                confidence_score DECIMAL(4,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Education to occupation probability matrix
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS education.occupation_probabilities (
                id SERIAL PRIMARY KEY,
                education_level_id INTEGER REFERENCES education_levels(id),
                occ_code VARCHAR(20) NOT NULL,
                state_code VARCHAR(2),
                probability DECIMAL(8,6),
                employment_share DECIMAL(8,4),
                median_age INTEGER,
                gender_distribution JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(education_level_id, occ_code, state_code)
            )
        """)
    
    def _create_tax_scenario_tables(self):
        """Create tables for tax scenario generation metadata"""
        
        # Generated family metadata
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tax_scenarios.family_generation_log (
                id SERIAL PRIMARY KEY,
                generation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_families_generated INTEGER,
                data_version VARCHAR(50),
                generation_parameters JSONB,
                validation_results JSONB
            )
        """)
        
        # Income bracket distributions for targeting
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tax_scenarios.income_brackets (
                id SERIAL PRIMARY KEY,
                bracket_name VARCHAR(100),
                min_income INTEGER,
                max_income INTEGER,
                filing_status VARCHAR(50),
                typical_deductions JSONB,
                tax_characteristics JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_indexes(self):
        """Create database indexes for performance"""
        indexes = [
            # Census indexes
            "CREATE INDEX IF NOT EXISTS idx_state_demographics_state_code ON census.state_demographics(state_code)",
            "CREATE INDEX IF NOT EXISTS idx_state_race_state_id ON census.state_race_ethnicity(state_id)",
            "CREATE INDEX IF NOT EXISTS idx_state_race_race_id ON census.state_race_ethnicity(race_id)",
            "CREATE INDEX IF NOT EXISTS idx_state_family_state_id ON census.state_family_structures(state_id)",
            
            # OEWS indexes
            "CREATE INDEX IF NOT EXISTS idx_oews_year_state ON oews.employment_wages(year, state_code)",
            "CREATE INDEX IF NOT EXISTS idx_oews_occupation ON oews.employment_wages(occ_code)",
            "CREATE INDEX IF NOT EXISTS idx_oews_state_occ ON oews.employment_wages(state_code, occ_code)",
            "CREATE INDEX IF NOT EXISTS idx_oews_emp_size ON oews.employment_wages(tot_emp) WHERE tot_emp IS NOT NULL",
            
            # Education indexes
            "CREATE INDEX IF NOT EXISTS idx_education_attainment_state ON education.attainment_demographics(state_code)",
            "CREATE INDEX IF NOT EXISTS idx_education_attainment_level ON education.attainment_demographics(education_level_id)",
            "CREATE INDEX IF NOT EXISTS idx_occupation_requirements_occ ON education.occupation_requirements(occ_code)",
            "CREATE INDEX IF NOT EXISTS idx_occupation_requirements_level ON education.occupation_requirements(typical_education_level_id)",
            "CREATE INDEX IF NOT EXISTS idx_occupation_probabilities_edu ON education.occupation_probabilities(education_level_id)",
            "CREATE INDEX IF NOT EXISTS idx_occupation_probabilities_state ON education.occupation_probabilities(state_code)"
        ]
        
        for index_sql in indexes:
            try:
                self.cursor.execute(index_sql)
            except Exception as e:
                self.logger.warning(f"Index creation warning: {e}")
    
    def close(self):
        """Close database connections"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

class CensusDataExtractor:
    """Extract demographic data from US Census Bureau APIs"""
    
    def __init__(self, db_manager: DatabaseManager, api_key: str = None):
        self.db = db_manager
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Tax-Scenario-Generator/1.0'
        })
        
        # State mapping
        self.state_info = {
            '01': {'name': 'Alabama', 'region': 'South'},
            '02': {'name': 'Alaska', 'region': 'West'},
            '04': {'name': 'Arizona', 'region': 'West'},
            '05': {'name': 'Arkansas', 'region': 'South'},
            '06': {'name': 'California', 'region': 'West'},
            '08': {'name': 'Colorado', 'region': 'West'},
            '09': {'name': 'Connecticut', 'region': 'Northeast'},
            '10': {'name': 'Delaware', 'region': 'South'},
            '11': {'name': 'District of Columbia', 'region': 'South'},
            '12': {'name': 'Florida', 'region': 'South'},
            '13': {'name': 'Georgia', 'region': 'South'},
            '15': {'name': 'Hawaii', 'region': 'West'},
            '16': {'name': 'Idaho', 'region': 'West'},
            '17': {'name': 'Illinois', 'region': 'Midwest'},
            '18': {'name': 'Indiana', 'region': 'Midwest'},
            '19': {'name': 'Iowa', 'region': 'Midwest'},
            '20': {'name': 'Kansas', 'region': 'Midwest'},
            '21': {'name': 'Kentucky', 'region': 'South'},
            '22': {'name': 'Louisiana', 'region': 'South'},
            '23': {'name': 'Maine', 'region': 'Northeast'},
            '24': {'name': 'Maryland', 'region': 'South'},
            '25': {'name': 'Massachusetts', 'region': 'Northeast'},
            '26': {'name': 'Michigan', 'region': 'Midwest'},
            '27': {'name': 'Minnesota', 'region': 'Midwest'},
            '28': {'name': 'Mississippi', 'region': 'South'},
            '29': {'name': 'Missouri', 'region': 'Midwest'},
            '30': {'name': 'Montana', 'region': 'West'},
            '31': {'name': 'Nebraska', 'region': 'Midwest'},
            '32': {'name': 'Nevada', 'region': 'West'},
            '33': {'name': 'New Hampshire', 'region': 'Northeast'},
            '34': {'name': 'New Jersey', 'region': 'Northeast'},
            '35': {'name': 'New Mexico', 'region': 'West'},
            '36': {'name': 'New York', 'region': 'Northeast'},
            '37': {'name': 'North Carolina', 'region': 'South'},
            '38': {'name': 'North Dakota', 'region': 'Midwest'},
            '39': {'name': 'Ohio', 'region': 'Midwest'},
            '40': {'name': 'Oklahoma', 'region': 'South'},
            '41': {'name': 'Oregon', 'region': 'West'},
            '42': {'name': 'Pennsylvania', 'region': 'Northeast'},
            '44': {'name': 'Rhode Island', 'region': 'Northeast'},
            '45': {'name': 'South Carolina', 'region': 'South'},
            '46': {'name': 'South Dakota', 'region': 'Midwest'},
            '47': {'name': 'Tennessee', 'region': 'South'},
            '48': {'name': 'Texas', 'region': 'South'},
            '49': {'name': 'Utah', 'region': 'West'},
            '50': {'name': 'Vermont', 'region': 'Northeast'},
            '51': {'name': 'Virginia', 'region': 'South'},
            '53': {'name': 'Washington', 'region': 'West'},
            '54': {'name': 'West Virginia', 'region': 'South'},
            '55': {'name': 'Wisconsin', 'region': 'Midwest'},
            '56': {'name': 'Wyoming', 'region': 'West'},
        }
    
    def extract_all_census_data(self):
        """Extract all Census data needed for family generation"""
        self.logger.info("Starting Census data extraction...")
        
        try:
            # Extract state-level demographic data
            self._extract_state_demographics()
            
            # Extract race/ethnicity distributions
            self._extract_race_ethnicity_data()
            
            # Extract family structure data
            self._extract_family_structure_data()
            
            # Extract employment data
            self._extract_employment_data()
            
            # Extract detailed educational attainment
            self._extract_education_attainment()
            
            self.logger.info("✓ Census data extraction completed")
            
        except Exception as e:
            self.logger.error(f"Census data extraction failed: {e}")
            raise
    
    def _make_census_request(self, url: str, params: Dict) -> List:
        """Make Census API request with retry logic"""
        if self.api_key:
            params['key'] = self.api_key
        
        for attempt in range(CONFIG['MAX_RETRIES']):
            try:
                time.sleep(CONFIG['RATE_LIMIT_DELAY'])
                response = self.session.get(url, params=params, timeout=CONFIG['REQUEST_TIMEOUT'])
                response.raise_for_status()
                return response.json()
            except Exception as e:
                self.logger.warning(f"Census API attempt {attempt + 1} failed: {e}")
                if attempt < CONFIG['MAX_RETRIES'] - 1:
                    time.sleep(CONFIG['RETRY_DELAY'] * (attempt + 1))
                else:
                    raise
    
    def _extract_state_demographics(self):
        """Extract basic state demographic data"""
        self.logger.info("Extracting state demographics...")
        
        url = f"https://api.census.gov/data/{CONFIG['CENSUS_YEAR']}/acs/acs1"
        variables = {
            'B01003_001E': 'total_population',
            'B25001_001E': 'total_housing_units',
            'B11001_001E': 'total_households'
        }
        
        params = {
            'get': ','.join(variables.keys()),
            'for': 'state:*'
        }
        
        data = self._make_census_request(url, params)
        
        for row in data[1:]:  # Skip header
            values = row[:-1]
            state_code = row[-1]
            
            if state_code not in self.state_info:
                continue
            
            # Parse values
            demo_data = {}
            for i, var_code in enumerate(variables.keys()):
                try:
                    demo_data[variables[var_code]] = int(values[i]) if values[i] else 0
                except (ValueError, TypeError):
                    demo_data[variables[var_code]] = 0
            
            # Get region ID
            region_name = self.state_info[state_code]['region']
            self.db.cursor.execute("SELECT id FROM regions WHERE region_name = %s", (region_name,))
            region_id = self.db.cursor.fetchone()[0]
            
            # Calculate population weight (for realistic family generation)
            total_us_pop = sum(int(row[0]) if row[0] else 0 for row in data[1:] if row[0])
            pop_weight = (demo_data['total_population'] / total_us_pop) * 100 if total_us_pop > 0 else 0
            
            # Store state demographics
            self.db.cursor.execute("""
                INSERT INTO census.state_demographics 
                (state_code, state_name, region_id, total_population, total_households, population_weight, data_year)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (state_code) DO UPDATE SET
                    state_name = EXCLUDED.state_name,
                    region_id = EXCLUDED.region_id,
                    total_population = EXCLUDED.total_population,
                    total_households = EXCLUDED.total_households,
                    population_weight = EXCLUDED.population_weight,
                    data_year = EXCLUDED.data_year,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                state_code, self.state_info[state_code]['name'], region_id,
                demo_data['total_population'], demo_data['total_households'],
                pop_weight, int(CONFIG['CENSUS_YEAR'])
            ))
        
        self.db.conn.commit()
        self.logger.info(f"✓ Stored demographics for {len(data) - 1} states")
    
    def _extract_race_ethnicity_data(self):
        """Extract race/ethnicity distribution by state"""
        self.logger.info("Extracting race/ethnicity data...")
        
        url = f"https://api.census.gov/data/{CONFIG['CENSUS_YEAR']}/acs/acs1"
        variables = {
            'B03002_001E': 'total_population',
            'B03002_003E': 'white_non_hispanic',
            'B03002_004E': 'black',
            'B03002_006E': 'asian',
            'B03002_005E': 'native_american',
            'B03002_007E': 'pacific_islander',
            'B03002_009E': 'two_or_more_races',
            'B03002_012E': 'hispanic_total'
        }
        
        params = {
            'get': ','.join(variables.keys()),
            'for': 'state:*'
        }
        
        data = self._make_census_request(url, params)
        
        for row in data[1:]:
            values = row[:-1]
            state_code = row[-1]
            
            if state_code not in self.state_info:
                continue
            
            # Get state ID
            self.db.cursor.execute("SELECT id FROM census.state_demographics WHERE state_code = %s", (state_code,))
            result = self.db.cursor.fetchone()
            if not result:
                continue
            state_id = result[0]
            
            # Parse race data
            race_data = {}
            for i, var_code in enumerate(variables.keys()):
                try:
                    race_data[variables[var_code]] = int(values[i]) if values[i] else 0
                except (ValueError, TypeError):
                    race_data[variables[var_code]] = 0
            
            total_pop = race_data['total_population']
            if total_pop == 0:
                continue
            
            # Map to our race categories
            race_mapping = {
                'WHITE_NON_HISPANIC': race_data['white_non_hispanic'],
                'BLACK': race_data['black'],
                'HISPANIC': race_data['hispanic_total'],
                'ASIAN': race_data['asian'],
                'NATIVE': race_data['native_american'],
                'PACIFIC_ISLANDER': race_data['pacific_islander'],
                'TWO_OR_MORE': race_data['two_or_more_races']
            }
            
            # Store race/ethnicity data
            for race_key, population_count in race_mapping.items():
                if population_count == 0:
                    continue
                
                # Get race ID
                self.db.cursor.execute("SELECT id FROM race_ethnicity WHERE race_key = %s", (race_key,))
                race_result = self.db.cursor.fetchone()
                if not race_result:
                    continue
                race_id = race_result[0]
                
                percentage = (population_count / total_pop) * 100
                
                self.db.cursor.execute("""
                    INSERT INTO census.state_race_ethnicity (state_id, race_id, population_count, population_percent)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (state_id, race_id) DO UPDATE SET
                        population_count = EXCLUDED.population_count,
                        population_percent = EXCLUDED.population_percent
                """, (state_id, race_id, population_count, percentage))
        
        self.db.conn.commit()
        self.logger.info("✓ Race/ethnicity data stored")
    
    def _extract_family_structure_data(self):
        """Extract family structure distribution by state"""
        self.logger.info("Extracting family structure data...")
        
        url = f"https://api.census.gov/data/{CONFIG['CENSUS_YEAR']}/acs/acs1"
        variables = {
            'B11001_001E': 'total_households',
            'B11001_002E': 'family_households',
            'B11001_003E': 'married_couple_families',
            'B11001_005E': 'male_householder_no_wife',
            'B11001_006E': 'female_householder_no_husband',
            'B11001_009E': 'living_alone'
        }
        
        params = {
            'get': ','.join(variables.keys()),
            'for': 'state:*'
        }
        
        data = self._make_census_request(url, params)
        
        for row in data[1:]:
            values = row[:-1]
            state_code = row[-1]
            
            if state_code not in self.state_info:
                continue
            
            # Get state ID
            self.db.cursor.execute("SELECT id FROM census.state_demographics WHERE state_code = %s", (state_code,))
            result = self.db.cursor.fetchone()
            if not result:
                continue
            state_id = result[0]
            
            # Parse family data
            family_data = {}
            for i, var_code in enumerate(variables.keys()):
                try:
                    family_data[variables[var_code]] = int(values[i]) if values[i] else 0
                except (ValueError, TypeError):
                    family_data[variables[var_code]] = 0
            
            total_households = family_data['total_households']
            if total_households == 0:
                continue
            
            # Map to our family structure categories
            structure_mapping = {
                'SINGLE_PERSON': family_data['living_alone'],
                'MARRIED_COUPLE': family_data['married_couple_families'],
                'SINGLE_PARENT_MALE': family_data['male_householder_no_wife'],
                'SINGLE_PARENT_FEMALE': family_data['female_householder_no_husband']
            }
            
            # Store family structure data
            for structure_key, household_count in structure_mapping.items():
                if household_count == 0:
                    continue
                
                # Get structure ID
                self.db.cursor.execute("SELECT id FROM census.family_structures WHERE structure_key = %s", (structure_key,))
                structure_result = self.db.cursor.fetchone()
                if not structure_result:
                    continue
                structure_id = structure_result[0]
                
                percentage = (household_count / total_households) * 100
                
                self.db.cursor.execute("""
                    INSERT INTO census.state_family_structures (state_id, structure_id, household_count, probability_percent)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (state_id, structure_id) DO UPDATE SET
                        household_count = EXCLUDED.household_count,
                        probability_percent = EXCLUDED.probability_percent
                """, (state_id, structure_id, household_count, percentage))
        
        self.db.conn.commit()
        self.logger.info("✓ Family structure data stored")
    
    def _extract_employment_data(self):
        """Extract employment statistics by state"""
        self.logger.info("Extracting employment data...")
        
        url = f"https://api.census.gov/data/{CONFIG['CENSUS_YEAR']}/acs/acs1"
        variables = {
            'B23025_001E': 'total_pop_16_over',
            'B23025_002E': 'in_labor_force',
            'B23025_004E': 'employed_civilian',
            'B23025_005E': 'unemployed_civilian',
            'B23025_007E': 'not_in_labor_force'
        }
        
        params = {
            'get': ','.join(variables.keys()),
            'for': 'state:*'
        }
        
        data = self._make_census_request(url, params)
        
        for row in data[1:]:
            values = row[:-1]
            state_code = row[-1]
            
            if state_code not in self.state_info:
                continue
            
            # Get state ID
            self.db.cursor.execute("SELECT id FROM census.state_demographics WHERE state_code = %s", (state_code,))
            result = self.db.cursor.fetchone()
            if not result:
                continue
            state_id = result[0]
            
            # Parse employment data
            emp_data = {}
            for i, var_code in enumerate(variables.keys()):
                try:
                    emp_data[variables[var_code]] = int(values[i]) if values[i] else 0
                except (ValueError, TypeError):
                    emp_data[variables[var_code]] = 0
            
            total_pop = emp_data['total_pop_16_over']
            labor_force = emp_data['in_labor_force']
            employed = emp_data['employed_civilian']
            unemployed = emp_data['unemployed_civilian']
            not_in_lf = emp_data['not_in_labor_force']
            
            if total_pop == 0 or labor_force == 0:
                continue
            
            # Calculate rates
            employment_rate = (employed / labor_force) * 100
            unemployment_rate = (unemployed / labor_force) * 100
            labor_force_participation_rate = (labor_force / total_pop) * 100
            
            # Store employment data
            self.db.cursor.execute("""
                INSERT INTO census.state_employment_stats 
                (state_id, total_civilian_labor_force, employed, unemployed, not_in_labor_force,
                 employment_rate, unemployment_rate, labor_force_participation_rate)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (state_id) DO UPDATE SET
                    total_civilian_labor_force = EXCLUDED.total_civilian_labor_force,
                    employed = EXCLUDED.employed,
                    unemployed = EXCLUDED.unemployed,
                    not_in_labor_force = EXCLUDED.not_in_labor_force,
                    employment_rate = EXCLUDED.employment_rate,
                    unemployment_rate = EXCLUDED.unemployment_rate,
                    labor_force_participation_rate = EXCLUDED.labor_force_participation_rate
            """, (state_id, labor_force, employed, unemployed, not_in_lf,
                  employment_rate, unemployment_rate, labor_force_participation_rate))
        
        self.db.conn.commit()
        self.logger.info("✓ Employment data stored")
    
    def _extract_education_attainment(self):
        """Extract detailed educational attainment by state"""
        self.logger.info("Extracting educational attainment data...")
        
        url = f"https://api.census.gov/data/{CONFIG['CENSUS_YEAR']}/acs/acs1"
        variables = {
            'B15003_001E': 'total_pop_25_over',
            'B15003_017E': 'high_school_graduate',
            'B15003_018E': 'ged_alternative_credential',
            'B15003_019E': 'some_college_less_1_year',
            'B15003_020E': 'some_college_1_or_more_years',
            'B15003_021E': 'associates_degree',
            'B15003_022E': 'bachelors_degree',
            'B15003_023E': 'masters_degree',
            'B15003_024E': 'professional_degree',
            'B15003_025E': 'doctorate_degree'
        }
        
        params = {
            'get': ','.join(variables.keys()),
            'for': 'state:*'
        }
        
        data = self._make_census_request(url, params)
        
        for row in data[1:]:
            values = row[:-1]
            state_code = row[-1]
            
            if state_code not in self.state_info:
                continue
            
            # Parse education data
            education_data = {}
            for i, var_code in enumerate(variables.keys()):
                try:
                    education_data[variables[var_code]] = int(values[i]) if values[i] else 0
                except (ValueError, TypeError):
                    education_data[variables[var_code]] = 0
            
            total_pop = education_data['total_pop_25_over']
            if total_pop == 0:
                continue
            
            # Aggregate into our education levels
            less_than_hs = total_pop - sum([
                education_data['high_school_graduate'],
                education_data['ged_alternative_credential'],
                education_data['some_college_less_1_year'],
                education_data['some_college_1_or_more_years'],
                education_data['associates_degree'],
                education_data['bachelors_degree'],
                education_data['masters_degree'],
                education_data['professional_degree'],
                education_data['doctorate_degree']
            ])
            
            education_counts = {
                'less_than_hs': max(0, less_than_hs),
                'high_school': education_data['high_school_graduate'] + education_data['ged_alternative_credential'],
                'some_college': (education_data['some_college_less_1_year'] + 
                               education_data['some_college_1_or_more_years'] + 
                               education_data['associates_degree']),
                'bachelors': education_data['bachelors_degree'],
                'graduate': (education_data['masters_degree'] + 
                           education_data['professional_degree'] + 
                           education_data['doctorate_degree'])
            }
            
            # Store education attainment data
            for education_key, count in education_counts.items():
                if count == 0:
                    continue
                
                # Get education level ID
                self.db.cursor.execute("SELECT id FROM education_levels WHERE level_key = %s", (education_key,))
                result = self.db.cursor.fetchone()
                if not result:
                    continue
                education_level_id = result[0]
                
                percentage = (count / total_pop) * 100
                
                self.db.cursor.execute("""
                    INSERT INTO education.attainment_demographics 
                    (state_code, race_key, age_group, gender, education_level_id, population_count, percentage)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (state_code, race_key, age_group, gender, education_level_id) 
                    DO UPDATE SET
                        population_count = EXCLUDED.population_count,
                        percentage = EXCLUDED.percentage
                """, (state_code, 'All', 'All', 'All', education_level_id, count, percentage))
        
        self.db.conn.commit()
        self.logger.info("✓ Education attainment data stored")

class OEWSDataExtractor:
    """Extract occupation and wage data from BLS OEWS"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Tax-Scenario-Generator/1.0)'
        })
        
        # OEWS URLs by year - state data only for faster processing
        self.urls = {
            2023: "https://www.bls.gov/oes/special-requests/oesm23st.zip",
            2022: "https://www.bls.gov/oes/special-requests/oesm22st.zip"
        }
    
    def extract_all_oews_data(self):
        """Extract all OEWS data for configured years"""
        self.logger.info("Starting OEWS data extraction...")
        
        for year in CONFIG['OEWS_YEARS']:
            if year not in self.urls:
                self.logger.warning(f"No URL configured for OEWS year {year}")
                continue
            
            self.logger.info(f"Processing OEWS {year} data...")
            self._extract_year_data(year)
        
        # Process occupation details and education requirements
        self._process_occupation_details()
        self._assign_education_requirements()
        
        self.logger.info("✓ OEWS data extraction completed")
    
    def _extract_year_data(self, year: int):
        """Extract OEWS data for a specific year"""
        url = self.urls[year]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download ZIP file
            zip_file = Path(temp_dir) / f"oews_{year}.zip"
            self._download_file(url, str(zip_file))
            
            # Extract Excel files
            extract_dir = Path(temp_dir) / "extracted"
            extract_dir.mkdir()
            excel_files = self._extract_zip(str(zip_file), str(extract_dir))
            
            # Process each Excel file
            total_rows = 0
            for excel_file in excel_files:
                df = self._read_excel_file(excel_file, year)
                if not df.empty:
                    self._store_oews_data(df)
                    total_rows += len(df)
            
            self.logger.info(f"✓ OEWS {year}: {len(excel_files)} files, {total_rows:,} rows processed")
    
    def _download_file(self, url: str, filename: str) -> bool:
        """Download file with retry logic"""
        for attempt in range(CONFIG['MAX_RETRIES']):
            try:
                self.logger.info(f"Downloading {url} (attempt {attempt + 1})")
                response = self.session.get(url, stream=True, timeout=CONFIG['REQUEST_TIMEOUT'])
                response.raise_for_status()
                
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                size_mb = os.path.getsize(filename) / (1024 * 1024)
                self.logger.info(f"✓ Downloaded {filename} ({size_mb:.1f} MB)")
                return True
                
            except Exception as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < CONFIG['MAX_RETRIES'] - 1:
                    time.sleep(CONFIG['RETRY_DELAY'] * (attempt + 1))
                else:
                    raise
    
    def _extract_zip(self, zip_path: str, extract_dir: str) -> List[str]:
        """Extract Excel files from ZIP"""
        files = []
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith(('.xlsx', '.xls')):
                        extracted = zip_ref.extract(file_info, extract_dir)
                        files.append(extracted)
                        self.logger.debug(f"Extracted: {file_info.filename}")
        except Exception as e:
            self.logger.error(f"ZIP extraction failed: {e}")
        return files
    
    def _read_excel_file(self, file_path: str, year: int) -> pd.DataFrame:
        """Read and clean Excel file"""
        try:
            # Try different header rows
            df = None
            for header_row in range(5):
                try:
                    df = pd.read_excel(file_path, header=header_row)
                    if len(df.columns) > 5 and len(df) > 10:
                        break
                except:
                    continue
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # Clean column names
            df.columns = df.columns.astype(str).str.strip().str.replace('\n', ' ')
            df = df.dropna(how='all')
            
            # Map columns to standard names
            column_map = self._map_columns(df.columns)
            if len(column_map) < 3:  # Need at least area, occupation, employment
                return pd.DataFrame()
            
            # Create clean DataFrame
            clean_df = pd.DataFrame()
            for target, source in column_map.items():
                if source in df.columns:
                    clean_df[target] = df[source]
            
            # Add metadata
            clean_df['year'] = year
            clean_df['source_file'] = Path(file_path).name
            
            # Clean data types
            self._clean_data_types(clean_df)
            
            return clean_df
            
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return pd.DataFrame()
    
    def _map_columns(self, columns: List[str]) -> Dict[str, str]:
        """Map Excel columns to standard column names"""
        column_map = {}
        
        for col in columns:
            col_lower = col.lower().strip()
            
            if 'area' in col_lower and 'title' in col_lower:
                column_map['area_title'] = col
            elif 'occ_code' in col_lower or col_lower == 'occ code':
                column_map['occ_code'] = col
            elif 'occ_title' in col_lower or 'occupation' in col_lower:
                column_map['occ_title'] = col
            elif col_lower in ['tot_emp', 'employment']:
                column_map['tot_emp'] = col
            elif 'h_mean' in col_lower or 'hourly mean' in col_lower:
                column_map['h_mean'] = col
            elif 'a_mean' in col_lower or 'annual mean' in col_lower:
                column_map['a_mean'] = col
            elif 'h_median' in col_lower or 'hourly median' in col_lower:
                column_map['h_median'] = col
            elif 'a_median' in col_lower or 'annual median' in col_lower:
                column_map['a_median'] = col
            # Add more percentile columns if available
            elif 'h_pct10' in col_lower:
                column_map['h_pct10'] = col
            elif 'h_pct25' in col_lower:
                column_map['h_pct25'] = col
            elif 'h_pct75' in col_lower:
                column_map['h_pct75'] = col
            elif 'h_pct90' in col_lower:
                column_map['h_pct90'] = col
        
        return column_map
    
    def _clean_data_types(self, df: pd.DataFrame):
        """Clean and convert data types"""
        # Numeric columns
        numeric_cols = ['tot_emp', 'h_mean', 'a_mean', 'h_median', 'a_median', 
                       'h_pct10', 'h_pct25', 'h_pct75', 'h_pct90',
                       'a_pct10', 'a_pct25', 'a_pct75', 'a_pct90']
        
        for col in numeric_cols:
            if col in df.columns:
                try:
                    # Convert to string first
                    df[col] = df[col].astype(str)
                    # Remove non-numeric characters
                    df[col] = df[col].str.replace(r'[*#$,()]', '', regex=True)
                    df[col] = df[col].str.strip()
                    # Replace empty strings with None
                    df[col] = df[col].replace(['', 'nan', 'NaN', 'N/A', '-'], None)
                    # Convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Cap extreme values
                    if col == 'tot_emp':
                        df[col] = df[col].where(df[col] <= 50_000_000, None)
                    elif col.startswith('a_'):  # Annual wages
                        df[col] = df[col].where(df[col] <= 5_000_000, None)
                    elif col.startswith('h_'):  # Hourly wages
                        df[col] = df[col].where(df[col] <= 5_000, None)
                        
                except Exception as e:
                    self.logger.warning(f"Error processing column {col}: {e}")
                    df[col] = None
        
        # String columns
        string_cols = ['area_title', 'occ_code', 'occ_title']
        for col in string_cols:
            if col in df.columns:
                try:
                    df[col] = df[col].astype(str)
                    df[col] = df[col].str.strip()
                    df[col] = df[col].replace(['nan', 'None', '<NA>', ''], None)
                except Exception as e:
                    self.logger.warning(f"Error processing string column {col}: {e}")
    
    def _store_oews_data(self, df: pd.DataFrame):
        """Store OEWS data in database"""
        if df.empty:
            return
        
        try:
            # Map state names to state codes
            df['state_code'] = df['area_title'].map(self._get_state_code_mapping())
            
            # Filter out non-state data (national, regional data)
            df = df[df['state_code'].notna()]
            
            if df.empty:
                return
            
            # Convert DataFrame to list of tuples for insertion
            columns = [col for col in df.columns if col != 'id']
            df_clean = df[columns].copy()
            
            # Handle NaN values
            df_clean = df_clean.where(pd.notnull(df_clean), None)
            
            values_list = []
            for _, row in df_clean.iterrows():
                row_values = []
                for val in row:
                    if pd.isna(val) or val is pd.NA:
                        row_values.append(None)
                    else:
                        row_values.append(val)
                values_list.append(tuple(row_values))
            
            # Insert data
            cols_str = ', '.join(columns)
            query = f"""
                INSERT INTO oews.employment_wages ({cols_str}) 
                VALUES %s 
                ON CONFLICT DO NOTHING
            """
            
            execute_values(
                self.db.cursor, query, values_list, 
                page_size=CONFIG['BATCH_SIZE']
            )
            
            self.db.conn.commit()
            self.logger.debug(f"Stored {len(values_list):,} OEWS records")
            
        except Exception as e:
            self.db.conn.rollback()
            self.logger.error(f"Error storing OEWS data: {e}")
            raise
    
    def _get_state_code_mapping(self) -> Dict[str, str]:
        """Create mapping from state names to state codes"""
        return {
            'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05',
            'California': '06', 'Colorado': '08', 'Connecticut': '09', 'Delaware': '10',
            'District of Columbia': '11', 'Florida': '12', 'Georgia': '13', 'Hawaii': '15',
            'Idaho': '16', 'Illinois': '17', 'Indiana': '18', 'Iowa': '19',
            'Kansas': '20', 'Kentucky': '21', 'Louisiana': '22', 'Maine': '23',
            'Maryland': '24', 'Massachusetts': '25', 'Michigan': '26', 'Minnesota': '27',
            'Mississippi': '28', 'Missouri': '29', 'Montana': '30', 'Nebraska': '31',
            'Nevada': '32', 'New Hampshire': '33', 'New Jersey': '34', 'New Mexico': '35',
            'New York': '36', 'North Carolina': '37', 'North Dakota': '38', 'Ohio': '39',
            'Oklahoma': '40', 'Oregon': '41', 'Pennsylvania': '42', 'Rhode Island': '44',
            'South Carolina': '45', 'South Dakota': '46', 'Tennessee': '47', 'Texas': '48',
            'Utah': '49', 'Vermont': '50', 'Virginia': '51', 'Washington': '53',
            'West Virginia': '54', 'Wisconsin': '55', 'Wyoming': '56'
        }
    
    def _process_occupation_details(self):
        """Extract and store occupation details"""
        self.logger.info("Processing occupation details...")
        
        # Get unique occupations from OEWS data
        self.db.cursor.execute("""
            SELECT DISTINCT occ_code, occ_title
            FROM oews.employment_wages 
            WHERE occ_code IS NOT NULL AND occ_code != ''
            ORDER BY occ_code
        """)
        
        occupations = self.db.cursor.fetchall()
        
        for occ_code, occ_title in occupations:
            if not occ_code or len(occ_code) < 2:
                continue
            
            # Parse occupation hierarchy
            major_group, minor_group, broad_occ, detailed_occ = self._parse_soc_hierarchy(occ_code, occ_title)
            occ_group = occ_code[:2]  # First 2 digits
            
            self.db.cursor.execute("""
                INSERT INTO oews.occupation_details 
                (occ_code, occ_title, major_group, minor_group, broad_occupation, detailed_occupation, occ_group)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (occ_code) DO UPDATE SET
                    occ_title = EXCLUDED.occ_title,
                    major_group = EXCLUDED.major_group,
                    minor_group = EXCLUDED.minor_group,
                    broad_occupation = EXCLUDED.broad_occupation,
                    detailed_occupation = EXCLUDED.detailed_occupation,
                    occ_group = EXCLUDED.occ_group
            """, (occ_code, occ_title, major_group, minor_group, broad_occ, detailed_occ, occ_group))
        
        self.db.conn.commit()
        self.logger.info(f"✓ Processed {len(occupations):,} occupation details")
    
    def _parse_soc_hierarchy(self, occ_code: str, occ_title: str) -> Tuple[str, str, str, str]:
        """Parse SOC occupation hierarchy"""
        # SOC major groups mapping
        major_groups = {
            '11': 'Management Occupations',
            '13': 'Business and Financial Operations Occupations',
            '15': 'Computer and Mathematical Occupations',
            '17': 'Architecture and Engineering Occupations',
            '19': 'Life, Physical, and Social Science Occupations',
            '21': 'Community and Social Service Occupations',
            '23': 'Legal Occupations',
            '25': 'Educational Instruction and Library Occupations',
            '27': 'Arts, Design, Entertainment, Sports, and Media Occupations',
            '29': 'Healthcare Practitioners and Technical Occupations',
            '31': 'Healthcare Support Occupations',
            '33': 'Protective Service Occupations',
            '35': 'Food Preparation and Serving Related Occupations',
            '37': 'Building and Grounds Cleaning and Maintenance Occupations',
            '39': 'Personal Care and Service Occupations',
            '41': 'Sales and Related Occupations',
            '43': 'Office and Administrative Support Occupations',
            '45': 'Farming, Fishing, and Forestry Occupations',
            '47': 'Construction and Extraction Occupations',
            '49': 'Installation, Maintenance, and Repair Occupations',
            '51': 'Production Occupations',
            '53': 'Transportation and Material Moving Occupations'
        }
        
        major_group = major_groups.get(occ_code[:2], 'Other Occupations')
        minor_group = f"{occ_code[:3]}-xxxx"  # Minor group pattern
        broad_occ = f"{occ_code[:4]}-x"  # Broad occupation pattern
        detailed_occ = occ_title  # The specific occupation title
        
        return major_group, minor_group, broad_occ, detailed_occ
    
    def _assign_education_requirements(self):
        """Assign education requirements to occupations"""
        self.logger.info("Assigning education requirements to occupations...")
        
        # Education requirement mapping by SOC major group
        education_mapping = {
            '11': 'bachelors',    # Management
            '13': 'bachelors',    # Business and Financial
            '15': 'bachelors',    # Computer and Mathematical
            '17': 'bachelors',    # Architecture and Engineering
            '19': 'bachelors',    # Life, Physical, and Social Science
            '21': 'bachelors',    # Community and Social Service
            '23': 'graduate',     # Legal
            '25': 'bachelors',    # Educational
            '27': 'bachelors',    # Arts, Design, Entertainment
            '29': 'graduate',     # Healthcare Practitioners
            '31': 'some_college', # Healthcare Support
            '33': 'high_school',  # Protective Service
            '35': 'less_than_hs', # Food Preparation
            '37': 'less_than_hs', # Building and Grounds
            '39': 'high_school',  # Personal Care
            '41': 'high_school',  # Sales
            '43': 'high_school',  # Office and Administrative
            '45': 'less_than_hs', # Farming, Fishing, Forestry
            '47': 'high_school',  # Construction and Extraction
            '49': 'some_college', # Installation, Maintenance, Repair
            '51': 'high_school',  # Production
            '53': 'high_school'   # Transportation and Material Moving
        }
        
        # Get all occupations
        self.db.cursor.execute("SELECT occ_code, occ_title FROM oews.occupation_details")
        occupations = self.db.cursor.fetchall()
        
        for occ_code, occ_title in occupations:
            if not occ_code or len(occ_code) < 2:
                continue
            
            # Get education requirement
            major_group = occ_code[:2]
            education_key = education_mapping.get(major_group, 'high_school')
            
            # Get education level ID
            self.db.cursor.execute("SELECT id FROM education_levels WHERE level_key = %s", (education_key,))
            result = self.db.cursor.fetchone()
            if not result:
                continue
            education_level_id = result[0]
            
            # Determine work experience and training requirements
            work_experience, on_job_training = self._get_experience_training_requirements(occ_code, occ_title)
            
            # Store education requirement
            self.db.cursor.execute("""
                INSERT INTO education.occupation_requirements 
                (occ_code, occ_title, typical_education_level_id, work_experience, on_job_training, source, confidence_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (occ_code) DO UPDATE SET
                    occ_title = EXCLUDED.occ_title,
                    typical_education_level_id = EXCLUDED.typical_education_level_id,
                    work_experience = EXCLUDED.work_experience,
                    on_job_training = EXCLUDED.on_job_training,
                    source = EXCLUDED.source,
                    confidence_score = EXCLUDED.confidence_score
            """, (occ_code, occ_title, education_level_id, work_experience, on_job_training, 'SOC_mapping', 0.8))
        
        self.db.conn.commit()
        self.logger.info(f"✓ Assigned education requirements to {len(occupations):,} occupations")
    
    def _get_experience_training_requirements(self, occ_code: str, occ_title: str) -> Tuple[str, str]:
        """Determine work experience and training requirements"""
        title_lower = occ_title.lower() if occ_title else ""
        major_group = occ_code[:2]
        
        # Work experience patterns
        if any(word in title_lower for word in ['manager', 'director', 'supervisor', 'chief', 'senior']):
            work_experience = "5 years or more"
        elif any(word in title_lower for word in ['lead', 'coordinator', 'specialist']):
            work_experience = "Less than 5 years"
        else:
            work_experience = "None"
        
        # Training patterns by occupation group
        training_mapping = {
            '11': 'None',  # Management
            '13': 'None',  # Business
            '15': 'None',  # Computer
            '17': 'None',  # Engineering
            '19': 'None',  # Science
            '21': 'None',  # Community Service
            '23': 'Internship/residency',  # Legal
            '25': 'Internship/residency',  # Education
            '27': 'Long-term on-the-job training',  # Arts
            '29': 'Internship/residency',  # Healthcare Practitioners
            '31': 'Short-term on-the-job training',  # Healthcare Support
            '33': 'Moderate-term on-the-job training',  # Protective Service
            '35': 'Short-term on-the-job training',  # Food
            '37': 'Short-term on-the-job training',  # Building/Grounds
            '39': 'Short-term on-the-job training',  # Personal Care
            '41': 'Short-term on-the-job training',  # Sales
            '43': 'Short-term on-the-job training',  # Office
            '45': 'Short-term on-the-job training',  # Farming
            '47': 'Moderate-term on-the-job training',  # Construction
            '49': 'Long-term on-the-job training',  # Installation/Repair
            '51': 'Moderate-term on-the-job training',  # Production
            '53': 'Short-term on-the-job training'   # Transportation
        }
        
        on_job_training = training_mapping.get(major_group, 'Short-term on-the-job training')
        
        return work_experience, on_job_training

class EducationOccupationProcessor:
    """Process education-occupation relationships and probabilities"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
    
    def calculate_education_occupation_probabilities(self):
        """Calculate probability matrix: P(Occupation | Education Level, State)"""
        self.logger.info("Calculating education-occupation probabilities...")
        
        try:
            # Calculate state-level probabilities
            self._calculate_state_probabilities()
            
            # Calculate national fallback probabilities
            self._calculate_national_probabilities()
            
            self.logger.info("✓ Education-occupation probabilities calculated")
            
        except Exception as e:
            self.logger.error(f"Error calculating probabilities: {e}")
            raise
    
    def _calculate_state_probabilities(self):
        """Calculate education-occupation probabilities by state"""
        
        # Get states with employment data
        self.db.cursor.execute("SELECT DISTINCT state_code FROM oews.employment_wages WHERE state_code IS NOT NULL")
        states = [row[0] for row in self.db.cursor.fetchall()]
        
        for state_code in states:
            self.logger.debug(f"Processing probabilities for state {state_code}")
            
            # Calculate employment by education level for this state
            self.db.cursor.execute("""
                WITH state_employment AS (
                    SELECT 
                        oer.typical_education_level_id,
                        ew.occ_code,
                        SUM(ew.tot_emp) as total_employment,
                        AVG(ew.a_mean) as avg_wage
                    FROM education.occupation_requirements oer
                    JOIN oews.employment_wages ew ON oer.occ_code = ew.occ_code
                    WHERE ew.state_code = %s AND ew.tot_emp IS NOT NULL AND ew.tot_emp > 0
                    GROUP BY oer.typical_education_level_id, ew.occ_code
                ),
                education_totals AS (
                    SELECT 
                        typical_education_level_id,
                        SUM(total_employment) as education_level_employment
                    FROM state_employment
                    GROUP BY typical_education_level_id
                )
                SELECT 
                    se.typical_education_level_id,
                    se.occ_code,
                    se.total_employment,
                    et.education_level_employment,
                    (se.total_employment::decimal / NULLIF(et.education_level_employment, 0)) as probability,
                    (se.total_employment::decimal / (SELECT SUM(total_employment) FROM state_employment)) * 100 as employment_share
                FROM state_employment se
                JOIN education_totals et ON se.typical_education_level_id = et.typical_education_level_id
                WHERE et.education_level_employment > 0
                ORDER BY se.typical_education_level_id, probability DESC
            """, (state_code,))
            
            probabilities = self.db.cursor.fetchall()
            
            # Store state-specific probabilities
            for education_level_id, occ_code, emp_count, total_emp, probability, emp_share in probabilities:
                if probability is None:
                    continue
                    
                self.db.cursor.execute("""
                    INSERT INTO education.occupation_probabilities 
                    (education_level_id, occ_code, state_code, probability, employment_share)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (education_level_id, occ_code, state_code) DO UPDATE SET
                        probability = EXCLUDED.probability,
                        employment_share = EXCLUDED.employment_share
                """, (education_level_id, occ_code, state_code, probability, emp_share))
            
            if len(probabilities) > 0:
                self.db.conn.commit()
    
    def _calculate_national_probabilities(self):
        """Calculate national education-occupation probabilities as fallback"""
        
        self.db.cursor.execute("""
            WITH national_employment AS (
                SELECT 
                    oer.typical_education_level_id,
                    ew.occ_code,
                    SUM(ew.tot_emp) as total_employment,
                    AVG(ew.a_mean) as avg_wage
                FROM education.occupation_requirements oer
                JOIN oews.employment_wages ew ON oer.occ_code = ew.occ_code
                WHERE ew.tot_emp IS NOT NULL AND ew.tot_emp > 0
                GROUP BY oer.typical_education_level_id, ew.occ_code
            ),
            education_totals AS (
                SELECT 
                    typical_education_level_id,
                    SUM(total_employment) as education_level_employment
                FROM national_employment
                GROUP BY typical_education_level_id
            )
            SELECT 
                ne.typical_education_level_id,
                ne.occ_code,
                ne.total_employment,
                et.education_level_employment,
                (ne.total_employment::decimal / NULLIF(et.education_level_employment, 0)) as probability,
                (ne.total_employment::decimal / (SELECT SUM(total_employment) FROM national_employment)) * 100 as employment_share
            FROM national_employment ne
            JOIN education_totals et ON ne.typical_education_level_id = et.typical_education_level_id
            WHERE et.education_level_employment > 0
            ORDER BY ne.typical_education_level_id, probability DESC
        """)
        
        probabilities = self.db.cursor.fetchall()
        
        # Store national probabilities (state_code = NULL indicates national)
        for education_level_id, occ_code, emp_count, total_emp, probability, emp_share in probabilities:
            if probability is None:
                continue
                
            self.db.cursor.execute("""
                INSERT INTO education.occupation_probabilities 
                (education_level_id, occ_code, state_code, probability, employment_share)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (education_level_id, occ_code, state_code) DO UPDATE SET
                    probability = EXCLUDED.probability,
                    employment_share = EXCLUDED.employment_share
            """, (education_level_id, occ_code, None, probability, emp_share))
        
        self.db.conn.commit()
        self.logger.info(f"✓ Calculated national probabilities for {len(probabilities):,} education-occupation pairs")

class DataValidator:
    """Validate extracted data for completeness and consistency"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
    
    def validate_all_data(self) -> Dict[str, Any]:
        """Run comprehensive data validation"""
        self.logger.info("Starting data validation...")
        
        validation_results = {
            'census_data': self._validate_census_data(),
            'oews_data': self._validate_oews_data(),
            'education_data': self._validate_education_data(),
            'data_quality': self._check_data_quality(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate overall score
        scores = [result.get('score', 0) for result in validation_results.values() if isinstance(result, dict)]
        validation_results['overall_score'] = sum(scores) / len(scores) if scores else 0
        
        self.logger.info(f"✓ Data validation completed (Score: {validation_results['overall_score']:.1f}/100)")
        
        return validation_results
    
    def _validate_census_data(self) -> Dict[str, Any]:
        """Validate Census demographic data"""
        
        # Check state coverage
        self.db.cursor.execute("SELECT COUNT(DISTINCT state_code) FROM census.state_demographics")
        states_count = self.db.cursor.fetchone()[0]
        
        # Check race data coverage
        self.db.cursor.execute("SELECT COUNT(*) FROM census.state_race_ethnicity")
        race_records = self.db.cursor.fetchone()[0]
        
        # Check family structure coverage
        self.db.cursor.execute("SELECT COUNT(*) FROM census.state_family_structures")
        family_records = self.db.cursor.fetchone()[0]
        
        # Check employment data coverage
        self.db.cursor.execute("SELECT COUNT(*) FROM census.state_employment_stats")
        employment_records = self.db.cursor.fetchone()[0]
        
        score = min(100, (states_count / 51) * 100)  # 50 states + DC
        
        return {
            'states_covered': states_count,
            'race_records': race_records,
            'family_records': family_records,
            'employment_records': employment_records,
            'score': score,
            'issues': [] if score > 90 else ['Incomplete state coverage']
        }
    
    def _validate_oews_data(self) -> Dict[str, Any]:
        """Validate OEWS occupation and wage data"""
        
        # Check occupation coverage
        self.db.cursor.execute("SELECT COUNT(DISTINCT occ_code) FROM oews.employment_wages")
        occupations_count = self.db.cursor.fetchone()[0]
        
        # Check state coverage in OEWS
        self.db.cursor.execute("SELECT COUNT(DISTINCT state_code) FROM oews.employment_wages WHERE state_code IS NOT NULL")
        oews_states_count = self.db.cursor.fetchone()[0]
        
        # Check wage data completeness
        self.db.cursor.execute("""
            SELECT COUNT(*) FROM oews.employment_wages 
            WHERE a_mean IS NOT NULL AND h_mean IS NOT NULL
        """)
        wage_records = self.db.cursor.fetchone()[0]
        
        # Check education requirements coverage
        self.db.cursor.execute("SELECT COUNT(*) FROM education.occupation_requirements")
        education_req_count = self.db.cursor.fetchone()[0]
        
        score = min(100, (occupations_count / 800) * 100)  # Expect ~800 detailed occupations
        
        return {
            'occupations_covered': occupations_count,
            'states_with_oews_data': oews_states_count,
            'wage_records': wage_records,
            'education_requirements': education_req_count,
            'score': score,
            'issues': [] if score > 80 else ['Low occupation coverage']
        }
    
    def _validate_education_data(self) -> Dict[str, Any]:
        """Validate education attainment and mapping data"""
        
        # Check education attainment coverage
        self.db.cursor.execute("SELECT COUNT(DISTINCT state_code) FROM education.attainment_demographics")
        edu_states_count = self.db.cursor.fetchone()[0]
        
        # Check education-occupation probabilities
        self.db.cursor.execute("SELECT COUNT(*) FROM education.occupation_probabilities")
        probability_records = self.db.cursor.fetchone()[0]
        
        # Check education levels coverage
        self.db.cursor.execute("SELECT COUNT(*) FROM education_levels")
        education_levels_count = self.db.cursor.fetchone()[0]
        
        score = min(100, (edu_states_count / 51) * 100)
        
        return {
            'education_states_covered': edu_states_count,
            'probability_records': probability_records,
            'education_levels': education_levels_count,
            'score': score,
            'issues': [] if score > 90 else ['Incomplete education data coverage']
        }
    
    def _check_data_quality(self) -> Dict[str, Any]:
        """Check overall data quality and consistency"""
        
        issues = []
        
        # Check for states missing in multiple datasets
        self.db.cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT state_code FROM census.state_demographics
                EXCEPT
                SELECT DISTINCT state_code FROM oews.employment_wages WHERE state_code IS NOT NULL
            ) AS missing_states
        """)
        missing_oews_states = self.db.cursor.fetchone()[0]
        if missing_oews_states > 0:
            issues.append(f"{missing_oews_states} states missing OEWS data")
        
        # Check for occupations without education requirements
        self.db.cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT occ_code FROM oews.employment_wages
                EXCEPT
                SELECT occ_code FROM education.occupation_requirements
            ) AS missing_requirements
        """)
        missing_education_reqs = self.db.cursor.fetchone()[0]
        if missing_education_reqs > 0:
            issues.append(f"{missing_education_reqs} occupations missing education requirements")
        
        # Check for reasonable wage distributions
        self.db.cursor.execute("""
            SELECT COUNT(*) FROM oews.employment_wages 
            WHERE a_mean > 500000 OR a_mean < 15000
        """)
        unreasonable_wages = self.db.cursor.fetchone()[0]
        if unreasonable_wages > 0:
            issues.append(f"{unreasonable_wages} records with unreasonable wages")
        
        score = max(0, 100 - len(issues) * 10)
        
        return {
            'issues_found': len(issues),
            'issues': issues,
            'score': score
        }

class ComprehensiveDataExtractor:
    """Main orchestrator for all data extraction"""
    
    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Environment variables
        self.connection_string = os.getenv('NEON_CONNECTION_STRING')
        self.census_api_key = os.getenv('CENSUS_API_KEY')
        
        if not self.connection_string:
            raise ValueError("NEON_CONNECTION_STRING environment variable is required")
        
        # Initialize components
        self.db_manager = DatabaseManager(self.connection_string)
        self.census_extractor = None
        self.oews_extractor = None
        self.edu_processor = None
        self.validator = None
    
    def run_full_extraction(self, census_only: bool = False, oews_only: bool = False):
        """Run complete data extraction pipeline"""
        self.logger.info("🗄️  COMPREHENSIVE TAX SCENARIO DATA EXTRACTION")
        self.logger.info("=" * 70)
        self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Connect to database
            self.db_manager.connect()
            
            # Create schema
            self.logger.info("Creating database schema...")
            self.db_manager.create_comprehensive_schema()
            
            # Initialize extractors
            self.census_extractor = CensusDataExtractor(self.db_manager, self.census_api_key)
            self.oews_extractor = OEWSDataExtractor(self.db_manager)
            self.edu_processor = EducationOccupationProcessor(self.db_manager)
            self.validator = DataValidator(self.db_manager)
            
            # Run extractions based on flags
            if not oews_only:
                self.logger.info("\n📊 Extracting Census demographic data...")
                self.census_extractor.extract_all_census_data()
            
            if not census_only:
                self.logger.info("\n💼 Extracting OEWS occupation and wage data...")
                self.oews_extractor.extract_all_oews_data()
                
                self.logger.info("\n🎓 Processing education-occupation relationships...")
                self.edu_processor.calculate_education_occupation_probabilities()
            
            # Validate data
            self.logger.info("\n✅ Validating extracted data...")
            validation_results = self.validator.validate_all_data()
            
            # Generate summary report
            self._generate_summary_report(validation_results)
            
            # Store extraction metadata
            self._store_extraction_metadata(validation_results)
            
            self.logger.info("\n🎉 DATA EXTRACTION COMPLETED SUCCESSFULLY!")
            self.logger.info("Database is ready for tax scenario family generation")
            
            return True
            
        except Exception as e:
            self.logger.error(f"\n❌ DATA EXTRACTION FAILED: {e}")
            return False
        finally:
            if self.db_manager:
                self.db_manager.close()
            self.logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _generate_summary_report(self, validation_results: Dict[str, Any]):
        """Generate comprehensive summary report"""
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("DATA EXTRACTION SUMMARY REPORT")
        self.logger.info("=" * 70)
        
        # Census data summary
        census_data = validation_results.get('census_data', {})
        self.logger.info(f"\n📊 Census Data:")
        self.logger.info(f"  States covered: {census_data.get('states_covered', 0)}")
        self.logger.info(f"  Race/ethnicity records: {census_data.get('race_records', 0):,}")
        self.logger.info(f"  Family structure records: {census_data.get('family_records', 0):,}")
        self.logger.info(f"  Employment records: {census_data.get('employment_records', 0):,}")
        
        # OEWS data summary
        oews_data = validation_results.get('oews_data', {})
        self.logger.info(f"\n💼 OEWS Employment Data:")
        self.logger.info(f"  Occupations covered: {oews_data.get('occupations_covered', 0):,}")
        self.logger.info(f"  States with wage data: {oews_data.get('states_with_oews_data', 0)}")
        self.logger.info(f"  Wage records: {oews_data.get('wage_records', 0):,}")
        self.logger.info(f"  Education requirements: {oews_data.get('education_requirements', 0):,}")
        
        # Education data summary
        education_data = validation_results.get('education_data', {})
        self.logger.info(f"\n🎓 Education Data:")
        self.logger.info(f"  States with education data: {education_data.get('education_states_covered', 0)}")
        self.logger.info(f"  Education-occupation probabilities: {education_data.get('probability_records', 0):,}")
        self.logger.info(f"  Education levels: {education_data.get('education_levels', 0)}")
        
        # Data quality summary
        data_quality = validation_results.get('data_quality', {})
        self.logger.info(f"\n✅ Data Quality:")
        self.logger.info(f"  Overall score: {validation_results.get('overall_score', 0):.1f}/100")
        self.logger.info(f"  Issues found: {data_quality.get('issues_found', 0)}")
        
        for issue in data_quality.get('issues', []):
            self.logger.info(f"    ⚠️  {issue}")
        
        # Sample data examples
        self._show_sample_data()
    
    def _show_sample_data(self):
        """Show sample data for verification"""
        self.logger.info(f"\n📋 Sample Data (Top 5 States by Population):")
        
        try:
            self.db_manager.cursor.execute("""
                SELECT sd.state_name, sd.total_population, r.region_name,
                       COUNT(DISTINCT ew.occ_code) as occupations,
                       AVG(ew.a_mean) as avg_wage
                FROM census.state_demographics sd
                JOIN regions r ON sd.region_id = r.id
                LEFT JOIN oews.employment_wages ew ON sd.state_code = ew.state_code
                GROUP BY sd.state_name, sd.total_population, r.region_name
                ORDER BY sd.total_population DESC NULLS LAST
                LIMIT 5
            """)
            
            for row in self.db_manager.cursor.fetchall():
                state_name, pop, region, occupations, avg_wage = row
                pop_str = f"{pop:,}" if pop else "Unknown"
                occ_str = f", {occupations} occupations" if occupations else ""
                wage_str = f", avg wage: ${avg_wage:,.0f}" if avg_wage else ""
                self.logger.info(f"  {state_name} ({region}): Pop {pop_str}{occ_str}{wage_str}")
        
        except Exception as e:
            self.logger.warning(f"Could not display sample data: {e}")
    
    def _store_extraction_metadata(self, validation_results: Dict[str, Any]):
        """Store extraction metadata for tracking"""
        
        try:
            generation_params = {
                'census_year': CONFIG['CENSUS_YEAR'],
                'oews_years': CONFIG['OEWS_YEARS'],
                'extraction_config': CONFIG
            }
            
            self.db_manager.cursor.execute("""
                INSERT INTO tax_scenarios.family_generation_log 
                (total_families_generated, data_version, generation_parameters, validation_results)
                VALUES (%s, %s, %s, %s)
            """, (
                0,  # No families generated yet
                f"v{datetime.now().strftime('%Y%m%d')}",
                json.dumps(generation_params),
                json.dumps(validation_results)
            ))
            
            self.db_manager.conn.commit()
            
        except Exception as e:
            self.logger.warning(f"Could not store extraction metadata: {e}")

def main():
    """Main execution function"""
    
    # Parse command line arguments
    census_only = '--census-only' in sys.argv
    oews_only = '--oews-only' in sys.argv
    validate_only = '--validate' in sys.argv
    
    if census_only and oews_only:
        print("Error: Cannot specify both --census-only and --oews-only")
        sys.exit(1)
    
    try:
        extractor = ComprehensiveDataExtractor()
        
        if validate_only:
            # Just run validation on existing data
            extractor.db_manager.connect()
            validator = DataValidator(extractor.db_manager)
            validation_results = validator.validate_all_data()
            
            print("\n" + "=" * 50)
            print("DATA VALIDATION RESULTS")
            print("=" * 50)
            print(json.dumps(validation_results, indent=2))
            
            extractor.db_manager.close()
            return True
        else:
            # Run full extraction
            success = extractor.run_full_extraction(census_only=census_only, oews_only=oews_only)
            return success
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Pipeline failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)