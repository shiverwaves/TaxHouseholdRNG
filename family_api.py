#!/usr/bin/env python3
"""
Family Generator API

A FastAPI backend service for generating realistic synthetic family data.
Converts the command-line family generator into a REST API.

Usage:
    uvicorn family_api:app --host 0.0.0.0 --port 8000 --reload
"""

import psycopg2
import random
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models for API
class FamilyMember(BaseModel):
    """Individual family member data model"""
    role: str
    age: int
    gender: str
    race: str
    education_level: str
    education_level_key: str
    employment_status: str
    annual_income: int = 0
    occupation: Optional[str] = None
    occupation_code: Optional[str] = None
    occupation_major_group: Optional[str] = None
    employment_type: Optional[str] = None

class Family(BaseModel):
    """Complete family data model"""
    family_id: str
    state_code: str
    state_name: str
    region: str
    race: str
    race_key: str
    family_type: str
    family_structure: str
    family_size: int
    total_household_income: int
    highest_education_level: str
    total_earners: int
    generation_date: str
    data_version: str
    members: List[FamilyMember]

class GenerationRequest(BaseModel):
    """Request model for family generation"""
    count: int = Field(default=5, ge=1, le=100, description="Number of families to generate (1-100)")
    state: Optional[str] = Field(default=None, description="Target state name or code (e.g., 'California' or 'CA')")

class GenerationResponse(BaseModel):
    """Response model for family generation"""
    success: bool
    message: str
    families_generated: int
    generation_time_ms: float
    families: List[Family]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    database_connected: bool
    version: str

class StateInfo(BaseModel):
    """State information model"""
    state_code: str
    state_name: str
    region: str
    population: int
    weight: float

class StatsResponse(BaseModel):
    """Statistics response model"""
    total_states: int
    total_occupations: int
    total_education_levels: int
    database_status: str
    cache_loaded: bool

# Import the enhanced family generator class
class EnhancedFamilyGenerator:
    """Enhanced family generator with realistic income calculations and rich database integration"""
    
    def __init__(self):
        self.connection_string = os.getenv('NEON_CONNECTION_STRING')
        if not self.connection_string:
            raise ValueError("NEON_CONNECTION_STRING environment variable is required")
        
        # Test database connection
        try:
            self.conn = psycopg2.connect(self.connection_string)
            self.cursor = self.conn.cursor()
            
            # Test basic connectivity
            self.cursor.execute("SELECT 1")
            logger.info("✓ Database connection established")
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
        
        # Cache for performance
        self._cache = {}
        self._load_cache_safely()
        
        # Current date for age-based logic
        self.current_year = datetime.now().year
    
    def _load_cache_safely(self):
        """Enhanced cache loading with real demographic data"""
        logger.info("Loading comprehensive data cache...")
        
        try:
            # Initialize cache structure (add new sections)
            self._cache = {
                'states': {},
                'state_weights': {},
                'state_race_data': {},
                'state_family_structures': {},
                'education_levels': {},
                'state_education_attainment': {},
                'occupations': {},
                'state_occupation_wages': {},
                'state_education_occupation_probs': {},
                'national_education_occupation_probs': {},
                # NEW: Real demographic data
                'state_employment_rates': {},
                'education_occupation_probs': {},
                'state_education_distributions': {}
            }
            
            # Load existing data
            self._load_states_safely()
            self._load_education_levels_safely()
            self._load_race_data_safely()
            self._load_family_structures_safely()
            self._load_occupation_data_safely()
            
            # Load NEW comprehensive data
            self._load_real_employment_rates()
            self._load_education_occupation_probabilities()
            self._load_real_education_distributions()
            
            logger.info("✓ Comprehensive cache loaded successfully")
            
        except Exception as e:
            logger.error(f"Cache loading failed: {e}")
            self._create_fallback_cache()

    def _load_real_employment_rates(self):
        """Load actual state-specific employment rates from database"""
        try:
            # Get employment rates by state
            self.cursor.execute("""
                SELECT 
                    sd.state_code,
                    sd.state_name,
                    ses.employment_rate,
                    ses.unemployment_rate,
                    ses.labor_force_participation_rate
                FROM census.state_demographics sd
                JOIN census.state_employment_stats ses ON sd.id = ses.state_id
            """)
            
            self._cache['state_employment_rates'] = {}
            for row in self.cursor.fetchall():
                state_code, state_name, emp_rate, unemp_rate, lfpr = row
                self._cache['state_employment_rates'][state_code] = {
                    'employment_rate': float(emp_rate) if emp_rate else 85.0,
                    'unemployment_rate': float(unemp_rate) if unemp_rate else 5.0,
                    'labor_force_participation_rate': float(lfpr) if lfpr else 63.0
                }
            
            logger.info(f"✓ Loaded real employment rates for {len(self._cache['state_employment_rates'])} states")
            
        except Exception as e:
            logger.warning(f"Failed to load employment rates: {e}")
            # Fallback to national averages
            self._cache['state_employment_rates'] = {}

    def _load_education_occupation_probabilities(self):
        """Load education-occupation probability matrix from database"""
        try:
            # Load state-specific probabilities
            self.cursor.execute("""
                SELECT 
                    el.level_key,
                    eop.occ_code,
                    eop.state_code,
                    eop.probability,
                    eop.employment_share,
                    od.occ_title,
                    od.major_group
                FROM education.occupation_probabilities eop
                JOIN education_levels el ON eop.education_level_id = el.id
                LEFT JOIN oews.occupation_details od ON eop.occ_code = od.occ_code
                WHERE eop.probability > 0.001  -- Only significant probabilities
                ORDER BY eop.state_code, el.level_key, eop.probability DESC
            """)
            
            self._cache['education_occupation_probs'] = {}
            for row in self.cursor.fetchall():
                education_key, occ_code, state_code, probability, emp_share, occ_title, major_group = row
                
                # Create nested structure: state -> education -> occupations
                if state_code not in self._cache['education_occupation_probs']:
                    self._cache['education_occupation_probs'][state_code] = {}
                
                if education_key not in self._cache['education_occupation_probs'][state_code]:
                    self._cache['education_occupation_probs'][state_code][education_key] = []
                
                self._cache['education_occupation_probs'][state_code][education_key].append({
                    'occ_code': occ_code,
                    'probability': float(probability),
                    'employment_share': float(emp_share) if emp_share else 0,
                    'title': occ_title,
                    'major_group': major_group
                })
            
            # Load national fallback probabilities
            self.cursor.execute("""
                SELECT 
                    el.level_key,
                    eop.occ_code,
                    eop.probability,
                    eop.employment_share,
                    od.occ_title,
                    od.major_group
                FROM education.occupation_probabilities eop
                JOIN education_levels el ON eop.education_level_id = el.id
                LEFT JOIN oews.occupation_details od ON eop.occ_code = od.occ_code
                WHERE eop.state_code IS NULL  -- National data
                  AND eop.probability > 0.001
                ORDER BY el.level_key, eop.probability DESC
            """)
            
            self._cache['national_education_occupation_probs'] = {}
            for row in self.cursor.fetchall():
                education_key, occ_code, probability, emp_share, occ_title, major_group = row
                
                if education_key not in self._cache['national_education_occupation_probs']:
                    self._cache['national_education_occupation_probs'][education_key] = []
                
                self._cache['national_education_occupation_probs'][education_key].append({
                    'occ_code': occ_code,
                    'probability': float(probability),
                    'employment_share': float(emp_share) if emp_share else 0,
                    'title': occ_title,
                    'major_group': major_group
                })
            
            logger.info(f"✓ Loaded education-occupation probabilities for {len(self._cache['education_occupation_probs'])} states")
            
        except Exception as e:
            logger.warning(f"Failed to load education-occupation probabilities: {e}")
            self._cache['education_occupation_probs'] = {}
            self._cache['national_education_occupation_probs'] = {}

    def _load_real_education_distributions(self):
        """Load actual education attainment by state from database"""
        try:
            self.cursor.execute("""
                SELECT 
                    ad.state_code,
                    el.level_key,
                    ad.percentage
                FROM education.attainment_demographics ad
                JOIN education_levels el ON ad.education_level_id = el.id
                WHERE ad.race_key = 'All' AND ad.age_group = 'All' AND ad.gender = 'All'
            """)
            
            self._cache['state_education_distributions'] = {}
            for row in self.cursor.fetchall():
                state_code, education_key, percentage = row
                
                if state_code not in self._cache['state_education_distributions']:
                    self._cache['state_education_distributions'][state_code] = {}
                
                self._cache['state_education_distributions'][state_code][education_key] = float(percentage)
            
            logger.info(f"✓ Loaded real education distributions for {len(self._cache['state_education_distributions'])} states")
            
        except Exception as e:
            logger.warning(f"Failed to load education distributions: {e}")
            self._cache['state_education_distributions'] = {}
    
    def _load_states_safely(self):
        """Load state data with error handling"""
        try:
            self.cursor.execute("""
                SELECT sd.state_code, sd.state_name, r.region_name, 
                       sd.total_population, sd.population_weight
                FROM census.state_demographics sd
                LEFT JOIN regions r ON sd.region_id = r.id
                WHERE sd.total_population > 0
                ORDER BY sd.population_weight DESC NULLS LAST
            """)
            
            rows = self.cursor.fetchall()
            
            for row in rows:
                state_code, state_name, region_name, total_pop, pop_weight = row
                
                # Safely handle None values
                region_name = region_name or 'Unknown'
                total_pop = total_pop or 0
                pop_weight = float(pop_weight) if pop_weight else 1.0
                
                self._cache['states'][state_code] = {
                    'name': state_name,
                    'region': region_name,
                    'population': total_pop,
                    'weight': pop_weight
                }
                self._cache['state_weights'][state_code] = pop_weight
            
            logger.info(f"✓ Loaded {len(rows)} states")
            
        except Exception as e:
            logger.warning(f"Failed to load state data: {e}")
            # Create minimal fallback
            self._cache['states']['06'] = {
                'name': 'California', 'region': 'West', 'population': 39000000, 'weight': 12.0
            }
            self._cache['state_weights']['06'] = 12.0
    
    def _load_education_levels_safely(self):
        """Load education levels with error handling"""
        try:
            self.cursor.execute("""
                SELECT level_key, level_name, sort_order, typical_years
                FROM education_levels
                ORDER BY sort_order
            """)
            
            for row in self.cursor.fetchall():
                level_key, level_name, sort_order, typical_years = row
                self._cache['education_levels'][level_key] = {
                    'name': level_name,
                    'sort_order': sort_order or 1,
                    'typical_years': typical_years or 12
                }
            
            logger.info(f"✓ Loaded {len(self._cache['education_levels'])} education levels")
            
        except Exception as e:
            logger.warning(f"Failed to load education levels: {e}")
            # Create fallback education levels
            self._cache['education_levels'] = {
                'less_than_hs': {'name': 'Less than High School', 'sort_order': 1, 'typical_years': 11},
                'high_school': {'name': 'High School Graduate', 'sort_order': 2, 'typical_years': 12},
                'some_college': {'name': 'Some College', 'sort_order': 3, 'typical_years': 14},
                'bachelors': {'name': "Bachelor's Degree", 'sort_order': 4, 'typical_years': 16},
                'graduate': {'name': 'Graduate Degree', 'sort_order': 5, 'typical_years': 18}
            }
    
    def _load_race_data_safely(self):
        """Load race/ethnicity data with error handling"""
        try:
            self.cursor.execute("""
                SELECT sd.state_code, re.race_key, re.race_name, sre.population_percent
                FROM census.state_race_ethnicity sre
                JOIN census.state_demographics sd ON sre.state_id = sd.id
                JOIN race_ethnicity re ON sre.race_id = re.id
                ORDER BY sd.state_code, sre.population_percent DESC
            """)
            
            for row in self.cursor.fetchall():
                state_code, race_key, race_name, race_percent = row
                
                if state_code not in self._cache['state_race_data']:
                    self._cache['state_race_data'][state_code] = {}
                
                self._cache['state_race_data'][state_code][race_key] = {
                    'name': race_name,
                    'percent': float(race_percent) if race_percent else 0.0
                }
            
            logger.info(f"✓ Loaded race data for {len(self._cache['state_race_data'])} states")
            
        except Exception as e:
            logger.warning(f"Failed to load race data: {e}")
            # Create fallback race data
            for state_code in self._cache['states'].keys():
                self._cache['state_race_data'][state_code] = {
                    'WHITE_NON_HISPANIC': {'name': 'White Non-Hispanic', 'percent': 60.0},
                    'HISPANIC': {'name': 'Hispanic or Latino', 'percent': 20.0},
                    'BLACK': {'name': 'Black or African American', 'percent': 12.0},
                    'ASIAN': {'name': 'Asian', 'percent': 8.0}
                }
    
    def _load_family_structures_safely(self):
        """Load family structure data with error handling"""
        try:
            self.cursor.execute("""
                SELECT sd.state_code, fs.structure_key, fs.structure_name, sfs.probability_percent
                FROM census.state_family_structures sfs
                JOIN census.state_demographics sd ON sfs.state_id = sd.id
                JOIN census.family_structures fs ON sfs.structure_id = fs.id
                ORDER BY sd.state_code, sfs.probability_percent DESC
            """)
            
            for row in self.cursor.fetchall():
                state_code, structure_key, structure_name, structure_percent = row
                
                if state_code not in self._cache['state_family_structures']:
                    self._cache['state_family_structures'][state_code] = {}
                
                self._cache['state_family_structures'][state_code][structure_key] = {
                    'name': structure_name,
                    'percent': float(structure_percent) if structure_percent else 0.0
                }
            
            logger.info(f"✓ Loaded family structures for {len(self._cache['state_family_structures'])} states")
            
        except Exception as e:
            logger.warning(f"Failed to load family structure data: {e}")
            # Create fallback family structures
            for state_code in self._cache['states'].keys():
                self._cache['state_family_structures'][state_code] = {
                    'MARRIED_COUPLE': {'name': 'Married Couple Family', 'percent': 50.0},
                    'SINGLE_PERSON': {'name': 'Single Person Household', 'percent': 30.0},
                    'SINGLE_PARENT_FEMALE': {'name': 'Single Mother Household', 'percent': 15.0},
                    'SINGLE_PARENT_MALE': {'name': 'Single Father Household', 'percent': 5.0}
                }
    
    def _load_occupation_data_safely(self):
        """Load occupation data with proper query logic for healthy database"""
        try:
            # Single comprehensive query - get everything we need at once
            self.cursor.execute("""
                SELECT DISTINCT 
                    occ_code, 
                    occ_title, 
                    a_median,           -- Use median (more realistic than mean)
                    a_mean,             -- Keep mean for reference
                    tot_emp,            -- Employment for weighting
                    state_code
                FROM oews.employment_wages
                WHERE occ_code IS NOT NULL 
                  AND occ_title IS NOT NULL 
                  AND a_median IS NOT NULL 
                  AND a_median > 15000    -- Filter unrealistic wages
                  AND a_median < 500000   -- Filter unrealistic wages
                  AND tot_emp IS NOT NULL 
                  AND tot_emp > 1000      -- Only jobs with significant employment
                ORDER BY tot_emp DESC     -- Prioritize common jobs
            """)
            
            occupations = {}
            state_wages = {}
            total_employment = 0
            
            for row in self.cursor.fetchall():
                occ_code, occ_title, a_median, a_mean, tot_emp, state_code = row
                
                # Build occupation info (once per occupation)
                if occ_code not in occupations:
                    occupations[occ_code] = {
                        'title': occ_title,
                        'major_group': self._get_major_group_from_code(occ_code),
                        'education_level': self._estimate_education_requirement(occ_title),
                        'median_wage': float(a_median),
                        'mean_wage': float(a_mean) if a_mean else float(a_median),
                        'total_employment': int(tot_emp),
                        'employment_weight': 0  # Will calculate after loop
                    }
                    total_employment += int(tot_emp)
                
                # Build state-specific wage data
                if state_code not in state_wages:
                    state_wages[state_code] = {}
                
                state_wages[state_code][occ_code] = {
                    'annual_median': float(a_median),
                    'annual_mean': float(a_mean) if a_mean else float(a_median)
                }
            
            # Calculate employment weights (probability of each job)
            if total_employment > 0:
                for occ_code, occ_data in occupations.items():
                    occ_data['employment_weight'] = occ_data['total_employment'] / total_employment
            
            # Store in cache
            if occupations:
                self._cache['occupations'] = occupations
                self._cache['state_occupation_wages'] = state_wages
                
                logger.info(f"✅ Loaded {len(occupations)} occupations from database")
                logger.info(f"✅ Loaded wage data for {len(state_wages)} states")
                logger.info(f"✅ Total employment represented: {total_employment:,}")
                
            else:
                logger.warning("❌ No occupations loaded from database - check data quality")
                self._create_fallback_occupations()
                
        except Exception as e:
            logger.error(f"Failed to load occupation data: {e}")
            self._create_fallback_occupations()
    
    def _create_fallback_occupations(self):
        """Create fallback occupations with realistic wages based on actual employment data"""
        
        # Based on actual BLS employment numbers and MEDIAN wages
        realistic_occupations = {
            # HIGH EMPLOYMENT, LOW-MODERATE WAGES (most common jobs)
            '41-2031': {  # Retail Salespersons - 3.2 million employed, $31k median
                'title': 'Retail Salespersons', 'major_group': 'Sales', 'education_level': 'high_school', 
                'median_wage': 31000, 'total_employment': 3200000, 'employment_weight': 0.08
            },
            '35-3021': {  # Food Prep Workers - 3.1 million employed, $26k median
                'title': 'Combined Food Preparation Workers', 'major_group': 'Food Service', 'education_level': 'less_than_hs',
                'median_wage': 26000, 'total_employment': 3100000, 'employment_weight': 0.078
            },
            '41-2011': {  # Cashiers - 2.9 million employed, $28k median
                'title': 'Cashiers', 'major_group': 'Sales', 'education_level': 'less_than_hs',
                'median_wage': 28000, 'total_employment': 2900000, 'employment_weight': 0.073
            },
            '43-4051': {  # Customer Service - 2.8 million employed, $37k median
                'title': 'Customer Service Representatives', 'major_group': 'Office Support', 'education_level': 'high_school',
                'median_wage': 37000, 'total_employment': 2800000, 'employment_weight': 0.07
            },
            '53-7062': {  # Laborers - 2.5 million employed, $32k median
                'title': 'Laborers and Freight Movers', 'major_group': 'Transportation', 'education_level': 'less_than_hs',
                'median_wage': 32000, 'total_employment': 2500000, 'employment_weight': 0.063
            },
            '29-1141': {  # Registered Nurses - 3.2 million employed, $75k median
                'title': 'Registered Nurses', 'major_group': 'Healthcare', 'education_level': 'bachelors',
                'median_wage': 75000, 'total_employment': 3200000, 'employment_weight': 0.08
            },
            '25-2021': {  # Elementary Teachers - 1.3 million employed, $58k median
                'title': 'Elementary School Teachers', 'major_group': 'Education', 'education_level': 'bachelors',
                'median_wage': 58000, 'total_employment': 1300000, 'employment_weight': 0.033
            },
            '15-1211': {  # Computer Analysts - 600k employed, $95k median
                'title': 'Computer Systems Analysts', 'major_group': 'Computer', 'education_level': 'bachelors',
                'median_wage': 95000, 'total_employment': 600000, 'employment_weight': 0.015
            }
        }
        
        self._cache['occupations'] = realistic_occupations
        
        # Create realistic wage data for each state
        for state_code in self._cache['states'].keys():
            self._cache['state_occupation_wages'][state_code] = {}
            for occ_code, occ_data in realistic_occupations.items():
                base_salary = occ_data['median_wage']
                # Add small state variation (±10%)
                state_variation = random.uniform(0.9, 1.1)
                
                self._cache['state_occupation_wages'][state_code][occ_code] = {
                    'annual_median': int(base_salary * state_variation),
                    'annual_mean': int(base_salary * state_variation * 1.1)
                }
        
        logger.info(f"✓ Created realistic fallback occupations with employment weighting")
    
    def _create_fallback_cache(self):
        """Create minimal fallback cache if loading fails"""
        logger.warning("Creating fallback cache with minimal data")
        
        self._cache = {
            'states': {
                '06': {'name': 'California', 'region': 'West', 'population': 39000000, 'weight': 12.0},
                '48': {'name': 'Texas', 'region': 'South', 'population': 29000000, 'weight': 9.0},
                '12': {'name': 'Florida', 'region': 'South', 'population': 22000000, 'weight': 7.0}
            },
            'state_weights': {
                '06': 12.0,
                '48': 9.0,
                '12': 7.0
            },
            'education_levels': {
                'less_than_hs': {'name': 'Less than High School', 'sort_order': 1, 'typical_years': 11},
                'high_school': {'name': 'High School Graduate', 'sort_order': 2, 'typical_years': 12},
                'some_college': {'name': 'Some College', 'sort_order': 3, 'typical_years': 14},
                'bachelors': {'name': "Bachelor's Degree", 'sort_order': 4, 'typical_years': 16},
                'graduate': {'name': 'Graduate Degree', 'sort_order': 5, 'typical_years': 18}
            },
            'state_race_data': {},
            'state_family_structures': {},
            'occupations': {},
            'state_occupation_wages': {},
            'state_education_attainment': {},
            'state_education_occupation_probs': {},
            'national_education_occupation_probs': {},
            'state_employment_rates': {},
            'education_occupation_probs': {},
            'state_education_distributions': {}
        }
        
        # Fill in basic data for each state
        for state_code in self._cache['states'].keys():
            # Race data
            self._cache['state_race_data'][state_code] = {
                'WHITE_NON_HISPANIC': {'name': 'White Non-Hispanic', 'percent': 60.0},
                'HISPANIC': {'name': 'Hispanic or Latino', 'percent': 20.0},
                'BLACK': {'name': 'Black or African American', 'percent': 12.0},
                'ASIAN': {'name': 'Asian', 'percent': 8.0}
            }
            
            # Family structures
            self._cache['state_family_structures'][state_code] = {
                'MARRIED_COUPLE': {'name': 'Married Couple Family', 'percent': 50.0},
                'SINGLE_PERSON': {'name': 'Single Person Household', 'percent': 30.0},
                'SINGLE_PARENT_FEMALE': {'name': 'Single Mother Household', 'percent': 15.0},
                'SINGLE_PARENT_MALE': {'name': 'Single Father Household', 'percent': 5.0}
            }

    # Include all the other methods from the original class
    # (I'm including the essential ones for the API, but in a real implementation,
    # you'd copy all methods from the original EnhancedFamilyGenerator class)
    
    def _get_major_group_from_code(self, occ_code: str) -> str:
        """Get major group from occupation code"""
        if not occ_code or len(occ_code) < 2:
            return 'Other'
        
        major_groups = {
            '11': 'Management',
            '13': 'Business and Financial',
            '15': 'Computer and Mathematical',
            '17': 'Architecture and Engineering',
            '19': 'Life, Physical, and Social Science',
            '21': 'Community and Social Service',
            '23': 'Legal',
            '25': 'Educational',
            '27': 'Arts and Entertainment',
            '29': 'Healthcare Practitioners',
            '31': 'Healthcare Support',
            '33': 'Protective Service',
            '35': 'Food Service',
            '37': 'Building and Maintenance',
            '39': 'Personal Care',
            '41': 'Sales',
            '43': 'Office Support',
            '45': 'Farming and Forestry',
            '47': 'Construction',
            '49': 'Installation and Repair',
            '51': 'Production',
            '53': 'Transportation'
        }
        
        return major_groups.get(occ_code[:2], 'Other')
    
    def _estimate_education_requirement(self, occ_title: str) -> str:
        """Estimate education requirement from occupation title"""
        title_lower = occ_title.lower() if occ_title else ""
        
        if any(word in title_lower for word in ['chief', 'director', 'manager', 'executive']):
            return 'graduate'
        elif any(word in title_lower for word in ['engineer', 'analyst', 'teacher', 'nurse']):
            return 'bachelors'
        elif any(word in title_lower for word in ['technician', 'assistant', 'specialist']):
            return 'some_college'
        elif any(word in title_lower for word in ['clerk', 'representative', 'operator']):
            return 'high_school'
        else:
            return 'high_school'

    def _select_occupation_using_probabilities(self, education_level: str, age: int, state_code: str) -> Optional[Dict[str, Any]]:
        """Select occupation using real education-occupation probability matrix"""
        
        if age < 16:
            return None
        elif 16 <= age <= 18:
            # Teenagers - use entry-level probabilities but limited to service jobs
            teen_suitable_groups = ['Food Preparation', 'Sales', 'Personal Care']
            return self._select_teen_occupation(education_level, state_code, teen_suitable_groups)
        
        # Get state-specific probabilities, fallback to national
        state_probs = self._cache['education_occupation_probs'].get(state_code, {})
        education_probs = state_probs.get(education_level, [])
        
        if not education_probs:
            # Fallback to national probabilities
            education_probs = self._cache['national_education_occupation_probs'].get(education_level, [])
        
        if not education_probs:
            return None
        
        # Apply employment rate for this education level and state
        employment_rate = self._get_real_employment_rate(education_level, state_code)
        
        if random.random() > employment_rate:
            return self._generate_unemployment_benefits(age, education_level, state_code)
        
        # Select occupation based on probabilities
        total_prob = sum(occ['probability'] for occ in education_probs)
        if total_prob <= 0:
            return None
        
        rand_val = random.uniform(0, total_prob)
        cumulative_prob = 0
        
        for occ_data in education_probs:
            cumulative_prob += occ_data['probability']
            if rand_val <= cumulative_prob:
                return self._get_occupation_with_real_wages(occ_data['occ_code'], state_code)
        
        # Fallback
        return self._get_occupation_with_real_wages(education_probs[0]['occ_code'], state_code)

    def _select_teen_occupation(self, education_level: str, state_code: str, suitable_groups: List[str]) -> Optional[Dict[str, Any]]:
        """Helper method for teen occupation selection"""
        teen_jobs = []
        for occ_code, occ_data in self._cache['occupations'].items():
            if occ_data.get('major_group') in suitable_groups:
                teen_jobs.append(occ_code)
        
        if teen_jobs and random.random() < 0.35:  # 35% teen employment rate
            occ_code = random.choice(teen_jobs)
            return self._get_occupation_with_real_wages(occ_code, state_code, part_time=True)
        return None

    def _get_real_employment_rate(self, education_level: str, state_code: str) -> float:
        """Get real employment rate for education level and state"""
        
        # Get state employment rate
        state_emp_data = self._cache['state_employment_rates'].get(state_code, {})
        base_employment_rate = state_emp_data.get('employment_rate', 85.0) / 100
        
        # Adjust based on education level (education improves employment prospects)
        education_multipliers = {
            'less_than_hs': 0.85,  # Below average
            'high_school': 0.95,   # Near average  
            'some_college': 1.02,  # Slightly above
            'bachelors': 1.08,     # Above average
            'graduate': 1.12       # Well above average
        }
        
        multiplier = education_multipliers.get(education_level, 1.0)
        adjusted_rate = min(0.98, base_employment_rate * multiplier)  # Cap at 98%
        
        return adjusted_rate

    def _select_education_level_using_real_data(self, age: int, state_code: str, race_key: str) -> str:
        """Select education level using real state demographic data"""
        
        if age < 18:
            return 'less_than_hs'
        
        # Get state-specific education distribution
        state_edu_dist = self._cache['state_education_distributions'].get(state_code, {})
        
        if state_edu_dist:
            # Use real state data
            education_weights = []
            education_levels = []
            
            for edu_level, percentage in state_edu_dist.items():
                education_levels.append(edu_level)
                education_weights.append(percentage)
            
            if education_weights and sum(education_weights) > 0:
                # Age-based adjustments to reflect realistic patterns
                if 18 <= age <= 24:
                    # Young adults - boost some_college, reduce graduate
                    adjusted_weights = []
                    for i, edu_level in enumerate(education_levels):
                        weight = education_weights[i]
                        if edu_level == 'some_college':
                            weight *= 1.5  # More young people in college
                        elif edu_level == 'graduate':
                            weight *= 0.3  # Fewer young graduates
                        elif edu_level == 'bachelors':
                            weight *= 0.7  # Some haven't finished yet
                        adjusted_weights.append(weight)
                    education_weights = adjusted_weights
                
                elif age >= 65:
                    # Older adults - reflect historical lower education
                    adjusted_weights = []
                    for i, edu_level in enumerate(education_levels):
                        weight = education_weights[i]
                        if edu_level in ['less_than_hs', 'high_school']:
                            weight *= 1.3  # Higher historical rates
                        elif edu_level in ['bachelors', 'graduate']:
                            weight *= 0.7  # Lower historical rates
                        adjusted_weights.append(weight)
                    education_weights = adjusted_weights
                
                # Weighted random selection
                return random.choices(education_levels, weights=education_weights)[0]
        
        # Fallback to age-based logic if no state data
        return self._fallback_education_selection(age)

    def _fallback_education_selection(self, age: int) -> str:
        """Fallback education selection based on age"""
        if age < 18:
            return 'less_than_hs'
        elif 18 <= age <= 24:
            return random.choices(
                ['less_than_hs', 'high_school', 'some_college', 'bachelors'],
                weights=[20, 50, 25, 5]
            )[0]
        elif age >= 65:
            return random.choices(
                ['less_than_hs', 'high_school', 'some_college', 'bachelors', 'graduate'],
                weights=[30, 40, 15, 12, 3]
            )[0]
        else:
            return random.choices(
                ['less_than_hs', 'high_school', 'some_college', 'bachelors', 'graduate'],
                weights=[12, 40, 28, 15, 5]
            )[0]

    def _get_occupation_with_real_wages(self, occ_code: str, state_code: str, part_time: bool = False) -> Dict[str, Any]:
        """Get occupation with real wages from OEWS database"""
        
        try:
            # Get real wage data from database
            self.cursor.execute("""
                SELECT 
                    ew.occ_title,
                    ew.a_median,
                    ew.a_mean,
                    ew.tot_emp,
                    od.major_group,
                    er.typical_education_level_id,
                    el.level_key
                FROM oews.employment_wages ew
                LEFT JOIN oews.occupation_details od ON ew.occ_code = od.occ_code
                LEFT JOIN education.occupation_requirements er ON ew.occ_code = er.occ_code
                LEFT JOIN education_levels el ON er.typical_education_level_id = el.id
                WHERE ew.occ_code = %s AND ew.state_code = %s
                ORDER BY ew.year DESC
                LIMIT 1
            """, (occ_code, state_code))
            
            result = self.cursor.fetchone()
            
            if result:
                occ_title, a_median, a_mean, tot_emp, major_group, edu_level_id, edu_key = result
                
                # Use median wage (more realistic)
                base_wage = float(a_median) if a_median else float(a_mean) if a_mean else 35000
                
                # Handle part-time work
                if part_time:
                    base_wage = int(base_wage * 0.4)  # ~40% of full-time
                
                # Small realistic variation (±10%)
                annual_income = int(base_wage * random.uniform(0.9, 1.1))
                
                return {
                    'occ_code': occ_code,
                    'title': occ_title or 'Unknown Occupation',
                    'major_group': major_group or 'Other',
                    'education_requirement': edu_key or 'high_school',
                    'annual_income': max(annual_income, 15000),  # Minimum wage floor
                    'employment_type': 'Part-time' if part_time else 'Full-time',
                    'data_source': 'database'
                }
        
        except Exception as e:
            logger.debug(f"Error getting real wages for {occ_code}: {e}")
        
        # Fallback to cache if database query fails
        return self._get_occupation_details_realistic(occ_code, state_code, part_time)

    def _generate_unemployment_benefits(self, age: int, education_level: str, state_code: str) -> Optional[Dict[str, Any]]:
        """Generate realistic unemployment benefits based on state and education"""
        
        if age < 18:
            return None
        elif age >= 67:
            # Social Security benefits
            monthly_ss = random.randint(800, 2800)  # Realistic SS range
            return {
                'occ_code': 'RETIRED',
                'title': 'Retired (Social Security)',
                'major_group': 'Retired',
                'education_requirement': education_level,
                'annual_income': monthly_ss * 12,
                'employment_type': 'Retired',
                'data_source': 'calculated'
            }
        
        # Get state unemployment rate to determine benefit probability
        state_emp_data = self._cache['state_employment_rates'].get(state_code, {})
        unemployment_rate = state_emp_data.get('unemployment_rate', 5.0)
        
        # Higher unemployment rate = higher chance of getting benefits
        benefit_probability = min(0.8, unemployment_rate / 10 + 0.4)
        
        if random.random() < benefit_probability:
            # Calculate unemployment benefits based on education level (proxy for prior wages)
            base_weekly_benefit = {
                'less_than_hs': 200,
                'high_school': 280,
                'some_college': 350,
                'bachelors': 450,
                'graduate': 520
            }.get(education_level, 280)
            
            # Add state variation (±20%)
            state_variation = random.uniform(0.8, 1.2)
            weekly_benefit = int(base_weekly_benefit * state_variation)
            
            # Assume 26 weeks of benefits (typical duration)
            annual_benefit = weekly_benefit * 26
            
            return {
                'occ_code': 'UNEMPLOYED',
                'title': 'Unemployed (Receiving Benefits)',
                'major_group': 'Unemployed',
                'education_requirement': education_level,
                'annual_income': annual_benefit,
                'employment_type': 'Unemployed',
                'data_source': 'calculated'
            }
        else:
            # Disability or other support
            monthly_benefit = random.randint(700, 1400)
            return {
                'occ_code': 'DISABLED',
                'title': 'Receiving Disability Benefits',
                'major_group': 'Disabled',
                'education_requirement': education_level,
                'annual_income': monthly_benefit * 12,
                'employment_type': 'Disabled',
                'data_source': 'calculated'
            }

    def _weighted_random_selection(self, items: Dict[str, Any], weight_key: str = 'percent') -> str:
        """Select item based on weighted probabilities - FIXED VERSION"""
        if not items:
            return None
        
        total_weight = 0
        cumulative_weights = []
        item_keys = list(items.keys())
        
        for key in item_keys:
            # Handle both dictionary values and direct numeric values
            if isinstance(items[key], dict):
                weight = items[key].get(weight_key, items[key].get('weight', 1))
            elif isinstance(items[key], (int, float)):
                # Direct numeric value (like state weights)
                weight = items[key]
            else:
                # Fallback
                weight = 1
            
            # Ensure weight is numeric
            try:
                weight = float(weight)
            except (ValueError, TypeError):
                weight = 1.0
            
            total_weight += weight
            cumulative_weights.append(total_weight)
        
        if total_weight == 0:
            return random.choice(item_keys)
        
        rand_val = random.uniform(0, total_weight)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return item_keys[i]
        
        return item_keys[0]
    
    def _select_state(self, target_state: str = None) -> str:
        """Select state based on population weights"""
        if target_state:
            target_state_upper = target_state.upper()
            if target_state_upper in self._cache['states']:
                return target_state_upper
            # Try to find by state name
            for state_code, state_data in self._cache['states'].items():
                if state_data['name'].upper() == target_state_upper:
                    return state_code
            raise ValueError(f"State '{target_state}' not found")
        
        return self._weighted_random_selection(self._cache['state_weights'], 'weight')
    
    def _select_race_ethnicity(self, state_code: str) -> Dict[str, str]:
        """Select race/ethnicity based on state demographics"""
        state_races = self._cache['state_race_data'].get(state_code, {})
        
        if not state_races:
            race_key = 'WHITE_NON_HISPANIC'  # Fallback
            race_name = 'White Non-Hispanic'
        else:
            race_key = self._weighted_random_selection(state_races, 'percent')
            race_name = state_races.get(race_key, {}).get('name', 'White Non-Hispanic')
        
        return {'race_key': race_key, 'race_name': race_name}
    
    def _select_family_structure(self, state_code: str) -> Dict[str, str]:
        """Select family structure based on state demographics"""
        state_structures = self._cache['state_family_structures'].get(state_code, {})
        
        if not state_structures:
            structure_key = random.choice(['MARRIED_COUPLE', 'SINGLE_PERSON', 'SINGLE_PARENT_FEMALE'])
        else:
            structure_key = self._weighted_random_selection(state_structures, 'percent')
        
        structure_name = state_structures.get(structure_key, {}).get('name', structure_key.replace('_', ' ').title())
        
        return {'structure_key': structure_key, 'structure_name': structure_name}
    
    def _generate_age(self, role: str, context: Dict = None) -> int:
        """Generate realistic age based on role and context"""
        context = context or {}
        
        if role == 'HEAD':
            if context.get('has_children'):
                return random.randint(25, 55)
            else:
                return random.randint(25, 75)
        elif role == 'SPOUSE':
            head_age = context.get('head_age', 40)
            age_diff = random.randint(-8, 8)
            return max(18, min(80, head_age + age_diff))
        elif role == 'CHILD':
            parent_age = context.get('parent_age', 35)
            return random.randint(0, min(17, parent_age - 18))
        elif role == 'SINGLE_PARENT':
            return random.randint(18, 55)
        else:
            return random.randint(18, 80)
    
    def _generate_gender(self, role: str, context: Dict = None) -> str:
        """Generate gender with realistic probabilities"""
        context = context or {}
        
        if role == 'SPOUSE':
            head_gender = context.get('head_gender', 'Male')
            return 'Female' if head_gender == 'Male' else 'Male'
        elif role == 'SINGLE_PARENT_FEMALE':
            return 'Female'
        elif role == 'SINGLE_PARENT_MALE':
            return 'Male'
        else:
            return 'Female' if random.random() < 0.51 else 'Male'
    
    def _determine_employment_status(self, age: int, education_level: str, occupation: Dict = None) -> str:
        """Determine employment status with realistic rates"""
        
        if age < 16:
            return "Child (Under 16)"
        elif age >= 67:
            return "Retired" if random.random() < 0.85 else "Employed"
        elif 16 <= age <= 18:
            if occupation:
                return "Student (Employed)"
            else:
                return "Student" if random.random() < 0.65 else "Unemployed"
        elif occupation:
            return "Employed"
        else:
            # More realistic unemployment and non-participation rates
            return random.choices(
                ["Unemployed", "Not in Labor Force"],
                weights=[30, 70]
            )[0]
    
    def _create_family_member(self, role: str, age: int, gender: str, race_name: str,
                             education_level: str, occupation: Dict = None, 
                             employment_status: str = "Not in Labor Force") -> Dict[str, Any]:
        """Create family member with all details"""
        
        member = {
            "role": role,
            "age": age,
            "gender": gender,
            "race": race_name,
            "education_level": self._cache['education_levels'].get(education_level, {}).get('name', education_level),
            "education_level_key": education_level,
            "employment_status": employment_status,
            "annual_income": 0
        }
        
        if occupation:
            member.update({
                "occupation": occupation['title'],
                "occupation_code": occupation['occ_code'],
                "occupation_major_group": occupation['major_group'],
                "annual_income": occupation['annual_income'],
                "employment_type": occupation.get('employment_type', 'Full-time')
            })
        
        return member
    
    def _generate_family_members(self, structure_key: str, race_info: Dict, state_code: str) -> List[Dict]:
        """Generate family members based on structure"""
        
        if structure_key == 'MARRIED_COUPLE':
            return self._generate_married_couple_family(race_info, state_code)
        elif structure_key == 'SINGLE_PERSON':
            return self._generate_single_person_household(race_info, state_code)
        elif structure_key == 'SINGLE_PARENT_FEMALE':
            return self._generate_single_parent_family(race_info, state_code, 'Female')
        elif structure_key == 'SINGLE_PARENT_MALE':
            return self._generate_single_parent_family(race_info, state_code, 'Male')
        else:
            return self._generate_married_couple_family(race_info, state_code)
    
    def _generate_married_couple_family(self, race_info: Dict, state_code: str) -> List[Dict]:
        """Generate married couple family"""
        members = []
        
        # Head of household
        head_age = self._generate_age('HEAD', {'has_children': True})
        head_gender = self._generate_gender('HEAD')
        head_education = self._select_education_level(head_age, state_code, race_info['race_key'])
        head_occupation = self._select_occupation(head_education, head_age, state_code)
        head_employment = self._determine_employment_status(head_age, head_education, head_occupation)
        
        head = self._create_family_member(
            'Head of Household', head_age, head_gender, race_info['race_name'],
            head_education, head_occupation, head_employment
        )
        members.append(head)
        
        # Spouse
        spouse_age = self._generate_age('SPOUSE', {'head_age': head_age})
        spouse_gender = self._generate_gender('SPOUSE', {'head_gender': head_gender})
        spouse_education = self._select_education_level(spouse_age, state_code, race_info['race_key'])
        spouse_occupation = self._select_occupation(spouse_education, spouse_age, state_code)
        spouse_employment = self._determine_employment_status(spouse_age, spouse_education, spouse_occupation)
        
        spouse = self._create_family_member(
            'Spouse', spouse_age, spouse_gender, race_info['race_name'],
            spouse_education, spouse_occupation, spouse_employment
        )
        members.append(spouse)
        
        # Children (maybe)
        if random.random() < 0.7:  # 70% chance of having children
            child_count = random.choices([1, 2, 3], weights=[40, 40, 20])[0]
            for i in range(child_count):
                child_age = self._generate_age('CHILD', {'parent_age': max(head_age, spouse_age)})
                child_gender = self._generate_gender('CHILD')
                child_education = self._select_education_level(child_age, state_code, race_info['race_key'])
                child_occupation = self._select_occupation(child_education, child_age, state_code)
                child_employment = self._determine_employment_status(child_age, child_education, child_occupation)
                
                child = self._create_family_member(
                    'Child', child_age, child_gender, race_info['race_name'],
                    child_education, child_occupation, child_employment
                )
                members.append(child)
        
        return members
    
    def _generate_single_parent_family(self, race_info: Dict, state_code: str, parent_gender: str) -> List[Dict]:
        """Generate single parent family"""
        members = []
        
        # Single parent
        parent_age = self._generate_age('SINGLE_PARENT')
        parent_education = self._select_education_level(parent_age, state_code, race_info['race_key'])
        parent_occupation = self._select_occupation(parent_education, parent_age, state_code)
        parent_employment = self._determine_employment_status(parent_age, parent_education, parent_occupation)
        
        parent = self._create_family_member(
            'Head of Household', parent_age, parent_gender, race_info['race_name'],
            parent_education, parent_occupation, parent_employment
        )
        members.append(parent)
        
        # Children
        child_count = random.choices([1, 2], weights=[70, 30])[0]
        for i in range(child_count):
            child_age = self._generate_age('CHILD', {'parent_age': parent_age})
            child_gender = self._generate_gender('CHILD')
            child_education = self._select_education_level(child_age, state_code, race_info['race_key'])
            child_occupation = self._select_occupation(child_education, child_age, state_code)
            child_employment = self._determine_employment_status(child_age, child_education, child_occupation)
            
            child = self._create_family_member(
                'Child', child_age, child_gender, race_info['race_name'],
                child_education, child_occupation, child_employment
            )
            members.append(child)
        
        return members
    
    def _generate_single_person_household(self, race_info: Dict, state_code: str) -> List[Dict]:
        """Generate single person household"""
        age = self._generate_age('HEAD', {'has_children': False})
        gender = self._generate_gender('HEAD')
        education = self._select_education_level(age, state_code, race_info['race_key'])
        occupation = self._select_occupation(education, age, state_code)
        employment = self._determine_employment_status(age, education, occupation)
        
        person = self._create_family_member(
            'Head of Household', age, gender, race_info['race_name'],
            education, occupation, employment
        )
        
        return [person]
    
    def _calculate_total_household_income(self, members: List[Dict]) -> int:
        """Calculate total household income"""
        return sum(member.get('annual_income', 0) for member in members)
    
    def _get_highest_education_level(self, members: List[Dict]) -> str:
        """Get highest education level in household"""
        education_order = ['less_than_hs', 'high_school', 'some_college', 'bachelors', 'graduate']
        
        highest_level = 'less_than_hs'
        for member in members:
            member_education = member.get('education_level_key', 'less_than_hs')
            if education_order.index(member_education) > education_order.index(highest_level):
                highest_level = member_education
        
        return highest_level

    def _select_occupation(self, education_level: str, age: int, state_code: str) -> Optional[Dict[str, Any]]:
        """Enhanced occupation selection using real database probabilities"""
        return self._select_occupation_using_probabilities(education_level, age, state_code)

    def _select_education_level(self, age: int, state_code: str, race_key: str) -> str:
        """Enhanced education selection using real state data"""
        return self._select_education_level_using_real_data(age, state_code, race_key)

    def generate_family(self, target_state: str = None) -> Dict[str, Any]:
        """Generate a complete family"""
        
        try:
            # Select state
            state_code = self._select_state(target_state)
            state_info = self._cache['states'][state_code]
            
            # Select race/ethnicity
            race_info = self._select_race_ethnicity(state_code)
            
            # Select family structure
            structure_info = self._select_family_structure(state_code)
            structure_key = structure_info['structure_key']
            
            # Generate family members
            members = self._generate_family_members(structure_key, race_info, state_code)
            
            # Calculate totals
            total_income = self._calculate_total_household_income(members)
            
            return {
                "family_id": f"FAM_{random.randint(100000, 999999)}",
                "state_code": state_code,
                "state_name": state_info['name'],
                "region": state_info['region'],
                "race": race_info['race_name'],
                "race_key": race_info['race_key'],
                "family_type": structure_info['structure_name'],
                "family_structure": structure_info['structure_name'],
                "family_size": len(members),
                "total_household_income": total_income,
                "highest_education_level": self._get_highest_education_level(members),
                "total_earners": sum(1 for member in members if member.get('annual_income', 0) > 0),
                "generation_date": datetime.now().isoformat(),
                "data_version": "enhanced_v1",
                "members": members
            }
            
        except Exception as e:
            logger.error(f"Family generation failed: {e}")
            raise
    
    def _get_occupation_details_realistic(self, occ_code: str, state_code: str, part_time: bool = False) -> Dict[str, Any]:
        """Fallback method for getting occupation details using cache"""
        
        occ_details = self._cache['occupations'].get(occ_code, {})
        state_wages = self._cache['state_occupation_wages'].get(state_code, {}).get(occ_code, {})
        
        # Use actual database wages (prefer median over mean)
        if state_wages and state_wages.get('annual_median'):
            base_income = state_wages['annual_median']
        elif occ_details.get('median_wage'):
            base_income = occ_details['median_wage']
        else:
            base_income = 35000  # Last resort fallback
        
        # Handle part-time work (mainly for teenagers)
        if part_time:
            base_income = int(base_income * 0.4)  # ~40% of full-time
        
        # Small realistic variation (±12%)
        annual_income = int(base_income * random.uniform(0.88, 1.12))
        
        return {
            'occ_code': occ_code,
            'title': occ_details.get('title', 'Unknown Occupation'),
            'major_group': occ_details.get('major_group', 'Other'),
            'education_requirement': occ_details.get('education_level', 'high_school'),
            'annual_income': max(annual_income, 15000),  # Minimum wage floor
            'employment_type': 'Part-time' if part_time else 'Full-time'
        }

    def generate_families(self, count: int = 5, target_state: str = None) -> List[Dict[str, Any]]:
        """Generate multiple families"""
        families = []
        
        logger.info(f"🏠 Starting generation of {count} families with enhanced database integration...")
        if target_state:
            logger.info(f"🎯 Target state: {target_state}")
        
        for i in range(count):
            try:
                logger.debug(f"Generating family {i + 1}/{count}...")
                family = self.generate_family(target_state)
                families.append(family)
                
                # More frequent progress updates for debugging
                if (i + 1) % 1 == 0 or i == count - 1:
                    logger.info(f"  ✅ Generated {i + 1}/{count} families")
                    
            except Exception as e:
                logger.warning(f"❌ Failed to generate family {i + 1}: {e}")
                logger.debug(f"Full error: {e}", exc_info=True)
                continue
        
        logger.info(f"🎉 Completed generation: {len(families)} families successfully created")
        return families
    
    def get_available_states(self) -> List[StateInfo]:
        """Get list of available states"""
        states = []
        for state_code, state_data in self._cache['states'].items():
            states.append(StateInfo(
                state_code=state_code,
                state_name=state_data['name'],
                region=state_data['region'],
                population=state_data['population'],
                weight=state_data['weight']
            ))
        return sorted(states, key=lambda x: x.state_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics"""
        return {
            'total_states': len(self._cache['states']),
            'total_occupations': len(self._cache['occupations']),
            'total_education_levels': len(self._cache['education_levels']),
            'database_status': 'connected' if self.conn else 'disconnected',
            'cache_loaded': bool(self._cache)
        }
    
    def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            self.cursor.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

# Global generator instance
generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global generator
    
    # Startup
    try:
        logger.info("🚀 Starting Family Generator API...")
        generator = EnhancedFamilyGenerator()
        logger.info("✅ Family Generator API started successfully")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to start API: {e}")
        raise
    finally:
        # Shutdown
        if generator:
            generator.close()
            logger.info("🔄 Family Generator API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Family Generator API",
    description="Generate realistic synthetic family data using demographic databases",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get generator instance
def get_generator() -> EnhancedFamilyGenerator:
    if generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")
    return generator

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Family Generator API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(gen: EnhancedFamilyGenerator = Depends(get_generator)):
    """Health check endpoint"""
    
    database_connected = gen.health_check()
    
    return HealthResponse(
        status="healthy" if database_connected else "unhealthy",
        timestamp=datetime.now().isoformat(),
        database_connected=database_connected,
        version="1.0.0"
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_families(
    request: GenerationRequest,
    gen: EnhancedFamilyGenerator = Depends(get_generator)
):
    """Generate synthetic families"""
    
    start_time = datetime.now()
    
    try:
        # Validate state if provided
        if request.state:
            available_states = gen.get_available_states()
            state_names = [s.state_name.lower() for s in available_states]
            state_codes = [s.state_code.lower() for s in available_states]
            
            if request.state.lower() not in state_names and request.state.lower() not in state_codes:
                raise HTTPException(
                    status_code=400,
                    detail=f"State '{request.state}' not found. Available states: {[s.state_name for s in available_states[:10]]}..."
                )
        
        # Generate families
        families = gen.generate_families(request.count, request.state)
        
        # Calculate generation time
        end_time = datetime.now()
        generation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return GenerationResponse(
            success=True,
            message=f"Successfully generated {len(families)} families",
            families_generated=len(families),
            generation_time_ms=round(generation_time_ms, 2),
            families=families
        )
        
    except Exception as e:
        logger.error(f"Family generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/generate", response_model=GenerationResponse)
async def generate_families_get(
    count: int = Query(default=5, ge=1, le=100, description="Number of families to generate"),
    state: Optional[str] = Query(default=None, description="Target state name or code"),
    gen: EnhancedFamilyGenerator = Depends(get_generator)
):
    """Generate synthetic families (GET version for easy testing)"""
    
    request = GenerationRequest(count=count, state=state)
    return await generate_families(request, gen)

@app.get("/states", response_model=List[StateInfo])
async def get_states(gen: EnhancedFamilyGenerator = Depends(get_generator)):
    """Get list of available states"""
    
    try:
        return gen.get_available_states()
    except Exception as e:
        logger.error(f"Failed to get states: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get states: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats(gen: EnhancedFamilyGenerator = Depends(get_generator)):
    """Get generator statistics"""
    
    try:
        stats = gen.get_stats()
        return StatsResponse(
            total_states=stats['total_states'],
            total_occupations=stats['total_occupations'],
            total_education_levels=stats['total_education_levels'],
            database_status=stats['database_status'],
            cache_loaded=stats['cache_loaded']
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "family_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )