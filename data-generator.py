#!/usr/bin/env python3
"""
Enhanced Family Generator with Real Database Integration + Tax Characteristics

This version integrates the rich database enhancements from generator_changes.py
to use actual employment rates, education-occupation probabilities, and real wage data,
PLUS adds comprehensive tax-specific characteristics including filing status, deductions,
and tax-relevant events.

Usage:
    python data-generator.py --count 5
    python data-generator.py --count 10 --state California
"""

import psycopg2
import random
import json
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFamilyGenerator:
    """Enhanced family generator with realistic income calculations, database integration, and tax characteristics"""
    
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
        
        # Initialize tax-related data
        self._init_tax_data()
    
    def _init_tax_data(self):
        """Initialize tax-related constants and probabilities"""
        # Tax brackets and thresholds for 2023 tax year
        self.tax_year = 2023
        
        # Income thresholds that affect tax behavior
        self.income_thresholds = {
            'low_income': 30000,
            'middle_income': 75000, 
            'high_income': 200000,
            'very_high_income': 500000
        }
        
        # Standard deduction amounts for 2023
        self.standard_deductions = {
            'single': 13850,
            'married_filing_jointly': 27700,
            'married_filing_separately': 13850,
            'head_of_household': 20800
        }
        
        # Deduction patterns by income level
        self.deduction_patterns = {
            'low_income': {
                'itemize_probability': 0.15,
                'avg_charitable_percent': 0.03,
                'avg_state_local_tax_percent': 0.08,
                'avg_mortgage_interest_percent': 0.12,
                'avg_medical_percent': 0.05
            },
            'middle_income': {
                'itemize_probability': 0.35,
                'avg_charitable_percent': 0.025,
                'avg_state_local_tax_percent': 0.09,
                'avg_mortgage_interest_percent': 0.15,
                'avg_medical_percent': 0.03
            },
            'high_income': {
                'itemize_probability': 0.65,
                'avg_charitable_percent': 0.04,
                'avg_state_local_tax_percent': 0.10,
                'avg_mortgage_interest_percent': 0.10,
                'avg_medical_percent': 0.02
            },
            'very_high_income': {
                'itemize_probability': 0.85,
                'avg_charitable_percent': 0.06,
                'avg_state_local_tax_percent': 0.05,
                'avg_mortgage_interest_percent': 0.08,
                'avg_medical_percent': 0.02
            }
        }
        
        # Tax event probabilities
        self.tax_event_probabilities = {
            'job_change': 0.15,
            'major_medical_expenses': 0.08,
            'home_purchase': 0.06,
            'home_sale': 0.04,
            'investment_activity': 0.25,
            'education_expenses': 0.12,
            'business_income': 0.08,
            'retirement_contribution': 0.45,
            'dependent_care_expenses': 0.20,
        }
    
    def _load_cache_safely(self):
        """Enhanced cache loading with real demographic data"""
        logger.info("Loading comprehensive data cache...")
        
        try:
            # Initialize cache structure
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
            
            # Load comprehensive data
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
            self._cache['state_employment_rates'] = {}

    def _load_education_occupation_probabilities(self):
        """Load education-occupation probability matrix from database"""
        try:
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
                WHERE eop.probability > 0.001
                ORDER BY eop.state_code, el.level_key, eop.probability DESC
            """)
            
            self._cache['education_occupation_probs'] = {}
            for row in self.cursor.fetchall():
                education_key, occ_code, state_code, probability, emp_share, occ_title, major_group = row
                
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
                WHERE eop.state_code IS NULL
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
            self.cursor.execute("""
                SELECT DISTINCT 
                    occ_code, 
                    occ_title, 
                    a_median,
                    a_mean,
                    tot_emp,
                    state_code
                FROM oews.employment_wages
                WHERE occ_code IS NOT NULL 
                  AND occ_title IS NOT NULL 
                  AND a_median IS NOT NULL 
                  AND a_median > 15000
                  AND a_median < 500000
                  AND tot_emp IS NOT NULL 
                  AND tot_emp > 1000
                ORDER BY tot_emp DESC
            """)
            
            occupations = {}
            state_wages = {}
            total_employment = 0
            
            for row in self.cursor.fetchall():
                occ_code, occ_title, a_median, a_mean, tot_emp, state_code = row
                
                if occ_code not in occupations:
                    occupations[occ_code] = {
                        'title': occ_title,
                        'major_group': self._get_major_group_from_code(occ_code),
                        'education_level': self._estimate_education_requirement(occ_title),
                        'median_wage': float(a_median),
                        'mean_wage': float(a_mean) if a_mean else float(a_median),
                        'total_employment': int(tot_emp),
                        'employment_weight': 0
                    }
                    total_employment += int(tot_emp)
                
                if state_code not in state_wages:
                    state_wages[state_code] = {}
                
                state_wages[state_code][occ_code] = {
                    'annual_median': float(a_median),
                    'annual_mean': float(a_mean) if a_mean else float(a_median)
                }
            
            if total_employment > 0:
                for occ_code, occ_data in occupations.items():
                    occ_data['employment_weight'] = occ_data['total_employment'] / total_employment
            
            if occupations:
                self._cache['occupations'] = occupations
                self._cache['state_occupation_wages'] = state_wages
                logger.info(f"✅ Loaded {len(occupations)} occupations from database")
            else:
                logger.warning("❌ No occupations loaded from database")
                self._create_fallback_occupations()
                
        except Exception as e:
            logger.error(f"Failed to load occupation data: {e}")
            self._create_fallback_occupations()
    
    def _create_fallback_cache(self):
        """Create minimal fallback cache if loading fails"""
        logger.warning("Creating fallback cache with minimal data")
        
        self._cache = {
            'states': {
                '06': {'name': 'California', 'region': 'West', 'population': 39000000, 'weight': 12.0},
                '48': {'name': 'Texas', 'region': 'South', 'population': 29000000, 'weight': 9.0},
                '12': {'name': 'Florida', 'region': 'South', 'population': 22000000, 'weight': 7.0}
            },
            'state_weights': {'06': 12.0, '48': 9.0, '12': 7.0},
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
        
        for state_code in self._cache['states'].keys():
            self._cache['state_race_data'][state_code] = {
                'WHITE_NON_HISPANIC': {'name': 'White Non-Hispanic', 'percent': 60.0},
                'HISPANIC': {'name': 'Hispanic or Latino', 'percent': 20.0},
                'BLACK': {'name': 'Black or African American', 'percent': 12.0},
                'ASIAN': {'name': 'Asian', 'percent': 8.0}
            }
            
            self._cache['state_family_structures'][state_code] = {
                'MARRIED_COUPLE': {'name': 'Married Couple Family', 'percent': 50.0},
                'SINGLE_PERSON': {'name': 'Single Person Household', 'percent': 30.0},
                'SINGLE_PARENT_FEMALE': {'name': 'Single Mother Household', 'percent': 15.0},
                'SINGLE_PARENT_MALE': {'name': 'Single Father Household', 'percent': 5.0}
            }
    
    def _create_fallback_occupations(self):
        """Create fallback occupations with realistic wages"""
        realistic_occupations = {
            '41-2031': {
                'title': 'Retail Salespersons', 'major_group': 'Sales', 'education_level': 'high_school', 
                'median_wage': 31000, 'total_employment': 3200000, 'employment_weight': 0.08
            },
            '35-3021': {
                'title': 'Combined Food Preparation Workers', 'major_group': 'Food Service', 'education_level': 'less_than_hs',
                'median_wage': 26000, 'total_employment': 3100000, 'employment_weight': 0.078
            },
            '29-1141': {
                'title': 'Registered Nurses', 'major_group': 'Healthcare', 'education_level': 'bachelors',
                'median_wage': 75000, 'total_employment': 3200000, 'employment_weight': 0.08
            },
            '15-1211': {
                'title': 'Computer Systems Analysts', 'major_group': 'Computer', 'education_level': 'bachelors',
                'median_wage': 95000, 'total_employment': 600000, 'employment_weight': 0.015
            }
        }
        
        self._cache['occupations'] = realistic_occupations
        
        for state_code in self._cache['states'].keys():
            self._cache['state_occupation_wages'][state_code] = {}
            for occ_code, occ_data in realistic_occupations.items():
                base_salary = occ_data['median_wage']
                state_variation = random.uniform(0.9, 1.1)
                
                self._cache['state_occupation_wages'][state_code][occ_code] = {
                    'annual_median': int(base_salary * state_variation),
                    'annual_mean': int(base_salary * state_variation * 1.1)
                }
        
        logger.info(f"✓ Created realistic fallback occupations")
    
    def _get_major_group_from_code(self, occ_code: str) -> str:
        """Get major group from occupation code"""
        if not occ_code or len(occ_code) < 2:
            return 'Other'
        
        major_groups = {
            '11': 'Management', '13': 'Business and Financial', '15': 'Computer and Mathematical',
            '17': 'Architecture and Engineering', '19': 'Life, Physical, and Social Science',
            '21': 'Community and Social Service', '23': 'Legal', '25': 'Educational',
            '27': 'Arts and Entertainment', '29': 'Healthcare Practitioners', '31': 'Healthcare Support',
            '33': 'Protective Service', '35': 'Food Service', '37': 'Building and Maintenance',
            '39': 'Personal Care', '41': 'Sales', '43': 'Office Support', '45': 'Farming and Forestry',
            '47': 'Construction', '49': 'Installation and Repair', '51': 'Production', '53': 'Transportation'
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
    
    def _weighted_random_selection(self, items: Dict[str, Any], weight_key: str = 'percent') -> str:
        """Select item based on weighted probabilities"""
        if not items:
            return None
        
        total_weight = 0
        cumulative_weights = []
        item_keys = list(items.keys())
        
        for key in item_keys:
            if isinstance(items[key], dict):
                weight = items[key].get(weight_key, items[key].get('weight', 1))
            elif isinstance(items[key], (int, float)):
                weight = items[key]
            else:
                weight = 1
            
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
            for state_code, state_data in self._cache['states'].items():
                if state_data['name'].upper() == target_state_upper:
                    return state_code
            raise ValueError(f"State '{target_state}' not found")
        
        return self._weighted_random_selection(self._cache['state_weights'], 'weight')
    
    def _select_race_ethnicity(self, state_code: str) -> Dict[str, str]:
        """Select race/ethnicity based on state demographics"""
        state_races = self._cache['state_race_data'].get(state_code, {})
        
        if not state_races:
            race_key = 'WHITE_NON_HISPANIC'
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
    
    def _select_education_level(self, age: int, state_code: str, race_key: str) -> str:
        """Select education level using real state demographic data"""
        
        if age < 18:
            return 'less_than_hs'
        
        state_edu_dist = self._cache['state_education_distributions'].get(state_code, {})
        
        if state_edu_dist:
            education_weights = []
            education_levels = []
            
            for edu_level, percentage in state_edu_dist.items():
                education_levels.append(edu_level)
                education_weights.append(percentage)
            
            if education_weights and sum(education_weights) > 0:
                if 18 <= age <= 24:
                    adjusted_weights = []
                    for i, edu_level in enumerate(education_levels):
                        weight = education_weights[i]
                        if edu_level == 'some_college':
                            weight *= 1.5
                        elif edu_level == 'graduate':
                            weight *= 0.3
                        elif edu_level == 'bachelors':
                            weight *= 0.7
                        adjusted_weights.append(weight)
                    education_weights = adjusted_weights
                
                elif age >= 65:
                    adjusted_weights = []
                    for i, edu_level in enumerate(education_levels):
                        weight = education_weights[i]
                        if edu_level in ['less_than_hs', 'high_school']:
                            weight *= 1.3
                        elif edu_level in ['bachelors', 'graduate']:
                            weight *= 0.7
                        adjusted_weights.append(weight)
                    education_weights = adjusted_weights
                
                return random.choices(education_levels, weights=education_weights)[0]
        
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
    
    def _select_occupation(self, education_level: str, age: int, state_code: str) -> Optional[Dict[str, Any]]:
        """Select occupation using real education-occupation probability matrix"""
        
        if age < 16:
            return None
        elif 16 <= age <= 18:
            teen_suitable_groups = ['Food Preparation', 'Sales', 'Personal Care']
            return self._select_teen_occupation(education_level, state_code, teen_suitable_groups)
        
        state_probs = self._cache['education_occupation_probs'].get(state_code, {})
        education_probs = state_probs.get(education_level, [])
        
        if not education_probs:
            education_probs = self._cache['national_education_occupation_probs'].get(education_level, [])
        
        if not education_probs:
            return None
        
        employment_rate = self._get_real_employment_rate(education_level, state_code)
        
        if random.random() > employment_rate:
            return self._generate_unemployment_benefits(age, education_level, state_code)
        
        total_prob = sum(occ['probability'] for occ in education_probs)
        if total_prob <= 0:
            return None
        
        rand_val = random.uniform(0, total_prob)
        cumulative_prob = 0
        
        for occ_data in education_probs:
            cumulative_prob += occ_data['probability']
            if rand_val <= cumulative_prob:
                return self._get_occupation_with_real_wages(occ_data['occ_code'], state_code)
        
        return self._get_occupation_with_real_wages(education_probs[0]['occ_code'], state_code)

    def _select_teen_occupation(self, education_level: str, state_code: str, suitable_groups: List[str]) -> Optional[Dict[str, Any]]:
        """Helper method for teen occupation selection"""
        teen_jobs = []
        for occ_code, occ_data in self._cache['occupations'].items():
            if occ_data.get('major_group') in suitable_groups:
                teen_jobs.append(occ_code)
        
        if teen_jobs and random.random() < 0.35:
            occ_code = random.choice(teen_jobs)
            return self._get_occupation_with_real_wages(occ_code, state_code, part_time=True)
        return None

    def _get_real_employment_rate(self, education_level: str, state_code: str) -> float:
        """Get real employment rate for education level and state"""
        
        state_emp_data = self._cache['state_employment_rates'].get(state_code, {})
        base_employment_rate = state_emp_data.get('employment_rate', 85.0) / 100
        
        education_multipliers = {
            'less_than_hs': 0.85, 'high_school': 0.95, 'some_college': 1.02,
            'bachelors': 1.08, 'graduate': 1.12
        }
        
        multiplier = education_multipliers.get(education_level, 1.0)
        adjusted_rate = min(0.98, base_employment_rate * multiplier)
        
        return adjusted_rate

    def _get_occupation_with_real_wages(self, occ_code: str, state_code: str, part_time: bool = False) -> Dict[str, Any]:
        """Get occupation with real wages from OEWS database"""
        
        try:
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
                
                base_wage = float(a_median) if a_median else float(a_mean) if a_mean else 35000
                
                if part_time:
                    base_wage = int(base_wage * 0.4)
                
                annual_income = int(base_wage * random.uniform(0.9, 1.1))
                
                return {
                    'occ_code': occ_code,
                    'title': occ_title or 'Unknown Occupation',
                    'major_group': major_group or 'Other',
                    'education_requirement': edu_key or 'high_school',
                    'annual_income': max(annual_income, 15000),
                    'employment_type': 'Part-time' if part_time else 'Full-time',
                    'data_source': 'database'
                }
        
        except Exception as e:
            logger.debug(f"Error getting real wages for {occ_code}: {e}")
        
        return self._get_occupation_details_realistic(occ_code, state_code, part_time)

    def _generate_unemployment_benefits(self, age: int, education_level: str, state_code: str) -> Optional[Dict[str, Any]]:
        """Generate realistic unemployment benefits based on state and education"""
        
        if age < 18:
            return None
        elif age >= 67:
            monthly_ss = random.randint(800, 2800)
            return {
                'occ_code': 'RETIRED',
                'title': 'Retired (Social Security)',
                'major_group': 'Retired',
                'education_requirement': education_level,
                'annual_income': monthly_ss * 12,
                'employment_type': 'Retired',
                'data_source': 'calculated'
            }
        
        state_emp_data = self._cache['state_employment_rates'].get(state_code, {})
        unemployment_rate = state_emp_data.get('unemployment_rate', 5.0)
        
        benefit_probability = min(0.8, unemployment_rate / 10 + 0.4)
        
        if random.random() < benefit_probability:
            base_weekly_benefit = {
                'less_than_hs': 200, 'high_school': 280, 'some_college': 350,
                'bachelors': 450, 'graduate': 520
            }.get(education_level, 280)
            
            state_variation = random.uniform(0.8, 1.2)
            weekly_benefit = int(base_weekly_benefit * state_variation)
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

    def _get_occupation_details_realistic(self, occ_code: str, state_code: str, part_time: bool = False) -> Dict[str, Any]:
        """Fallback method for getting occupation details using cache"""
        
        occ_details = self._cache['occupations'].get(occ_code, {})
        state_wages = self._cache['state_occupation_wages'].get(state_code, {}).get(occ_code, {})
        
        if state_wages and state_wages.get('annual_median'):
            base_income = state_wages['annual_median']
        elif occ_details.get('median_wage'):
            base_income = occ_details['median_wage']
        else:
            base_income = 35000
        
        if part_time:
            base_income = int(base_income * 0.4)
        
        annual_income = int(base_income * random.uniform(0.88, 1.12))
        
        return {
            'occ_code': occ_code,
            'title': occ_details.get('title', 'Unknown Occupation'),
            'major_group': occ_details.get('major_group', 'Other'),
            'education_requirement': occ_details.get('education_level', 'high_school'),
            'annual_income': max(annual_income, 15000),
            'employment_type': 'Part-time' if part_time else 'Full-time'
        }
    
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
        if random.random() < 0.7:
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

    # TAX CHARACTERISTICS METHODS
    def _generate_tax_characteristics(self, family: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive tax characteristics for a single family"""
        
        # 1. Determine appropriate filing status
        filing_status = self._determine_filing_status(family)
        
        # 2. Generate deduction scenario
        deduction_scenario = self._generate_deduction_scenario(family, filing_status)
        
        # 3. Add tax-relevant events
        tax_events = self._generate_tax_events(family)
        
        # 4. Calculate estimated tax liability
        tax_calculation = self._estimate_tax_liability(family, filing_status, deduction_scenario)
        
        return {
            'tax_year': self.tax_year,
            'filing_status': filing_status,
            'deduction_scenario': deduction_scenario,
            'tax_events': tax_events,
            'estimated_tax_liability': tax_calculation,
            'generation_timestamp': datetime.now().isoformat()
        }
    
    def _determine_filing_status(self, family: Dict[str, Any]) -> Dict[str, Any]:
        """Determine realistic filing status based on family structure and income"""
        
        family_type = family.get('family_type', '')
        total_income = family.get('total_household_income', 0)
        members = family.get('members', [])
        
        adults = [m for m in members if m.get('age', 0) >= 18]
        dependents = [m for m in members if m.get('age', 0) < 18]
        
        if 'Married Couple' in family_type:
            if total_income > self.income_thresholds['very_high_income']:
                files_separately = random.random() < 0.15
            elif total_income > self.income_thresholds['high_income']:
                files_separately = random.random() < 0.08
            else:
                files_separately = random.random() < 0.02
            
            if files_separately:
                spouse_1_income, spouse_2_income = self._split_household_income(members)
                return {
                    'status': 'married_filing_separately',
                    'status_name': 'Married Filing Separately',
                    'spouse_1_income': spouse_1_income,
                    'spouse_2_income': spouse_2_income,
                    'reason': 'Tax optimization strategy'
                }
            else:
                return {
                    'status': 'married_filing_jointly',
                    'status_name': 'Married Filing Jointly',
                    'joint_income': total_income
                }
        
        elif 'Single Mother' in family_type or 'Single Father' in family_type:
            return {
                'status': 'head_of_household',
                'status_name': 'Head of Household',
                'dependents': len(dependents),
                'qualifying_children': len([d for d in dependents if d.get('age', 0) < 17])
            }
        
        else:
            if dependents:
                return {
                    'status': 'head_of_household', 
                    'status_name': 'Head of Household',
                    'dependents': len(dependents)
                }
            else:
                return {
                    'status': 'single',
                    'status_name': 'Single'
                }
    
    def _split_household_income(self, members: List[Dict]) -> tuple:
        """Split household income between spouses for separate filing"""
        adults = [m for m in members if m.get('age', 0) >= 18 and m.get('annual_income', 0) > 0]
        
        if len(adults) >= 2:
            spouse_1_income = adults[0].get('annual_income', 0)
            spouse_2_income = adults[1].get('annual_income', 0)
        elif len(adults) == 1:
            spouse_1_income = adults[0].get('annual_income', 0)
            spouse_2_income = 0
        else:
            spouse_1_income = 0
            spouse_2_income = 0
        
        return spouse_1_income, spouse_2_income
    
    def _generate_deduction_scenario(self, family: Dict[str, Any], filing_status: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic deduction scenario"""
        
        total_income = family.get('total_household_income', 0)
        
        income_bracket = self._get_income_bracket(total_income)
        patterns = self.deduction_patterns[income_bracket]
        
        status = filing_status['status']
        standard_deduction = self.standard_deductions.get(status, 13850)
        
        will_itemize = random.random() < patterns['itemize_probability']
        
        if will_itemize:
            itemized_deductions = self._generate_itemized_deductions(family, total_income, patterns)
            
            total_itemized = sum(itemized_deductions.values())
            if total_itemized > standard_deduction:
                return {
                    'deduction_type': 'itemized',
                    'standard_deduction_amount': standard_deduction,
                    'itemized_deductions': itemized_deductions,
                    'total_deductions': total_itemized,
                    'tax_benefit': total_itemized - standard_deduction
                }
        
        return {
            'deduction_type': 'standard',
            'standard_deduction_amount': standard_deduction,
            'total_deductions': standard_deduction,
            'itemize_considered': will_itemize,
            'reason_for_standard': 'Standard deduction is higher' if will_itemize else 'Standard deduction chosen'
        }
    
    def _get_income_bracket(self, income: int) -> str:
        """Determine income bracket for deduction patterns"""
        if income < self.income_thresholds['low_income']:
            return 'low_income'
        elif income < self.income_thresholds['middle_income']:
            return 'middle_income'
        elif income < self.income_thresholds['high_income']:
            return 'high_income'
        else:
            return 'very_high_income'
    
    def _generate_itemized_deductions(self, family: Dict, income: int, patterns: Dict) -> Dict[str, int]:
        """Generate realistic itemized deductions"""
        
        deductions = {}
        
        # 1. State and Local Tax Deduction (SALT) - capped at $10,000
        salt_percent = patterns['avg_state_local_tax_percent']
        salt_amount = int(income * salt_percent * random.uniform(0.7, 1.3))
        deductions['state_local_taxes'] = min(salt_amount, 10000)
        
        # 2. Mortgage Interest - if family likely owns home
        home_ownership_prob = self._get_home_ownership_probability(income, family)
        if random.random() < home_ownership_prob:
            mortgage_percent = patterns['avg_mortgage_interest_percent']
            mortgage_interest = int(income * mortgage_percent * random.uniform(0.5, 1.5))
            max_mortgage_interest = min(50000, income * 0.25)
            deductions['mortgage_interest'] = min(mortgage_interest, max_mortgage_interest)
        
        # 3. Charitable Contributions
        charitable_percent = patterns['avg_charitable_percent']
        charitable_amount = int(income * charitable_percent * random.uniform(0.3, 2.0))
        max_charitable = int(income * 0.15)
        deductions['charitable_contributions'] = min(charitable_amount, max_charitable)
        
        # 4. Medical Expenses (only excess over 7.5% AGI threshold)
        medical_percent = patterns['avg_medical_percent']
        total_medical = int(income * medical_percent * random.uniform(0.5, 3.0))
        medical_threshold = int(income * 0.075)
        if total_medical > medical_threshold:
            deductions['medical_expenses'] = total_medical - medical_threshold
        
        return deductions
    
    def _get_home_ownership_probability(self, income: int, family: Dict) -> float:
        """Estimate home ownership probability"""
        
        base_prob = 0.65
        
        if income < 30000:
            income_multiplier = 0.4
        elif income < 50000:
            income_multiplier = 0.6
        elif income < 75000:
            income_multiplier = 0.8
        elif income < 100000:
            income_multiplier = 1.0
        else:
            income_multiplier = 1.2
        
        family_type = family.get('family_type', '')
        if 'Married' in family_type:
            family_multiplier = 1.3
        elif 'Single Person' in family_type:
            family_multiplier = 0.7
        else:
            family_multiplier = 1.0
        
        final_prob = min(0.95, base_prob * income_multiplier * family_multiplier)
        return final_prob
    
    def _generate_tax_events(self, family: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate realistic tax-relevant events"""
        
        events = []
        members = family.get('members', [])
        total_income = family.get('total_household_income', 0)
        
        working_adults = [m for m in members if m.get('annual_income', 0) > 1000 and m.get('age', 0) >= 18]
        children = [m for m in members if m.get('age', 0) < 18]
        
        # Generate various tax events based on probabilities
        for adult in working_adults:
            if random.random() < self.tax_event_probabilities['job_change']:
                events.append(self._generate_job_change_event(adult))
        
        if random.random() < self.tax_event_probabilities['major_medical_expenses']:
            events.append(self._generate_medical_expense_event(family, total_income))
        
        if random.random() < self.tax_event_probabilities['home_purchase']:
            events.append(self._generate_home_purchase_event(family, total_income))
        elif random.random() < self.tax_event_probabilities['home_sale']:
            events.append(self._generate_home_sale_event(family, total_income))
        
        if random.random() < self.tax_event_probabilities['investment_activity']:
            events.append(self._generate_investment_event(total_income))
        
        if children or any(m.get('age', 0) < 25 for m in members):
            if random.random() < self.tax_event_probabilities['education_expenses']:
                events.append(self._generate_education_expense_event(family))
        
        for adult in working_adults:
            if random.random() < self.tax_event_probabilities['business_income']:
                events.append(self._generate_business_income_event(adult))
        
        for adult in working_adults:
            if adult.get('age', 0) >= 21 and random.random() < self.tax_event_probabilities['retirement_contribution']:
                events.append(self._generate_retirement_contribution_event(adult))
        
        young_children = [c for c in children if c.get('age', 0) < 13]
        if young_children and random.random() < self.tax_event_probabilities['dependent_care_expenses']:
            events.append(self._generate_dependent_care_event(family, len(young_children)))
        
        return events
    
    def _generate_job_change_event(self, adult: Dict) -> Dict[str, Any]:
        """Generate job change event"""
        
        old_income = adult.get('annual_income', 0)
        change_type = random.choices(
            ['promotion', 'new_job_same_pay', 'new_job_higher_pay', 'new_job_lower_pay', 'temporary_unemployment'],
            weights=[20, 30, 25, 15, 10]
        )[0]
        
        if change_type == 'promotion':
            income_change = random.randint(int(old_income * 0.1), int(old_income * 0.3))
            new_income = old_income + income_change
        elif change_type == 'new_job_higher_pay':
            income_change = random.randint(int(old_income * 0.05), int(old_income * 0.25))
            new_income = old_income + income_change
        elif change_type == 'new_job_lower_pay':
            income_change = random.randint(int(old_income * 0.05), int(old_income * 0.20))
            new_income = old_income - income_change
        elif change_type == 'temporary_unemployment':
            unemployment_months = random.randint(2, 4)
            lost_income = int(old_income * unemployment_months / 12)
            new_income = old_income - lost_income
        else:
            new_income = old_income
        
        return {
            'event_type': 'job_change',
            'event_date': self._random_date_in_tax_year(),
            'person': adult.get('role', 'Adult'),
            'change_type': change_type,
            'old_annual_income': old_income,
            'new_annual_income': max(0, new_income),
            'income_impact': new_income - old_income
        }
    
    def _generate_medical_expense_event(self, family: Dict, income: int) -> Dict[str, Any]:
        """Generate major medical expense event"""
        
        members = family.get('members', [])
        elderly = [m for m in members if m.get('age', 0) >= 65]
        
        if elderly:
            base_expense = random.randint(15000, 50000)
        else:
            base_expense = random.randint(8000, 25000)
        
        if income > 100000:
            expense_multiplier = random.uniform(0.6, 1.0)
        else:
            expense_multiplier = random.uniform(0.8, 1.4)
        
        total_expense = int(base_expense * expense_multiplier)
        agi_threshold = int(income * 0.075)
        deductible_amount = max(0, total_expense - agi_threshold)
        
        expense_types = [
            'Emergency surgery', 'Cancer treatment', 'Chronic condition management',
            'Dental procedures', 'Mental health treatment', 'Physical therapy'
        ]
        
        return {
            'event_type': 'major_medical_expenses',
            'event_date': self._random_date_in_tax_year(),
            'expense_type': random.choice(expense_types),
            'total_medical_expenses': total_expense,
            'agi_threshold': agi_threshold,
            'potentially_deductible_amount': deductible_amount,
            'tax_benefit_estimate': int(deductible_amount * 0.22)
        }
    
    def _generate_home_purchase_event(self, family: Dict, income: int) -> Dict[str, Any]:
        """Generate home purchase event"""
        
        price_multiplier = random.uniform(2.5, 5.0)
        state_code = family.get('state_code', '06')
        if state_code in ['06', '11', '22', '25', '36']:
            price_multiplier = random.uniform(4.0, 7.0)
        
        home_price = max(150000, int(income * price_multiplier))
        down_payment_percent = random.uniform(0.05, 0.20)
        down_payment = int(home_price * down_payment_percent)
        mortgage_amount = home_price - down_payment
        interest_rate = random.uniform(0.035, 0.075)
        annual_interest = int(mortgage_amount * interest_rate * 0.9)
        
        return {
            'event_type': 'home_purchase',
            'event_date': self._random_date_in_tax_year(),
            'home_price': home_price,
            'down_payment': down_payment,
            'mortgage_amount': mortgage_amount,
            'interest_rate': round(interest_rate * 100, 2),
            'estimated_annual_mortgage_interest': annual_interest,
            'closing_costs': int(home_price * 0.03)
        }
    
    def _generate_home_sale_event(self, family: Dict, income: int) -> Dict[str, Any]:
        """Generate home sale event"""
        
        current_value = random.randint(200000, int(income * 4))
        ownership_years = random.randint(2, 15)
        annual_appreciation = random.uniform(0.02, 0.06)
        original_price = int(current_value / (1 + annual_appreciation) ** ownership_years)
        capital_gain = current_value - original_price
        selling_costs = int(current_value * 0.07)
        
        if ownership_years >= 2:
            exclusion_amount = 250000
            if 'Married' in family.get('family_type', ''):
                exclusion_amount = 500000
            taxable_gain = max(0, capital_gain - exclusion_amount)
        else:
            taxable_gain = capital_gain
        
        return {
            'event_type': 'home_sale',
            'event_date': self._random_date_in_tax_year(),
            'sale_price': current_value,
            'original_purchase_price': original_price,
            'ownership_years': ownership_years,
            'capital_gain': capital_gain,
            'selling_costs': selling_costs,
            'eligible_for_exclusion': ownership_years >= 2,
            'taxable_capital_gain': max(0, taxable_gain),
            'estimated_tax_owed': int(max(0, taxable_gain) * 0.15)
        }
    
    def _generate_investment_event(self, income: int) -> Dict[str, Any]:
        """Generate investment income/loss event"""
        
        if income < 50000:
            investment_range = (500, 5000)
        elif income < 100000:
            investment_range = (1000, 15000) 
        elif income < 200000:
            investment_range = (5000, 50000)
        else:
            investment_range = (10000, 100000)
        
        event_type = random.choices(
            ['capital_gains', 'capital_losses', 'dividend_income', 'interest_income'],
            weights=[35, 25, 25, 15]
        )[0]
        
        amount = random.randint(*investment_range)
        if event_type == 'capital_losses':
            amount = -amount
        
        is_long_term = random.random() < 0.6 if event_type in ['capital_gains', 'capital_losses'] else None
        tax_rate = 0.15 if is_long_term else 0.22
        
        return {
            'event_type': 'investment_activity',
            'investment_type': event_type,
            'amount': amount,
            'is_long_term': is_long_term,
            'estimated_tax_impact': int(abs(amount) * tax_rate) if amount > 0 else int(abs(amount) * 0.22 * -1),
            'event_date': self._random_date_in_tax_year()
        }
    
    def _generate_education_expense_event(self, family: Dict) -> Dict[str, Any]:
        """Generate education expense event"""
        
        members = family.get('members', [])
        students = [m for m in members if 16 <= m.get('age', 0) <= 24]
        
        if students:
            student = random.choice(students)
            age = student.get('age', 18)
            if age >= 18:
                expense_type = 'college_tuition'
                base_cost = random.randint(8000, 25000)
            else:
                expense_type = 'educational_programs'
                base_cost = random.randint(1000, 5000)
            
            books_supplies = random.randint(500, 1500)
            total_expenses = base_cost + books_supplies
            
            if expense_type == 'college_tuition':
                max_credit = 2500 if age <= 24 else 2000
                estimated_credit = min(max_credit, int(total_expenses * 0.4))
            else:
                estimated_credit = 0
            
            return {
                'event_type': 'education_expenses',
                'student_age': age,
                'student_role': student.get('role', 'Child'),
                'expense_type': expense_type,
                'tuition_fees': base_cost,
                'books_supplies': books_supplies,
                'total_expenses': total_expenses,
                'estimated_tax_credit': estimated_credit,
                'event_date': self._random_date_in_tax_year()
            }
        
        return {
            'event_type': 'education_expenses',
            'expense_type': 'family_education',
            'total_expenses': random.randint(500, 3000),
            'estimated_tax_credit': 0,
            'event_date': self._random_date_in_tax_year()
        }
    
    def _generate_business_income_event(self, adult: Dict) -> Dict[str, Any]:
        """Generate business/gig income event"""
        
        business_types = [
            'freelance_consulting', 'ride_sharing', 'food_delivery', 'online_sales',
            'tutoring', 'handyman_services', 'pet_sitting', 'rental_property'
        ]
        
        business_type = random.choice(business_types)
        
        if business_type in ['freelance_consulting', 'rental_property']:
            income_range = (5000, 30000)
        elif business_type in ['ride_sharing', 'online_sales']:
            income_range = (2000, 15000)
        else:
            income_range = (500, 8000)
        
        gross_income = random.randint(*income_range)
        expense_ratio = random.uniform(0.15, 0.50)
        business_expenses = int(gross_income * expense_ratio)
        net_income = gross_income - business_expenses
        se_tax = int(net_income * 0.1413)
        
        return {
            'event_type': 'business_income',
            'business_type': business_type,
            'gross_business_income': gross_income,
            'business_expenses': business_expenses,
            'net_business_income': net_income,
            'estimated_se_tax': se_tax,
            'quarter_1099_issued': random.random() < 0.7,
            'event_date': self._random_date_in_tax_year()
        }
    
    def _generate_retirement_contribution_event(self, adult: Dict) -> Dict[str, Any]:
        """Generate retirement contribution event"""
        
        income = adult.get('annual_income', 30000)
        age = adult.get('age', 35)
        
        if age >= 50:
            contribution_limit_401k = 30000
            contribution_limit_ira = 7500
        else:
            contribution_limit_401k = 22500
            contribution_limit_ira = 6500
        
        contribution_percent = random.uniform(0.03, 0.15)
        desired_contribution = int(income * contribution_percent)
        
        account_types = ['401k', 'traditional_ira', 'roth_ira']
        weights = [60, 25, 15]
        
        if income > 75000:
            weights = [75, 15, 10]
        
        account_type = random.choices(account_types, weights=weights)[0]
        
        if account_type == '401k':
            max_contribution = min(desired_contribution, contribution_limit_401k)
            employer_match_rate = random.uniform(0, 0.06)
            employer_match = min(int(income * employer_match_rate), int(max_contribution * 1.0))
        else:
            max_contribution = min(desired_contribution, contribution_limit_ira)
            employer_match = 0
        
        tax_deduction = max_contribution if account_type != 'roth_ira' else 0
        
        return {
            'event_type': 'retirement_contribution',
            'account_type': account_type,
            'employee_contribution': max_contribution,
            'employer_match': employer_match,
            'total_contribution': max_contribution + employer_match,
            'tax_deductible_amount': tax_deduction,
            'estimated_tax_savings': int(tax_deduction * 0.22),
            'contribution_percent_of_income': round((max_contribution / income) * 100, 1),
            'event_date': self._random_date_in_tax_year()
        }
    
    def _generate_dependent_care_event(self, family: Dict, num_children: int) -> Dict[str, Any]:
        """Generate dependent care expense event"""
        
        per_child_cost = random.randint(8000, 15000)
        total_cost = per_child_cost * num_children
        credit_eligible_expenses = min(total_cost, 6000)
        
        income = family.get('total_household_income', 50000)
        if income < 15000:
            credit_percent = 0.35
        elif income < 43000:
            credit_percent = 0.35 - ((income - 15000) / 28000) * 0.15
        else:
            credit_percent = 0.20
        
        estimated_credit = int(credit_eligible_expenses * credit_percent)
        
        care_types = ['daycare_center', 'family_daycare', 'nanny', 'after_school_program', 'summer_camp']
        
        return {
            'event_type': 'dependent_care_expenses',
            'number_of_children': num_children,
            'care_type': random.choice(care_types),
            'total_expenses': total_cost,
            'credit_eligible_expenses': credit_eligible_expenses,
            'estimated_credit': estimated_credit,
            'monthly_cost': int(total_cost / 12),
            'event_date': self._random_date_in_tax_year()
        }
    
    def _estimate_tax_liability(self, family: Dict, filing_status: Dict, deduction_scenario: Dict) -> Dict[str, Any]:
        """Estimate tax liability for the family"""
        
        income = family.get('total_household_income', 0)
        status = filing_status.get('status', 'single')
        total_deductions = deduction_scenario.get('total_deductions', 0)
        
        taxable_income = max(0, income - total_deductions)
        federal_tax = self._calculate_federal_tax(taxable_income, status)
        state_code = family.get('state_code', '06')
        state_tax = self._estimate_state_tax(taxable_income, state_code)
        fica_tax = min(income * 0.0765, 160200 * 0.062 + income * 0.0145)
        total_tax = federal_tax + state_tax + fica_tax
        effective_rate = (total_tax / income * 100) if income > 0 else 0
        
        return {
            'gross_income': income,
            'total_deductions': total_deductions,
            'taxable_income': taxable_income,
            'federal_income_tax': int(federal_tax),
            'state_income_tax': int(state_tax),
            'fica_tax': int(fica_tax),
            'total_tax_liability': int(total_tax),
            'effective_tax_rate': round(effective_rate, 2),
            'marginal_tax_bracket': self._get_marginal_bracket(taxable_income, status)
        }
    
    def _calculate_federal_tax(self, taxable_income: int, filing_status: str) -> float:
        """Calculate federal income tax using 2023 tax brackets"""
        
        if filing_status in ['single', 'married_filing_separately']:
            brackets = [
                (11000, 0.10), (44725, 0.12), (95375, 0.22), (197050, 0.24),
                (231250, 0.32), (578125, 0.35), (float('inf'), 0.37)
            ]
        elif filing_status == 'married_filing_jointly':
            brackets = [
                (22000, 0.10), (89450, 0.12), (190750, 0.22), (364200, 0.24),
                (462500, 0.32), (693750, 0.35), (float('inf'), 0.37)
            ]
        else:
            brackets = [
                (15700, 0.10), (59850, 0.12), (95350, 0.22), (193350, 0.24),
                (231250, 0.32), (578100, 0.35), (float('inf'), 0.37)
            ]
        
        tax = 0
        prev_bracket = 0
        
        for bracket_limit, rate in brackets:
            if taxable_income <= prev_bracket:
                break
            
            taxable_in_bracket = min(taxable_income - prev_bracket, bracket_limit - prev_bracket)
            tax += taxable_in_bracket * rate
            prev_bracket = bracket_limit
            
            if taxable_income <= bracket_limit:
                break
        
        return tax
    
    def _estimate_state_tax(self, taxable_income: int, state_code: str) -> float:
        """Estimate state income tax"""
        
        no_tax_states = ['02', '12', '32', '53', '48', '50', '82', '03', '47']
        if state_code in no_tax_states:
            return 0
        
        high_tax_states = ['06', '36', '34', '09', '25']
        
        if state_code in high_tax_states:
            if taxable_income < 50000:
                rate = 0.05
            elif taxable_income < 100000:
                rate = 0.07
            else:
                rate = 0.09
        else:
            if taxable_income < 50000:
                rate = 0.03
            elif taxable_income < 100000:
                rate = 0.05
            else:
                rate = 0.06
        
        return taxable_income * rate
    
    def _get_marginal_bracket(self, taxable_income: int, filing_status: str) -> str:
        """Get marginal tax bracket"""
        
        if filing_status in ['single', 'married_filing_separately']:
            if taxable_income <= 11000:
                return "10%"
            elif taxable_income <= 44725:
                return "12%"
            elif taxable_income <= 95375:
                return "22%"
            elif taxable_income <= 197050:
                return "24%"
            elif taxable_income <= 231250:
                return "32%"
            elif taxable_income <= 578125:
                return "35%"
            else:
                return "37%"
        else:
            return "22%"
    
    def _random_date_in_tax_year(self) -> str:
        """Generate random date in tax year"""
        start_date = datetime(self.tax_year, 1, 1)
        end_date = datetime(self.tax_year, 12, 31)
        
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime('%Y-%m-%d')
    
    # MAIN GENERATION METHODS
    def generate_family(self, target_state: str = None, include_tax: bool = True) -> Dict[str, Any]:
        """Generate a complete family with integrated tax characteristics"""
        
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
            
            # Create base family data
            family = {
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
                "data_version": "enhanced_v1_with_tax",
                "members": members
            }
            
            # Add tax characteristics during family generation
            if include_tax:
                try:
                    tax_data = self._generate_tax_characteristics(family)
                    family['tax_characteristics'] = tax_data
                except Exception as e:
                    logger.warning(f"Failed to add tax characteristics to family {family['family_id']}: {e}")
            
            return family
            
        except Exception as e:
            logger.error(f"Family generation failed: {e}")
            raise
    
    def generate_families(self, count: int = 5, target_state: str = None, include_tax: bool = True) -> List[Dict[str, Any]]:
        """Generate multiple families with integrated tax characteristics"""
        families = []
        
        tax_status = "with tax characteristics" if include_tax else "basic demographic data only"
        logger.info(f"🏠 Starting generation of {count} families {tax_status}...")
        if target_state:
            logger.info(f"🎯 Target state: {target_state}")
        
        for i in range(count):
            try:
                family = self.generate_family(target_state, include_tax)
                families.append(family)
                
                if (i + 1) % 1 == 0 or i == count - 1:
                    logger.info(f"  ✅ Generated {i + 1}/{count} families")
                    
            except Exception as e:
                logger.warning(f"❌ Failed to generate family {i + 1}: {e}")
                continue
        
        logger.info(f"🎉 Completed generation: {len(families)} families successfully created")
        return families

    def print_family_summary(self, families: List[Dict]):
        """Print comprehensive summary including tax characteristics"""
        if not families:
            logger.info("No families to summarize")
            return
        
        logger.info(f"\n=== ENHANCED FAMILY GENERATION SUMMARY ===")
        logger.info(f"Total families generated: {len(families)}")
        
        # Check if families have tax characteristics
        has_tax_data = any(f.get('tax_characteristics') for f in families)
        
        # Income statistics
        incomes = [f['total_household_income'] for f in families]
        if incomes:
            avg_income = sum(incomes) / len(incomes)
            median_income = sorted(incomes)[len(incomes) // 2]
            logger.info(f"Average household income: ${avg_income:,.0f}")
            logger.info(f"Median household income: ${median_income:,.0f}")
        
        # Tax statistics (if available)
        if has_tax_data:
            filing_statuses = {}
            deduction_types = {'standard': 0, 'itemized': 0}
            tax_rates = []
            total_events = 0
            
            for family in families:
                tax_chars = family.get('tax_characteristics', {})
                if tax_chars:
                    # Filing status
                    status = tax_chars.get('filing_status', {}).get('status_name', 'Unknown')
                    filing_statuses[status] = filing_statuses.get(status, 0) + 1
                    
                    # Deduction type
                    deduction_type = tax_chars.get('deduction_scenario', {}).get('deduction_type', 'standard')
                    deduction_types[deduction_type] += 1
                    
                    # Tax rates
                    tax_calc = tax_chars.get('estimated_tax_liability', {})
                    if tax_calc.get('effective_tax_rate'):
                        tax_rates.append(tax_calc['effective_tax_rate'])
                    
                    # Count events
                    events = tax_chars.get('tax_events', [])
                    total_events += len(events)
            
            logger.info(f"\n📊 TAX CHARACTERISTICS:")
            if tax_rates:
                avg_tax_rate = sum(tax_rates) / len(tax_rates)
                logger.info(f"Average effective tax rate: {avg_tax_rate:.1f}%")
            
            logger.info(f"Filing status distribution:")
            for status, count in filing_statuses.items():
                pct = (count / len([f for f in families if f.get('tax_characteristics')])) * 100
                logger.info(f"  {status}: {count} ({pct:.1f}%)")
            
            logger.info(f"Deduction type distribution:")
            for deduction_type, count in deduction_types.items():
                pct = (count / len([f for f in families if f.get('tax_characteristics')])) * 100
                logger.info(f"  {deduction_type.title()}: {count} ({pct:.1f}%)")
            
            logger.info(f"Tax events: {total_events} total across all families")
            logger.info(f"Average events per family: {total_events / len(families):.1f}")
        
        # Show detailed family examples
        display_count = min(len(families), 5)
        logger.info(f"\n📋 DETAILED FAMILY EXAMPLES ({display_count} of {len(families)}):")
        
        for i, family in enumerate(families[:display_count]):
            logger.info(f"\n--- Family {i + 1} (ID: {family['family_id']}) ---")
            logger.info(f"  Location: {family['state_name']}")
            logger.info(f"  Type: {family['family_type']}")
            logger.info(f"  Size: {family['family_size']} members")
            logger.info(f"  Household Income: ${family['total_household_income']:,}")
            
            # Show family members
            for member in family['members']:
                occupation = member.get('occupation', 'No occupation')
                income = member.get('annual_income', 0)
                income_str = f", ${income:,}" if income > 0 else ""
                logger.info(f"    {member['role']}: Age {member['age']}, {member['education_level']}, {occupation}{income_str}")
            
            # Show tax characteristics (if available)
            tax_chars = family.get('tax_characteristics', {})
            if tax_chars:
                filing_status = tax_chars.get('filing_status', {})
                deduction_info = tax_chars.get('deduction_scenario', {})
                tax_calc = tax_chars.get('estimated_tax_liability', {})
                events = tax_chars.get('tax_events', [])
                
                logger.info(f"  📋 Tax Profile:")
                logger.info(f"    Filing Status: {filing_status.get('status_name', 'Unknown')}")
                logger.info(f"    Deduction Strategy: {deduction_info.get('deduction_type', 'standard').title()}")
                logger.info(f"    Total Deductions: ${deduction_info.get('total_deductions', 0):,}")
                
                if tax_calc:
                    logger.info(f"    Estimated Tax Liability: ${tax_calc.get('total_tax_liability', 0):,}")
                    logger.info(f"    Effective Tax Rate: {tax_calc.get('effective_tax_rate', 0):.1f}%")
                
                if events:
                    logger.info(f"    Tax Events ({len(events)}):")
                    for event in events[:3]:  # Show first 3 events
                        event_type = event.get('event_type', 'Unknown').replace('_', ' ').title()
                        logger.info(f"      • {event_type}")
                    if len(events) > 3:
                        logger.info(f"      • ... and {len(events) - 3} more events")
        
        if len(families) > 5:
            logger.info(f"\n... and {len(families) - 5} more families")
        
        # Final statistics
        logger.info(f"\n📊 SUMMARY STATISTICS:")
        logger.info(f"Family types: {set(f['family_type'] for f in families)}")
        logger.info(f"Income range: ${min(incomes):,} - ${max(incomes):,}")
        logger.info(f"Average family size: {sum(f['family_size'] for f in families) / len(families):.1f}")
        
        if has_tax_data:
            logger.info(f"✅ Includes comprehensive tax characteristics")
        logger.info(f"✅ Database integration with real demographic data")
        logger.info(f"✅ Realistic income calculations using OEWS wage data")
        logger.info(f"Data version: enhanced_v1_with_tax")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

def main():
    """Main execution function with integrated tax characteristics"""
    
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate realistic synthetic families with enhanced database integration and tax characteristics')
    parser.add_argument('--count', type=int, default=5, 
                       help='Number of families to generate (default: 5)')
    parser.add_argument('--state', type=str, default=None,
                       help='Target state (e.g., California, Texas)')
    parser.add_argument('--no-tax', action='store_true', 
                       help='Skip tax characteristics generation')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (optional)')
    
    args = parser.parse_args()
    count = args.count
    target_state = args.state
    include_tax = not args.no_tax  # Invert the flag
    output_file = args.output
    
    # Debug logging
    tax_status = "with tax characteristics" if include_tax else "basic data only"
    logger.info(f"🎯 Parsed arguments: count={count}, state={target_state}, mode={tax_status}")
    
    try:
        logger.info("🏠 ENHANCED FAMILY GENERATOR v1 WITH INTEGRATED TAX DATA")
        logger.info("="*65)
        
        # Initialize generator and generate families with integrated tax characteristics
        generator = EnhancedFamilyGenerator()
        
        # Generate families (tax characteristics included during generation)
        families = generator.generate_families(count, target_state, include_tax)
        
        # Print comprehensive results (includes tax data in main summary)
        generator.print_family_summary(families)
        
        # Save output (if requested)
        if output_file:
            logger.info(f"\n💾 Saving results to {output_file}...")
            try:
                with open(output_file, 'w') as f:
                    json.dump(families, f, indent=2, default=str)
                logger.info(f"✅ Successfully saved {len(families)} families to {output_file}")
            except Exception as e:
                logger.warning(f"Failed to save to {output_file}: {e}")
        
        # Final summary
        logger.info(f"\n🎉 GENERATION COMPLETE!")
        logger.info(f"Successfully generated {len(families)} families")
        if include_tax:
            logger.info(f"✅ Tax characteristics integrated during family generation")
        else:
            logger.info(f"ℹ️  Basic demographic data only (tax characteristics skipped)")
        
        generator.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)