#!/usr/bin/env python3
"""
Comprehensive Fix for Enhanced Family Generator

This is a corrected version that handles all the data type issues and provides
better error handling. This version should work even with incomplete data.

Usage:
    python comprehensive_fix.py --count 5
"""

import psycopg2
import random
import json
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
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

class FixedFamilyGenerator:
    """Fixed family generator with robust error handling"""
    
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
        """Load cache with comprehensive error handling"""
        logger.info("Loading data cache safely...")
        
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
                'national_education_occupation_probs': {}
            }
            
            # Load each data type safely
            self._load_states_safely()
            self._load_education_levels_safely()
            self._load_race_data_safely()
            self._load_family_structures_safely()
            self._load_occupation_data_safely()
            
            logger.info("✓ Cache loaded successfully")
            
        except Exception as e:
            logger.error(f"Cache loading failed: {e}")
            # Create minimal fallback data
            self._create_fallback_cache()
    
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
        """Load occupation data with error handling"""
        try:
            # Try to load occupation details
            self.cursor.execute("""
                SELECT DISTINCT occ_code, occ_title
                FROM oews.employment_wages
                WHERE occ_code IS NOT NULL AND occ_title IS NOT NULL
                LIMIT 100
            """)
            
            for row in self.cursor.fetchall():
                occ_code, occ_title = row
                self._cache['occupations'][occ_code] = {
                    'title': occ_title,
                    'major_group': self._get_major_group_from_code(occ_code),
                    'education_level': self._estimate_education_requirement(occ_title)
                }
            
            # Try to load wage data
            self.cursor.execute("""
                SELECT state_code, occ_code, a_mean, a_median
                FROM oews.employment_wages
                WHERE state_code IS NOT NULL AND a_mean IS NOT NULL
                LIMIT 1000
            """)
            
            for row in self.cursor.fetchall():
                state_code, occ_code, a_mean, a_median = row
                
                if state_code not in self._cache['state_occupation_wages']:
                    self._cache['state_occupation_wages'][state_code] = {}
                
                self._cache['state_occupation_wages'][state_code][occ_code] = {
                    'annual_mean': float(a_mean) if a_mean else 50000,
                    'annual_median': float(a_median) if a_median else 45000
                }
            
            logger.info(f"✓ Loaded {len(self._cache['occupations'])} occupations")
            
        except Exception as e:
            logger.warning(f"Failed to load occupation data: {e}")
            # Create fallback occupations
            self._create_fallback_occupations()
    
    def _create_fallback_occupations(self):
        """Create fallback occupation data"""
        fallback_occupations = {
            '11-1011': {'title': 'Chief Executives', 'major_group': 'Management', 'education_level': 'graduate'},
            '15-1211': {'title': 'Computer Systems Analysts', 'major_group': 'Computer', 'education_level': 'bachelors'},
            '25-2021': {'title': 'Elementary School Teachers', 'major_group': 'Education', 'education_level': 'bachelors'},
            '43-4051': {'title': 'Customer Service Representatives', 'major_group': 'Office Support', 'education_level': 'high_school'},
            '35-3021': {'title': 'Combined Food Preparation Workers', 'major_group': 'Food Service', 'education_level': 'less_than_hs'},
            '53-3032': {'title': 'Heavy Truck Drivers', 'major_group': 'Transportation', 'education_level': 'high_school'},
            '31-1014': {'title': 'Nursing Assistants', 'major_group': 'Healthcare Support', 'education_level': 'some_college'},
            '41-2031': {'title': 'Retail Salespersons', 'major_group': 'Sales', 'education_level': 'high_school'}
        }
        
        self._cache['occupations'] = fallback_occupations
        
        # Create fallback wage data
        for state_code in self._cache['states'].keys():
            self._cache['state_occupation_wages'][state_code] = {}
            for occ_code, occ_data in fallback_occupations.items():
                # Estimate wages based on education level
                base_wage = {
                    'less_than_hs': 30000,
                    'high_school': 35000,
                    'some_college': 45000,
                    'bachelors': 65000,
                    'graduate': 85000
                }.get(occ_data['education_level'], 40000)
                
                self._cache['state_occupation_wages'][state_code][occ_code] = {
                    'annual_mean': base_wage,
                    'annual_median': int(base_wage * 0.9)
                }
    
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
            'national_education_occupation_probs': {}
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
    
    def _select_education_level(self, age: int, state_code: str, race_key: str) -> str:
        """Select education level based on age and demographics"""
        
        # Age-based education logic
        if age < 18:
            return 'less_than_hs'  # Still in school
        elif 18 <= age <= 24:
            # Young adults - mix of high school, some college, early bachelors
            return random.choices(
                ['high_school', 'some_college', 'bachelors'],
                weights=[40, 45, 15]
            )[0]
        elif age >= 65:
            # Older adults - reflect historical education patterns
            return random.choices(
                ['less_than_hs', 'high_school', 'some_college', 'bachelors', 'graduate'],
                weights=[15, 40, 20, 20, 5]
            )[0]
        
        # Adults 25-64: Use general education distribution
        return random.choices(
            ['less_than_hs', 'high_school', 'some_college', 'bachelors', 'graduate'],
            weights=[10, 30, 30, 20, 10]
        )[0]
    
    def _select_occupation(self, education_level: str, age: int, state_code: str) -> Optional[Dict[str, Any]]:
        """Select occupation based on education level"""
        
        # Students and very young people typically don't have professional occupations
        if age < 16:
            return None
        elif 16 <= age <= 18:
            # Teenage jobs
            return {
                'occ_code': '35-3021',
                'title': 'Combined Food Preparation and Serving Workers',
                'major_group': 'Food Service',
                'education_requirement': 'less_than_hs',
                'annual_income': random.randint(15000, 25000),
                'employment_type': 'Part-time'
            }
        
        # Filter occupations by education level
        suitable_occupations = []
        for occ_code, occ_data in self._cache['occupations'].items():
            if occ_data.get('education_level') == education_level:
                suitable_occupations.append(occ_code)
        
        if not suitable_occupations:
            # Fallback to any occupation
            suitable_occupations = list(self._cache['occupations'].keys())
        
        if not suitable_occupations:
            return None
        
        # Select random occupation
        occ_code = random.choice(suitable_occupations)
        return self._get_occupation_details(occ_code, state_code)
    
    def _get_occupation_details(self, occ_code: str, state_code: str) -> Dict[str, Any]:
        """Get complete occupation details including wages"""
        
        occ_details = self._cache['occupations'].get(occ_code, {})
        wage_data = self._cache['state_occupation_wages'].get(state_code, {}).get(occ_code, {})
        
        # Calculate income
        if wage_data and wage_data.get('annual_mean'):
            base_income = wage_data['annual_mean']
        else:
            # Fallback income based on education
            education_level = occ_details.get('education_level', 'high_school')
            base_income = {
                'less_than_hs': 30000,
                'high_school': 35000,
                'some_college': 45000,
                'bachelors': 65000,
                'graduate': 85000
            }.get(education_level, 40000)
        
        # Add variation
        annual_income = int(base_income * random.uniform(0.8, 1.2))
        
        return {
            'occ_code': occ_code,
            'title': occ_details.get('title', 'Unknown Occupation'),
            'major_group': occ_details.get('major_group', 'Other'),
            'education_requirement': occ_details.get('education_level', 'high_school'),
            'annual_income': annual_income,
            'employment_type': 'Full-time'
        }
    
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
        """Determine employment status"""
        
        if age < 16:
            return "Child (Under 16)"
        elif age >= 67:
            return "Retired" if random.random() < 0.8 else "Employed"
        elif 16 <= age <= 18:
            return "Student" if random.random() < 0.7 else "Student (Employed)"
        elif occupation:
            return "Employed"
        else:
            # Use general employment rates
            if random.random() < 0.65:
                return "Employed"
            elif random.random() < 0.1:
                return "Unemployed"
            else:
                return "Not in Labor Force"
    
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
                "data_version": "fixed_v1",
                "members": members
            }
            
        except Exception as e:
            logger.error(f"Family generation failed: {e}")
            raise
    
    def generate_families(self, count: int = 5, target_state: str = None) -> List[Dict[str, Any]]:
        """Generate multiple families"""
        families = []
        
        logger.info(f"Generating {count} families...")
        
        for i in range(count):
            try:
                family = self.generate_family(target_state)
                families.append(family)
                
                if (i + 1) % 5 == 0 or i == count - 1:
                    logger.info(f"  Generated {i + 1}/{count} families")
                    
            except Exception as e:
                logger.warning(f"Failed to generate family {i + 1}: {e}")
                continue
        
        return families
    
    def print_family_summary(self, families: List[Dict]):
        """Print summary of generated families"""
        if not families:
            logger.info("No families to summarize")
            return
        
        logger.info(f"\n=== FAMILY GENERATION SUMMARY ===")
        logger.info(f"Total families generated: {len(families)}")
        
        # Income statistics
        incomes = [f['total_household_income'] for f in families]
        if incomes:
            avg_income = sum(incomes) / len(incomes)
            logger.info(f"Average household income: ${avg_income:,.0f}")
        
        # Show sample families
        logger.info(f"\nSample families:")
        for i, family in enumerate(families[:3]):
            logger.info(f"\nFamily {i + 1}:")
            logger.info(f"  Location: {family['state_name']}")
            logger.info(f"  Type: {family['family_type']}")
            logger.info(f"  Size: {family['family_size']} members")
            logger.info(f"  Income: ${family['total_household_income']:,}")
            
            for member in family['members']:
                occupation = member.get('occupation', 'No occupation')
                income = member.get('annual_income', 0)
                income_str = f", ${income:,}" if income > 0 else ""
                logger.info(f"    {member['role']}: Age {member['age']}, {member['education_level']}, {occupation}{income_str}")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

def main():
    """Main execution function"""
    
    # Parse command line arguments
    count = 5
    target_state = None
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--count' and i + 1 < len(sys.argv):
            count = int(sys.argv[i + 1])
            i += 2
        elif arg == '--state' and i + 1 < len(sys.argv):
            target_state = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    try:
        logger.info("🏠 FIXED FAMILY GENERATOR")
        logger.info("="*40)
        
        # Initialize generator
        generator = FixedFamilyGenerator()
        
        # Generate families
        families = generator.generate_families(count, target_state)
        
        # Print results
        generator.print_family_summary(families)
        
        logger.info(f"\n✅ Successfully generated {len(families)} families!")
        
        generator.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
