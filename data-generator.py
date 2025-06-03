#!/usr/bin/env python3
"""
Simplified Enhanced Family Generator

A streamlined version with flexible family generation parameters.

Usage:
    python family_generator.py --count 5
    python family_generator.py --count 10 --state California --family-type married_couple
    python family_generator.py --count 3 --race hispanic --income-range 50000-80000
    python family_generator.py --count 2 --education bachelors --children 2
"""

import psycopg2
import random
import json
import sys
import os
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class FamilyGenerator:
    """Simplified family generator with flexible parameters"""
    
    # Class constants for easy configuration
    FALLBACK_DATA = {
        'states': {
            '06': {'name': 'California', 'region': 'West', 'population': 39000000, 'weight': 12.0},
            '48': {'name': 'Texas', 'region': 'South', 'population': 29000000, 'weight': 9.0},
            '12': {'name': 'Florida', 'region': 'South', 'population': 22000000, 'weight': 7.0},
            '36': {'name': 'New York', 'region': 'Northeast', 'population': 20000000, 'weight': 6.0}
        },
        'races': {
            'WHITE_NON_HISPANIC': {'name': 'White Non-Hispanic', 'percent': 60.0},
            'HISPANIC': {'name': 'Hispanic or Latino', 'percent': 20.0},
            'BLACK': {'name': 'Black or African American', 'percent': 12.0},
            'ASIAN': {'name': 'Asian', 'percent': 8.0}
        },
        'family_structures': {
            'MARRIED_COUPLE': {'name': 'Married Couple Family', 'percent': 50.0},
            'SINGLE_PERSON': {'name': 'Single Person Household', 'percent': 30.0},
            'SINGLE_PARENT_FEMALE': {'name': 'Single Mother Household', 'percent': 15.0},
            'SINGLE_PARENT_MALE': {'name': 'Single Father Household', 'percent': 5.0}
        },
        'education_levels': {
            'less_than_hs': {'name': 'Less than High School', 'sort_order': 1, 'typical_years': 11},
            'high_school': {'name': 'High School Graduate', 'sort_order': 2, 'typical_years': 12},
            'some_college': {'name': 'Some College', 'sort_order': 3, 'typical_years': 14},
            'bachelors': {'name': "Bachelor's Degree", 'sort_order': 4, 'typical_years': 16},
            'graduate': {'name': 'Graduate Degree', 'sort_order': 5, 'typical_years': 18}
        },
        'occupations': {
            '11-1011': {'title': 'Chief Executives', 'group': 'Management', 'education': 'graduate', 'base_salary': 85000},
            '15-1211': {'title': 'Computer Systems Analysts', 'group': 'Computer', 'education': 'bachelors', 'base_salary': 65000},
            '25-2021': {'title': 'Elementary School Teachers', 'group': 'Education', 'education': 'bachelors', 'base_salary': 55000},
            '43-4051': {'title': 'Customer Service Representatives', 'group': 'Office Support', 'education': 'high_school', 'base_salary': 35000},
            '35-3021': {'title': 'Food Preparation Workers', 'group': 'Food Service', 'education': 'less_than_hs', 'base_salary': 25000},
            '53-3032': {'title': 'Heavy Truck Drivers', 'group': 'Transportation', 'education': 'high_school', 'base_salary': 45000},
            '31-1014': {'title': 'Nursing Assistants', 'group': 'Healthcare Support', 'education': 'some_college', 'base_salary': 35000},
            '41-2031': {'title': 'Retail Salespersons', 'group': 'Sales', 'education': 'high_school', 'base_salary': 30000}
        }
    }
    
    def __init__(self):
        self.current_year = datetime.now().year
        self.data = self.FALLBACK_DATA.copy()
        self.conn = None
        self.cursor = None
        self._init_database()
        self._load_real_data()
    
    def _init_database(self):
        """Initialize database connection if available"""
        try:
            # Load environment variables
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            
            connection_string = os.getenv('NEON_CONNECTION_STRING')
            if connection_string:
                self.conn = psycopg2.connect(connection_string)
                self.cursor = self.conn.cursor()
                self.cursor.execute("SELECT 1")
                logger.info("✓ Database connected")
        except Exception as e:
            logger.warning(f"Database not available, using fallback data: {e}")
    
    def _load_real_data(self):
        """Load real data from database if available, otherwise use fallbacks"""
        if not self.cursor:
            logger.info("Using fallback demographic data")
            return
        
        try:
            self._load_states()
            self._load_demographics()
            self._load_occupations()
            logger.info("✓ Real demographic data loaded")
        except Exception as e:
            logger.warning(f"Failed to load real data: {e}, using fallbacks")
    
    def _load_states(self):
        """Load state data from database"""
        try:
            self.cursor.execute("""
                SELECT sd.state_code, sd.state_name, r.region_name, 
                       sd.total_population, sd.population_weight
                FROM census.state_demographics sd
                LEFT JOIN regions r ON sd.region_id = r.id
                WHERE sd.total_population > 0
                ORDER BY sd.population_weight DESC NULLS LAST
                LIMIT 50
            """)
            
            states = {}
            for row in self.cursor.fetchall():
                state_code, state_name, region_name, total_pop, pop_weight = row
                states[state_code] = {
                    'name': state_name,
                    'region': region_name or 'Unknown',
                    'population': total_pop or 0,
                    'weight': float(pop_weight) if pop_weight else 1.0
                }
            
            if states:
                self.data['states'] = states
                
        except Exception as e:
            logger.warning(f"Failed to load states: {e}")
    
    def _load_demographics(self):
        """Load demographic data (race, family structures) from database"""
        try:
            # Load race data
            self.cursor.execute("""
                SELECT re.race_key, re.race_name, AVG(sre.population_percent) as avg_percent
                FROM census.state_race_ethnicity sre
                JOIN race_ethnicity re ON sre.race_id = re.id
                GROUP BY re.race_key, re.race_name
                ORDER BY avg_percent DESC
                LIMIT 10
            """)
            
            races = {}
            for row in self.cursor.fetchall():
                race_key, race_name, avg_percent = row
                races[race_key] = {
                    'name': race_name,
                    'percent': float(avg_percent) if avg_percent else 1.0
                }
            
            if races:
                self.data['races'] = races
                
        except Exception as e:
            logger.warning(f"Failed to load demographics: {e}")
    
    def _load_occupations(self):
        """Load occupation data from database"""
        try:
            self.cursor.execute("""
                SELECT DISTINCT occ_code, occ_title, a_mean
                FROM oews.employment_wages
                WHERE occ_code IS NOT NULL AND occ_title IS NOT NULL 
                  AND a_mean IS NOT NULL AND a_mean > 0
                ORDER BY a_mean DESC
                LIMIT 50
            """)
            
            occupations = {}
            for row in self.cursor.fetchall():
                occ_code, occ_title, a_mean = row
                occupations[occ_code] = {
                    'title': occ_title,
                    'group': self._get_occupation_group(occ_code),
                    'education': self._estimate_education_requirement(occ_title),
                    'base_salary': float(a_mean) if a_mean else 40000
                }
            
            if occupations:
                self.data['occupations'] = occupations
                
        except Exception as e:
            logger.warning(f"Failed to load occupations: {e}")
    
    def _get_occupation_group(self, occ_code: str) -> str:
        """Get occupation group from SOC code"""
        groups = {
            '11': 'Management', '13': 'Business', '15': 'Computer', '17': 'Engineering',
            '19': 'Science', '21': 'Community Service', '23': 'Legal', '25': 'Education',
            '27': 'Arts', '29': 'Healthcare', '31': 'Healthcare Support', '33': 'Protective',
            '35': 'Food Service', '37': 'Maintenance', '39': 'Personal Care', '41': 'Sales',
            '43': 'Office Support', '45': 'Farming', '47': 'Construction', '49': 'Installation',
            '51': 'Production', '53': 'Transportation'
        }
        return groups.get(occ_code[:2], 'Other')
    
    def _estimate_education_requirement(self, occ_title: str) -> str:
        """Estimate education requirement from job title"""
        title_lower = occ_title.lower()
        if any(word in title_lower for word in ['chief', 'director', 'manager', 'executive']):
            return 'graduate'
        elif any(word in title_lower for word in ['engineer', 'analyst', 'teacher', 'nurse']):
            return 'bachelors'
        elif any(word in title_lower for word in ['technician', 'assistant', 'specialist']):
            return 'some_college'
        else:
            return 'high_school'
    
    def _weighted_choice(self, choices: Dict[str, Any], weight_key: str = 'percent') -> str:
        """Make weighted random choice from dictionary"""
        if not choices:
            return None
        
        items = list(choices.items())
        weights = []
        
        for key, value in items:
            if isinstance(value, dict):
                weight = value.get(weight_key, value.get('weight', 1))
            else:
                weight = value
            weights.append(max(0, float(weight)))
        
        if sum(weights) == 0:
            return random.choice(list(choices.keys()))
        
        return random.choices(list(choices.keys()), weights=weights)[0]
    
    def generate_family(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a family with optional parameters"""
        params = params or {}
        
        # Select characteristics
        state_code = self._select_state(params.get('state'))
        race_info = self._select_race(params.get('race'), state_code)
        structure_info = self._select_family_structure(params.get('family_type'))
        
        # Generate members
        members = self._generate_members(structure_info, race_info, state_code, params)
        
        # Apply constraints
        members = self._apply_constraints(members, params)
        
        # Calculate totals
        total_income = sum(m.get('annual_income', 0) for m in members)
        
        return {
            "family_id": f"FAM_{random.randint(100000, 999999)}",
            "state_code": state_code,
            "state_name": self.data['states'][state_code]['name'],
            "region": self.data['states'][state_code]['region'],
            "race": race_info['name'],
            "race_key": race_info['key'],
            "family_type": structure_info['name'],
            "family_size": len(members),
            "total_household_income": total_income,
            "highest_education": self._get_highest_education(members),
            "total_earners": sum(1 for m in members if m.get('annual_income', 0) > 0),
            "generation_date": datetime.now().isoformat(),
            "members": members
        }
    
    def _select_state(self, target_state: str = None) -> str:
        """Select state based on target or population weights"""
        if target_state:
            # Try to find by name or code
            target_upper = target_state.upper()
            for code, data in self.data['states'].items():
                if code == target_upper or data['name'].upper() == target_upper:
                    return code
            raise ValueError(f"State '{target_state}' not found")
        
        weights = {code: data['weight'] for code, data in self.data['states'].items()}
        return self._weighted_choice(weights, 'weight')
    
    def _select_race(self, target_race: str = None, state_code: str = None) -> Dict[str, str]:
        """Select race/ethnicity"""
        if target_race:
            target_upper = target_race.upper()
            for key, data in self.data['races'].items():
                if key == target_upper or target_upper in data['name'].upper():
                    return {'key': key, 'name': data['name']}
            raise ValueError(f"Race '{target_race}' not found")
        
        race_key = self._weighted_choice(self.data['races'])
        return {'key': race_key, 'name': self.data['races'][race_key]['name']}
    
    def _select_family_structure(self, target_type: str = None) -> Dict[str, str]:
        """Select family structure"""
        if target_type:
            target_upper = target_type.upper()
            for key, data in self.data['family_structures'].items():
                if key == target_upper or target_upper in key:
                    return {'key': key, 'name': data['name']}
            raise ValueError(f"Family type '{target_type}' not found")
        
        structure_key = self._weighted_choice(self.data['family_structures'])
        return {'key': structure_key, 'name': self.data['family_structures'][structure_key]['name']}
    
    def _generate_members(self, structure_info: Dict, race_info: Dict, state_code: str, params: Dict) -> List[Dict]:
        """Generate family members based on structure"""
        structure_key = structure_info['key']
        
        if structure_key == 'MARRIED_COUPLE':
            return self._generate_married_couple(race_info, state_code, params)
        elif structure_key == 'SINGLE_PERSON':
            return self._generate_single_person(race_info, state_code, params)
        elif structure_key in ['SINGLE_PARENT_FEMALE', 'SINGLE_PARENT_MALE']:
            return self._generate_single_parent(race_info, state_code, params, structure_key)
        else:
            return self._generate_married_couple(race_info, state_code, params)
    
    def _generate_married_couple(self, race_info: Dict, state_code: str, params: Dict) -> List[Dict]:
        """Generate married couple family"""
        members = []
        
        # Generate head of household
        head = self._generate_person('Head of Household', race_info, state_code, params)
        members.append(head)
        
        # Generate spouse
        spouse_params = params.copy()
        spouse_params['age_range'] = self._get_spouse_age_range(head['age'])
        spouse_params['gender'] = 'Female' if head['gender'] == 'Male' else 'Male'
        spouse = self._generate_person('Spouse', race_info, state_code, spouse_params)
        members.append(spouse)
        
        # Generate children if specified or randomly
        child_count = params.get('children')
        if child_count is None:
            child_count = random.choices([0, 1, 2, 3], weights=[30, 30, 30, 10])[0]
        
        for i in range(int(child_count)):
            child_params = params.copy()
            child_params['age_range'] = (0, min(17, min(head['age'], spouse['age']) - 18))
            child = self._generate_person('Child', race_info, state_code, child_params)
            members.append(child)
        
        return members
    
    def _generate_single_parent(self, race_info: Dict, state_code: str, params: Dict, structure_key: str) -> List[Dict]:
        """Generate single parent family"""
        members = []
        
        # Generate parent
        parent_params = params.copy()
        parent_params['gender'] = 'Female' if 'FEMALE' in structure_key else 'Male'
        parent = self._generate_person('Head of Household', race_info, state_code, parent_params)
        members.append(parent)
        
        # Generate children
        child_count = params.get('children')
        if child_count is None:
            child_count = random.choices([1, 2], weights=[70, 30])[0]
        
        for i in range(int(child_count)):
            child_params = params.copy()
            child_params['age_range'] = (0, min(17, parent['age'] - 18))
            child = self._generate_person('Child', race_info, state_code, child_params)
            members.append(child)
        
        return members
    
    def _generate_single_person(self, race_info: Dict, state_code: str, params: Dict) -> List[Dict]:
        """Generate single person household"""
        person = self._generate_person('Head of Household', race_info, state_code, params)
        return [person]
    
    def _generate_person(self, role: str, race_info: Dict, state_code: str, params: Dict) -> Dict[str, Any]:
        """Generate a single person with all attributes"""
        # Generate basic attributes
        age = self._generate_age(role, params.get('age_range'))
        gender = params.get('gender') or self._generate_gender()
        education = self._generate_education(age, params.get('education'))
        occupation = self._generate_occupation(education, age, params)
        employment = self._generate_employment_status(age, occupation)
        
        # Create person record
        person = {
            "role": role,
            "age": age,
            "gender": gender,
            "race": race_info['name'],
            "education_level": self.data['education_levels'][education]['name'],
            "education_level_key": education,
            "employment_status": employment,
            "annual_income": 0
        }
        
        # Add occupation details if employed
        if occupation and employment == "Employed":
            income = self._calculate_income(occupation, state_code, params)
            person.update({
                "occupation": occupation['title'],
                "occupation_code": occupation.get('code'),
                "occupation_group": occupation['group'],
                "annual_income": income,
                "employment_type": "Part-time" if age < 18 else "Full-time"
            })
        
        return person
    
    def _generate_age(self, role: str, age_range: Tuple[int, int] = None) -> int:
        """Generate age based on role and constraints"""
        if age_range:
            return random.randint(max(0, age_range[0]), min(100, age_range[1]))
        
        if role == 'Head of Household':
            return random.randint(25, 75)
        elif role == 'Spouse':
            return random.randint(25, 75)
        elif role == 'Child':
            return random.randint(0, 17)
        else:
            return random.randint(18, 80)
    
    def _generate_gender(self) -> str:
        """Generate gender with realistic distribution"""
        return 'Female' if random.random() < 0.51 else 'Male'
    
    def _generate_education(self, age: int, target_education: str = None) -> str:
        """Generate education level based on age and target"""
        if target_education:
            if target_education in self.data['education_levels']:
                return target_education
            # Try to find partial match
            for key, data in self.data['education_levels'].items():
                if target_education.lower() in data['name'].lower():
                    return key
        
        # Age-based education logic
        if age < 18:
            return 'less_than_hs'
        elif 18 <= age <= 24:
            return random.choices(
                ['high_school', 'some_college', 'bachelors'],
                weights=[40, 45, 15]
            )[0]
        elif age >= 65:
            return random.choices(
                ['less_than_hs', 'high_school', 'some_college', 'bachelors', 'graduate'],
                weights=[15, 40, 20, 20, 5]
            )[0]
        else:
            return random.choices(
                ['less_than_hs', 'high_school', 'some_college', 'bachelors', 'graduate'],
                weights=[10, 30, 30, 20, 10]
            )[0]
    
    def _generate_occupation(self, education: str, age: int, params: Dict) -> Optional[Dict]:
        """Generate occupation based on education and age"""
        if age < 16:
            return None
        
        # Filter occupations by education level
        suitable_occupations = []
        for code, occ_data in self.data['occupations'].items():
            if occ_data['education'] == education:
                suitable_occupations.append({**occ_data, 'code': code})
        
        if not suitable_occupations:
            # Fallback to any occupation
            for code, occ_data in self.data['occupations'].items():
                suitable_occupations.append({**occ_data, 'code': code})
        
        return random.choice(suitable_occupations) if suitable_occupations else None
    
    def _generate_employment_status(self, age: int, occupation: Dict = None) -> str:
        """Generate employment status"""
        if age < 16:
            return "Child (Under 16)"
        elif age >= 67:
            return "Retired" if random.random() < 0.8 else "Employed"
        elif 16 <= age <= 18:
            return "Student" if random.random() < 0.7 else "Student (Employed)"
        elif occupation:
            return "Employed"
        else:
            return random.choices(
                ["Employed", "Unemployed", "Not in Labor Force"],
                weights=[65, 10, 25]
            )[0]
    
    def _calculate_income(self, occupation: Dict, state_code: str, params: Dict) -> int:
        """Calculate income with variation and constraints"""
        base_salary = occupation.get('base_salary', 40000)
        
        # Apply income range constraint if specified
        income_range = params.get('income_range')
        if income_range:
            min_income, max_income = income_range
            # Adjust base salary to fit within range
            base_salary = max(min_income, min(max_income, base_salary))
        
        # Add variation (±20%)
        income = int(base_salary * random.uniform(0.8, 1.2))
        
        # Ensure within income range if specified
        if income_range:
            income = max(min_income, min(max_income, income))
        
        return income
    
    def _get_spouse_age_range(self, head_age: int) -> Tuple[int, int]:
        """Get reasonable age range for spouse"""
        return (max(18, head_age - 10), min(80, head_age + 10))
    
    def _apply_constraints(self, members: List[Dict], params: Dict) -> List[Dict]:
        """Apply income and other constraints to family"""
        income_range = params.get('income_range')
        if not income_range:
            return members
        
        min_income, max_income = income_range
        current_income = sum(m.get('annual_income', 0) for m in members)
        
        # Adjust incomes proportionally if needed
        if current_income > 0 and (current_income < min_income or current_income > max_income):
            target_income = random.randint(min_income, max_income)
            adjustment_factor = target_income / current_income
            
            for member in members:
                if member.get('annual_income', 0) > 0:
                    member['annual_income'] = int(member['annual_income'] * adjustment_factor)
        
        return members
    
    def _get_highest_education(self, members: List[Dict]) -> str:
        """Get highest education level in household"""
        education_order = ['less_than_hs', 'high_school', 'some_college', 'bachelors', 'graduate']
        highest = 'less_than_hs'
        
        for member in members:
            education = member.get('education_level_key', 'less_than_hs')
            if education_order.index(education) > education_order.index(highest):
                highest = education
        
        return self.data['education_levels'][highest]['name']
    
    def generate_families(self, count: int, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate multiple families"""
        families = []
        logger.info(f"Generating {count} families...")
        
        for i in range(count):
            try:
                family = self.generate_family(params)
                families.append(family)
                
                if (i + 1) % 10 == 0 or i == count - 1:
                    logger.info(f"  Generated {i + 1}/{count} families")
                    
            except Exception as e:
                logger.warning(f"Failed to generate family {i + 1}: {e}")
        
        return families
    
    def print_summary(self, families: List[Dict]):
        """Print summary of generated families"""
        if not families:
            logger.info("No families generated")
            return
        
        print(f"\n=== FAMILY GENERATION SUMMARY ===")
        print(f"Total families: {len(families)}")
        
        # Statistics
        total_income = sum(f['total_household_income'] for f in families)
        avg_income = total_income / len(families) if families else 0
        avg_size = sum(f['family_size'] for f in families) / len(families) if families else 0
        
        print(f"Average household income: ${avg_income:,.0f}")
        print(f"Average family size: {avg_size:.1f}")
        
        # Sample families
        print(f"\nSample families:")
        for i, family in enumerate(families[:3]):
            print(f"\nFamily {i + 1} ({family['family_id']}):")
            print(f"  Location: {family['state_name']}, {family['region']}")
            print(f"  Type: {family['family_type']}")
            print(f"  Race: {family['race']}")
            print(f"  Size: {family['family_size']} members")
            print(f"  Income: ${family['total_household_income']:,}")
            print(f"  Education: {family['highest_education']}")
            
            for member in family['members']:
                occupation = member.get('occupation', 'No occupation')
                income = member.get('annual_income', 0)
                income_str = f", ${income:,}" if income > 0 else ""
                print(f"    {member['role']}: {member['gender']}, Age {member['age']}, "
                      f"{member['education_level']}, {occupation}{income_str}")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

def parse_income_range(income_str: str) -> Tuple[int, int]:
    """Parse income range string like '50000-80000'"""
    if '-' in income_str:
        min_str, max_str = income_str.split('-')
        return (int(min_str), int(max_str))
    else:
        # Single value - create range around it
        value = int(income_str)
        return (int(value * 0.8), int(value * 1.2))

def main():
    """Main execution with argument parsing"""
    parser = argparse.ArgumentParser(description='Generate synthetic families with demographic realism')
    
    # Basic parameters
    parser.add_argument('--count', type=int, default=5, help='Number of families to generate')
    parser.add_argument('--state', type=str, help='Target state (name or code)')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    
    # Family characteristics
    parser.add_argument('--family-type', type=str, 
                       choices=['married_couple', 'single_person', 'single_parent_female', 'single_parent_male'],
                       help='Type of family structure')
    parser.add_argument('--race', type=str, help='Race/ethnicity (e.g., hispanic, white, black, asian)')
    parser.add_argument('--education', type=str,
                       choices=['less_than_hs', 'high_school', 'some_college', 'bachelors', 'graduate'],
                       help='Target education level')
    parser.add_argument('--income-range', type=str, help='Income range (e.g., 50000-80000)')
    parser.add_argument('--children', type=int, help='Number of children (for applicable family types)')
    parser.add_argument('--age-range', type=str, help='Age range for head of household (e.g., 30-50)')
    
    args = parser.parse_args()
    
    try:
        # Build parameters dictionary
        params = {}
        if args.state:
            params['state'] = args.state
        if args.family_type:
            params['family_type'] = args.family_type
        if args.race:
            params['race'] = args.race
        if args.education:
            params['education'] = args.education
        if args.income_range:
            params['income_range'] = parse_income_range(args.income_range)
        if args.children is not None:
            params['children'] = args.children
        if args.age_range:
            min_age, max_age = map(int, args.age_range.split('-'))
            params['age_range'] = (min_age, max_age)
        
        # Generate families
        logger.info("🏠 ENHANCED FAMILY GENERATOR")
        logger.info("="*40)
        
        generator = FamilyGenerator()
        families = generator.generate_families(args.count, params)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(families, f, indent=2, default=str)
            logger.info(f"✓ Results saved to {args.output}")
        
        generator.print_summary(families)
        generator.close()
        
        logger.info(f"\n✅ Successfully generated {len(families)} families!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
