import json
import pandas as pd
from datetime import datetime
import numpy as np


def parse_date_ranges(date_ranges):
    """Convert list of date ranges into a list of all dates within those ranges."""
    if not date_ranges:
        return []

    all_dates = []
    for date_range in date_ranges:
        start = datetime.strptime(date_range[0], '%Y-%m-%d')
        end = datetime.strptime(date_range[1], '%Y-%m-%d')
        date_range_list = pd.date_range(start=start, end=end)
        all_dates.extend(date_range_list)
    return all_dates


def create_employee_calendar(data):
    # Extract employee data
    employees_data = data['_default']

    # Create a date range for the entire year
    year_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

    # Initialize the DataFrame
    df = pd.DataFrame(index=year_dates)

    # Process each employee's data
    for emp_id, emp_data in employees_data.items():
        name = emp_data['name']

        # Initialize employee's column with their type
        df[f"{name} ({emp_data['type']})"] = ''

        # Process each type of absence
        absence_types = {
            'vacation': 'Vacation',
            'sick': 'Sick',
            'Free': 'Free',
            'child_sick': 'Child Sick',
            'maternity': 'Maternity',
            'School': 'School',
            'training': 'Training'
        }

        for absence_type, label in absence_types.items():
            if absence_type in emp_data:
                dates = parse_date_ranges(emp_data[absence_type])
                for date in dates:
                    df.loc[date, f"{name} ({emp_data['type']})"] = label

    return df


def save_to_excel(json_data, output_file='employee_schedule.xlsx'):
    # Create the calendar DataFrame
    calendar_df = create_employee_calendar(json_data)

    # Create Excel writer object
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Write the calendar
        calendar_df.to_excel(writer, sheet_name='Schedule', index=True)

        # Get the worksheet
        worksheet = writer.sheets['Schedule']

        # Format the worksheet
        worksheet.freeze_panes = 'B2'  # Freeze the first row and column

        # Adjust column widths
        worksheet.column_dimensions['A'].width = 12  # Date column
        for col in range(1, worksheet.max_column):
            worksheet.column_dimensions[chr(66 + col)].width = 20  # Other columns

        # Create color mapping sheet
        colors_df = pd.DataFrame(json_data['colors'].items(), columns=['Type', 'Color'])
        colors_df.to_excel(writer, sheet_name='Color Codes', index=False)


# Main execution
if __name__ == "__main__":
    # Read the JSON data
    with open('celerobase-default-rtdb-export.json', 'r') as file:
        json_data = json.load(file)

    # Convert and save to Excel
    save_to_excel(json_data)
    print("Excel file has been created successfully!")