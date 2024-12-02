import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from tinydb import TinyDB, Query
from openai import OpenAI
from streamlit_calendar import calendar
import random
from typing import List, Dict, Tuple, Optional
import altair as alt
import firebase_admin
from firebase_admin import credentials, db
import requests
import json

# Page configuration
st.set_page_config(layout="wide")

# Constants
try:
    API = st.secrets['OPENAI_API_KEY']
    OPENAI_API_KEY = API
except KeyError:
    st.error("OPENAI_API_KEY is missing in the secrets configuration.")
    st.stop()

EVENT_TYPES = ["Vacation", "Sick", "Child Sick", "Training"]
DEFAULT_CALENDAR_OPTIONS = {
    "editable": True,
    "navLinks": True,
    "selectable": True
}

# Database initialization
db = TinyDB("ferie_db.json")
events_table = db.table('events')
User = Query()

JSONBIN_API_KEY = st.secrets['JSONBIN_API_KEY']

class ColorManager:
    TYPE_COLORS = {
        "Person": "#34a853",
        "Hund": "#c19b41",
        "Abe": "#f045c1",
        "Tømrer": "#971a5b"
    }

    @staticmethod
    def get_color_for_type(type_name: str) -> str:
        """Get a consistent color for a given type."""
        if type_name in ColorManager.TYPE_COLORS:
            return ColorManager.TYPE_COLORS[type_name]

        # Generate a random color for new types
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        ColorManager.TYPE_COLORS[type_name] = color
        return color


class JSONBinManager:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.jsonbin.io/v3"
        self.headers = {
            "Content-Type": "application/json",
            "X-Master-key": self.api_key
        }
        self.bin_id = self._get_or_create_bin()

    def _get_or_create_bin(self):
        """Retrieve or create a JSONBin for storing data."""
        try:
            # Check if a bin ID is stored in Streamlit secrets
            bin_id = st.secrets.get('JSONBIN_BIN_ID')

            if not bin_id:
                # Create a new bin if no existing ID
                create_url = f"{self.base_url}/b"
                initial_data = {"users": []}
                response = requests.post(
                    create_url,
                    json=initial_data,
                    headers=self.headers
                )
                response.raise_for_status()
                bin_id = response.json()['metadata']['id']

                st.write(f"Created new JSONBin with ID: {bin_id}")

            return bin_id
        except Exception as e:
            st.error(f"Error getting/creating bin: {e}")
            return None

    def get_events(self, selected_user_name=None, selected_user_type=None):
        """Retrieve events from JSONBin, with optional filtering."""
        try:
            response = requests.get(
                f"{self.base_url}/b/{self.bin_id}",
                headers=self.headers
            )
            response.raise_for_status()

            data = response.json()['record'].get('users', [])

            # Apply filters
            if selected_user_name and selected_user_name != "All":
                data = [user for user in data if user.get('name') == selected_user_name]
            elif selected_user_type:
                data = [user for user in data if user.get('type', '').lower() == selected_user_type.lower()]

            return self._convert_to_fullcalendar(data)

        except Exception as e:
            st.error(f"Error retrieving events: {e}")
            return []

    def _convert_to_fullcalendar(self, events_data):
        """Convert events to FullCalendar format."""
        fullcalendar_events = []

        for event in events_data:
            event_type = event.get("type", "Unknown")
            color = ColorManager.get_color_for_type(event_type)

            for event_type_key in ["vacation", "sick", "child_sick", "training"]:
                dates = event.get(event_type_key, [])
                if not dates:
                    continue

                for start, end in dates:
                    try:
                        start_date = datetime.strptime(start, '%Y-%m-%d').date()
                        end_date = datetime.strptime(end, '%Y-%m-%d').date()
                        end_date = end_date + timedelta(days=1)

                        fullcalendar_events.append({
                            "title": f"{event.get('name', 'Unknown')} - {event_type_key.capitalize()}",
                            "start": start_date.isoformat(),
                            "end": end_date.isoformat(),
                            "color": color,
                            "description": f"{event_type_key.capitalize()} event for {event.get('name', 'Unknown')}"
                        })
                    except (ValueError, TypeError) as e:
                        st.error(f"Error processing dates: {e}")

        return fullcalendar_events

    def add_or_update_user(self, name, type, **events):
        """Add or update a user in JSONBin."""
        try:
            # Fetch current data
            response = requests.get(
                f"{self.base_url}/b/{self.bin_id}",
                headers=self.headers
            )
            response.raise_for_status()
            current_data = response.json()['record']

            # Convert dates to ISO format
            formatted_events = {
                k: [(s.isoformat(), e.isoformat()) for s, e in v] if v else []
                for k, v in events.items()
            }

            # Generate color
            color = ColorManager.get_color_for_type(type)

            # Check if user exists and update
            users = current_data.get('users', [])
            existing_user_index = next(
                (index for index, user in enumerate(users)
                 if user.get('name') == name),
                None
            )

            if existing_user_index is not None:
                # Update existing user
                users[existing_user_index].update({
                    "type": type,
                    "color": color,
                    **formatted_events
                })
            else:
                # Add new user
                users.append({
                    "name": name,
                    "type": type,
                    "color": color,
                    **formatted_events
                })

            # Update data in JSONBin
            updated_data = {"users": users}
            response = requests.put(
                f"{self.base_url}/b/{self.bin_id}",
                json=updated_data,
                headers=self.headers
            )
            response.raise_for_status()

            st.success(f"User {name} added/updated successfully.")
            return users

        except Exception as e:
            st.error(f"Error updating JSONBin: {e}")
            return None

    def get_unique_names_and_types(self):
        """Retrieve unique names and types from JSONBin."""
        try:
            response = requests.get(
                f"{self.base_url}/b/{self.bin_id}",
                headers=self.headers
            )
            response.raise_for_status()

            users = response.json()['record'].get('users', [])

            names = list(set(user.get('name') for user in users if user.get('name'))) + ["All"]
            types = list(set(user.get('type') for user in users if user.get('type')))

            return names, types

        except Exception as e:
            st.error(f"Error retrieving names and types: {e}")
            return [], []


class CalendarApp:
    def __init__(self):
        self.jsonbin_manager = JSONBinManager(JSONBIN_API_KEY)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.calendar_modes = {
            "daygrid": {
                "initialView": "dayGridMonth",
                "headerToolbar": {
                    "left": "prev,next today",
                    "center": "title",
                    "right": "dayGridMonth,dayGridWeek,dayGridDay"
                },
                "views": {
                    "dayGridMonth": {"titleFormat": {"month": "long", "year": "numeric"}},
                    "dayGridWeek": {"titleFormat": {"month": "long", "day": "numeric"}},
                    "dayGridDay": {"titleFormat": {"weekday": "long", "month": "long", "day": "numeric"}}
                }
            },
            "timegrid": {
                "initialView": "timeGridWeek",
                "headerToolbar": {
                    "left": "prev,next today",
                    "center": "title",
                    "right": "timeGridWeek,timeGridDay"
                },
                "slotMinTime": "06:00:00",
                "slotMaxTime": "20:00:00",
                "slotDuration": "00:30:00",
                "views": {
                    "timeGridWeek": {"titleFormat": {"month": "long", "day": "numeric"}},
                    "timeGridDay": {"titleFormat": {"weekday": "long", "month": "long", "day": "numeric"}}
                }
            },
            "timeline": {
                "initialView": "timelineMonth",
                "headerToolbar": {
                    "left": "prev,next today",
                    "center": "title",
                    "right": "timelineMonth,timelineWeek,timelineDay"
                },
                "views": {
                    "timelineMonth": {"titleFormat": {"month": "long", "year": "numeric"}},
                    "timelineWeek": {"titleFormat": {"month": "long", "day": "numeric"}},
                    "timelineDay": {"titleFormat": {"weekday": "long", "month": "long", "day": "numeric"}}
                },
                "slotMinTime": "06:00:00",
                "slotMaxTime": "20:00:00"
            },
            "list": {
                "initialView": "listMonth",
                "headerToolbar": {
                    "left": "prev,next today",
                    "center": "title",
                    "right": "listMonth,listWeek,listDay"
                },
                "views": {
                    "listMonth": {"titleFormat": {"month": "long", "year": "numeric"}},
                    "listWeek": {"titleFormat": {"month": "long", "day": "numeric"}},
                    "listDay": {"titleFormat": {"weekday": "long", "month": "long", "day": "numeric"}}
                }
            },
            "multimonth": {
                "initialView": "multiMonthYear",
                "headerToolbar": {
                    "left": "prev,next today",
                    "center": "title",
                    "right": "multiMonthYear"
                },
                "views": {
                    "multiMonthYear": {
                        "titleFormat": {"year": "numeric"},
                        "multiMonthMaxColumns": 3,
                        "multiMonthMinWidth": 350
                    }
                }
            }
        }


    def display_calendar(self, user_override=None, mode=True, types=None):
        """Display the calendar with filtered events."""
        if mode:
            mode = st.selectbox("Calendar Mode:",
                                ["daygrid", "timegrid", "timeline", "list", "multimonth"])

        # Handle user selection
        if not types:
            names, _ = self.jsonbin_manager.get_unique_names_and_types()
            selected_user = (user_override if user_override else
                             st.sidebar.selectbox("Select User", names))
        else:
            selected_user = None

        # Update events in session state
        events_need_update = (
                "events" not in st.session_state or
                st.session_state.get("selected_user") != selected_user or
                st.session_state.get("selected_type") != types
        )

        if events_need_update:
            if types:
                events = self.jsonbin_manager.get_events(selected_user_type=types)
            else:
                events = self.jsonbin_manager.get_events(selected_user_name=selected_user)

            st.session_state.events = events
            st.session_state.selected_user = selected_user
            st.session_state.selected_type = types

        # Display calendar
        state = calendar(
            events=st.session_state.events,
            options=DEFAULT_CALENDAR_OPTIONS
        )

        if state.get("eventsSet") is not None:
            st.session_state["events"] = state["eventsSet"]

    def ask_assistant(self, content: str, database: List[Dict], data: Dict) -> str:
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "Du er en professionel mønstergenkendelsesrobot. "
                                   "Du vil analysere mønstre i folks fridage fra arbejde. "
                                   "Du modtager en database med alle personers orlov. "
                                   "Du vil derefter analysere for mønstre og markere, "
                                   "når nogen har vist et mønster i deres orlov."
                    },
                    {
                        "role": "user",
                        "content": f"Hvem? {content} "
                                   f"Og her er dataene {data} i ugedagsformat. "
                                   f"Her er hele databasen med datoerne {database}"
                    }
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error getting AI response: {e}")
            return "Der opstod en fejl ved behandling af din forespørgsel."

    def main(self):
        st.title("Vacation Calendar")
        self.setup_sidebar()

        page_choice = st.sidebar.selectbox("Choose a page", ["Calendar", "Login"])
        if page_choice == "Calendar":
            self.display_calendar()
        else:
            self.login_page()

    def setup_sidebar(self):
        st.sidebar.header("User Management")
        with st.sidebar.form(key="add_user_form"):
            name = st.text_input("Name")
            user_type = st.text_input("Type")
            start = st.date_input("Start")
            end = st.date_input("End")
            off_type = st.selectbox("Absence Type", EVENT_TYPES, index=0)

            if st.form_submit_button("Add/Update User"):
                event_type = off_type.lower().replace(" ", "_")
                self.jsonbin_manager.add_or_update_user(
                    name, user_type, **{event_type: [(start, end)]}
                )

    def login_page(self):
        """Handle login page display and functionality."""
        # Get unique names and types
        names, types = self.jsonbin_manager.get_unique_names_and_types()

        switch = st.radio("Select by:", ["Name", "Type"])
        view_mode = st.radio("View Mode:", ["Calendar", "Statistics"])

        chosen = None
        chosen_type = None

        if switch == "Name":
            chosen = st.selectbox("Select a name:", names)
            if chosen == "All":
                events = self.jsonbin_manager.get_events()
            elif chosen:
                events = self.jsonbin_manager.get_events(selected_user_name=chosen)
        else:
            chosen_type = st.selectbox("Select a type:", types)
            if chosen_type:
                events = self.jsonbin_manager.get_events(selected_user_type=chosen_type)

        if view_mode == "Calendar":
            if chosen == "All":
                self.display_calendar()
            elif chosen:
                self.display_calendar(user_override=chosen)
            elif chosen_type:
                self.display_calendar(types=chosen_type)
        else:  # Statistics view
            if events:
                display_statistics(events)

        # AI Assistant section
        question = st.text_input("Ask the assistant about the leave patterns:")
        if question:
            pass  # Implement AI assistant logic if needed

def migrate_database_colors():
    """Update all existing database entries to use type-based colors."""
    all_users = db.all()
    for user in all_users:
        type_name = user.get("type")
        if type_name:
            color = ColorManager.get_color_for_type(type_name)
            db.update({"color": color}, User.name == user["name"])


def calculate_statistics(events):
    """Calculate statistics from events data with enhanced output for Altair."""
    from collections import defaultdict

    # Define weekday names
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    day_stats = defaultdict(int)
    category_stats = defaultdict(int)
    combined_stats = []

    for event in events:
        start = datetime.strptime(event['start'], '%Y-%m-%d')
        end = datetime.strptime(event['end'], '%Y-%m-%d')

        # Get category from title
        category = event['title'].split(' - ')[1]

        # Count days between start and end
        current = start
        while current < end:
            # Get day name using weekday index
            day_name = weekday_names[current.weekday()]
            day_stats[day_name] += 1
            category_stats[category] += 1

            # Add to combined stats for detailed visualization
            combined_stats.append({
                'Day': day_name,
                'Category': category,
                'Count': 1  # Each day counts as 1
            })

            current += timedelta(days=1)

    # Aggregate combined stats
    df_combined = pd.DataFrame(combined_stats)
    if df_combined.empty:
        # Create empty DataFrame with correct structure if no data
        return pd.DataFrame(columns=['Day', 'Category', 'Count'])

    df_agg = df_combined.groupby(['Day', 'Category']).sum().reset_index()

    # Ensure all day-category combinations exist
    categories = df_agg['Category'].unique()

    # Create all possible combinations
    index = pd.MultiIndex.from_product([weekday_names, categories], names=['Day', 'Category'])
    df_full = pd.DataFrame(index=index).reset_index()

    # Merge with actual data
    df_final = df_full.merge(df_agg, on=['Day', 'Category'], how='left').fillna(0)

    return df_final


def create_leave_dashboard(events):
    """Create an interactive leave statistics dashboard using Altair."""

    # Calculate statistics
    df = calculate_statistics(events)

    if df.empty:
        st.warning("No data available for visualization")
        return None, {'total_days': 0, 'avg_per_category': 0, 'most_common_day': 'N/A'}

    # Create base charts
    base = alt.Chart(df).properties(height=300)

    # Color scheme
    color_scale = alt.Scale(scheme='category10')

    # Day of Week Chart
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    col1, col2, col3 = st.columns(3)

    with col1:
        day_chart = base.mark_bar().encode(
            x=alt.X('Day:N', sort=days_order, title='Day of Week'),
            y=alt.Y('sum(Count):Q', title='Number of Days'),
            color=alt.Color('Category:N', scale=color_scale),
            tooltip=[
                alt.Tooltip('Day:N'),
                alt.Tooltip('Category:N'),
                alt.Tooltip('sum(Count):Q', title='Days', format='.0f')
            ]
        ).properties(
            title='Leave Distribution by Day of Week'
        )
        st.altair_chart(day_chart, use_container_width=True)

    with col2:
        # Category Chart
        category_chart = base.mark_bar().encode(
            x=alt.X('Category:N', title='Leave Category'),
            y=alt.Y('sum(Count):Q', title='Number of Days'),
            color=alt.Color('Category:N', scale=color_scale),
            tooltip=[
                alt.Tooltip('Category:N'),
                alt.Tooltip('sum(Count):Q', title='Total Days', format='.0f')
            ]
        ).properties(
            title='Leave Distribution by Category'
        )
        st.altair_chart(category_chart, use_container_width=True)

    # Heatmap
    with col3:
        heatmap = base.mark_rect().encode(
            x=alt.X('Day:N', sort=days_order, title='Day of Week'),
            y=alt.Y('Category:N', title='Leave Category'),
            color=alt.Color('sum(Count):Q',
                            scale=alt.Scale(scheme='viridis'),
                            title='Number of Days'),
            tooltip=[
                alt.Tooltip('Day:N'),
                alt.Tooltip('Category:N'),
                alt.Tooltip('sum(Count):Q', title='Days', format='.0f')
            ]
        ).properties(
            title='Leave Heatmap'
        )
        st.altair_chart(heatmap, use_container_width=True)

    # Remove the previous alt.vconcat code

    # Add overall metrics
    total_days = df['Count'].sum()
    avg_per_category = df.groupby('Category')['Count'].sum().mean()
    most_common_day = df.groupby('Day')['Count'].sum().idxmax()

    # Return metrics
    return None, {
        'total_days': int(total_days),
        'avg_per_category': round(avg_per_category, 1),
        'most_common_day': most_common_day
    }


# Update the statistics section in your login_page method:
def display_statistics(events):
    """Display the statistics dashboard."""
    if events:
        chart, metrics = create_leave_dashboard(events)

        if chart is not None:
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Leave Days", metrics['total_days'])
            with col2:
                st.metric("Average Days per Category", metrics['avg_per_category'])
            with col3:
                st.metric("Most Common Day", metrics['most_common_day'])

            # Display the combined chart
            st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No events data available for visualization")

# Add this to your main() function or where you initialize the app
if __name__ == "__main__":
    # Migrate existing database entries to use type-based colors
    migrate_database_colors()
    app = CalendarApp()
    app.main()
