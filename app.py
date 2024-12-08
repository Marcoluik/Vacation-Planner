import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from tinydb import TinyDB, Query
from openai import OpenAI
from streamlit_calendar import calendar
import random
from typing import List, Dict, Tuple, Optional
import altair as alt

#FIREBASE
import firebase_admin
from firebase_admin import credentials, db
import json


class FirebaseManager:
    @staticmethod
    def initialize_firebase():
        """Initialize Firebase app if not already initialized."""
        if not firebase_admin._apps:
            try:
                # Load Firebase credentials from Streamlit secrets
                print("Attempting to initialize Firebase...")
                load = json.loads(st.secrets["FIREBASE_KEY"], strict=False)
                cred = credentials.Certificate(load)
                firebase_admin.initialize_app(cred, {
                    "databaseURL": st.secrets["FIREBASE_DATABASE_URL"]
                })
                print("Firebase initialization successful")
            except KeyError as ke:
                print(f"Firebase configuration error: {ke}")
                st.error("Firebase configuration is missing in the secrets.")
                st.stop()
            except Exception as e:
                print(f"Unexpected Firebase initialization error: {e}")

    @classmethod
    def get_ref(cls, path: str = "/"):
        """Get a reference to a specific path in the database."""
        cls.initialize_firebase()
        return db.reference(path)

class DatabaseManager:
    @staticmethod
    def all():
        """
        Retrieve all users from the database.

        :return: List of all user records
        """
        try:
            ref = FirebaseManager.get_ref("/_default")
            data = ref.get() or {}
            #print(f"Retrieved users data: {data}")

            # Convert Firebase's nested dictionary to a list of users
            if '_default' in data:
                users = [
                    user for user in data['_default']
                    if isinstance(user, dict)
                ]
            else:
                # Directly use root-level data
                users = [
                    user for user in data.values()
                    if isinstance(user, dict)
                ]

            #print(f"Processed users: {users}")
            return users

        except Exception as e:
            print(f"Error retrieving all users: {e}")
            return []

    @staticmethod
    @staticmethod
    def add_or_update_user(name: str, type: str, **events) -> None:
        """
        Add or update a user in the Firebase Realtime Database.
        Automatically adds or updates a color entry for the type.

        :param name: User's name
        :param type: User's type
        :param events: Dictionary of event types and their date ranges
        """
        print(f"Adding/Updating user: {name}, Type: {type}")

        if not name or not type:
            print("Error: Name and type are required!")
            st.error("Name and type are required!")
            return

        # Get a reference to the root and colors nodes
        ref = FirebaseManager.get_ref("/_default")
        colors_ref = FirebaseManager.get_ref("colors")

        # Convert dates to ISO format strings
        formatted_events = {
            k: [(s.isoformat(), e.isoformat()) for s, e in v] if v else []
            for k, v in events.items()
        }

        # Get color based on type
        color = ColorManager.get_color_for_type(type)

        # Prepare user data
        user_data = {
            "name": name,
            "type": type,
            "color": color,
            **formatted_events
        }

        try:
            # Find existing user by name
            existing_users = ref.get() or {}
            print(f"Existing users: {existing_users}")

            # Check if the type color already exists
            existing_colors = colors_ref.get() or {}
            if type not in existing_colors:
                # Add new color entry for the type
                colors_ref.child(type).set(color)
                print(f"Added new color entry for type: {type}")

            user_key = None
            for key, user in existing_users.items():
                if isinstance(user, dict) and user.get('name') == name:
                    user_key = key
                    break

            if user_key:
                # Update existing user
                print(f"Updating existing user with key: {user_key}")
                ref.child(user_key).update(user_data)
                st.success(f"Updated events for {name}")
            else:
                # Add new user
                print("Adding new user")
                ref.push(user_data)
                st.success(f"User {name} added successfully.")

        except Exception as e:
            print(f"Error adding/updating user: {e}")
            st.error(f"Error adding/updating user: {e}")


    @staticmethod
    def search(query):
        """
        Simulate TinyDB's search method.

        :param query: A query object (in this case, we'll simulate a simple name search)
        :return: List of matching users
        """
        # Check if the query is looking for a user by name
        if hasattr(query, 'name'):
            ref = FirebaseManager.get_ref("/_default")
            users_data = ref.get() or {}

            return [
                user for _, user in users_data.items()
                if isinstance(user, dict) and user.get('name') == query.name
            ]

        return []

    @staticmethod
    def load_events(selected_user_name: Optional[str] = None,
                    selected_user_type: Optional[str] = None) -> List[Dict]:
        print("load_events")
        try:
            # Get reference to users
            ref = FirebaseManager.get_ref("/_default")
            users_data = ref.get() or {}
            #print(f"Raw users data: {users_data}")  # Debug print

            # Convert users to list, filtering as needed
            filtered_users = []

            # Handle the '_default' key if it exists
            if '_default' in users_data:
                users = users_data['_default']
                # Skip the first None element
                users = [user for user in users if user is not None]
            else:
                users = list(users_data.values())

            for user in users:
                #print(f"Processing user: {user}")  # Debug print
                if not isinstance(user, dict):
                    continue

                # Apply name filter
                if selected_user_name and selected_user_name != "All":
                    if user.get('name') != selected_user_name:
                        continue

                # Apply type filter
                if selected_user_type:
                    if user.get('type', '').lower() != selected_user_type.lower():
                        continue

                filtered_users.append(user)

            #print(f"Filtered users: {filtered_users}")  # Debug print

            # Convert to FullCalendar events
            events = EventManager.convert_to_fullcalendar(filtered_users)
            #print(f"FullCalendar events: {events}")  # Debug print
            return events

        except Exception as e:
            st.error(f"Error loading events: {e}")
            return []

    @staticmethod
    def find_by_name(name: str) -> Optional[Dict]:
        """
        Find a user by name in the Firebase database.

        :param name: Name of the user to find
        :return: User data or None if not found
        """
        ref = FirebaseManager.get_ref("/")
        users = ref.get() or {}

        for _, user in users.items():
            if isinstance(user, dict) and user.get('name') == name:
                return user

        return None


    @staticmethod
    def delete_user(name: str) -> bool:
        """
        Delete a user from the Firebase database.

        :param name: Name of the user to delete
        :return: True if deleted, False otherwise
        """
        ref = FirebaseManager.get_ref("/")
        users = ref.get() or {}

        for key, user in users.items():
            if isinstance(user, dict) and user.get('name') == name:
                ref.child(key).delete()
                st.success(f"User {name} deleted successfully.")
                return True

        st.error(f"No user found with name {name}")
        return False

# Page configuration
st.set_page_config(layout="wide", page_title="FeriePlan", page_icon=":calendar:")

# Constants
try:
    API = st.secrets['OPENAI_API_KEY']
    OPENAI_API_KEY = API
except KeyError:
    st.error("OPENAI_API_KEY is missing in the secrets configuration.")
    st.stop()

EVENT_TYPES = ["vacation", "sick", "child_sick", "training"]
DEFAULT_CALENDAR_OPTIONS = {
    "editable": True,
    "navLinks": True,
    "selectable": True
}


class ColorManager:
    color_ref = FirebaseManager.get_ref("/colors")

    @staticmethod
    def get_color_for_type(type_name: str) -> str:
        """
        Get a consistent color for a given type, using Firebase as the source of truth.

        :param type_name: The type to get a color for
        :return: A color hex code
        """
        try:
            # First, try to fetch the color from Firebase
            existing_color = ColorManager.color_ref.child(type_name).get()

            # If color exists in Firebase, return it
            if existing_color:
                return existing_color

            # If no color in Firebase, generate a new color
            new_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

            # Store the new color in Firebase
            ColorManager.color_ref.child(type_name).set(new_color)

            return new_color

        except Exception as e:
            print(f"Error in get_color_for_type: {e}")

            # Fallback to a default color if Firebase access fails
            return "#808080"  # A neutral gray color

class EventManager:
    @staticmethod
    def convert_to_fullcalendar(events_data: List[Dict], types: Optional[Dict] = None) -> List[Dict]:
        print("Converting events to FullCalendar format...")
        fullcalendar_events = []
        for event in events_data:
            # Get color based on type
            event_type = event.get("type", "Unknown")
            color = ColorManager.get_color_for_type(event_type)

            # Define a mapping of event keys to readable names
            event_key_mapping = {
                "vacation": "Ferie",
                "sick": "Sygdom",
                "child_sick": "Barnsygdom",
                "training": "Kursus"
            }

            for event_key, readable_name in event_key_mapping.items():
                dates = event.get(event_key, [])
                if not dates:
                    continue

                for start, end in dates:
                    try:
                        start_date = datetime.strptime(start, '%Y-%m-%d').date()
                        end_date = datetime.strptime(end, '%Y-%m-%d').date()

                        # Adjust end date to be inclusive
                        end_date = end_date + timedelta(days=1)

                        fullcalendar_events.append({
                            "title": f"{event.get('name', 'Unknown')} - {readable_name}",
                            "start": start_date.isoformat(),
                            "end": end_date.isoformat(),
                            "color": color,  # Color now based on type
                            "description": f"{readable_name} event for {event.get('name', 'Unknown')}"
                        })
                    except (ValueError, TypeError) as e:
                        st.error(f"Error processing dates for {event.get('name', 'Unknown')}: {e}")

        #print(f"Generated FullCalendar events: {fullcalendar_events}")
        return fullcalendar_events
class CalendarApp:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        # Existing calendar mode definitions remain the same
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

    def display_calendar(self, user_override: Optional[str] = None,
                         mode: bool = True, types: Optional[str] = None) -> None:
        """Display the calendar with filtered events."""
        print("Displaying calendar...")

        #if mode:
            #mode = st.selectbox("Kalender Mode:",
                                #["daygrid", "timegrid", "timeline", "list", "multimonth"])
       
        mode = "daygrid"

        # Handle user selection
        if not types:
            selected_user = (user_override if user_override else
                             st.sidebar.selectbox("Vælg bruger",
                                                  [None] + [user["name"] for user in DatabaseManager.all()]))
        else:
            selected_user = None

        # Only update events if selection changes or events don't exist
        if ("events" not in st.session_state or 
                st.session_state.get("selected_user") != selected_user or 
                st.session_state.get("selected_type") != types):
            
            # Load new events based on the current user or type
            if types:
                events = DatabaseManager.load_events(selected_user_type=types)
            elif selected_user:
                events = DatabaseManager.load_events(selected_user_name=selected_user)
            else:
                events = DatabaseManager.load_events()

            # Store events and selections in session state
            st.session_state.events = events
            st.session_state.selected_user = selected_user
            st.session_state.selected_type = types

        # Display calendar with current events
        state = calendar(
            events=st.session_state.events,
            options=DEFAULT_CALENDAR_OPTIONS
        )

        # Only update events if they've been modified through the calendar
        if state.get("eventsSet") is not None:
            st.session_state.events = state["eventsSet"]

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
        st.title("Ferieplan")
        self.setup_sidebar()
        



        page_choice = st.sidebar.selectbox("Vælg side", ["Calendar", "Stats"])
        if page_choice == "Calendar":
            self.display_calendar()
        else:
            self.login_page()

    def setup_sidebar(self):
        st.sidebar.header("Brugerdata")
        with st.sidebar.form(key="add_user_form"):
            name = st.text_input("Navn")
            user_type = st.text_input("Brugertype")
            start = st.date_input("Start")
            end = st.date_input("Slut")
            off_type = st.selectbox("Type fravær", EVENT_TYPES, index=0)

            if st.form_submit_button("Tilføj"):
                event_type = off_type
                DatabaseManager.add_or_update_user(
                    name, user_type, **{event_type: [(start, end)]}
                )

    def login_page(self):
        """Handle login page display and functionality."""
        # Get unique names and types
        names = [item.get('name') for item in DatabaseManager.all() if item.get('name')] + ["All"]
        types = list(set(item.get('type') for item in DatabaseManager.all() if item.get('type')))

        # Create two columns for the control panel
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Kontrol Panel")
            switch = st.radio("Sorter efter:", ["Navn", "Type"])
            view_mode = st.radio("Vis data:", ["Calendar", "Statistics"])

        with col2:
            st.subheader("Vælg Data")
            chosen = None
            chosen_type = None

            if switch == "Navn":
                chosen = st.selectbox("Vælg navn:", names)
            else:
                chosen_type = st.selectbox("Vælg type:", types)
            # AI Assistant section
            question = st.text_input("Sprøg Kunsitg inteligens:")
            if question:
                dball = DatabaseManager.all()
                current_data = chosen if chosen and chosen != "All" else dball
                answer = self.ask_assistant(question, dball, current_data)
                st.subheader("Assistant's Response:")
                st.write(answer)

        # Rest of your existing code...
        if chosen == "All":
            events = DatabaseManager.load_events()
        elif chosen:
            print(f"Loading events for user: {chosen}")
            events = DatabaseManager.load_events(selected_user_name=chosen)
        elif chosen_type:
            events = DatabaseManager.load_events(selected_user_type=chosen_type)

        if view_mode == "Calendar":
            if chosen == "All":
                self.display_calendar()
                print("Displaying calendar for all users")
            elif chosen:
                self.display_calendar(user_override=chosen)
                print(f"Displaying calendar for user: {chosen}")
            elif chosen_type:
                self.display_calendar(types=chosen_type)
                print(f"Displaying calendar for type: {chosen_type}")
        else:  # Statistics view
            if events:
                display_statistics(events)

        # Debug information and AI Assistant section
        if st.checkbox("-$-"):
            st.write("Valgt:")
            st.write({
                "Chosen Name": chosen,
                "Chosen Type": chosen_type,
                "Number of Events": len(events) if events else 0,
                "Database Records": len(DatabaseManager.all())
            })

        




def calculate_statistics(events, date_range: str = "all"):
    """Calculate statistics from events data with date range filtering."""
    from collections import defaultdict
    
    # Define date ranges
    now = datetime.now()
    date_filters = {
        "last_month": now - timedelta(days=30),
        "last_6_months": now - timedelta(days=180),
        "last_year": now - timedelta(days=365),
        "all": datetime.min
    }
    
    filter_date = date_filters.get(date_range, datetime.min)
    
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    combined_stats = []

    for event in events:
        start = datetime.strptime(event['start'], '%Y-%m-%d')
        end = datetime.strptime(event['end'], '%Y-%m-%d')
        
        # Skip events outside the selected date range
        if start < filter_date:
            continue
            
        category = event['title'].split(' - ')[1]
        
        current = start
        while current < end:
            day_name = weekday_names[current.weekday()]
            combined_stats.append({
                'Day': day_name,
                'Category': category,
                'Count': 1
            })
            current += timedelta(days=1)

    # Create DataFrame and process as before
    df_combined = pd.DataFrame(combined_stats)
    if df_combined.empty:
        return pd.DataFrame(columns=['Day', 'Category', 'Count'])

    # Only aggregate the 'Count' column
    df_agg = df_combined.groupby(['Day', 'Category'])['Count'].sum().reset_index()
    
    # Create all possible combinations
    categories = df_agg['Category'].unique()
    index = pd.MultiIndex.from_product([weekday_names, categories], names=['Day', 'Category'])
    df_full = pd.DataFrame(index=index).reset_index()
    
    df_final = df_full.merge(df_agg, on=['Day', 'Category'], how='left').fillna(0)
    return df_final

def display_statistics(events):
    """Display the statistics dashboard with date range filtering."""
    # Add date range selector
    date_range = st.selectbox(
        "Vælg tidsperiode:",
        ["all", "last_month", "last_6_months", "last_year"],
        format_func=lambda x: {
            "all": "Alle data",
            "last_month": "Sidste måned",
            "last_6_months": "Sidste 6 måneder",
            "last_year": "Sidste år"
        }[x]
    )

    if events:
        # Calculate statistics with date range filter
        df = calculate_statistics(events, date_range)
        
        # Debug print
        st.write("Debug: Number of events in DataFrame:", len(df))
        
        if not df.empty:
            # Create visualizations with filtered data
            col1, col2, col3 = st.columns(3)

            with col1:
                # Day of Week Chart
                day_chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('Day:N', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
                    y='sum(Count):Q',
                    color='Category:N',
                    tooltip=['Day:N', 'Category:N', 'Count:Q']
                ).properties(
                    title='Daglig fordeling',
                    height=300
                )
                st.altair_chart(day_chart, use_container_width=True)

            with col2:
                # Category Chart
                category_chart = alt.Chart(df).mark_bar().encode(
                    x='Category:N',
                    y='sum(Count):Q',
                    color='Category:N',
                    tooltip=['Category:N', 'Count:Q']
                ).properties(
                    title='Fraværstyper',
                    height=300
                )
                st.altair_chart(category_chart, use_container_width=True)

            with col3:
                # Heatmap
                heatmap = alt.Chart(df).mark_rect().encode(
                    x=alt.X('Day:N', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
                    y='Category:N',
                    color='Count:Q',
                    tooltip=['Day:N', 'Category:N', 'Count:Q']
                ).properties(
                    title='Fravær Heatmap',
                    height=300
                )
                st.altair_chart(heatmap, use_container_width=True)

            # Calculate and display metrics
            total_days = int(df['Count'].sum())
            avg_per_category = round(df.groupby('Category')['Count'].sum().mean(), 1)
            day_sums = df.groupby('Day')['Count'].sum()
            most_common_day = day_sums.index[day_sums.values.argmax()] if not day_sums.empty else "N/A"

            # Display metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Total fravær dage", total_days)
            with metrics_col2:
                st.metric("Gennemsnit per kategori", avg_per_category)
            with metrics_col3:
                st.metric("Mest sete dag", most_common_day)

            # Debug: Display raw data
            if st.checkbox("Vis rådata"):
                st.write(df)
        else:
            st.warning("Ingen data fundet for den valgte periode")
    else:
        st.warning("Ingen fraværsdata tilgængelig")

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
            x=alt.X('Day:N', sort=days_order, title='Ugedag'),
            y=alt.Y('sum(Count):Q', title='Antal dage'),
            color=alt.Color('Category:N', scale=color_scale),
            tooltip=[
                alt.Tooltip('Day:N'),
                alt.Tooltip('Category:N'),
                alt.Tooltip('sum(Count):Q', title='Days', format='.0f')
            ]
        ).properties(
            title='Daglig graf'
        )
        st.altair_chart(day_chart, use_container_width=True)

    with col2:
        # Category Chart
        category_chart = base.mark_bar().encode(
            x=alt.X('Category:N', title='Kategori'),
            y=alt.Y('sum(Count):Q', title='Antal dage'),
            color=alt.Color('Category:N', scale=color_scale),
            tooltip=[
                alt.Tooltip('Category:N'),
                alt.Tooltip('sum(Count):Q', title='Total Days', format='.0f')
            ]
        ).properties(
            title='Fraværs type graf'
        )
        st.altair_chart(category_chart, use_container_width=True)

    # Heatmap
    with col3:
        heatmap = base.mark_rect().encode(
            x=alt.X('Day:N', sort=days_order, title='Ugedag'),
            y=alt.Y('Category:N', title='Kategori'),
            color=alt.Color('sum(Count):Q',
                            scale=alt.Scale(scheme='viridis'),
                            title='Antal dage'),
            tooltip=[
                alt.Tooltip('Day:N'),
                alt.Tooltip('Category:N'),
                alt.Tooltip('sum(Count):Q', title='Days', format='.0f')
            ]
        ).properties(
            title='Fravær Heatmap'
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

def debug_database_contents():
    users = DatabaseManager.all()
    st.write("Database Contents:")
    for user in users:
        st.write(user)
# Add this to your main() function or where you initialize the app
if __name__ == "__main__":
    # Migrate existing database entries to use type-based colors
    app = CalendarApp()
    app.main()
