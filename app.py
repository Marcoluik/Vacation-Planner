import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from tinydb import TinyDB, Query
from openai import OpenAI
from streamlit_calendar import calendar
import random
from typing import List, Dict, Tuple, Optional
import altair as alt
import os
import google.generativeai as genai

#FIREBASE
import firebase_admin
from firebase_admin import credentials, db
import json
st.set_page_config(layout="wide", page_title="FeriePlan", page_icon=":calendar:")

try:
    API = st.secrets['OPENAI_API_KEY']
    OPENAI_API_KEY = API
except KeyError:
    st.error("OPENAI_API_KEY is missing in the secrets configuration.")
    st.stop()

try:
    API = st.secrets['GENAI_API_KEY']
    genai.configure(api_key=API)
    # Create the model

except KeyError:
    st.error("GENAI_API_KEY is missing in the secrets configuration.")

EVENT_KEY_MAPPING = {
    "vacation": "Ferie",
    "sick": "Sygdom",
    "child_sick": "Barnsygdom",
    "training": "Kursus",
    "maternity": "Barsel",
    "Free": "Fri",
    "School": "Skole"
}

EVENT_TYPES = list(EVENT_KEY_MAPPING.keys())

DEFAULT_CALENDAR_OPTIONS = {
    "editable": True,
    "navLinks": True,
    "selectable": True
}




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

        try:
            # Find existing user by name
            existing_users = ref.get() or {}
            print(f"Existing users: {existing_users}")

            # Check if the type color already exists
            existing_colors = colors_ref.get() or {}
            if type not in existing_colors:
                # Add new color entry for the type
                color = ColorManager.get_color_for_type(type)
                colors_ref.child(type).set(color)
                print(f"Added new color entry for type: {type}")

            user_key = None
            existing_user_data = None
            for key, user in existing_users.items():
                if isinstance(user, dict) and user.get('name') == name:
                    user_key = key
                    existing_user_data = user
                    break

            # Prepare the new event data
            formatted_events = {}
            for event_type, date_ranges in events.items():
                if date_ranges:  # Only process if there are new date ranges
                    new_ranges = [(s.isoformat(), e.isoformat()) for s, e in date_ranges]
                    
                    # If updating existing user, merge with existing events
                    if existing_user_data and event_type in existing_user_data:
                        existing_ranges = existing_user_data.get(event_type, [])
                        # Combine existing and new ranges
                        formatted_events[event_type] = existing_ranges + new_ranges
                    else:
                        formatted_events[event_type] = new_ranges

            if user_key:
                # Update existing user
                print(f"Updating existing user with key: {user_key}")
                # Merge existing data with updates
                update_data = {
                    "name": name,
                    "type": type,
                    "color": ColorManager.get_color_for_type(type)
                }
                # Only update event types that have new data
                for event_type, ranges in formatted_events.items():
                    update_data[event_type] = ranges
                
                # Preserve existing event types that aren't being updated
                for event_type in EVENT_TYPES:
                    if event_type not in formatted_events and event_type in existing_user_data:
                        update_data[event_type] = existing_user_data[event_type]
                
                ref.child(user_key).update(update_data)
                st.success(f"Updated events for {name}")
            else:
                # Add new user
                print("Adding new user")
                user_data = {
                    "name": name,
                    "type": type,
                    "color": ColorManager.get_color_for_type(type),
                    **formatted_events
                }
                ref.push(user_data)
                st.success(f"User {name} added successfully.")

        except Exception as e:
            print(f"Error adding/updating user: {e}")
            st.error(f"Error adding/updating user: {e}")

        # Clear calendar events cache to force reload
        if "calendar_events" in st.session_state:
            del st.session_state.calendar_events

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

    @staticmethod
    def delete_event(name: str, event_type: str, start_date: str, end_date: str) -> bool:
        """
        Delete a specific event for a user from the Firebase database.

        :param name: Name of the user
        :param event_type: Type of event to delete (vacation, sick, child_sick, training)
        :param start_date: Start date of the event to delete
        :param end_date: End date of the event to delete
        :return: True if deleted successfully, False otherwise
        """
        try:
            ref = FirebaseManager.get_ref("/_default")
            users = ref.get() or {}

            # Find the user
            user_key = None
            user_data = None
            for key, user in users.items():
                if isinstance(user, dict) and user.get('name') == name:
                    user_key = key
                    user_data = user
                    break

            if not user_data:
                st.error(f"Ingen bruger fundet med navnet {name}")
                return False

            # Get the event list for the specified type
            event_list = user_data.get(event_type, [])
            if not event_list:
                st.error(f"Ingen {event_type} perioder fundet for {name}")
                return False

            # Find and remove the specific event
            new_event_list = [
                event for event in event_list
                if event[0] != start_date or event[1] != end_date
            ]

            if len(new_event_list) == len(event_list):
                st.error("Periode ikke fundet")
                return False

            # Update the database
            ref.child(user_key).child(event_type).set(new_event_list)
            return True

        except Exception as e:
            st.error(f"Fejl ved sletning af periode: {e}")
            return False




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
        print("\n=== display_calendar called ===")
        print(f"Session state keys: {st.session_state.keys()}")
        print(f"User override: {user_override}")
        print(f"Types: {types}")

        # Initialize calendar_events if not present
        if "calendar_events" not in st.session_state:
            print(">>> Loading events (not in session state)")
            events = DatabaseManager.load_events(
                selected_user_name=user_override,
                selected_user_type=types
            )
            print(f">>> Events loaded: {len(events)} events")
            st.session_state.calendar_events = events
        else:
            print(">>> Using cached events from session state")
            events = st.session_state.calendar_events

        print(">>> Rendering calendar...")

        # Create a callback for the selectbox
        def on_view_change():
            st.session_state.calendar_key = f"calendar_{id(events)}_{st.session_state.selected_view}"

        # Initialize the calendar key if not present
        if 'calendar_key' not in st.session_state:
            st.session_state.calendar_key = f"calendar_{id(events)}_initial"

        # Create the selectbox with the callback
        selected_view = st.selectbox(
            "Vælg visning",
            list(self.calendar_modes.keys()),
            key="selected_view",
            on_change=on_view_change
        )

        # Get the calendar mode configuration
        mode = self.calendar_modes[selected_view]
        initial_view = mode["initialView"]
        header_toolbar = mode["headerToolbar"]
        views = mode["views"]

        # Set up calendar options
        calendar_options = {
            **DEFAULT_CALENDAR_OPTIONS,
            "loading": False,
            "rerenderDelay": 0,
            "handleWindowResize": False,
            "weekNumbers": True,
            "initialView": initial_view,
            "headerToolbar": header_toolbar,
            "views": views,
            "firstDay": 1,
            "locale": "da",

        }

        print(f">>> Using calendar key: {st.session_state.calendar_key}")
        print(f">>> About to render calendar with {len(events)} events")

        # Render the calendar with the dynamic key
        state = calendar(
            events=events,
            options=calendar_options,
            key=st.session_state.calendar_key
        )
        print(">>> Calendar rendered")
        print("=== display_calendar complete ===\n")

    def ask_assistant(self, content: str, database: List[Dict], data: Dict) -> str:
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Du er en specialiseret analyseekspert med fokus på fraværsmønstre i arbejdsmiljøer.

Din primære opgave er at:
1. Analysere medarbejderes fraværsdata for at identificere mønstre
2. Vurdere om disse mønstre er statistisk signifikante
3. Kategorisere mønstre i følgende typer:
   - Tidsmæssige mønstre (bestemte ugedage, måneder, eller årstider)
   - Længdemønstre (varighed af fravær)
   - Frekvensbaserede mønstre (hyppighed af fravær)
   - Kategorimønstre (typer af fravær)

Format din analyse således:
1. Identificerede mønstre
2. Statistisk relevans
3. Mulige årsagssammenhænge
4. Anbefalinger (hvis relevant)

Databasestruktur:
- Hver post indeholder: navn, fraværstype, start- og slutdato
- Fraværstyper inkluderer: Ferie, Sygdom, Barnsygdom, Kursus

Vær objektiv og faktabaseret i din analyse, og undgå at drage forhastede konklusioner."""
                },
                {
                    "role": "user",
                    "content": f"""Analysér følgende fraværsdata:

Forespørgsel: {content}
Specifik data: {data}
Komplet database: {database}

Identificér eventuelle mønstre og vurdér deres statistiske signifikans."""
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
            st.error(f"Error getting AI response: {e}, trying gemini-2.0-flash-exp")
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                generation_config=generation_config,
            )
            contentuere = [
                    {
                        "role": "system",
                        "content": """Du er en specialiseret analyseekspert med fokus på fraværsmønstre i arbejdsmiljøer.

Din primære opgave er at:
1. Analysere medarbejderes fraværsdata for at identificere mønstre
2. Vurdere om disse mønstre er statistisk signifikante
3. Kategorisere mønstre i følgende typer:
   - Tidsmæssige mønstre (bestemte ugedage, måneder, eller årstider)
   - Længdemønstre (varighed af fravær)
   - Frekvensbaserede mønstre (hyppighed af fravær)
   - Kategorimønstre (typer af fravær)

Format din analyse således:
1. Identificerede mønstre
2. Statistisk relevans
3. Mulige årsagssammenhænge
4. Anbefalinger (hvis relevant)

Databasestruktur:
- Hver post indeholder: navn, fraværstype, start- og slutdato
- Fraværstyper inkluderer: Ferie, Sygdom, Barnsygdom, Kursus

Vær objektiv og faktabaseret i din analyse, og undgå at drage forhastede konklusioner."""
                },
                {
                    "role": "user",
                    "content": f"""Analysér følgende fraværsdata:

Forespørgsel: {content}
Specifik data: {data}
Komplet database: {database}

Identificér eventuelle mønstre og vurdér deres statistiske signifikans."""
                    },
                    {
                        "role": "user",
                        "content": f"Hvem? {content} "
                                   f"Og her er dataene {data} i ugedagsformat. "
                                   f"Her er hele databasen med datoerne {database}"
                    }
                ]

            chat_session = model.start_chat()
            response = chat_session.send_message(content, contentuere)
            return response

    def main(self):
        st.title("Ferieplan")


        # Check if the user is logged in
        if "user" not in st.session_state:
            self.login_page()
            return

        page_choice = st.sidebar.selectbox("Vælg side", ["Kalender", "Statistik"])
        if page_choice == "Kalender":
            self.display_calendar()
            print("page choice: ", page_choice)
        else:
            self.secrets()

        if st.session_state.user == "admin":
            self.setup_sidebar()
            self.setup_deletion_ui()



    def login_page(self):
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in st.secrets["ADMIN_CREDENTIALS"] and password == st.secrets["ADMIN_CREDENTIALS"][username]:
                st.session_state.user = "admin"
                st.success(f"Logged in as {username} (admin)")
                self.display_calendar()
            elif username == st.secrets["USER_USERNAME"] and password == st.secrets["USER_PASSWORD"]:
                st.session_state.user = "user"
                st.success("Logged in as user")
                self.display_calendar()
            else:
                st.error("Incorrect username or password")

    def setup_sidebar(self):
        st.sidebar.header("Brugerdata")
        is_new_user = st.sidebar.checkbox("Ny bruger", key="Ny bruger")

        # Handle user/type selection outside the form
        if not is_new_user:
            nametype = {}
            all_types = set()  
            
            for item in DatabaseManager.all():
                name = item.get("name")
                type_val = item.get("type")
                if name and type_val:
                    nametype[name] = type_val
                    all_types.add(type_val)

            # Move name selection outside the form
            selected_name = st.sidebar.selectbox("Navne", nametype.keys(), key='name_select')
            
            # Get the current user's type
            current_type = nametype.get(selected_name)
            all_types_list = list(all_types)
            
            # Find the index of the current type
            default_index = all_types_list.index(current_type) if current_type in all_types_list else 0
            
            # Move type selection outside the form
            selected_type = st.sidebar.selectbox("Type", all_types_list, 
                                        index=default_index,
                                        key=f"type_{selected_name}")

        # Now handle the form for date inputs and submission
        with st.sidebar.form(key="add_user_form"):
            if is_new_user:
                name = st.text_input("Navn")
                user_type = st.text_input("Brugertype")
            else:
                # Use the selected values from outside the form
                name = selected_name
                user_type = selected_type
                
            start = st.date_input("Start")
            end = st.date_input("Slut")

            reverse_mapping = {v: k for k, v in EVENT_KEY_MAPPING.items()}
            selected_label = st.selectbox(
                "Type fravær",
                options=EVENT_KEY_MAPPING.values(),
                index=0
            )

            if st.form_submit_button("Tilføj"):
                event_type = reverse_mapping[selected_label]
                DatabaseManager.add_or_update_user(
                    name, user_type, **{event_type: [(start, end)]}
                )

    def setup_deletion_ui(self):
        """Add deletion UI to the sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("Slet fraværsperiode")

        # Get all users and their events
        users = DatabaseManager.all()
        if not users:
            st.sidebar.warning("Ingen brugere fundet")
            return

        # Create selectbox for users
        user_names = [user.get('name') for user in users if user.get('name')]
        selected_user = st.sidebar.selectbox(
            "Vælg medarbejder",
            options=user_names,
            key="delete_user_select"
        )

        if selected_user:
            # Get user's events
            user_events = []
            for user in users:
                if user.get('name') == selected_user:
                    # Create a list of all events for this user
                    for event_type, danish_name in EVENT_KEY_MAPPING.items():
                        if event_type in user:
                            for start, end in user[event_type]:
                                user_events.append({
                                    'type': danish_name,
                                    'start': start,
                                    'end': end,
                                    'event_key': event_type
                                })

            if not user_events:
                st.sidebar.info(f"Ingen fraværsperioder fundet for {selected_user}")
                return

            # Format events for display
            event_options = [
                f"{evt['type']}: {evt['start']} - {evt['end']}"
                for evt in user_events
            ]

            selected_event_idx = st.sidebar.selectbox(
                "Vælg fraværsperiode",
                range(len(event_options)),
                format_func=lambda x: event_options[x],
                key="delete_event_select"
            )

            if st.sidebar.button("Slet fraværsperiode", type="primary"):
                selected_event = user_events[selected_event_idx]
                success = DatabaseManager.delete_event(
                    name=selected_user,
                    event_type=selected_event['event_key'],
                    start_date=selected_event['start'],
                    end_date=selected_event['end']
                )

                if success:
                    # Clear calendar events cache to force reload
                    if "calendar_events" in st.session_state:
                        del st.session_state.calendar_events
                    st.sidebar.success(f"Fraværsperiode slettet for {selected_user}")
                    st.rerun()
    def secrets(self):
        # Get unique names and types
        names = [item.get('name') for item in DatabaseManager.all() if item.get('name')] + ["All"]
        types = list(set(item.get('type') for item in DatabaseManager.all() if item.get('type')))

        # Create two columns for the control panel
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Kontrol Panel")
            switch = st.radio("Sorter efter:", ["Navn", "Type"])
            view_mode = st.radio("Vis data:", ["Kalender", "Statistik"])

        with col2:
            st.subheader("Vælg Data")
            chosen = None
            chosen_types = None

            if switch == "Navn":
                chosen = st.multiselect("Vælg navne:", names)
            else:
                chosen_types = st.multiselect("Vælg typer:", types)

            # AI Assistant section
            question = st.text_input("Spørg kunstig intelligens:")
            if question:
                dball = DatabaseManager.all()
                current_data = chosen if chosen and "All" not in chosen else dball
                answer = self.ask_assistant(question, dball, current_data)
                st.subheader("Assistant's Response:")
                st.write(answer)

        # Load events based on selections
        events = []
        if chosen:
            if "All" in chosen:
                events = DatabaseManager.load_events()
            else:
                for name in chosen:
                    events.extend(DatabaseManager.load_events(selected_user_name=name))
        elif chosen_types:
            for type_ in chosen_types:
                events.extend(DatabaseManager.load_events(selected_user_type=type_))

        def on_view_change():
            # Update the calendar key when the view changes
            st.session_state.calendar_key = f"calendar_{id(events)}_{st.session_state.selected_view}"

        if view_mode == "Kalender":
            # Initialize the calendar key if not present
            if 'calendar_key' not in st.session_state:
                st.session_state.calendar_key = f"calendar_{id(events)}_initial"

            # Create the selectbox with the callback
            selected_view = st.selectbox(
                "Vælg visning",
                list(self.calendar_modes.keys()),
                key="selected_view",
                on_change=on_view_change
            )

            # Get the calendar mode configuration
            mode = self.calendar_modes[selected_view]
            initial_view = mode["initialView"]
            header_toolbar = mode["headerToolbar"]
            views = mode["views"]

            # Use the same calendar display logic as the sidebar
            calendar_options = {
                **DEFAULT_CALENDAR_OPTIONS,
                "loading": False,
                "rerenderDelay": 0,
                "handleWindowResize": False,
                "weekNumbers": True,
                "initialView": initial_view,
                "headerToolbar": header_toolbar,
                "views": views,
                "firstDay": 1,
                "locale": "da",
            }

            # Force calendar rerender with unique key based on selection and view
            calendar_key = f"calendar_{'-'.join(chosen) if chosen else '-'.join(chosen_types) if chosen_types else 'all'}_{selected_view}"


            # Display calendar with current events
            state = calendar(
                events=events,
                options=calendar_options,
                key=calendar_key
            )
        else:  # Statistics view
            if events:
                display_statistics(events)

        # Debug information
        if st.checkbox("-$-"):
            st.write("Valgt:")
            st.write({
                "Chosen Names": chosen,
                "Chosen Types": chosen_types,
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
    
    # Remove Saturday and Sunday from weekday names
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
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
            # Skip weekends (5 = Saturday, 6 = Sunday)
            if current.weekday() < 5:  # Only process weekdays
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
    
    # Create all possible combinations (excluding weekends)
    categories = df_agg['Category'].unique()
    index = pd.MultiIndex.from_product([weekday_names, categories], names=['Day', 'Category'])
    df_full = pd.DataFrame(index=index).reset_index()
    
    df_final = df_full.merge(df_agg, on=['Day', 'Category'], how='left').fillna(0)
    return df_final

def display_statistics(events):
    """Display the statistics dashboard with date range filtering."""
    # First handle preset selection before creating date inputs
    preset_range = st.selectbox(
        "Forudindstillede perioder",
        ["Custom", "Sidste måned", "Sidste 6 måneder", "Sidste år", "Alle data"],
        key="preset_range"
    )
    
    # Calculate default dates based on preset
    today = datetime.now()
    if preset_range == "Sidste måned":
        default_start = today - timedelta(days=30)
    elif preset_range == "Sidste 6 måneder":
        default_start = today - timedelta(days=180)
    elif preset_range == "Sidste år":
        default_start = today - timedelta(days=365)
    elif preset_range == "Alle data":
        default_start = datetime(2024, 10, 1)
    else:  # Custom
        default_start = today - timedelta(days=30)
    
    # Create columns for date selection
    date_col1, date_col2 = st.columns(2)
    
    with date_col1:
        start_date = st.date_input(
            "Fra dato",
            value=default_start,
            key="stats_start_date"
        )
    
    with date_col2:
        end_date = st.date_input(
            "Til dato",
            value=today,
            key="stats_end_date"
        )

    if events:
        # Filter events based on selected date range
        filtered_events = []
        for event in events:
            event_start = datetime.strptime(event['start'], '%Y-%m-%d').date()
            event_end = datetime.strptime(event['end'], '%Y-%m-%d').date()
            
            # Check if event overlaps with selected date range
            if (event_start <= end_date and event_end >= start_date):
                filtered_events.append(event)

        # Calculate statistics with filtered events
        df = calculate_statistics(filtered_events)
        
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
