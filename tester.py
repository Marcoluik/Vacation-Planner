import json
import streamlit as st
from firebase_admin import credentials, db
import firebase_admin
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import random

print("Starting Firebase Initialization...")
try:
    load = json.loads(st.secrets["FIREBASE_KEY"], strict=False)
    print("Firebase Key loaded successfully")
    cred = credentials.Certificate(load)
    print("Credentials created")
    firebase_admin.initialize_app(cred, {"databaseURL": st.secrets["FIREBASE_DATABASE_URL"]})
    print("Firebase app initialized")
except Exception as e:
    print(f"Firebase Initialization Error: {e}")

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
        print(f"Getting reference for path: {path}")
        return db.reference(path)

class DatabaseManager:
    @staticmethod
    def all():
        """
        Retrieve all users from the database.

        :return: List of all user records
        """
        try:
            ref = FirebaseManager.get_ref("/")
            data = ref.get() or {}
            print(f"Retrieved users data: {data}")

            # Convert Firebase's nested dictionary to a list of users
            if '_default' in data:
                print("Using '_default' key")
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

            print(f"Processed users: {users}")
            return users

        except Exception as e:
            print(f"Error retrieving all users: {e}")
            return []

    @staticmethod
    def add_or_update_user(name: str, type: str, **events) -> None:
        """
        Add or update a user in the Firebase Realtime Database.

        :param name: User's name
        :param type: User's type
        :param events: Dictionary of event types and their date ranges
        """
        print(f"Adding/Updating user: {name}, Type: {type}")
        print(f"Events: {events}")

        if not name or not type:
            print("Error: Name and type are required!")
            st.error("Name and type are required!")
            return

        # Get a reference to the users node
        ref = FirebaseManager.get_ref("/")

        # Convert dates to ISO format strings
        formatted_events = {
            k: [(s.isoformat(), e.isoformat()) for s, e in v] if v else []
            for k, v in events.items()
        }
        print(f"Formatted events: {formatted_events}")

        # Get color based on type
        color = ColorManager.get_color_for_type(type)
        print(f"Assigned color: {color}")

        # Prepare user data
        user_data = {
            "name": name,
            "type": type,
            "color": color,
            **formatted_events
        }
        print(f"User data to be saved: {user_data}")

        try:
            # Find existing user by name
            existing_users = ref.get() or {}
            print(f"Existing users: {existing_users}")

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
            ref = FirebaseManager.get_ref("/")
            users_data = ref.get() or {}

            return [
                user for _, user in users_data.items()
                if isinstance(user, dict) and user.get('name') == query.name
            ]

        return []

    @staticmethod
    def load_events(selected_user_name: Optional[str] = None,
                    selected_user_type: Optional[str] = None) -> List[Dict]:
        """
        Load events from Firebase database based on filters.

        :param selected_user_name: Filter by specific user name
        :param selected_user_type: Filter by user type
        :return: List of events in FullCalendar format
        """
        try:
            # Get reference to users
            ref = FirebaseManager.get_ref("/")
            users_data = ref.get() or {}

            # Convert users to list, filtering as needed
            filtered_users = []
            for _, user in users_data.items():
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

            # Convert to FullCalendar events
            events = EventManager.convert_to_fullcalendar(filtered_users)
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

class ColorManager:
    # Predefined colors for types
    TYPE_COLORS = {
        "Bygningskonstruktør": "#34a853",
        "Kontor": "#c19b41",
        "Tømrer": "#f045c1",
        "Formand": "#971a5b",
        "Nedriver": "#d9d9d9",
        "Kørsel/service": "#ff0000",
    }

    @staticmethod
    def get_color_for_type(type_name: str) -> str:
        """Get a consistent color for a given type."""
        # If type already has a color, return it
        if type_name in ColorManager.TYPE_COLORS:
            return ColorManager.TYPE_COLORS[type_name]

        # If it's a new type, generate a color and store it
        new_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        ColorManager.TYPE_COLORS[type_name] = new_color
        return new_color

class EventManager:
    @staticmethod
    def convert_to_fullcalendar(events_data: List[Dict], types: Optional[Dict] = None) -> List[Dict]:
        fullcalendar_events = []

        for event in events_data:
            # Get color based on type
            event_type = event.get("type", "Unknown")
            color = ColorManager.get_color_for_type(event_type)

            for event_type in ["Ferie", "Sygdom", "Barnsygdom", "Kursus"]:
                dates = event.get(event_type, [])
                if not dates:
                    continue

                for start, end in dates:
                    try:
                        start_date = datetime.strptime(start, '%Y-%m-%d').date()
                        end_date = datetime.strptime(end, '%Y-%m-%d').date()

                        # Adjust end date to be inclusive
                        end_date = end_date + timedelta(days=1)

                        fullcalendar_events.append({
                            "title": f"{event.get('name', 'Unknown')} - {event_type.capitalize()}",
                            "start": start_date.isoformat(),
                            "end": end_date.isoformat(),
                            "color": color,  # Color now based on type
                            "description": f"{event_type.capitalize()} event for {event.get('name', 'Unknown')}"
                        })
                    except (ValueError, TypeError) as e:
                        st.error(f"Error processing dates for {event.get('name', 'Unknown')}: {e}")

        return fullcalendar_events


def test_database_manager():
    print("=== DatabaseManager Comprehensive Test ===")

    # 1. Test retrieving all users
    print("\n1. Testing DatabaseManager.all()")
    all_users = DatabaseManager.all()
    print(f"Total users retrieved: {len(all_users)}")
    for user in all_users:
        print(f"User: {user.get('name', 'N/A')} - Type: {user.get('type', 'N/A')}")

    # 2. Test adding/updating a user
    print("\n2. Testing add_or_update_user()")
    try:
        # Example events (you'll need to adapt these to your actual date input method)
        from datetime import date
        DatabaseManager.add_or_update_user(
            name="Test User",
            type="Kontor",
            Ferie=[(date(2024, 7, 1), date(2024, 7, 15))],
            Sygdom=[(date(2024, 8, 5), date(2024, 8, 7))]
        )
    except Exception as e:
        print(f"Error adding/updating user: {e}")

    # 3. Test finding a user by name
    print("\n3. Testing find_by_name()")
    test_name = "Test User"
    found_user = DatabaseManager.find_by_name(test_name)
    if found_user:
        print(f"Found user: {found_user}")
    else:
        print(f"No user found with name {test_name}")

    # 4. Test loading events
    print("\n4. Testing load_events()")
    print("Loading all events:")
    all_events = DatabaseManager.load_events()
    print(f"Total events: {len(all_events)}")
    for event in all_events[:5]:  # Print first 5 events
        print(event)

    # 5. Test loading events with filters
    print("\n5. Testing load_events with filters")
    print("Events for specific user:")
    filtered_events = DatabaseManager.load_events(selected_user_name="Test User")
    print(f"Events for Test User: {len(filtered_events)}")
    for event in filtered_events:
        print(event)

    # 6. Test deletion (be careful with this!)
    print("\n6. Testing delete_user()")
    delete_result = DatabaseManager.delete_user("Test User")
    print(f"Deletion result: {delete_result}")

    print("\n=== Test Complete ===")


# Run the comprehensive test
test_database_manager()