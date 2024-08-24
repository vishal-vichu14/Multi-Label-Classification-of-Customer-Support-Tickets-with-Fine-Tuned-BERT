import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import json
import streamlit_lottie as st_lottie

# Function to load a Lottie animation from a JSON file
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Load the Lottie animation
lottie_animation = load_lottie("/content/Animation - 1723210002909.json")
# Load the model and tokenizer
model_path = "multi-class-model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)

# Connect to the SQLite database
conn = sqlite3.connect('test1.db')
cursor = conn.cursor()

# Create tables if they do not exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS Complaints (
        Ticket INTEGER,
        Complaint TEXT,
        Team TEXT,
        Status TEXT
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS Team (
        Member_id INTEGER,
        Name TEXT,
        M_Team TEXT,
        M_Status TEXT
    )
""")
# Create the TicketCounter table if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS TicketCounter (
        id INTEGER PRIMARY KEY,
        last_ticket INTEGER
    )
""")
cursor.execute("SELECT COUNT(*) FROM TicketCounter")
if cursor.fetchone()[0] == 0:
    cursor.execute("INSERT INTO TicketCounter (last_ticket) VALUES (0)")
    conn.commit()

# Load existing complaints and team data for display
rough = pd.read_sql_query("SELECT * FROM Complaints", conn)
rough2 = pd.read_sql_query("SELECT * FROM Team", conn)

st.title("MULTI LABEL CLASSIFICATION")

# Initialize session state
if 'ticket_generated' not in st.session_state:
    st.session_state.ticket_generated = False

# Define the pages
def customer_page():
    st.title("Customer Service")
    st_lottie.st_lottie(lottie_animation, height=200, key="animation")
    section = st.selectbox("Choose", ["Complaints", "Complaint Status"])
    
    with st.container():
        st.write('---')
        if section == "Complaints":
            user_input = st.text_input("Enter your complaint:")
            submit_button = st.button("Submit Complaint")
            
            if submit_button and user_input:
              default = "Pending"
              
              # Generate the next sequential ticket number
              cursor.execute("SELECT last_ticket FROM TicketCounter WHERE id = 1")
              last_ticket = cursor.fetchone()[0]
              ticket = last_ticket + 1
              cursor.execute("UPDATE TicketCounter SET last_ticket = ? WHERE id = 1", (ticket,))
              conn.commit()

              # Predict the team using the BERT model
              inputs = tokenizer(user_input, padding=True, truncation=True, max_length=512, return_tensors="pt").to('cuda')
              output = model(**inputs)
              probs = output[0].softmax(1)
              pred_label_idx = probs.argmax()
              team = model.config.id2label[pred_label_idx.item()]

              # Check for available team members
              cursor.execute("SELECT Member_id, Name FROM Team WHERE M_Team = ? AND M_Status = 'Free'", (team,))
              available_members = cursor.fetchall()

              if available_members:
                  # Assign the first available member
                  member_id, member_name = available_members[0]
                  cursor.execute("INSERT INTO Complaints (Ticket, Complaint, Team, Status) VALUES (?, ?, ?, ?)",
                                (ticket, user_input, team, "Assigned"))
                  cursor.execute("UPDATE Team SET M_Status = 'Assigned' WHERE Member_id = ?", (member_id,))
                  conn.commit()
                  st.write(f"## Ticket {ticket} ## Your ticket has been raised. {member_name} from the {team} will handle your problem.")
              else:
                  cursor.execute("INSERT INTO Complaints (Ticket, Complaint, Team, Status) VALUES (?, ?, ?, ?)",
                                (ticket, user_input, team, default))
                  conn.commit()
                  st.write(f"## Ticket {ticket} ## Your ticket has been raised. However, there are no available members in the {team} to handle your problem right now. Your complaint is pending.")

            # Reset ticket generation state for new entries
            st.session_state.ticket_generated = False   

        elif section == "Complaint Status":
            ticket_number = st.text_input("Enter your ticket number:")
            check_status_button = st.button("Check Status")

            if check_status_button and ticket_number:
                try:
                    ticket_number = int(ticket_number)
                    cursor.execute("SELECT Status FROM Complaints WHERE Ticket = ?", (ticket_number,))
                    result = cursor.fetchone()

                    if result:
                        st.write(f"The status of your ticket {ticket_number} is {result[0]}")
                    else:
                        st.write(f"No complaint found with ticket number {ticket_number}")

                except ValueError:
                    st.write("Please enter a valid ticket number")

def admin_page():
    section = st.selectbox("Choose", ["COMPLAINT", "TEAMS", "STATUS"])

    if section == "COMPLAINT":
        st.write(rough.drop_duplicates())
    elif section == "TEAMS":
        st.write(rough2)
    elif section == "STATUS":
        st.title("Status Overview")

        # Query to get complaints and their status
        cursor.execute("SELECT Ticket, Complaint, Team, Status FROM Complaints")
        complaints_data = cursor.fetchall()

        # Query to get team members and their statuses
        cursor.execute("SELECT Member_id, Name, M_Team, M_Status FROM Team")
        team_data = cursor.fetchall()

        # Combine the data based on the requirements
        output_data = []
        for complaint in complaints_data:
            ticket, complaint_text, team, status = complaint
            if status == 'Assigned':
                # Find the assigned team member
                cursor.execute("SELECT Member_id, Name FROM Team WHERE M_Team = ? AND M_Status = 'Assigned'", (team,))
                assigned_members = cursor.fetchall()
                if assigned_members:
                    member_id, member_name = assigned_members[0]
                    output_data.append((ticket, member_id, member_name, status))
                else:
                    output_data.append((ticket, "N/A", "N/A", status))
            else:
                output_data.append((ticket, "N/A", "N/A", status))

        # Create a DataFrame for the combined data
        output_df = pd.DataFrame(output_data, columns=['Ticket', 'Member ID', 'Name', 'Status'])

        # Display the combined data
        st.write("Combined Status Table")
        st.table(output_df)

def resolve_complaint(team_name):
    st.title("Resolve Complaint")

    # Input field to enter the ticket number
    ticket_number = st.text_input("Enter the ticket number to resolve:")
    resolve_button = st.button("Resolve")

    if resolve_button and ticket_number:
        try:
            ticket_number = int(ticket_number)
            # Update the status of the complaint to "Resolved"
            cursor.execute("UPDATE Complaints SET Status = 'Resolved' WHERE Ticket = ? AND Team = ?", (ticket_number, team_name))
            conn.commit()

            # Find the team member assigned to this complaint
            cursor.execute("SELECT Member_id FROM Team WHERE M_Team = ? AND M_Status = 'Assigned'", (team_name,))
            member = cursor.fetchone()
            if member:
                member_id = member[0]
                # Update the team member's status to "Free"
                cursor.execute("UPDATE Team SET M_Status = 'Free' WHERE Member_id = ?", (member_id,))
                conn.commit()
                st.write(f"Ticket {ticket_number} resolved and team member {member_id} set to Free.")

                # Check for pending complaints and assign to the newly freed member
                cursor.execute("SELECT Ticket, Complaint FROM Complaints WHERE Team = ? AND Status = 'Pending'", (team_name,))
                pending_complaint = cursor.fetchone()
                if pending_complaint:
                    pending_ticket, pending_complaint_text = pending_complaint
                    cursor.execute("UPDATE Complaints SET Status = 'Assigned' WHERE Ticket = ?", (pending_ticket,))
                    cursor.execute("UPDATE Team SET M_Status = 'Assigned' WHERE Member_id = ?", (member_id,))
                    conn.commit()
                    st.write(f"Pending ticket {pending_ticket} assigned to team member {member_id}.")

            else:
                st.write(f"No assigned member found for team {team_name}.")

        except ValueError:
            st.write("Please enter a valid ticket number")

def employees_page():
    section = st.selectbox("Choose", [
        "TEAM + & -",
        "Audio Systems Team",
        "Imaging and Optics Team",
        "Customer Experience Team",
        "Protection and Installation Team",
        "Power and Battery Solutions",
        "Peripheral Devices Team",
        "Connectivity Solutions Team"
    ])

    if section == "TEAM + & -":
        st.write(rough2)
        st.write("Add or Remove Team")
        mid = st.text_input("Enter ID:")
        mname = st.text_input("Enter Name:")
        mteam = st.selectbox("Select Team:", [
            "Audio Systems Team",
            "Imaging and Optics Team",
            "Customer Experience Team",
            "Protection and Installation Team",
            "Power and Battery Solutions",
            "Peripheral Devices Team",
            "Connectivity Solutions Team"
        ])
        mstatus = st.selectbox("Select Status:", ["Free", "Assigned"])
        submit_button = st.button("Submit")

        if submit_button and mid and mname and mteam and mstatus:
            cursor.execute("INSERT INTO Team (Member_id, Name, M_Team, M_Status) VALUES (?, ?, ?, ?)",
                           (mid, mname, mteam, mstatus))
            conn.commit()
            st.write("Team member added successfully")

    elif section in [
        "Audio Systems Team",
        "Imaging and Optics Team",
        "Customer Experience Team",
        "Protection and Installation Team",
        "Power and Battery Solutions",
        "Peripheral Devices Team",
        "Connectivity Solutions Team"
    ]:
        resolve_complaint(section)

# Create a dropdown in the sidebar to select the page
page = st.sidebar.selectbox("Select a page", ["Customer", "Admin", "Employees"])

# Render the selected page
if page == "Customer":
    customer_page()
elif page == "Admin":
    admin_page()
elif page == "Employees":
    employees_page()

conn.commit()
