import os
import csv
from datetime import datetime
import dearpygui.dearpygui as dpg
from google.oauth2 import service_account
from googleapiclient.discovery import build

# === CONFIGURATION ===
# Percorso al file CSV generato
CSV_PATH = os.path.join("exams", "deep_learning", "studio_scheduler.csv")
# Calendar API credentials
SERVICE_ACCOUNT_FILE = "path/to/service-account.json"  # aggiorna con il tuo file
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
CALENDAR_ID = 'primary'  # o sostituisci con l'ID del calendario desiderato
TIMEZONE = 'Europe/Rome'

# === FUNZIONI DI SUPPORTO ===

def read_csv_events(csv_path):
    """
    Legge il CSV con colonne: Data, Fascia oraria, Ore, Attività
    e ritorna liste di eventi Python.
    """
    events = []
    with open("./exams/deep_learning/studio_scheduler.csv", encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row['Data']
            period = row['Fascia oraria']
            summary = row['Attività']
            # period esempio: "08:00 - 12:00"
            try:
                start_str, end_str = [t.strip() for t in period.split('-')]
                start_dt = datetime.fromisoformat(f"{date_str}T{start_str}:00")
                end_dt   = datetime.fromisoformat(f"{date_str}T{end_str}:00")
            except Exception:
                # in caso di formato differente
                start_dt = None
                end_dt = None
            events.append({
                'start': start_dt,
                'end': end_dt,
                'summary': summary
            })
    return events


def init_google_service():
    """
    Inizializza il client per Google Calendar API (readonly).
    """
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('calendar', 'v3', credentials=creds)


def fetch_google_events(service, calendar_id, time_min):
    """
    Recupera eventi futuri da Google Calendar, a partire da time_min UTC.
    """
    iso_min = time_min.isoformat() + 'Z'
    result = service.events().list(
        calendarId=calendar_id,
        timeMin=iso_min,
        maxResults=250,
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    return result.get('items', [])


# === MAIN GUI ===

def build_gui():
    # Leggi eventi dal CSV
    csv_events = read_csv_events(CSV_PATH)

    # Imposta Window
    dpg.create_context()
    with dpg.window(label="Study Scheduler - Deep Learning", width=800, height=600):
        dpg.add_text("Eventi dal file CSV:")
        with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(label="Data")
            dpg.add_table_column(label="Inizio")
            dpg.add_table_column(label="Fine")
            dpg.add_table_column(label="Attività")
            for evt in csv_events:
                start = evt['start'].strftime('%Y-%m-%d %H:%M') if evt['start'] else '-'
                end   = evt['end'].strftime('%Y-%m-%d %H:%M')   if evt['end']   else '-'
                dpg.add_table_row(
                    dpg.add_text(start),
                    dpg.add_text(end),
                    dpg.add_text(evt['summary'])
                )

        # Separatore
        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Pulsante per caricare da Google Calendar
        def on_load_google(sender, app_data):
            # Inizializza servizio e recupera eventi
            service = init_google_service()
            now = datetime.utcnow()
            g_events = fetch_google_events(service, CALENDAR_ID, now)

            # Crea nuova finestra per eventi Google
            with dpg.window(label="Google Calendar Events", width=800, height=400):
                dpg.add_text("Eventi da Google Calendar:")
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp):
                    dpg.add_table_column(label="Data Inizio")
                    dpg.add_table_column(label="Data Fine")
                    dpg.add_table_column(label="Titolo")
                    for evt in g_events:
                        start = evt['start'].get('dateTime', evt['start'].get('date'))
                        end   = evt['end'].get('dateTime', evt['end'].get('date'))
                        summary = evt.get('summary', '')
                        dpg.add_table_row(
                            dpg.add_text(start),
                            dpg.add_text(end),
                            dpg.add_text(summary)
                        )

        dpg.add_button(label="Carica da Google Calendar", callback=on_load_google)

    dpg.create_viewport(title="Study Scheduler GUI", width=820, height=650)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    build_gui()