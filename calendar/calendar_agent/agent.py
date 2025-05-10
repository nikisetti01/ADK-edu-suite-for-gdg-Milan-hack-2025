from google.adk.agents import Agent
from datetime import datetime, timedelta
import csv
import json
import re
import os

def create_calendar(exam_date: str, understanding_level: str, daily_hours: int,
                    study_period: str, exam_title: str) -> str:
    """
    Genera un calendario di studio in tre blocchi consecutivi e lo esporta in CSV
    nella cartella ./exams/{exam_title}/studio_scheduler.csv, creando la directory
    se necessario. Ritorna una stringa JSON con esito e percorso file.
    """
    today = datetime.today().date()
    end_date = datetime.strptime(exam_date, "%Y-%m-%d").date()
    total_days = (end_date - today).days + 1

    if total_days <= 0:
        return json.dumps({
            "success": False,
            "message": "La data dell'esame deve essere futura."
        })

    # Proporzioni per livello
    proportions = {
        "Base":  (0.6, 0.3, 0.1),
        "Medio": (0.4, 0.4, 0.2),
        "Alto":  (0.3, 0.4, 0.3)
    }
    p_ind, p_gym, p_test = proportions.get(understanding_level, proportions["Base"])

    # Calcolo giorni per blocchi
    days_ind  = int(round(total_days * p_ind))
    days_gym  = int(round(total_days * p_gym))
    days_test = total_days - days_ind - days_gym

    # Generazione righe
    rows = []
    current_day = today
    def add_block(n_days, label):
        nonlocal current_day
        for _ in range(n_days):
            rows.append([
                str(current_day),
                study_period,
                daily_hours,
                label
            ])
            current_day += timedelta(days=1)

    add_block(days_ind,  "Studio individuale")
    add_block(days_gym,  "Ripasso attivo in modalità GYM")
    add_block(days_test, "Simulazione o test")

    # Costruisci il path e crea la cartella
    exam_slug = re.sub(r'[^0-9A-Za-z_]', '',
                   exam_title.strip().replace(' ', '_'))
    dir_path = os.path.join("exams", exam_slug)
    os.makedirs(dir_path, exist_ok=True)

    csv_path = os.path.join(dir_path, "studio_scheduler.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Data", "Fascia oraria", "Ore", "Attività"])
        writer.writerows(rows)

    return json.dumps({
        "success":    True,
        "message":    "Piano generato con successo",
        "exam_title": exam_title,
        "total_days": total_days,
        "csv_file":   csv_path
    })



date = None
typeActivity = None
fascia = None
Nore = None

root_agent = Agent(
    model='gemini-2.0-flash-001',
    name='root_agent',
    description="""
    Sei uno scheduling agent specializzato nell'organizzazione dello studio universitario. Il tuo compito è creare un piano di studio personalizzato dall'oggi fino alla data di un esame specifico.
    Ricevi i seguenti input dall’utente:
    - Data dell’esame finale (formato YYYY-MM-DD)
    - Grado di comprensione da raggiungere (Base, Medio, Alto)
    - Ore giornaliere disponibili per studiare (numero intero)
    - Periodo del giorno in cui l’utente può studiare (Mattina, Pomeriggio, Sera)
    - Titolo o nome dell’esame

    Con queste informazioni, devi generare uno **scheduler giornaliero** che divide le ore in 3 fasi di studio, a seconda del periodo:
    1. **Studio individuale** (focus sui contenuti teorici e apprendimento autonomo)
    2. **Modalità GYM** (ripasso attivo, schemi, flashcard, spiegazioni orali, problem solving veloce)
    3. **Test** (simulazioni d’esame, quiz, domande aperte, verifica degli obiettivi)

    Distribuisci le fasi tenendo conto di:
    - L’obiettivo finale di comprensione (più è alto, più serve intensificare la GYM e i test verso la fine)
    - Il tempo disponibile (ore al giorno * giorni disponibili)
    - Progressione graduale: all’inizio più studio individuale, poi GYM, poi test sempre più frequenti
    - Mantieni almeno un giorno di scarico o test leggero ogni 7 giorni

    Genera per ogni giorno:
    - Data
    - Fascia oraria (coerente con il periodo selezionato)
    - Numero di ore
    - Attività prevista (Studio individuale / GYM / Test)

    Rispetta sempre le ore indicate dall’utente. Concentrati sulla qualità del tempo, non sul riempire tutti gli slot disponibili. L’obiettivo è un piano sostenibile, adattivo e coerente con il livello di apprendimento desiderato.
    """,
    instruction=f'{date}, {fascia} {Nore} {typeActivity}',
    tools=[create_calendar]
)
