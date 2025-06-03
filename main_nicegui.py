from nicegui import ui, events, app
from pathlib import Path

# ----------------
# --- VÁLTOZÓK ---
# ----------------

# Feltöltött videók könyvtára
INPUT_DIR = Path("input_videos")
INPUT_DIR.mkdir(exist_ok=True)

# ----------------
# --- FUNKCIÓK ---
# ----------------

# Videó feltöltése és mentése a megadott könyvtárba
def save_uploaded_video(e: events.UploadEventArguments):
    target_path = INPUT_DIR / e.name
    # Videó átmásolása a megadott könyvtárba
    with open(target_path, 'wb') as f:
        f.write(e.content.read())
    ui.notify(f"Videó feltöltve: {e.name}")


# ---------------
# --- OLDALAK ---
# ---------------

# --- Kezdőlap ("/") ---
@ui.page("/")
def start_page():
    with ui.column().classes("absolute-center items-center gap-12"):
        # Kezdőlap címe
        ui.label("Futballanalízis Alkalmazás").classes("text-3xl font-bold")
        # Indítás gomb -> teszt oldalra navigál
        ui.button("Indítás", on_click=lambda: ui.navigate.to("/main_page")).classes("w-48 text-lg")
        # Kilépés gomb -> alkalmazás leállítása
        ui.button("Kilépés", on_click=app.shutdown).classes("w-48 text-lg")

# --- Teszt oldal ("/teszt") ---
@ui.page("/main_page")
def main_page():
    with ui.column().classes("absolute-center items-center"):
        # Teszt oldal címe
        ui.label("Videó elemzése").classes("text-2xl font-semibold")
        # Videó feltöltése
        ui.upload(on_upload=save_uploaded_video,
                  multiple=True, 
                  on_rejected=lambda e: ui.notify("Csak mp4/avi kiterjesztésű videók engedélyezettek!")
                  ).props('accept=".mp4,.avi,.mkv" auto-upload')  
        # Videó feltöltésének leírása
        ui.markdown("Itt töltheti fel az elemezni kívánt videót (mp4/avi/mkv).").classes("text-center")
        # Feltöltött videók -> feltöltött videók oldalára navigál
        ui.button("Feltöltött videók", on_click=lambda: ui.navigate.to("/uploaded_videos")).classes("mt-4")
        # Vissza a kezdőlapra gomb -> kezdőlapra navigál
        ui.button("Vissza a kezdőlapra", on_click=lambda: ui.navigate.to("/")).classes("mt-4")

# --- Feltöltött videók oldala ("/uploaded_videos") ---
@ui.page("/uploaded_videos")
def uploaded_videos_page():
    with ui.column().classes("absolute-center items-center gap-4"):
        # Feltöltött videók oldalának címe
        ui.label("Feltöltött videók").classes("text-2xl font-semibold")
        # Feltöltött videók mappájának ellenőrzése
        video_files = list(INPUT_DIR.glob("*.mp4")) + list(INPUT_DIR.glob("*.avi")) + list(INPUT_DIR.glob("*.mkv"))
        if not video_files:
            ui.label("A Feltöltött videók mappája üres!").classes("text-white")
        else:
            with ui.row().classes("justify-center items-start flex-wrap gap-8"):
                for video_file in video_files:
                    with ui.column().classes("items-center"):
                        # Thumbnail + videólejátszó
                        ui.video(f'/videos/{video_file.name}').props('controls').classes("w-64 h-36 rounded shadow")
                        # Videók címei
                        ui.label(video_file.name).classes("text-sm text-center text-white mt-2")
        # Visszalépés a menübe
        ui.button("Vissza a menübe", on_click=lambda: ui.navigate.to("/main_page")).classes("mt-4")

# --- GUI indítása natív módban, teljes képernyőn ---
app.add_static_files('/videos', INPUT_DIR)
ui.run(native=True, fullscreen=True, title="Futballanalízis", dark=True)